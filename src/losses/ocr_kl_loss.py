"""Frozen differentiable PP-OCRv5 CTC KL-divergence loss (``OCRKLLoss``).

Adapted from ``yqwu905/VOSR`` (``models/ocr_recognizers.py`` and ``models/pixel_losses.py``)
for SRTrainer's RGB ``[0, 1]`` tensor convention.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from utils.text_geometry import canonical_quad, rectify_text_roi

__all__ = [
    "PPOCR_CHECKPOINT",
    "PPOCRv5ServerRec",
    "load_ppocr_recognizer",
    "OCRKLLoss",
    "KLOCRLoss",
]

PPOCR_CHECKPOINT = "PaddlePaddle/PP-OCRv5_server_rec_safetensors"
_VALID_INPUT_RANGES: frozenset[str] = frozenset({"0_1", "-1_1"})
_GEOMETRY_KEYS: tuple[str, ...] = ("quad", "points", "bbox", "box")
_IGNORED_TRANSCRIPTIONS: frozenset[str] = frozenset({"###", "*"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _normalize_amp_dtype(value: Any) -> str:
    """Normalize OCR-KL autocast mode to ``'bf16' | 'fp16' | 'fp32' | 'inherit'``."""
    if value is None:
        return "fp32"
    if isinstance(value, bool):
        raise ValueError(
            f"amp_dtype must be 'bf16', 'fp16', 'fp32', 'inherit', or None, got bool {value!r}."
        )
    if not isinstance(value, str):
        raise ValueError(
            f"amp_dtype must be a string or None, got {type(value).__name__}: {value!r}."
        )
    norm = value.strip().lower()
    if norm in {"bf16", "bfloat16"}:
        return "bf16"
    if norm in {"fp16", "float16"}:
        return "fp16"
    if norm in {"fp32", "float32", "no", "none", "off", ""}:
        return "fp32"
    if norm in {"inherit", "auto"}:
        return "inherit"
    raise ValueError(
        f"Unsupported amp_dtype={value!r}; expected 'bf16', 'fp16', 'fp32', or 'inherit'."
    )


class ConvBN(nn.Module):
    """Conv2d (no bias) + BatchNorm2d + optional activation matching HF ``HGNetV2ConvLayer``."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        act: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        k = (kernel, kernel) if isinstance(kernel, int) else tuple(kernel)
        padding = ((k[0] - 1) // 2, (k[1] - 1) // 2)
        self.convolution = nn.Conv2d(cin, cout, k, stride, padding, groups=groups, bias=False)
        self.normalization = nn.BatchNorm2d(cout)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(self.convolution(x))
        return self.act(x) if self.act is not None else x


class LightConvBN(nn.Module):
    """Pointwise + depthwise separable convolution block used in HGNetV2 stages 2-3."""

    def __init__(self, cin: int, cout: int, kernel: int) -> None:
        super().__init__()
        self.conv1 = ConvBN(cin, cout, 1)
        self.conv2 = ConvBN(cout, cout, kernel, groups=cout, act=F.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class HGBlock(nn.Module):
    """High-performance GPU-friendly dense aggregation block."""

    def __init__(
        self,
        cin: int,
        mid: int,
        cout: int,
        layers: int,
        kernel: int,
        residual: bool,
        light: bool,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            LightConvBN(cin if i == 0 else mid, mid, kernel)
            if light
            else ConvBN(cin if i == 0 else mid, mid, kernel, act=F.relu)
            for i in range(layers)
        )
        self.aggregation = nn.Sequential(
            ConvBN(cin + layers * mid, cout // 2, 1, act=F.relu),
            ConvBN(cout // 2, cout, 1, act=F.relu),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for layer in self.layers:
            outputs.append(layer(outputs[-1]))
        y = self.aggregation(torch.cat(outputs, dim=1))
        return y + x if self.residual else y


class HGStage(nn.Module):
    """HGNetV2 stage with depthwise strided downsampling followed by ``HGBlock`` layers."""

    def __init__(
        self,
        cin: int,
        mid: int,
        cout: int,
        blocks: int,
        light: bool,
        kernel: int,
        layers: int,
        stride: tuple[int, int],
    ) -> None:
        super().__init__()
        self.downsample = ConvBN(cin, cin, 3, stride, groups=cin)
        self.blocks = nn.ModuleList(
            HGBlock(cin if i == 0 else cout, mid, cout, layers, kernel, i > 0, light)
            for i in range(blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


class HGStem(nn.Module):
    """PPHGNetV2 stem block."""

    def __init__(self) -> None:
        super().__init__()
        self.stem1 = ConvBN(3, 32, 3, 2, act=F.relu)
        self.stem2a = ConvBN(32, 16, 2, act=F.relu)
        self.stem2b = ConvBN(16, 32, 2, act=F.relu)
        self.stem3 = ConvBN(64, 32, 3, act=F.relu)
        self.stem4 = ConvBN(32, 48, 1, act=F.relu)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        branch = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat((self.pool(x), branch), dim=1)))


class HGNetV2B4Rec(nn.Module):
    """PPHGNetV2-B4 backbone with PP-OCRv5_server_rec text-recognition strides."""

    STAGES = (
        (48, 48, 128, 1, False, 3, 6, (2, 1)),
        (128, 96, 512, 1, False, 3, 6, (1, 2)),
        (512, 192, 1024, 3, True, 5, 6, (2, 1)),
        (1024, 384, 2048, 1, True, 5, 6, (2, 1)),
    )

    def __init__(self) -> None:
        super().__init__()
        self.embedder = HGStem()
        self.encoder = nn.Module()
        self.encoder.stages = nn.ModuleList(HGStage(*stage) for stage in self.STAGES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for stage in self.encoder.stages:
            x = stage(x)
        return x


class SVTRBlock(nn.Module):
    """Self-attention + SiLU MLP block inside the PP-OCRv5 SVTR neck."""

    def __init__(self, dim: int = 120, heads: int = 8, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.heads = heads
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = nn.Module()
        self.self_attn.qkv = nn.Linear(dim, 3 * dim)
        self.self_attn.projection = nn.Linear(dim, dim)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = (
            self.self_attn.qkv(self.layer_norm1(x))
            .reshape(b, n, 3, self.heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        attention = (q @ k.transpose(-1, -2) * (q.shape[-1] ** -0.5)).softmax(dim=-1)
        x = x + self.self_attn.projection((attention @ v).transpose(1, 2).reshape(b, n, c))
        return x + self.mlp.fc2(F.silu(self.mlp.fc1(self.layer_norm2(x))))


class SVTRNeck(nn.Module):
    """SVTR sequence encoder neck projecting 2D backbone maps to 1D CTC frames."""

    def __init__(self, cin: int = 2048, dim: int = 120, depth: int = 2) -> None:
        super().__init__()
        self.conv_block = nn.ModuleList(
            [
                ConvBN(cin, cin // 8, (1, 3), act=F.silu),
                ConvBN(cin // 8, dim, 1, act=F.silu),
                ConvBN(dim, cin, 1, act=F.silu),
                ConvBN(2 * cin, cin // 8, (1, 3), act=F.silu),
                ConvBN(cin // 8, dim, 1, act=F.silu),
            ]
        )
        self.svtr_block = nn.ModuleList(SVTRBlock(dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_block[1](self.conv_block[0](x))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.svtr_block:
            x = block(x)
        x = self.norm(x).transpose(1, 2).reshape(b, c, h, w)
        x = self.conv_block[3](torch.cat((residual, self.conv_block[2](x)), dim=1))
        return self.conv_block[4](x).squeeze(2).transpose(1, 2)


class PPOCRv5ServerRec(nn.Module):
    """PyTorch port of ``PP-OCRv5_server_rec`` CTC branch (458 official safetensors tensors).

    Expects ``(N, 3, 48, W)`` RGB in ``[-1, 1]`` (internally flipped to BGR to match PaddleOCR)
    and outputs per-frame CTC logits of shape ``(N, W // 8, num_classes)``.
    """

    input_height: int = 48

    def __init__(self, num_classes: int = 18385) -> None:
        super().__init__()
        if not _is_positive_int(num_classes):
            raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}.")
        self.model = nn.Module()
        self.model.backbone = HGNetV2B4Rec()
        self.head = nn.Module()
        self.head.encoder = SVTRNeck()
        self.head.head = nn.Linear(120, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                f"PPOCRv5ServerRec expects 4D RGB input (N, 3, 48, W), got {tuple(x.shape)}."
            )
        if x.shape[-2] != self.input_height:
            raise ValueError(
                f"PP-OCRv5_server_rec expects {self.input_height}-pixel-high text lines, "
                f"got height {x.shape[-2]}."
            )
        x = F.avg_pool2d(self.model.backbone(x.flip(1)), (3, 2))
        return self.head.head(self.head.encoder(x))


def load_ppocr_recognizer(
    checkpoint: str | Path = PPOCR_CHECKPOINT,
) -> PPOCRv5ServerRec:
    """Strictly load ``PPOCRv5ServerRec`` from a local file/dir or HuggingFace Hub repo id."""
    path = Path(checkpoint)
    if not path.exists():
        ckpt_str = str(checkpoint)
        if path.is_absolute() or ckpt_str.count("/") != 1:
            raise FileNotFoundError(f"PP-OCR checkpoint not found: {checkpoint}")
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(ckpt_str, "model.safetensors")).parent
        hf_hub_download(ckpt_str, "config.json")

    if path.is_dir():
        cfg_file = path / "config.json"
        if cfg_file.is_file():
            model_type = json.loads(cfg_file.read_text(encoding="utf-8")).get("model_type")
            if model_type != "pp_ocrv5_server_rec":
                raise ValueError(
                    f"Expected PP-OCRv5_server_rec checkpoint, got model_type={model_type!r}."
                )
        path = path / "model.safetensors"

    if not path.is_file():
        raise FileNotFoundError(f"PP-OCR weights file not found: {path}")

    state = load_file(str(path), device="cpu")
    if "head.head.weight" not in state:
        raise ValueError(f"{path} is not a valid PP-OCRv5_server_rec safetensors file.")

    model = PPOCRv5ServerRec(num_classes=int(state["head.head.weight"].shape[0]))
    incompatible = model.load_state_dict(state, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.endswith("num_batches_tracked")]
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            f"PP-OCR weights do not match PPOCRv5ServerRec: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}."
        )
    return model


class OCRKLLoss(nn.Module):
    """KL(OCR(target) || OCR(pred)) over frozen PP-OCRv5 per-frame CTC distributions on text lines.

    Sets ``needs_batch = True`` and operates exclusively on samples carrying crop-relative
    ``text_lines`` annotations (from ``text_lines=...`` or ``batch['text_lines']``). Each line
    geometry (``quad`` / ``points`` / ``bbox`` / ``box``, or the bounding hull of ``glyphs``) is
    differentiably rectified via :func:`rectify_text_roi`, auto-rotated when vertical
    (``height / width >= 1.5``), sorted by width, grouped into chunks of ``strip_batch_size``,
    right-padded to a bucket width, and scored strictly over valid (unpadded) CTC frames.

    Samples or batches without ``text_lines`` annotations are skipped cleanly (returning a
    graph-connected zero loss when no valid text lines exist in the local batch while still
    participating in DDP frame-count synchronization). Recognizer weights are frozen in ``eval()``
    mode and run under ``amp_dtype`` (default ``"bf16"``) while ``log_softmax`` / ``kl_div`` run
    in FP32 for numerical stability. When ``use_checkpoint=True``, each chunk's recognizer forward
    and KL reduction are activation-checkpointed to cap peak memory at one ``strip_batch_size``.
    """

    needs_batch: bool = True

    def __init__(
        self,
        checkpoint: str | Path = PPOCR_CHECKPOINT,
        temperature: float = 1.0,
        input_range: str = "0_1",
        strip_batch_size: int = 4,
        max_lines_per_image: int | None = None,
        max_strip_width: int = 512,
        width_bucket_step: int | None = 32,
        amp_dtype: str | None = "bf16",
        use_checkpoint: bool = True,
        sync_ddp_counts: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
            raise ValueError(f"temperature must be a positive finite real, got {temperature!r}.")
        temp_f = float(temperature)
        if not math.isfinite(temp_f) or temp_f <= 0.0:
            raise ValueError(f"temperature must be finite and > 0, got {temperature!r}.")
        if input_range not in _VALID_INPUT_RANGES:
            raise ValueError(
                f"Unsupported input_range={input_range!r}; expected one of {sorted(_VALID_INPUT_RANGES)}."
            )
        if not _is_positive_int(strip_batch_size):
            raise ValueError(
                f"strip_batch_size must be a positive integer, got {strip_batch_size!r}."
            )
        if max_lines_per_image is not None and not _is_positive_int(max_lines_per_image):
            raise ValueError(
                f"max_lines_per_image must be a positive integer or None, got {max_lines_per_image!r}."
            )
        if not _is_positive_int(max_strip_width) or max_strip_width < 8 or max_strip_width % 8 != 0:
            raise ValueError(
                f"max_strip_width must be a positive multiple of 8 (>= 8), got {max_strip_width!r}."
            )
        if width_bucket_step is not None and (
            not _is_positive_int(width_bucket_step) or width_bucket_step % 8 != 0
        ):
            raise ValueError(
                f"width_bucket_step must be a positive multiple of 8 or None, got {width_bucket_step!r}."
            )
        if not isinstance(use_checkpoint, bool):
            raise ValueError(
                f"use_checkpoint must be a bool, got {type(use_checkpoint).__name__}: {use_checkpoint!r}."
            )
        if not isinstance(sync_ddp_counts, bool):
            raise ValueError(
                f"sync_ddp_counts must be a bool, got {type(sync_ddp_counts).__name__}: {sync_ddp_counts!r}."
            )

        norm_amp = _normalize_amp_dtype(amp_dtype)

        self.checkpoint = str(checkpoint)
        self.temperature = temp_f
        self.input_range = input_range
        self.strip_batch_size = int(strip_batch_size)
        self.max_lines_per_image = int(max_lines_per_image) if max_lines_per_image is not None else None
        self.max_strip_width = int(max_strip_width)
        self.width_bucket_step = int(width_bucket_step) if width_bucket_step is not None else None
        self.amp_dtype = norm_amp
        self.use_checkpoint = use_checkpoint
        self.sync_ddp_counts = sync_ddp_counts
        self._warned_amp_fallback: set[tuple[str, str]] = set()
        self._warned_missing_text_lines: bool = False
        self.last_counts: tuple[int, int] = (0, 0)  # (local_lines, local_frames)

        rec_model = load_ppocr_recognizer(checkpoint)
        resolved_height = int(getattr(rec_model, "input_height", PPOCRv5ServerRec.input_height))
        if not _is_positive_int(resolved_height):
            raise ValueError(f"input_height must be a positive integer, got {resolved_height!r}.")

        self.input_height: int = resolved_height
        self.recognizer: nn.Module = rec_model.eval().requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> OCRKLLoss:
        """Keep the frozen OCR recognizer permanently locked in eval mode."""
        return super().train(False)

    def _to_recognizer_range(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        if self.input_range == "0_1":
            return x * 2.0 - 1.0
        return x

    def _amp_context(self, device: torch.device) -> contextlib.AbstractContextManager[Any]:
        if self.amp_dtype == "inherit":
            return contextlib.nullcontext()
        device_type = device.type
        if self.amp_dtype == "fp32":
            return torch.amp.autocast(device_type=device_type, enabled=False)

        is_avail = torch.amp.is_autocast_available(device_type)
        if self.amp_dtype == "bf16":
            if (device_type == "cuda" and not torch.cuda.is_bf16_supported()) or not is_avail:
                key = ("bf16", device_type)
                if key not in self._warned_amp_fallback:
                    self._warned_amp_fallback.add(key)
                    logging.getLogger("srtrainer").warning(
                        "OCRKLLoss: amp_dtype='bf16' is unavailable on device_type=%r; "
                        "falling back to FP32.",
                        device_type,
                    )
                return torch.amp.autocast(device_type=device_type, enabled=False)
            return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)

        if self.amp_dtype == "fp16":
            if device_type == "cpu" or not is_avail:
                key = ("fp16", device_type)
                if key not in self._warned_amp_fallback:
                    self._warned_amp_fallback.add(key)
                    logging.getLogger("srtrainer").warning(
                        "OCRKLLoss: amp_dtype='fp16' is unsupported on device_type=%r; "
                        "falling back to FP32.",
                        device_type,
                    )
                return torch.amp.autocast(device_type=device_type, enabled=False)
            return torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=True)

        return contextlib.nullcontext()

    def _extract_line_quad(
        self,
        line: Mapping[str, Any],
        context: str,
    ) -> np.ndarray | None:
        if line.get("valid", True) is False or line.get("alignment_valid", True) is False:
            return None
        if line.get("ignore", False) is True or line.get("difficult", False) is True:
            return None
        transcript = line.get("transcript", line.get("transcription", line.get("text", "")))
        if isinstance(transcript, str) and transcript.strip() in _IGNORED_TRANSCRIPTIONS:
            return None

        for geom_key in _GEOMETRY_KEYS:
            if geom_key in line and line[geom_key] is not None:
                return canonical_quad(line[geom_key], name=f"{context}.{geom_key}")

        glyphs = line.get("glyphs")
        if isinstance(glyphs, Sequence) and not isinstance(glyphs, (str, bytes)) and len(glyphs) > 0:
            all_pts: list[np.ndarray] = []
            for g_idx, glyph in enumerate(glyphs):
                if not isinstance(glyph, Mapping):
                    continue
                geom = glyph.get("quad", glyph.get("bbox"))
                if geom is None:
                    continue
                all_pts.append(canonical_quad(geom, name=f"{context}.glyphs[{g_idx}]"))
            if all_pts:
                stacked = np.concatenate(all_pts, axis=0)
                x0, y0 = float(stacked[:, 0].min()), float(stacked[:, 1].min())
                x1, y1 = float(stacked[:, 0].max()), float(stacked[:, 1].max())
                if x1 > x0 and y1 > y0:
                    return canonical_quad([x0, y0, x1, y1], name=f"{context}.glyphs_hull")

        return None

    def _collect_line_entries(
        self,
        text_lines: Sequence[Sequence[Mapping[str, Any]] | None],
    ) -> list[tuple[int, np.ndarray, int]]:
        entries: list[tuple[int, np.ndarray, int]] = []
        for img_idx, per_image in enumerate(text_lines):
            if per_image is None:
                continue
            if not isinstance(per_image, Sequence) or isinstance(per_image, (str, bytes)):
                raise ValueError(
                    f"Expected sequence of line mappings (or None) in text_lines[{img_idx}], "
                    f"got {type(per_image).__name__}."
                )
            img_entries: list[tuple[int, np.ndarray, int]] = []
            for line_idx, item in enumerate(per_image):
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"Expected mapping in text_lines[{img_idx}][{line_idx}], "
                        f"got {type(item).__name__}."
                    )
                quad = self._extract_line_quad(item, f"text_lines[{img_idx}][{line_idx}]")
                if quad is None:
                    continue

                top, right, bottom, left = np.linalg.norm(
                    np.roll(quad, -1, axis=0) - quad, axis=1
                )
                nat_h = max(1.0, float(max(left, right)))
                nat_w = max(1.0, float(max(top, bottom)))
                # Rotate vertical text lines 90 deg CCW to match horizontal PP-OCRv5 input.
                if nat_h / nat_w >= 1.5:
                    quad = quad[[1, 2, 3, 0]]
                    nat_h, nat_w = nat_w, nat_h

                raw_w = round(nat_w * self.input_height / nat_h / 8.0) * 8
                target_w = max(8, min(self.max_strip_width, int(raw_w)))
                img_entries.append((img_idx, quad, target_w))

            if self.max_lines_per_image is not None and len(img_entries) > self.max_lines_per_image:
                keep_indices = torch.randperm(len(img_entries))[: self.max_lines_per_image].tolist()
                img_entries = [img_entries[idx] for idx in keep_indices]

            entries.extend(img_entries)

        # Sort by target width so each strip_batch_size chunk shares similar width with minimal pad.
        entries.sort(key=lambda item: item[2])
        return entries

    def _resolve_pad_width(self, widths: Sequence[int]) -> int:
        max_w = max(widths)
        if self.width_bucket_step is None:
            pad_w = max_w
        else:
            step = self.width_bucket_step
            bucketed = math.ceil(max_w / step) * step
            pad_w = min(self.max_strip_width, bucketed)
            if pad_w < max_w:
                pad_w = max_w
        return max(8, math.ceil(pad_w / 8) * 8)

    def _chunk_kl_sum(
        self,
        pred_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        valid_widths: tuple[int, ...],
        pad_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._amp_context(pred_chunk.device):
            with torch.no_grad():
                target_logits = self.recognizer(target_chunk)
            pred_logits = self.recognizer(pred_chunk)

        with torch.amp.autocast(device_type=pred_chunk.device.type, enabled=False):
            target_f32 = target_logits.float() / self.temperature
            pred_f32 = pred_logits.float() / self.temperature
            if pred_f32.shape != target_f32.shape or pred_f32.ndim != 3:
                raise ValueError(
                    f"OCR recognizer must return matching (N, T, C) logits, "
                    f"got pred={tuple(pred_f32.shape)}, target={tuple(target_f32.shape)}."
                )

            batch_n, total_t, _ = pred_f32.shape
            if all(w == pad_w for w in valid_widths):
                target_log = F.log_softmax(target_f32, dim=-1)
                pred_log = F.log_softmax(pred_f32, dim=-1)
                kl_sum = F.kl_div(pred_log, target_log, reduction="sum", log_target=True)
                valid_frames = batch_n * total_t
            else:
                valid_lens = [
                    max(1, min(total_t, round(w * total_t / pad_w))) for w in valid_widths
                ]
                frame_idx = torch.arange(total_t, device=pred_f32.device).unsqueeze(0)
                len_t = torch.tensor(valid_lens, device=pred_f32.device).unsqueeze(1)
                valid_mask = frame_idx < len_t
                target_log = F.log_softmax(target_f32[valid_mask], dim=-1)
                pred_log = F.log_softmax(pred_f32[valid_mask], dim=-1)
                kl_sum = F.kl_div(pred_log, target_log, reduction="sum", log_target=True)
                valid_frames = int(sum(valid_lens))

        return kl_sum, torch.tensor(valid_frames, device=pred_chunk.device, dtype=torch.long)

    def _run_chunk_kl(
        self,
        pred_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        valid_widths: tuple[int, ...],
        pad_w: int,
    ) -> tuple[torch.Tensor, int]:
        if self.use_checkpoint and pred_chunk.requires_grad and torch.is_grad_enabled():

            def _ckpt_fn(p_in: torch.Tensor, t_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return self._chunk_kl_sum(p_in, t_in, valid_widths, pad_w)

            kl_sum, frames_t = checkpoint(
                _ckpt_fn,
                pred_chunk,
                target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            kl_sum, frames_t = self._chunk_kl_sum(pred_chunk, target_chunk, valid_widths, pad_w)
        return kl_sum, int(frames_t.item())

    def _forward_text_lines_chunked(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        entries: list[tuple[int, np.ndarray, int]],
    ) -> tuple[torch.Tensor, int, int]:
        with torch.no_grad():
            target_norm = self._to_recognizer_range(target)
        pred_norm = self._to_recognizer_range(pred)

        total_kl: torch.Tensor | None = None
        total_frames = 0
        for start in range(0, len(entries), self.strip_batch_size):
            chunk = entries[start : start + self.strip_batch_size]
            widths = tuple(item[2] for item in chunk)
            pad_w = self._resolve_pad_width(widths)

            p_crops: list[torch.Tensor] = []
            t_crops: list[torch.Tensor] = []
            for img_idx, quad, target_w in chunk:
                out_hw = (self.input_height, target_w)
                p_roi = rectify_text_roi(pred_norm[img_idx : img_idx + 1], quad, out_hw)
                with torch.no_grad():
                    t_roi = rectify_text_roi(target_norm[img_idx : img_idx + 1], quad, out_hw)
                if target_w < pad_w:
                    pad_right = pad_w - target_w
                    p_roi = F.pad(p_roi, (0, pad_right, 0, 0), mode="constant", value=0.0)
                    t_roi = F.pad(t_roi, (0, pad_right, 0, 0), mode="constant", value=0.0)
                p_crops.append(p_roi)
                t_crops.append(t_roi)

            p_chunk = torch.cat(p_crops, dim=0)
            t_chunk = torch.cat(t_crops, dim=0)
            chunk_kl, chunk_frames = self._run_chunk_kl(p_chunk, t_chunk, widths, pad_w)
            total_kl = chunk_kl if total_kl is None else total_kl + chunk_kl
            total_frames += chunk_frames

        assert total_kl is not None
        return total_kl, len(entries), total_frames

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        text_lines: Sequence[Sequence[Mapping[str, Any]] | None] | None = None,
        batch: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                f"OCRKLLoss expects 4D inputs (B, 3, H, W), "
                f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}."
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred has shape {tuple(pred.shape)}, "
                f"target has shape {tuple(target.shape)}."
            )
        if pred.shape[1] != 3:
            raise ValueError(
                f"OCRKLLoss expects 3-channel RGB inputs, got {pred.shape[1]} channels."
            )

        resolved_lines = text_lines
        if resolved_lines is None and batch is not None:
            resolved_lines = batch.get("text_lines")

        if resolved_lines is None:
            if not self._warned_missing_text_lines:
                self._warned_missing_text_lines = True
                logging.getLogger("srtrainer").warning(
                    "OCRKLLoss received batch without `text_lines` annotations; skipping OCR-KL "
                    "on unannotated samples (returning 0). Run `tools/detect_text_lines.py` to "
                    "generate `text_lines` annotations for training manifests."
                )
            entries: list[tuple[int, np.ndarray, int]] = []
        else:
            if isinstance(resolved_lines, (str, bytes)) or not isinstance(resolved_lines, Sequence):
                raise ValueError(
                    f"text_lines must be a per-image sequence, got {type(resolved_lines).__name__}."
                )
            if len(resolved_lines) != pred.shape[0]:
                raise ValueError(
                    f"Length of text_lines ({len(resolved_lines)}) must match batch size ({pred.shape[0]})."
                )
            entries = self._collect_line_entries(resolved_lines)

        if not entries:
            self.last_counts = (0, 0)
            total_frames = 0
            # Connected zero keeps autograd / DDP backward graphs valid on unannotated/text-free batches.
            loss = pred.flatten()[:0].float().sum()
        else:
            total_kl, num_lines, total_frames = self._forward_text_lines_chunked(
                pred, target, entries
            )
            self.last_counts = (num_lines, total_frames)
            loss = (total_kl / max(1, total_frames)) * (self.temperature**2)

        if (
            self.sync_ddp_counts
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            world = dist.get_world_size()
            counts = torch.tensor([total_frames], device=pred.device, dtype=torch.long)
            dist.all_reduce(counts)
            global_frames = counts[0].clamp_min(1)
            loss = loss * (total_frames * world / global_frames.to(dtype=loss.dtype))

        return loss


KLOCRLoss = OCRKLLoss
