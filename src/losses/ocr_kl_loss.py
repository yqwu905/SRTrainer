"""Frozen differentiable PP-OCRv5 CTC KL-divergence loss (``OCRKLLoss``).

Adapted from ``yqwu905/VOSR`` (``models/ocr_recognizers.py`` and ``models/pixel_losses.py``)
for SRTrainer's RGB ``[0, 1]`` tensor convention.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

__all__ = [
    "PPOCR_CHECKPOINT",
    "PPOCRv5ServerRec",
    "load_ppocr_recognizer",
    "extract_text_strips",
    "OCRKLLoss",
    "KLOCRLoss",
]

PPOCR_CHECKPOINT = "PaddlePaddle/PP-OCRv5_server_rec_safetensors"
_VALID_INPUT_RANGES: frozenset[str] = frozenset({"0_1", "-1_1"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


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


def extract_text_strips(images: torch.Tensor, height: int, stride: int) -> torch.Tensor:
    """Slice 4D images into overlapping full-width horizontal strips covering every row."""
    full = int(images.shape[-2])
    eff_height = min(height, full)
    starts = list(range(0, full - eff_height + 1, stride))
    if not starts or starts[-1] + eff_height < full:
        starts.append(full - eff_height)
    return torch.cat([images[..., start : start + eff_height, :] for start in starts], dim=0)


class OCRKLLoss(nn.Module):
    """KL(OCR(target) || OCR(pred)) over frozen PP-OCRv5 per-frame CTC distributions.

    Cuts ``pred`` and ``target`` into matching overlapping horizontal strips, resizes each
    strip to the recognizer's line height (48 px for PP-OCRv5), and computes temperature-scaled
    frame-averaged KL divergence in FP32 outside autocast. Recognizer weights are frozen and
    permanently locked in ``eval()`` mode so BatchNorm statistics never mutate; gradients flow
    exclusively into ``pred``.

    By default ``input_range="0_1"`` rescales SRTrainer ``[0, 1]`` RGB images to ``[-1, 1]``
    before passing them to ``PPOCRv5ServerRec`` (use ``input_range="-1_1"`` for raw ``[-1, 1]``).
    """

    def __init__(
        self,
        checkpoint: str | Path = PPOCR_CHECKPOINT,
        strip_height: int = 64,
        strip_stride: int = 32,
        temperature: float = 1.0,
        input_range: str = "0_1",
    ) -> None:
        super().__init__()
        if not _is_positive_int(strip_height):
            raise ValueError(f"strip_height must be a positive integer, got {strip_height!r}.")
        if not _is_positive_int(strip_stride):
            raise ValueError(f"strip_stride must be a positive integer, got {strip_stride!r}.")
        if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
            raise ValueError(f"temperature must be a positive finite real, got {temperature!r}.")
        temp_f = float(temperature)
        if not math.isfinite(temp_f) or temp_f <= 0.0:
            raise ValueError(f"temperature must be finite and > 0, got {temperature!r}.")
        if input_range not in _VALID_INPUT_RANGES:
            raise ValueError(
                f"Unsupported input_range={input_range!r}; expected one of {sorted(_VALID_INPUT_RANGES)}."
            )

        self.checkpoint = str(checkpoint)
        self.strip_height = int(strip_height)
        self.strip_stride = int(strip_stride)
        self.temperature = temp_f
        self.input_range = input_range

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

    def logits(self, images: torch.Tensor) -> torch.Tensor:
        """Compute temperature-scaled per-frame CTC logits on horizontal strips of ``images``."""
        x = self._to_recognizer_range(images)
        strips = extract_text_strips(x, self.strip_height, self.strip_stride)
        width = max(8, round(strips.shape[-1] * self.input_height / strips.shape[-2] / 8) * 8)
        strips = F.interpolate(
            strips,
            size=(self.input_height, width),
            mode="bilinear",
            align_corners=False,
        )
        return self.recognizer(strips).float() / self.temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
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

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            with torch.no_grad():
                target_log = F.log_softmax(self.logits(target), dim=-1)
            pred_log = F.log_softmax(self.logits(pred), dim=-1)
            if pred_log.shape != target_log.shape or pred_log.ndim != 3:
                raise ValueError(
                    f"OCR recognizer must return matching (N, T, C) logits, "
                    f"got pred={tuple(pred_log.shape)}, target={tuple(target_log.shape)}."
                )
            frames = pred_log.shape[0] * pred_log.shape[1]
            kl = F.kl_div(pred_log, target_log, reduction="sum", log_target=True) / frames
        return kl * (self.temperature**2)


KLOCRLoss = OCRKLLoss
