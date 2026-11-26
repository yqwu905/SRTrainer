"""Shared tiled inference engine with auto tile sizing, 2D HR feathering, and padding guards."""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn

from config.schema import BlendMode, MixedPrecision, TileConfig
from config.validation import load_tile_config
from utils.config import normalize_mixed_precision
from utils.misc import pad_to_min_size, pad_to_multiple, resolve_2d_size, unpad_tensor

__all__ = [
    "TiledInferencer",
    "build_blend_window",
    "compute_tile_starts",
]

_VALID_BLEND_MODES: frozenset[str] = frozenset(m.value for m in BlendMode)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalize_blend_mode(blend_mode: Any) -> str:
    if isinstance(blend_mode, BlendMode):
        return blend_mode.value
    if not isinstance(blend_mode, str):
        raise ValueError(
            f"blend_mode must be a string in {sorted(_VALID_BLEND_MODES)}, "
            f"got {type(blend_mode).__name__}: {blend_mode!r}."
        )
    norm = blend_mode.strip().lower()
    if norm not in _VALID_BLEND_MODES:
        raise ValueError(
            f"Unsupported blend_mode={blend_mode!r}; expected one of {sorted(_VALID_BLEND_MODES)}."
        )
    return norm


def compute_tile_starts(dim: int, tile_dim: int, overlap: int) -> list[int]:
    """Generate sorted, deduplicated tile start indices covering ``[0, dim)``."""
    if not _is_positive_int(dim):
        raise ValueError(f"dim must be a positive integer, got {dim!r}.")
    if not _is_positive_int(tile_dim):
        raise ValueError(f"tile_dim must be a positive integer, got {tile_dim!r}.")
    if not _is_non_negative_int(overlap):
        raise ValueError(f"overlap must be a non-negative integer, got {overlap!r}.")
    if overlap >= tile_dim:
        raise ValueError(f"overlap ({overlap}) must be smaller than tile_dim ({tile_dim}).")

    if dim <= tile_dim:
        return [0]

    stride = tile_dim - overlap
    starts = list(range(0, dim - tile_dim + 1, stride))
    last_start = dim - tile_dim
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _build_1d_window(length: int, ramp: int, blend_mode: str, eps: float = 1e-3) -> torch.Tensor:
    w = torch.ones(length, dtype=torch.float32)
    if ramp <= 0 or blend_mode == "mean":
        return w
    if 2 * ramp > length:
        raise ValueError(
            f"Double ramp width (2 * {ramp} = {2 * ramp}) exceeds window length ({length})."
        )
    if ramp == 1:
        edge = torch.tensor([eps], dtype=torch.float32)
    else:
        t = torch.linspace(0.0, 1.0, steps=ramp, dtype=torch.float32)
        if blend_mode == "linear":
            edge = eps + (1.0 - eps) * t
        elif blend_mode == "cosine":
            edge = eps + (1.0 - eps) * 0.5 * (1.0 - torch.cos(math.pi * t))
        else:
            raise ValueError(f"Unsupported blend_mode={blend_mode!r}.")
    w[:ramp] = edge
    w[-ramp:] = torch.flip(edge, dims=[0])
    return w


def build_blend_window(
    tile_h: int,
    tile_w: int,
    scale: int,
    overlap: int,
    blend_mode: str | BlendMode,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Build a 2D HR feathering window of shape ``[1, 1, tile_h * scale, tile_w * scale]``."""
    if not _is_positive_int(tile_h) or not _is_positive_int(tile_w):
        raise ValueError(
            f"tile_h and tile_w must be positive integers, got tile_h={tile_h!r}, tile_w={tile_w!r}."
        )
    if not _is_positive_int(scale):
        raise ValueError(f"scale must be a positive integer, got {scale!r}.")
    if not _is_non_negative_int(overlap):
        raise ValueError(f"overlap must be a non-negative integer, got {overlap!r}.")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not (0.0 < float(eps) <= 1.0):
        raise ValueError(f"eps must be a float in (0, 1], got {eps!r}.")

    norm_mode = _normalize_blend_mode(blend_mode)
    h_hr = tile_h * scale
    w_hr = tile_w * scale
    ramp = overlap * scale

    if ramp <= 0 or norm_mode == "mean":
        return torch.ones((1, 1, h_hr, w_hr), dtype=torch.float32)

    eps_f = float(eps)
    wh = _build_1d_window(h_hr, ramp, norm_mode, eps=eps_f)
    ww = _build_1d_window(w_hr, ramp, norm_mode, eps=eps_f)
    return (wh[:, None] * ww[None, :]).view(1, 1, h_hr, w_hr).contiguous()


class TiledInferencer:
    """Shared tiled/untiled inference engine used by both validation and ``tools/infer.py``.

    Supports:
    - Auto tile size derivation (``tile_size=None`` -> per-axis ``gt_size // scale``) with graceful
      untiled fallback + warning when ``gt_size`` is unavailable.
    - ``gt_size`` as a square ``int`` or a rectangular ``[gt_h, gt_w]`` sequence (per-axis
      divisibility by ``scale`` is enforced).
    - Explicit square (``int``) or rectangular (``[tile_h, tile_w]``) tile sizes.
    - Overlap validation (``0 <= tile_overlap < min(tile_h, tile_w) // 2``; ``0`` = hard stitch).
    - 2D HR feathering windows (``cosine`` | ``linear`` | ``mean``) with FP32 accumulation.
    - Small-image and odd-dimension ``reflect`` -> ``replicate`` padding and exact HR cropping.
    """

    def __init__(
        self,
        model: nn.Module,
        scale: int,
        tile_enabled: bool = True,
        tile_size: int | Sequence[int] | None = None,
        tile_overlap: int = 16,
        tile_batch_size: int = 4,
        blend_mode: str | BlendMode = BlendMode.cosine,
        pad_multiple: int = 8,
        gt_size: int | Sequence[int] | None = None,
        mixed_precision: str | MixedPrecision = MixedPrecision.no,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}.")
        if not _is_positive_int(scale):
            raise ValueError(f"scale must be a positive integer, got {scale!r}.")
        if not isinstance(tile_enabled, bool):
            raise ValueError(
                f"tile_enabled must be a bool, got {type(tile_enabled).__name__}: {tile_enabled!r}."
            )
        if not _is_non_negative_int(tile_overlap):
            raise ValueError(
                f"tile_overlap must be a non-negative integer, got {tile_overlap!r}."
            )
        if not _is_positive_int(tile_batch_size):
            raise ValueError(
                f"tile_batch_size must be a positive integer, got {tile_batch_size!r}."
            )
        if not _is_positive_int(pad_multiple):
            raise ValueError(f"pad_multiple must be a positive integer, got {pad_multiple!r}.")

        norm_blend_mode = _normalize_blend_mode(blend_mode)
        norm_mp = normalize_mixed_precision(mixed_precision)

        resolved_gt_size: tuple[int, int] | None = None
        if gt_size is not None:
            resolved_gt_size = resolve_2d_size(gt_size, "gt_size")
            if resolved_gt_size[0] % scale != 0 or resolved_gt_size[1] % scale != 0:
                raise ValueError(
                    f"gt_size ({resolved_gt_size}) must be evenly divisible by scale ({scale})."
                )

        resolved_device: torch.device | None
        if device is None:
            resolved_device = None
        elif isinstance(device, (torch.device, str)):
            resolved_device = torch.device(device)
        else:
            raise TypeError(
                f"device must be torch.device, str, or None, got {type(device).__name__}."
            )

        self.model = model
        self.scale = scale
        self.tile_overlap = tile_overlap
        self.tile_batch_size = tile_batch_size
        self.blend_mode = norm_blend_mode
        self.pad_multiple = pad_multiple
        self.gt_size = resolved_gt_size
        self.mixed_precision = norm_mp
        self.device = resolved_device
        self._warned_amp_fallback: set[tuple[str, str]] = set()

        self.tile_enabled: bool = False
        self.tile_size: tuple[int, int] | None = None
        self.weight_window: torch.Tensor | None = None

        if not tile_enabled:
            return

        resolved_tile_size = self._resolve_tile_size(
            tile_size, gt_size=resolved_gt_size, scale=scale
        )
        if resolved_tile_size is None:
            logging.getLogger("srtrainer").warning(
                "TiledInferencer: tile_size=None (auto) requires gt_size, which is None; "
                "falling back to untiled whole-image inference with pad_multiple=%d.",
                self.pad_multiple,
            )
            return

        tile_h, tile_w = resolved_tile_size
        min_tile = min(tile_h, tile_w)
        if self.tile_overlap > 0 and self.tile_overlap >= min_tile // 2:
            raise ValueError(
                f"tile_overlap={self.tile_overlap} must satisfy "
                f"0 <= tile_overlap < min(tile_h, tile_w) // 2 ({min_tile // 2}) "
                f"for tile_size={resolved_tile_size}."
            )

        self.tile_enabled = True
        self.tile_size = resolved_tile_size
        self.weight_window = build_blend_window(
            tile_h=tile_h,
            tile_w=tile_w,
            scale=self.scale,
            overlap=self.tile_overlap,
            blend_mode=self.blend_mode,
        )

    @staticmethod
    def _resolve_tile_size(
        tile_size: int | Sequence[int] | None,
        gt_size: tuple[int, int] | None,
        scale: int,
    ) -> tuple[int, int] | None:
        if tile_size is None:
            if gt_size is None:
                return None
            return (gt_size[0] // scale, gt_size[1] // scale)

        if isinstance(tile_size, bool):
            raise ValueError(f"tile_size cannot be a bool, got {tile_size!r}.")

        if isinstance(tile_size, int):
            if tile_size < 1:
                raise ValueError(f"tile_size must be a positive integer, got {tile_size!r}.")
            return (tile_size, tile_size)

        if isinstance(tile_size, Sequence) and not isinstance(tile_size, (str, bytes, bytearray)):
            if len(tile_size) != 2:
                raise ValueError(
                    f"Sequence tile_size must have length 2 ([tile_h, tile_w]), "
                    f"got length {len(tile_size)}: {tile_size!r}."
                )
            h_raw, w_raw = tile_size[0], tile_size[1]
            if not _is_positive_int(h_raw) or not _is_positive_int(w_raw):
                raise ValueError(
                    f"tile_size elements must be positive integers, got {tile_size!r}."
                )
            return (int(h_raw), int(w_raw))

        raise ValueError(
            f"Unsupported tile_size={tile_size!r} ({type(tile_size).__name__}); "
            "expected None (auto), positive int, or 2-element sequence [tile_h, tile_w]."
        )

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        scale: int,
        tile_cfg: TileConfig | DictConfig | Mapping[str, Any] | None,
        gt_size: int | Sequence[int] | None = None,
        mixed_precision: str | MixedPrecision = MixedPrecision.no,
        device: torch.device | str | None = None,
    ) -> TiledInferencer:
        """Instantiate ``TiledInferencer`` from a typed ``TileConfig`` (or structured mapping)."""
        if tile_cfg is not None and not isinstance(tile_cfg, (TileConfig, DictConfig, Mapping)):
            raise TypeError(
                f"Expected TileConfig, DictConfig, Mapping, or None for tile_cfg, "
                f"got {type(tile_cfg).__name__}."
            )
        tile = tile_cfg if isinstance(tile_cfg, TileConfig) else load_tile_config(tile_cfg)
        return cls(
            model=model,
            scale=scale,
            tile_enabled=tile.enabled,
            tile_size=tile.size,
            tile_overlap=tile.overlap,
            tile_batch_size=tile.batch_size,
            blend_mode=tile.blend_mode,
            pad_multiple=tile.pad_multiple,
            gt_size=gt_size,
            mixed_precision=mixed_precision,
            device=device,
        )

    def _resolve_execution_device(self, lq: torch.Tensor) -> torch.device:
        if self.device is not None:
            return self.device
        first_param = next(self.model.parameters(), None)
        if first_param is not None:
            return first_param.device
        first_buf = next(self.model.buffers(), None)
        if first_buf is not None:
            return first_buf.device
        return lq.device

    def _autocast_context(self, exec_device: torch.device) -> contextlib.AbstractContextManager[Any]:
        if self.mixed_precision == "no":
            return contextlib.nullcontext()

        device_type = exec_device.type
        is_avail = torch.amp.is_autocast_available(device_type)

        if self.mixed_precision == "fp16":
            if device_type == "cpu" or not is_avail:
                key = ("fp16", device_type)
                if key not in self._warned_amp_fallback:
                    self._warned_amp_fallback.add(key)
                    logging.getLogger("srtrainer").warning(
                        "TiledInferencer: mixed_precision='fp16' is unsupported on device_type=%r; "
                        "running forward in FP32.",
                        device_type,
                    )
                return contextlib.nullcontext()
            return torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=True)

        if self.mixed_precision == "bf16":
            if device_type == "cuda" and not torch.cuda.is_bf16_supported():
                key = ("bf16", device_type)
                if key not in self._warned_amp_fallback:
                    self._warned_amp_fallback.add(key)
                    logging.getLogger("srtrainer").warning(
                        "TiledInferencer: CUDA device does not support bf16 autocast; "
                        "running forward in FP32."
                    )
                return contextlib.nullcontext()
            if not is_avail:
                key = ("bf16", device_type)
                if key not in self._warned_amp_fallback:
                    self._warned_amp_fallback.add(key)
                    logging.getLogger("srtrainer").warning(
                        "TiledInferencer: mixed_precision='bf16' is unavailable on device_type=%r; "
                        "running forward in FP32.",
                        device_type,
                    )
                return contextlib.nullcontext()
            return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)

        return contextlib.nullcontext()

    def _run_untiled(self, x_dev: torch.Tensor, exec_device: torch.device) -> torch.Tensor:
        lq_pad, (orig_h, orig_w) = pad_to_multiple(x_dev, self.pad_multiple, mode="reflect")
        batch_size = lq_pad.shape[0]
        pad_h, pad_w = int(lq_pad.shape[-2]), int(lq_pad.shape[-1])
        expected_h = pad_h * self.scale
        expected_w = pad_w * self.scale

        with self._autocast_context(exec_device):
            pred = self.model(lq_pad)

        if not isinstance(pred, torch.Tensor):
            raise RuntimeError(
                f"Expected model to return torch.Tensor, got {type(pred).__name__}."
            )
        if (
            pred.ndim != 4
            or pred.shape[0] != batch_size
            or pred.shape[-2:] != (expected_h, expected_w)
        ):
            raise RuntimeError(
                f"Expected model output shape ({batch_size}, C_out, {expected_h}, {expected_w}), "
                f"got {tuple(pred.shape)}."
            )

        pred_fp32 = pred.to(device=exec_device, dtype=torch.float32)
        return unpad_tensor(pred_fp32, orig_h, orig_w, scale=self.scale).clamp(0.0, 1.0)

    def _run_tiled(self, x_dev: torch.Tensor, exec_device: torch.device) -> torch.Tensor:
        if self.tile_size is None or self.weight_window is None:
            raise RuntimeError("Tiled execution requested without initialized tile_size/window.")

        tile_h, tile_w = self.tile_size
        lq_pad, (orig_h, orig_w) = pad_to_min_size(x_dev, tile_h, tile_w, mode="reflect")
        batch_size = int(lq_pad.shape[0])
        pad_h, pad_w = int(lq_pad.shape[-2]), int(lq_pad.shape[-1])

        y_starts = compute_tile_starts(pad_h, tile_h, self.tile_overlap)
        x_starts = compute_tile_starts(pad_w, tile_w, self.tile_overlap)
        coords = [(b, y, x) for b in range(batch_size) for y in y_starts for x in x_starts]

        tile_h_hr = tile_h * self.scale
        tile_w_hr = tile_w * self.scale
        weight_2d = self.weight_window.to(device=exec_device, dtype=torch.float32).squeeze(0)

        output_acc: torch.Tensor | None = None
        weight_acc: torch.Tensor | None = None

        for start_idx in range(0, len(coords), self.tile_batch_size):
            chunk = coords[start_idx : start_idx + self.tile_batch_size]
            tile_batch = torch.stack(
                [lq_pad[b, :, y : y + tile_h, x : x + tile_w] for b, y, x in chunk],
                dim=0,
            ).contiguous()

            with self._autocast_context(exec_device):
                pred_batch = self.model(tile_batch)

            if not isinstance(pred_batch, torch.Tensor):
                raise RuntimeError(
                    f"Expected model to return torch.Tensor, got {type(pred_batch).__name__}."
                )
            if (
                pred_batch.ndim != 4
                or pred_batch.shape[0] != len(chunk)
                or pred_batch.shape[-2:] != (tile_h_hr, tile_w_hr)
            ):
                raise RuntimeError(
                    f"Expected model tile output shape "
                    f"({len(chunk)}, C_out, {tile_h_hr}, {tile_w_hr}), "
                    f"got {tuple(pred_batch.shape)}."
                )

            pred_batch_fp32 = pred_batch.to(device=exec_device, dtype=torch.float32)

            if output_acc is None or weight_acc is None:
                c_out = int(pred_batch_fp32.shape[1])
                output_acc = torch.zeros(
                    (batch_size, c_out, pad_h * self.scale, pad_w * self.scale),
                    device=exec_device,
                    dtype=torch.float32,
                )
                weight_acc = torch.zeros(
                    (batch_size, 1, pad_h * self.scale, pad_w * self.scale),
                    device=exec_device,
                    dtype=torch.float32,
                )
            elif int(pred_batch_fp32.shape[1]) != int(output_acc.shape[1]):
                raise RuntimeError(
                    f"Inconsistent output channel count across tiles: expected "
                    f"{output_acc.shape[1]}, got {pred_batch_fp32.shape[1]}."
                )

            for idx, (b, y, x) in enumerate(chunk):
                y_hr = y * self.scale
                x_hr = x * self.scale
                output_acc[b, :, y_hr : y_hr + tile_h_hr, x_hr : x_hr + tile_w_hr] += (
                    pred_batch_fp32[idx] * weight_2d
                )
                weight_acc[b, :, y_hr : y_hr + tile_h_hr, x_hr : x_hr + tile_w_hr] += weight_2d

        if output_acc is None or weight_acc is None:
            raise RuntimeError("No tiles were generated during tiled inference.")

        out = (output_acc / weight_acc).clamp(0.0, 1.0)
        return unpad_tensor(out, orig_h, orig_w, scale=self.scale)

    def infer_tensor(self, lq: torch.Tensor) -> torch.Tensor:
        """Run tiled or untiled inference on a float tensor ``[C, H, W]`` or ``[B, C, H, W]``.

        Returns a contiguous ``float32`` tensor clamped to ``[0, 1]`` with exact spatial
        shape ``(H * scale, W * scale)`` and preserved batch rank. Preserves
        ``self.model.training`` on both normal exit and exceptions.
        """
        if not isinstance(lq, torch.Tensor):
            raise TypeError(f"Expected lq to be a torch.Tensor, got {type(lq).__name__}.")
        if not lq.is_floating_point():
            raise TypeError(
                f"Expected floating-point lq tensor in [0, 1], got dtype {lq.dtype}."
            )
        if lq.ndim not in (3, 4):
            raise ValueError(
                f"Expected 3D [C, H, W] or 4D [B, C, H, W] lq tensor, got shape {tuple(lq.shape)}."
            )

        squeeze_batch = lq.ndim == 3
        x = lq.unsqueeze(0) if squeeze_batch else lq
        if any(dim < 1 for dim in x.shape):
            raise ValueError(
                f"All lq tensor dimensions must be >= 1, got shape {tuple(lq.shape)}."
            )

        exec_device = self._resolve_execution_device(x)
        x_dev = x.to(device=exec_device, dtype=torch.float32)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                if self.tile_enabled:
                    out = self._run_tiled(x_dev, exec_device)
                else:
                    out = self._run_untiled(x_dev, exec_device)
        finally:
            if was_training:
                self.model.train(True)

        return out.squeeze(0).contiguous() if squeeze_batch else out.contiguous()
