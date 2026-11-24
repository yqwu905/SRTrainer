"""GPU masked 2nd-order Real-ESRGAN degradation pipeline, filter2d, USMSharp, and TrainingPool."""

from __future__ import annotations

import math
import numbers
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from torch import nn
from torch.nn import functional as F

from data.degradation.diffjpeg import DiffJPEG

__all__ = [
    "filter2d",
    "USMSharp",
    "TrainingPool",
    "RealESRGANDegrader",
]

_INTERP_MODES: tuple[str, ...] = ("area", "bilinear", "bicubic")


def _normalize_omegaconf(obj: Any, context: str) -> Any:
    if OmegaConf.is_config(obj):
        try:
            return OmegaConf.to_container(obj, resolve=True, throw_on_missing=True)
        except OmegaConfBaseException as exc:
            raise ValueError(f"Invalid OmegaConf value for {context}: {exc}") from exc
    return obj


def _validate_int(value: Any, name: str, min_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {min_value}, got {value!r}.")
    int_val = int(value)
    if int_val < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {int_val}.")
    return int_val


def _validate_scalar_prob(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {value!r}.")
    val = float(value)
    if not math.isfinite(val) or val < 0.0 or val > 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0], got {value!r}.")
    return val


def _to_1d_list(obj: Any, context: str) -> list[Any]:
    normalized = _normalize_omegaconf(obj, context)
    if isinstance(normalized, np.ndarray):
        if normalized.ndim != 1:
            raise ValueError(
                f"{context} must be a 1D sequence, got ndarray with shape {normalized.shape}."
            )
        return normalized.tolist()
    if isinstance(normalized, (str, bytes, Mapping)) or not isinstance(normalized, Sequence):
        raise ValueError(f"{context} must be a sequence, got {type(obj).__name__}.")
    return list(normalized)


def _validate_range(
    rng_pair: Sequence[float],
    name: str,
    min_bound: float | None = None,
    max_bound: float | None = None,
    strictly_positive: bool = False,
) -> tuple[float, float]:
    normalized = _to_1d_list(rng_pair, name)
    if len(normalized) != 2:
        raise ValueError(f"{name} must have length 2, got length {len(normalized)}.")

    vals: list[float] = []
    for idx, v in enumerate(normalized):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"{name}[{idx}] must be a real number, got {v!r}.")
        vf = float(v)
        if not math.isfinite(vf):
            raise ValueError(f"{name}[{idx}] must be finite, got {v!r}.")
        vals.append(vf)

    lo, hi = vals[0], vals[1]
    if lo > hi:
        raise ValueError(f"{name} requires low <= high, got ({lo}, {hi}).")
    if strictly_positive and lo <= 0.0:
        raise ValueError(f"{name} requires strictly positive bounds, got ({lo}, {hi}).")
    if min_bound is not None and lo < min_bound:
        raise ValueError(f"{name} requires low >= {min_bound}, got ({lo}, {hi}).")
    if max_bound is not None and hi > max_bound:
        raise ValueError(f"{name} requires high <= {max_bound}, got ({lo}, {hi}).")
    return lo, hi


def _validate_prob_triple(probs: Sequence[float], name: str) -> tuple[float, float, float]:
    normalized = _to_1d_list(probs, name)
    if len(normalized) != 3:
        raise ValueError(f"{name} must have length 3, got length {len(normalized)}.")

    clean: list[float] = []
    for idx, p in enumerate(normalized):
        if isinstance(p, bool) or not isinstance(p, numbers.Real):
            raise ValueError(f"{name}[{idx}] must be a real number, got {p!r}.")
        pf = float(p)
        if not math.isfinite(pf) or pf < 0.0:
            raise ValueError(f"{name}[{idx}] must be finite and >= 0.0, got {p!r}.")
        clean.append(pf)

    total = sum(clean)
    if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(f"{name} must sum to 1.0 (within 1e-4), got sum={total} for {clean!r}.")
    return clean[0] / total, clean[1] / total, clean[2] / total


def _slice_kernel(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    target_device: torch.device,
) -> torch.Tensor:
    """Slice a kernel tensor with a device-aligned boolean mask and move to target_device in FP32."""
    mask_k = mask if mask.device == tensor.device else mask.to(device=tensor.device)
    return tensor[mask_k].to(device=target_device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# NPU depthwise-conv fast path limits (measured on Ascend 910B3 / torch_npu 2.9)
#
# ``F.conv2d`` with ``groups == C`` (i.e. cin=1 per group) has a fast AICore path
# only for *small* kernels and *small* spatial extents:
#
#   kernel 3x3, 5x5  @ any H,W        -> 1-2 ms @ 2048^2
#   kernel 7x7+      @ 2048^2         -> 1112 ms  (7x7), 2756 ms (11x11)
#   kernel 21x21     @ H,W <= 640     -> 2 ms
#   kernel 21x21     @ 768^2 / 1024^2 -> 1404 ms / 2494 ms
#   kernel 21x21     @ 2048^2         -> 9970 ms
#
# i.e. beyond these limits the op silently falls into a ~3 orders-of-magnitude
# slower path (per-tile im2col / AICPU style), which made the 2048^2 Real-ESRGAN
# degradation the dominant cost of a training step. Tiling the *output* and
# keeping every conv call inside the fast window restores the fast path while
# producing bit-identical results (disjoint output tiles on the padded input).
# Doc/limit source: see ``filter2d`` docstring + ``docs/`` perf notes.
_DEPTHWISE_FAST_MAX_KERNEL: int = 5
_DEPTHWISE_FAST_MAX_HW: int = 640
_DEPTHWISE_TILE: int = 512


def _depthwise_conv(img_pad: torch.Tensor, weight: torch.Tensor, b: int, c: int) -> torch.Tensor:
    """Grouped (per-channel) conv2d on an already padded ``[B, C, Hp, Wp]`` tensor."""
    hp, wp = int(img_pad.shape[-2]), int(img_pad.shape[-1])
    img_grouped = img_pad.reshape(1, b * c, hp, wp)
    return F.conv2d(img_grouped, weight, groups=b * c)


def _needs_tiling(device_type: str, hp: int, wp: int, kh: int, kw: int) -> bool:
    """Return True when the depthwise conv must be split to stay on the NPU fast path."""
    if device_type != "npu":
        return False
    if max(kh, kw) <= _DEPTHWISE_FAST_MAX_KERNEL and max(hp, wp) <= _DEPTHWISE_FAST_MAX_HW:
        return False
    return max(hp, wp) > _DEPTHWISE_FAST_MAX_HW or max(kh, kw) > _DEPTHWISE_FAST_MAX_KERNEL


def _filter2d_tiled(
    img_pad: torch.Tensor,
    weight: torch.Tensor,
    b: int,
    c: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
) -> torch.Tensor:
    """Exact tiled depthwise conv: disjoint output tiles, halo taken from the padded input."""
    # Conv input window is tile + (k - 1); keep it below the fast-path extent.
    tile = min(_DEPTHWISE_TILE, max(64, _DEPTHWISE_FAST_MAX_HW - (max(kh, kw) - 1)))
    out = img_pad.new_empty((b, c, h, w))
    for i0 in range(0, h, tile):
        i1 = min(i0 + tile, h)
        for j0 in range(0, w, tile):
            j1 = min(j0 + tile, w)
            window = img_pad[:, :, i0 : i1 + kh - 1, j0 : j1 + kw - 1]
            # _depthwise_conv 返回 [1, b*c, h_tile, w_tile]（batch 展平进 channel），
            # 必须先还原成 [b, c, h_tile, w_tile] 再写回 out；否则 b>1 时维度对不上
            # （b=1 时靠 broadcast 侥幸通过，但语义也不对）。
            out[:, :, i0:i1, j0:j1] = _depthwise_conv(window.contiguous(), weight, b, c).view(
                b, c, i1 - i0, j1 - j0
            )
    return out


def filter2d(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply per-sample 2D spatial filtering via grouped ``F.conv2d`` with reflect->replicate padding.

    On NPU the conv is automatically split into disjoint output tiles so that every
    ``F.conv2d`` call stays inside the AICore fast window (small kernel / small
    spatial extent); the result is numerically identical to the untiled version.

    Args:
        img: 4D tensor of shape ``[B, C, H, W]``.
        kernel: 3D tensor of shape ``[B, K_h, K_w]`` or 2D tensor ``[K_h, K_w]`` broadcast across ``B``.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected img to be a torch.Tensor, got {type(img).__name__}.")
    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Expected kernel to be a torch.Tensor, got {type(kernel).__name__}.")
    if img.ndim != 4:
        raise ValueError(f"Expected 4D img tensor [B, C, H, W], got shape {tuple(img.shape)}.")

    b, c, h, w = img.shape
    if b < 1 or c < 1 or h < 1 or w < 1:
        raise ValueError(f"Expected non-empty img tensor, got shape {tuple(img.shape)}.")

    if kernel.ndim == 2:
        k = kernel.unsqueeze(0).expand(b, -1, -1)
    elif kernel.ndim == 3:
        if kernel.shape[0] != b:
            raise ValueError(
                f"Kernel batch size ({kernel.shape[0]}) must match img batch size ({b})."
            )
        k = kernel
    else:
        raise ValueError(
            f"Expected 2D [K_h, K_w] or 3D [B, K_h, K_w] kernel tensor, got shape {tuple(kernel.shape)}."
        )

    kh, kw = int(k.shape[-2]), int(k.shape[-1])
    if kh < 1 or kw < 1:
        raise ValueError(f"Kernel dimensions must be positive, got {(kh, kw)}.")

    k = k.to(device=img.device, dtype=img.dtype)
    ph1, ph2 = (kh - 1) // 2, kh // 2
    pw1, pw2 = (kw - 1) // 2, kw // 2
    pad_mode = "replicate" if (pw1 >= w or pw2 >= w or ph1 >= h or ph2 >= h) else "reflect"
    img_pad = F.pad(img, (pw1, pw2, ph1, ph2), mode=pad_mode)
    hp, wp = h + kh - 1, w + kw - 1
    weight = k.unsqueeze(1).expand(b, c, kh, kw).reshape(b * c, 1, kh, kw).contiguous()

    if _needs_tiling(img.device.type, hp, wp, kh, kw):
        return _filter2d_tiled(img_pad, weight, b, c, h, w, kh, kw)

    out = _depthwise_conv(img_pad, weight, b, c)
    return out.view(b, c, h, w)


def _poisson_unit_noise(img_c: torch.Tensor, vals: float) -> torch.Tensor:
    """Return ``torch.poisson(img_c * vals) / vals - img_c`` without leaving the device.

    ``aten::poisson`` has no NPU kernel: on Ascend it silently falls back to CPU
    (a full D2H -> CPU sample -> H2D round-trip), measured at ~0.7 s per 2048^2 RGB
    call — i.e. ~1.4 s per training step with the 2nd-order pipeline.

    For large rates (here ``lambda = img_c * 256 >= 30`` for any pixel above 0.12)
    a Gaussian with the *same first two moments* is an excellent approximation of
    Poisson:  ``Poisson(l) ~ l + sqrt(l) * N(0,1)``, relative distribution error
    O(1/sqrt(l)) <= 0.12 quantization level at l=40 and < 0.02 level at l=256.
    Dividing by ``vals`` and subtracting ``img_c`` gives exactly
    ``randn * sqrt(img_c / vals)`` with mean 0 and variance ``img_c / vals``,
    matching the original shot-noise statistics to first order.
    """
    if img_c.device.type == "npu":
        return torch.randn_like(img_c) * torch.sqrt(img_c.clamp_min(0.0) / vals)
    return torch.poisson(img_c * vals) / vals - img_c


class USMSharp(nn.Module):
    """Pure-PyTorch unsharp masking (USM) sharpener using separable 1D Gaussian convolutions."""

    kernel_1d: torch.Tensor

    def __init__(self, radius: int = 50, sigma: float = 0.0) -> None:
        super().__init__()
        self.radius = _validate_int(radius, "radius", min_value=1)
        if isinstance(sigma, bool) or not isinstance(sigma, numbers.Real):
            raise ValueError(f"sigma must be a finite float >= 0.0, got {sigma!r}.")
        sigma_f = float(sigma)
        if not math.isfinite(sigma_f) or sigma_f < 0.0:
            raise ValueError(f"sigma must be finite and >= 0.0, got {sigma!r}.")
        self.sigma = sigma_f

        sigma_eff = 0.3 * (self.radius - 1) + 0.8 if sigma_f <= 0.0 else sigma_f
        coords = torch.arange(-self.radius, self.radius + 1, dtype=torch.float64)
        g1d = torch.exp(-0.5 * (coords / sigma_eff) ** 2)
        g1d = (g1d / g1d.sum()).to(dtype=torch.float32)
        self.register_buffer("kernel_1d", g1d, persistent=True)

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        r = self.radius
        pad_mode = "replicate" if (r >= h or r >= w) else "reflect"
        padded = F.pad(x, (r, r, r, r), mode=pad_mode)
        k1d = self.kernel_1d.to(device=x.device, dtype=torch.float32)
        k_h = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
        k_v = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
        out = F.conv2d(padded, k_h, groups=c)
        return F.conv2d(out, k_v, groups=c)

    def forward(
        self,
        img: torch.Tensor,
        weight: float = 0.5,
        threshold: int = 10,
    ) -> torch.Tensor:
        """Apply FP32 unsharp masking to a 4D ``[B, C, H, W]`` tensor in ``[0, 1]``."""
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected img to be a torch.Tensor, got {type(img).__name__}.")
        if (
            img.ndim != 4
            or img.shape[0] < 1
            or img.shape[1] < 1
            or img.shape[2] < 1
            or img.shape[3] < 1
        ):
            raise ValueError(f"Expected non-empty 4D tensor [B, C, H, W], got {tuple(img.shape)}.")
        if isinstance(weight, bool) or not isinstance(weight, numbers.Real):
            raise ValueError(f"weight must be a finite real number, got {weight!r}.")
        weight_f = float(weight)
        if not math.isfinite(weight_f):
            raise ValueError(f"weight must be finite, got {weight!r}.")
        if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
            raise ValueError(f"threshold must be a finite real number, got {threshold!r}.")
        threshold_f = float(threshold)
        if not math.isfinite(threshold_f):
            raise ValueError(f"threshold must be finite, got {threshold!r}.")

        with torch.amp.autocast(device_type=img.device.type, enabled=False):
            img_f = img.float()
            blur = self._blur(img_f)
            residual = img_f - blur
            mask = (torch.abs(residual) * 255.0 > threshold_f).to(dtype=torch.float32)
            soft_mask = self._blur(mask)
            sharp = torch.clamp(img_f + weight_f * residual, 0.0, 1.0)
            return torch.clamp(soft_mask * sharp + (1.0 - soft_mask) * img_f, 0.0, 1.0)


class TrainingPool:
    """Post-degradation FIFO/shuffled pair buffer to increase intra-batch degradation diversity."""

    def __init__(self, queue_size: int) -> None:
        self.queue_size: int = _validate_int(queue_size, "queue_size", min_value=1)
        self.queue_lr: torch.Tensor | None = None
        self.queue_gt: torch.Tensor | None = None
        self.queue_ptr: int = 0

    def reset(self) -> None:
        """Clear internal pool buffers and reset pointer."""
        self.queue_lr = None
        self.queue_gt = None
        self.queue_ptr = 0

    def dequeue_and_enqueue(
        self,
        lq: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Enqueue ``(lq, gt)`` and return either live batch (warm-up) or shuffled historical pairs."""
        if not isinstance(lq, torch.Tensor) or not isinstance(gt, torch.Tensor):
            raise TypeError("lq and gt must be torch.Tensor instances.")
        if lq.ndim != 4 or gt.ndim != 4:
            raise ValueError(
                f"Expected 4D [B, C, H, W] tensors, got lq={tuple(lq.shape)}, gt={tuple(gt.shape)}."
            )
        b = int(lq.shape[0])
        if b < 1 or int(gt.shape[0]) != b:
            raise ValueError(
                f"Expected matching non-zero batch sizes, got lq={b}, gt={int(gt.shape[0])}."
            )
        if self.queue_size % b != 0:
            raise ValueError(
                f"TrainingPool queue_size ({self.queue_size}) must be divisible by batch size ({b})."
            )

        if self.queue_lr is None or self.queue_gt is None:
            self.queue_lr = torch.zeros(
                (self.queue_size, *lq.shape[1:]), device=lq.device, dtype=lq.dtype
            )
            self.queue_gt = torch.zeros(
                (self.queue_size, *gt.shape[1:]), device=gt.device, dtype=gt.dtype
            )
            self.queue_ptr = 0
        else:
            if tuple(lq.shape[1:]) != tuple(self.queue_lr.shape[1:]) or tuple(gt.shape[1:]) != tuple(
                self.queue_gt.shape[1:]
            ):
                raise ValueError(
                    f"Incoming tensor shapes lq={tuple(lq.shape[1:])}, gt={tuple(gt.shape[1:])} "
                    f"do not match pool shapes lq={tuple(self.queue_lr.shape[1:])}, "
                    f"gt={tuple(self.queue_gt.shape[1:])}."
                )
            if self.queue_ptr < self.queue_size and self.queue_ptr + b > self.queue_size:
                raise ValueError(
                    f"Batch size {b} exceeds remaining warm-up capacity "
                    f"({self.queue_size - self.queue_ptr}) in "
                    f"TrainingPool(queue_size={self.queue_size})."
                )
            if self.queue_lr.device != lq.device or self.queue_lr.dtype != lq.dtype:
                self.queue_lr = self.queue_lr.to(device=lq.device, dtype=lq.dtype)
            if self.queue_gt.device != gt.device or self.queue_gt.dtype != gt.dtype:
                self.queue_gt = self.queue_gt.to(device=gt.device, dtype=gt.dtype)

        if self.queue_ptr < self.queue_size:
            out_lq = lq.detach().clone()
            out_gt = gt.detach().clone()
            self.queue_lr[self.queue_ptr : self.queue_ptr + b] = out_lq.clone()
            self.queue_gt[self.queue_ptr : self.queue_ptr + b] = out_gt.clone()
            self.queue_ptr += b
            return out_lq, out_gt

        idx = torch.randperm(self.queue_size, device=lq.device)
        idx_gt = idx if self.queue_gt.device == idx.device else idx.to(device=self.queue_gt.device)
        self.queue_lr = self.queue_lr[idx]
        self.queue_gt = self.queue_gt[idx_gt]

        out_lq = self.queue_lr[:b].detach().clone()
        out_gt = self.queue_gt[:b].detach().clone()

        self.queue_lr[:b] = lq.detach().clone()
        self.queue_gt[:b] = gt.detach().clone()
        return out_lq, out_gt


class RealESRGANDegrader(nn.Module):
    """GPU masked 2nd-order Real-ESRGAN online degradation pipeline (forced no_grad + FP32)."""

    def __init__(
        self,
        scale: int = 1,
        use_usm: bool = False,
        queue_size: int = 0,
        kernel_cfg: Mapping[str, Any] | None = None,
        # 1st-order degradation params
        resize_prob: Sequence[float] = (0.2, 0.7, 0.1),
        resize_range: Sequence[float] = (0.15, 1.5),
        gaussian_noise_prob: float = 0.5,
        noise_range: Sequence[float] = (1.0, 30.0),
        poisson_scale_range: Sequence[float] = (0.05, 3.0),
        gray_noise_prob: float = 0.4,
        jpeg_range: Sequence[float] = (30.0, 95.0),
        # 2nd-order degradation params
        second_blur_prob: float = 0.8,
        resize_prob2: Sequence[float] = (0.3, 0.4, 0.3),
        resize_range2: Sequence[float] = (0.3, 1.2),
        gaussian_noise_prob2: float = 0.5,
        noise_range2: Sequence[float] = (1.0, 25.0),
        poisson_scale_range2: Sequence[float] = (0.05, 2.5),
        gray_noise_prob2: float = 0.4,
        jpeg_range2: Sequence[float] = (30.0, 95.0),
        # Optional USM hyperparameters
        usm_radius: int = 50,
        usm_sigma: float = 0.0,
        usm_weight: float = 0.5,
        usm_threshold: int = 10,
    ) -> None:
        super().__init__()
        self.scale: int = _validate_int(scale, "scale", min_value=1)
        if not isinstance(use_usm, bool):
            raise TypeError(f"use_usm must be a bool, got {type(use_usm).__name__}.")
        self.use_usm: bool = use_usm
        self.queue_size: int = _validate_int(queue_size, "queue_size", min_value=0)

        if kernel_cfg is None:
            self.kernel_cfg: dict[str, Any] | None = None
        else:
            norm_kcfg = _normalize_omegaconf(kernel_cfg, "kernel_cfg")
            if not isinstance(norm_kcfg, Mapping):
                raise TypeError(
                    f"kernel_cfg must be a mapping or None, got {type(norm_kcfg).__name__}."
                )
            self.kernel_cfg = dict(norm_kcfg)

        # 1st-order parameters
        self.resize_prob = _validate_prob_triple(resize_prob, "resize_prob")
        self.resize_range = _validate_range(resize_range, "resize_range", strictly_positive=True)
        self.gaussian_noise_prob = _validate_scalar_prob(
            gaussian_noise_prob, "gaussian_noise_prob"
        )
        self.noise_range = _validate_range(noise_range, "noise_range", min_bound=0.0)
        self.poisson_scale_range = _validate_range(
            poisson_scale_range, "poisson_scale_range", min_bound=0.0
        )
        self.gray_noise_prob = _validate_scalar_prob(gray_noise_prob, "gray_noise_prob")
        self.jpeg_range = _validate_range(
            jpeg_range, "jpeg_range", min_bound=1.0, max_bound=100.0
        )

        # 2nd-order parameters
        self.second_blur_prob = _validate_scalar_prob(second_blur_prob, "second_blur_prob")
        self.resize_prob2 = _validate_prob_triple(resize_prob2, "resize_prob2")
        self.resize_range2 = _validate_range(
            resize_range2, "resize_range2", strictly_positive=True
        )
        self.gaussian_noise_prob2 = _validate_scalar_prob(
            gaussian_noise_prob2, "gaussian_noise_prob2"
        )
        self.noise_range2 = _validate_range(noise_range2, "noise_range2", min_bound=0.0)
        self.poisson_scale_range2 = _validate_range(
            poisson_scale_range2, "poisson_scale_range2", min_bound=0.0
        )
        self.gray_noise_prob2 = _validate_scalar_prob(gray_noise_prob2, "gray_noise_prob2")
        self.jpeg_range2 = _validate_range(
            jpeg_range2, "jpeg_range2", min_bound=1.0, max_bound=100.0
        )

        # USM & Pool & DiffJPEG submodules
        if isinstance(usm_weight, bool) or not isinstance(usm_weight, numbers.Real):
            raise ValueError(f"usm_weight must be a finite real number, got {usm_weight!r}.")
        usm_weight_f = float(usm_weight)
        if not math.isfinite(usm_weight_f):
            raise ValueError(f"usm_weight must be finite, got {usm_weight!r}.")
        self.usm_weight: float = usm_weight_f
        self.usm_threshold: int = _validate_int(usm_threshold, "usm_threshold", min_value=0)

        self.jpeger = DiffJPEG(differentiable=False)
        self.usm_sharpener: USMSharp | None = (
            USMSharp(radius=usm_radius, sigma=usm_sigma) if self.use_usm else None
        )
        self.pool: TrainingPool | None = (
            TrainingPool(self.queue_size) if self.queue_size > 0 else None
        )

    @staticmethod
    def _random_resize(
        x: torch.Tensor,
        ref_h: int,
        ref_w: int,
        resize_prob: tuple[float, float, float],
        resize_range: tuple[float, float],
    ) -> torch.Tensor:
        updown_type = random.choices(("up", "down", "keep"), weights=resize_prob, k=1)[0]
        lo, hi = resize_range
        if updown_type == "up":
            scale_factor = random.uniform(1.0, hi) if hi >= 1.0 else random.uniform(lo, hi)
        elif updown_type == "down":
            scale_factor = (
                random.uniform(lo, min(1.0, hi)) if lo <= 1.0 else random.uniform(lo, hi)
            )
        else:
            scale_factor = 1.0

        new_h = max(1, int(round(ref_h * scale_factor)))
        new_w = max(1, int(round(ref_w * scale_factor)))
        if updown_type == "keep" and new_h == x.shape[-2] and new_w == x.shape[-1]:
            return x
        mode = random.choice(_INTERP_MODES)
        return F.interpolate(x, size=(new_h, new_w), mode=mode)

    @staticmethod
    def _add_random_noise(
        x: torch.Tensor,
        gaussian_noise_prob: float,
        noise_range: tuple[float, float],
        poisson_scale_range: tuple[float, float],
        gray_noise_prob: float,
    ) -> torch.Tensor:
        b, _, h, w = x.shape
        device = x.device
        gray_mask = torch.rand((b, 1, 1, 1), device=device) < gray_noise_prob

        if random.random() < gaussian_noise_prob:
            sigma = (
                torch.empty((b, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
                    noise_range[0], noise_range[1]
                )
                / 255.0
            )
            noise_rgb = torch.randn_like(x) * sigma
            noise_gray = (
                torch.randn((b, 1, h, w), device=device, dtype=torch.float32).expand_as(x) * sigma
            )
            return (x + torch.where(gray_mask, noise_gray, noise_rgb)).clamp(0.0, 1.0)

        # Clamp strictly to [0, 1] before Poisson so bicubic negative undershoot never NaNs.
        img_c = x.clamp(0.0, 1.0)
        vals = 256.0
        gray = (
            0.299 * img_c[:, 0:1] + 0.587 * img_c[:, 1:2] + 0.114 * img_c[:, 2:3]
        ).clamp(0.0, 1.0)
        noise_rgb = _poisson_unit_noise(img_c, vals)
        noise_gray = _poisson_unit_noise(gray, vals).expand_as(img_c)
        scale = torch.empty((b, 1, 1, 1), device=device, dtype=torch.float32).uniform_(
            poisson_scale_range[0], poisson_scale_range[1]
        )
        return (img_c + torch.where(gray_mask, noise_gray, noise_rgb) * scale).clamp(0.0, 1.0)

    @torch.no_grad()
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply masked 2nd-order degradation to ``batch['need_degradation'] == True`` samples."""
        if not isinstance(batch, Mapping):
            raise TypeError(f"Expected batch to be a mapping, got {type(batch).__name__}.")
        for req_key in ("gt", "lq", "need_degradation"):
            if req_key not in batch:
                raise KeyError(f"Batch is missing required key {req_key!r}.")

        gt = batch["gt"]
        lq = batch["lq"]
        raw_mask = batch["need_degradation"]

        if not isinstance(gt, torch.Tensor) or not isinstance(lq, torch.Tensor):
            raise TypeError("batch['gt'] and batch['lq'] must be torch.Tensor instances.")
        if not isinstance(raw_mask, torch.Tensor):
            raise TypeError("batch['need_degradation'] must be a torch.Tensor.")

        if gt.ndim != 4 or gt.shape[1] != 3:
            raise ValueError(f"Expected batch['gt'] of shape [B, 3, H, W], got {tuple(gt.shape)}.")
        b, _, gt_h, gt_w = gt.shape
        if b < 1 or gt_h < self.scale or gt_w < self.scale:
            raise ValueError(
                f"batch['gt'] shape {tuple(gt.shape)} is too small for scale={self.scale}."
            )
        if gt_h % self.scale != 0 or gt_w % self.scale != 0:
            raise ValueError(
                f"batch['gt'] spatial dims {(gt_h, gt_w)} must be divisible by scale={self.scale}."
            )

        target_h, target_w = gt_h // self.scale, gt_w // self.scale
        if lq.ndim != 4 or tuple(lq.shape) != (b, 3, target_h, target_w):
            raise ValueError(
                f"Expected batch['lq'] of shape {(b, 3, target_h, target_w)}, "
                f"got {tuple(lq.shape)}."
            )

        if raw_mask.dtype != torch.bool:
            raise ValueError(
                f"Expected batch['need_degradation'] to have dtype torch.bool, "
                f"got {raw_mask.dtype}."
            )
        valid_mask_shape = (
            raw_mask.shape == (b,)
            or raw_mask.shape == (b, 1)
            or (b == 1 and raw_mask.ndim == 0)
        )
        if not valid_mask_shape:
            expected_shape = f"({b},), ({b}, 1), or ()" if b == 1 else f"({b},) or ({b}, 1)"
            raise ValueError(
                f"Expected batch['need_degradation'] of shape {expected_shape}, "
                f"got {tuple(raw_mask.shape)}."
            )

        device = gt.device
        if self.pool is not None and "text_lines" in batch:
            raise ValueError(
                "RealESRGANDegrader TrainingPool (queue_size > 0) cannot be combined with "
                "per-sample text_lines/glyph annotations because pooling swaps samples across batches."
            )
        mask = raw_mask.to(device=device).reshape(b)
        with torch.amp.autocast(device_type=device.type, enabled=False):
            # Zero-overhead passthrough when no sample needs degradation.
            if not bool(mask.any()):
                gt_out = gt.detach() if gt.dtype == torch.float32 else gt.detach().float()
                lq_out = (
                    lq.detach()
                    if (lq.dtype == torch.float32 and lq.device == device)
                    else lq.detach().to(device=device, dtype=torch.float32)
                )
                if self.pool is not None:
                    lq_out, gt_out = self.pool.dequeue_and_enqueue(lq_out, gt_out)
                return {**batch, "lq": lq_out, "gt": gt_out}

            for k_name in ("kernel1", "kernel2", "sinc_kernel"):
                if k_name not in batch:
                    raise KeyError(
                        f"Batch requires {k_name!r} when any sample has need_degradation=True."
                    )
                kt = batch[k_name]
                if not isinstance(kt, torch.Tensor) or kt.ndim != 3 or kt.shape[0] != b:
                    raise ValueError(
                        f"Expected batch[{k_name!r}] to be a 3D tensor [B, K, K] with B={b}, "
                        f"got {tuple(kt.shape) if isinstance(kt, torch.Tensor) else type(kt).__name__}."
                    )

            gt_out = gt.detach().float().clone()
            lq_out = lq.detach().to(device=device, dtype=torch.float32).clone()

            gt_sub = gt_out[mask]
            b_sub = int(gt_sub.shape[0])
            k1 = _slice_kernel(batch["kernel1"], mask, device)
            k2 = _slice_kernel(batch["kernel2"], mask, device)
            sk = _slice_kernel(batch["sinc_kernel"], mask, device)

            # Optional USM sharpening on GT[mask] only (GT[~mask] stays bit-exact).
            if self.usm_sharpener is not None:
                gt_out[mask] = self.usm_sharpener(
                    gt_sub, weight=self.usm_weight, threshold=self.usm_threshold
                )

            # -------- 1st-order degradation --------
            out = filter2d(gt_sub, k1)
            out = self._random_resize(
                out,
                ref_h=gt_h,
                ref_w=gt_w,
                resize_prob=self.resize_prob,
                resize_range=self.resize_range,
            )
            out = self._add_random_noise(
                out,
                gaussian_noise_prob=self.gaussian_noise_prob,
                noise_range=self.noise_range,
                poisson_scale_range=self.poisson_scale_range,
                gray_noise_prob=self.gray_noise_prob,
            )
            jpeg_p1 = torch.empty((b_sub,), device=device, dtype=torch.float32).uniform_(
                self.jpeg_range[0], self.jpeg_range[1]
            )
            out = self.jpeger(out.clamp(0.0, 1.0), quality=jpeg_p1)

            # -------- 2nd-order degradation --------
            if random.random() < self.second_blur_prob:
                out = filter2d(out, k2)

            out = self._random_resize(
                out,
                ref_h=target_h,
                ref_w=target_w,
                resize_prob=self.resize_prob2,
                resize_range=self.resize_range2,
            )
            out = self._add_random_noise(
                out,
                gaussian_noise_prob=self.gaussian_noise_prob2,
                noise_range=self.noise_range2,
                poisson_scale_range=self.poisson_scale_range2,
                gray_noise_prob=self.gray_noise_prob2,
            )

            final_mode = random.choice(_INTERP_MODES)
            jpeg_p2 = torch.empty((b_sub,), device=device, dtype=torch.float32).uniform_(
                self.jpeg_range2[0], self.jpeg_range2[1]
            )
            if random.random() < 0.5:
                out = F.interpolate(out, size=(target_h, target_w), mode=final_mode)
                out = filter2d(out, sk)
                out = self.jpeger(out.clamp(0.0, 1.0), quality=jpeg_p2)
            else:
                out = self.jpeger(out.clamp(0.0, 1.0), quality=jpeg_p2)
                out = F.interpolate(out, size=(target_h, target_w), mode=final_mode)
                out = filter2d(out, sk)

            # 8-bit quantization rounding
            out = torch.clamp((out * 255.0).round(), 0.0, 255.0) / 255.0
            lq_out[mask] = out

            if self.pool is not None:
                lq_out, gt_out = self.pool.dequeue_and_enqueue(lq_out, gt_out)

            return {**batch, "lq": lq_out, "gt": gt_out}
