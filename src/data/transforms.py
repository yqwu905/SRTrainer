"""Spatial cropping and synchronized dihedral augmentation transforms for SR tensors."""

from __future__ import annotations

import numbers
import random
from collections.abc import Sequence
from typing import Any

import torch

from utils.misc import pad_to_min_size

__all__ = [
    "paired_random_crop",
    "paired_random_crop_with_coords",
    "random_crop",
    "random_crop_with_coords",
    "augment_tensors",
    "augment_tensors_with_flags",
]


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def paired_random_crop_with_coords(
    gt: torch.Tensor,
    lq: torch.Tensor,
    gt_size: int,
    scale: int,
    *,
    focus_center: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    """Crop aligned ``(gt, lq)`` patches and return ``(gt_crop, lq_crop, gt_top, gt_left, orig_h, orig_w)``."""
    if not isinstance(gt, torch.Tensor):
        raise TypeError(f"Expected gt to be a torch.Tensor, got {type(gt).__name__}.")
    if not isinstance(lq, torch.Tensor):
        raise TypeError(f"Expected lq to be a torch.Tensor, got {type(lq).__name__}.")
    if gt.ndim != 3:
        raise ValueError(f"Expected 3D [C, H, W] gt tensor, got shape {tuple(gt.shape)}.")
    if lq.ndim != 3:
        raise ValueError(f"Expected 3D [C, H, W] lq tensor, got shape {tuple(lq.shape)}.")

    scale_i = _validate_positive_int(scale, "scale")
    gt_size_i = _validate_positive_int(gt_size, "gt_size")
    if gt_size_i % scale_i != 0:
        raise ValueError(f"gt_size ({gt_size_i}) must be divisible by scale ({scale_i}).")

    gt_h, gt_w = int(gt.shape[1]), int(gt.shape[2])
    lq_h, lq_w = int(lq.shape[1]), int(lq.shape[2])
    if gt_h <= 0 or gt_w <= 0 or lq_h <= 0 or lq_w <= 0:
        raise ValueError(
            f"Image spatial dimensions must be positive, got gt={(gt_h, gt_w)}, lq={(lq_h, lq_w)}."
        )

    if gt_h != lq_h * scale_i or gt_w != lq_w * scale_i:
        raise ValueError(
            f"Paired GT/LQ dimension mismatch for scale={scale_i}: "
            f"gt=({gt_h}, {gt_w}), lq=({lq_h}, {lq_w}), "
            f"expected gt=({lq_h * scale_i}, {lq_w * scale_i})."
        )

    lq_size = gt_size_i // scale_i
    if lq_h < lq_size or lq_w < lq_size:
        raise ValueError(
            f"Paired image smaller than crop size: lq=({lq_h}, {lq_w}) < lq_size={lq_size} "
            f"(gt=({gt_h}, {gt_w}) < gt_size={gt_size_i}, scale={scale_i})."
        )

    if focus_center is not None:
        cx, cy = focus_center
        jitter = gt_size_i * 0.25
        gt_left_f = cx - gt_size_i * 0.5 + random.uniform(-jitter, jitter)
        gt_top_f = cy - gt_size_i * 0.5 + random.uniform(-jitter, jitter)
        lq_left = max(0, min(lq_w - lq_size, round(gt_left_f / scale_i)))
        lq_top = max(0, min(lq_h - lq_size, round(gt_top_f / scale_i)))
    else:
        lq_top = random.randint(0, lq_h - lq_size)
        lq_left = random.randint(0, lq_w - lq_size)

    gt_top = lq_top * scale_i
    gt_left = lq_left * scale_i

    gt_crop = gt[:, gt_top : gt_top + gt_size_i, gt_left : gt_left + gt_size_i].contiguous()
    lq_crop = lq[:, lq_top : lq_top + lq_size, lq_left : lq_left + lq_size].contiguous()
    return gt_crop, lq_crop, gt_top, gt_left, gt_h, gt_w


def paired_random_crop(
    gt: torch.Tensor,
    lq: torch.Tensor,
    gt_size: int,
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly crop aligned ``(gt, lq)`` patches with strict ``gt_size == lq_size * scale`` check."""
    gt_crop, lq_crop, _, _, _, _ = paired_random_crop_with_coords(
        gt, lq, gt_size=gt_size, scale=scale
    )
    return gt_crop, lq_crop


def random_crop_with_coords(
    gt: torch.Tensor,
    gt_size: int,
    pad_if_smaller: bool = True,
    *,
    focus_center: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Crop ``[C, gt_size, gt_size]`` from ``gt`` and return ``(crop, top, left, orig_h, orig_w)``."""
    if not isinstance(gt, torch.Tensor):
        raise TypeError(f"Expected gt to be a torch.Tensor, got {type(gt).__name__}.")
    if gt.ndim != 3:
        raise ValueError(f"Expected 3D [C, H, W] gt tensor, got shape {tuple(gt.shape)}.")

    gt_size_i = _validate_positive_int(gt_size, "gt_size")
    orig_h, orig_w = int(gt.shape[1]), int(gt.shape[2])
    h, w = orig_h, orig_w
    if h <= 0 or w <= 0:
        raise ValueError(f"Image spatial dimensions must be positive, got {(h, w)}.")

    if h < gt_size_i or w < gt_size_i:
        if pad_if_smaller:
            gt, _ = pad_to_min_size(gt, gt_size_i, gt_size_i, mode="reflect")
            h, w = int(gt.shape[1]), int(gt.shape[2])
        else:
            raise ValueError(
                f"GT image size {(h, w)} is smaller than gt_size={gt_size_i} "
                "with pad_if_smaller=False."
            )

    if focus_center is not None:
        cx, cy = focus_center
        jitter = gt_size_i * 0.25
        left = max(0, min(w - gt_size_i, int(round(cx - gt_size_i * 0.5 + random.uniform(-jitter, jitter)))))
        top = max(0, min(h - gt_size_i, int(round(cy - gt_size_i * 0.5 + random.uniform(-jitter, jitter)))))
    else:
        top = random.randint(0, h - gt_size_i)
        left = random.randint(0, w - gt_size_i)
    crop = gt[:, top : top + gt_size_i, left : left + gt_size_i].contiguous()
    return crop, top, left, orig_h, orig_w


def random_crop(
    gt: torch.Tensor,
    gt_size: int,
    pad_if_smaller: bool = True,
) -> torch.Tensor:
    """Randomly crop a ``[C, gt_size, gt_size]`` patch from ``gt``, optionally padding small images."""
    crop, _, _, _, _ = random_crop_with_coords(gt, gt_size=gt_size, pad_if_smaller=pad_if_smaller)
    return crop


def augment_tensors_with_flags(
    tensors: Sequence[torch.Tensor],
    hflip: bool = True,
    vflip: bool = True,
    rot90: bool = True,
) -> tuple[list[torch.Tensor], bool]:
    """Apply identical random hflip/vflip/rot90 transforms and report whether geometry changed."""
    if isinstance(tensors, (str, bytes)) or not isinstance(tensors, Sequence):
        raise TypeError(
            f"Expected tensors to be a sequence of torch.Tensor, got {type(tensors).__name__}."
        )

    for idx, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(
                f"Expected tensors[{idx}] to be a torch.Tensor, got {type(t).__name__}."
            )
        if t.ndim < 2:
            raise ValueError(
                f"Expected tensors[{idx}] to have at least 2 spatial dims, got shape {tuple(t.shape)}."
            )

    if len(tensors) == 0:
        return [], False

    do_hflip = bool(hflip) and (random.random() < 0.5)
    do_vflip = bool(vflip) and (random.random() < 0.5)
    rot_k = random.randint(0, 3) if bool(rot90) else 0
    changed = do_hflip or do_vflip or (rot_k != 0)

    out: list[torch.Tensor] = []
    for t in tensors:
        x = t
        if do_hflip:
            x = torch.flip(x, dims=(-1,))
        if do_vflip:
            x = torch.flip(x, dims=(-2,))
        if rot_k:
            x = torch.rot90(x, k=rot_k, dims=(-2, -1))
        out.append(x.contiguous())
    return out, changed


def augment_tensors(
    tensors: Sequence[torch.Tensor],
    hflip: bool = True,
    vflip: bool = True,
    rot90: bool = True,
) -> list[torch.Tensor]:
    """Apply identical random hflip/vflip/rot90 transforms to a sequence of image tensors."""
    out, _ = augment_tensors_with_flags(tensors, hflip=hflip, vflip=vflip, rot90=rot90)
    return out
