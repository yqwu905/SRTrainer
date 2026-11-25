"""Spatial cropping and synchronized dihedral augmentation transforms for SR tensors."""

from __future__ import annotations

import numbers
import random
from collections.abc import Sequence
from typing import Any

import torch

from utils.misc import normalize_gt_size, pad_to_min_size

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


def _focused_start(center: float, extent: int) -> float:
    jitter = extent * 0.25
    return center - extent * 0.5 + random.uniform(-jitter, jitter)


def paired_random_crop_with_coords(
    gt: torch.Tensor,
    lq: torch.Tensor,
    gt_size: int | Sequence[int],
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
    crop_h, crop_w = normalize_gt_size(gt_size, scale_i)

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

    patch_h, patch_w = crop_h // scale_i, crop_w // scale_i
    if lq_h < patch_h or lq_w < patch_w:
        raise ValueError(
            f"Paired image smaller than crop size: lq=({lq_h}, {lq_w}) < lq_size={(patch_h, patch_w)} "
            f"(gt=({gt_h}, {gt_w}) < gt_size={(crop_h, crop_w)}, scale={scale_i})."
        )

    if focus_center is not None:
        cx, cy = focus_center
        gt_left_f = _focused_start(cx, crop_w)
        gt_top_f = _focused_start(cy, crop_h)
        lq_left = max(0, min(lq_w - patch_w, round(gt_left_f / scale_i)))
        lq_top = max(0, min(lq_h - patch_h, round(gt_top_f / scale_i)))
    else:
        lq_top = random.randint(0, lq_h - patch_h)
        lq_left = random.randint(0, lq_w - patch_w)

    gt_top = lq_top * scale_i
    gt_left = lq_left * scale_i

    gt_crop = gt[:, gt_top : gt_top + crop_h, gt_left : gt_left + crop_w].contiguous()
    lq_crop = lq[:, lq_top : lq_top + patch_h, lq_left : lq_left + patch_w].contiguous()
    return gt_crop, lq_crop, gt_top, gt_left, gt_h, gt_w


def paired_random_crop(
    gt: torch.Tensor,
    lq: torch.Tensor,
    gt_size: int | Sequence[int],
    scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly crop aligned ``(gt, lq)`` patches with strict ``gt_size == lq_size * scale`` check."""
    gt_crop, lq_crop, _, _, _, _ = paired_random_crop_with_coords(
        gt, lq, gt_size=gt_size, scale=scale
    )
    return gt_crop, lq_crop


def random_crop_with_coords(
    gt: torch.Tensor,
    gt_size: int | Sequence[int],
    pad_if_smaller: bool = True,
    *,
    focus_center: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Crop ``[C, crop_h, crop_w]`` from ``gt`` and return ``(crop, top, left, orig_h, orig_w)``."""
    if not isinstance(gt, torch.Tensor):
        raise TypeError(f"Expected gt to be a torch.Tensor, got {type(gt).__name__}.")
    if gt.ndim != 3:
        raise ValueError(f"Expected 3D [C, H, W] gt tensor, got shape {tuple(gt.shape)}.")

    crop_h, crop_w = normalize_gt_size(gt_size)
    orig_h, orig_w = int(gt.shape[1]), int(gt.shape[2])
    h, w = orig_h, orig_w
    if h <= 0 or w <= 0:
        raise ValueError(f"Image spatial dimensions must be positive, got {(h, w)}.")

    if h < crop_h or w < crop_w:
        if pad_if_smaller:
            gt, _ = pad_to_min_size(gt, crop_h, crop_w, mode="reflect")
            h, w = int(gt.shape[1]), int(gt.shape[2])
        else:
            raise ValueError(
                f"GT image size {(h, w)} is smaller than gt_size={(crop_h, crop_w)} "
                "with pad_if_smaller=False."
            )

    if focus_center is not None:
        cx, cy = focus_center
        left = max(0, min(w - crop_w, round(_focused_start(cx, crop_w))))
        top = max(0, min(h - crop_h, round(_focused_start(cy, crop_h))))
    else:
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
    crop = gt[:, top : top + crop_h, left : left + crop_w].contiguous()
    return crop, top, left, orig_h, orig_w


def random_crop(
    gt: torch.Tensor,
    gt_size: int | Sequence[int],
    pad_if_smaller: bool = True,
) -> torch.Tensor:
    """Randomly crop a ``[C, crop_h, crop_w]`` patch from ``gt``, optionally padding small images."""
    crop, _, _, _, _ = random_crop_with_coords(gt, gt_size=gt_size, pad_if_smaller=pad_if_smaller)
    return crop


def augment_tensors_with_flags(
    tensors: Sequence[torch.Tensor],
    hflip: bool = True,
    vflip: bool = True,
    rot90: bool = True,
    *,
    preserve_shape: bool = False,
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
    # Odd quarter turns swap the height and width of a rectangular training patch.
    if preserve_shape and tensors[0].shape[-2] != tensors[0].shape[-1]:
        rot_k = (rot_k // 2) * 2
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
