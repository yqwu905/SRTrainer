"""Image I/O utilities converting between disk images and RGB float32 tensors in ``[0, 1]``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp"}

_PIL_FORMAT_MAP: dict[str, str] = {
    "png": "PNG",
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "webp": "WEBP",
}


def _to_3d_rgb(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"Expected {name} of shape [3, H, W] or [1, 3, H, W], got {tuple(tensor.shape)}."
        )
    return tensor


def load_image(path: str | Path) -> torch.Tensor:
    """Load an image file into a contiguous RGB float32 tensor of shape ``[3, H, W]`` in ``[0, 1]``."""
    img_path = Path(path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_image(
    tensor: torch.Tensor,
    path: str | Path,
    save_format: str | None = None,
    jpg_quality: int = 95,
) -> Path:
    """Save a ``[3, H, W]`` or ``[1, 3, H, W]`` RGB tensor in ``[0, 1]`` to disk."""
    img_3d = _to_3d_rgb(tensor, "tensor")
    if (
        isinstance(jpg_quality, bool)
        or not isinstance(jpg_quality, int)
        or not (1 <= jpg_quality <= 100)
    ):
        raise ValueError(f"jpg_quality must be an integer in [1, 100], got {jpg_quality!r}.")

    out_path = Path(path)
    if save_format is None:
        fmt = out_path.suffix.lstrip(".").lower()
        if fmt not in _PIL_FORMAT_MAP:
            raise ValueError(
                f"Cannot infer supported image format from path {out_path}; "
                f"expected extension in {sorted(IMAGE_EXTENSIONS)} or explicit save_format."
            )
    else:
        fmt = save_format.strip().lstrip(".").lower()
        if fmt not in _PIL_FORMAT_MAP:
            raise ValueError(
                f"Unsupported save_format={save_format!r}; "
                f"expected one of {sorted(_PIL_FORMAT_MAP)}."
            )
        existing_fmt = out_path.suffix.lstrip(".").lower()
        if (
            existing_fmt not in _PIL_FORMAT_MAP
            or _PIL_FORMAT_MAP[existing_fmt] != _PIL_FORMAT_MAP[fmt]
        ):
            out_path = out_path.with_suffix(f".{fmt}")

    arr_u8 = (
        img_3d.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(arr_u8)
    pil_fmt = _PIL_FORMAT_MAP[fmt]
    if pil_fmt in {"JPEG", "WEBP"}:
        pil_img.save(out_path, format=pil_fmt, quality=jpg_quality)
    else:
        pil_img.save(out_path, format=pil_fmt)
    return out_path


def make_comparison_image(
    lq: torch.Tensor,
    sr: torch.Tensor,
    gt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a horizontal comparison strip ``[LQ_bicubic | SR | GT]`` (or ``[LQ_bicubic | SR]``).

    Returns a contiguous float32 tensor of shape ``[3, H_sr, W_total]`` clamped to ``[0, 1]``.
    """
    sr_3d = _to_3d_rgb(sr, "sr").detach().to(dtype=torch.float32).clamp(0.0, 1.0)
    target_device = sr_3d.device
    h_sr, w_sr = sr_3d.shape[1], sr_3d.shape[2]

    lq_3d = _to_3d_rgb(lq, "lq").detach().to(device=target_device, dtype=torch.float32)
    if (lq_3d.shape[1], lq_3d.shape[2]) != (h_sr, w_sr):
        lq_up = (
            F.interpolate(
                lq_3d.unsqueeze(0),
                size=(h_sr, w_sr),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze(0)
            .clamp(0.0, 1.0)
        )
    else:
        lq_up = lq_3d.clamp(0.0, 1.0)

    panels = [lq_up, sr_3d]
    if gt is not None:
        gt_3d = _to_3d_rgb(gt, "gt").detach().to(device=target_device, dtype=torch.float32)
        if (gt_3d.shape[1], gt_3d.shape[2]) != (h_sr, w_sr):
            raise ValueError(
                f"Spatial shape mismatch between sr {(h_sr, w_sr)} and gt "
                f"{(gt_3d.shape[1], gt_3d.shape[2])}."
            )
        panels.append(gt_3d.clamp(0.0, 1.0))

    return torch.cat(panels, dim=2).contiguous()
