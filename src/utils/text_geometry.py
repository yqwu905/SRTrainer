"""Differentiable text-ROI geometry helpers shared across dataset manifests and losses."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "canonical_quad",
    "rectify_text_roi",
]


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    return (
        abs(
            float(
                np.sum(
                    pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)
                )
            )
        )
        * 0.5
    )


def canonical_quad(value: Any, name: str = "glyph") -> np.ndarray:
    """Validate and canonicalize an xyxy AABB, 8-float quad, or 4x2 cyclic TL/TR/BR/BL quad."""
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: expected finite xyxy AABB or four cyclic (x, y) corners."
        ) from exc

    if points.shape == (4,):
        left, top, right, bottom = points
        if not np.isfinite(points).all() or right <= left or bottom <= top:
            raise ValueError(f"{name}: AABB must be finite with x1 > x0 and y1 > y0.")
        points = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float64,
        )
    elif points.shape == (8,):
        points = points.reshape(4, 2)

    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{name}: expected finite xyxy AABB or four cyclic (x, y) corners.")

    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 1e-8) or np.all(cross < -1e-8)) or _polygon_area(points) < 1e-6:
        raise ValueError(f"{name}: quad must be convex, non-degenerate, and cyclic (not crossed).")
    if np.all(cross < 0):
        points = points[[0, 3, 2, 1]]
    return points


def rectify_text_roi(
    images: torch.Tensor,
    quad: Any,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Differentiable perspective crop from cyclic 4-corner geometry.

    Solves the constant 8x8 homography on CPU in float64 (device-agnostic), then samples via
    ``F.grid_sample`` so gradients propagate cleanly back to ``images``.
    """
    if images.ndim != 4:
        raise ValueError(f"rectify_text_roi expects 4D NCHW images, got shape {tuple(images.shape)}.")
    pts = canonical_quad(quad, "glyph ROI")
    out_h, out_w = output_size
    if not _is_positive_int(out_h) or not _is_positive_int(out_w):
        raise ValueError(
            f"output_size must contain positive integers (H, W), got {output_size!r}."
        )

    matrix: list[list[float]] = []
    rhs: list[float] = []
    for (u, v), (x, y) in zip(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), pts, strict=True):
        matrix.extend(
            (
                [u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x],
                [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y],
            )
        )
        rhs.extend((float(x), float(y)))

    try:
        h = torch.linalg.solve(
            torch.tensor(matrix, dtype=torch.float64),
            torch.tensor(rhs, dtype=torch.float64),
        )
    except RuntimeError as exc:
        raise ValueError("Glyph quad does not define a stable perspective transform.") from exc

    transform = torch.cat((h, torch.ones(1, dtype=torch.float64))).reshape(3, 3)
    if not torch.isfinite(transform).all():
        raise ValueError("Glyph quad produced a non-finite perspective transform.")

    transform = transform.to(device=images.device, dtype=torch.float32)
    ys = (torch.arange(out_h, device=images.device, dtype=torch.float32) + 0.5) / out_h
    xs = (torch.arange(out_w, device=images.device, dtype=torch.float32) + 0.5) / out_w
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    source = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1) @ transform.T
    denom = source[..., 2:]
    if (denom.abs() < 1e-7).any():
        raise ValueError("Glyph quad crosses the perspective horizon.")
    source_xy = source[..., :2] / denom
    grid = source_xy / source_xy.new_tensor([images.shape[-1], images.shape[-2]]) * 2.0 - 1.0
    return F.grid_sample(
        images.float(),
        grid[None].expand(images.shape[0], -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
