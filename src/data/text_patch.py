"""Place paired text patches on GT and LQ crops using a shared GT-coordinate quad."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.schema import TextPatchConfig
from data.manifest import _load_json_list, is_valid_rel_path
from utils.text_geometry import canonical_quad


@dataclass(frozen=True)
class TextPatchSample:
    lq_path: Path
    gt_path: Path
    label: str


def validate_text_patch_paths(
    json_path: str | Path, root_dir: str | Path, *, field_prefix: str = "text_patch"
) -> None:
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"{field_prefix}.json_path must be an existing file: {path}")
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"{field_prefix}.root_dir does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"{field_prefix}.root_dir must be a directory: {root}")


def load_text_patch_manifest(
    json_path: str | Path,
    root_dir: str | Path,
    *,
    field_prefix: str = "text_patch",
) -> list[TextPatchSample]:
    validate_text_patch_paths(json_path, root_dir, field_prefix=field_prefix)
    try:
        path, entries = _load_json_list(json_path)
    except ValueError as exc:
        raise ValueError(f"{field_prefix}.json_path: {exc}") from exc
    root = Path(root_dir)
    samples = []
    for idx, entry in enumerate(entries):
        for key in ("lq", "gt"):
            value = entry.get(key)
            if not is_valid_rel_path(value) or Path(value).is_absolute():
                raise ValueError(
                    f"{field_prefix}.json_path ({path}[{idx}].{key}) must be a non-empty relative image path."
                )
        aliases = ("label", "text", "transcript", "transcription")
        label = next((entry[k] for k in aliases if k in entry), None)
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{field_prefix}.json_path ({path}[{idx}]) needs a non-empty text label.")
        samples.append(TextPatchSample(root / entry["lq"], root / entry["gt"], label))
    return samples


def polygons_overlap(quad_a: np.ndarray, quad_b: np.ndarray) -> bool:
    """Test positive-area intersection of convex cyclic quads (edge contact is allowed)."""
    a, b = np.asarray(quad_a), np.asarray(quad_b)
    edges = np.concatenate((np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b))
    axes = np.stack((-edges[:, 1], edges[:, 0]))
    pa, pb = a @ axes, b @ axes
    return bool(np.all((pa.min(0) < pb.max(0)) & (pb.min(0) < pa.max(0))))


def _read_patch(path: Path) -> torch.Tensor:
    # Premultiply before any interpolation so RGB hidden behind transparency cannot bleed.
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:]
    return torch.from_numpy(rgba).permute(2, 0, 1).contiguous()


def _place_quad(
    patch_hw: tuple[int, int],
    canvas_hw: tuple[int, int],
    scale: int,
    zoom: float,
    angle: float,
) -> np.ndarray | None:
    h, w = patch_hw
    canvas_h, canvas_w = canvas_hw
    width, height = w * zoom, h * zoom
    if min(width, height) < max(2, scale) or max(width, height) > math.hypot(canvas_h, canvas_w):
        return None
    if zoom == 1 and angle == 0:
        if w > canvas_w or h > canvas_h:
            return None
        x = scale * random.randint(0, (canvas_w - w) // scale)
        y = scale * random.randint(0, (canvas_h - h) // scale)
        return canonical_quad([x, y, x + w, y + h])

    # Positive angles rotate counterclockwise in the image (y increases downwards).
    if angle % 90 == 0:
        cos, sin = ((1, 0), (0, 1), (-1, 0), (0, -1))[int(angle % 360 // 90)]
    else:
        theta = math.radians(angle % 360)
        cos, sin = math.cos(theta), math.sin(theta)
    offsets = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [width / 2, height / 2]
    offsets = offsets @ np.array([[cos, -sin], [sin, cos]])
    rx, ry = offsets.max(axis=0)
    if 2 * rx > canvas_w or 2 * ry > canvas_h:
        return None
    center = [random.uniform(rx, canvas_w - rx), random.uniform(ry, canvas_h - ry)]
    return canonical_quad(np.clip(offsets + center, 0.0, [float(canvas_w), float(canvas_h)]))


def _quad_roi(quad: np.ndarray, canvas_hw: tuple[int, int]) -> tuple[slice, slice]:
    h, w = canvas_hw
    x0, y0 = np.clip(np.floor(quad.min(axis=0)), 0, [w, h]).astype(int)
    x1, y1 = np.clip(np.ceil(quad.max(axis=0)), 0, [w, h]).astype(int)
    return slice(y0, y1), slice(x0, x1)


@dataclass
class _RenderedPatch:
    roi: tuple[slice, slice]
    rgba: torch.Tensor
    covered: torch.Tensor

    def fits(self, occupied: torch.Tensor) -> bool:
        return bool(self.covered.any()) and not bool((self.covered & occupied[self.roi]).any())

    def commit(
        self, canvas: torch.Tensor, occupied: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        roi = (slice(None), *self.roi)
        alpha = self.rgba[3:]
        canvas[roi].mul_(1 - alpha).add_(self.rgba[:3])
        occupied[self.roi] |= self.covered
        if mask is not None:
            coverage = mask[roi]
            coverage.add_(alpha * (1 - coverage))


def _render_patch(
    canvas_hw: tuple[int, int],
    patch: torch.Tensor,
    quad: np.ndarray,
    zoom: float,
    aligned: bool,
) -> _RenderedPatch | None:
    """Render premultiplied RGBA at pixel centers within the clipped bounding box."""
    rows, cols = _quad_roi(quad, canvas_hw)
    x0, x1, y0, y1 = cols.start, cols.stop, rows.start, rows.stop
    if x1 <= x0 or y1 <= y0:
        return None
    if not aligned:
        if zoom < 1:
            size = tuple(max(1, round(n * zoom)) for n in patch.shape[-2:])
            patch = F.interpolate(
                patch[None], size=size, mode="bilinear", align_corners=False, antialias=True
            )[0]
        # Project each pixel center onto the patch's two orthogonal edges.
        ys = torch.arange(y0, y1, dtype=torch.float64) + 0.5 - quad[0, 1]
        xs = torch.arange(x0, x1, dtype=torch.float64) + 0.5 - quad[0, 0]
        dy, dx = torch.meshgrid(ys, xs, indexing="ij")
        axes = quad[[1, 3]] - quad[0]
        axes = axes / (axes * axes).sum(axis=1, keepdims=True)
        uv = torch.stack([dx * ax + dy * ay for ax, ay in axes], dim=-1)
        inside = ((uv >= 0) & (uv < 1)).all(dim=-1)
        patch = (
            F.grid_sample(
                patch[None],
                (uv * 2 - 1).float()[None],
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )[0].clamp_(0, 1)
            * inside
        )
    covered = patch[3] > 1e-6
    if not aligned:
        patch = patch.masked_fill(~covered, 0)
    return _RenderedPatch((rows, cols), patch, covered)


def _visible_source_lines(lines: list[dict], gt_occupied: torch.Tensor) -> list[dict]:
    def visible(quad: Any) -> bool:
        points = np.asarray(quad)
        rows, cols = _quad_roi(points, gt_occupied.shape)
        ys, xs = np.nonzero(gt_occupied[rows, cols].numpy())
        if not xs.size:
            return True
        # A pixel center inside a convex cyclic quad lies on the same side of every edge.
        x = (xs + cols.start + 0.5)[:, None] - points[:, 0]
        y = (ys + rows.start + 0.5)[:, None] - points[:, 1]
        edges = np.roll(points, -1, axis=0) - points
        cross = edges[:, 0] * y - edges[:, 1] * x
        return not np.any((cross >= 0).all(axis=1) | (cross <= 0).all(axis=1))

    result = []
    for line in lines:
        kept = dict(line)
        if "quad" in kept and not visible(kept["quad"]):
            del kept["quad"]
            kept.pop("transcript", None)
        if "glyphs" in kept:
            kept["glyphs"] = [g for g in kept["glyphs"] if visible(g["quad"])]
        if "quad" in kept:
            result.append(kept)
        elif glyphs := kept.get("glyphs"):
            # OCR falls back to the bounding box of all remaining glyphs. If that box
            # crosses a patch, supervise each visible glyph at its own exact quad.
            points = np.concatenate([g["quad"] for g in glyphs])
            hull = canonical_quad([*points.min(axis=0), *points.max(axis=0)])
            if visible(hull):
                result.append(kept)
            else:
                result.extend({**kept, "quad": g["quad"], "glyphs": [g]} for g in glyphs)
    return result


class TextPatchOverlay:
    def __init__(
        self,
        cfg: TextPatchConfig,
        scale: int,
        *,
        samples: list[TextPatchSample] | None = None,
    ) -> None:
        assert cfg.json_path is not None and cfg.root_dir is not None
        self.samples = (
            list(samples)
            if samples is not None
            else load_text_patch_manifest(cfg.json_path, cfg.root_dir)
        )
        self.cfg = cfg
        self.scale = scale
        self._gt_sizes: dict[TextPatchSample, tuple[int, int]] = {}

    def _gt_size(self, sample: TextPatchSample) -> tuple[int, int]:
        if sample not in self._gt_sizes:
            with Image.open(sample.gt_path) as gt, Image.open(sample.lq_path) as lq:
                if gt.size != (lq.width * self.scale, lq.height * self.scale):
                    raise ValueError(
                        f"Patch {sample.gt_path} and {sample.lq_path} must match at SR scale "
                        f"{self.scale}; got GT {gt.size} and LQ {lq.size} (width, height)."
                    )
                self._gt_sizes[sample] = (gt.height, gt.width)
        return self._gt_sizes[sample]

    def apply(
        self,
        gt: torch.Tensor,
        lq: torch.Tensor,
        lines: list[dict],
        mask: torch.Tensor | None = None,
    ) -> list[dict]:
        """Update the canvases (and optional LQ coverage) and return their visible annotations."""
        if random.random() >= self.cfg.prob:
            return lines
        quads: list[np.ndarray] = []
        added = []
        gt_occupied = torch.zeros(gt.shape[-2:], dtype=torch.bool)
        lq_occupied = torch.zeros(lq.shape[-2:], dtype=torch.bool)

        @lru_cache(maxsize=16)
        def read_pair(sample: TextPatchSample) -> tuple[torch.Tensor, torch.Tensor]:
            return _read_patch(sample.gt_path), _read_patch(sample.lq_path)

        min_angle, max_angle = self.cfg.rotate_range
        for _ in range(random.randint(*self.cfg.num_patches)):
            for _attempt in range(self.cfg.max_attempts):
                sample = random.choice(self.samples)
                zoom = random.uniform(*self.cfg.scale_range)
                # Halving the bounds keeps their difference finite even for opposite extremes.
                angle = 2 * random.uniform(min_angle / 2, max_angle / 2)
                quad = _place_quad(self._gt_size(sample), gt.shape[-2:], self.scale, zoom, angle)
                if quad is None or any(polygons_overlap(quad, p) for p in quads):
                    continue
                aligned = zoom == 1 and angle == 0
                gt_patch, lq_patch = read_pair(sample)
                gt_render = _render_patch(gt.shape[-2:], gt_patch, quad, zoom, aligned)
                lq_render = _render_patch(lq.shape[-2:], lq_patch, quad / self.scale, zoom, aligned)
                if gt_render is None or not gt_render.fits(gt_occupied):
                    continue
                if lq_render is None or not lq_render.fits(lq_occupied):
                    continue
                gt_render.commit(gt, gt_occupied)
                lq_render.commit(lq, lq_occupied, mask)
                break
            else:
                break
            quads.append(quad)
            added.append(
                {
                    "quad": quad.tolist(),
                    "transcript": sample.label,
                    "valid": sample.label.strip() not in ("###", "*"),
                    "alignment_valid": True,
                }
            )
        return _visible_source_lines(lines, gt_occupied) + added if added else lines
