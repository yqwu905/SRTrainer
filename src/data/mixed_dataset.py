"""Sample-level mixed training dataset supporting auto-detected gt_only and paired JSON sources."""

from __future__ import annotations

import bisect
import itertools
import logging
import math
import numbers
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from torch.utils.data import Dataset, default_collate

from config.schema import AugmentConfig, DataSourceConfig, KernelConfig
from config.validation import load_augment_config, load_kernel_config
from data.degradation.kernels import KernelGenerator
from data.manifest import SourceStats, TrainSample, validate_sources_config
from data.transforms import (
    augment_tensors_with_flags,
    paired_random_crop_with_coords,
    random_crop_with_coords,
    resolve_gt_size,
)
from utils.image import load_image
from utils.text_geometry import canonical_quad

__all__ = ["MixedSRDataset", "train_collate_fn"]


def _normalize_omegaconf(obj: Any, context: str) -> Any:
    if OmegaConf.is_config(obj):
        try:
            return OmegaConf.to_container(obj, resolve=True, throw_on_missing=True)
        except OmegaConfBaseException as exc:
            raise ValueError(f"Invalid OmegaConf value for {context}: {exc}") from exc
    return obj


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _validate_prob(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real in [0.0, 1.0], got {value!r}.")
    val = float(value)
    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}.")
    return val


def _pick_text_focus_center(
    text_lines: tuple[dict[str, Any], ...] | None,
) -> tuple[float, float] | None:
    if not text_lines:
        return None
    centers: list[tuple[float, float]] = []
    for line in text_lines:
        if not line.get("valid", True) or not line.get("alignment_valid", True):
            continue
        if "quad" in line:
            pts = np.asarray(line["quad"], dtype=np.float64)
            centers.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
        else:
            for glyph in line.get("glyphs", ()):
                pts = np.asarray(glyph["quad"], dtype=np.float64)
                centers.append((float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    if not centers:
        return None
    return random.choice(centers)


def _polygon_area_2d(pts: np.ndarray) -> float:
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


def _crop_line_quad(
    quad: Any,
    offset: np.ndarray,
    extent: np.ndarray,
    *,
    min_side: float = 4.0,
    min_axis_retention: float = 0.7,
    min_area_retention: float = 0.25,
    min_rectangularity: float = 0.85,
) -> list[list[float]] | None:
    """Translate a line quad into crop coordinates, clipping near-axis-aligned boundary boxes."""
    pts = np.asarray(quad, dtype=np.float64) - offset
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return None
    if (pts >= 0.0).all() and (pts <= extent).all():
        return pts.tolist()

    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    aabb_w = x1 - x0
    aabb_h = y1 - y0
    if aabb_w <= 0.0 or aabb_h <= 0.0:
        return None

    # Only clip boxes that are close to axis-aligned; rotated boxes clipped as AABBs pull in
    # excessive background corners, so keep rotated boxes only when fully inside the crop.
    poly_area = _polygon_area_2d(pts)
    aabb_area = aabb_w * aabb_h
    if poly_area / aabb_area < min_rectangularity:
        return None

    ix0 = max(0.0, x0)
    iy0 = max(0.0, y0)
    ix1 = min(float(extent[0]), x1)
    iy1 = min(float(extent[1]), y1)
    iw = ix1 - ix0
    ih = iy1 - iy0
    if iw < min_side or ih < min_side:
        return None

    # Keep horizontal lines truncated left/right (preserving line height) or vertical lines
    # truncated top/bottom (preserving line width), provided enough of the box remains visible.
    major_axis_ok = (ih / aabb_h >= min_axis_retention) if aabb_w >= aabb_h else (
        iw / aabb_w >= min_axis_retention
    )
    if not major_axis_ok or (iw * ih) / aabb_area < min_area_retention:
        return None

    clipped = canonical_quad([ix0, iy0, ix1, iy1], name="cropped line quad")
    return clipped.tolist()


def _crop_text_lines(
    text_lines: tuple[dict[str, Any], ...] | None,
    *,
    crop_left: int,
    crop_top: int,
    crop_h: int,
    crop_w: int,
    orig_h: int,
    orig_w: int,
) -> list[dict[str, Any]]:
    """Translate HR line quads and glyph quads to crop-relative coordinates."""
    if not text_lines:
        return []
    max_x = float(min(crop_w, orig_w - crop_left))
    max_y = float(min(crop_h, orig_h - crop_top))
    if max_x <= 0.0 or max_y <= 0.0:
        return []

    visible_lines: list[dict[str, Any]] = []
    offset = np.array([float(crop_left), float(crop_top)], dtype=np.float64)
    extent = np.array([max_x, max_y], dtype=np.float64)
    for line in text_lines:
        if not line.get("valid", True) or not line.get("alignment_valid", True):
            continue
        kept_line: dict[str, Any] = {}
        if "quad" in line and line["quad"] is not None:
            kept_quad = _crop_line_quad(line["quad"], offset, extent)
            if kept_quad is not None:
                kept_line["quad"] = kept_quad

        kept_glyphs: list[dict[str, Any]] = []
        for glyph in line.get("glyphs", ()):
            pts = np.asarray(glyph["quad"], dtype=np.float64) - offset
            if (pts >= 0.0).all() and (pts <= extent).all():
                kept_glyphs.append({"char": glyph["char"], "quad": pts.tolist()})
        if kept_glyphs:
            kept_line["glyphs"] = kept_glyphs

        if kept_line:
            visible_lines.append(kept_line)
    return visible_lines


def train_collate_fn(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate training batches while preserving variable-length ``text_lines`` per image."""
    if not items:
        raise ValueError("Cannot collate an empty training batch.")
    has_lines = any("text_lines" in item for item in items)
    if not has_lines:
        return default_collate(list(items))
    tensor_items = [
        {k: v for k, v in item.items() if k != "text_lines"} for item in items
    ]
    batch: dict[str, Any] = default_collate(tensor_items)
    batch["text_lines"] = [list(item.get("text_lines", ())) for item in items]
    return batch


class MixedSRDataset(Dataset[dict[str, Any]]):
    """Multi-source mixed super-resolution training dataset with per-sample auto-detection.

    ``gt_size`` accepts either a positive int (square crops) or a 2-element ``[gt_h, gt_w]``
    sequence (rectangular crops, ``gt_h != gt_w`` allowed); the paired LQ crop is derived per
    axis as ``(gt_h // scale, gt_w // scale)``. Rectangular crops require ``augment.rot90`` to
    be disabled, since 90-degree rotations would swap the patch dimensions.

    Supports both ``(source_id, sample_idx)`` tuple indexing (used by
    ``DistributedWeightedInfiniteSampler``) and flat integer indexing. Emits uniform tensor
    dictionaries with zero placeholders so ``gt_only`` and ``paired`` samples collate seamlessly
    into a single batch for GPU masked degradation. When any source carries glyph annotations
    (inline ``text_lines``/``glyphs`` or external ``glyphs_json``), emits crop-relative
    ``text_lines`` per sample for ``GlyphLoss``.
    """

    def __init__(
        self,
        sources: Sequence[DataSourceConfig | Mapping[str, Any]],
        gt_size: int | Sequence[int],
        scale: int = 1,
        augment: AugmentConfig | Mapping[str, Any] | None = None,
        has_degradation: bool = True,
        kernel_cfg: KernelConfig | Mapping[str, Any] | None = None,
        is_main_process: bool = True,
        logger: Any = None,
        text_crop_prob: float = 0.0,
    ) -> None:
        self.scale: int = _validate_positive_int(scale, "scale")
        self.gt_size: tuple[int, int] = resolve_gt_size(gt_size, self.scale)
        self.lq_size: tuple[int, int] = (
            self.gt_size[0] // self.scale,
            self.gt_size[1] // self.scale,
        )
        self.text_crop_prob: float = _validate_prob(text_crop_prob, "text_crop_prob")

        if not isinstance(has_degradation, bool):
            raise TypeError(
                f"has_degradation must be a bool, got {type(has_degradation).__name__}."
            )
        self.has_degradation: bool = has_degradation

        if augment is None:
            aug_cfg = AugmentConfig(hflip=False, vflip=False, rot90=False)
        elif isinstance(augment, AugmentConfig):
            aug_cfg = augment
        else:
            norm_aug = _normalize_omegaconf(augment, "augment")
            if not isinstance(norm_aug, Mapping):
                raise TypeError(
                    f"augment must be an AugmentConfig, mapping, or None, got {type(norm_aug).__name__}."
                )
            for key in ("hflip", "vflip", "rot90"):
                if key in norm_aug and not isinstance(norm_aug[key], bool):
                    raise TypeError(
                        f"augment[{key!r}] must be a bool, got {type(norm_aug[key]).__name__}."
                    )
            aug_cfg = load_augment_config(
                {
                    "hflip": norm_aug.get("hflip", False),
                    "vflip": norm_aug.get("vflip", False),
                    "rot90": norm_aug.get("rot90", False),
                }
            )
        self._hflip = aug_cfg.hflip
        self._vflip = aug_cfg.vflip
        self._rot90 = aug_cfg.rot90

        if self._rot90 and self.gt_size[0] != self.gt_size[1]:
            raise ValueError(
                f"augment.rot90=True is incompatible with rectangular gt_size={self.gt_size}: "
                "90-degree rotations would emit (gt_h, gt_w) and (gt_w, gt_h) crops in the same "
                "batch. Disable augment.rot90 (or use a square gt_size)."
            )

        # Intentionally do NOT store `logger` on `self` to keep dataset picklable for num_workers>0.
        samples_per_source, source_stats = validate_sources_config(
            sources=sources,
            has_degradation=has_degradation,
            is_main_process=is_main_process,
            logger=logger,
        )

        self._samples_per_source: list[list[TrainSample]] = samples_per_source
        self._source_stats: list[SourceStats] = source_stats
        self._source_sizes: list[int] = [len(s) for s in samples_per_source]
        self._source_ratios: list[float] = [st.ratio for st in source_stats]
        self._cum_sizes: list[int] = list(itertools.accumulate(self._source_sizes))
        self._total_len: int = self._cum_sizes[-1]
        self.has_text_annotations: bool = any(
            sample.text_lines is not None
            for src_samples in samples_per_source
            for sample in src_samples
        )

        if self.has_text_annotations and (self._hflip or self._vflip or self._rot90) and is_main_process:
            active_logger = logger if logger is not None else logging.getLogger("srtrainer")
            active_logger.warning(
                "[Data] Glyph annotations detected with geometric augmentations enabled "
                "(hflip=%s, vflip=%s, rot90=%s). Flipped/rotated crops will omit glyph "
                "supervision to avoid scoring mirrored text; disable augment for text training.",
                self._hflip,
                self._vflip,
                self._rot90,
            )

        if self.has_degradation:
            if kernel_cfg is not None and not isinstance(kernel_cfg, (KernelConfig, Mapping)):
                raise TypeError(
                    f"kernel_cfg must be a KernelConfig, mapping, or None, got {type(kernel_cfg).__name__}."
                )
            resolved_kcfg = (
                kernel_cfg
                if isinstance(kernel_cfg, KernelConfig)
                else (load_kernel_config(kernel_cfg) if kernel_cfg is not None else KernelConfig())
            )
            self.kernel_generator: KernelGenerator | None = KernelGenerator(resolved_kcfg)
            self.kernel_size: int | None = self.kernel_generator.kernel_size
        else:
            self.kernel_generator = None
            self.kernel_size = None

    @property
    def source_sizes(self) -> list[int]:
        """Return a defensive copy of per-source sample counts."""
        return list(self._source_sizes)

    @property
    def source_ratios(self) -> list[float]:
        """Return a defensive copy of per-source sampling ratios."""
        return list(self._source_ratios)

    @property
    def source_stats(self) -> list[SourceStats]:
        """Return a defensive copy of per-source manifest statistics."""
        return list(self._source_stats)

    def __len__(self) -> int:
        return self._total_len

    def _resolve_index(self, index: tuple[int, int] | int) -> tuple[int, int]:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError(
                    f"Expected (source_id, sample_idx) 2-tuple, got tuple of length {len(index)}."
                )
            raw_sid, raw_idx = index
            if (
                isinstance(raw_sid, bool)
                or isinstance(raw_idx, bool)
                or not isinstance(raw_sid, (int, np.integer))
                or not isinstance(raw_idx, (int, np.integer))
            ):
                raise TypeError(f"Tuple index entries must be integers, got {index!r}.")
            source_id = int(raw_sid)
            sample_idx = int(raw_idx)
            if source_id < 0 or source_id >= len(self._samples_per_source):
                raise IndexError(
                    f"source_id {source_id} out of range for {len(self._samples_per_source)} sources."
                )
            if sample_idx < 0 or sample_idx >= self._source_sizes[source_id]:
                raise IndexError(
                    f"sample_idx {sample_idx} out of range for source {source_id} "
                    f"(size={self._source_sizes[source_id]})."
                )
            return source_id, sample_idx

        if isinstance(index, (int, np.integer)) and not isinstance(index, bool):
            flat_idx = int(index)
            if flat_idx < 0:
                flat_idx += self._total_len
            if flat_idx < 0 or flat_idx >= self._total_len:
                raise IndexError(
                    f"Index {index} out of range for dataset of length {self._total_len}."
                )
            source_id = bisect.bisect_right(self._cum_sizes, flat_idx)
            prev_end = self._cum_sizes[source_id - 1] if source_id > 0 else 0
            sample_idx = flat_idx - prev_end
            return source_id, sample_idx

        raise TypeError(
            f"Index must be an int or a (source_id, sample_idx) tuple, got {type(index).__name__}."
        )

    def __getitem__(self, index: tuple[int, int] | int) -> dict[str, Any]:
        source_id, sample_idx = self._resolve_index(index)
        sample = self._samples_per_source[source_id][sample_idx]
        do_aug = self._hflip or self._vflip or self._rot90

        focus_center: tuple[float, float] | None = None
        if (
            self.has_text_annotations
            and self.text_crop_prob > 0.0
            and sample.text_lines
            and random.random() < self.text_crop_prob
        ):
            focus_center = _pick_text_focus_center(sample.text_lines)

        if not sample.need_degradation:
            assert sample.lq_path is not None
            gt = load_image(sample.gt_path)
            lq = load_image(sample.lq_path)
            gt, lq, gt_top, gt_left, orig_h, orig_w = paired_random_crop_with_coords(
                gt,
                lq,
                gt_size=self.gt_size,
                scale=self.scale,
                focus_center=focus_center,
            )
            geom_changed = False
            if do_aug:
                [gt, lq], geom_changed = augment_tensors_with_flags(
                    [gt, lq],
                    hflip=self._hflip,
                    vflip=self._vflip,
                    rot90=self._rot90,
                )

            item: dict[str, Any] = {
                "gt": gt.contiguous(),
                "lq": lq.contiguous(),
                "need_degradation": torch.tensor(False, dtype=torch.bool),
                "source_id": torch.tensor(source_id, dtype=torch.long),
            }
            if self.has_degradation:
                assert self.kernel_size is not None
                k = self.kernel_size
                item["kernel1"] = torch.zeros((k, k), dtype=torch.float32)
                item["kernel2"] = torch.zeros((k, k), dtype=torch.float32)
                item["sinc_kernel"] = torch.zeros((k, k), dtype=torch.float32)
            if self.has_text_annotations:
                item["text_lines"] = (
                    []
                    if geom_changed
                    else _crop_text_lines(
                        sample.text_lines,
                        crop_left=gt_left,
                        crop_top=gt_top,
                        crop_h=self.gt_size[0],
                        crop_w=self.gt_size[1],
                        orig_h=orig_h,
                        orig_w=orig_w,
                    )
                )
            return item

        assert self.kernel_generator is not None
        gt = load_image(sample.gt_path)
        gt, gt_top, gt_left, orig_h, orig_w = random_crop_with_coords(
            gt,
            gt_size=self.gt_size,
            pad_if_smaller=True,
            focus_center=focus_center,
        )
        geom_changed = False
        if do_aug:
            [gt], geom_changed = augment_tensors_with_flags(
                [gt],
                hflip=self._hflip,
                vflip=self._vflip,
                rot90=self._rot90,
            )

        lq = torch.zeros((3, self.lq_size[0], self.lq_size[1]), dtype=torch.float32)
        kernel1, kernel2, sinc_kernel = self.kernel_generator.sample()
        out_item: dict[str, Any] = {
            "gt": gt.contiguous(),
            "lq": lq,
            "need_degradation": torch.tensor(True, dtype=torch.bool),
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "kernel1": kernel1,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
        }
        if self.has_text_annotations:
            out_item["text_lines"] = (
                []
                if geom_changed
                else _crop_text_lines(
                    sample.text_lines,
                    crop_left=gt_left,
                    crop_top=gt_top,
                    crop_h=self.gt_size[0],
                    crop_w=self.gt_size[1],
                    orig_h=orig_h,
                    orig_w=orig_w,
                )
            )
        return out_item
