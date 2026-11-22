"""Validation image dataset and batch_size=1 meta-preserving collate function."""

from __future__ import annotations

import copy
import numbers
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from srtrainer.data.manifest import ValSample, load_val_manifest
from srtrainer.utils.image import load_image

__all__ = [
    "ValImageDataset",
    "val_collate_fn",
]


class ValImageDataset(Dataset[dict[str, Any]]):
    """Validation dataset backed by a unified list-of-dicts JSON manifest.

    - ``lq`` is required (non-empty path).
    - ``gt`` is optional (auto-detected; absent/null/empty -> ``None``).
    - All extra JSON fields plus resolved ``lq_path``, ``gt_path``, and 0-based
      ``index`` are returned in ``meta`` as an isolated deep copy per access.
    """

    def __init__(
        self,
        root_dir: str | Path | None,
        json_path: str | Path,
    ) -> None:
        self.samples: list[ValSample] = load_val_manifest(
            json_path=json_path,
            root_dir=root_dir,
        )
        self.json_path: Path = Path(json_path)
        self.root_dir: Path | None = (
            Path(root_dir)
            if isinstance(root_dir, Path)
            or (isinstance(root_dir, str) and len(root_dir.strip()) > 0)
            else None
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, numbers.Integral):
            raise TypeError(f"Index must be an integer, got {type(index).__name__}.")
        sample = self.samples[int(index)]
        lq = load_image(sample.lq_path)
        gt = load_image(sample.gt_path) if sample.gt_path is not None else None
        return {
            "lq": lq,
            "gt": gt,
            "meta": copy.deepcopy(sample.meta),
        }


def val_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function strictly for ``batch_size=1`` validation DataLoaders.

    Unsqueezes ``lq`` and non-``None`` ``gt`` to ``[1, 3, H, W]`` while preserving
    ``gt=None`` and the raw Python ``meta`` dict untouched without mutating ``batch[0]``.
    """
    if len(batch) != 1:
        raise ValueError(
            f"val_collate_fn strictly requires batch_size=1, got len(batch)={len(batch)}."
        )

    sample = batch[0]
    if not isinstance(sample, Mapping) or "lq" not in sample or "meta" not in sample:
        raise ValueError(
            "Validation sample must be a mapping containing 'lq' and 'meta' keys."
        )

    lq = sample["lq"]
    if not isinstance(lq, torch.Tensor) or lq.ndim != 3:
        raise ValueError(
            f"Expected sample['lq'] to be a 3D [C, H, W] tensor, got "
            f"{tuple(lq.shape) if isinstance(lq, torch.Tensor) else type(lq).__name__}."
        )

    gt = sample.get("gt")
    if gt is not None:
        if not isinstance(gt, torch.Tensor) or gt.ndim != 3:
            raise ValueError(
                f"Expected sample['gt'] to be a 3D [C, H, W] tensor or None, got "
                f"{tuple(gt.shape) if isinstance(gt, torch.Tensor) else type(gt).__name__}."
            )
        gt_out: torch.Tensor | None = gt.unsqueeze(0)
    else:
        gt_out = None

    return {
        "lq": lq.unsqueeze(0),
        "gt": gt_out,
        "meta": sample["meta"],
    }
