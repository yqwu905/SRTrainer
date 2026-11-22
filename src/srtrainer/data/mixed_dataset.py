"""Sample-level mixed training dataset supporting auto-detected gt_only and paired JSON sources."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from torch.utils.data import Dataset

from srtrainer.data.degradation.kernels import KernelGenerator
from srtrainer.data.manifest import SourceStats, TrainSample, validate_sources_config
from srtrainer.data.transforms import augment_tensors, paired_random_crop, random_crop
from srtrainer.utils.image import load_image

__all__ = ["MixedSRDataset"]


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


class MixedSRDataset(Dataset[dict[str, torch.Tensor]]):
    """Multi-source mixed super-resolution training dataset with per-sample auto-detection.

    Supports both ``(source_id, sample_idx)`` tuple indexing (used by
    ``DistributedWeightedInfiniteSampler``) and flat integer indexing. Emits uniform tensor
    dictionaries with zero placeholders so ``gt_only`` and ``paired`` samples collate seamlessly
    into a single batch for GPU masked degradation.
    """

    def __init__(
        self,
        sources: Sequence[Mapping[str, Any]],
        gt_size: int,
        scale: int = 4,
        augment: Mapping[str, Any] | None = None,
        has_degradation: bool = True,
        kernel_cfg: Mapping[str, Any] | None = None,
        is_main_process: bool = True,
        logger: Any = None,
    ) -> None:
        self.gt_size: int = _validate_positive_int(gt_size, "gt_size")
        self.scale: int = _validate_positive_int(scale, "scale")
        if self.gt_size % self.scale != 0:
            raise ValueError(
                f"gt_size ({self.gt_size}) must be divisible by scale ({self.scale})."
            )
        self.lq_size: int = self.gt_size // self.scale

        if not isinstance(has_degradation, bool):
            raise TypeError(
                f"has_degradation must be a bool, got {type(has_degradation).__name__}."
            )
        self.has_degradation: bool = has_degradation

        if augment is None:
            self._hflip = False
            self._vflip = False
            self._rot90 = False
        else:
            norm_aug = _normalize_omegaconf(augment, "augment")
            if not isinstance(norm_aug, Mapping):
                raise TypeError(
                    f"augment must be a mapping or None, got {type(norm_aug).__name__}."
                )
            flags: dict[str, bool] = {}
            for key in ("hflip", "vflip", "rot90"):
                val = norm_aug.get(key, False)
                if not isinstance(val, bool):
                    raise TypeError(
                        f"augment[{key!r}] must be a bool, got {type(val).__name__}."
                    )
                flags[key] = val
            self._hflip = flags["hflip"]
            self._vflip = flags["vflip"]
            self._rot90 = flags["rot90"]

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

        if self.has_degradation:
            if kernel_cfg is None:
                plain_kernel_cfg: dict[str, Any] = {}
            else:
                norm_kcfg = _normalize_omegaconf(kernel_cfg, "kernel_cfg")
                if not isinstance(norm_kcfg, Mapping):
                    raise TypeError(
                        f"kernel_cfg must be a mapping or None, got {type(norm_kcfg).__name__}."
                    )
                plain_kernel_cfg = dict(norm_kcfg)
            self.kernel_generator: KernelGenerator | None = KernelGenerator(plain_kernel_cfg)
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

    def __getitem__(self, index: tuple[int, int] | int) -> dict[str, torch.Tensor]:
        source_id, sample_idx = self._resolve_index(index)
        sample = self._samples_per_source[source_id][sample_idx]
        do_aug = self._hflip or self._vflip or self._rot90

        if not sample.need_degradation:
            assert sample.lq_path is not None
            gt = load_image(sample.gt_path)
            lq = load_image(sample.lq_path)
            gt, lq = paired_random_crop(gt, lq, gt_size=self.gt_size, scale=self.scale)
            if do_aug:
                gt, lq = augment_tensors(
                    [gt, lq],
                    hflip=self._hflip,
                    vflip=self._vflip,
                    rot90=self._rot90,
                )

            item: dict[str, torch.Tensor] = {
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
            return item

        assert self.kernel_generator is not None
        gt = load_image(sample.gt_path)
        gt = random_crop(gt, gt_size=self.gt_size, pad_if_smaller=True)
        if do_aug:
            [gt] = augment_tensors(
                [gt],
                hflip=self._hflip,
                vflip=self._vflip,
                rot90=self._rot90,
            )

        lq = torch.zeros((3, self.lq_size, self.lq_size), dtype=torch.float32)
        kernel1, kernel2, sinc_kernel = self.kernel_generator.sample()
        return {
            "gt": gt.contiguous(),
            "lq": lq,
            "need_degradation": torch.tensor(True, dtype=torch.bool),
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "kernel1": kernel1,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
        }
