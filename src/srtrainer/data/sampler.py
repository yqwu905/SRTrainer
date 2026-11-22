"""Distributed ratio-weighted infinite sampler with O(1) deterministic resume."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from torch.utils.data import Sampler

__all__ = ["DistributedWeightedInfiniteSampler"]


def _normalize_sequence(value: Any, name: str) -> list[Any]:
    if OmegaConf.is_config(value):
        try:
            value = OmegaConf.to_container(value, resolve=True, throw_on_missing=True)
        except OmegaConfBaseException as exc:
            raise ValueError(f"Invalid OmegaConf value for {name}: {exc}") from exc
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a 1D sequence, got ndarray with shape {value.shape}.")
        return value.tolist()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence, got {type(value).__name__}.")
    return list(value)


def _require_int(value: Any, name: str, min_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {min_value}, got {value!r}.")
    int_val = int(value)
    if int_val < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {int_val}.")
    return int_val


def _draw_indices_for_source(
    rng: np.random.Generator,
    source_size: int,
    count: int,
) -> np.ndarray:
    """Draw ``count`` sample indices in ``[0, source_size)`` without replacement when possible."""
    if count <= source_size:
        return rng.choice(source_size, size=count, replace=False)
    if source_size == 1:
        return np.zeros(count, dtype=np.int64)
    full_rounds, rem = divmod(count, source_size)
    parts = [rng.permutation(source_size) for _ in range(full_rounds)]
    if rem > 0:
        parts.append(rng.choice(source_size, size=rem, replace=False))
    return np.concatenate(parts)


class DistributedWeightedInfiniteSampler(Sampler[tuple[int, int]]):
    """Infinite ratio-weighted multi-source sampler with deterministic per-step multi-rank slicing.

    Yields ``(source_id, sample_idx)`` tuples where source selection probabilities depend strictly
    on normalized ``ratios`` (completely decoupled from ``source_sizes``). Each global
    ``micro_step`` seeds a local ``np.random.Generator(base_seed + micro_step)``, draws
    ``world_size * batch_size_per_gpu`` tuples, and yields slice ``[rank :: world_size]`` so
    distributed ranks receive disjoint slices and resuming at ``start_micro_step`` is O(1).
    """

    def __init__(
        self,
        source_sizes: Sequence[int],
        ratios: Sequence[float],
        batch_size_per_gpu: int,
        rank: int = 0,
        world_size: int = 1,
        base_seed: int = 42,
        start_micro_step: int = 0,
    ) -> None:
        super().__init__()

        raw_sizes = _normalize_sequence(source_sizes, "source_sizes")
        if len(raw_sizes) == 0:
            raise ValueError("source_sizes must be non-empty.")
        clean_sizes = [
            _require_int(size, f"source_sizes[{idx}]", min_value=1)
            for idx, size in enumerate(raw_sizes)
        ]

        raw_ratios = _normalize_sequence(ratios, "ratios")
        if len(raw_ratios) != len(clean_sizes):
            raise ValueError(
                f"Length of ratios ({len(raw_ratios)}) must match length of "
                f"source_sizes ({len(clean_sizes)})."
            )

        clean_ratios: list[float] = []
        for idx, ratio in enumerate(raw_ratios):
            if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
                raise ValueError(
                    f"ratios[{idx}] must be a finite real number >= 0, got {ratio!r}."
                )
            r_f = float(ratio)
            if not math.isfinite(r_f) or r_f < 0.0:
                raise ValueError(f"ratios[{idx}] must be finite and >= 0, got {ratio!r}.")
            clean_ratios.append(r_f)

        ratio_arr = np.asarray(clean_ratios, dtype=np.float64)
        max_ratio = float(ratio_arr.max())
        if max_ratio <= 0.0:
            raise ValueError(f"Sum of ratios must be positive, got {clean_ratios!r}.")
        scaled = ratio_arr / max_ratio
        probs = scaled / float(scaled.sum())

        self.source_sizes: list[int] = clean_sizes
        self.ratios: list[float] = clean_ratios
        self._probs: np.ndarray = probs

        self.batch_size_per_gpu: int = _require_int(
            batch_size_per_gpu, "batch_size_per_gpu", min_value=1
        )
        self.world_size: int = _require_int(world_size, "world_size", min_value=1)
        rank_i = _require_int(rank, "rank", min_value=0)
        if rank_i >= self.world_size:
            raise ValueError(
                f"rank ({rank_i}) must satisfy 0 <= rank < world_size ({self.world_size})."
            )
        self.rank: int = rank_i
        self.base_seed: int = _require_int(base_seed, "base_seed", min_value=0)
        self.start_micro_step: int = _require_int(
            start_micro_step, "start_micro_step", min_value=0
        )

    @property
    def probs(self) -> np.ndarray:
        """Return a defensive copy of the normalized source sampling probabilities."""
        return self._probs.copy()

    def set_start_micro_step(self, start_micro_step: int) -> None:
        """Set the starting micro-step for O(1) deterministic resume."""
        self.start_micro_step = _require_int(start_micro_step, "start_micro_step", min_value=0)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        micro_step = self.start_micro_step
        num_sources = len(self.source_sizes)
        global_batch = self.world_size * self.batch_size_per_gpu

        while True:
            rng = np.random.default_rng(self.base_seed + micro_step)
            source_ids = rng.choice(num_sources, size=global_batch, p=self._probs)
            global_indices: list[tuple[int, int]] = [(0, 0)] * global_batch

            for sid in range(num_sources):
                positions = np.flatnonzero(source_ids == sid)
                count = int(positions.size)
                if count == 0:
                    continue
                chosen = _draw_indices_for_source(rng, self.source_sizes[sid], count)
                for pos, sample_idx in zip(positions.tolist(), chosen.tolist(), strict=True):
                    global_indices[pos] = (sid, int(sample_idx))

            yield from global_indices[self.rank :: self.world_size]
            micro_step += 1
