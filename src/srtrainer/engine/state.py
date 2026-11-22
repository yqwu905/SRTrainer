"""Resumable training progress state registered with Accelerate checkpointing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["TrainState"]


def _validate_global_step(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"global_step must be a non-negative integer, got {type(value).__name__}: {value!r}."
        )
    return int(value)


def _coerce_numeric(value: Any, label: str) -> float:
    raw = value
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be numeric (not bool), got {value!r}.")
    if not isinstance(raw, (int, float)):
        if hasattr(raw, "item"):
            try:
                raw = raw.item()
            except Exception as exc:
                raise ValueError(
                    f"{label} must be numeric, got {type(value).__name__}: {value!r}."
                ) from exc
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"{label} must be numeric, got {type(value).__name__}: {value!r}."
            )
    return float(raw)


def _validate_best_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"best_metrics must be a mapping of str -> float, got {type(value).__name__}."
        )
    cleaned: dict[str, float] = {}
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(
                f"best_metrics keys must be non-empty strings, got {type(k).__name__}: {k!r}."
            )
        fv = _coerce_numeric(v, f"best_metrics[{k!r}]")
        if math.isnan(fv):
            raise ValueError(f"best_metrics[{k!r}] cannot be NaN.")
        cleaned[str(k)] = fv
    return cleaned


@dataclass
class TrainState:
    """Mutable iteration and best-metric state registered with ``accelerator.register_for_checkpointing``."""

    global_step: int = 0
    best_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.global_step = _validate_global_step(self.global_step)
        self.best_metrics = _validate_best_metrics(self.best_metrics)

    def state_dict(self) -> dict[str, Any]:
        """Return a pure-primitive, defensive-copied state dict safe for ``weights_only=True``."""
        return {
            "global_step": _validate_global_step(self.global_step),
            "best_metrics": _validate_best_metrics(self.best_metrics),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Atomically validate and restore ``global_step`` and ``best_metrics``."""
        if not isinstance(state, Mapping):
            raise TypeError(
                f"TrainState state_dict must be a Mapping, got {type(state).__name__}."
            )
        if "global_step" not in state:
            raise ValueError("TrainState state_dict is missing required key 'global_step'.")

        new_step = _validate_global_step(state["global_step"])
        raw_best = state["best_metrics"] if "best_metrics" in state else {}
        new_best = _validate_best_metrics(raw_best)

        self.global_step = new_step
        self.best_metrics = new_best

    def update_best(
        self,
        key: str,
        value: float,
        higher_is_better: bool = True,
    ) -> bool:
        """Update ``best_metrics[key]`` iff ``value`` strictly improves over the stored best.

        Returns ``False`` without mutating state when ``value`` is ``NaN``; permits ``+inf``
        (e.g., exact-match PSNR).
        """
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"best metric key must be a non-empty string, got {key!r}.")
        if not isinstance(higher_is_better, bool):
            raise ValueError(
                f"higher_is_better must be a bool, got {type(higher_is_better).__name__}."
            )

        fv = _coerce_numeric(value, f"best_metrics[{key!r}]")
        if math.isnan(fv):
            return False

        prev = self.best_metrics.get(key)
        if prev is None:
            self.best_metrics[key] = fv
            return True

        improved = fv > prev if higher_is_better else fv < prev
        if improved:
            self.best_metrics[key] = fv
            return True
        return False
