"""Deferred-import custom BaseMetric implementations for SRTrainer E2E tests.

Zero module-level imports of ``srtrainer`` or optional ML libraries are used so
``python -m pytest --collect-only tests/e2e/`` always succeeds before Milestone 3
is implemented, while every instantiated metric is a genuine ``BaseMetric``
subclass loadable via Hydra ``_target_: tests.e2e.custom_metrics.<Class>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BOUND_CACHE: dict[type, type] = {}
_BOUND_MODULE_CACHE: dict[type, type] = {}


class _DeferredBaseMetricMixin:
    """Dynamically binds subclasses to ``srtrainer.metrics.BaseMetric`` on instantiation."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        bound = _BOUND_CACHE.get(cls)
        if bound is None:
            from srtrainer.metrics import BaseMetric

            class _Bound(cls, BaseMetric):  # type: ignore[misc, valid-type]
                def __new__(subcls: type, *a: Any, **kw: Any) -> Any:
                    return object.__new__(subcls)

            _Bound.__name__ = cls.__name__
            _Bound.__qualname__ = cls.__qualname__
            _Bound.__module__ = cls.__module__
            _BOUND_CACHE[cls] = _Bound
            bound = _Bound
        return object.__new__(bound)


class MockOCRAMetric(_DeferredBaseMetricMixin):
    """GT-less OCR-A style custom metric consuming ``meta[label_key]``."""

    higher_is_better: bool = True

    def __init__(
        self,
        label_key: str = "label",
        strict: bool = True,
        scale_factor: float = 1.0,
        trace_file: str | None = None,
        trace_tag: str = "val",
    ) -> None:
        super().__init__()
        self.label_key = label_key
        self.strict = bool(strict)
        self.scale_factor = float(scale_factor)
        self.trace_file: Path | None = Path(trace_file) if trace_file is not None else None
        self.trace_tag = str(trace_tag)
        self.seen_metas: list[dict[str, Any] | None] = []

    def compute(
        self,
        pred: Any,
        target: Any | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del target  # Explicitly supports target=None (GT-less evaluation)
        self.seen_metas.append(meta)
        if self.trace_file is not None:
            self.trace_file.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_file.open("a", encoding="utf-8") as fh:
                fh.write(f"{self.trace_tag}\n")
        if meta is None or self.label_key not in meta:
            if self.strict:
                raise KeyError(f"Missing required meta key: {self.label_key!r}")
            label_text = ""
        else:
            label_text = str(meta[self.label_key])

        pred_mean = float(pred.detach().float().clamp(0.0, 1.0).mean().item())
        label_score = min(len(label_text) / 20.0, 1.0)
        return float(self.scale_factor * (0.5 * pred_mean + 0.5 * label_score))


class MockLowerIsBetterMetric(_DeferredBaseMetricMixin):
    """Lower-is-better custom metric (``higher_is_better = False``).

    Supports either direct mean-distance scoring or deterministic scripted
    sequences (optionally persisted via ``state_file`` across runs).
    """

    higher_is_better: bool = False

    def __init__(
        self,
        target_mean: float = 0.5,
        weight: float = 1.0,
        sequence: list[float] | None = None,
        state_file: str | None = None,
    ) -> None:
        super().__init__()
        self.target_mean = float(target_mean)
        self.weight = float(weight)
        self.sequence: list[float] | None = (
            [float(v) for v in sequence] if sequence is not None else None
        )
        self.state_file: Path | None = Path(state_file) if state_file is not None else None
        self._call_idx: int = 0

    def _next_sequence_value(self) -> float:
        assert self.sequence is not None and len(self.sequence) > 0
        if self.state_file is not None:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                idx = int(data.get("idx", 0))
            else:
                idx = 0
            val = self.sequence[min(idx, len(self.sequence) - 1)]
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"idx": idx + 1}), encoding="utf-8")
            return float(val)

        val = self.sequence[min(self._call_idx, len(self.sequence) - 1)]
        self._call_idx += 1
        return float(val)

    def compute(
        self,
        pred: Any,
        target: Any | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del target, meta
        if self.sequence is not None:
            return self._next_sequence_value()
        pred_mean = float(pred.detach().float().clamp(0.0, 1.0).mean().item())
        return float(self.weight * abs(pred_mean - self.target_mean))


class MockNaNMetric(_DeferredBaseMetricMixin):
    """Custom metric returning ``float('nan')`` for NaN best-checkpoint guard tests."""

    higher_is_better: bool = True

    def __init__(self) -> None:
        super().__init__()

    def compute(
        self,
        pred: Any,
        target: Any | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del pred, target, meta
        return float("nan")


class _DeferredModuleMixin:
    """Dynamically binds subclasses to ``torch.nn.Module`` on first instantiation."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        bound = _BOUND_MODULE_CACHE.get(cls)
        if bound is None:
            import torch.nn as nn

            class _BoundModule(cls, nn.Module):  # type: ignore[misc, valid-type]
                def __new__(subcls: type, *a: Any, **kw: Any) -> Any:
                    return object.__new__(subcls)

            _BoundModule.__name__ = cls.__name__
            _BoundModule.__qualname__ = cls.__qualname__
            _BoundModule.__module__ = cls.__module__
            _BOUND_MODULE_CACHE[cls] = _BoundModule
            bound = _BoundModule
        return object.__new__(bound)


class RecordingL1Loss(_DeferredModuleMixin):
    """Genuine ``nn.Module`` wrapping ``srtrainer.losses.L1Loss`` and logging each forward call."""

    def __init__(
        self,
        trace_file: str,
        reduction: str = "mean",
        trace_tag: str = "train_micro",
    ) -> None:
        super().__init__()
        from srtrainer.losses import L1Loss

        self.criterion = L1Loss(reduction=reduction)
        self.trace_file = Path(trace_file)
        self.trace_tag = str(trace_tag)

    def forward(self, pred: Any, target: Any) -> Any:
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{self.trace_tag}\n")
        return self.criterion(pred, target)
