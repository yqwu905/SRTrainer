"""Unified abstract evaluation metric protocol (``BaseMetric``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

__all__ = ["BaseMetric"]


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics used during validation.

    Supports full-reference metrics (consuming ``pred`` and ``target``),
    no-reference metrics (``target=None``), and auxiliary-label metrics
    (e.g., OCR-A reading ``meta["label"]`` with optional or absent ``target``).
    """

    higher_is_better: bool = True

    def to(self, device: torch.device | str) -> BaseMetric:
        """Move metric state or underlying models to ``device`` in-place and return ``self``."""
        return self

    @abstractmethod
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        """Compute a scalar evaluation metric value.

        Args:
            pred: Model output RGB tensor of shape ``[1, 3, H, W]`` (or ``[N, 3, H, W]``) in ``[0, 1]``.
            target: Optional GT RGB tensor in ``[0, 1]``, or ``None`` when no GT is available.
            meta: Optional sample metadata dictionary forwarded from the validation dataset
                (e.g., auxiliary fields like ``label``, ``text_lines``, ``lq_path``, ``gt_path``, ``index``).

        Returns:
            Scalar Python ``float`` metric value.
        """

    def compute_stats(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Compute additive ``(numerator, denominator)`` statistics for global weighted reduction.

        The default implementation wraps :meth:`compute` with unit weight ``1.0`` so existing
        scalar metrics remain backward-compatible without modification. Metrics whose dataset-level
        score is a ratio of summed counts (e.g., OCR-A, CER, and 1-NED) override this method to
        return exact per-sample ``(numerator, denominator)`` statistics and ``(0.0, 0.0)`` for
        ignored or zero-valid-box samples.
        """
        return (float(self.compute(pred, target=target, meta=meta)), 1.0)
