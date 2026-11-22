"""Pixel-domain loss module (``L1Loss``)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

_VALID_REDUCTIONS = ("mean", "sum", "none")


def _validate_reduction(reduction: str) -> None:
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(
            f"Unsupported reduction={reduction!r}; expected one of {_VALID_REDUCTIONS}."
        )


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred has shape {tuple(pred.shape)}, "
            f"target has shape {tuple(target.shape)}."
        )


class L1Loss(nn.Module):
    """Pixel-wise L1 loss."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        _validate_reduction(reduction)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _validate_pair(pred, target)
        return F.l1_loss(pred, target, reduction=self.reduction)
