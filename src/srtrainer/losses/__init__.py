"""Loss functions re-exported for short Hydra ``_target_`` paths."""

from __future__ import annotations

from .perceptual_loss import LPIPSLoss
from .pixel_loss import L1Loss

__all__ = ["L1Loss", "LPIPSLoss"]
