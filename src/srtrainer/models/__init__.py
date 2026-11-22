"""Training model paradigms and EMA re-exported for short Hydra ``_target_`` paths."""

from __future__ import annotations

from .base_model import BaseModel
from .ema import ModelEMA
from .sr_model import SRModel

__all__ = ["BaseModel", "SRModel", "ModelEMA"]
