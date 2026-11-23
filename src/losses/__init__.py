"""Loss functions re-exported for short Hydra ``_target_`` paths."""

from __future__ import annotations

from .glyph_loss import GlyphLoss
from .ocr_kl_loss import OCRKLLoss
from .perceptual_loss import LPIPSLoss
from .pixel_loss import L1Loss

__all__ = ["L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"]
