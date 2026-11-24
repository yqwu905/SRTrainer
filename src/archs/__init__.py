"""Neural network architectures re-exported for short Hydra ``_target_`` paths."""

from __future__ import annotations

from .vosr2 import AutoencoderKLQwenImage2D, DINOv2Encoder, VOSR2DiT

__all__ = [
    "AutoencoderKLQwenImage2D",
    "VOSR2DiT",
    "DINOv2Encoder",
]
