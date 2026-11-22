"""Abstract base class decoupling ``Trainer`` from algorithm-specific training paradigms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

if TYPE_CHECKING:
    from accelerate import Accelerator

__all__ = ["BaseModel"]


class BaseModel(ABC):
    """Abstract base class for all training paradigms managed by ``Trainer``."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        if isinstance(cfg, DictConfig):
            self.cfg: DictConfig = cfg
        elif isinstance(cfg, Mapping):
            created = OmegaConf.create(dict(cfg))
            if not isinstance(created, DictConfig):
                raise TypeError(f"Expected mapping config, got {type(cfg).__name__}.")
            self.cfg = created
        else:
            raise TypeError(
                f"Expected DictConfig or Mapping for cfg, got {type(cfg).__name__}."
            )
        self.accelerator: Accelerator | None = None

    @abstractmethod
    def setup(self, accelerator: Accelerator) -> None:
        """Build networks, losses, degrader, optimizers, schedulers, and EMA on ``accelerator``."""

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Optional device-side batch preprocessing hook (e.g., online GPU degradation)."""
        return batch

    @abstractmethod
    def train_step(
        self,
        batch: dict[str, Any],
        global_step: int,
    ) -> tuple[dict[str, float], bool]:
        """Run one micro-step; return ``(log_scalars, stepped)`` where ``stepped == sync_gradients``."""

    @abstractmethod
    def get_infer_network(self, use_ema: bool = True) -> nn.Module:
        """Return an unwrapped inference-ready network (EMA preferred when enabled and requested)."""

    @abstractmethod
    def export_weights(
        self,
        accelerator: Accelerator,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Export clean, prefix-free inference weight snapshots (e.g., ``{'net_g': ...}``)."""

    def get_lr_dict(self) -> dict[str, float]:
        """Return current learning rates for structured logging and trackers."""
        return {}
