"""Abstract base class decoupling ``Trainer`` from algorithm-specific training paradigms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import hydra
import torch
from torch import nn

from config.schema import InferRootConfig, TrainConfig

if TYPE_CHECKING:
    from accelerate import Accelerator

__all__ = ["BaseModel", "build_infer_model", "build_model"]


class BaseModel(ABC):
    """Shared interface for training and standalone inference setup."""

    def __init__(self, cfg: TrainConfig | InferRootConfig) -> None:
        if not isinstance(cfg, (TrainConfig, InferRootConfig)):
            raise TypeError(
                f"BaseModel requires a validated TrainConfig or InferRootConfig, got {type(cfg).__name__}."
            )
        self.cfg = cfg
        self.accelerator: Accelerator | None = None

    def setup_infer(self, device: torch.device | str | None = None) -> nn.Module:
        """Build an evaluation network with model-configured weights on the requested device."""
        raise NotImplementedError(f"{type(self).__name__} does not implement standalone inference.")

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
        """Export clean, prefix-free inference weight snapshots (e.g., per-arch dicts)."""

    def export_infer_weights(
        self,
        accelerator: Accelerator,
        use_ema: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Export the inference weight snapshot(s) corresponding to ``get_infer_network(use_ema)``."""
        exported = self.export_weights(accelerator)
        if "net_g" in exported or "net_g_ema" in exported:
            selected = (
                exported["net_g_ema"]
                if (use_ema and "net_g_ema" in exported)
                else exported["net_g"]
            )
            return {"net_g": selected}
        return exported

    def get_lr_dict(self) -> dict[str, float]:
        """Return current learning rates for structured logging and trackers."""
        return {}

    def set_capture_train_visuals(self, enabled: bool, max_images: int = 0) -> None:
        """Configure whether subsequent ``train_step`` calls snapshot training visuals on CPU."""
        _ = (enabled, max_images)

    def pop_train_visuals(self) -> dict[str, torch.Tensor] | None:
        """Return and clear the latest captured training visual tensors (e.g. ``lq``, ``gt``, ``pred``)."""
        return None


def build_model(cfg: TrainConfig) -> BaseModel:
    """Instantiate the ``BaseModel`` class named by ``cfg.model._target_`` with typed ``TrainConfig``."""
    if not isinstance(cfg, TrainConfig):
        raise TypeError(f"build_model requires a validated TrainConfig, got {type(cfg).__name__}.")
    model_cls = hydra.utils.get_class(cfg.model._target_)
    model_obj = model_cls(cfg)
    if not isinstance(model_obj, BaseModel):
        raise TypeError(
            f"Instantiated cfg.model ({cfg.model._target_!r}) must be a BaseModel, "
            f"got {type(model_obj).__name__}."
        )
    return model_obj


def build_infer_model(cfg: InferRootConfig, device: torch.device | str | None = None) -> nn.Module:
    """Build the configured model for standalone inference."""
    model_cfg = cfg.model
    target = (
        model_cfg.get("_target_") if isinstance(model_cfg, Mapping)
        else getattr(model_cfg, "_target_", None)
    )
    if not target:
        raise ValueError("Standalone inference requires model._target_.")
    target_obj = hydra.utils.get_object(target)
    if isinstance(target_obj, type) and issubclass(target_obj, BaseModel):
        return target_obj(cfg).setup_infer(device=device)

    net = hydra.utils.instantiate(model_cfg)
    if not isinstance(net, nn.Module):
        raise TypeError(f"Expected {target} to instantiate an nn.Module, got {type(net).__name__}.")
    return net.to(device=device).eval()
