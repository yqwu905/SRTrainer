"""Hydra ConfigStore registration for all SRTrainer Structured Configs."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from config.schema import (
    DataConfig,
    InferRootConfig,
    InferSettingsConfig,
    MetricConfig,
    RealESRGANDegraderConfig,
    TrainConfig,
    TrainerConfig,
    VOSRModelConfig,
)

__all__ = ["register_configs"]


def register_configs() -> None:
    """Register all declarative Structured Config schemas in Hydra's global ``ConfigStore``."""
    cs = ConfigStore.instance()
    cs.store(name="base_train", node=TrainConfig)
    cs.store(group="model", name="base_vosr", node=VOSRModelConfig)
    cs.store(group="trainer", name="base_trainer", node=TrainerConfig)
    cs.store(group="data", name="base_data", node=DataConfig)
    cs.store(group="data/degradation", name="base_realesrgan", node=RealESRGANDegraderConfig)
    cs.store(group="metric", name="base_metric", node=MetricConfig)

    cs.store(name="base_infer", node=InferRootConfig)
    cs.store(group="infer", name="base_infer_settings", node=InferSettingsConfig)
