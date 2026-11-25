"""Hydra CLI entrypoint for single-process and multi-GPU/CPU distributed SRTrainer training."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate.state import AcceleratorState, GradientState, PartialState
from omegaconf import DictConfig

from config import TrainConfig, load_train_config, register_configs
from engine import Trainer
from utils.npu import setup_npu

IS_NPU = setup_npu()
register_configs()

__all__ = ["main", "run_train"]

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _reenable_srtrainer_loggers() -> None:
    """Re-enable ``srtrainer`` loggers disabled by Hydra's ``job_logging: disabled``."""
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if (name == "srtrainer" or name.startswith("srtrainer.")) and isinstance(
            obj, logging.Logger
        ):
            obj.disabled = False


def run_train(cfg: TrainConfig | DictConfig | Mapping[str, Any]) -> Trainer:
    """Validate config at entry, run a full training lifecycle, and return the ``Trainer``."""
    train_cfg = load_train_config(cfg)
    had_existing_state = bool(PartialState._shared_state)
    had_existing_dist = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    trainer: Trainer | None = None
    try:
        _reenable_srtrainer_loggers()
        trainer = Trainer(train_cfg)
        trainer.fit()
        return trainer
    finally:
        if trainer is not None:
            acc = getattr(trainer, "accelerator", None)
            if acc is not None and not getattr(trainer, "_training_ended", False):
                try:
                    acc.end_training()
                except Exception:
                    pass
        if (
            not had_existing_dist
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()
        if not had_existing_state:
            AcceleratorState._reset_state(reset_partial_state=True)
            GradientState._reset_state()


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint for training."""
    run_train(cfg)


if __name__ == "__main__":
    main()
