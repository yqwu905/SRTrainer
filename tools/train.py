"""Hydra CLI entrypoint for single-process and multi-GPU/CPU distributed SRTrainer training."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate.state import AcceleratorState, GradientState, PartialState
from omegaconf import DictConfig, OmegaConf

from engine import Trainer
from utils.npu import setup_npu
IS_NPU = setup_npu()

__all__ = ["main", "run_train"]

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _as_dict_config(cfg: DictConfig | Mapping[str, Any]) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    if isinstance(cfg, Mapping):
        created = OmegaConf.create(dict(cfg))
        if isinstance(created, DictConfig):
            return created
    raise TypeError(f"Expected DictConfig or Mapping, got {type(cfg).__name__}.")


def _reenable_srtrainer_loggers() -> None:
    """Re-enable ``srtrainer`` loggers disabled by Hydra's ``job_logging: disabled``."""
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if (name == "srtrainer" or name.startswith("srtrainer.")) and isinstance(
            obj, logging.Logger
        ):
            obj.disabled = False


def run_train(cfg: DictConfig) -> Trainer:
    """Run a full training lifecycle from a Hydra config and return the ``Trainer`` instance."""
    cfg_oc = _as_dict_config(cfg)
    had_existing_state = bool(PartialState._shared_state)
    had_existing_dist = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    trainer: Trainer | None = None
    try:
        _reenable_srtrainer_loggers()
        trainer = Trainer(cfg_oc)
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
