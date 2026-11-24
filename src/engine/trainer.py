"""Pure-iteration Accelerate training engine with distributed exact validation and dual-track saving."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import (
    DistributedWeightedInfiniteSampler,
    MixedSRDataset,
    ValImageDataset,
    val_collate_fn,
)
from engine.checkpointer import Checkpointer
from engine.inferencer import TiledInferencer
from engine.state import TrainState
from metrics.base import BaseMetric
from models.base_model import BaseModel
from models.ema import unwrap_model_for_ema
from utils.config import (
    flatten_config_for_tracker,
    normalize_mixed_precision,
    resolve_work_dir,
    save_resolved_config,
    setup_experiment_dirs,
)
from utils.image import load_image, make_comparison_image, save_image
from utils.logging import (
    format_train_log,
    log_image_dict_to_trackers,
    log_images_to_trackers,
    setup_logger,
)
from utils.misc import normalize_gt_size

__all__ = ["Trainer"]


@dataclass
class _ValSuite:
    name: str
    dataset: ValImageDataset
    loader: Any
    metrics: dict[str, BaseMetric]
    max_save_images: int
    vis_stems: list[str]


def _as_dict_config(cfg: DictConfig | Mapping[str, Any]) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    if isinstance(cfg, Mapping):
        created = OmegaConf.create(dict(cfg))
        if isinstance(created, DictConfig):
            return created
    raise TypeError(f"Expected DictConfig or Mapping, got {type(cfg).__name__}.")


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return int(value)


def _require_interval(value: Any, name: str) -> int:
    if value is None:
        return 0
    return _require_non_negative_int(value, name)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}: {value!r}.")
    return value


def _optional_stripped_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None, got {type(value).__name__}: {value!r}.")
    stripped = value.strip()
    return stripped if stripped else None


_VALID_TRAIN_LOG_IMAGE_MODES: frozenset[str] = frozenset({"comparison", "separate"})


def _require_train_log_image_mode(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be one of {sorted(_VALID_TRAIN_LOG_IMAGE_MODES)}, "
            f"got {type(value).__name__}: {value!r}."
        )
    normalized = value.strip().lower()
    if normalized not in _VALID_TRAIN_LOG_IMAGE_MODES:
        raise ValueError(
            f"{name} must be one of {sorted(_VALID_TRAIN_LOG_IMAGE_MODES)}, got {value!r}."
        )
    return normalized


def _normalize_log_with(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if raw is None:
        return None
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        return [stripped]
    if isinstance(raw, (list, tuple)):
        cleaned: list[str] = []
        for idx, elem in enumerate(raw):
            if elem is None:
                continue
            if not isinstance(elem, str):
                raise ValueError(
                    f"trainer.log_with[{idx}] must be a string, got {type(elem).__name__}: {elem!r}."
                )
            stripped = elem.strip()
            if not stripped or stripped.lower() in {"none", "null"}:
                continue
            cleaned.append(stripped)
        return cleaned if cleaned else None
    raise ValueError(
        f"trainer.log_with must be null, str, or list[str], got {type(raw).__name__}: {raw!r}."
    )


def _sanitize_tracker_init_kwargs(
    raw: Any,
    active_log_with: list[str] | None,
) -> dict[str, dict[str, Any]]:
    if raw is None or active_log_with is None:
        return {}
    if OmegaConf.is_config(raw):
        raw = OmegaConf.to_container(raw, resolve=True)
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"trainer.tracker_init_kwargs must be a mapping or None, got {type(raw).__name__}."
        )

    keep_all = any(item.lower() == "all" for item in active_log_with)
    active_set = {item.lower() for item in active_log_with}

    cleaned: dict[str, dict[str, Any]] = {}
    for tracker_key, kwargs_node in raw.items():
        if kwargs_node is None:
            continue
        if not isinstance(kwargs_node, Mapping):
            raise ValueError(
                f"trainer.tracker_init_kwargs[{tracker_key!r}] must be a mapping or None, "
                f"got {type(kwargs_node).__name__}."
            )
        key_str = str(tracker_key)
        if not keep_all and key_str.lower() not in active_set:
            continue
        cleaned[key_str] = dict(kwargs_node)
    return cleaned


def _validate_resume_early(raw_resume: Any) -> None:
    if raw_resume is None or raw_resume is False:
        return
    if isinstance(raw_resume, bool):
        raise ValueError(
            "Ambiguous trainer.resume=True; specify 'auto', a checkpoint directory path, or null."
        )
    if not isinstance(raw_resume, (str, Path)):
        raise ValueError(
            f"trainer.resume must be str, Path, or None, got {type(raw_resume).__name__}: {raw_resume!r}."
        )
    text = str(raw_resume).strip()
    if text.lower() in {"", "none", "null", "false", "auto"}:
        return
    explicit = Path(text)
    if not explicit.exists():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {explicit}")
    if not explicit.is_dir():
        raise ValueError(
            f"Resume path must be a checkpoint state directory, got file: {explicit} "
            "(use model.pretrain_path for weight files)."
        )


def _reenable_srtrainer_loggers() -> None:
    """Re-enable ``srtrainer`` and ``srtrainer.*`` loggers disabled by Hydra job_logging."""
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if (name == "srtrainer" or name.startswith("srtrainer.")) and isinstance(
            obj, logging.Logger
        ):
            obj.disabled = False


def _build_vis_stems(dataset: ValImageDataset) -> list[str]:
    raw_stems = [
        (Path(sample.lq_path).stem or f"img_{idx:04d}")
        for idx, sample in enumerate(dataset.samples)
    ]
    counts = Counter(raw_stems)
    return [
        stem if counts[stem] == 1 else f"{stem}_{idx:04d}"
        for idx, stem in enumerate(raw_stems)
    ]


class Trainer:
    """Iteration-driven Accelerate training engine with distributed exact validation."""

    def __init__(self, cfg: DictConfig | Mapping[str, Any]) -> None:
        cfg_oc = _as_dict_config(cfg)

        # Capture existing singleton/distributed state BEFORE anything touches PartialState.
        self._had_existing_state: bool = bool(PartialState._shared_state)
        self._had_existing_dist: bool = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self._training_ended: bool = False
        self.accelerator: Accelerator | None = None

        # Step 1: Pure fail-fast config validation (no directories or Accelerate state created yet).
        trainer_cfg = OmegaConf.select(cfg_oc, "trainer", default=None)
        if not isinstance(trainer_cfg, (DictConfig, Mapping)):
            raise ValueError("Configuration must contain a `trainer` mapping.")

        self.total_iters: int = _require_positive_int(
            OmegaConf.select(cfg_oc, "trainer.total_iters", default=None),
            "trainer.total_iters",
        )
        self.log_interval: int = _require_interval(
            OmegaConf.select(cfg_oc, "trainer.log_interval", default=100),
            "trainer.log_interval",
        )
        self.val_interval: int = _require_interval(
            OmegaConf.select(cfg_oc, "trainer.val_interval", default=5000),
            "trainer.val_interval",
        )
        self.save_interval: int = _require_interval(
            OmegaConf.select(cfg_oc, "trainer.save_interval", default=5000),
            "trainer.save_interval",
        )

        raw_max_keep = OmegaConf.select(cfg_oc, "trainer.max_keep_states", default=3)
        if raw_max_keep is not None and (
            isinstance(raw_max_keep, bool) or not isinstance(raw_max_keep, int)
        ):
            raise ValueError(
                f"trainer.max_keep_states must be an integer or None, got {raw_max_keep!r}."
            )
        self.max_keep_states: int | None = (
            None if (raw_max_keep is None or raw_max_keep <= 0) else int(raw_max_keep)
        )

        self.grad_accum_steps: int = _require_positive_int(
            OmegaConf.select(cfg_oc, "trainer.gradient_accumulation_steps", default=1),
            "trainer.gradient_accumulation_steps",
        )
        self.mixed_precision: str = normalize_mixed_precision(
            OmegaConf.select(cfg_oc, "trainer.mixed_precision", default="no")
        )
        self.train_log_image_interval: int = _require_interval(
            OmegaConf.select(cfg_oc, "trainer.train_log_image_interval", default=0),
            "trainer.train_log_image_interval",
        )
        self.train_max_log_images: int = _require_non_negative_int(
            OmegaConf.select(cfg_oc, "trainer.train_max_log_images", default=4),
            "trainer.train_max_log_images",
        )
        self.train_log_image_mode: str = _require_train_log_image_mode(
            OmegaConf.select(cfg_oc, "trainer.train_log_image_mode", default="comparison"),
            "trainer.train_log_image_mode",
        )
        self.val_save_visuals: bool = _require_bool(
            OmegaConf.select(cfg_oc, "trainer.val_save_visuals", default=True),
            "trainer.val_save_visuals",
        )
        self.val_max_save_images: int = _require_non_negative_int(
            OmegaConf.select(cfg_oc, "trainer.val_max_save_images", default=8),
            "trainer.val_max_save_images",
        )
        self.val_use_ema: bool = _require_bool(
            OmegaConf.select(cfg_oc, "trainer.val_use_ema", default=True),
            "trainer.val_use_ema",
        )

        self.seed: int = _require_non_negative_int(
            OmegaConf.select(cfg_oc, "seed", default=42),
            "seed",
        )
        self.scale: int = _require_positive_int(
            OmegaConf.select(cfg_oc, "scale", default=None),
            "scale",
        )

        data_cfg = OmegaConf.select(cfg_oc, "data", default=None)
        if not isinstance(data_cfg, (DictConfig, Mapping)):
            raise ValueError("Configuration must contain a `data` mapping.")
        data_scale_raw = OmegaConf.select(cfg_oc, "data.scale", default=None)
        if data_scale_raw is not None:
            data_scale = _require_positive_int(data_scale_raw, "data.scale")
            if data_scale != self.scale:
                raise ValueError(
                    f"data.scale ({data_scale}) must match cfg.scale ({self.scale})."
                )
        self.gt_size = normalize_gt_size(OmegaConf.select(cfg_oc, "data.gt_size"), self.scale)
        batch_size_per_gpu = _require_positive_int(
            OmegaConf.select(cfg_oc, "data.batch_size_per_gpu", default=None),
            "data.batch_size_per_gpu",
        )
        num_workers = _require_non_negative_int(
            OmegaConf.select(cfg_oc, "data.num_workers", default=0),
            "data.num_workers",
        )

        raw_resume = OmegaConf.select(cfg_oc, "trainer.resume", default=None)
        _validate_resume_early(raw_resume)

        normalized_log_with = _normalize_log_with(
            OmegaConf.select(cfg_oc, "trainer.log_with", default=None)
        )
        sanitized_tracker_init_kwargs = _sanitize_tracker_init_kwargs(
            OmegaConf.select(cfg_oc, "trainer.tracker_init_kwargs", default=None),
            normalized_log_with,
        )

        # Validate metric.val_suites and primary_dataset / primary_metric fail-fast.
        raw_suites = OmegaConf.select(cfg_oc, "metric.val_suites", default=None)
        raw_primary_dataset = _optional_stripped_str(
            OmegaConf.select(cfg_oc, "metric.primary_dataset", default=None),
            "metric.primary_dataset",
        )
        raw_primary_metric = _optional_stripped_str(
            OmegaConf.select(cfg_oc, "metric.primary_metric", default=None),
            "metric.primary_metric",
        )

        validated_suite_cfgs: dict[str, dict[str, Any]] = {}
        if raw_suites is not None and not isinstance(raw_suites, (DictConfig, Mapping)):
            raise ValueError(
                f"metric.val_suites must be a mapping or null, got {type(raw_suites).__name__}."
            )
        if raw_suites is not None:
            for suite_name, suite_cfg in raw_suites.items():
                s_name = str(suite_name).strip()
                if not s_name:
                    raise ValueError("Validation suite names in metric.val_suites must be non-empty.")
                if not isinstance(suite_cfg, (DictConfig, Mapping)):
                    raise ValueError(
                        f"metric.val_suites[{s_name!r}] must be a mapping, got {type(suite_cfg).__name__}."
                    )
                ds_cfg = suite_cfg.get("dataset", None)
                if not isinstance(ds_cfg, (DictConfig, Mapping)):
                    raise ValueError(
                        f"metric.val_suites[{s_name!r}].dataset must be a mapping with 'json_path'."
                    )
                json_path = ds_cfg.get("json_path", None)
                if not isinstance(json_path, (str, Path)) or not str(json_path).strip():
                    raise ValueError(
                        f"metric.val_suites[{s_name!r}].dataset.json_path must be a non-empty path."
                    )

                metrics_cfg = suite_cfg.get("metrics", None)
                if metrics_cfg is None:
                    metrics_map: dict[str, Any] = {}
                elif isinstance(metrics_cfg, (DictConfig, Mapping)):
                    metrics_map = {str(k): v for k, v in metrics_cfg.items() if v is not None}
                else:
                    raise ValueError(
                        f"metric.val_suites[{s_name!r}].metrics must be a mapping or null."
                    )

                raw_suite_max_vis = suite_cfg.get("max_save_images", None)
                suite_max_vis = (
                    self.val_max_save_images
                    if raw_suite_max_vis is None
                    else _require_non_negative_int(
                        raw_suite_max_vis,
                        f"metric.val_suites[{s_name!r}].max_save_images",
                    )
                )
                validated_suite_cfgs[s_name] = {
                    "dataset": ds_cfg,
                    "metrics": metrics_map,
                    "max_save_images": suite_max_vis,
                }

        if (raw_primary_dataset is None) != (raw_primary_metric is None):
            raise ValueError(
                "Both metric.primary_dataset and metric.primary_metric must be set together "
                "or both null."
            )
        if raw_primary_dataset is not None and raw_primary_metric is not None:
            if raw_primary_dataset not in validated_suite_cfgs:
                raise ValueError(
                    f"metric.primary_dataset={raw_primary_dataset!r} not found in "
                    f"metric.val_suites {list(validated_suite_cfgs.keys())}."
                )
            primary_suite_metrics = validated_suite_cfgs[raw_primary_dataset]["metrics"]
            if raw_primary_metric not in primary_suite_metrics:
                raise ValueError(
                    f"metric.primary_metric={raw_primary_metric!r} not found in "
                    f"metric.val_suites[{raw_primary_dataset!r}].metrics "
                    f"{list(primary_suite_metrics.keys())}."
                )
        self.primary_dataset: str | None = raw_primary_dataset
        self.primary_metric: str | None = raw_primary_metric

        # Steps 2-10: Clean up Accelerate/distributed singletons if construction fails mid-way.
        try:
            # Step 2: Multi-rank safe work_dir resolution BEFORE Accelerator(...).
            raw_base_work_dir = OmegaConf.select(cfg_oc, "work_dir", default=None)
            base_work_dir: Path | None = (
                Path(str(raw_base_work_dir))
                if raw_base_work_dir is not None and str(raw_base_work_dir).strip()
                else None
            )
            self.work_dir: Path = resolve_work_dir(cfg_oc)
            resolved_container = OmegaConf.to_container(cfg_oc, resolve=True)
            cfg_resolved = OmegaConf.create(resolved_container)
            if not isinstance(cfg_resolved, DictConfig):
                raise TypeError("Resolved configuration must be a DictConfig.")
            cfg_resolved.work_dir = str(self.work_dir)
            self.cfg: DictConfig = cfg_resolved

            # Step 3: Initialize Accelerator + ProjectConfiguration.
            project_config = ProjectConfiguration(
                project_dir=str(self.work_dir),
                logging_dir=str(self.work_dir / "trackers"),
                automatic_checkpoint_naming=False,
            )
            self.accelerator = Accelerator(
                mixed_precision=self.mixed_precision,
                gradient_accumulation_steps=self.grad_accum_steps,
                log_with=normalized_log_with,
                project_config=project_config,
            )

            # Step 4: Rank-0 experiment directory creation & config_resolved.yaml persistence.
            self.accelerator.wait_for_everyone()
            if (
                self.accelerator.is_main_process
                and base_work_dir is not None
                and base_work_dir.resolve() != self.work_dir.resolve()
                and base_work_dir.is_dir()
            ):
                # Remove empty placeholder dir created by Hydra's hydra.run.dir=${work_dir}
                # before append_timestamp resolved the actual broadcast timestamped work_dir.
                try:
                    base_work_dir.rmdir()
                except OSError:
                    pass
            self.dirs: dict[str, Path] = setup_experiment_dirs(
                self.work_dir,
                is_main_process=self.accelerator.is_main_process,
            )
            if self.accelerator.is_main_process:
                save_resolved_config(self.cfg, self.work_dir / "config_resolved.yaml")
            self.accelerator.wait_for_everyone()

            # Step 5: Rank-0 structured logger + Accelerate trackers.
            self.logger: logging.Logger = setup_logger(
                self.work_dir / "train.log",
                is_main_process=self.accelerator.is_main_process,
                name="srtrainer",
            )
            _reenable_srtrainer_loggers()

            if self.accelerator.log_with:
                flat_cfg = flatten_config_for_tracker(self.cfg)
                project_name = str(OmegaConf.select(self.cfg, "exp_name", default="srtrainer"))
                self.accelerator.init_trackers(
                    project_name=project_name,
                    config=flat_cfg,
                    init_kwargs=sanitized_tracker_init_kwargs,
                )

            # Step 6: Phase-A identical cross-rank seeding before model & metric initialization.
            set_seed(self.seed, device_specific=False)

            # Step 7: Instantiate & setup model, TrainState, Checkpointer, and deterministic registration.
            # NOTE: Never wrap model.setup() in main_process_first() because setup() invokes
            # collective accelerator.prepare() (DDP sync). Pretrained loss downloads are guarded
            # inside SRModel.setup() Step 1 before prepare().
            model_obj = hydra.utils.instantiate(self.cfg.model, cfg=self.cfg, _recursive_=False)
            if not isinstance(model_obj, BaseModel):
                raise TypeError(
                    f"Instantiated cfg.model must be a BaseModel, got {type(model_obj).__name__}."
                )
            model_obj.setup(self.accelerator)
            self.model: BaseModel = model_obj

            self.train_state: TrainState = TrainState()
            self.accelerator.register_for_checkpointing(self.train_state)
            self.checkpointer: Checkpointer = Checkpointer(
                work_dir=self.work_dir,
                max_keep_states=self.max_keep_states,
            )

            # Step 8: Training data pipeline (unprepared train_loader + dedicated Generator).
            has_degradation = OmegaConf.select(self.cfg, "data.degradation", default=None) is not None
            kernel_cfg = (
                OmegaConf.select(self.cfg, "data.degradation.kernel_cfg", default=None)
                if has_degradation
                else None
            )
            self.train_dataset = MixedSRDataset(
                sources=self.cfg.data.sources,
                gt_size=self.gt_size,
                scale=self.scale,
                augment=OmegaConf.select(self.cfg, "data.augment", default=None),
                has_degradation=has_degradation,
                kernel_cfg=kernel_cfg,
                is_main_process=self.accelerator.is_main_process,
                logger=self.logger,
                text_crop_prob=float(OmegaConf.select(self.cfg, "data.text_crop_prob", default=0.0)),
            )
            self.train_sampler = DistributedWeightedInfiniteSampler(
                source_sizes=self.train_dataset.source_sizes,
                ratios=self.train_dataset.source_ratios,
                batch_size_per_gpu=batch_size_per_gpu,
                rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes,
                base_seed=self.seed,
                start_micro_step=0,
            )
            pin_memory = (
                bool(OmegaConf.select(self.cfg, "data.pin_memory", default=True))
                and self.accelerator.device.type == "cuda"
            )
            persistent_workers = (
                bool(OmegaConf.select(self.cfg, "data.persistent_workers", default=False))
                and num_workers > 0
            )
            train_dl_gen = torch.Generator()
            train_dl_gen.manual_seed(self.seed + self.accelerator.process_index)
            from data.mixed_dataset import train_collate_fn

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size_per_gpu,
                sampler=self.train_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=train_collate_fn if self.train_dataset.has_text_annotations else None,
                generator=train_dl_gen,
            )

            # Step 9: Validation suites (val_loader prepared ONCE in __init__, metrics cached on device).
            self.val_suites: dict[str, _ValSuite] = {}
            if validated_suite_cfgs:
                for s_name, s_info in validated_suite_cfgs.items():
                    ds_cfg = s_info["dataset"]
                    val_ds = ValImageDataset(
                        root_dir=ds_cfg.get("root_dir", None),
                        json_path=ds_cfg["json_path"],
                    )
                    vis_stems = _build_vis_stems(val_ds)
                    val_dl_gen = torch.Generator()
                    val_dl_gen.manual_seed(self.seed)
                    raw_val_loader = DataLoader(
                        val_ds,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=val_collate_fn,
                        pin_memory=pin_memory,
                        generator=val_dl_gen,
                    )
                    prepared_val_loader = self.accelerator.prepare(raw_val_loader)

                    instantiated_metrics: dict[str, BaseMetric] = {}
                    with self.accelerator.main_process_first():
                        for m_name, m_cfg in s_info["metrics"].items():
                            metric = hydra.utils.instantiate(m_cfg)
                            if not isinstance(metric, BaseMetric):
                                raise TypeError(
                                    f"Instantiated metric {s_name}.{m_name} must be a BaseMetric, "
                                    f"got {type(metric).__name__}."
                                )
                            metric.to(self.accelerator.device)
                            instantiated_metrics[m_name] = metric

                    self.val_suites[s_name] = _ValSuite(
                        name=s_name,
                        dataset=val_ds,
                        loader=prepared_val_loader,
                        metrics=instantiated_metrics,
                        max_save_images=s_info["max_save_images"],
                        vis_stems=vis_stems,
                    )

            # Step 10: Phase-B per-rank seeding & optional resume restoration.
            set_seed(self.seed, device_specific=True)
            resolved_resume_cfg = OmegaConf.select(self.cfg, "trainer.resume", default=None)
            resume_path = self.checkpointer.resolve_resume_path(resolved_resume_cfg)
            if resume_path is None:
                if (
                    isinstance(resolved_resume_cfg, str)
                    and resolved_resume_cfg.strip().lower() == "auto"
                ):
                    self.logger.warning(
                        "[Resume] trainer.resume='auto' requested, but no checkpoint state was "
                        "found under %s; starting fresh from step 0.",
                        self.checkpointer.checkpoints_dir,
                    )
            else:
                self.accelerator.load_state(str(resume_path))
                if self.accelerator.step % self.grad_accum_steps != 0:
                    self.accelerator.step = self.train_state.global_step * self.grad_accum_steps
                start_micro_step = self.train_state.global_step * self.grad_accum_steps
                self.train_sampler.set_start_micro_step(start_micro_step)
                self.logger.info(
                    "[Resume] Restored state from %s at global_step=%d (start_micro_step=%d).",
                    resume_path,
                    self.train_state.global_step,
                    start_micro_step,
                )

            self.accelerator.wait_for_everyone()
        except BaseException:
            self._cleanup_on_init_failure()
            raise

    def _cleanup_on_init_failure(self) -> None:
        if self.accelerator is not None and not self._training_ended:
            self._training_ended = True
            try:
                self.accelerator.end_training()
            except Exception:
                pass
        if (
            not self._had_existing_dist
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()
        if not self._had_existing_state:
            AcceleratorState._reset_state(reset_partial_state=True)
            GradientState._reset_state()

    def _flush_logger(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass

    def _get_mem_gb(self) -> float:
        assert self.accelerator is not None
        if self.accelerator.device.type == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated(self.accelerator.device)) / (1024.0**3)
        return 0.0

    def _save_checkpoint(self, step: int) -> None:
        assert self.accelerator is not None
        self.checkpointer.save_state(self.accelerator, step)
        exported = self.model.export_weights(self.accelerator)
        self.checkpointer.save_weights(self.accelerator, exported, step)

    def _should_log_train_images(self, step: int) -> bool:
        return (
            self.accelerator is not None
            and self.accelerator.is_main_process
            and bool(self.accelerator.trackers)
            and self.train_log_image_interval > 0
            and self.train_max_log_images > 0
            and (step % self.train_log_image_interval == 0 or step == self.total_iters)
        )

    def _log_train_images(self, step: int) -> None:
        assert self.accelerator is not None
        visuals = self.model.pop_train_visuals()
        if not visuals:
            return
        if not self.accelerator.is_main_process or not self.accelerator.trackers:
            return

        lq = visuals.get("lq")
        gt = visuals.get("gt")
        pred = visuals.get("pred")
        if (
            not isinstance(lq, torch.Tensor)
            or not isinstance(gt, torch.Tensor)
            or not isinstance(pred, torch.Tensor)
            or lq.ndim != 4
            or gt.ndim != 4
            or pred.ndim != 4
        ):
            return

        num_images = min(
            self.train_max_log_images,
            int(lq.shape[0]),
            int(gt.shape[0]),
            int(pred.shape[0]),
        )
        if num_images <= 0:
            return

        lq_clean = torch.nan_to_num(
            lq[:num_images].detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        pred_clean = torch.nan_to_num(
            pred[:num_images].detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        gt_clean = torch.nan_to_num(
            gt[:num_images].detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        image_payload: dict[str, torch.Tensor] = {}
        if self.train_log_image_mode == "comparison":
            for idx in range(num_images):
                image_payload[f"train/lq_pred_gt/sample_{idx:02d}"] = make_comparison_image(
                    lq=lq_clean[idx],
                    sr=pred_clean[idx],
                    gt=gt_clean[idx],
                )
        else:
            for idx in range(num_images):
                image_payload[f"train/sample_{idx:02d}/lq"] = lq_clean[idx]
                image_payload[f"train/sample_{idx:02d}/pred"] = pred_clean[idx]
                image_payload[f"train/sample_{idx:02d}/gt"] = gt_clean[idx]

        log_image_dict_to_trackers(self.accelerator, image_payload, step=step)

    def validate(self, step: int) -> dict[str, dict[str, float]]:
        """Run distributed exact validation across all ``metric.val_suites`` using ``TiledInferencer``."""
        assert self.accelerator is not None
        if not self.val_suites:
            return {}

        infer_net = unwrap_model_for_ema(self.model.get_infer_network(use_ema=self.val_use_ema))
        was_training = infer_net.training
        infer_net.eval()

        inferencer = TiledInferencer.from_config(
            model=infer_net,
            scale=self.scale,
            tile_cfg=OmegaConf.select(self.cfg, "trainer.val_tile", default=None),
            gt_size=self.gt_size,
            mixed_precision=self.mixed_precision,
            device=self.accelerator.device,
        )

        results: dict[str, dict[str, float]] = {}
        tracker_metrics: dict[str, float] = {}
        metric_dtype = (
            torch.float32 if self.accelerator.device.type in {"mps", "npu"} else torch.float64
        )

        try:
            for suite_name, suite in self.val_suites.items():
                metric_numerators: dict[str, float] = {m_name: 0.0 for m_name in suite.metrics}
                metric_denominators: dict[str, float] = {m_name: 0.0 for m_name in suite.metrics}
                local_vis_cache: dict[int, torch.Tensor] = {}

                for metric in suite.metrics.values():
                    if hasattr(metric, "clear_sample_cache"):
                        metric.clear_sample_cache()

                try:
                    for batch in suite.loader:
                        lq = batch["lq"]
                        gt = batch.get("gt")
                        meta = batch["meta"]

                        pred = inferencer.infer_tensor(lq)
                        if gt is not None:
                            gt = gt.to(device=pred.device, dtype=torch.float32)

                        for m_name, metric in suite.metrics.items():
                            raw_stats = metric.compute_stats(pred, target=gt, meta=meta)
                            if (
                                not isinstance(raw_stats, (tuple, list))
                                or len(raw_stats) != 2
                                or isinstance(raw_stats[0], bool)
                                or isinstance(raw_stats[1], bool)
                            ):
                                raise ValueError(
                                    f"Metric {suite_name}.{m_name}.compute_stats() must return a "
                                    f"(numerator, denominator) numeric 2-tuple, got {raw_stats!r}."
                                )
                            num_val = float(raw_stats[0])
                            den_val = float(raw_stats[1])
                            if not math.isfinite(den_val) or den_val < 0.0:
                                raise ValueError(
                                    f"Metric {suite_name}.{m_name}.compute_stats() returned "
                                    f"invalid denominator {den_val!r}; expected finite >= 0."
                                )
                            stats_tensor = torch.tensor(
                                [[num_val, den_val]],
                                dtype=metric_dtype,
                                device=self.accelerator.device,
                            )
                            gathered = self.accelerator.gather_for_metrics(stats_tensor)
                            gathered_cpu = gathered.detach().cpu()
                            if gathered_cpu.ndim != 2 or gathered_cpu.shape[1] != 2:
                                raise RuntimeError(
                                    f"Expected gathered stats shape [N, 2] for {suite_name}.{m_name}, "
                                    f"got {tuple(gathered_cpu.shape)}."
                                )
                            for row_num, row_den in gathered_cpu.tolist():
                                metric_numerators[m_name] += float(row_num)
                                metric_denominators[m_name] += float(row_den)

                        gs = self.accelerator.gradient_state
                        is_padded_tail = (
                            self.accelerator.num_processes > 1
                            and gs.end_of_dataloader
                            and gs.remainder > 0
                            and self.accelerator.process_index >= gs.remainder
                        )
                        sample_idx = int(meta.get("index", 0))
                        if (
                            self.val_save_visuals
                            and suite.max_save_images > 0
                            and (not is_padded_tail)
                            and 0 <= sample_idx < suite.max_save_images
                        ):
                            cmp_img = make_comparison_image(lq=lq, sr=pred, gt=gt)
                            vis_stem = suite.vis_stems[sample_idx]
                            save_path = (
                                self.dirs["visuals"]
                                / f"iter_{step:06d}"
                                / suite_name
                                / f"{vis_stem}_cmp.png"
                            )
                            save_image(cmp_img, save_path)
                            if self.accelerator.is_main_process and self.accelerator.trackers:
                                local_vis_cache[sample_idx] = cmp_img.detach().cpu()
                finally:
                    for metric in suite.metrics.values():
                        if hasattr(metric, "clear_sample_cache"):
                            metric.clear_sample_cache()
                    if hasattr(suite.loader, "end"):
                        suite.loader.end()
                    if hasattr(suite.loader, "reset"):
                        suite.loader.reset()
                    gs = self.accelerator.gradient_state
                    if gs.in_dataloader and gs.active_dataloader is suite.loader:
                        gs._remove_dataloader(suite.loader)

                self.accelerator.wait_for_everyone()

                # Deterministic Rank-0 tracker image logging after all ranks finish saving visuals.
                if (
                    self.accelerator.is_main_process
                    and self.accelerator.trackers
                    and self.val_save_visuals
                    and suite.max_save_images > 0
                ):
                    vis_limit = min(len(suite.dataset), suite.max_save_images)
                    for idx in range(vis_limit):
                        vis_stem = suite.vis_stems[idx]
                        img_to_log = local_vis_cache.get(idx)
                        if img_to_log is None:
                            save_path = (
                                self.dirs["visuals"]
                                / f"iter_{step:06d}"
                                / suite_name
                                / f"{vis_stem}_cmp.png"
                            )
                            if save_path.is_file():
                                img_to_log = load_image(save_path)
                        if img_to_log is not None:
                            log_images_to_trackers(
                                self.accelerator,
                                tag=f"val/{suite_name}/{vis_stem}",
                                image=img_to_log,
                                step=step,
                            )

                suite_results: dict[str, float] = {}
                for m_name in suite.metrics:
                    total_num = metric_numerators[m_name]
                    total_den = metric_denominators[m_name]
                    mean_val = float(total_num / total_den) if total_den > 0.0 else float("nan")
                    suite_results[m_name] = mean_val
                    tracker_metrics[f"val/{suite_name}/{m_name}"] = mean_val
                results[suite_name] = suite_results

                if suite_results:
                    summary = " | ".join(f"{k}: {v:.4f}" for k, v in suite_results.items())
                    self.logger.info(
                        "[Val Iter %06d] %s | %s (N=%d)",
                        step,
                        suite_name,
                        summary,
                        len(suite.dataset),
                    )
                else:
                    self.logger.info(
                        "[Val Iter %06d] %s | (visuals only, N=%d)",
                        step,
                        suite_name,
                        len(suite.dataset),
                    )

            if tracker_metrics:
                self.accelerator.log(tracker_metrics, step=step)

            if self.primary_dataset is not None and self.primary_metric is not None:
                current_val = results[self.primary_dataset][self.primary_metric]
                higher_is_better = bool(
                    self.val_suites[self.primary_dataset].metrics[self.primary_metric].higher_is_better
                )
                best_key = f"{self.primary_dataset}/{self.primary_metric}"
                prev_best = self.train_state.best_metrics.get(best_key)
                is_new_best = self.train_state.update_best(
                    best_key,
                    current_val,
                    higher_is_better=higher_is_better,
                )
                if is_new_best:
                    exported = self.model.export_weights(self.accelerator)
                    best_sd = (
                        exported["net_g_ema"]
                        if (self.val_use_ema and "net_g_ema" in exported)
                        else exported["net_g"]
                    )
                    self.checkpointer.save_best(
                        self.accelerator,
                        best_sd,
                        self.primary_dataset,
                        self.primary_metric,
                        step=step,
                        metric_value=current_val,
                    )
                    prev_str = "None" if prev_best is None else f"{prev_best:.4f}"
                    self.logger.info(
                        "[Best] Updated %s: %s -> %.4f (saved best_%s_%s.safetensors)",
                        best_key,
                        prev_str,
                        current_val,
                        self.primary_dataset,
                        self.primary_metric,
                    )
        finally:
            if was_training:
                infer_net.train(True)

        self.accelerator.wait_for_everyone()
        return results

    def fit(self) -> TrainState:
        """Execute the pure-iteration training loop until ``train_state.global_step == total_iters``."""
        assert self.accelerator is not None
        try:
            start_step = self.train_state.global_step
            if start_step >= self.total_iters:
                self.logger.info(
                    "[Train] Already at global_step=%d >= total_iters=%d; skipping training loop.",
                    start_step,
                    self.total_iters,
                )
                self.accelerator.wait_for_everyone()
                return self.train_state

            self.logger.info(
                "[Train] Starting training from step %d to %d (grad_accum=%d, processes=%d, mp=%s, work_dir=%s)",
                start_step,
                self.total_iters,
                self.grad_accum_steps,
                self.accelerator.num_processes,
                self.mixed_precision,
                self.work_dir,
            )

            train_iter = iter(self.train_loader)
            scalar_buf: dict[str, list[float]] = {}
            window_steps = 0
            window_start = time.perf_counter()
            steps_run = 0
            last_val_step: int | None = None
            last_save_step: int | None = None

            while self.train_state.global_step < self.total_iters:
                batch = next(train_iter)
                batch = self.model.preprocess_batch(batch)
                next_step = self.train_state.global_step + 1
                capture_train_visuals = self._should_log_train_images(next_step)
                self.model.set_capture_train_visuals(
                    capture_train_visuals,
                    max_images=self.train_max_log_images if capture_train_visuals else 0,
                )
                step_scalars, stepped = self.model.train_step(batch, self.train_state.global_step)

                if self.log_interval > 0:
                    for key, val in step_scalars.items():
                        scalar_buf.setdefault(key, []).append(float(val))

                if not stepped:
                    continue

                self.train_state.global_step += 1
                step = self.train_state.global_step
                steps_run += 1
                window_steps += 1

                # 1) Periodic training scalar logging.
                if self.log_interval > 0 and (
                    step % self.log_interval == 0 or step == self.total_iters
                ):
                    elapsed = max(time.perf_counter() - window_start, 1e-9)
                    it_per_sec = window_steps / elapsed
                    remaining = max(0, self.total_iters - step)
                    eta_sec = remaining / it_per_sec if it_per_sec > 0.0 else float("inf")
                    mem_gb = self._get_mem_gb()

                    avg_scalars = {
                        k: float(sum(vals) / len(vals))
                        for k, vals in scalar_buf.items()
                        if vals
                    }
                    lr_dict = self.model.get_lr_dict()

                    log_line = format_train_log(
                        step=step,
                        total_iters=self.total_iters,
                        lr_dict=lr_dict,
                        scalars=avg_scalars,
                        it_per_sec=it_per_sec,
                        mem_gb=mem_gb,
                        eta_sec=eta_sec,
                    )
                    self.logger.info(log_line)

                    tracker_payload: dict[str, float] = dict(avg_scalars)
                    for lr_k, lr_v in lr_dict.items():
                        t_key = (
                            lr_k if (lr_k == "lr" or lr_k.startswith("lr/")) else f"lr/{lr_k}"
                        )
                        tracker_payload[t_key] = float(lr_v)
                    self.accelerator.log(tracker_payload, step=step)

                    scalar_buf.clear()
                    window_steps = 0
                    window_start = time.perf_counter()

                # 2) Periodic training visuals, validation, and checkpointing (paused from it/s).
                pause_start = time.perf_counter()
                did_pause = False

                if self._should_log_train_images(step):
                    self._log_train_images(step)
                    did_pause = True
                else:
                    self.model.pop_train_visuals()

                # 3) Periodic / final validation BEFORE checkpoint save so best_metrics & RNG stay synced.
                should_val = bool(self.val_suites) and (
                    (self.val_interval > 0 and step % self.val_interval == 0)
                    or step == self.total_iters
                )
                if should_val:
                    self.validate(step)
                    last_val_step = step
                    did_pause = True

                # 4) Periodic / final dual-track checkpoint save.
                should_save = (
                    (self.save_interval > 0 and step % self.save_interval == 0)
                    or step == self.total_iters
                )
                if should_save:
                    self._save_checkpoint(step)
                    last_save_step = step
                    did_pause = True

                if did_pause:
                    window_start += time.perf_counter() - pause_start

            # End-of-loop defensive fallback (no-op when step == total_iters already ran val & save).
            if steps_run > 0:
                final_step = self.train_state.global_step
                if self.val_suites and last_val_step != final_step:
                    self.validate(final_step)
                    last_val_step = final_step
                if last_save_step != final_step:
                    self._save_checkpoint(final_step)
                    last_save_step = final_step

            self.accelerator.wait_for_everyone()
            self.logger.info(
                "[Train] Finished training at step %d/%d.",
                self.train_state.global_step,
                self.total_iters,
            )
            return self.train_state
        finally:
            self.model.set_capture_train_visuals(False, max_images=0)
            self._flush_logger()
            if not self._training_ended:
                self._training_ended = True
                self.accelerator.end_training()
            if (
                not self._had_existing_dist
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.destroy_process_group()
