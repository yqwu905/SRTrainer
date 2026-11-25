"""Centralized config loading, OmegaConf Structured Config binding, and business validation."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from config.schema import (
    VOSR_COMPONENTS,
    VOSR_REQUIRED_COMPONENTS,
    ArchGroupConfig,
    AugmentConfig,
    DataConfig,
    EMAConfig,
    InferRootConfig,
    InferSettingsConfig,
    KernelConfig,
    LoRAConfig,
    LossEntryConfig,
    MetricConfig,
    OptimConfig,
    PretrainEntryConfig,
    TileConfig,
    TrainConfig,
    TrainerConfig,
    VOSRModelConfig,
)

__all__ = [
    "ConfigValidationError",
    "load_structured_config",
    "load_train_config",
    "load_infer_config",
    "load_tile_config",
    "load_augment_config",
    "load_kernel_config",
    "load_model_config",
    "validate_train_config",
    "validate_infer_config",
    "validate_tile_config",
    "to_resolved_dict_config",
]

T = TypeVar("T")


class ConfigValidationError(ValueError):
    """Raised when structured config schema or business invariant validation fails."""


def _to_omegaconf_input(raw: Any, context: str) -> DictConfig:
    if isinstance(raw, DictConfig):
        return raw
    if is_dataclass(raw) and not isinstance(raw, type):
        return OmegaConf.structured(raw)
    if isinstance(raw, Mapping):
        try:
            created = OmegaConf.create(dict(raw))
        except OmegaConfBaseException as exc:
            raise ConfigValidationError(f"Invalid {context}: {exc}") from exc
        if isinstance(created, DictConfig):
            return created
    raise ConfigValidationError(
        f"Expected {context} mapping or dataclass, got {type(raw).__name__}."
    )


def load_structured_config(
    schema_cls: type[T],
    raw: T | DictConfig | Mapping[str, Any],
    *,
    context: str | None = None,
) -> T:
    """Merge ``raw`` into ``schema_cls`` via OmegaConf Structured Config and return a dataclass."""
    label = context or schema_cls.__name__
    if isinstance(raw, schema_cls):
        return raw

    raw_dc = _to_omegaconf_input(raw, label)
    try:
        base = OmegaConf.structured(schema_cls)
        merged = OmegaConf.merge(base, raw_dc)
        OmegaConf.resolve(merged)
        obj = OmegaConf.to_object(merged)
    except OmegaConfBaseException as exc:
        raise ConfigValidationError(f"Invalid {label}: {exc}") from exc

    if not isinstance(obj, schema_cls):
        raise ConfigValidationError(
            f"Expected {schema_cls.__name__} after structured merge, got {type(obj).__name__}."
        )
    return obj


def _require_non_empty_str(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ConfigValidationError(f"{field_name} must be a non-empty string.")
    return stripped


def _require_optional_non_empty_str(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ConfigValidationError(
            f"{field_name} cannot be an empty string; pass null/None to omit."
        )
    return stripped


def _require_target(node: Any, field_name: str) -> None:
    if isinstance(node, Mapping):
        target = node.get("_target_")
    elif is_dataclass(node) and not isinstance(node, type):
        target = getattr(node, "_target_", None)
    else:
        raise ConfigValidationError(f"{field_name} must be a mapping or dataclass defining '_target_'.")
    if not isinstance(target, str) or not target.strip():
        raise ConfigValidationError(f"{field_name} must define a non-empty string '_target_'.")


def validate_tile_config(
    tile: TileConfig,
    *,
    field_prefix: str = "tile",
    scale: int | None = None,
    gt_size: int | None = None,
) -> None:
    """Validate business ranges and cross-field constraints on ``TileConfig``."""
    if tile.overlap < 0:
        raise ConfigValidationError(
            f"{field_prefix}.overlap must be >= 0, got {tile.overlap!r}."
        )
    if tile.batch_size < 1:
        raise ConfigValidationError(
            f"{field_prefix}.batch_size must be >= 1, got {tile.batch_size!r}."
        )
    if tile.pad_multiple < 1:
        raise ConfigValidationError(
            f"{field_prefix}.pad_multiple must be >= 1, got {tile.pad_multiple!r}."
        )
    if tile.size is not None:
        if len(tile.size) != 2 or any(dim < 1 for dim in tile.size):
            raise ConfigValidationError(
                f"{field_prefix}.size must be null or a 2-element list [tile_h, tile_w] of "
                f"positive integers, got {tile.size!r}."
            )
        if tile.enabled and tile.overlap > 0:
            min_tile = min(tile.size[0], tile.size[1])
            if tile.overlap >= min_tile // 2:
                raise ConfigValidationError(
                    f"{field_prefix}.overlap={tile.overlap} must satisfy "
                    f"0 <= overlap < min(tile_h, tile_w) // 2 ({min_tile // 2}) "
                    f"for size={tile.size}."
                )
    elif tile.enabled and tile.overlap > 0 and scale is not None and gt_size is not None:
        if scale >= 1 and gt_size >= 1 and gt_size % scale == 0:
            auto_dim = gt_size // scale
            if tile.overlap >= auto_dim // 2:
                raise ConfigValidationError(
                    f"{field_prefix}.overlap={tile.overlap} must satisfy "
                    f"0 <= overlap < (gt_size // scale) // 2 ({auto_dim // 2}) "
                    f"for auto tile size ({auto_dim}, {auto_dim})."
                )


def _validate_ema_config(ema: EMAConfig) -> None:
    if not math.isfinite(ema.decay) or not (0.0 < ema.decay <= 1.0):
        raise ConfigValidationError(f"model.ema.decay must be finite in (0, 1], got {ema.decay!r}.")
    if ema.warmup_steps < 0:
        raise ConfigValidationError(
            f"model.ema.warmup_steps must be >= 0, got {ema.warmup_steps!r}."
        )


def _validate_lora_config(lora: LoRAConfig) -> None:
    if lora.lora_rank < 1:
        raise ConfigValidationError(
            f"model.lora.lora_rank must be >= 1, got {lora.lora_rank!r}."
        )
    if not math.isfinite(lora.lora_alpha) or lora.lora_alpha <= 0.0:
        raise ConfigValidationError(
            f"model.lora.lora_alpha must be finite and > 0, got {lora.lora_alpha!r}."
        )
    _require_optional_non_empty_str(lora.checkpoint, "model.lora.checkpoint")


def _validate_pretrain_entry(entry: PretrainEntryConfig, comp_name: str) -> None:
    _require_optional_non_empty_str(entry.path, f"model.pretrain.{comp_name}.path")
    if entry.key is not None and not entry.key.strip():
        entry.key = None


def _validate_vosr_model_and_arch(
    model: VOSRModelConfig,
    arch: ArchGroupConfig,
    *,
    require_vosr_arch: bool = True,
) -> None:
    _require_non_empty_str(model._target_, "model._target_")
    if model.scale < 1:
        raise ConfigValidationError(f"model.scale must be >= 1, got {model.scale!r}.")
    _validate_ema_config(model.ema)
    _validate_lora_config(model.lora)
    for comp_name in VOSR_COMPONENTS:
        _validate_pretrain_entry(model.pretrain.get(comp_name), comp_name)

    if not require_vosr_arch:
        return

    active_components: list[str] = []
    for comp_name in VOSR_COMPONENTS:
        comp_cfg = arch.get(comp_name)
        if comp_cfg is None:
            if comp_name in VOSR_REQUIRED_COMPONENTS:
                raise ConfigValidationError(
                    f"arch.{comp_name} is required for VOSRModel and cannot be null."
                )
            if model.trainable.get(comp_name):
                raise ConfigValidationError(
                    f"Cannot set model.trainable.{comp_name}=True when arch.{comp_name} is null."
                )
            if model.pretrain.get(comp_name).path is not None:
                raise ConfigValidationError(
                    f"Cannot set model.pretrain.{comp_name}.path when arch.{comp_name} is null."
                )
            continue
        _require_target(comp_cfg, f"arch.{comp_name}")
        active_components.append(comp_name)

    if not any(model.trainable.get(comp_name) for comp_name in active_components):
        raise ConfigValidationError(
            "At least one active VOSR arch component must be trainable "
            f"(model.trainable={model.trainable.as_dict()})."
        )


def _validate_losses(loss: dict[str, LossEntryConfig]) -> None:
    if not loss:
        raise ConfigValidationError("loss must be a non-empty mapping of named loss entries.")
    active_count = 0
    for loss_name, entry in loss.items():
        if not math.isfinite(entry.weight):
            raise ConfigValidationError(
                f"loss.{loss_name}.weight must be finite, got {entry.weight!r}."
            )
        if entry.weight > 0.0:
            _require_target(entry.criterion, f"loss.{loss_name}.criterion")
            active_count += 1
    if active_count == 0:
        raise ConfigValidationError(
            "loss must contain at least one active loss entry with weight > 0."
        )


def _validate_optim(optim: OptimConfig) -> None:
    _require_target(optim.optimizer, "optim.optimizer")
    if optim.scheduler is not None:
        _require_target(optim.scheduler, "optim.scheduler")


def _validate_data(data: DataConfig, global_scale: int) -> None:
    if data.scale < 1:
        raise ConfigValidationError(f"data.scale must be >= 1, got {data.scale!r}.")
    if data.scale != global_scale:
        raise ConfigValidationError(
            f"data.scale ({data.scale}) must match scale ({global_scale})."
        )
    if data.gt_size < 1:
        raise ConfigValidationError(f"data.gt_size must be >= 1, got {data.gt_size!r}.")
    if data.gt_size % global_scale != 0:
        raise ConfigValidationError(
            f"data.gt_size ({data.gt_size}) must be evenly divisible by scale ({global_scale})."
        )
    if data.batch_size_per_gpu < 1:
        raise ConfigValidationError(
            f"data.batch_size_per_gpu must be >= 1, got {data.batch_size_per_gpu!r}."
        )
    if data.num_workers < 0:
        raise ConfigValidationError(f"data.num_workers must be >= 0, got {data.num_workers!r}.")
    if not math.isfinite(data.text_crop_prob) or not (0.0 <= data.text_crop_prob <= 1.0):
        raise ConfigValidationError(
            f"data.text_crop_prob must be finite in [0.0, 1.0], got {data.text_crop_prob!r}."
        )
    if not data.sources:
        raise ConfigValidationError("data.sources must be a non-empty list.")
    ratios: list[float] = []
    for idx, src in enumerate(data.sources):
        _require_non_empty_str(src.name, f"data.sources[{idx}].name")
        _require_non_empty_str(src.root_dir, f"data.sources[{idx}].root_dir")
        _require_non_empty_str(src.json_path, f"data.sources[{idx}].json_path")
        _require_optional_non_empty_str(src.glyphs_json, f"data.sources[{idx}].glyphs_json")
        if not math.isfinite(src.ratio) or src.ratio < 0.0:
            raise ConfigValidationError(
                f"data.sources[{idx}].ratio must be finite and >= 0, got {src.ratio!r}."
            )
        ratios.append(src.ratio)
    if max(ratios) <= 0.0:
        raise ConfigValidationError(
            f"Sum of data.sources ratios must be finite and > 0, got {ratios!r}."
        )
    if data.degradation is not None:
        _require_target(data.degradation, "data.degradation")


def _validate_metric(metric: MetricConfig) -> None:
    if metric.primary_dataset is not None:
        metric.primary_dataset = metric.primary_dataset.strip() or None
    if metric.primary_metric is not None:
        metric.primary_metric = metric.primary_metric.strip() or None

    for suite_name, suite in metric.val_suites.items():
        if not suite_name.strip():
            raise ConfigValidationError("Validation suite names in metric.val_suites must be non-empty.")
        _require_non_empty_str(
            suite.dataset.json_path, f"metric.val_suites[{suite_name!r}].dataset.json_path"
        )
        if suite.max_save_images is not None and suite.max_save_images < 0:
            raise ConfigValidationError(
                f"metric.val_suites[{suite_name!r}].max_save_images must be >= 0, "
                f"got {suite.max_save_images!r}."
            )
        for m_name, m_cfg in suite.metrics.items():
            _require_target(m_cfg, f"metric.val_suites[{suite_name!r}].metrics[{m_name!r}]")

    if (metric.primary_dataset is None) != (metric.primary_metric is None):
        raise ConfigValidationError(
            "Both metric.primary_dataset and metric.primary_metric must be set together or both null."
        )
    if metric.primary_dataset is not None and metric.primary_metric is not None:
        if metric.primary_dataset not in metric.val_suites:
            raise ConfigValidationError(
                f"metric.primary_dataset={metric.primary_dataset!r} not found in "
                f"metric.val_suites {list(metric.val_suites.keys())}."
            )
        primary_metrics = metric.val_suites[metric.primary_dataset].metrics
        if metric.primary_metric not in primary_metrics:
            raise ConfigValidationError(
                f"metric.primary_metric={metric.primary_metric!r} not found in "
                f"metric.val_suites[{metric.primary_dataset!r}].metrics "
                f"{list(primary_metrics.keys())}."
            )


def _validate_resume(resume: str | None) -> None:
    if resume is None:
        return
    stripped = resume.strip()
    if not stripped:
        raise ConfigValidationError(
            "trainer.resume cannot be empty; specify null, 'auto', or a checkpoint directory path."
        )
    if stripped.lower() == "auto":
        return
    explicit = Path(stripped)
    if not explicit.exists():
        raise FileNotFoundError(f"Resume checkpoint directory not found: {explicit}")
    if not explicit.is_dir():
        raise ConfigValidationError(
            f"Resume path must be a checkpoint state directory, got file: {explicit} "
            "(use model.pretrain.<arch>.path for weight files)."
        )


def _validate_trainer(
    trainer: TrainerConfig,
    *,
    scale: int,
    gt_size: int,
    has_val_suites: bool,
) -> None:
    if trainer.total_iters < 1:
        raise ConfigValidationError(f"trainer.total_iters must be >= 1, got {trainer.total_iters!r}.")
    for interval_name in (
        "log_interval",
        "val_interval",
        "save_interval",
        "train_log_image_interval",
    ):
        val = getattr(trainer, interval_name)
        if val < 0:
            raise ConfigValidationError(f"trainer.{interval_name} must be >= 0, got {val!r}.")
    if trainer.max_keep_states is not None and trainer.max_keep_states < 0:
        raise ConfigValidationError(
            f"trainer.max_keep_states must be >= 0 or null, got {trainer.max_keep_states!r}."
        )
    if trainer.gradient_accumulation_steps < 1:
        raise ConfigValidationError(
            "trainer.gradient_accumulation_steps must be >= 1, "
            f"got {trainer.gradient_accumulation_steps!r}."
        )
    if trainer.grad_clip_norm is not None:
        if not math.isfinite(trainer.grad_clip_norm) or trainer.grad_clip_norm <= 0.0:
            raise ConfigValidationError(
                f"trainer.grad_clip_norm must be finite and > 0 (or null), "
                f"got {trainer.grad_clip_norm!r}."
            )
    if trainer.train_max_log_images < 0:
        raise ConfigValidationError(
            f"trainer.train_max_log_images must be >= 0, got {trainer.train_max_log_images!r}."
        )
    if trainer.val_max_save_images < 0:
        raise ConfigValidationError(
            f"trainer.val_max_save_images must be >= 0, got {trainer.val_max_save_images!r}."
        )
    for idx, entry in enumerate(trainer.log_with):
        if not entry.strip():
            raise ConfigValidationError(
                f"trainer.log_with[{idx}] must be a non-empty tracker name, got {entry!r}."
            )
    _validate_resume(trainer.resume)
    check_auto_tile = trainer.val_interval > 0 and has_val_suites
    validate_tile_config(
        trainer.val_tile,
        field_prefix="trainer.val_tile",
        scale=scale if check_auto_tile else None,
        gt_size=gt_size if check_auto_tile else None,
    )


def validate_train_config(
    cfg: TrainConfig,
    *,
    require_data: bool = True,
    require_trainer_full: bool = True,
) -> TrainConfig:
    """Run centralized business-rule validation on a materialized ``TrainConfig``."""
    _require_non_empty_str(cfg.exp_name, "exp_name")
    _require_non_empty_str(cfg.work_dir, "work_dir")
    if cfg.scale < 1:
        raise ConfigValidationError(f"scale must be >= 1, got {cfg.scale!r}.")
    if cfg.seed < 0:
        raise ConfigValidationError(f"seed must be >= 0, got {cfg.seed!r}.")

    is_vosr_model = cfg.model._target_ == "models.VOSRModel"
    _validate_vosr_model_and_arch(cfg.model, cfg.arch, require_vosr_arch=is_vosr_model)
    if is_vosr_model:
        _validate_losses(cfg.loss)
        _validate_optim(cfg.optim)

    if require_data:
        _validate_data(cfg.data, global_scale=cfg.scale)

    _validate_metric(cfg.metric)

    if require_trainer_full:
        _validate_trainer(
            cfg.trainer,
            scale=cfg.scale,
            gt_size=cfg.data.gt_size,
            has_val_suites=bool(cfg.metric.val_suites),
        )
    elif cfg.trainer.grad_clip_norm is not None:
        if not math.isfinite(cfg.trainer.grad_clip_norm) or cfg.trainer.grad_clip_norm <= 0.0:
            raise ConfigValidationError(
                f"trainer.grad_clip_norm must be finite and > 0 (or null), "
                f"got {cfg.trainer.grad_clip_norm!r}."
            )

    return cfg


def load_train_config(
    raw: TrainConfig | DictConfig | Mapping[str, Any],
    *,
    require_data: bool = True,
    require_trainer_full: bool = True,
) -> TrainConfig:
    """Merge ``raw`` into ``TrainConfig``, materialize dataclasses, and validate business rules."""
    if not isinstance(raw, TrainConfig):
        raw_trainer = raw.get("trainer") if isinstance(raw, (Mapping, DictConfig)) else None
        if isinstance(raw_trainer, (Mapping, DictConfig)) and isinstance(
            raw_trainer.get("resume"), bool
        ):
            raise ConfigValidationError(
                "trainer.resume must be null, 'auto', or a checkpoint directory path, not a bool."
            )
    cfg = load_structured_config(TrainConfig, raw, context="train config")
    return validate_train_config(
        cfg,
        require_data=require_data,
        require_trainer_full=require_trainer_full,
    )


def validate_infer_config(cfg: InferRootConfig) -> InferRootConfig:
    """Run centralized business-rule validation on a materialized ``InferRootConfig``."""
    if cfg.scale < 1:
        raise ConfigValidationError(f"scale must be >= 1, got {cfg.scale!r}.")
    inf: InferSettingsConfig = cfg.infer
    _require_non_empty_str(inf.output_dir, "infer.output_dir")
    _require_optional_non_empty_str(inf.exp_dir, "infer.exp_dir")
    _require_optional_non_empty_str(inf.weights_path, "infer.weights_path")
    _require_optional_non_empty_str(inf.input_root, "infer.input_root")
    for comp in VOSR_COMPONENTS:
        _require_optional_non_empty_str(
            inf.component_weights.get(comp), f"infer.component_weights.{comp}"
        )
    if inf.weights_path is not None and inf.component_weights.any_set():
        raise ConfigValidationError(
            "Cannot set both infer.weights_path and infer.component_weights simultaneously."
        )
    if inf.weights_key is not None and inf.component_weights_key.any_set():
        raise ConfigValidationError(
            "Cannot set both infer.weights_key and infer.component_weights_key simultaneously."
        )

    forbidden = {"/", "\\", os.sep, "\x00"}
    if os.altsep:
        forbidden.add(os.altsep)
    if any(tok in inf.suffix for tok in forbidden) or ".." in inf.suffix:
        raise ConfigValidationError(
            f"infer.suffix={inf.suffix!r} contains forbidden path separator or traversal characters."
        )

    validate_tile_config(inf.tile, field_prefix="infer.tile")
    if cfg.model is not None:
        if cfg.model.scale < 1:
            raise ConfigValidationError(f"model.scale must be >= 1, got {cfg.model.scale!r}.")
        _validate_ema_config(cfg.model.ema)
        _validate_lora_config(cfg.model.lora)
    return cfg


def load_infer_config(raw: InferRootConfig | DictConfig | Mapping[str, Any]) -> InferRootConfig:
    """Merge ``raw`` into ``InferRootConfig``, materialize dataclasses, and validate business rules."""
    cfg = load_structured_config(InferRootConfig, raw, context="infer config")
    return validate_infer_config(cfg)


def load_tile_config(raw: TileConfig | DictConfig | Mapping[str, Any] | None) -> TileConfig:
    """Materialize and validate a ``TileConfig``."""
    if raw is None:
        return TileConfig()
    tile = load_structured_config(TileConfig, raw, context="tile config")
    validate_tile_config(tile)
    return tile


def load_augment_config(
    raw: AugmentConfig | DictConfig | Mapping[str, Any] | None,
) -> AugmentConfig:
    """Materialize an ``AugmentConfig`` (defaulting to all-False when explicitly ``None``)."""
    if raw is None:
        return AugmentConfig(hflip=False, vflip=False, rot90=False)
    return load_structured_config(AugmentConfig, raw, context="augment config")


def _validate_kernel_config(kcfg: KernelConfig) -> None:
    if kcfg.kernel_size < 3 or kcfg.kernel_size % 2 == 0:
        raise ConfigValidationError(
            f"kernel_cfg.kernel_size must be an odd integer >= 3, got {kcfg.kernel_size!r}."
        )
    if kcfg.kernel_range is not None:
        if not kcfg.kernel_range:
            raise ConfigValidationError("kernel_cfg.kernel_range must be non-empty when set.")
        for k in kcfg.kernel_range:
            if k < 1 or k > kcfg.kernel_size or k % 2 == 0:
                raise ConfigValidationError(
                    f"kernel_cfg.kernel_range entries must be odd integers in "
                    f"[1, {kcfg.kernel_size}], got {kcfg.kernel_range!r}."
                )


def load_kernel_config(raw: KernelConfig | DictConfig | Mapping[str, Any] | None) -> KernelConfig:
    """Materialize and validate a ``KernelConfig``."""
    if raw is None:
        return KernelConfig()
    kcfg = load_structured_config(KernelConfig, raw, context="kernel_cfg")
    _validate_kernel_config(kcfg)
    return kcfg


def load_model_config(raw: VOSRModelConfig | DictConfig | Mapping[str, Any]) -> VOSRModelConfig:
    """Materialize a ``VOSRModelConfig``."""
    return load_structured_config(VOSRModelConfig, raw, context="model config")


def to_resolved_dict_config(cfg: TrainConfig | InferRootConfig | DictConfig | Mapping[str, Any]) -> DictConfig:
    """Convert a structured config dataclass or mapping into a fully resolved ``DictConfig``."""
    dc = _to_omegaconf_input(cfg, "config")
    container = OmegaConf.to_container(dc, resolve=True, enum_to_str=True)
    resolved = OmegaConf.create(container)
    if not isinstance(resolved, DictConfig):
        raise ConfigValidationError(f"Expected DictConfig root, got {type(resolved).__name__}.")
    return resolved
