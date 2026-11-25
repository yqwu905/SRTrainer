"""Dual-mode and multi-GPU/CPU distributed offline inference CLI backed by TiledInferencer."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import hydra
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from omegaconf import DictConfig, OmegaConf
from torch import nn

from config import (
    VOSR_COMPONENTS,
    VOSR_REQUIRED_COMPONENTS,
    InferRootConfig,
    MetricConfig,
    SaveFormat,
    TrainerConfig,
    VOSRModelConfig,
    load_infer_config,
    load_structured_config,
    register_configs,
)
from config.validation import _validate_lora_config
from engine.inferencer import TiledInferencer
from models.vosr_model import (
    VOSRPipeline,
    _apply_dit_lora,
    _clone_arch_cfg_skip_pretrained,
    _load_component_weights,
    should_attach_dit_lora,
)
from utils.config import load_resolved_config
from utils.image import IMAGE_EXTENSIONS, load_image, save_image
from utils.logging import setup_logger
from utils.misc import load_weights, normalize_gt_size

register_configs()

__all__ = [
    "ALLOWED_SAVE_FORMATS",
    "InferItem",
    "InferSummary",
    "ResolvedInferConfig",
    "collect_input_items",
    "main",
    "normalize_save_format",
    "normalize_suffix",
    "resolve_exp_component_weights",
    "resolve_exp_weights",
    "resolve_infer_config",
    "run_infer",
]

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
ALLOWED_SAVE_FORMATS: frozenset[str] = frozenset(fmt.value for fmt in SaveFormat)


@dataclass(frozen=True)
class ResolvedInferConfig:
    """Resolved model architecture, scale, optional gt_size, and weight selection."""

    mode: Literal["exp_dir", "standalone"]
    arch_cfg: DictConfig
    scale: int
    gt_size: tuple[int, int] | None
    weights_path: Path
    weights_key: str | None
    exp_dir: Path | None = None
    component_weights: dict[str, Path | None] | None = None
    component_weights_keys: dict[str, str | None] | None = None
    model_cfg: VOSRModelConfig | None = None


@dataclass(frozen=True)
class InferItem:
    """Single resolved inference input/output pair."""

    input_path: Path
    rel_path: Path
    output_path: Path


@dataclass(frozen=True)
class InferSummary:
    """Summary returned by ``run_infer`` after synchronized completion."""

    mode: Literal["exp_dir", "standalone"]
    total_images: int
    local_images: int
    elapsed_sec: float
    images_per_sec: float
    output_dir: Path
    weights_path: Path
    component_weights: dict[str, Path | None] | None = None


def normalize_save_format(save_format: SaveFormat | str) -> str:
    """Validate and normalize ``infer.save_format`` to ``'png'`` or ``'jpg'``."""
    if isinstance(save_format, SaveFormat):
        return save_format.value
    if not isinstance(save_format, str):
        raise ValueError(
            f"infer.save_format must be a string in {sorted(ALLOWED_SAVE_FORMATS)}, "
            f"got {type(save_format).__name__}: {save_format!r}."
        )
    norm = save_format.strip().lstrip(".").lower()
    if norm not in ALLOWED_SAVE_FORMATS:
        raise ValueError(
            f"Unsupported infer.save_format={save_format!r}; "
            f"expected one of {sorted(ALLOWED_SAVE_FORMATS)}."
        )
    return norm


def normalize_suffix(suffix: str | None) -> str:
    """Validate output filename suffix, rejecting path separators and traversal tokens."""
    if suffix is None:
        return ""
    if not isinstance(suffix, str):
        raise ValueError(f"infer.suffix must be a string, got {type(suffix).__name__}.")
    forbidden = {"/", "\\", os.sep, "\x00"}
    if os.altsep:
        forbidden.add(os.altsep)
    if any(tok in suffix for tok in forbidden) or ".." in suffix:
        raise ValueError(
            f"infer.suffix={suffix!r} contains forbidden path separator or traversal characters."
        )
    return suffix


def _require_non_empty_path(value: str | Path | None, field_name: str) -> Path:
    if value is None or not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{field_name} must be a non-empty path.")
    return Path(str(value).strip())


def _is_non_empty_arch_override(raw_arch: Any) -> bool:
    if raw_arch is None:
        return False
    if isinstance(raw_arch, Mapping):
        return any(v is not None for v in raw_arch.values())
    return True


def _is_multi_arch_cfg(arch_cfg: Any) -> bool:
    if not isinstance(arch_cfg, (DictConfig, Mapping)):
        return False
    if arch_cfg.get("_target_", None) is not None:
        return False
    return any(k in arch_cfg and arch_cfg.get(k) is not None for k in VOSR_COMPONENTS)


def _validate_arch_cfg(arch_cfg: Any, context: str) -> DictConfig:
    if not isinstance(arch_cfg, (DictConfig, Mapping)):
        raise ValueError(f"{context} must provide a mapping `arch` config with `_target_`.")
    arch_dc = arch_cfg if isinstance(arch_cfg, DictConfig) else OmegaConf.create(dict(arch_cfg))
    assert isinstance(arch_dc, DictConfig)
    target = arch_dc.get("_target_", None)
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"{context} `arch` config must define a non-empty `_target_`.")
    return arch_dc


def _validate_multi_arch_cfg(arch_cfg: Any, context: str) -> tuple[DictConfig, tuple[str, ...]]:
    if not isinstance(arch_cfg, (DictConfig, Mapping)):
        raise ValueError(
            f"{context} must provide a mapping `arch` config containing 'vae' and 'dit'."
        )
    arch_dc = arch_cfg if isinstance(arch_cfg, DictConfig) else OmegaConf.create(dict(arch_cfg))
    assert isinstance(arch_dc, DictConfig)
    active: list[str] = []
    for comp in VOSR_COMPONENTS:
        sub = arch_dc.get(comp, None)
        if sub is None:
            if comp in VOSR_REQUIRED_COMPONENTS:
                raise ValueError(f"{context} `arch.{comp}` is required for VOSR inference.")
            continue
        _validate_arch_cfg(sub, context=f"{context} `arch.{comp}`")
        active.append(comp)
    return arch_dc, tuple(active)


def _arch_has_builtin_pretrained(comp_cfg: DictConfig | Mapping[str, Any]) -> bool:
    pretrained_path = comp_cfg.get("pretrained_path", None)
    if isinstance(pretrained_path, (str, Path)) and str(pretrained_path).strip():
        return True
    return comp_cfg.get("load_hub_weights", False) is True


def _extract_exp_subconfigs(
    train_cfg: DictConfig,
) -> tuple[VOSRModelConfig, MetricConfig, TrainerConfig]:
    """Load typed model/metric/trainer sub-configs from a resolved training YAML."""
    raw_model = train_cfg.get("model", None)
    model_cfg = (
        load_structured_config(VOSRModelConfig, raw_model, context="config_resolved.model")
        if raw_model is not None
        else VOSRModelConfig()
    )
    _validate_lora_config(model_cfg.lora)
    raw_metric = train_cfg.get("metric", None)
    metric_cfg = (
        load_structured_config(MetricConfig, raw_metric, context="config_resolved.metric")
        if raw_metric is not None
        else MetricConfig()
    )
    raw_trainer = train_cfg.get("trainer", None)
    trainer_cfg = (
        load_structured_config(TrainerConfig, raw_trainer, context="config_resolved.trainer")
        if raw_trainer is not None
        else TrainerConfig()
    )
    return model_cfg, metric_cfg, trainer_cfg


def resolve_exp_weights(
    exp_dir: Path,
    train_cfg: DictConfig,
    use_ema: bool,
    explicit_weights_path: str | Path | None = None,
) -> Path:
    """Resolve weight file for Mode 1 single-arch ``exp_dir`` auto-restore."""
    if explicit_weights_path is not None:
        explicit = _require_non_empty_path(explicit_weights_path, "infer.weights_path")
        if not explicit.is_file():
            raise FileNotFoundError(f"Weight file not found: {explicit}")
        return explicit

    weights_dir = Path(exp_dir) / "weights"
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights directory not found under exp_dir: {weights_dir}")

    _, metric_cfg, trainer_cfg = _extract_exp_subconfigs(train_cfg)
    primary_dataset = metric_cfg.primary_dataset
    primary_metric = metric_cfg.primary_metric
    val_use_ema = trainer_cfg.val_use_ema

    allow_best = bool(use_ema) or (not val_use_ema)
    checked: list[str] = []

    if allow_best:
        if primary_dataset and primary_metric:
            primary_best = weights_dir / f"best_{primary_dataset}_{primary_metric}.safetensors"
            checked.append(primary_best.name)
            if primary_best.is_file():
                return primary_best

        best_candidates = [p for p in weights_dir.glob("best_*.safetensors") if p.is_file()]
        checked.append("best_*.safetensors")
        if best_candidates:
            best_candidates.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
            return best_candidates[0]

    if bool(use_ema):
        latest_ema = weights_dir / "latest_ema.safetensors"
        checked.append(latest_ema.name)
        if latest_ema.is_file():
            return latest_ema

    latest = weights_dir / "latest.safetensors"
    checked.append(latest.name)
    if latest.is_file():
        return latest

    raise FileNotFoundError(
        f"No eligible weight file found in {weights_dir} "
        f"(use_ema={bool(use_ema)}, val_use_ema={val_use_ema}; checked: {checked})."
    )


def resolve_exp_component_weights(
    exp_dir: Path,
    train_cfg: DictConfig,
    active_components: tuple[str, ...],
    use_ema: bool,
    explicit_weights: Mapping[str, Path | None] | None = None,
) -> dict[str, Path | None]:
    """Resolve per-arch weight files for Mode 1 multi-arch VOSR ``exp_dir`` auto-restore."""
    weights_dir = Path(exp_dir) / "weights"
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights directory not found under exp_dir: {weights_dir}")

    model_cfg, metric_cfg, trainer_cfg = _extract_exp_subconfigs(train_cfg)
    primary_dataset = metric_cfg.primary_dataset
    primary_metric = metric_cfg.primary_metric
    val_use_ema = trainer_cfg.val_use_ema
    allow_best = bool(use_ema) or (not val_use_ema)

    # Choose one consistent best_* prefix across all components when allow_best is True.
    winning_best_prefix: str | None = None
    if allow_best:
        if primary_dataset and primary_metric:
            cand_prefix = f"best_{primary_dataset}_{primary_metric}"
            if any((weights_dir / f"{cand_prefix}_{c}.safetensors").is_file() for c in active_components):
                winning_best_prefix = cand_prefix
        if winning_best_prefix is None:
            prefix_mtime: dict[str, int] = {}
            for comp in active_components:
                suffix_str = f"_{comp}.safetensors"
                for path in weights_dir.glob(f"best_*{suffix_str}"):
                    if not path.is_file():
                        continue
                    prefix = path.name[: -len(suffix_str)]
                    if not prefix.startswith("best_"):
                        continue
                    mtime = path.stat().st_mtime_ns
                    if prefix not in prefix_mtime or mtime > prefix_mtime[prefix]:
                        prefix_mtime[prefix] = mtime
            if prefix_mtime:
                winning_best_prefix = max(
                    prefix_mtime.items(),
                    key=lambda item: (item[1], item[0]),
                )[0]

    explicit = explicit_weights or {}
    resolved: dict[str, Path | None] = {}
    for comp in active_components:
        if comp in explicit and explicit[comp] is not None:
            p = explicit[comp]
            assert p is not None
            if not p.is_file():
                raise FileNotFoundError(f"Weight file not found for {comp!r}: {p}")
            resolved[comp] = p
            continue

        checked: list[str] = []
        chosen: Path | None = None
        if allow_best and winning_best_prefix is not None:
            cand = weights_dir / f"{winning_best_prefix}_{comp}.safetensors"
            checked.append(cand.name)
            if cand.is_file():
                chosen = cand

        if chosen is None and bool(use_ema):
            cand_ema = weights_dir / f"latest_{comp}_ema.safetensors"
            checked.append(cand_ema.name)
            if cand_ema.is_file():
                chosen = cand_ema

        if chosen is None:
            cand_latest = weights_dir / f"latest_{comp}.safetensors"
            checked.append(cand_latest.name)
            if cand_latest.is_file():
                chosen = cand_latest

        if chosen is None:
            comp_cfg = train_cfg.arch.get(comp, None) if "arch" in train_cfg else None
            if not model_cfg.trainable.get(comp):
                pretrain_path_str = model_cfg.pretrain.get(comp).path
                if pretrain_path_str is not None:
                    pretrain_path = Path(pretrain_path_str)
                    if pretrain_path.is_file():
                        resolved[comp] = pretrain_path
                        continue
                if isinstance(comp_cfg, (DictConfig, Mapping)) and _arch_has_builtin_pretrained(comp_cfg):
                    resolved[comp] = None
                    continue
            raise FileNotFoundError(
                f"No eligible weight file found for component {comp!r} in {weights_dir} "
                f"(use_ema={bool(use_ema)}, val_use_ema={val_use_ema}; checked: {checked})."
            )
        resolved[comp] = chosen

    return resolved


def _resolve_component_override_paths(
    infer_root: InferRootConfig,
    active_components: tuple[str, ...],
) -> tuple[dict[str, Path | None], dict[str, str | None]]:
    inf = infer_root.infer
    if inf.weights_path is not None:
        raise ValueError(
            "For multi-arch VOSR inference, use `infer.component_weights.{vae,dit,dino}` "
            "instead of scalar `infer.weights_path`."
        )
    explicit_paths: dict[str, Path | None] = {}
    for comp in VOSR_COMPONENTS:
        raw_p = inf.component_weights.get(comp)
        if raw_p is None:
            continue
        if comp not in active_components:
            raise ValueError(
                f"infer.component_weights.{comp} was provided, but arch.{comp} is not active."
            )
        p = Path(raw_p)
        if not p.is_file():
            raise FileNotFoundError(f"Weight file not found for {comp!r}: {p}")
        explicit_paths[comp] = p

    comp_keys: dict[str, str | None] = {}
    for comp in active_components:
        if inf.weights_key is not None:
            comp_keys[comp] = inf.weights_key.strip() or None
        else:
            raw_k = inf.component_weights_key.get(comp)
            comp_keys[comp] = raw_k.strip() if (raw_k and raw_k.strip()) else None
    return explicit_paths, comp_keys


def resolve_infer_config(
    cfg: InferRootConfig | DictConfig | Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> ResolvedInferConfig:
    """Resolve Mode 1 (``infer.exp_dir``) or Mode 2 (standalone ``infer.yaml``) settings."""
    infer_root = load_infer_config(cfg)
    inf = infer_root.infer

    if inf.exp_dir is not None:
        exp_dir = Path(inf.exp_dir)
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

        config_path = exp_dir / "config_resolved.yaml"
        train_cfg = load_resolved_config(config_path)

        arch_raw = train_cfg.get("arch", None)
        if _is_non_empty_arch_override(infer_root.arch) and logger is not None:
            logger.warning(
                "infer.exp_dir=%s is set; restoring arch from %s and ignoring CLI arch override.",
                exp_dir,
                config_path,
            )

        raw_scale = train_cfg.get("scale", None)
        if raw_scale is None and "model" in train_cfg and train_cfg.model is not None:
            raw_scale = train_cfg.model.get("scale", None)
        if raw_scale is None and isinstance(arch_raw, (DictConfig, Mapping)):
            raw_scale = arch_raw.get("scale", None)
        if raw_scale is None and "data" in train_cfg and train_cfg.data is not None:
            raw_scale = train_cfg.data.get("scale", None)
        if raw_scale is None:
            raw_scale = infer_root.scale
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, int) or raw_scale < 1:
            raise ValueError(f"scale must be a positive integer, got {raw_scale!r}.")
        scale = int(raw_scale)

        raw_gt_size = (
            train_cfg.data.get("gt_size", None)
            if ("data" in train_cfg and train_cfg.data is not None)
            else None
        )
        gt_size = (
            normalize_gt_size(raw_gt_size, scale, field_name="data.gt_size")
            if raw_gt_size is not None
            else None
        )

        model_cfg, _, _ = _extract_exp_subconfigs(train_cfg)

        if _is_multi_arch_cfg(arch_raw):
            arch_cfg, active_components = _validate_multi_arch_cfg(
                arch_raw, context=f"config_resolved.yaml ({config_path})"
            )
            explicit_map, comp_keys = _resolve_component_override_paths(
                infer_root, active_components
            )
            comp_weights = resolve_exp_component_weights(
                exp_dir=exp_dir,
                train_cfg=train_cfg,
                active_components=active_components,
                use_ema=inf.use_ema,
                explicit_weights=explicit_map,
            )
            summary_path = (
                comp_weights.get("dit")
                or next((p for p in comp_weights.values() if p is not None), None)
                or (exp_dir / "weights")
            )
            return ResolvedInferConfig(
                mode="exp_dir",
                arch_cfg=arch_cfg,
                scale=scale,
                gt_size=gt_size,
                weights_path=summary_path,
                weights_key=None,
                exp_dir=exp_dir,
                component_weights=comp_weights,
                component_weights_keys=comp_keys,
                model_cfg=model_cfg,
            )

        if inf.component_weights.any_set():
            raise ValueError(
                "infer.component_weights is only supported for multi-arch VOSR models; "
                "use infer.weights_path for single-arch inference."
            )
        arch_cfg = _validate_arch_cfg(arch_raw, context=f"config_resolved.yaml ({config_path})")
        weights_key = inf.weights_key.strip() if (inf.weights_key and inf.weights_key.strip()) else None
        weights_path = resolve_exp_weights(
            exp_dir=exp_dir,
            train_cfg=train_cfg,
            use_ema=inf.use_ema,
            explicit_weights_path=inf.weights_path,
        )
        return ResolvedInferConfig(
            mode="exp_dir",
            arch_cfg=arch_cfg,
            scale=scale,
            gt_size=gt_size,
            weights_path=weights_path,
            weights_key=weights_key,
            exp_dir=exp_dir,
        )

    # Mode 2: Standalone infer.yaml
    arch_raw = infer_root.arch
    if not _is_non_empty_arch_override(arch_raw):
        raise ValueError(
            "Standalone inference mode (infer.exp_dir=null) requires `arch` "
            "(e.g. `arch/vae=qwen_vae2d arch/dit=vosr2_dit arch/dino=dinov2_vitl14`)."
        )

    scale = infer_root.scale
    model_cfg = infer_root.model if infer_root.model is not None else VOSRModelConfig(scale=scale)

    if _is_multi_arch_cfg(arch_raw):
        arch_cfg, active_components = _validate_multi_arch_cfg(
            arch_raw, context="Standalone inference config"
        )
        explicit_map, comp_keys = _resolve_component_override_paths(infer_root, active_components)
        comp_weights: dict[str, Path | None] = {}
        for comp in active_components:
            if comp in explicit_map and explicit_map[comp] is not None:
                comp_weights[comp] = explicit_map[comp]
            elif _arch_has_builtin_pretrained(arch_cfg[comp]):
                comp_weights[comp] = None
            else:
                raise ValueError(
                    f"Standalone VOSR inference requires `infer.component_weights.{comp}` or "
                    f"`arch.{comp}.pretrained_path` for component {comp!r}."
                )
        summary_path = (
            comp_weights.get("dit")
            or next((p for p in comp_weights.values() if p is not None), None)
            or Path(str(arch_cfg.dit.get("pretrained_path", "builtin")))
        )
        return ResolvedInferConfig(
            mode="standalone",
            arch_cfg=arch_cfg,
            scale=scale,
            gt_size=None,
            weights_path=summary_path,
            weights_key=None,
            exp_dir=None,
            component_weights=comp_weights,
            component_weights_keys=comp_keys,
            model_cfg=model_cfg,
        )

    if inf.component_weights.any_set():
        raise ValueError(
            "infer.component_weights is only supported for multi-arch VOSR models; "
            "use infer.weights_path for single-arch inference."
        )
    arch_cfg = _validate_arch_cfg(arch_raw, context="Standalone inference config")
    weights_key = inf.weights_key.strip() if (inf.weights_key and inf.weights_key.strip()) else None
    if inf.weights_path is None:
        raise ValueError(
            "Standalone inference mode (infer.exp_dir=null) requires `infer.weights_path`."
        )
    explicit_weights = Path(inf.weights_path)
    if not explicit_weights.is_file():
        raise FileNotFoundError(f"Weight file not found: {explicit_weights}")

    return ResolvedInferConfig(
        mode="standalone",
        arch_cfg=arch_cfg,
        scale=scale,
        gt_size=None,
        weights_path=explicit_weights,
        weights_key=weights_key,
        exp_dir=None,
    )


def _extract_manifest_image_path(record: dict[str, Any], index: int) -> str:
    for field_name in ("lq", "gt"):
        if field_name in record and record[field_name] is not None:
            val = record[field_name]
            if not isinstance(val, str):
                raise ValueError(
                    f"Manifest item {index} field {field_name!r} must be a string, "
                    f"got {type(val).__name__}."
                )
            stripped = val.strip()
            if stripped:
                return stripped
    raise ValueError(f"Manifest item {index} must define a non-empty 'lq' or 'gt' string path.")


def _derive_safe_manifest_rel_path(
    raw_str: str,
    resolved: Path,
    root_dir: Path | None,
    manifest_dir: Path,
) -> Path:
    raw_posix = raw_str.replace("\\", "/")
    raw_path = Path(raw_str)
    if not raw_path.is_absolute() and not PurePosixPath(raw_posix).is_absolute():
        norm = PurePosixPath(os.path.normpath(raw_posix))
        if (
            norm.parts
            and norm.parts[0] != ".."
            and not norm.is_absolute()
            and norm != PurePosixPath(".")
        ):
            return Path(*norm.parts)

    resolved_abs = resolved.resolve()
    if root_dir is not None:
        try:
            return resolved_abs.relative_to(root_dir.resolve())
        except ValueError:
            pass
    try:
        return resolved_abs.relative_to(manifest_dir.resolve())
    except ValueError:
        pass
    return Path(resolved_abs.name)


def _make_output_path(
    output_dir: Path,
    rel_path: Path,
    suffix: str,
    save_format: str,
    ordinal: int | None = None,
) -> Path:
    stem = rel_path.stem
    if ordinal is not None:
        filename = f"{stem}_{ordinal}{suffix}.{save_format}"
    else:
        filename = f"{stem}{suffix}.{save_format}"
    return output_dir / rel_path.parent / filename


def _assert_within_output_dir(
    out_path: Path,
    resolved_output_dir: Path,
    display_output_dir: Path,
) -> None:
    resolved_target = out_path.resolve()
    if resolved_output_dir not in resolved_target.parents:
        raise ValueError(
            f"Refusing to write output path {out_path} outside output_dir {display_output_dir}."
        )


def collect_input_items(
    input_spec: str | Path | None,
    output_dir: str | Path,
    input_root: str | Path | None = None,
    recursive: bool = False,
    suffix: str = "",
    save_format: str | SaveFormat = "png",
) -> list[InferItem]:
    """Collect, validate, disambiguate, and lexicographically sort inference items."""
    if not isinstance(recursive, bool):
        raise ValueError(f"infer.recursive must be a bool, got {recursive!r}.")

    norm_format = normalize_save_format(save_format)
    norm_suffix = normalize_suffix(suffix)
    output_dir_path = _require_non_empty_path(output_dir, "infer.output_dir")
    input_path = _require_non_empty_path(input_spec, "infer.input")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    raw_entries: list[tuple[Path, Path]] = []

    if input_path.is_dir():
        iterator = input_path.rglob("*") if recursive else input_path.iterdir()
        for candidate in iterator:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                raw_entries.append((candidate, candidate.relative_to(input_path)))
        if not raw_entries:
            raise ValueError(
                f"No supported images ({sorted(IMAGE_EXTENSIONS)}) found in directory "
                f"{input_path} (recursive={recursive})."
            )
    elif input_path.is_file() and input_path.suffix.lower() == ".json":
        try:
            raw_text = input_path.read_text(encoding="utf-8")
            records = json.loads(raw_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to parse JSON manifest {input_path}: {exc}") from exc

        if not isinstance(records, list) or len(records) == 0:
            raise ValueError(
                f"JSON manifest {input_path} must be a non-empty list of dicts, "
                f"got {type(records).__name__} (len={len(records) if isinstance(records, list) else 'N/A'})."
            )

        root_dir = (
            _require_non_empty_path(input_root, "infer.input_root")
            if input_root is not None
            else None
        )
        if root_dir is not None and not root_dir.is_dir():
            raise FileNotFoundError(f"infer.input_root directory not found: {root_dir}")

        manifest_dir = input_path.parent
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Manifest item {idx} in {input_path} must be a dict, "
                    f"got {type(record).__name__}."
                )
            raw_str = _extract_manifest_image_path(record, idx)
            raw_p = Path(raw_str)
            if raw_p.is_absolute():
                resolved = raw_p
                if not resolved.is_file():
                    raise FileNotFoundError(f"Manifest item {idx} image not found: {resolved}")
            elif root_dir is not None:
                resolved = root_dir / raw_p
                if not resolved.is_file():
                    raise FileNotFoundError(f"Manifest item {idx} image not found: {resolved}")
            else:
                cand_manifest = manifest_dir / raw_p
                cand_cwd = Path.cwd() / raw_p
                if cand_manifest.is_file():
                    resolved = cand_manifest
                elif cand_cwd.is_file():
                    resolved = cand_cwd
                else:
                    raise FileNotFoundError(
                        f"Manifest item {idx} image not found "
                        f"(checked {cand_manifest} and {cand_cwd})."
                    )

            if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Manifest item {idx} has unsupported image extension {resolved.suffix!r} "
                    f"({resolved}); expected one of {sorted(IMAGE_EXTENSIONS)}."
                )

            rel_path = _derive_safe_manifest_rel_path(
                raw_str=raw_str,
                resolved=resolved,
                root_dir=root_dir,
                manifest_dir=manifest_dir,
            )
            raw_entries.append((resolved, rel_path))
    elif input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported input file extension {input_path.suffix!r} for {input_path}; "
                f"expected one of {sorted(IMAGE_EXTENSIONS)} or '.json'."
            )
        raw_entries.append((input_path, Path(input_path.name)))
    else:
        raise ValueError(
            f"Unsupported input path (neither a regular file nor directory): {input_path}"
        )

    # Step 1: Deterministic sort & commonpath disambiguation for colliding basenames
    raw_entries.sort(key=lambda pair: (pair[1].as_posix(), pair[0].resolve().as_posix()))
    initial_groups: dict[Path, list[tuple[Path, Path]]] = {}
    for inp, rel in raw_entries:
        target = _make_output_path(output_dir_path, rel, norm_suffix, norm_format)
        initial_groups.setdefault(target.resolve(), []).append((inp, rel))

    adjusted: list[tuple[Path, Path]] = []
    for group in initial_groups.values():
        if len(group) > 1:
            resolved_parents = [inp.resolve().parent for inp, _ in group]
            if len(set(resolved_parents)) > 1:
                try:
                    common = Path(os.path.commonpath([str(p) for p in resolved_parents]))
                except ValueError:
                    common = None
                if common is not None:
                    for inp, _ in group:
                        adjusted.append((inp, inp.resolve().relative_to(common)))
                    continue
        adjusted.extend(group)

    # Step 2: Re-group and apply deterministic ordinal fallback for any remaining collisions
    adjusted.sort(key=lambda pair: (pair[1].as_posix(), pair[0].resolve().as_posix()))
    final_groups: dict[Path, list[tuple[Path, Path]]] = {}
    for inp, rel in adjusted:
        target = _make_output_path(output_dir_path, rel, norm_suffix, norm_format)
        final_groups.setdefault(target.resolve(), []).append((inp, rel))

    used_targets: set[Path] = {
        target_key for target_key, grp in final_groups.items() if len(grp) == 1
    }
    resolved_output_dir = output_dir_path.resolve()
    items: list[InferItem] = []

    for grp in final_groups.values():
        if len(grp) == 1:
            inp, rel = grp[0]
            out_path = _make_output_path(output_dir_path, rel, norm_suffix, norm_format)
            _assert_within_output_dir(out_path, resolved_output_dir, output_dir_path)
            items.append(InferItem(input_path=inp, rel_path=rel, output_path=out_path))
        else:
            sorted_grp = sorted(
                grp, key=lambda pair: (pair[0].resolve().as_posix(), pair[1].as_posix())
            )
            for inp, rel in sorted_grp:
                ordinal = 1
                while True:
                    cand = _make_output_path(
                        output_dir_path,
                        rel,
                        norm_suffix,
                        norm_format,
                        ordinal=ordinal,
                    )
                    cand_resolved = cand.resolve()
                    if cand_resolved not in used_targets:
                        used_targets.add(cand_resolved)
                        _assert_within_output_dir(cand, resolved_output_dir, output_dir_path)
                        items.append(InferItem(input_path=inp, rel_path=rel, output_path=cand))
                        break
                    ordinal += 1

    items.sort(
        key=lambda item: (
            item.rel_path.as_posix(),
            item.output_path.as_posix(),
            item.input_path.resolve().as_posix(),
        )
    )
    return items


def _reenable_srtrainer_loggers() -> None:
    """Re-enable existing ``srtrainer`` loggers disabled by Hydra ``job_logging: disabled``."""
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if (name == "srtrainer" or name.startswith("srtrainer.")) and isinstance(
            obj, logging.Logger
        ):
            obj.disabled = False


def _build_infer_model(
    resolved: ResolvedInferConfig,
    accelerator: Accelerator,
) -> nn.Module:
    if resolved.component_weights is not None:
        model_cfg = resolved.model_cfg or VOSRModelConfig(scale=resolved.scale)
        dit_lora_active = should_attach_dit_lora(model_cfg)
        comp_modules: dict[str, nn.Module] = {}
        comp_keys = resolved.component_weights_keys or {}
        for comp in VOSR_COMPONENTS:
            if comp not in resolved.component_weights:
                continue
            weight_path = resolved.component_weights[comp]
            comp_cfg = resolved.arch_cfg[comp]
            inst_cfg = (
                _clone_arch_cfg_skip_pretrained(comp_cfg)
                if weight_path is not None
                else comp_cfg
            )
            with accelerator.main_process_first():
                comp_mod = hydra.utils.instantiate(inst_cfg)
            if not isinstance(comp_mod, nn.Module):
                raise TypeError(
                    f"Instantiated arch.{comp} must be torch.nn.Module, got {type(comp_mod).__name__}."
                )
            if comp == "dit" and dit_lora_active:
                comp_mod = _apply_dit_lora(comp_mod, model_cfg.lora)
            if weight_path is not None:
                if comp == "dit" and dit_lora_active:
                    load_weights(
                        comp_mod,
                        weight_path,
                        strict=True,
                        key=comp_keys.get(comp),
                    )
                else:
                    _load_component_weights(
                        comp_mod,
                        weight_path,
                        strict=True,
                        key=comp_keys.get(comp),
                    )
            elif comp == "dit" and dit_lora_active and model_cfg.lora.checkpoint is not None:
                load_weights(
                    comp_mod,
                    Path(model_cfg.lora.checkpoint),
                    strict=False,
                )
            comp_modules[comp] = comp_mod

        model: nn.Module = VOSRPipeline.from_model_config(
            vae=comp_modules["vae"],
            dit=comp_modules["dit"],
            dino=comp_modules.get("dino"),
            model_cfg=model_cfg,
            scale=resolved.scale,
            trainable=None,
        )
        model.to(accelerator.device)
        model.eval()
        return model

    with accelerator.main_process_first():
        model = hydra.utils.instantiate(resolved.arch_cfg)
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Instantiated arch must be torch.nn.Module, got {type(model).__name__}."
        )
    load_weights(model, resolved.weights_path, strict=True, key=resolved.weights_key)
    model.to(accelerator.device)
    model.eval()
    return model


def run_infer(cfg: InferRootConfig | DictConfig | Mapping[str, Any]) -> InferSummary:
    """Validate config at entry and execute single-process or multi-process inference."""
    infer_root = load_infer_config(cfg)
    inf = infer_root.infer
    mixed_precision = inf.mixed_precision.value
    save_format = inf.save_format.value
    output_dir = Path(inf.output_dir)

    had_existing_state = bool(PartialState._shared_state)
    accelerator = Accelerator(mixed_precision=mixed_precision)
    try:
        logger = setup_logger(
            log_file=None,
            is_main_process=accelerator.is_main_process,
            name="srtrainer",
        )
        _reenable_srtrainer_loggers()

        resolved = resolve_infer_config(infer_root, logger=logger)
        items = collect_input_items(
            input_spec=inf.input,
            output_dir=output_dir,
            input_root=inf.input_root,
            recursive=inf.recursive,
            suffix=inf.suffix,
            save_format=save_format,
        )

        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        model = _build_infer_model(resolved, accelerator)

        inferencer = TiledInferencer.from_config(
            model=model,
            scale=resolved.scale,
            tile_cfg=inf.tile,
            gt_size=resolved.gt_size,
            mixed_precision=mixed_precision,
            device=accelerator.device,
        )

        weights_desc = (
            {k: str(v) if v is not None else "arch_pretrained" for k, v in resolved.component_weights.items()}
            if resolved.component_weights is not None
            else str(resolved.weights_path)
        )
        logger.info(
            "[Infer] mode=%s | scale=%d | weights=%s | total_images=%d | processes=%d | device=%s",
            resolved.mode,
            resolved.scale,
            weights_desc,
            len(items),
            accelerator.num_processes,
            accelerator.device,
        )

        start_time = time.perf_counter()
        local_count = 0
        with accelerator.split_between_processes(items, apply_padding=False) as local_items:
            for item in local_items:
                lq = load_image(item.input_path).unsqueeze(0)
                sr = inferencer.infer_tensor(lq)
                save_image(sr, item.output_path, save_format=save_format)
                local_count += 1

        accelerator.wait_for_everyone()
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        images_per_sec = len(items) / elapsed

        logger.info(
            "[Infer] Done: %d images in %.2fs (%.2f img/s) | output_dir=%s",
            len(items),
            elapsed,
            images_per_sec,
            output_dir,
        )

        return InferSummary(
            mode=resolved.mode,
            total_images=len(items),
            local_images=local_count,
            elapsed_sec=elapsed,
            images_per_sec=images_per_sec,
            output_dir=output_dir,
            weights_path=resolved.weights_path,
            component_weights=resolved.component_weights,
        )
    finally:
        accelerator.end_training()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if not had_existing_state:
            AcceleratorState._reset_state(reset_partial_state=True)
            GradientState._reset_state()


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="infer")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint for single-process and distributed inference."""
    run_infer(cfg)


if __name__ == "__main__":
    main()
