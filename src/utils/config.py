"""Configuration helpers: mixed precision normalization, work_dir broadcast, resolved config I/O, tracker flattening."""

from __future__ import annotations

import enum
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from config.schema import InferRootConfig, MixedPrecision, TrainConfig


def normalize_mixed_precision(value: Any) -> str:
    """Normalize YAML/CLI mixed precision values to Accelerate's ``'no' | 'fp16' | 'bf16'``."""
    if isinstance(value, MixedPrecision):
        return value.value
    if value is None:
        return "no"
    if isinstance(value, bool):
        if value is False:
            return "no"
        raise ValueError(
            "Ambiguous mixed_precision=True; explicitly specify 'no', 'fp16', or 'bf16'."
        )
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"", "no", "none", "false", "fp32", "off"}:
            return "no"
        if norm in {"fp16", "bf16"}:
            return norm
        raise ValueError(
            f"Unsupported mixed_precision={value!r}; expected 'no', 'fp16', or 'bf16'."
        )
    raise ValueError(
        f"Unsupported mixed_precision type {type(value).__name__}: {value!r}; "
        "expected 'no', 'fp16', or 'bf16'."
    )


def _as_dict_config(cfg: TrainConfig | InferRootConfig | DictConfig | Mapping[str, Any]) -> DictConfig:
    if isinstance(cfg, DictConfig):
        return cfg
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return OmegaConf.structured(cfg)
    if isinstance(cfg, Mapping):
        created = OmegaConf.create(dict(cfg))
        if isinstance(created, DictConfig):
            return created
    raise TypeError(f"Expected TrainConfig, DictConfig, or Mapping, got {type(cfg).__name__}.")


def _parse_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _is_distributed_env() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return True
    if _parse_int_env("LOCAL_RANK", -1) != -1:
        return True
    for key in ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"):
        if _parse_int_env(key, 1) > 1:
            return True
    if os.environ.get("ACCELERATE_USE_SAGEMAKER", "false").lower() == "true":
        return True
    return False


def _should_use_cpu_for_partial_state() -> bool:
    from accelerate.utils.environment import parse_flag_from_env

    if parse_flag_from_env("ACCELERATE_USE_CPU", False):
        return True
    try:
        from accelerate.utils.imports import get_torch_accelerator_backend

        device_type, _ = get_torch_accelerator_backend()
        return device_type is None
    except Exception:
        return not torch.cuda.is_available()


def _with_timestamp(base: Path) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if base.name:
        return str(base.parent / f"{base.name}_{ts}")
    return f"{str(base).rstrip('/')}_{ts}"


def resolve_work_dir(cfg: TrainConfig | DictConfig | Mapping[str, Any]) -> Path:
    """Resolve experiment working directory with multi-rank safe timestamp broadcast.

    - When ``trainer.append_timestamp`` is False or ``trainer.resume`` is set, returns
      ``Path(cfg.work_dir)`` directly without touching Accelerate state.
    - When ``trainer.append_timestamp`` is True and not resuming:
      - In single-process runs before ``Accelerator`` exists, appends the timestamp without
        initializing ``PartialState`` (so subsequent ``Accelerator(cpu=..., mixed_precision=...)``
        remains completely unpoisoned).
      - In distributed runs (or when ``PartialState`` is already initialized), initializes
        ``PartialState(cpu=...)`` appropriately and broadcasts Rank 0's timestamped path via
        ``accelerate.utils.broadcast_object_list``.
    """
    if isinstance(cfg, TrainConfig):
        raw_work_dir: Any = cfg.work_dir
        append_timestamp = bool(cfg.trainer.append_timestamp)
        resume: Any = cfg.trainer.resume
    else:
        cfg_oc = _as_dict_config(cfg)
        raw_work_dir = OmegaConf.select(cfg_oc, "work_dir", default=None)
        append_timestamp = bool(OmegaConf.select(cfg_oc, "trainer.append_timestamp", default=False))
        resume = OmegaConf.select(cfg_oc, "trainer.resume", default=None)

    if raw_work_dir is None or not str(raw_work_dir).strip():
        raise ValueError("cfg.work_dir must be a non-empty path.")

    base_dir = Path(str(raw_work_dir))
    is_resuming = resume is not None and str(resume).strip().lower() not in {
        "",
        "none",
        "null",
        "false",
    }

    if not append_timestamp or is_resuming:
        return base_dir

    from accelerate.state import PartialState
    from accelerate.utils import broadcast_object_list

    already_initialized = bool(PartialState._shared_state)
    if not already_initialized and not _is_distributed_env():
        return Path(_with_timestamp(base_dir))

    state = (
        PartialState()
        if already_initialized
        else PartialState(cpu=_should_use_cpu_for_partial_state())
    )
    obj_list: list[str | None] = [_with_timestamp(base_dir) if state.is_main_process else None]
    broadcast_object_list(obj_list, from_process=0)
    if obj_list[0] is None:
        raise RuntimeError("Failed to broadcast resolved work_dir from main process.")
    return Path(obj_list[0])


def setup_experiment_dirs(work_dir: Path, is_main_process: bool) -> dict[str, Path]:
    """Return canonical experiment directory paths and create them on Rank 0."""
    root = Path(work_dir)
    dirs: dict[str, Path] = {
        "work_dir": root,
        "trackers": root / "trackers",
        "checkpoints": root / "checkpoints",
        "weights": root / "weights",
        "visuals": root / "visuals",
    }
    if is_main_process:
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_resolved_config(
    cfg: TrainConfig | InferRootConfig | DictConfig | Mapping[str, Any],
    path: Path,
) -> None:
    """Atomically save a fully resolved OmegaConf config to YAML."""
    cfg_oc = _as_dict_config(cfg)
    yaml_text = OmegaConf.to_yaml(cfg_oc, resolve=True)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(yaml_text)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_resolved_config(path: Path) -> DictConfig:
    """Load a persisted ``config_resolved.yaml`` into an OmegaConf ``DictConfig``."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Resolved config not found: {cfg_path}")

    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
        raw = yaml.safe_load(raw_text)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to parse YAML config {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected YAML mapping (dict) in {cfg_path}, got {type(raw).__name__}."
        )

    try:
        loaded = OmegaConf.create(raw)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid OmegaConf mapping in {cfg_path}: {exc}") from exc

    if not isinstance(loaded, DictConfig):
        raise ValueError(
            f"Expected YAML mapping (DictConfig) in {cfg_path}, got {type(loaded).__name__}."
        )
    return loaded


def _unwrap_scalar(obj: Any) -> Any:
    if getattr(type(obj), "__module__", "").startswith("numpy"):
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            if obj.ndim == 0 or obj.size == 1:
                return obj.item()
            return obj.tolist()
    return obj


def _normalize_list_item(item: Any) -> Any:
    item = _unwrap_scalar(item)
    if isinstance(item, enum.Enum):
        return str(item.name)
    if isinstance(item, (list, tuple)):
        return [_normalize_list_item(sub) for sub in item]
    return item


def _json_default(obj: Any) -> Any:
    unwrapped = _unwrap_scalar(obj)
    if unwrapped is not obj:
        return unwrapped
    if isinstance(obj, enum.Enum):
        return str(obj.name)
    return str(obj)


def _flatten_into(
    obj: Any,
    prefix: str,
    out: dict[str, int | float | str | bool],
) -> None:
    obj = _unwrap_scalar(obj)

    if isinstance(obj, Mapping):
        if len(obj) == 0:
            if prefix:
                out[prefix] = "{}"
            return
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_into(v, key, out)
        return

    if isinstance(obj, (list, tuple)):
        if any(isinstance(item, Mapping) for item in obj):
            for idx, item in enumerate(obj):
                key = f"{prefix}.{idx}" if prefix else str(idx)
                _flatten_into(item, key, out)
        else:
            out[prefix] = json.dumps(
                [_normalize_list_item(item) for item in obj],
                ensure_ascii=False,
                default=_json_default,
            )
        return

    if obj is None:
        out[prefix] = "null"
        return

    if isinstance(obj, enum.Enum):
        out[prefix] = str(obj.name)
        return

    if isinstance(obj, bool):
        out[prefix] = bool(obj)
        return

    if isinstance(obj, int):
        out[prefix] = int(obj)
        return

    if isinstance(obj, float):
        out[prefix] = float(obj)
        return

    if isinstance(obj, str):
        out[prefix] = str(obj)
        return

    out[prefix] = str(obj)


def flatten_config_for_tracker(
    cfg: TrainConfig | InferRootConfig | DictConfig | Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Flatten a resolved config into a tracker-safe flat dict of ``bool | int | float | str``.

    Safe for PyTorch TensorBoard ``SummaryWriter.add_hparams`` and ``yaml.dump`` (no ``None``,
    no nested dicts/lists, no OmegaConf container nodes).
    """
    if is_dataclass(cfg) and not isinstance(cfg, type):
        container: Any = OmegaConf.to_container(
            OmegaConf.structured(cfg), resolve=True, enum_to_str=True
        )
    elif OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    elif isinstance(cfg, Mapping):
        container = dict(cfg)
    else:
        raise TypeError(f"Expected TrainConfig, DictConfig, or Mapping, got {type(cfg).__name__}.")

    if not isinstance(container, Mapping):
        raise ValueError(
            f"Expected mapping root config, got {type(container).__name__}."
        )

    flat: dict[str, int | float | str | bool] = {}
    _flatten_into(container, "", flat)
    return flat
