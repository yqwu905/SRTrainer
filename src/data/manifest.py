"""Unified list-of-dicts JSON manifest parsing and training source validation."""

from __future__ import annotations

import copy
import json
import logging
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

__all__ = [
    "TrainSample",
    "ValSample",
    "SourceStats",
    "is_valid_rel_path",
    "load_train_manifest",
    "load_val_manifest",
    "validate_sources_config",
]

_REQUIRED_SOURCE_KEYS: tuple[str, ...] = ("name", "root_dir", "json_path", "ratio")


@dataclass(frozen=True)
class TrainSample:
    """Single training sample parsed from a unified list-of-dicts JSON manifest."""

    gt_path: Path
    lq_path: Path | None
    need_degradation: bool
    source_name: str = ""


@dataclass(frozen=True)
class ValSample:
    """Single validation sample parsed from a unified list-of-dicts JSON manifest."""

    lq_path: Path
    gt_path: Path | None
    meta: dict[str, Any]


@dataclass(frozen=True)
class SourceStats:
    """Per-source auto-detection statistics for a training manifest."""

    name: str
    total: int
    gt_only: int
    paired: int
    ratio: float = 1.0

    def format_summary(self) -> str:
        """Format the Rank-0 per-source auto-detection summary line."""
        return (
            f"[Data] {self.name}: {self.total} samples "
            f"(gt_only: {self.gt_only}, paired: {self.paired})"
        )


def is_valid_rel_path(val: Any) -> bool:
    """Return True iff ``val`` is a non-empty, non-whitespace string."""
    return isinstance(val, str) and len(val.strip()) > 0


def _normalize_omegaconf(obj: Any, context: str) -> Any:
    """Convert OmegaConf containers to plain Python objects, rejecting unresolved missing values."""
    if OmegaConf.is_config(obj):
        try:
            return OmegaConf.to_container(obj, resolve=True, throw_on_missing=True)
        except OmegaConfBaseException as exc:
            raise ValueError(f"Invalid OmegaConf configuration for {context}: {exc}") from exc
    return obj


def _validate_path_arg(val: Any, name: str) -> Path:
    """Validate a path argument accepting ``Path`` or non-empty, non-whitespace ``str``."""
    if isinstance(val, Path):
        return val
    if is_valid_rel_path(val):
        return Path(val)
    raise ValueError(f"{name} must be a Path or non-empty string, got {val!r}.")


def _load_json_list(json_path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Read and validate a non-empty list-of-dicts JSON manifest."""
    path = _validate_path_arg(json_path, "json_path")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest JSON file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
        raw = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse JSON manifest {path}: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(
            f"Manifest {path} must be a JSON list of dicts, got {type(raw).__name__}."
        )
    if len(raw) == 0:
        raise ValueError(f"Manifest {path} must not be empty.")

    entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Manifest {path}[{idx}] must be a JSON object (dict), got {type(entry).__name__}."
            )
        entries.append(entry)

    return path, entries


def load_train_manifest(
    json_path: str | Path,
    root_dir: str | Path,
    source_name: str = "",
) -> tuple[list[TrainSample], SourceStats]:
    """Parse a unified training JSON manifest with per-sample ``gt_only`` vs ``paired`` detection."""
    path, entries = _load_json_list(json_path)
    root = _validate_path_arg(root_dir, "root_dir")

    if not isinstance(source_name, str):
        raise ValueError(f"source_name must be a string, got {type(source_name).__name__}.")
    resolved_name = source_name.strip() if source_name.strip() else path.stem

    samples: list[TrainSample] = []
    gt_only_count = 0
    paired_count = 0

    for idx, entry in enumerate(entries):
        raw_gt = entry.get("gt")
        if not is_valid_rel_path(raw_gt):
            raise ValueError(
                f"Train manifest {path}[{idx}] (source={resolved_name!r}) is missing a valid "
                f"non-empty 'gt' path, got {raw_gt!r}."
            )
        gt_path = root / raw_gt

        raw_lq = entry.get("lq")
        if raw_lq is None:
            lq_path: Path | None = None
            need_degradation = True
        elif isinstance(raw_lq, str):
            if is_valid_rel_path(raw_lq):
                lq_path = root / raw_lq
                need_degradation = False
            else:
                lq_path = None
                need_degradation = True
        else:
            raise ValueError(
                f"Train manifest {path}[{idx}] (source={resolved_name!r}) has invalid 'lq' field: "
                f"expected string or null, got {type(raw_lq).__name__}."
            )

        if need_degradation:
            gt_only_count += 1
        else:
            paired_count += 1

        samples.append(
            TrainSample(
                gt_path=gt_path,
                lq_path=lq_path,
                need_degradation=need_degradation,
                source_name=resolved_name,
            )
        )

    stats = SourceStats(
        name=resolved_name,
        total=len(samples),
        gt_only=gt_only_count,
        paired=paired_count,
    )
    return samples, stats


def validate_sources_config(
    sources: Sequence[Mapping[str, Any]] | ListConfig,
    has_degradation: bool,
    is_main_process: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[list[list[TrainSample]], list[SourceStats]]:
    """Validate training ``data.sources``, parse all manifests, log Rank-0 stats, and fail-fast."""
    if not isinstance(has_degradation, bool):
        raise TypeError(
            f"has_degradation must be a bool, got {type(has_degradation).__name__}."
        )
    normalized_sources = _normalize_omegaconf(sources, "data.sources")
    if isinstance(normalized_sources, (str, bytes, Mapping)) or not isinstance(
        normalized_sources, Sequence
    ):
        raise ValueError(
            f"data.sources must be a non-empty sequence of source mappings, "
            f"got {type(normalized_sources).__name__}."
        )
    if len(normalized_sources) == 0:
        raise ValueError("data.sources must be a non-empty list.")

    parsed_configs: list[tuple[str, Path, Path, float]] = []
    ratios: list[float] = []

    for idx, raw_src in enumerate(normalized_sources):
        src = _normalize_omegaconf(raw_src, f"data.sources[{idx}]")
        if not isinstance(src, Mapping):
            raise ValueError(
                f"data.sources[{idx}] must be a mapping, got {type(src).__name__}."
            )

        # Explicitly reject legacy 'mode' key before any other validation.
        if "mode" in src:
            raise ValueError(
                f"Legacy 'mode' key is not allowed in data.sources[{idx}] "
                f"(name={src.get('name')!r}). Remove 'mode'; gt_only vs paired is "
                "auto-detected per sample from JSON."
            )

        missing = [k for k in _REQUIRED_SOURCE_KEYS if k not in src]
        if missing:
            raise ValueError(
                f"data.sources[{idx}] is missing required key(s): {', '.join(missing)}."
            )

        raw_name = src["name"]
        if not is_valid_rel_path(raw_name):
            raise ValueError(
                f"data.sources[{idx}]['name'] must be a non-empty string, got {raw_name!r}."
            )
        src_name = raw_name.strip()

        src_root_dir = _validate_path_arg(src["root_dir"], f"data.sources[{idx}]['root_dir']")
        src_json_path = _validate_path_arg(src["json_path"], f"data.sources[{idx}]['json_path']")

        raw_ratio = src["ratio"]
        if isinstance(raw_ratio, bool) or not isinstance(raw_ratio, numbers.Real):
            raise ValueError(
                f"data.sources[{idx}]['ratio'] must be a finite real number >= 0, "
                f"got {raw_ratio!r}."
            )
        ratio_f = float(raw_ratio)
        if not math.isfinite(ratio_f) or ratio_f < 0.0:
            raise ValueError(
                f"data.sources[{idx}]['ratio'] must be finite and >= 0, got {raw_ratio!r}."
            )

        parsed_configs.append((src_name, src_root_dir, src_json_path, ratio_f))
        ratios.append(ratio_f)

    if max(ratios) <= 0.0:
        raise ValueError(
            f"Sum of data.sources ratios must be finite and > 0, got {ratios!r}."
        )

    all_samples: list[list[TrainSample]] = []
    all_stats: list[SourceStats] = []
    for src_name, src_root_dir, src_json_path, ratio_f in parsed_configs:
        samples, stats = load_train_manifest(
            json_path=src_json_path,
            root_dir=src_root_dir,
            source_name=src_name,
        )
        all_samples.append(samples)
        all_stats.append(
            SourceStats(
                name=stats.name,
                total=stats.total,
                gt_only=stats.gt_only,
                paired=stats.paired,
                ratio=ratio_f,
            )
        )

    if is_main_process:
        active_logger = logger if logger is not None else logging.getLogger("srtrainer")
        for st in all_stats:
            active_logger.info(st.format_summary())

    gt_only_sources = [st for st in all_stats if st.gt_only > 0]
    if not has_degradation and gt_only_sources:
        details = ", ".join(f"{st.name} ({st.gt_only} gt_only)" for st in gt_only_sources)
        raise ValueError(
            f"Found gt_only samples in [{details}], but degradation is not configured "
            "(has_degradation=False). Configure data.degradation or supply paired 'lq' "
            "paths for all samples."
        )

    return all_samples, all_stats


def _resolve_val_path(raw_path: str, base_dir: Path | None, json_parent: Path) -> Path:
    """Resolve a validation image path relative to ``base_dir`` or the manifest directory."""
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if base_dir is not None:
        return base_dir / p
    candidate = json_parent / p
    if candidate.exists() or not p.exists():
        return candidate
    return p


def load_val_manifest(
    json_path: str | Path,
    root_dir: str | Path | None = None,
) -> list[ValSample]:
    """Parse a validation JSON manifest with required ``lq``, optional ``gt``, and full ``meta``."""
    path, entries = _load_json_list(json_path)

    if root_dir is None:
        base_dir: Path | None = None
    elif isinstance(root_dir, Path):
        base_dir = root_dir
    elif isinstance(root_dir, str):
        base_dir = Path(root_dir) if len(root_dir.strip()) > 0 else None
    else:
        raise ValueError(
            f"root_dir must be a Path, string, or None, got {type(root_dir).__name__}."
        )

    samples: list[ValSample] = []
    for idx, entry in enumerate(entries):
        raw_lq = entry.get("lq")
        if not is_valid_rel_path(raw_lq):
            raise ValueError(
                f"Validation manifest {path}[{idx}] is missing a valid non-empty 'lq' path, "
                f"got {raw_lq!r}."
            )
        lq_path = _resolve_val_path(raw_lq, base_dir, path.parent)

        raw_gt = entry.get("gt")
        if raw_gt is None:
            gt_path: Path | None = None
        elif isinstance(raw_gt, str):
            gt_path = (
                _resolve_val_path(raw_gt, base_dir, path.parent)
                if is_valid_rel_path(raw_gt)
                else None
            )
        else:
            raise ValueError(
                f"Validation manifest {path}[{idx}] has invalid 'gt' field: "
                f"expected string or null, got {type(raw_gt).__name__}."
            )

        meta: dict[str, Any] = {
            k: copy.deepcopy(v) for k, v in entry.items() if k not in ("lq", "gt")
        }
        meta["lq_path"] = str(lq_path)
        meta["gt_path"] = str(gt_path) if gt_path is not None else None
        meta["index"] = idx

        samples.append(ValSample(lq_path=lq_path, gt_path=gt_path, meta=meta))

    return samples
