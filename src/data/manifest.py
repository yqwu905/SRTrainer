"""Unified list-of-dicts JSON manifest parsing and training source validation."""

from __future__ import annotations

import copy
import json
import logging
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from config.schema import DataSourceConfig
from config.validation import load_structured_config
from utils.text_geometry import canonical_quad

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
_IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


@dataclass(frozen=True)
class TrainSample:
    """Single training sample parsed from a unified list-of-dicts JSON manifest."""

    gt_path: Path
    lq_path: Path | None
    need_degradation: bool
    source_name: str = ""
    text_lines: tuple[dict[str, Any], ...] | None = None


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


def _normalize_glyph_entry(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context}: glyph entry must be a dict, got {type(raw).__name__}.")
    ch = raw.get("char")
    if not isinstance(ch, str) or len(ch) != 1 or ch.isspace():
        raise ValueError(f"{context}: each glyph needs a single non-whitespace 'char', got {ch!r}.")
    geom_keys = [k for k in ("quad", "bbox") if k in raw]
    if len(geom_keys) != 1:
        raise ValueError(f"{context}: glyph {ch!r} needs exactly one of 'quad' or 'bbox'.")
    pts = canonical_quad(raw[geom_keys[0]], name=f"{context} glyph {ch!r}")
    return {"char": ch, "quad": pts.tolist()}


def _normalize_text_lines(
    raw: Any,
    context: str,
    *,
    image_alignment_valid: bool = True,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(image_alignment_valid, bool):
        raise ValueError(
            f"{context}.alignment_valid must be a bool, got {type(image_alignment_valid).__name__}."
        )
    if not isinstance(raw, list):
        raise ValueError(f"{context}: text_lines must be a list, got {type(raw).__name__}.")
    normalized: list[dict[str, Any]] = []
    for idx, line in enumerate(raw):
        if not isinstance(line, Mapping):
            raise ValueError(
                f"{context}[{idx}]: line entry must be a dict, got {type(line).__name__}."
            )
        valid = line.get("valid", True)
        if not isinstance(valid, bool):
            raise ValueError(f"{context}[{idx}].valid must be a bool.")
        line_aligned = line.get("alignment_valid", True)
        if not isinstance(line_aligned, bool):
            raise ValueError(f"{context}[{idx}].alignment_valid must be a bool.")
        aligned = image_alignment_valid and line_aligned
        for flag in ("difficult", "ignore"):
            flag_val = line.get(flag, False)
            if not isinstance(flag_val, bool):
                raise ValueError(f"{context}[{idx}].{flag} must be a bool.")
            if flag_val:
                valid = False
        transcript = line.get("transcript", line.get("transcription", line.get("text", "")))
        if transcript in ("###", "*"):
            valid = False

        raw_glyphs = line.get("glyphs")
        if raw_glyphs is None:
            glyphs: list[dict[str, Any]] = []
        else:
            if not isinstance(raw_glyphs, list):
                raise ValueError(f"{context}[{idx}].glyphs must be a list.")
            glyphs = [
                _normalize_glyph_entry(g, f"{context}[{idx}].glyphs[{g_idx}]")
                for g_idx, g in enumerate(raw_glyphs)
            ]

        entry: dict[str, Any] = {
            "valid": valid,
            "alignment_valid": aligned,
            "glyphs": glyphs,
        }
        for geom_key in ("quad", "points", "bbox", "box"):
            if geom_key in line:
                entry["quad"] = canonical_quad(line[geom_key], name=f"{context}[{idx}]").tolist()
                break
        normalized.append(entry)
    return tuple(normalized)


def _normalize_flat_glyphs(raw: Any, context: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{context}: glyphs must be a list, got {type(raw).__name__}.")
    glyphs = [
        _normalize_glyph_entry(g, f"{context}[{g_idx}]") for g_idx, g in enumerate(raw)
    ]
    return ({"valid": True, "alignment_valid": True, "glyphs": glyphs},)


def _normalize_glyph_key(raw_key: str) -> str:
    """Normalize an image path/key using pure string operations.

    Converts backslashes to ``/``, collapses redundant separators and ``.`` segments,
    and preserves case, meaningful spaces, and ``..`` segments without touching the filesystem.
    """
    replaced = raw_key.replace("\\", "/")
    is_abs = replaced.startswith("/")
    parts = [part for part in replaced.split("/") if part not in ("", ".")]
    if is_abs:
        return "/" + "/".join(parts)
    if not parts:
        return "."
    return "/".join(parts)


def _glyph_key_aliases(norm_key: str) -> tuple[str, str]:
    """Return ``(basename, stem)`` aliases from an already-normalized glyph key."""
    basename = PurePosixPath(norm_key).name
    if not basename:
        return "", ""
    base_path = PurePosixPath(basename)
    stem = base_path.stem if base_path.suffix.lower() in _IMAGE_SUFFIXES else basename
    return basename, stem


@dataclass(frozen=True)
class _ExternalGlyphsIndex:
    """Structured external glyph index separating exact full keys from basename/stem aliases."""

    source_path: Path
    exact: dict[str, tuple[dict[str, Any], ...]]
    by_basename: dict[str, set[str]]
    by_stem: dict[str, set[str]]


def _load_external_glyphs_map(glyphs_json: str | Path) -> _ExternalGlyphsIndex:
    path = _validate_path_arg(glyphs_json, "glyphs_json")
    if not path.is_file():
        raise FileNotFoundError(f"Glyph annotations JSON file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse glyphs JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(
            f"Glyph annotations JSON {path} must map image names/stems to line lists, "
            f"got {type(raw).__name__}."
        )
    exact: dict[str, tuple[dict[str, Any], ...]] = {}
    raw_keys_by_norm: dict[str, str] = {}
    by_basename: dict[str, set[str]] = {}
    by_stem: dict[str, set[str]] = {}
    for raw_name, records in raw.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{path}: image keys must be non-empty strings.")
        norm_key = _normalize_glyph_key(raw_name)
        if norm_key in exact:
            prev_raw = raw_keys_by_norm[norm_key]
            raise ValueError(
                f"Duplicate normalized glyph key {norm_key!r} in {path}: "
                f"raw keys {sorted((prev_raw, raw_name))!r} collide after path normalization."
            )
        if isinstance(records, Mapping):
            image_aligned = records.get("alignment_valid", True)
            if not isinstance(image_aligned, bool):
                raise ValueError(
                    f"{path}:{raw_name}.alignment_valid must be a bool, "
                    f"got {type(image_aligned).__name__}."
                )
            line_list = records.get("lines")
        else:
            image_aligned = True
            line_list = records
        parsed = _normalize_text_lines(
            line_list,
            f"{path}:{raw_name}",
            image_alignment_valid=image_aligned,
        )
        exact[norm_key] = parsed
        raw_keys_by_norm[norm_key] = raw_name
        basename, stem = _glyph_key_aliases(norm_key)
        if basename:
            by_basename.setdefault(basename, set()).add(norm_key)
        if stem:
            by_stem.setdefault(stem, set()).add(norm_key)
    return _ExternalGlyphsIndex(
        source_path=path,
        exact=exact,
        by_basename=by_basename,
        by_stem=by_stem,
    )


def _lookup_external_glyphs(
    index: _ExternalGlyphsIndex,
    raw_gt: str,
    *,
    manifest_path: Path,
    sample_idx: int,
    source_name: str,
) -> tuple[dict[str, Any], ...]:
    norm_gt = _normalize_glyph_key(raw_gt)
    if norm_gt in index.exact:
        return index.exact[norm_gt]

    gt_basename, gt_stem = _glyph_key_aliases(norm_gt)
    if gt_basename and gt_basename in index.by_basename:
        matches = index.by_basename[gt_basename]
        if len(matches) == 1:
            return index.exact[next(iter(matches))]
        candidates = sorted(matches)
        raise ValueError(
            f"Ambiguous external glyph annotation match in {index.source_path} for "
            f"Train manifest {manifest_path}[{sample_idx}] (source={source_name!r}, "
            f"gt={raw_gt!r}) at basename level {gt_basename!r}: candidates={candidates!r}."
        )

    if gt_stem and gt_stem in index.by_stem:
        matches = index.by_stem[gt_stem]
        if len(matches) == 1:
            return index.exact[next(iter(matches))]
        candidates = sorted(matches)
        raise ValueError(
            f"Ambiguous external glyph annotation match in {index.source_path} for "
            f"Train manifest {manifest_path}[{sample_idx}] (source={source_name!r}, "
            f"gt={raw_gt!r}) at stem level {gt_stem!r}: candidates={candidates!r}."
        )

    return ()


def load_train_manifest(
    json_path: str | Path,
    root_dir: str | Path,
    source_name: str = "",
    glyphs_json: str | Path | None = None,
) -> tuple[list[TrainSample], SourceStats]:
    """Parse a unified training JSON manifest with per-sample ``gt_only`` vs ``paired`` detection."""
    path, entries = _load_json_list(json_path)
    root = _validate_path_arg(root_dir, "root_dir")

    if not isinstance(source_name, str):
        raise ValueError(f"source_name must be a string, got {type(source_name).__name__}.")
    resolved_name = source_name.strip() if source_name.strip() else path.stem

    external_glyphs = (
        _load_external_glyphs_map(glyphs_json) if glyphs_json is not None else None
    )

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

        sample_lines: tuple[dict[str, Any], ...] | None = None
        if "text_lines" in entry and entry["text_lines"] is not None:
            sample_lines = _normalize_text_lines(
                entry["text_lines"], f"Train manifest {path}[{idx}].text_lines"
            )
        elif "glyphs" in entry and entry["glyphs"] is not None:
            sample_lines = _normalize_flat_glyphs(
                entry["glyphs"], f"Train manifest {path}[{idx}].glyphs"
            )
        elif external_glyphs is not None:
            sample_lines = _lookup_external_glyphs(
                external_glyphs,
                raw_gt,
                manifest_path=path,
                sample_idx=idx,
                source_name=resolved_name,
            )

        samples.append(
            TrainSample(
                gt_path=gt_path,
                lq_path=lq_path,
                need_degradation=need_degradation,
                source_name=resolved_name,
                text_lines=sample_lines,
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
    sources: Sequence[DataSourceConfig | Mapping[str, Any]] | ListConfig,
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
            f"data.sources must be a non-empty sequence of source entries, "
            f"got {type(normalized_sources).__name__}."
        )
    if len(normalized_sources) == 0:
        raise ValueError("data.sources must be a non-empty list.")

    parsed_configs: list[tuple[str, Path, Path, float, Path | None]] = []
    ratios: list[float] = []

    for idx, raw_src in enumerate(normalized_sources):
        if isinstance(raw_src, DataSourceConfig):
            src_cfg = raw_src
        else:
            src_map = _normalize_omegaconf(raw_src, f"data.sources[{idx}]")
            if not isinstance(src_map, Mapping):
                raise ValueError(
                    f"data.sources[{idx}] must be a DataSourceConfig or mapping, "
                    f"got {type(src_map).__name__}."
                )
            if "mode" in src_map:
                raise ValueError(
                    f"Legacy 'mode' key is not allowed in data.sources[{idx}] "
                    f"(name={src_map.get('name')!r}). Remove 'mode'; gt_only vs paired is "
                    "auto-detected per sample from JSON."
                )
            missing = [k for k in _REQUIRED_SOURCE_KEYS if k not in src_map]
            if missing:
                raise ValueError(
                    f"data.sources[{idx}] is missing required key(s): {', '.join(missing)}."
                )
            raw_ratio_pre = src_map["ratio"]
            if isinstance(raw_ratio_pre, bool) or not isinstance(raw_ratio_pre, numbers.Real):
                raise ValueError(
                    f"data.sources[{idx}]['ratio'] must be a finite real number >= 0, "
                    f"got {raw_ratio_pre!r}."
                )
            # Coerce Path fields before OmegaConf structured merge if caller passed Path objects.
            coerced = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in src_map.items()
            }
            src_cfg = load_structured_config(
                DataSourceConfig, coerced, context=f"data.sources[{idx}]"
            )

        if not is_valid_rel_path(src_cfg.name):
            raise ValueError(
                f"data.sources[{idx}]['name'] must be a non-empty string, got {src_cfg.name!r}."
            )
        src_name = src_cfg.name.strip()
        src_root_dir = _validate_path_arg(src_cfg.root_dir, f"data.sources[{idx}]['root_dir']")
        src_json_path = _validate_path_arg(src_cfg.json_path, f"data.sources[{idx}]['json_path']")
        src_glyphs_json = (
            _validate_path_arg(src_cfg.glyphs_json, f"data.sources[{idx}]['glyphs_json']")
            if src_cfg.glyphs_json is not None
            else None
        )
        ratio_f = float(src_cfg.ratio)
        if not math.isfinite(ratio_f) or ratio_f < 0.0:
            raise ValueError(
                f"data.sources[{idx}]['ratio'] must be finite and >= 0, got {src_cfg.ratio!r}."
            )

        parsed_configs.append((src_name, src_root_dir, src_json_path, ratio_f, src_glyphs_json))
        ratios.append(ratio_f)

    if max(ratios) <= 0.0:
        raise ValueError(
            f"Sum of data.sources ratios must be finite and > 0, got {ratios!r}."
        )

    all_samples: list[list[TrainSample]] = []
    all_stats: list[SourceStats] = []
    for src_name, src_root_dir, src_json_path, ratio_f, src_glyphs_json in parsed_configs:
        samples, stats = load_train_manifest(
            json_path=src_json_path,
            root_dir=src_root_dir,
            source_name=src_name,
            glyphs_json=src_glyphs_json,
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
