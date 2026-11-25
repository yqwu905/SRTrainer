"""Shared RapidOCR 3.x / ONNX Runtime helpers for metrics and offline text detection."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_DET_MODEL",
    "DEFAULT_REC_MODEL",
    "DEFAULT_CLS_MODEL",
    "DET_MODELS",
    "REC_MODELS",
    "CLS_MODELS",
    "import_rapidocr",
    "get_rapidocr_model_root",
    "resolve_onnx_model",
    "onnx_device_params",
    "strip_accelerator_device_params",
    "ensure_rapidocr_models_ready",
    "create_rapidocr_engine",
]

DEFAULT_DET_MODEL: str = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL: str = "PP-OCRv5_server_rec"
DEFAULT_CLS_MODEL: str = "PP-LCNet_x1_0_textline_ori"

DET_MODELS: dict[str, tuple[str, str]] = {
    "PP-OCRv5_server_det": ("PP-OCRv5", "server"),
    "PP-OCRv5_mobile_det": ("PP-OCRv5", "mobile"),
    "PP-OCRv4_server_det": ("PP-OCRv4", "server"),
    "PP-OCRv4_mobile_det": ("PP-OCRv4", "mobile"),
}
REC_MODELS: dict[str, tuple[str, str]] = {
    "PP-OCRv4_server_rec": ("PP-OCRv4", "server"),
    "PP-OCRv4_mobile_rec": ("PP-OCRv4", "mobile"),
    DEFAULT_REC_MODEL: ("PP-OCRv5", "server"),
    "PP-OCRv5_mobile_rec": ("PP-OCRv5", "mobile"),
}
CLS_MODELS: dict[str, tuple[str, str]] = {
    DEFAULT_CLS_MODEL: ("PP-OCRv5", "server"),
    "PP-LCNet_x0_25_textline_ori": ("PP-OCRv5", "mobile"),
}

_READY_MANIFEST_NAME = ".ready_models.json"
_ACCELERATOR_PARAM_PREFIXES: tuple[str, ...] = (
    "EngineConfig.onnxruntime.use_cuda",
    "EngineConfig.onnxruntime.cuda_ep_cfg.",
    "EngineConfig.onnxruntime.use_cann",
    "EngineConfig.onnxruntime.cann_ep_cfg.",
)


def import_rapidocr(context: str = "OCR features") -> tuple[Any, Any]:
    """Lazily import ``rapidocr`` and ``onnxruntime`` with an actionable error message."""
    try:
        import onnxruntime
        import rapidocr
    except ImportError as exc:
        raise ImportError(
            f"{context} require RapidOCR and ONNX Runtime. Install optional OCR dependencies "
            "with `uv sync --extra ocr`."
        ) from exc
    return rapidocr, onnxruntime


def get_rapidocr_model_root() -> Path:
    """Return the shared RapidOCR model cache directory under XDG cache home."""
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return cache_root / "srtrainer" / "rapidocr"


def resolve_onnx_model(
    model_name: str,
    model_dir: str | Path | None,
    role_label: str,
    known_models: Mapping[str, tuple[str, str]],
) -> tuple[str, str, Path | None]:
    """Resolve a built-in RapidOCR model name or local ONNX file/directory."""
    if model_dir is None:
        if model_name not in known_models:
            raise ValueError(
                f"Unknown OCR {role_label} model {model_name!r}. Available for automatic "
                f"download: {', '.join(known_models)}. To use other weights, supply an ONNX model."
            )
        version, model_type = known_models[model_name]
        return version, model_type, None

    path = Path(model_dir).expanduser()
    if path.is_dir():
        named_model = path / f"{model_name}.onnx"
        candidates = sorted(candidate for candidate in path.glob("*.onnx") if candidate.is_file())
        if named_model in candidates:
            path = named_model
        elif len(candidates) == 1:
            path = candidates[0]
        elif candidates:
            raise ValueError(f"Multiple ONNX models in {path}. Specify the {role_label} ONNX file.")
    if path.suffix.lower() != ".onnx" or not path.is_file():
        raise FileNotFoundError(
            f"OCR {role_label} ONNX model not found at {path}. Supply an .onnx file or "
            "a directory containing the ONNX export of your model."
        )
    version, model_type = known_models.get(model_name, ("PP-OCRv5", "server"))
    return version, model_type, path


def onnx_device_params(
    device: str | None,
    onnxruntime: Any,
    *,
    cpu_fallback_hint: str = 'ocr_device="cpu"',
) -> dict[str, Any]:
    """Build RapidOCR ONNX Runtime execution-provider params for ``cpu | gpu | cuda | npu``."""
    raw = (device or "cpu").strip().lower()
    name, separator, device_id = raw.partition(":")
    if name == "cpu" and not separator:
        return {}
    providers = {
        "gpu": ("cuda", "CUDAExecutionProvider"),
        "cuda": ("cuda", "CUDAExecutionProvider"),
        "npu": ("cann", "CANNExecutionProvider"),
    }
    if name not in providers or (separator and not device_id.isdecimal()):
        raise ValueError(
            f"Unsupported OCR device {device!r}. Use cpu, gpu[:ID], cuda[:ID], or npu[:ID]."
        )
    flag, provider = providers[name]
    available = onnxruntime.get_available_providers()
    if provider not in available:
        raise ValueError(
            f"OCR device {device!r} requires {provider}. Install an ONNX Runtime build with "
            f"that provider or set {cpu_fallback_hint}."
        )
    return {
        f"EngineConfig.onnxruntime.use_{flag}": True,
        f"EngineConfig.onnxruntime.{flag}_ep_cfg.device_id": int(device_id or 0),
    }


def strip_accelerator_device_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return a CPU-only copy of RapidOCR ``params`` without CUDA/CANN EP settings."""
    return {
        key: val
        for key, val in params.items()
        if not any(key == pref or key.startswith(pref) for pref in _ACCELERATOR_PARAM_PREFIXES)
    }


def _uses_accelerator(params: Mapping[str, Any]) -> bool:
    return bool(
        params.get("EngineConfig.onnxruntime.use_cuda")
        or params.get("EngineConfig.onnxruntime.use_cann")
    )


def _read_ready_manifest(model_root: Path) -> dict[str, list[str]]:
    manifest_path = model_root / _READY_MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, list[str]] = {}
    for key, files in raw.items():
        if isinstance(key, str) and isinstance(files, list) and all(isinstance(f, str) for f in files):
            cleaned[key] = files
    return cleaned


def _write_ready_manifest(model_root: Path, data: Mapping[str, list[str]]) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    target = model_root / _READY_MANIFEST_NAME
    fd, tmp_name = tempfile.mkstemp(dir=model_root, prefix=".ready_models.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(data), handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _are_download_keys_ready(model_root: Path, download_keys: Sequence[str]) -> bool:
    if not download_keys:
        return True
    ready = _read_ready_manifest(model_root)
    for key in download_keys:
        recorded_files = ready.get(key)
        if not recorded_files:
            return False
        for rel_file in recorded_files:
            full = model_root / rel_file
            if not full.is_file() or full.stat().st_size <= 0:
                return False
    return True


def _snapshot_onnx_files(model_root: Path) -> set[str]:
    if not model_root.is_dir():
        return set()
    files: set[str] = set()
    for path in model_root.rglob("*.onnx"):
        if path.is_file() and path.stat().st_size > 0:
            files.add(path.relative_to(model_root).as_posix())
    return files


def ensure_rapidocr_models_ready(
    params: Mapping[str, Any],
    *,
    rapidocr: Any,
    model_root: Path,
    download_keys: Sequence[str],
) -> Any | None:
    """Ensure auto-downloaded RapidOCR ONNX models exist on disk without serializing GPU/NPU init.

    Fast path: if all ``download_keys`` are already recorded in ``model_root/.ready_models.json``
    and their ``.onnx`` files exist on disk, returns ``None`` immediately without acquiring any
    lock.

    Cold-start path: acquires ``WeakFileLock`` only long enough to verify or download missing
    weights using CPU-only params (never holding the lock during CUDA/CANN session compilation).
    Returns the temporary CPU engine when one was created so CPU callers can reuse it directly.
    """
    keys = tuple(dict.fromkeys(download_keys))
    if not keys or _are_download_keys_ready(model_root, keys):
        return None

    from huggingface_hub.utils import WeakFileLock

    lock_path = model_root / ".download.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with WeakFileLock(lock_path):
        if _are_download_keys_ready(model_root, keys):
            return None

        before_files = _snapshot_onnx_files(model_root)
        cpu_params = strip_accelerator_device_params(params)
        engine = rapidocr.RapidOCR(params=cpu_params)
        after_files = _snapshot_onnx_files(model_root)

        added_files = sorted(after_files - before_files)
        recorded_files = added_files if added_files else sorted(after_files)
        if recorded_files:
            ready = _read_ready_manifest(model_root)
            for key in keys:
                ready[key] = recorded_files
            _write_ready_manifest(model_root, ready)
        return engine


def create_rapidocr_engine(
    params: Mapping[str, Any],
    *,
    rapidocr: Any,
    model_root: Path,
    download_keys: Sequence[str],
) -> Any:
    """Create a RapidOCR engine safely across multi-process GPU/NPU/CPU workers.

    Serializes only cold-start model file downloads via :func:`ensure_rapidocr_models_ready`;
    actual CUDA/CANN ONNX Runtime session creation always runs concurrently outside any file lock.
    """
    prefetched_cpu_engine = ensure_rapidocr_models_ready(
        params,
        rapidocr=rapidocr,
        model_root=model_root,
        download_keys=download_keys,
    )
    if prefetched_cpu_engine is not None and not _uses_accelerator(params):
        return prefetched_cpu_engine
    return rapidocr.RapidOCR(params=dict(params))
