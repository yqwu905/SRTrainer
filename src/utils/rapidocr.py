"""Shared RapidOCR 3.x / PyTorch helpers for metrics and offline text detection.

All RapidOCR PyTorch weights (det / rec / cls) and character dictionaries live in the
project-local ``experiments/weights/rapidocr`` directory instead of RapidOCR's default
``site-packages/rapidocr/models`` location. Override the weights root with the
``SRTRAINER_WEIGHTS_DIR`` environment variable when the project tree is read-only.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "DEFAULT_DET_MODEL",
    "DEFAULT_REC_MODEL",
    "DEFAULT_CLS_MODEL",
    "DET_MODELS",
    "REC_MODELS",
    "CLS_MODELS",
    "WEIGHTS_DIR_ENV_VAR",
    "RAPIDOCR_WEIGHTS_DIRNAME",
    "import_rapidocr",
    "get_project_root",
    "get_weights_dir",
    "get_rapidocr_model_root",
    "default_torch_model_path",
    "resolve_torch_model",
    "resolve_rec_dict_path",
    "torch_device_params",
    "strip_accelerator_device_params",
    "ensure_rapidocr_models_ready",
    "create_rapidocr_engine",
]

DEFAULT_DET_MODEL: str = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL: str = "PP-OCRv5_server_rec"
DEFAULT_CLS_MODEL: str = "ch_ppocr_mobile_v2.0_cls"

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
# RapidOCR 3.x PyTorch backend provides the PP-OCRv4 MobileNetV3 orientation classifier.
CLS_MODELS: dict[str, tuple[str, str]] = {
    DEFAULT_CLS_MODEL: ("PP-OCRv4", "mobile"),
    "ch_ppocr_mobile_v2.0_cls_mobile": ("PP-OCRv4", "mobile"),
}

# Canonical filename stems matching RapidOCR's PyTorch arch_config.yaml keys.
_CANONICAL_TORCH_STEMS: dict[str, str] = {
    "PP-OCRv5_server_det": "ch_PP-OCRv5_det_server",
    "PP-OCRv5_mobile_det": "ch_PP-OCRv5_det_mobile",
    "PP-OCRv4_server_det": "ch_PP-OCRv4_det_server",
    "PP-OCRv4_mobile_det": "ch_PP-OCRv4_det_mobile",
    "PP-OCRv5_server_rec": "ch_PP-OCRv5_rec_server",
    "PP-OCRv5_mobile_rec": "ch_PP-OCRv5_rec_mobile",
    "PP-OCRv4_server_rec": "ch_PP-OCRv4_rec_server",
    "PP-OCRv4_mobile_rec": "ch_PP-OCRv4_rec_mobile",
    "ch_ppocr_mobile_v2.0_cls": "ch_ppocr_mobile_v2.0_cls_mobile",
    "ch_ppocr_mobile_v2.0_cls_mobile": "ch_ppocr_mobile_v2.0_cls_mobile",
}

_VERSION_DICT_NAMES: dict[str, str] = {
    "PP-OCRv5": "ppocrv5_dict.txt",
    "PP-OCRv4": "ppocr_keys_v1.txt",
}

_TORCH_WEIGHT_SUFFIXES: tuple[str, ...] = (".pth", ".pt")
_MODEL_SNAPSHOT_SUFFIXES: frozenset[str] = frozenset({".pth", ".pt", ".txt"})
_READY_MANIFEST_NAME = ".ready_models.json"
_ACCELERATOR_PARAM_PREFIXES: tuple[str, ...] = (
    "EngineConfig.torch.use_cuda",
    "EngineConfig.torch.cuda_ep_cfg.",
    "EngineConfig.torch.use_npu",
    "EngineConfig.torch.npu_ep_cfg.",
    "EngineConfig.torch.use_mps",
)


def import_rapidocr(context: str = "OCR features") -> Any:
    """Lazily import ``rapidocr`` with an actionable error message."""
    try:
        import rapidocr
    except ImportError as exc:
        raise ImportError(
            f"{context} require RapidOCR (PyTorch backend). Install optional OCR dependencies "
            "with `uv sync --extra ocr`."
        ) from exc
    return rapidocr


WEIGHTS_DIR_ENV_VAR: str = "SRTRAINER_WEIGHTS_DIR"
#: Environment variable overriding the project-local ``experiments/weights`` directory.

RAPIDOCR_WEIGHTS_DIRNAME: str = "rapidocr"
#: Sub-directory of ``experiments/weights`` holding the RapidOCR PyTorch det / rec / cls weights.

#: Project-local weights root, relative to the repository root.
WEIGHTS_DIR_RELPATH: tuple[str, ...] = ("experiments", "weights")


def get_project_root() -> Path:
    """Return the SRTrainer repository root (the closest parent holding ``pyproject.toml``)."""
    source_file = Path(__file__).resolve()
    for parent in source_file.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return source_file.parents[2]


def get_weights_dir() -> Path:
    """Return the project weights root ``experiments/weights`` (override via the env var)."""
    override = os.environ.get(WEIGHTS_DIR_ENV_VAR)
    if override is not None and override.strip():
        return Path(override).expanduser()
    return get_project_root().joinpath(*WEIGHTS_DIR_RELPATH)


def get_rapidocr_model_root() -> Path:
    """Return the project-local RapidOCR model directory ``experiments/weights/rapidocr``.

    Both explicitly loaded ``.pth`` files and any auto-downloaded weights or character
    dictionaries are kept here, never in RapidOCR's default ``site-packages/rapidocr/models``
    nor in a user cache directory.
    """
    return get_weights_dir() / RAPIDOCR_WEIGHTS_DIRNAME


def default_torch_model_path(
    model_name: str,
    *,
    model_root: str | Path | None = None,
) -> Path:
    """Return the project-local ``experiments/weights/rapidocr/<canonical_stem>.pth`` path."""
    stem = _CANONICAL_TORCH_STEMS.get(model_name, model_name)
    root = get_rapidocr_model_root() if model_root is None else Path(model_root).expanduser()
    return root / f"{stem}.pth"


def resolve_torch_model(
    model_name: str,
    model_dir: str | Path | None,
    role_label: str,
    known_models: Mapping[str, tuple[str, str]],
) -> tuple[str, str, Path | None]:
    """Resolve a built-in RapidOCR PyTorch model name or local ``.pth`` file/directory.

    With ``model_dir=None`` the project-local weights copy
    (``experiments/weights/rapidocr/<canonical_stem>.pth``) is preferred; when that file is
    still absent RapidOCR downloads the weights into the same directory.
    """
    if model_dir is None:
        if model_name not in known_models:
            raise ValueError(
                f"Unknown OCR {role_label} model {model_name!r}. Available for automatic "
                f"download (PyTorch backend): {', '.join(known_models)}. "
                "To use other weights, supply a RapidOCR PyTorch .pth model file."
            )
        version, model_type = known_models[model_name]
        weights_path = default_torch_model_path(model_name)
        if weights_path.is_file():
            return version, model_type, weights_path
        return version, model_type, None

    path = Path(model_dir).expanduser()
    if path.is_dir():
        candidates = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in _TORCH_WEIGHT_SUFFIXES
        )
        preferred_stems = list(
            dict.fromkeys(
                stem
                for stem in (_CANONICAL_TORCH_STEMS.get(model_name), model_name)
                if stem is not None
            )
        )
        matched = next(
            (
                path / f"{stem}{suffix}"
                for stem in preferred_stems
                for suffix in _TORCH_WEIGHT_SUFFIXES
                if (path / f"{stem}{suffix}") in candidates
            ),
            None,
        )
        if matched is not None:
            path = matched
        elif len(candidates) == 1:
            path = candidates[0]
        elif candidates:
            raise ValueError(
                f"Multiple PyTorch models in {path}. Specify the {role_label} .pth file."
            )
    if path.suffix.lower() not in _TORCH_WEIGHT_SUFFIXES or not path.is_file():
        raise FileNotFoundError(
            f"OCR {role_label} PyTorch model not found at {path}. Supply a .pth file (named to "
            "match RapidOCR's arch_config.yaml, e.g. ch_PP-OCRv5_rec_server.pth) or a directory "
            f"containing it, or place it in the project weights directory "
            f"{get_rapidocr_model_root()}."
        )
    default_meta = ("PP-OCRv4", "mobile") if role_label == "orientation" else ("PP-OCRv5", "server")
    version, model_type = known_models.get(model_name, default_meta)
    return version, model_type, path


def resolve_rec_dict_path(rec_path: Path, ocr_version: str) -> Path | None:
    """Locate an accompanying recognition character dictionary next to a local ``.pth`` file."""
    candidates: list[Path] = [rec_path.with_name("dict.txt")]
    version_dict = _VERSION_DICT_NAMES.get(ocr_version)
    if version_dict is not None:
        candidates.append(rec_path.with_name(version_dict))
    candidates.append(rec_path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _is_npu_available() -> tuple[bool, int]:
    """Check Ascend NPU availability via ``torch_npu`` without failing when absent."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        pass
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not hasattr(npu_mod, "is_available"):
        return False, 0
    try:
        if not npu_mod.is_available():
            return False, 0
        count = int(npu_mod.device_count()) if hasattr(npu_mod, "device_count") else 0
        return True, count
    except Exception:
        return False, 0


def torch_device_params(
    device: str | None,
    *,
    cpu_fallback_hint: str = 'ocr_device="cpu"',
) -> dict[str, Any]:
    """Build RapidOCR PyTorch engine device params for ``cpu | gpu | cuda | npu | mps``."""
    raw = (device or "cpu").strip().lower()
    name, separator, device_id = raw.partition(":")
    if name == "cpu" and not separator:
        return {}
    if name not in {"gpu", "cuda", "npu", "mps"} or (separator and not device_id.isdecimal()):
        raise ValueError(
            f"Unsupported OCR device {device!r}. Use cpu, gpu[:ID], cuda[:ID], npu[:ID], or mps."
        )

    dev_idx = int(device_id or 0)
    if name in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise ValueError(
                f"OCR device {device!r} requires CUDA in PyTorch. Install a CUDA-enabled PyTorch "
                f"build on a CUDA host or set {cpu_fallback_hint}."
            )
        cuda_count = int(torch.cuda.device_count())
        if cuda_count > 0 and dev_idx >= cuda_count:
            raise ValueError(
                f"OCR device {device!r} is out of range for {cuda_count} available CUDA device(s)."
            )
        return {
            "EngineConfig.torch.use_cuda": True,
            "EngineConfig.torch.cuda_ep_cfg.device_id": dev_idx,
        }

    if name == "npu":
        available, npu_count = _is_npu_available()
        if not available:
            raise ValueError(
                f"OCR device {device!r} requires Ascend NPU support via `torch_npu`. Install "
                f"`torch-npu` on an Ascend NPU host or set {cpu_fallback_hint}."
            )
        if npu_count > 0 and dev_idx >= npu_count:
            raise ValueError(
                f"OCR device {device!r} is out of range for {npu_count} available NPU device(s)."
            )
        return {
            "EngineConfig.torch.use_npu": True,
            "EngineConfig.torch.npu_ep_cfg.device_id": dev_idx,
        }

    # Apple Silicon MPS
    if separator and dev_idx != 0:
        raise ValueError(f"Unsupported OCR device {device!r}. MPS only supports device index 0.")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None or not mps_backend.is_available():
        raise ValueError(
            f"OCR device {device!r} requires Apple Silicon MPS support in PyTorch, "
            f"or set {cpu_fallback_hint}."
        )
    return {"EngineConfig.torch.use_mps": True}


def strip_accelerator_device_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return a CPU-only copy of RapidOCR ``params`` without CUDA/NPU/MPS PyTorch settings."""
    return {
        key: val
        for key, val in params.items()
        if not any(key == pref or key.startswith(pref) for pref in _ACCELERATOR_PARAM_PREFIXES)
    }


def _uses_accelerator(params: Mapping[str, Any]) -> bool:
    return bool(
        params.get("EngineConfig.torch.use_cuda")
        or params.get("EngineConfig.torch.use_npu")
        or params.get("EngineConfig.torch.use_mps")
    )


def _configure_npu_kernel_cache_dir(model_root: Path) -> None:
    """Redirect RapidOCR's NPU ACL compiler cache from site-packages to writable ``model_root``."""
    try:
        from rapidocr.inference_engine.pytorch import device_config as torch_device_cfg
    except ImportError:
        return
    if hasattr(torch_device_cfg, "model_dir"):
        model_root.mkdir(parents=True, exist_ok=True)
        torch_device_cfg.model_dir = model_root


def _eager_load_enabled_stages(engine: Any, params: Mapping[str, Any]) -> None:
    """Trigger RapidOCR 3.10+ lazy stage loaders so weights and dicts download under lock."""
    if params.get("Global.use_det", True) and hasattr(engine, "_load_det_model"):
        engine._load_det_model()
    if params.get("Global.use_cls", False) and hasattr(engine, "_load_cls_model"):
        engine._load_cls_model()
    if params.get("Global.use_rec", True) and hasattr(engine, "_load_rec_model"):
        engine._load_rec_model()


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


def _snapshot_model_files(model_root: Path) -> set[str]:
    if not model_root.is_dir():
        return set()
    files: set[str] = set()
    for path in model_root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in _MODEL_SNAPSHOT_SUFFIXES:
            continue
        rel = path.relative_to(model_root)
        if rel.parts and rel.parts[0] == "kernel_meta":
            continue
        if path.stat().st_size > 0:
            files.add(rel.as_posix())
    return files


def ensure_rapidocr_models_ready(
    params: Mapping[str, Any],
    *,
    rapidocr: Any,
    model_root: Path,
    download_keys: Sequence[str],
) -> Any | None:
    """Ensure auto-downloaded RapidOCR PyTorch weights & dicts exist without serializing GPU/NPU init.

    Fast path: if all ``download_keys`` are already recorded in ``model_root/.ready_models.json``
    and their weight/dictionary files exist on disk, returns ``None`` immediately without acquiring
    any lock.

    Cold-start path: acquires ``WeakFileLock`` only long enough to verify or download missing
    weights using CPU-only params (never holding the lock during CUDA/NPU/MPS initialization).
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

        before_files = _snapshot_model_files(model_root)
        cpu_params = strip_accelerator_device_params(params)
        engine = rapidocr.RapidOCR(params=cpu_params)
        _eager_load_enabled_stages(engine, cpu_params)
        after_files = _snapshot_model_files(model_root)

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
    """Create a RapidOCR PyTorch engine safely across multi-process GPU/NPU/MPS/CPU workers.

    Serializes only cold-start model/dict file downloads via :func:`ensure_rapidocr_models_ready`;
    actual CUDA/NPU/MPS PyTorch session creation always runs concurrently outside any file lock.
    """
    prefetched_cpu_engine = ensure_rapidocr_models_ready(
        params,
        rapidocr=rapidocr,
        model_root=model_root,
        download_keys=download_keys,
    )
    if prefetched_cpu_engine is not None and not _uses_accelerator(params):
        return prefetched_cpu_engine
    del prefetched_cpu_engine
    if params.get("EngineConfig.torch.use_npu"):
        _configure_npu_kernel_cache_dir(model_root)
    return rapidocr.RapidOCR(params=dict(params))
