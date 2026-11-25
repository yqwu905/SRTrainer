"""Offline RapidOCR text-line detection & recognition CLI for SRTrainer JSON manifests.

Reads a unified list-of-dicts training/validation JSON manifest containing ``gt`` image paths,
runs RapidOCR detection (and recognition unless ``--det-only`` is set) on each ``gt`` image in
parallel across multiple CUDA GPUs, Ascend NPUs, or CPU workers, overwrites existing
``text_lines`` (or inserts ``text_lines`` when absent), and atomically writes the result to a new
JSON file while preserving all other fields and sample order.

Examples:
    # Auto-detect all available GPUs / NPUs and run PP-OCRv5 det + rec
    uv run python tools/detect_text_lines.py \\
        --input-json /data/train.json \\
        --output-json /data/train_with_lines.json \\
        --root-dir /data

    # Explicit 8-card Ascend NPU parallel detection + recognition
    uv run python tools/detect_text_lines.py \\
        --input-json /data/train.json \\
        --output-json /data/train_with_lines.json \\
        --root-dir /data \\
        --devices npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7

    # Explicit 4-GPU detection only (faster when transcripts are not needed)
    uv run python tools/detect_text_lines.py \\
        --input-json /data/train.json \\
        --output-json /data/train_det_only.json \\
        --root-dir /data \\
        --devices cuda:0,cuda:1,cuda:2,cuda:3 \\
        --det-only
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.text_geometry import canonical_quad  # noqa: E402

__all__ = [
    "DEFAULT_DET_MODEL",
    "DEFAULT_REC_MODEL",
    "DEFAULT_CLS_MODEL",
    "DetectConfig",
    "DetectSummary",
    "parse_devices",
    "resolve_gt_path",
    "build_rapidocr_params",
    "extract_valid_text_lines",
    "run_detect_manifest",
    "main",
]

DEFAULT_DET_MODEL: str = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL: str = "PP-OCRv5_server_rec"
DEFAULT_CLS_MODEL: str = "PP-LCNet_x1_0_textline_ori"

_DET_MODELS: dict[str, tuple[str, str]] = {
    "PP-OCRv5_server_det": ("PP-OCRv5", "server"),
    "PP-OCRv5_mobile_det": ("PP-OCRv5", "mobile"),
    "PP-OCRv4_server_det": ("PP-OCRv4", "server"),
    "PP-OCRv4_mobile_det": ("PP-OCRv4", "mobile"),
}
_REC_MODELS: dict[str, tuple[str, str]] = {
    "PP-OCRv5_server_rec": ("PP-OCRv5", "server"),
    "PP-OCRv5_mobile_rec": ("PP-OCRv5", "mobile"),
    "PP-OCRv4_server_rec": ("PP-OCRv4", "server"),
    "PP-OCRv4_mobile_rec": ("PP-OCRv4", "mobile"),
}
_CLS_MODELS: dict[str, tuple[str, str]] = {
    "PP-LCNet_x1_0_textline_ori": ("PP-OCRv5", "server"),
    "PP-LCNet_x0_25_textline_ori": ("PP-OCRv5", "mobile"),
}

_LOGGER = logging.getLogger("srtrainer.detect_text_lines")


@dataclass(frozen=True)
class DetectConfig:
    """RapidOCR detection/recognition parameters shared across workers."""

    det_model: str = DEFAULT_DET_MODEL
    det_model_dir: str | None = None
    rec_model: str = DEFAULT_REC_MODEL
    rec_model_dir: str | None = None
    cls_model: str = DEFAULT_CLS_MODEL
    cls_model_dir: str | None = None
    det_only: bool = False
    use_cls: bool = False
    rec_batch_size: int = 6
    limit_side_len: int = 960
    limit_type: str = "max"
    box_thresh: float = 0.5
    unclip_ratio: float = 1.6
    text_score: float = 0.5
    min_box_size: float = 3.0
    intra_threads: int | None = None
    on_error: str = "raise"


@dataclass(frozen=True)
class DetectSummary:
    """Execution summary returned by :func:`run_detect_manifest`."""

    total_samples: int
    overwritten_samples: int
    added_samples: int
    total_lines: int
    empty_samples: int
    skipped_errors: int
    dropped_boxes: int
    devices: tuple[str, ...]
    elapsed_sec: float
    output_json: Path


def _import_rapidocr() -> tuple[Any, Any]:
    try:
        import onnxruntime
        import rapidocr
    except ImportError as exc:
        raise ImportError(
            "Offline text-line detection requires RapidOCR and ONNX Runtime. "
            "Install optional OCR dependencies with `uv sync --extra ocr`."
        ) from exc
    return rapidocr, onnxruntime


def _resolve_onnx_model(
    model_name: str,
    model_dir: str | Path | None,
    role_label: str,
    known_models: Mapping[str, tuple[str, str]],
) -> tuple[str, str, Path | None]:
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
        candidates = sorted(p for p in path.glob("*.onnx") if p.is_file())
        if named_model in candidates:
            path = named_model
        elif len(candidates) == 1:
            path = candidates[0]
        elif candidates:
            raise ValueError(
                f"Multiple ONNX models found in {path}. Specify the exact {role_label} .onnx file."
            )
    if path.suffix.lower() != ".onnx" or not path.is_file():
        raise FileNotFoundError(
            f"OCR {role_label} ONNX model not found at {path}. Supply an .onnx file or "
            "a directory containing the ONNX export of your model."
        )
    version, model_type = known_models.get(model_name, ("PP-OCRv5", "server"))
    return version, model_type, path


def _onnx_device_params(device: str, onnxruntime: Any) -> dict[str, Any]:
    name, separator, device_id = device.strip().lower().partition(":")
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
            f"OCR device {device!r} requires {provider}, but available ONNX Runtime providers "
            f"are {available}. Install the matching ONNX Runtime build or pass `--devices cpu`."
        )
    return {
        f"EngineConfig.onnxruntime.use_{flag}": True,
        f"EngineConfig.onnxruntime.{flag}_ep_cfg.device_id": int(device_id or 0),
    }


def _detect_accelerator_devices(device_type: str = "auto") -> list[str]:
    """Auto-detect available CUDA or Ascend NPU devices supported by ONNX Runtime."""
    norm_type = device_type.strip().lower()
    if norm_type not in {"auto", "cuda", "gpu", "npu", "cpu"}:
        raise ValueError(
            f"Unsupported device_type={device_type!r}; expected auto, cuda, npu, or cpu."
        )
    if norm_type == "cpu":
        return ["cpu"]

    _, onnxruntime = _import_rapidocr()
    providers = set(onnxruntime.get_available_providers())

    def _cuda_count() -> int:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except Exception:
            pass
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if vis and vis not in {"-1", "none", "NoDevFiles"}:
            return len([tok for tok in vis.split(",") if tok.strip()])
        return 0

    def _npu_count() -> int:
        try:
            import torch

            try:
                import torch_npu  # noqa: F401
            except ImportError:
                pass
            if hasattr(torch, "npu") and torch.npu.is_available():
                return int(torch.npu.device_count())
        except Exception:
            pass
        vis = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
        if vis:
            return len([tok for tok in vis.split(",") if tok.strip()])
        return 0

    if norm_type in {"auto", "cuda", "gpu"} and "CUDAExecutionProvider" in providers:
        count = _cuda_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
        if norm_type in {"cuda", "gpu"}:
            return ["cuda:0"]

    if norm_type in {"auto", "npu"} and "CANNExecutionProvider" in providers:
        count = _npu_count()
        if count > 0:
            return [f"npu:{i}" for i in range(count)]
        if norm_type == "npu":
            return ["npu:0"]

    if norm_type in {"cuda", "gpu", "npu"}:
        raise RuntimeError(
            f"Requested --device-type={norm_type!r}, but no matching ONNX Runtime execution "
            f"provider is available (available={sorted(providers)})."
        )
    return ["cpu"]


def parse_devices(
    devices: str | Sequence[str] | None,
    *,
    device_type: str = "auto",
    workers_per_device: int = 1,
) -> list[str]:
    """Parse ``--devices`` specification into a flat worker device list."""
    if isinstance(workers_per_device, bool) or workers_per_device < 1:
        raise ValueError(
            f"workers_per_device must be a positive integer, got {workers_per_device!r}."
        )

    if devices is None:
        base_devices = _detect_accelerator_devices(device_type=device_type)
    else:
        raw_tokens = (
            [tok.strip() for tok in devices.split(",") if tok.strip()]
            if isinstance(devices, str)
            else [str(tok).strip() for tok in devices if str(tok).strip()]
        )
        if not raw_tokens:
            raise ValueError("devices must not be empty when specified.")

        inferred_prefix: str | None = None
        norm_dt = device_type.strip().lower()
        if norm_dt in {"cuda", "gpu"}:
            inferred_prefix = "cuda"
        elif norm_dt == "npu":
            inferred_prefix = "npu"

        base_devices = []
        for token in raw_tokens:
            low = token.lower()
            if low.isdecimal():
                if inferred_prefix is None:
                    detected = _detect_accelerator_devices("auto")
                    inferred_prefix = detected[0].partition(":")[0]
                    if inferred_prefix == "cpu":
                        raise ValueError(
                            f"Bare device index {token!r} requires a GPU/NPU backend or explicit "
                            "`--device-type cuda|npu` (or write `cuda:{token}` / `npu:{token}`)."
                        )
                base_devices.append(f"{inferred_prefix}:{low}")
            elif low == "cpu":
                base_devices.append("cpu")
            else:
                name, sep, dev_id = low.partition(":")
                if name not in {"cuda", "gpu", "npu"} or (sep and not dev_id.isdecimal()):
                    raise ValueError(
                        f"Invalid device entry {token!r}; expected cpu, cuda[:ID], gpu[:ID], "
                        "npu[:ID], or a numeric device index."
                    )
                canonical_name = "cuda" if name == "gpu" else name
                base_devices.append(f"{canonical_name}:{dev_id or '0'}")

    expanded: list[str] = []
    for dev in base_devices:
        expanded.extend([dev] * workers_per_device)
    return expanded


def resolve_gt_path(raw_gt: str, root_dir: Path | None, manifest_parent: Path) -> Path:
    """Resolve a manifest ``gt`` path consistently with SRTrainer manifest conventions."""
    if not isinstance(raw_gt, str) or not raw_gt.strip():
        raise ValueError(f"Expected non-empty string 'gt' path, got {raw_gt!r}.")
    path = Path(raw_gt)
    if path.is_absolute():
        return path
    if root_dir is not None:
        return root_dir / path
    candidate = manifest_parent / path
    if candidate.exists() or not path.exists():
        return candidate
    return path


def build_rapidocr_params(
    cfg: DetectConfig,
    device: str,
    rapidocr: Any,
    onnxruntime: Any,
) -> tuple[dict[str, Any], bool, Path]:
    """Build RapidOCR 3.x ``params`` dictionary matching ``src/metrics/ocr_metrics.py``."""
    det_version, det_type, det_path = _resolve_onnx_model(
        cfg.det_model, cfg.det_model_dir, "detection", _DET_MODELS
    )
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    model_root = cache_root / "srtrainer" / "rapidocr"

    det_lang: Any = rapidocr.LangDet("ch") if hasattr(rapidocr, "LangDet") else "ch"
    params: dict[str, Any] = {
        "Global.use_det": True,
        "Global.use_cls": bool(cfg.use_cls),
        "Global.use_rec": not bool(cfg.det_only),
        "Global.text_score": float(cfg.text_score),
        "Global.model_root_dir": str(model_root),
        "Det.engine_type": rapidocr.EngineType.ONNXRUNTIME,
        "Det.ocr_version": rapidocr.OCRVersion(det_version),
        "Det.model_type": rapidocr.ModelType(det_type),
        "Det.lang_type": det_lang,
        "Det.model_path": str(det_path) if det_path is not None else None,
        "Det.limit_side_len": int(cfg.limit_side_len),
        "Det.limit_type": str(cfg.limit_type),
        "Det.box_thresh": float(cfg.box_thresh),
        "Det.unclip_ratio": float(cfg.unclip_ratio),
        **_onnx_device_params(device, onnxruntime),
    }
    if cfg.intra_threads is not None:
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = int(cfg.intra_threads)

    needs_download = det_path is None
    if not cfg.det_only:
        rec_version, rec_type, rec_path = _resolve_onnx_model(
            cfg.rec_model, cfg.rec_model_dir, "recognition", _REC_MODELS
        )
        needs_download |= rec_path is None
        params.update(
            {
                "Rec.engine_type": rapidocr.EngineType.ONNXRUNTIME,
                "Rec.ocr_version": rapidocr.OCRVersion(rec_version),
                "Rec.model_type": rapidocr.ModelType(rec_type),
                "Rec.lang_type": "ch",
                "Rec.model_path": str(rec_path) if rec_path is not None else None,
                "Rec.rec_batch_num": int(cfg.rec_batch_size),
            }
        )
        if rec_path is not None:
            dict_path = rec_path.with_name("dict.txt")
            if dict_path.is_file():
                params["Rec.rec_keys_path"] = str(dict_path)

    if cfg.use_cls:
        cls_version, cls_type, cls_path = _resolve_onnx_model(
            cfg.cls_model, cfg.cls_model_dir, "orientation", _CLS_MODELS
        )
        needs_download |= cls_path is None
        params.update(
            {
                "Cls.engine_type": rapidocr.EngineType.ONNXRUNTIME,
                "Cls.ocr_version": rapidocr.OCRVersion(cls_version),
                "Cls.model_type": rapidocr.ModelType(cls_type),
                "Cls.model_path": str(cls_path) if cls_path is not None else None,
                "Cls.cls_batch_num": int(cfg.rec_batch_size),
            }
        )

    return params, needs_download, model_root


def _create_rapidocr_engine(cfg: DetectConfig, device: str) -> Any:
    rapidocr, onnxruntime = _import_rapidocr()
    params, needs_download, model_root = build_rapidocr_params(cfg, device, rapidocr, onnxruntime)
    if needs_download:
        from huggingface_hub.utils import WeakFileLock

        lock_path = model_root / ".download.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with WeakFileLock(lock_path):
            return rapidocr.RapidOCR(params=params)
    return rapidocr.RapidOCR(params=params)


def _load_bgr_image(image_path: Path) -> np.ndarray:
    if not image_path.is_file():
        raise FileNotFoundError(f"GT image not found: {image_path}")
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def extract_valid_text_lines(
    result: Any,
    image_hw: tuple[int, int],
    *,
    min_box_size: float = 3.0,
    det_only: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Convert a RapidOCR 3.x result object into manifest-ready ``text_lines`` dicts.

    Clamps coordinates to image bounds, rejects degenerate/crossed quads via ``canonical_quad``,
    and guarantees every returned entry passes ``data.manifest.load_train_manifest`` validation.
    """
    if result is None:
        return [], 0
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], 0

    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    img_h, img_w = int(image_hw[0]), int(image_hw[1])

    lines: list[dict[str, Any]] = []
    dropped = 0
    for idx, raw_box in enumerate(boxes):
        try:
            pts = np.asarray(raw_box, dtype=np.float64)
            if pts.shape == (8,):
                pts = pts.reshape(4, 2)
            if pts.shape != (4, 2) or not np.isfinite(pts).all():
                dropped += 1
                continue

            pts[:, 0] = np.clip(pts[:, 0], 0.0, float(img_w))
            pts[:, 1] = np.clip(pts[:, 1], 0.0, float(img_h))
            norm_pts = canonical_quad(pts, name=f"rapidocr_box[{idx}]")

            top, right, bottom, left = np.linalg.norm(
                np.roll(norm_pts, -1, axis=0) - norm_pts, axis=1
            )
            box_w = float(max(top, bottom))
            box_h = float(max(left, right))
            if box_w < min_box_size or box_h < min_box_size:
                dropped += 1
                continue

            rounded = [[round(float(x), 2), round(float(y), 2)] for x, y in norm_pts.tolist()]
            # Re-verify after rounding so serialized JSON always passes strict manifest loading.
            canonical_quad(rounded, name=f"rapidocr_box_rounded[{idx}]")
        except ValueError:
            dropped += 1
            continue

        txt = "" if (det_only or txts is None or idx >= len(txts) or txts[idx] is None) else str(txts[idx])
        line_dict: dict[str, Any] = {
            "transcript": txt,
            "quad": rounded,
        }
        if scores is not None and idx < len(scores) and scores[idx] is not None:
            try:
                score_val = float(scores[idx])
                if math.isfinite(score_val):
                    line_dict["score"] = round(score_val, 4)
            except (TypeError, ValueError):
                pass
        lines.append(line_dict)

    return lines, dropped


def _process_one_image(
    engine: Any,
    image_path: Path,
    cfg: DetectConfig,
) -> tuple[list[dict[str, Any]], int, bool]:
    try:
        bgr = _load_bgr_image(image_path)
        result = engine(bgr)
        lines, dropped = extract_valid_text_lines(
            result,
            (int(bgr.shape[0]), int(bgr.shape[1])),
            min_box_size=cfg.min_box_size,
            det_only=cfg.det_only,
        )
        return lines, dropped, False
    except Exception:
        if cfg.on_error == "skip":
            _LOGGER.warning("Failed to detect text on %s; writing empty text_lines.", image_path)
            return [], 0, True
        raise


def _worker_loop(
    worker_id: int,
    device: str,
    cfg: DetectConfig,
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        engine = _create_rapidocr_engine(cfg, device)
    except BaseException:
        result_queue.put(("error", worker_id, device, traceback.format_exc()))
        return

    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):
            break
        if task is None:
            break
        chunk_out: list[tuple[int, list[dict[str, Any]], int, bool]] = []
        try:
            for sample_idx, gt_path_str in task:
                lines, dropped, had_err = _process_one_image(engine, Path(gt_path_str), cfg)
                chunk_out.append((sample_idx, lines, dropped, had_err))
        except BaseException:
            result_queue.put(("error", worker_id, device, traceback.format_exc()))
            return
        result_queue.put(("ok", worker_id, device, chunk_out))


def _atomic_write_json(path: Path, payload: Any, *, indent: int | None) -> None:
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
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def run_detect_manifest(
    input_json: str | Path,
    output_json: str | Path,
    *,
    root_dir: str | Path | None = None,
    devices: str | Sequence[str] | None = None,
    device_type: str = "auto",
    workers_per_device: int = 1,
    chunk_size: int = 16,
    config: DetectConfig | None = None,
    indent: int | None = 2,
    engine_factory: Any = None,
) -> DetectSummary:
    """Run offline RapidOCR detection/recognition on every ``gt`` in a manifest JSON.

    Existing ``text_lines`` fields are overwritten; entries without ``text_lines`` receive a new
    ``text_lines`` list. All other fields and original sample ordering are preserved.
    """
    cfg = config if config is not None else DetectConfig()
    if cfg.on_error not in {"raise", "skip"}:
        raise ValueError(f"Unsupported on_error={cfg.on_error!r}; expected 'raise' or 'skip'.")
    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    if cfg.limit_type not in {"max", "min"}:
        raise ValueError(f"limit_type must be 'max' or 'min', got {cfg.limit_type!r}.")

    in_path = Path(input_json)
    out_path = Path(output_json)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input manifest JSON not found: {in_path}")

    raw_data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(
            f"Input manifest {in_path} must be a JSON list of dicts, got {type(raw_data).__name__}."
        )

    resolved_root = Path(root_dir) if root_dir is not None else None
    tasks: list[tuple[int, str]] = []
    overwritten = 0
    added = 0
    for idx, entry in enumerate(raw_data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Input manifest {in_path}[{idx}] must be a JSON object (dict), "
                f"got {type(entry).__name__}."
            )
        if "gt" not in entry:
            raise ValueError(f"Input manifest {in_path}[{idx}] is missing required key 'gt'.")
        gt_path = resolve_gt_path(entry["gt"], resolved_root, in_path.parent)
        tasks.append((idx, str(gt_path)))
        if "text_lines" in entry:
            overwritten += 1
        else:
            added += 1

    worker_devices = parse_devices(
        devices,
        device_type=device_type,
        workers_per_device=workers_per_device,
    )
    if engine_factory is not None:
        # Custom factory is intended for in-process testing.
        worker_devices = [worker_devices[0]]

    num_samples = len(tasks)
    start_time = time.perf_counter()
    results_by_idx: list[list[dict[str, Any]] | None] = [None] * num_samples
    dropped_boxes = 0
    skipped_errors = 0

    _LOGGER.info(
        "[Detect] Starting offline RapidOCR (%s) on %d samples | workers=%d (%s) | input=%s",
        "det-only" if cfg.det_only else "det+rec",
        num_samples,
        len(worker_devices),
        ", ".join(worker_devices),
        in_path,
    )

    if num_samples == 0:
        _atomic_write_json(out_path, [], indent=indent)
        return DetectSummary(
            total_samples=0,
            overwritten_samples=0,
            added_samples=0,
            total_lines=0,
            empty_samples=0,
            skipped_errors=0,
            dropped_boxes=0,
            devices=tuple(worker_devices),
            elapsed_sec=max(time.perf_counter() - start_time, 1e-9),
            output_json=out_path,
        )

    progress_bar: Any = None
    try:
        from tqdm import tqdm

        progress_bar = tqdm(total=num_samples, desc="Detecting text_lines", unit="img")
    except Exception:
        progress_bar = None

    try:
        if len(worker_devices) == 1 or num_samples == 1:
            dev = worker_devices[0]
            engine = (
                engine_factory(cfg, dev)
                if engine_factory is not None
                else _create_rapidocr_engine(cfg, dev)
            )
            for sample_idx, gt_path_str in tasks:
                lines, dropped, had_err = _process_one_image(engine, Path(gt_path_str), cfg)
                results_by_idx[sample_idx] = lines
                dropped_boxes += dropped
                skipped_errors += int(had_err)
                if progress_bar is not None:
                    progress_bar.update(1)
        else:
            chunks = [tasks[i : i + chunk_size] for i in range(0, num_samples, chunk_size)]
            num_workers = min(len(worker_devices), len(chunks))
            active_devices = worker_devices[:num_workers]
            worker_devices = active_devices

            ctx = mp.get_context("spawn")
            task_q: Any = ctx.Queue()
            result_q: Any = ctx.Queue()

            for chunk in chunks:
                task_q.put(chunk)
            for _ in range(num_workers):
                task_q.put(None)

            procs = [
                ctx.Process(
                    target=_worker_loop,
                    args=(wid, dev, cfg, task_q, result_q),
                    daemon=False,
                )
                for wid, dev in enumerate(active_devices)
            ]
            for proc in procs:
                proc.start()

            completed_chunks = 0
            try:
                while completed_chunks < len(chunks):
                    try:
                        msg = result_q.get(timeout=1.0)
                    except queue.Empty:
                        dead = [
                            (wid, proc.exitcode)
                            for wid, proc in enumerate(procs)
                            if not proc.is_alive() and proc.exitcode not in (0, None)
                        ]
                        if dead:
                            raise RuntimeError(
                                f"RapidOCR worker(s) exited unexpectedly: {dead}."
                            ) from None
                        continue

                    status = msg[0]
                    if status == "error":
                        _, wid, dev, tb_text = msg
                        raise RuntimeError(
                            f"RapidOCR worker {wid} on {dev!r} failed:\n{tb_text}"
                        )
                    _, _, _, chunk_out = msg
                    for sample_idx, lines, dropped, had_err in chunk_out:
                        results_by_idx[sample_idx] = lines
                        dropped_boxes += dropped
                        skipped_errors += int(had_err)
                    completed_chunks += 1
                    if progress_bar is not None:
                        progress_bar.update(len(chunk_out))
            except BaseException:
                for proc in procs:
                    if proc.is_alive():
                        proc.terminate()
                for proc in procs:
                    proc.join(timeout=5.0)
                raise
            else:
                for proc in procs:
                    proc.join()
                    if proc.exitcode != 0:
                        raise RuntimeError(
                            f"RapidOCR worker PID={proc.pid} exited with code {proc.exitcode}."
                        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    output_records: list[dict[str, Any]] = []
    total_lines = 0
    empty_samples = 0
    for idx, orig_entry in enumerate(raw_data):
        detected = results_by_idx[idx]
        if detected is None:
            raise RuntimeError(f"Missing detection result for sample index {idx}.")
        record = copy.deepcopy(orig_entry)
        record["text_lines"] = detected
        output_records.append(record)
        total_lines += len(detected)
        if not detected:
            empty_samples += 1

    _atomic_write_json(out_path, output_records, indent=indent)
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    _LOGGER.info(
        "[Detect] Done: %d samples (%d lines, %d empty, %d dropped boxes, %d errors) in %.2fs "
        "(%.2f img/s) | overwritten=%d added=%d | saved=%s",
        num_samples,
        total_lines,
        empty_samples,
        dropped_boxes,
        skipped_errors,
        elapsed,
        num_samples / elapsed,
        overwritten,
        added,
        out_path,
    )
    return DetectSummary(
        total_samples=num_samples,
        overwritten_samples=overwritten,
        added_samples=added,
        total_lines=total_lines,
        empty_samples=empty_samples,
        skipped_errors=skipped_errors,
        dropped_boxes=dropped_boxes,
        devices=tuple(worker_devices),
        elapsed_sec=elapsed,
        output_json=out_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline multi-device RapidOCR text-line detection & recognition for SRTrainer JSON "
            "manifests. Overwrites existing `text_lines` or inserts new `text_lines` per sample."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Input manifest JSON path (list of dicts containing 'gt').",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output manifest JSON path (written atomically; may equal --input-json).",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Optional root directory prepended to relative 'gt' paths.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated device list, e.g. 'cuda:0,cuda:1', 'npu:0,npu:1', '0,1,2,3', or "
            "'cpu'. Omit to auto-use all available GPUs/NPUs (or CPU)."
        ),
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="auto",
        choices=("auto", "cuda", "gpu", "npu", "cpu"),
        help="Backend type for auto-detection or bare numeric --devices indices (default: auto).",
    )
    parser.add_argument(
        "--workers-per-device",
        type=int,
        default=1,
        help="Number of parallel worker processes spawned per device (default: 1).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="Number of images dispatched per dynamic queue chunk (default: 16).",
    )
    parser.add_argument(
        "--det-only",
        action="store_true",
        help="Run text detection only (skip recognition; faster when transcripts are unneeded).",
    )
    parser.add_argument(
        "--use-cls",
        action="store_true",
        help="Enable 0/180-deg text-line orientation classification.",
    )
    parser.add_argument(
        "--det-model",
        type=str,
        default=DEFAULT_DET_MODEL,
        help=f"RapidOCR detection model name (default: {DEFAULT_DET_MODEL}).",
    )
    parser.add_argument(
        "--det-model-dir",
        type=str,
        default=None,
        help="Optional local detection .onnx file or directory containing it.",
    )
    parser.add_argument(
        "--rec-model",
        type=str,
        default=DEFAULT_REC_MODEL,
        help=f"RapidOCR recognition model name (default: {DEFAULT_REC_MODEL}).",
    )
    parser.add_argument(
        "--rec-model-dir",
        type=str,
        default=None,
        help="Optional local recognition .onnx file or directory containing it.",
    )
    parser.add_argument(
        "--cls-model",
        type=str,
        default=DEFAULT_CLS_MODEL,
        help=f"RapidOCR orientation model name when --use-cls is set (default: {DEFAULT_CLS_MODEL}).",
    )
    parser.add_argument(
        "--cls-model-dir",
        type=str,
        default=None,
        help="Optional local orientation .onnx file or directory containing it.",
    )
    parser.add_argument(
        "--rec-batch-size",
        type=int,
        default=6,
        help="RapidOCR recognition batch size per image (default: 6).",
    )
    parser.add_argument(
        "--limit-side-len",
        type=int,
        default=960,
        help="RapidOCR Det.limit_side_len (default: 960; raise e.g. 1280/1536 for tiny text).",
    )
    parser.add_argument(
        "--limit-type",
        type=str,
        default="max",
        choices=("max", "min"),
        help="RapidOCR Det.limit_type (default: max).",
    )
    parser.add_argument(
        "--box-thresh",
        type=float,
        default=0.5,
        help="RapidOCR Det.box_thresh confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--unclip-ratio",
        type=float,
        default=1.6,
        help="RapidOCR Det.unclip_ratio polygon expansion factor (default: 1.6).",
    )
    parser.add_argument(
        "--text-score",
        type=float,
        default=0.5,
        help="RapidOCR Global.text_score recognition filter threshold (default: 0.5).",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=3.0,
        help="Minimum box width/height in pixels to keep (default: 3.0).",
    )
    parser.add_argument(
        "--intra-threads",
        type=int,
        default=None,
        help="Optional ONNX Runtime intra_op_num_threads per worker process.",
    )
    parser.add_argument(
        "--on-error",
        type=str,
        default="raise",
        choices=("raise", "skip"),
        help="Behavior on missing/corrupt images: 'raise' (default) or 'skip' with empty list.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Output JSON indentation spaces (<= 0 writes compact single-line JSON; default: 2).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> DetectSummary:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = _build_arg_parser().parse_args(argv)
    cfg = DetectConfig(
        det_model=args.det_model,
        det_model_dir=args.det_model_dir,
        rec_model=args.rec_model,
        rec_model_dir=args.rec_model_dir,
        cls_model=args.cls_model,
        cls_model_dir=args.cls_model_dir,
        det_only=bool(args.det_only),
        use_cls=bool(args.use_cls),
        rec_batch_size=args.rec_batch_size,
        limit_side_len=args.limit_side_len,
        limit_type=args.limit_type,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
        text_score=args.text_score,
        min_box_size=args.min_box_size,
        intra_threads=args.intra_threads,
        on_error=args.on_error,
    )
    indent = args.indent if args.indent > 0 else None
    return run_detect_manifest(
        input_json=args.input_json,
        output_json=args.output_json,
        root_dir=args.root_dir,
        devices=args.devices,
        device_type=args.device_type,
        workers_per_device=args.workers_per_device,
        chunk_size=args.chunk_size,
        config=cfg,
        indent=indent,
    )


if __name__ == "__main__":
    main()
