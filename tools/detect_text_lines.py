"""Offline RapidOCR text-line detection & recognition CLI for SRTrainer JSON manifests.

Reads a unified list-of-dicts training/validation JSON manifest containing ``gt`` image paths,
runs RapidOCR detection (and recognition unless ``--det-only`` is set) on each ``gt`` image in
parallel across multiple CUDA GPUs, Ascend NPUs, or CPU workers, overwrites existing
``text_lines`` (or inserts ``text_lines`` when absent), persists progress incrementally to a
sidecar JSONL checkpoint for automatic resume after interruption, and atomically writes the
final ordered result to ``--output-json``.

Examples:
    # Auto-detect all available GPUs / NPUs and run PP-OCRv5 det + rec (auto-resumes if interrupted)
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

    # Use a non-default weights directory for the det / rec PyTorch weights
    uv run python tools/detect_text_lines.py \\
        --input-json /data/train.json \\
        --output-json /data/train_with_lines.json \\
        --root-dir /data \\
        --weights-dir /mnt/nfs/srtrainer_weights

    # Force fresh run ignoring any previous sidecar checkpoint
    uv run python tools/detect_text_lines.py \\
        --input-json /data/train.json \\
        --output-json /data/train_with_lines.json \\
        --root-dir /data \\
        --no-resume
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.rapidocr import (  # noqa: E402
    CLS_MODELS,
    DEFAULT_CLS_MODEL,
    DEFAULT_DET_MODEL,
    DEFAULT_REC_MODEL,
    DET_MODELS,
    REC_MODELS,
    WEIGHTS_DIR_ENV_VAR,
    create_rapidocr_engine,
    ensure_rapidocr_models_ready,
    get_rapidocr_model_root,
    import_rapidocr,
    resolve_rec_dict_path,
    resolve_torch_model,
    torch_device_params,
)
from utils.text_geometry import canonical_quad  # noqa: E402

__all__ = [
    "DEFAULT_DET_MODEL",
    "DEFAULT_REC_MODEL",
    "DEFAULT_CLS_MODEL",
    "DetectConfig",
    "DetectSummary",
    "default_checkpoint_path",
    "parse_devices",
    "resolve_gt_path",
    "build_rapidocr_params",
    "extract_valid_text_lines",
    "run_detect_manifest",
    "main",
]

_CHECKPOINT_VERSION = 1
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
    resumed_samples: int
    processed_samples: int
    total_lines: int
    empty_samples: int
    skipped_errors: int
    dropped_boxes: int
    devices: tuple[str, ...]
    elapsed_sec: float
    output_json: Path
    checkpoint_path: Path | None = None


def default_checkpoint_path(output_json: str | Path) -> Path:
    """Return the default sidecar JSONL checkpoint path paired with ``output_json``."""
    out = Path(output_json)
    return out.parent / f"{out.name}.ckpt.jsonl"


def _compute_run_fingerprint(
    gt_entries: Sequence[str],
    cfg: DetectConfig,
) -> str:
    """Compute a stable SHA-256 fingerprint binding checkpoint records to input order & OCR mode."""
    hasher = hashlib.sha256()
    meta = {
        "version": _CHECKPOINT_VERSION,
        "count": len(gt_entries),
        "det_model": cfg.det_model,
        "det_model_dir": cfg.det_model_dir,
        "rec_model": None if cfg.det_only else cfg.rec_model,
        "rec_model_dir": None if cfg.det_only else cfg.rec_model_dir,
        "cls_model": cfg.cls_model if cfg.use_cls else None,
        "cls_model_dir": cfg.cls_model_dir if cfg.use_cls else None,
        "det_only": cfg.det_only,
        "use_cls": cfg.use_cls,
        "limit_side_len": cfg.limit_side_len,
        "limit_type": cfg.limit_type,
        "box_thresh": cfg.box_thresh,
        "unclip_ratio": cfg.unclip_ratio,
        "text_score": cfg.text_score,
        "min_box_size": cfg.min_box_size,
    }
    hasher.update(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for raw_gt in gt_entries:
        hasher.update(b"\0")
        hasher.update(raw_gt.encode("utf-8"))
    return hasher.hexdigest()


def _detect_accelerator_devices(device_type: str = "auto") -> list[str]:
    """Auto-detect available CUDA, Ascend NPU, or MPS devices supported by PyTorch."""
    norm_type = device_type.strip().lower()
    if norm_type not in {"auto", "cuda", "gpu", "npu", "mps", "cpu"}:
        raise ValueError(
            f"Unsupported device_type={device_type!r}; expected auto, cuda, npu, mps, or cpu."
        )
    if norm_type == "cpu":
        return ["cpu"]

    def _cuda_count() -> int:
        try:
            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except Exception:
            pass
        return 0

    def _npu_count() -> int:
        try:
            try:
                import torch_npu  # noqa: F401
            except ImportError:
                pass
            if hasattr(torch, "npu") and torch.npu.is_available():
                return int(torch.npu.device_count())
        except Exception:
            pass
        return 0

    def _mps_available() -> bool:
        try:
            mps_backend = getattr(torch.backends, "mps", None)
            return bool(mps_backend is not None and mps_backend.is_available())
        except Exception:
            return False

    if norm_type in {"auto", "cuda", "gpu"}:
        count = _cuda_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
        if norm_type in {"cuda", "gpu"}:
            raise RuntimeError(
                f"Requested --device-type={norm_type!r}, but CUDA is not available in PyTorch."
            )

    if norm_type in {"auto", "npu"}:
        count = _npu_count()
        if count > 0:
            return [f"npu:{i}" for i in range(count)]
        if norm_type == "npu":
            raise RuntimeError(
                "Requested --device-type='npu', but Ascend NPU is not available via `torch_npu`."
            )

    if norm_type in {"auto", "mps"}:
        if _mps_available():
            return ["mps"]
        if norm_type == "mps":
            raise RuntimeError(
                "Requested --device-type='mps', but Apple Silicon MPS is not available in PyTorch."
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
                    if inferred_prefix in {"cpu", "mps"}:
                        raise ValueError(
                            f"Bare device index {token!r} requires a multi-device CUDA/NPU backend "
                            "or explicit `--device-type cuda|npu` (or write "
                            f"`cuda:{token}` / `npu:{token}`)."
                        )
                base_devices.append(f"{inferred_prefix}:{low}")
            elif low == "cpu":
                base_devices.append("cpu")
            elif low in {"mps", "mps:0"}:
                base_devices.append("mps")
            else:
                name, sep, dev_id = low.partition(":")
                if name not in {"cuda", "gpu", "npu"} or (sep and not dev_id.isdecimal()):
                    raise ValueError(
                        f"Invalid device entry {token!r}; expected cpu, cuda[:ID], gpu[:ID], "
                        "npu[:ID], mps, or a numeric device index."
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
) -> tuple[dict[str, Any], tuple[str, ...], Path]:
    """Build RapidOCR 3.x PyTorch ``params`` dict, auto-download model keys, and cache root.

    Local ``.pth`` weights are resolved from the project weights directory
    (``experiments/weights/rapidocr`` by default, or ``SRTRAINER_WEIGHTS_DIR``) and only fall
    back to ``Global.model_root_dir`` downloads into that same directory.
    """
    if cfg.intra_threads is not None and (
        isinstance(cfg.intra_threads, bool) or int(cfg.intra_threads) < 1
    ):
        raise ValueError(f"intra_threads must be a positive integer, got {cfg.intra_threads!r}.")

    det_version, det_type, det_path = resolve_torch_model(
        cfg.det_model, cfg.det_model_dir, "detection", DET_MODELS
    )
    model_root = get_rapidocr_model_root()
    det_lang: Any = rapidocr.LangDet("ch") if hasattr(rapidocr, "LangDet") else "ch"
    params: dict[str, Any] = {
        "Global.use_det": True,
        "Global.use_cls": bool(cfg.use_cls),
        "Global.use_rec": not bool(cfg.det_only),
        "Global.text_score": float(cfg.text_score),
        "Global.model_root_dir": str(model_root),
        "Det.engine_type": rapidocr.EngineType.TORCH,
        "Det.ocr_version": rapidocr.OCRVersion(det_version),
        "Det.model_type": rapidocr.ModelType(det_type),
        "Det.lang_type": det_lang,
        "Det.model_path": str(det_path) if det_path is not None else None,
        "Det.limit_side_len": int(cfg.limit_side_len),
        "Det.limit_type": str(cfg.limit_type),
        "Det.box_thresh": float(cfg.box_thresh),
        "Det.unclip_ratio": float(cfg.unclip_ratio),
        **torch_device_params(device, cpu_fallback_hint="`--devices cpu`"),
    }

    download_keys: list[str] = []
    if det_path is None:
        download_keys.append(f"torch:det:{det_version}:{det_type}:ch")

    if not cfg.det_only:
        rec_version, rec_type, rec_path = resolve_torch_model(
            cfg.rec_model, cfg.rec_model_dir, "recognition", REC_MODELS
        )
        if rec_path is None:
            download_keys.append(f"torch:rec:{rec_version}:{rec_type}:ch")
        params.update(
            {
                "Rec.engine_type": rapidocr.EngineType.TORCH,
                "Rec.ocr_version": rapidocr.OCRVersion(rec_version),
                "Rec.model_type": rapidocr.ModelType(rec_type),
                "Rec.lang_type": "ch",
                "Rec.model_path": str(rec_path) if rec_path is not None else None,
                "Rec.rec_batch_num": int(cfg.rec_batch_size),
            }
        )
        if rec_path is not None:
            dict_path = resolve_rec_dict_path(rec_path, rec_version)
            if dict_path is not None:
                params["Rec.rec_keys_path"] = str(dict_path)

    if cfg.use_cls:
        cls_version, cls_type, cls_path = resolve_torch_model(
            cfg.cls_model, cfg.cls_model_dir, "orientation", CLS_MODELS
        )
        if cls_path is None:
            download_keys.append(f"torch:cls:{cls_version}:{cls_type}")
        params.update(
            {
                "Cls.engine_type": rapidocr.EngineType.TORCH,
                "Cls.ocr_version": rapidocr.OCRVersion(cls_version),
                "Cls.model_type": rapidocr.ModelType(cls_type),
                "Cls.model_path": str(cls_path) if cls_path is not None else None,
                "Cls.cls_batch_num": int(cfg.rec_batch_size),
            }
        )

    return params, tuple(download_keys), model_root


def _prefetch_models_before_spawn(cfg: DetectConfig) -> None:
    """Pre-download any uncached PyTorch weights & dicts once on CPU before spawning workers."""
    rapidocr = import_rapidocr(context="Offline text-line detection")
    params, download_keys, model_root = build_rapidocr_params(cfg, "cpu", rapidocr)
    if download_keys:
        ensure_rapidocr_models_ready(
            params,
            rapidocr=rapidocr,
            model_root=model_root,
            download_keys=download_keys,
        )


def _create_rapidocr_engine(cfg: DetectConfig, device: str, *, skip_download_lock: bool = False) -> Any:
    if cfg.intra_threads is not None:
        if isinstance(cfg.intra_threads, bool) or int(cfg.intra_threads) < 1:
            raise ValueError(
                f"intra_threads must be a positive integer, got {cfg.intra_threads!r}."
            )
        torch.set_num_threads(int(cfg.intra_threads))
    rapidocr = import_rapidocr(context="Offline text-line detection")
    params, download_keys, model_root = build_rapidocr_params(cfg, device, rapidocr)
    return create_rapidocr_engine(
        params,
        rapidocr=rapidocr,
        model_root=model_root,
        download_keys=() if skip_download_lock else download_keys,
    )


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
        engine = _create_rapidocr_engine(cfg, device, skip_download_lock=True)
    except BaseException:
        result_queue.put(("error", worker_id, device, traceback.format_exc()))
        return

    while True:
        try:
            task = task_queue.get()
        except (EOFError, OSError):
            result_queue.put(("error", worker_id, device, "Task queue closed unexpectedly."))
            return
        if task is None:
            result_queue.put(("done", worker_id, device, None))
            return
        chunk_out: list[tuple[int, list[dict[str, Any]], int, bool]] = []
        try:
            for sample_idx, gt_path_str in task:
                lines, dropped, had_err = _process_one_image(engine, Path(gt_path_str), cfg)
                chunk_out.append((sample_idx, lines, dropped, had_err))
        except BaseException:
            result_queue.put(("error", worker_id, device, traceback.format_exc()))
            return
        result_queue.put(("ok", worker_id, device, chunk_out))


def _load_checkpoint(
    ckpt_path: Path,
    *,
    expected_fingerprint: str,
    gt_entries: Sequence[str],
) -> tuple[dict[int, tuple[list[dict[str, Any]], int, bool]], int, int]:
    """Load completed per-sample results from a sidecar JSONL checkpoint.

    Validates the run fingerprint and per-sample ``gt`` path, and safely ignores a torn
    final line caused by a crash mid-write.
    """
    raw_lines = ckpt_path.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        return {}, 0, 0

    try:
        header = json.loads(raw_lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Corrupted checkpoint header in {ckpt_path}; pass --no-resume to restart fresh."
        ) from exc

    if not isinstance(header, dict) or header.get("type") != "header":
        raise ValueError(
            f"Invalid checkpoint header in {ckpt_path}; pass --no-resume to restart fresh."
        )
    if header.get("version") != _CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {header.get('version')!r} in {ckpt_path}; "
            "pass --no-resume to restart fresh."
        )
    if header.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            f"Checkpoint {ckpt_path} does not match current input manifest or OCR parameters "
            "(fingerprint mismatch). Pass --no-resume to overwrite the checkpoint or choose a "
            "different --output-json."
        )

    restored: dict[int, tuple[list[dict[str, Any]], int, bool]] = {}
    num_samples = len(gt_entries)
    total_lines = len(raw_lines)
    for line_no in range(1, total_lines):
        raw_line = raw_lines[line_no].strip()
        if not raw_line:
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            if line_no == total_lines - 1:
                _LOGGER.warning(
                    "[Detect] Ignoring torn trailing line %d in checkpoint %s; that sample will be re-run.",
                    line_no + 1,
                    ckpt_path,
                )
                break
            raise ValueError(
                f"Corrupted JSON on line {line_no + 1} of checkpoint {ckpt_path}; "
                "pass --no-resume to restart fresh."
            ) from None

        if not isinstance(record, dict) or record.get("type") != "result":
            continue
        idx = record.get("idx")
        gt_val = record.get("gt")
        text_lines = record.get("text_lines")
        if (
            isinstance(idx, bool)
            or not isinstance(idx, int)
            or not (0 <= idx < num_samples)
            or gt_val != gt_entries[idx]
            or not isinstance(text_lines, list)
        ):
            raise ValueError(
                f"Inconsistent checkpoint record on line {line_no + 1} of {ckpt_path} "
                f"(idx={idx!r}, gt={gt_val!r}); pass --no-resume to restart fresh."
            )
        dropped = int(record.get("dropped_boxes", 0))
        had_err = bool(record.get("had_error", False))
        restored[idx] = (text_lines, dropped, had_err)

    dropped_boxes = sum(entry[1] for entry in restored.values())
    skipped_errors = sum(int(entry[2]) for entry in restored.values())
    return restored, dropped_boxes, skipped_errors


def _init_checkpoint_file(
    ckpt_path: Path,
    *,
    fingerprint: str,
    total_samples: int,
    restored_results: dict[int, tuple[list[dict[str, Any]], int, bool]] | None = None,
    gt_entries: Sequence[str] = (),
) -> Any:
    """Create or reopen the JSONL checkpoint file in append mode with a clean header."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "header",
        "version": _CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "total_samples": total_samples,
    }
    # Rewriting compacted valid records atomically on resume also truncates any torn trailing line.
    fd, tmp_name = tempfile.mkstemp(
        dir=ckpt_path.parent,
        prefix=f".{ckpt_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            if restored_results:
                for idx in sorted(restored_results):
                    lines, dropped, had_err = restored_results[idx]
                    rec = {
                        "type": "result",
                        "idx": idx,
                        "gt": gt_entries[idx],
                        "text_lines": lines,
                        "dropped_boxes": dropped,
                        "had_error": had_err,
                    }
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, ckpt_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    return ckpt_path.open("a", encoding="utf-8")


def _append_checkpoint_records(
    ckpt_handle: Any,
    records: Sequence[tuple[int, str, list[dict[str, Any]], int, bool]],
) -> None:
    if ckpt_handle is None or not records:
        return
    for sample_idx, raw_gt, lines, dropped, had_err in records:
        payload = {
            "type": "result",
            "idx": sample_idx,
            "gt": raw_gt,
            "text_lines": lines,
            "dropped_boxes": dropped,
            "had_error": had_err,
        }
        ckpt_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    ckpt_handle.flush()


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


def _drain_multiprocess_workers(
    procs: Sequence[Any],
    active_devices: Sequence[str],
    result_q: Any,
    *,
    total_chunks: int,
    on_chunk: Any,
) -> None:
    """Consume worker results while failing fast on crashes OR premature exitcode==0 exits."""
    completed_chunks = 0
    finished_workers: set[int] = set()
    num_workers = len(procs)

    def _handle_msg(msg: tuple[Any, ...]) -> None:
        nonlocal completed_chunks
        status = msg[0]
        if status == "error":
            _, wid, dev, tb_text = msg
            raise RuntimeError(f"RapidOCR worker {wid} on {dev!r} failed:\n{tb_text}")
        if status == "done":
            _, wid, _, _ = msg
            finished_workers.add(int(wid))
            return
        if status == "ok":
            _, _, _, chunk_out = msg
            on_chunk(chunk_out)
            completed_chunks += 1
            return
        raise RuntimeError(f"Unexpected worker message: {msg!r}.")

    try:
        while completed_chunks < total_chunks:
            try:
                msg = result_q.get(timeout=0.5)
            except queue.Empty:
                # Re-check queue non-blockingly before declaring a worker prematurely dead to
                # avoid racing between a worker enqueuing its last message and exiting.
                try:
                    msg = result_q.get_nowait()
                except queue.Empty:
                    failed_nonzero = [
                        (wid, active_devices[wid], proc.exitcode)
                        for wid, proc in enumerate(procs)
                        if not proc.is_alive() and proc.exitcode not in (0, None)
                    ]
                    if failed_nonzero:
                        raise RuntimeError(
                            f"RapidOCR worker(s) crashed unexpectedly: {failed_nonzero}."
                        ) from None

                    premature_zero = [
                        (wid, active_devices[wid], proc.exitcode)
                        for wid, proc in enumerate(procs)
                        if wid not in finished_workers
                        and not proc.is_alive()
                        and proc.exitcode == 0
                    ]
                    if premature_zero:
                        raise RuntimeError(
                            f"RapidOCR worker(s) exited prematurely with exitcode=0 before "
                            f"signaling completion ({completed_chunks}/{total_chunks} chunks done): "
                            f"{premature_zero}."
                        ) from None

                    if len(finished_workers) >= num_workers or all(
                        not proc.is_alive() for proc in procs
                    ):
                        raise RuntimeError(
                            f"All RapidOCR workers exited with incomplete progress "
                            f"({completed_chunks}/{total_chunks} chunks done)."
                        ) from None
                    continue

            _handle_msg(msg)

        # Drain remaining "done" messages so workers finish cleanly without blocking on queue pipe.
        deadline = time.monotonic() + 30.0
        while len(finished_workers) < num_workers and time.monotonic() < deadline:
            try:
                msg = result_q.get(timeout=0.2)
            except queue.Empty:
                if all(not proc.is_alive() for proc in procs):
                    break
                continue
            _handle_msg(msg)
    except BaseException:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5.0)
        raise
    else:
        for wid, proc in enumerate(procs):
            proc.join(timeout=15.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
                raise RuntimeError(
                    f"RapidOCR worker {wid} (PID={proc.pid}) failed to exit cleanly after work finished."
                )
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"RapidOCR worker {wid} (PID={proc.pid}) exited with code {proc.exitcode}."
                )


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
    resume: bool = True,
    checkpoint_file: str | Path | None = None,
    keep_checkpoint: bool = False,
    engine_factory: Any = None,
) -> DetectSummary:
    """Run offline RapidOCR detection/recognition on every ``gt`` in a manifest JSON.

    Existing ``text_lines`` fields are overwritten; entries without ``text_lines`` receive a new
    ``text_lines`` list. Progress is flushed incrementally to a sidecar JSONL checkpoint so an
    interrupted run automatically resumes when re-invoked with ``resume=True`` (default).
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
    ckpt_path = (
        Path(checkpoint_file)
        if checkpoint_file is not None
        else default_checkpoint_path(out_path)
    )
    if not in_path.is_file():
        raise FileNotFoundError(f"Input manifest JSON not found: {in_path}")

    raw_data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(
            f"Input manifest {in_path} must be a JSON list of dicts, got {type(raw_data).__name__}."
        )

    resolved_root = Path(root_dir) if root_dir is not None else None
    all_tasks: list[tuple[int, str]] = []
    gt_entries: list[str] = []
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
        gt_entries.append(str(entry["gt"]))
        all_tasks.append((idx, str(gt_path)))
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

    num_samples = len(all_tasks)
    start_time = time.perf_counter()

    if num_samples == 0:
        _atomic_write_json(out_path, [], indent=indent)
        if not keep_checkpoint:
            ckpt_path.unlink(missing_ok=True)
        return DetectSummary(
            total_samples=0,
            overwritten_samples=0,
            added_samples=0,
            resumed_samples=0,
            processed_samples=0,
            total_lines=0,
            empty_samples=0,
            skipped_errors=0,
            dropped_boxes=0,
            devices=tuple(worker_devices),
            elapsed_sec=max(time.perf_counter() - start_time, 1e-9),
            output_json=out_path,
            checkpoint_path=ckpt_path if keep_checkpoint else None,
        )

    fingerprint = _compute_run_fingerprint(gt_entries, cfg)
    results_by_idx: list[list[dict[str, Any]] | None] = [None] * num_samples
    restored_map: dict[int, tuple[list[dict[str, Any]], int, bool]] = {}
    dropped_boxes = 0
    skipped_errors = 0

    if resume and ckpt_path.is_file():
        restored_map, dropped_boxes, skipped_errors = _load_checkpoint(
            ckpt_path,
            expected_fingerprint=fingerprint,
            gt_entries=gt_entries,
        )
        for idx, (lines, _, _) in restored_map.items():
            results_by_idx[idx] = lines
        _LOGGER.info(
            "[Detect] Auto-resumed %d/%d completed samples from checkpoint %s.",
            len(restored_map),
            num_samples,
            ckpt_path,
        )
    elif not resume and ckpt_path.is_file():
        ckpt_path.unlink(missing_ok=True)

    pending_tasks = [task for task in all_tasks if results_by_idx[task[0]] is None]
    resumed_samples = len(restored_map)
    processed_samples = len(pending_tasks)

    _LOGGER.info(
        "[Detect] Starting offline RapidOCR (%s) | total=%d | resumed=%d | pending=%d | "
        "workers=%d (%s) | input=%s",
        "det-only" if cfg.det_only else "det+rec",
        num_samples,
        resumed_samples,
        processed_samples,
        len(worker_devices),
        ", ".join(worker_devices),
        in_path,
    )

    progress_bar: Any = None
    try:
        from tqdm import tqdm

        progress_bar = tqdm(
            total=num_samples,
            initial=resumed_samples,
            desc="Detecting text_lines",
            unit="img",
        )
    except Exception:
        progress_bar = None

    ckpt_handle = _init_checkpoint_file(
        ckpt_path,
        fingerprint=fingerprint,
        total_samples=num_samples,
        restored_results=restored_map,
        gt_entries=gt_entries,
    )
    try:
        if processed_samples == 0:
            pass
        elif len(worker_devices) == 1 or processed_samples == 1:
            dev = worker_devices[0]
            engine = (
                engine_factory(cfg, dev)
                if engine_factory is not None
                else _create_rapidocr_engine(cfg, dev)
            )
            for sample_idx, gt_path_str in pending_tasks:
                lines, dropped, had_err = _process_one_image(engine, Path(gt_path_str), cfg)
                results_by_idx[sample_idx] = lines
                dropped_boxes += dropped
                skipped_errors += int(had_err)
                _append_checkpoint_records(
                    ckpt_handle,
                    [(sample_idx, gt_entries[sample_idx], lines, dropped, had_err)],
                )
                if progress_bar is not None:
                    progress_bar.update(1)
        else:
            # Pre-download any missing PyTorch weights & dicts once on CPU so all workers initialize
            # concurrently without contending on WeakFileLock.
            _prefetch_models_before_spawn(cfg)

            chunks = [
                pending_tasks[i : i + chunk_size]
                for i in range(0, processed_samples, chunk_size)
            ]
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

            def _on_chunk(chunk_out: Sequence[tuple[int, list[dict[str, Any]], int, bool]]) -> None:
                nonlocal dropped_boxes, skipped_errors
                ckpt_batch: list[tuple[int, str, list[dict[str, Any]], int, bool]] = []
                for sample_idx, lines, dropped, had_err in chunk_out:
                    results_by_idx[sample_idx] = lines
                    dropped_boxes += dropped
                    skipped_errors += int(had_err)
                    ckpt_batch.append(
                        (sample_idx, gt_entries[sample_idx], lines, dropped, had_err)
                    )
                _append_checkpoint_records(ckpt_handle, ckpt_batch)
                if progress_bar is not None:
                    progress_bar.update(len(chunk_out))

            _drain_multiprocess_workers(
                procs,
                active_devices,
                result_q,
                total_chunks=len(chunks),
                on_chunk=_on_chunk,
            )
    finally:
        ckpt_handle.close()
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
    if not keep_checkpoint:
        ckpt_path.unlink(missing_ok=True)

    elapsed = max(time.perf_counter() - start_time, 1e-9)
    _LOGGER.info(
        "[Detect] Done: %d samples (resumed=%d, processed=%d, lines=%d, empty=%d, dropped=%d, "
        "errors=%d) in %.2fs (%.2f img/s) | overwritten=%d added=%d | saved=%s",
        num_samples,
        resumed_samples,
        processed_samples,
        total_lines,
        empty_samples,
        dropped_boxes,
        skipped_errors,
        elapsed,
        processed_samples / elapsed if processed_samples > 0 else 0.0,
        overwritten,
        added,
        out_path,
    )
    return DetectSummary(
        total_samples=num_samples,
        overwritten_samples=overwritten,
        added_samples=added,
        resumed_samples=resumed_samples,
        processed_samples=processed_samples,
        total_lines=total_lines,
        empty_samples=empty_samples,
        skipped_errors=skipped_errors,
        dropped_boxes=dropped_boxes,
        devices=tuple(worker_devices),
        elapsed_sec=elapsed,
        output_json=out_path,
        checkpoint_path=ckpt_path if keep_checkpoint else None,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline multi-device RapidOCR text-line detection & recognition for SRTrainer JSON "
            "manifests with incremental JSONL checkpointing and automatic resume."
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
            "Comma-separated device list, e.g. 'cuda:0,cuda:1', 'npu:0,npu:1', '0,1,2,3', 'mps', "
            "or 'cpu'. Omit to auto-use all available GPUs/NPUs/MPS (or CPU)."
        ),
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="auto",
        choices=("auto", "cuda", "gpu", "npu", "mps", "cpu"),
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
        "--checkpoint-file",
        type=Path,
        default=None,
        help="Optional path for incremental JSONL progress checkpoint (default: <output-json>.ckpt.jsonl).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore and discard any existing checkpoint file and start from scratch.",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep the sidecar JSONL checkpoint file after the final output JSON is written.",
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
        "--weights-dir",
        type=Path,
        default=None,
        help=(
            "Optional weights root directory holding the RapidOCR PyTorch det/rec/cls .pth files "
            "(default: <repo>/experiments/weights, i.e. <repo>/experiments/weights/rapidocr "
            f"for RapidOCR; also settable via ${WEIGHTS_DIR_ENV_VAR})."
        ),
    )
    parser.add_argument(
        "--det-model-dir",
        type=str,
        default=None,
        help=(
            "Optional local detection PyTorch .pth file or directory containing it "
            "(default: experiments/weights/rapidocr/ch_PP-OCRv5_det_server.pth when present)."
        ),
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
        help=(
            "Optional local recognition PyTorch .pth file or directory containing it (with "
            "dict.txt/ppocrv5_dict.txt). Default: "
            "experiments/weights/rapidocr/ch_PP-OCRv5_rec_server.pth when present."
        ),
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
        help=(
            "Optional local orientation PyTorch .pth file or directory containing it "
            "(default: experiments/weights/rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.pth when "
            "present)."
        ),
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
        help="Optional PyTorch CPU intra-op thread count (torch.set_num_threads) per worker process.",
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
    if args.weights_dir is not None:
        os.environ[WEIGHTS_DIR_ENV_VAR] = str(Path(args.weights_dir).expanduser())
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
        resume=not bool(args.no_resume),
        checkpoint_file=args.checkpoint_file,
        keep_checkpoint=bool(args.keep_checkpoint),
    )


if __name__ == "__main__":
    main()
