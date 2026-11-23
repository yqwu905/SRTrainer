"""RapidOCR text recognition evaluation metrics: OCR-A, CER, and 1-NED.

Aligned with ``yqwu905/VOSR`` ``evaluate.py`` (commit ``9fbcf8fe22b399a6fbb41e011748322363ae9ee0``)
for NFKC/whitespace text normalization, unit-cost Unicode Levenshtein distance,
OpenCV ``INTER_CUBIC``/``BORDER_REPLICATE`` perspective cropping with 1.5x vertical rotation,
and exact dataset-level weighted reduction via :meth:`BaseMetric.compute_stats`.

Differences documented by design:
- ``ocr_device`` defaults to ``"cpu"`` independently of the training device.
- OpenCV, RapidOCR, and ONNX Runtime are optional dependencies imported lazily.
- Same-process metrics sharing identical OCR configuration share one recognizer instance
  and a bounded 1-sample recognition cache keyed by image/annotation/config digest.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import os
import threading
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.text_geometry import canonical_quad

from .base import BaseMetric

__all__ = [
    "DEFAULT_OCR_MODEL",
    "IGNORED_TRANSCRIPTIONS",
    "normalize_text",
    "edit_distance",
    "rectified_corners",
    "crop_text_region",
    "RapidOCRTextRecognizer",
    "PaddleTextRecognizer",
    "clear_ocr_caches",
    "OCRAccuracyMetric",
    "CERMetric",
    "OneMinusNEDMetric",
]

DEFAULT_OCR_MODEL: str = "PP-OCRv5_server_rec"
IGNORED_TRANSCRIPTIONS: frozenset[str] = frozenset({"###", "*"})

_TRANSCRIPT_KEYS: tuple[str, ...] = ("transcript", "transcription", "text")
_GEOMETRY_KEYS: tuple[str, ...] = ("points", "quad", "bbox", "box")
_MAX_RUNTIME_CACHE_ENTRIES: int = 16


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}: {value!r}.")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _validate_str_option(
    value: Any,
    name: str,
    *,
    required: bool = False,
    allow_path: bool = False,
) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{name} must be a non-empty string, got None.")
        return None
    if allow_path and isinstance(value, Path):
        text = str(value).strip()
    elif isinstance(value, str):
        text = value.strip()
    else:
        expected = "a path string, Path, or None" if allow_path else "a string"
        raise ValueError(f"{name} must be {expected}, got {value!r}.")
    if not text:
        raise ValueError(f"{name} must be non-empty, got {value!r}.")
    return text


def normalize_text(text: str, ignore_case: bool = False) -> str:
    """Apply Unicode NFKC normalization, remove all Unicode whitespace, and optionally lowercase."""
    if not isinstance(text, str):
        raise TypeError(f"Expected text to be a str, got {type(text).__name__}.")
    _require_bool(ignore_case, "ignore_case")
    normalized = "".join(unicodedata.normalize("NFKC", text).split())
    return normalized.lower() if ignore_case else normalized


def edit_distance(first: str, second: str) -> int:
    """Compute unit-cost Levenshtein distance over Unicode code points."""
    if not isinstance(first, str) or not isinstance(second, str):
        raise TypeError(
            "edit_distance expects two str arguments, got "
            f"{type(first).__name__} and {type(second).__name__}."
        )
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row_idx, char_first in enumerate(first, 1):
        current = [row_idx]
        for col_idx, char_second in enumerate(second, 1):
            current.append(
                min(
                    previous[col_idx] + 1,
                    current[col_idx - 1] + 1,
                    previous[col_idx - 1] + (char_first != char_second),
                )
            )
        previous = current
    return previous[-1]


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OCR text-box cropping requires OpenCV (`cv2`). Install optional OCR dependencies "
            "with `uv sync --extra ocr` (or `pip install opencv-python-headless`)."
        ) from exc
    return cv2


def _import_rapidocr() -> tuple[Any, Any]:
    try:
        import onnxruntime
        import rapidocr
    except ImportError as exc:
        raise ImportError(
            "OCR metrics require RapidOCR and ONNX Runtime. Install optional OCR dependencies "
            "with `uv sync --extra ocr`."
        ) from exc
    return rapidocr, onnxruntime


def _contains_bool(obj: Any) -> bool:
    if isinstance(obj, (bool, np.bool_)):
        return True
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.bool_:
            return True
        return obj.dtype == object and any(_contains_bool(item) for item in obj.flat)
    if isinstance(obj, (list, tuple)):
        return any(_contains_bool(item) for item in obj)
    return False


def _polygon_area(points: np.ndarray) -> float:
    coords = np.asarray(points, dtype=np.float64)
    if coords.ndim != 2 or len(coords) < 3 or coords.shape[1] != 2:
        return 0.0
    cross_sum = np.sum(
        coords[:, 0] * np.roll(coords[:, 1], -1) - coords[:, 1] * np.roll(coords[:, 0], -1)
    )
    return abs(float(cross_sum)) * 0.5


def _orientation(pt_a: np.ndarray, pt_b: np.ndarray, pt_c: np.ndarray) -> float:
    return float((pt_b[0] - pt_a[0]) * (pt_c[1] - pt_a[1]) - (pt_b[1] - pt_a[1]) * (pt_c[0] - pt_a[0]))


def _on_segment(pt_a: np.ndarray, pt_b: np.ndarray, probe: np.ndarray, eps: float = 1e-8) -> bool:
    return (
        min(float(pt_a[0]), float(pt_b[0])) - eps
        <= float(probe[0])
        <= max(float(pt_a[0]), float(pt_b[0])) + eps
        and min(float(pt_a[1]), float(pt_b[1])) - eps
        <= float(probe[1])
        <= max(float(pt_a[1]), float(pt_b[1])) + eps
    )


def _segments_intersect(
    seg1_start: np.ndarray,
    seg1_end: np.ndarray,
    seg2_start: np.ndarray,
    seg2_end: np.ndarray,
    eps: float = 1e-8,
) -> bool:
    ori_1 = _orientation(seg1_start, seg1_end, seg2_start)
    ori_2 = _orientation(seg1_start, seg1_end, seg2_end)
    ori_3 = _orientation(seg2_start, seg2_end, seg1_start)
    ori_4 = _orientation(seg2_start, seg2_end, seg1_end)

    if ((ori_1 > eps and ori_2 < -eps) or (ori_1 < -eps and ori_2 > eps)) and (
        (ori_3 > eps and ori_4 < -eps) or (ori_3 < -eps and ori_4 > eps)
    ):
        return True
    return (
        (abs(ori_1) <= eps and _on_segment(seg1_start, seg1_end, seg2_start, eps))
        or (abs(ori_2) <= eps and _on_segment(seg1_start, seg1_end, seg2_end, eps))
        or (abs(ori_3) <= eps and _on_segment(seg2_start, seg2_end, seg1_start, eps))
        or (abs(ori_4) <= eps and _on_segment(seg2_start, seg2_end, seg1_end, eps))
    )


def _validate_polygon_points(points: np.ndarray, name: str) -> np.ndarray:
    """Validate an N>=4 polygon (quad via ``canonical_quad``, N>4 as non-degenerate simple polygon)."""
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 4:
        raise ValueError(
            f"{name}: expected polygon of shape (N, 2) with N >= 4, got shape {tuple(points.shape)}."
        )
    if not np.isfinite(points).all():
        raise ValueError(f"{name}: polygon coordinates must be finite.")
    if points.shape[0] == 4:
        return canonical_quad(points, name=name)
    if _polygon_area(points) < 1e-6:
        raise ValueError(f"{name}: polygon must be non-degenerate with positive area.")

    num_vertices = points.shape[0]
    for first_idx in range(num_vertices):
        seg1_start = points[first_idx]
        seg1_end = points[(first_idx + 1) % num_vertices]
        if float(np.linalg.norm(seg1_end - seg1_start)) < 1e-8:
            continue
        for second_idx in range(first_idx + 2, num_vertices):
            if first_idx == 0 and second_idx == num_vertices - 1:
                continue
            seg2_start = points[second_idx]
            seg2_end = points[(second_idx + 1) % num_vertices]
            if float(np.linalg.norm(seg2_end - seg2_start)) < 1e-8:
                continue
            if _segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
                raise ValueError(f"{name}: polygon must not be self-intersecting (crossed).")
    return points


def _parse_line_geometry(line: Mapping[str, Any], context: str) -> np.ndarray:
    """Extract and strictly validate one text-line geometry from ``points``/``quad``/``bbox``/``box``."""
    geom_key = next((key for key in _GEOMETRY_KEYS if key in line), None)
    if geom_key is None:
        raise ValueError(f"{context}: missing geometry field; expected one of {_GEOMETRY_KEYS}.")
    raw_geom = line[geom_key]
    if raw_geom is None or isinstance(raw_geom, (str, bytes, Mapping)) or _contains_bool(raw_geom):
        raise ValueError(f"{context}.{geom_key}: invalid geometry value {raw_geom!r}.")
    try:
        coords = np.asarray(raw_geom, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}.{geom_key}: geometry must be numeric.") from exc

    if geom_key in ("bbox", "box"):
        if coords.shape != (4,):
            raise ValueError(
                f"{context}.{geom_key}: expected xyxy [x0, y0, x1, y1] of shape (4,), "
                f"got shape {tuple(coords.shape)}."
            )
        return canonical_quad(coords, name=f"{context}.{geom_key}")
    if geom_key == "quad":
        if coords.shape not in ((4, 2), (8,)):
            raise ValueError(
                f"{context}.quad: expected 4x2 cyclic corners or 8 floats, "
                f"got shape {tuple(coords.shape)}."
            )
        return canonical_quad(coords, name=f"{context}.quad")
    return _validate_polygon_points(coords, name=f"{context}.points")


def _parse_annotation_size(meta: Mapping[str, Any]) -> tuple[float, float] | None:
    raw_size = meta.get("annotation_size")
    if raw_size is None:
        return None
    if isinstance(raw_size, (str, bytes, Mapping)):
        raise ValueError(
            f"meta['annotation_size'] must be a 2-element [width, height] sequence, got {raw_size!r}."
        )
    if isinstance(raw_size, np.ndarray):
        if raw_size.ndim != 1 or raw_size.shape[0] != 2 or _contains_bool(raw_size):
            raise ValueError(
                f"meta['annotation_size'] must be a 2-element [width, height] sequence, got {raw_size!r}."
            )
        size_items = raw_size.tolist()
    elif isinstance(raw_size, Sequence):
        if len(raw_size) != 2:
            raise ValueError(
                f"meta['annotation_size'] must have length 2 ([width, height]), got length {len(raw_size)}."
            )
        size_items = list(raw_size)
    else:
        raise ValueError(
            f"meta['annotation_size'] must be a 2-element [width, height] sequence, "
            f"got {type(raw_size).__name__}."
        )

    dims: list[float] = []
    for axis_name, val in zip(("width", "height"), size_items, strict=True):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, numbers.Real):
            raise ValueError(
                f"meta['annotation_size'] {axis_name} must be a positive finite number, got {val!r}."
            )
        float_val = float(val)
        if not math.isfinite(float_val) or float_val <= 0.0:
            raise ValueError(
                f"meta['annotation_size'] {axis_name} must be positive and finite, got {val!r}."
            )
        dims.append(float_val)
    return (dims[0], dims[1])


def _scale_points(
    points: np.ndarray,
    src_size: tuple[float, float],
    dst_size: tuple[int, int],
) -> np.ndarray:
    scale_x = float(dst_size[0]) / float(src_size[0])
    scale_y = float(dst_size[1]) / float(src_size[1])
    return np.asarray(points, dtype=np.float32) * np.float32([scale_x, scale_y])


def rectified_corners(points: Any) -> np.ndarray:
    """Return clockwise ``(4, 2)`` ``float32`` crop corners matching VOSR ``evaluate.py``.

    4-point quadrilaterals are made clockwise while preserving vertex 0; polygons with >4
    vertices use OpenCV's minimum-area rectangle ordered ``[TL, TR, BR, BL]``.
    """
    if isinstance(points, (str, bytes, Mapping)) or _contains_bool(points):
        raise ValueError(f"Invalid polygon points: {points!r}.")
    pts_f64 = _validate_polygon_points(np.asarray(points, dtype=np.float64), name="points")
    pts_f32 = pts_f64.astype(np.float32)

    if len(pts_f32) == 4:
        signed_area2 = float(
            np.sum(
                pts_f32[:, 0] * np.roll(pts_f32[:, 1], -1)
                - np.roll(pts_f32[:, 0], -1) * pts_f32[:, 1]
            )
        )
        if signed_area2 < 0.0:
            pts_f32 = pts_f32[[0, 3, 2, 1]]
        return np.ascontiguousarray(pts_f32, dtype=np.float32)

    cv2 = _import_cv2()
    rect = cv2.minAreaRect(pts_f32)
    rect_width, rect_height = float(rect[1][0]), float(rect[1][1])
    if (
        not (math.isfinite(rect_width) and math.isfinite(rect_height))
        or rect_width <= 1e-6
        or rect_height <= 1e-6
    ):
        raise ValueError("Polygon minimum-area rectangle is degenerate.")
    box = sorted(cv2.boxPoints(rect).tolist())
    left_pts = sorted(box[:2], key=lambda point: point[1])
    right_pts = sorted(box[2:], key=lambda point: point[1])
    return np.float32([left_pts[0], right_pts[0], right_pts[1], left_pts[1]])


def crop_text_region(bgr_image: np.ndarray, points: Any) -> np.ndarray:
    """Rectify one text region from a CPU uint8 BGR image matching VOSR ``evaluate.py``.

    Uses OpenCV ``INTER_CUBIC`` and ``BORDER_REPLICATE`` perspective warping and rotates
    crops with ``height / width >= 1.5`` by 90 degrees counter-clockwise.
    """
    if not isinstance(bgr_image, np.ndarray) or bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(
            "bgr_image must be a 3D [H, W, 3] numpy array, got "
            f"{type(bgr_image).__name__} with shape {getattr(bgr_image, 'shape', None)}."
        )
    if bgr_image.dtype != np.uint8:
        raise ValueError(f"bgr_image must have dtype uint8, got {bgr_image.dtype}.")
    if bgr_image.shape[0] < 1 or bgr_image.shape[1] < 1:
        raise ValueError(f"bgr_image must have positive spatial size, got {bgr_image.shape}.")

    cv2 = _import_cv2()
    corners = rectified_corners(points)
    width = max(
        1,
        int(max(np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[2] - corners[3]))),
    )
    height = max(
        1,
        int(max(np.linalg.norm(corners[0] - corners[3]), np.linalg.norm(corners[1] - corners[2]))),
    )
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(corners, target)
    crop = cv2.warpPerspective(
        bgr_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return np.ascontiguousarray(np.rot90(crop) if height / width >= 1.5 else crop)


def _pred_tensor_to_bgr_and_digest(pred: torch.Tensor) -> tuple[np.ndarray, bytes]:
    """Validate RGB ``[1, 3, H, W]`` float tensor, convert to CPU uint8 BGR, and hash content."""
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Expected pred to be a torch.Tensor, got {type(pred).__name__}.")
    if pred.ndim != 4 or pred.shape[0] != 1 or pred.shape[1] != 3:
        raise ValueError(
            f"OCR metrics expect a 4D RGB tensor [1, 3, H, W], got shape {tuple(pred.shape)}."
        )
    if pred.shape[2] < 1 or pred.shape[3] < 1:
        raise ValueError(
            f"OCR metrics expect positive spatial dimensions (H, W >= 1), got shape {tuple(pred.shape)}."
        )
    if not pred.is_floating_point():
        raise ValueError(
            f"OCR metrics expect a floating-point tensor in [0, 1], got dtype {pred.dtype}."
        )

    with torch.no_grad():
        cpu_f32 = pred[0].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(cpu_f32).all():
            raise ValueError("OCR metrics received pred tensor containing NaN or Inf values.")
        digest_src = cpu_f32 + 0.0
        pred_digest = hashlib.blake2b(
            memoryview(digest_src.numpy()).cast("B"), digest_size=32
        ).digest()
        rgb_u8 = (
            cpu_f32.clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
    return np.ascontiguousarray(rgb_u8[..., ::-1]), pred_digest


def _require_meta_bool(mapping: Mapping[str, Any], key: str, default: bool, context: str) -> bool:
    if key not in mapping:
        return default
    return _require_bool(mapping[key], f"{context}.{key}")


def _should_skip_annotation(
    mapping: Mapping[str, Any],
    context: str,
    raw_text: str,
    norm_label: str,
    *,
    image_alignment_valid: bool,
    include_difficult: bool,
    check_alignment: bool,
) -> bool:
    is_valid = _require_meta_bool(mapping, "valid", True, context)
    is_ignored = _require_meta_bool(mapping, "ignore", False, context)
    is_difficult = _require_meta_bool(mapping, "difficult", False, context)
    line_aligned = (
        _require_meta_bool(mapping, "alignment_valid", True, context) if check_alignment else True
    )
    return (
        (not image_alignment_valid)
        or (not is_valid)
        or is_ignored
        or (not line_aligned)
        or (is_difficult and not include_difficult)
        or (raw_text.strip() in IGNORED_TRANSCRIPTIONS)
        or (not norm_label)
    )


@dataclass(frozen=True)
class _OCRInstance:
    raw_label: str
    norm_label: str
    points: np.ndarray | None


def _extract_ocr_instances(
    meta: Mapping[str, Any] | None,
    *,
    pred_size: tuple[int, int],
    label_key: str,
    ignore_case: bool,
    include_difficult: bool,
) -> tuple[str, tuple[float, float] | None, list[_OCRInstance]]:
    """Parse ``meta`` into valid OCR targets according to the single-line / ``text_lines`` contract."""
    if meta is None or not isinstance(meta, Mapping):
        raise ValueError(
            "OCR metrics require a metadata dict `meta` containing 'text_lines' or "
            f"{label_key!r}."
        )

    annotation_size = _parse_annotation_size(meta)
    image_alignment_valid = _require_meta_bool(meta, "alignment_valid", True, "meta")

    if "text_lines" in meta:
        raw_lines = meta["text_lines"]
        if not isinstance(raw_lines, list):
            raise ValueError(
                f"meta['text_lines'] must be a list, got {type(raw_lines).__name__}."
            )
        instances: list[_OCRInstance] = []
        for idx, line in enumerate(raw_lines):
            ctx = f"meta.text_lines[{idx}]"
            if not isinstance(line, Mapping):
                raise ValueError(f"{ctx} must be a dict, got {type(line).__name__}.")

            transcript_key = next((key for key in _TRANSCRIPT_KEYS if key in line), None)
            if transcript_key is None:
                raise ValueError(
                    f"{ctx}: missing transcript field; expected one of {_TRANSCRIPT_KEYS}."
                )
            raw_text = line[transcript_key]
            if not isinstance(raw_text, str):
                raise ValueError(
                    f"{ctx}.{transcript_key} must be a str, got {type(raw_text).__name__}: {raw_text!r}."
                )

            pts = _parse_line_geometry(line, context=ctx)
            norm_label = normalize_text(raw_text, ignore_case=ignore_case)
            if _should_skip_annotation(
                line,
                ctx,
                raw_text,
                norm_label,
                image_alignment_valid=image_alignment_valid,
                include_difficult=include_difficult,
                check_alignment=True,
            ):
                continue

            scaled_pts = (
                _scale_points(pts, annotation_size, pred_size)
                if annotation_size is not None
                else np.asarray(pts, dtype=np.float32)
            )
            instances.append(
                _OCRInstance(raw_label=raw_text, norm_label=norm_label, points=scaled_pts)
            )
        return "boxes", annotation_size, instances

    if label_key not in meta:
        raise ValueError(
            f"OCR metrics require meta to contain 'text_lines' or {label_key!r}; "
            f"available keys={sorted(meta.keys())}."
        )
    raw_label = meta[label_key]
    if not isinstance(raw_label, str):
        raise ValueError(
            f"meta[{label_key!r}] must be a str, got {type(raw_label).__name__}: {raw_label!r}."
        )
    norm_label = normalize_text(raw_label, ignore_case=ignore_case)
    if _should_skip_annotation(
        meta,
        "meta",
        raw_label,
        norm_label,
        image_alignment_valid=image_alignment_valid,
        include_difficult=include_difficult,
        check_alignment=False,
    ):
        return "whole_image", annotation_size, []
    return (
        "whole_image",
        annotation_size,
        [_OCRInstance(raw_label=raw_label, norm_label=norm_label, points=None)],
    )


_REC_MODELS = {
    "PP-OCRv4_server_rec": ("PP-OCRv4", "server"),
    "PP-OCRv4_mobile_rec": ("PP-OCRv4", "mobile"),
    DEFAULT_OCR_MODEL: ("PP-OCRv5", "server"),
    "PP-OCRv5_mobile_rec": ("PP-OCRv5", "mobile"),
}
_CLS_MODELS = {
    "PP-LCNet_x1_0_textline_ori": ("PP-OCRv5", "server"),
    "PP-LCNet_x0_25_textline_ori": ("PP-OCRv5", "mobile"),
}


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


def _onnx_device_params(device: str | None, onnxruntime: Any) -> dict[str, Any]:
    name, separator, device_id = (device or "cpu").partition(":")
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
    if provider not in onnxruntime.get_available_providers():
        raise ValueError(
            f"OCR device {device!r} requires {provider}. Install an ONNX Runtime build with "
            'that provider or set ocr_device="cpu".'
        )
    return {
        f"EngineConfig.onnxruntime.use_{flag}": True,
        f"EngineConfig.onnxruntime.{flag}_ep_cfg.device_id": int(device_id or 0),
    }


class RapidOCRTextRecognizer:
    """Batched RapidOCR text recognition on BGR crops with optional 0/180-deg orientation."""

    def __init__(
        self,
        model_name: str = DEFAULT_OCR_MODEL,
        *,
        model_dir: str | Path | None = None,
        cls_model_name: str | None = None,
        cls_model_dir: str | Path | None = None,
        device: str | None = "cpu",
        batch_size: int = 1,
    ) -> None:
        self.batch_size = _require_positive_int(batch_size, "batch_size")
        clean_model_name = _validate_str_option(model_name, "model_name", required=True)
        assert clean_model_name is not None
        if cls_model_dir is not None and cls_model_name is None:
            raise ValueError("cls_model_dir requires cls_model_name to be specified.")

        rec_version, rec_type, rec_path = _resolve_onnx_model(
            clean_model_name, model_dir, "recognition", _REC_MODELS
        )
        cls_model = (
            _resolve_onnx_model(cls_model_name, cls_model_dir, "orientation", _CLS_MODELS)
            if cls_model_name is not None
            else None
        )
        self.use_cls = cls_model is not None
        rapidocr, onnxruntime = _import_rapidocr()
        cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        model_root = cache_root / "srtrainer" / "rapidocr"
        params: dict[str, Any] = {
            "Global.use_det": False,
            "Global.use_cls": self.use_cls,
            "Global.model_root_dir": str(model_root),
            "Rec.engine_type": rapidocr.EngineType.ONNXRUNTIME,
            "Rec.ocr_version": rapidocr.OCRVersion(rec_version),
            "Rec.model_type": rapidocr.ModelType(rec_type),
            "Rec.lang_type": "ch",
            "Rec.model_path": str(rec_path) if rec_path is not None else None,
            "Rec.rec_batch_num": self.batch_size,
            **_onnx_device_params(device, onnxruntime),
        }
        if rec_path is not None:
            dict_path = rec_path.with_name("dict.txt")
            if dict_path.is_file():
                params["Rec.rec_keys_path"] = str(dict_path)
        needs_download = rec_path is None
        if cls_model is not None:
            cls_version, cls_type, cls_path = cls_model
            needs_download |= cls_path is None
            params.update(
                {
                    "Cls.engine_type": rapidocr.EngineType.ONNXRUNTIME,
                    "Cls.ocr_version": rapidocr.OCRVersion(cls_version),
                    "Cls.model_type": rapidocr.ModelType(cls_type),
                    "Cls.model_path": str(cls_path) if cls_path is not None else None,
                    "Cls.cls_batch_num": self.batch_size,
                    # Apply the classifier's top label regardless of its confidence.
                    "Cls.cls_thresh": -1.0,
                }
            )
        self.engine = rapidocr.RapidOCR(params=params)
        # RapidOCR downloads directly to the final path. Serialize the first call across ranks.
        self._download_lock = model_root / ".download.lock" if needs_download else None

    def __call__(self, crops: Sequence[np.ndarray]) -> list[tuple[str, float]]:
        crop_list = list(crops)
        if not crop_list:
            return []

        lock_path = self._download_lock
        if lock_path is not None:
            from huggingface_hub.utils import WeakFileLock

            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with WeakFileLock(lock_path):
                outputs = self._recognize(crop_list)
            self._download_lock = None
            return outputs
        return self._recognize(crop_list)

    def _recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        # Use the individual RapidOCR stages to keep the project's crop geometry, raw text,
        # and batching. A crop that decodes to an empty string still contributes to the metrics.
        expected_count = len(crops)
        if self.use_cls:
            crops, result = self.engine.cls_and_rotate(crops)
            if len(crops) != expected_count or len(result.cls_res) != expected_count:
                raise RuntimeError(
                    f"OCR orientation classifier returned {len(result.cls_res)} predictions "
                    f"and {len(crops)} images for {expected_count} crops."
                )
        result = self.engine.recognize_txt(crops)
        if len(result.txts) != expected_count or len(result.scores) != expected_count:
            raise RuntimeError(
                f"OCR recognizer returned {len(result.txts)} texts and {len(result.scores)} scores "
                f"for {expected_count} crops."
            )
        return [
            (text, float(score)) for text, score in zip(result.txts, result.scores, strict=True)
        ]


# Preserve the import used by callers of the former adapter.
PaddleTextRecognizer = RapidOCRTextRecognizer


@dataclass(frozen=True)
class _RecognizerConfig:
    ocr_model: str
    ocr_model_dir: str | None
    ocr_cls_model: str | None
    ocr_cls_model_dir: str | None
    ocr_device: str | None
    ocr_batch_size: int


class _SharedOCRRuntime:
    """Shared recognizer with a bounded 1-sample cache holding only hashes and text tuples."""

    def __init__(self, recognizer: Any) -> None:
        self.recognizer = recognizer
        self._lock = threading.Lock()
        self._last_key: tuple[Any, ...] | None = None
        self._last_texts: tuple[str, ...] | None = None

    def recognize_cached(
        self,
        cache_key: tuple[Any, ...],
        crop_builder: Callable[[], list[np.ndarray]],
        expected_count: int,
    ) -> tuple[str, ...]:
        with self._lock:
            if self._last_key == cache_key and self._last_texts is not None:
                return self._last_texts

        crops = crop_builder()
        if len(crops) != expected_count:
            raise RuntimeError(f"Expected {expected_count} text crops, built {len(crops)}.")

        raw_outputs = list(self.recognizer(crops))
        if len(raw_outputs) != expected_count:
            raise RuntimeError(
                f"OCR recognizer returned {len(raw_outputs)} predictions for "
                f"{expected_count} text crops."
            )

        texts: list[str] = []
        for idx, item in enumerate(raw_outputs):
            if isinstance(item, tuple) and len(item) >= 1 and isinstance(item[0], str):
                texts.append(item[0])
            elif isinstance(item, str):
                texts.append(item)
            else:
                raise RuntimeError(
                    f"OCR recognizer returned invalid prediction at index {idx}: {item!r}."
                )

        result = tuple(texts)
        with self._lock:
            self._last_key = cache_key
            self._last_texts = result
        return result

    def clear_sample_cache(self) -> None:
        with self._lock:
            self._last_key = None
            self._last_texts = None


_RUNTIME_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE: OrderedDict[tuple[Any, ...], _SharedOCRRuntime] = OrderedDict()


def _build_recognizer(config: _RecognizerConfig) -> RapidOCRTextRecognizer:
    return RapidOCRTextRecognizer(
        model_name=config.ocr_model,
        model_dir=config.ocr_model_dir,
        cls_model_name=config.ocr_cls_model,
        cls_model_dir=config.ocr_cls_model_dir,
        device=config.ocr_device,
        batch_size=config.ocr_batch_size,
    )


def _runtime_cache_key(config: _RecognizerConfig) -> tuple[Any, ...]:
    return (config, RapidOCRTextRecognizer)


def _get_shared_runtime(config: _RecognizerConfig) -> _SharedOCRRuntime:
    lookup_key = _runtime_cache_key(config)
    with _RUNTIME_CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(lookup_key)
        if runtime is not None:
            _RUNTIME_CACHE.move_to_end(lookup_key)
            return runtime

    recognizer = _build_recognizer(config)
    runtime = _SharedOCRRuntime(recognizer)
    with _RUNTIME_CACHE_LOCK:
        existing = _RUNTIME_CACHE.get(lookup_key)
        if existing is not None:
            _RUNTIME_CACHE.move_to_end(lookup_key)
            return existing
        _RUNTIME_CACHE[lookup_key] = runtime
        while len(_RUNTIME_CACHE) > _MAX_RUNTIME_CACHE_ENTRIES:
            _RUNTIME_CACHE.popitem(last=False)
        return runtime


def clear_ocr_caches() -> None:
    """Clear all cached OCR recognizers and bounded single-sample prediction caches."""
    with _RUNTIME_CACHE_LOCK:
        for runtime in _RUNTIME_CACHE.values():
            runtime.clear_sample_cache()
        _RUNTIME_CACHE.clear()


class _BaseOCRMetric(BaseMetric):
    """Shared base class for RapidOCR-backed text recognition metrics (OCR-A, CER, 1-NED)."""

    def __init__(
        self,
        ocr_model: str = DEFAULT_OCR_MODEL,
        ocr_model_dir: str | Path | None = None,
        ocr_cls_model: str | None = None,
        ocr_cls_model_dir: str | Path | None = None,
        ocr_device: str | None = "cpu",
        ocr_batch_size: int = 1,
        ignore_case: bool = False,
        include_difficult: bool = False,
        label_key: str = "label",
    ) -> None:
        resolved_model = _validate_str_option(ocr_model, "ocr_model", required=True)
        resolved_label_key = _validate_str_option(label_key, "label_key", required=True)
        assert resolved_model is not None and resolved_label_key is not None

        self.ocr_model: str = resolved_model
        self.ocr_model_dir: str | None = _validate_str_option(
            ocr_model_dir, "ocr_model_dir", allow_path=True
        )
        self.ocr_cls_model: str | None = _validate_str_option(ocr_cls_model, "ocr_cls_model")
        self.ocr_cls_model_dir: str | None = _validate_str_option(
            ocr_cls_model_dir, "ocr_cls_model_dir", allow_path=True
        )
        if self.ocr_cls_model_dir is not None and self.ocr_cls_model is None:
            raise ValueError("ocr_cls_model_dir requires ocr_cls_model to be specified.")

        self.ocr_device: str | None = _validate_str_option(ocr_device, "ocr_device")
        self.ocr_batch_size: int = _require_positive_int(ocr_batch_size, "ocr_batch_size")
        self.ignore_case: bool = _require_bool(ignore_case, "ignore_case")
        self.include_difficult: bool = _require_bool(include_difficult, "include_difficult")
        self.label_key: str = resolved_label_key

        self._recognizer_config = _RecognizerConfig(
            ocr_model=self.ocr_model,
            ocr_model_dir=self.ocr_model_dir,
            ocr_cls_model=self.ocr_cls_model,
            ocr_cls_model_dir=self.ocr_cls_model_dir,
            ocr_device=self.ocr_device,
            ocr_batch_size=self.ocr_batch_size,
        )

    def to(self, device: torch.device | str) -> _BaseOCRMetric:
        """Preserve the explicit ``ocr_device`` without triggering lazy initialization."""
        del device
        return self

    def clear_sample_cache(self) -> None:
        """Clear the bounded single-sample cache on the shared runtime without lazy init."""
        key = _runtime_cache_key(self._recognizer_config)
        with _RUNTIME_CACHE_LOCK:
            runtime = _RUNTIME_CACHE.get(key)
        if runtime is not None:
            runtime.clear_sample_cache()

    def _recognize_pairs(
        self,
        pred: torch.Tensor,
        meta: dict[str, Any] | None,
    ) -> list[tuple[str, str]]:
        bgr_u8, pred_digest = _pred_tensor_to_bgr_and_digest(pred)
        pred_height, pred_width = int(bgr_u8.shape[0]), int(bgr_u8.shape[1])

        mode, annotation_size, instances = _extract_ocr_instances(
            meta,
            pred_size=(pred_width, pred_height),
            label_key=self.label_key,
            ignore_case=self.ignore_case,
            include_difficult=self.include_difficult,
        )
        if not instances:
            return []

        ann_sig = (
            mode,
            annotation_size,
            tuple(
                (
                    inst.raw_label,
                    None if inst.points is None else inst.points.tobytes(),
                    None if inst.points is None else inst.points.shape,
                )
                for inst in instances
            ),
        )
        sample_cache_key = ((pred_height, pred_width), pred_digest, ann_sig)

        def _build_crops() -> list[np.ndarray]:
            if mode == "whole_image":
                crop = (
                    np.ascontiguousarray(np.rot90(bgr_u8))
                    if pred_height / pred_width >= 1.5
                    else bgr_u8
                )
                return [crop]
            return [
                crop_text_region(bgr_u8, inst.points)
                for inst in instances
                if inst.points is not None
            ]

        runtime = _get_shared_runtime(self._recognizer_config)
        raw_predictions = runtime.recognize_cached(
            cache_key=sample_cache_key,
            crop_builder=_build_crops,
            expected_count=len(instances),
        )
        return [
            (normalize_text(raw_pred_text, ignore_case=self.ignore_case), inst.norm_label)
            for raw_pred_text, inst in zip(raw_predictions, instances, strict=True)
        ]

    def _stats_from_pairs(self, pairs: list[tuple[str, str]]) -> tuple[float, float]:
        raise NotImplementedError

    def compute_stats(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Return additive ``(numerator, denominator)`` counts for exact global reduction."""
        del target
        pairs = self._recognize_pairs(pred, meta)
        if not pairs:
            return (0.0, 0.0)
        return self._stats_from_pairs(pairs)

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        """Compute scalar metric for one image (returns ``nan`` when the image has 0 valid boxes)."""
        numerator, denominator = self.compute_stats(pred, target=target, meta=meta)
        return float(numerator / denominator) if denominator > 0.0 else float("nan")


class OCRAccuracyMetric(_BaseOCRMetric):
    """Exact-match text recognition accuracy (**OCR-A**, higher is better)."""

    higher_is_better: bool = True

    def _stats_from_pairs(self, pairs: list[tuple[str, str]]) -> tuple[float, float]:
        exact_matches = sum(1 for pred_text, label_text in pairs if pred_text == label_text)
        return (float(exact_matches), float(len(pairs)))


class CERMetric(_BaseOCRMetric):
    """Character Error Rate (**CER**, lower is better, unclipped and allowed to exceed 1.0)."""

    higher_is_better: bool = False

    def _stats_from_pairs(self, pairs: list[tuple[str, str]]) -> tuple[float, float]:
        total_edits = sum(edit_distance(pred_text, label_text) for pred_text, label_text in pairs)
        total_chars = sum(len(label_text) for _, label_text in pairs)
        return (float(total_edits), float(total_chars))


class OneMinusNEDMetric(_BaseOCRMetric):
    """One minus Normalized Edit Distance (**1-NED**, higher is better)."""

    higher_is_better: bool = True

    def _stats_from_pairs(self, pairs: list[tuple[str, str]]) -> tuple[float, float]:
        total_score = sum(
            1.0 - edit_distance(pred_text, label_text) / max(len(pred_text), len(label_text))
            for pred_text, label_text in pairs
        )
        return (float(total_score), float(len(pairs)))
