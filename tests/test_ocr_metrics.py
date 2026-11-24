"""Comprehensive unit, integration, and 2-process CPU DDP tests for OCR-A, CER, and 1-NED."""

from __future__ import annotations

import builtins
import copy
import gc
import importlib
import json
import logging
import math
import os
import socket
import subprocess
import sys
import types
import unicodedata
import weakref
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import metrics
import metrics.ocr_metrics as ocr_mod
from engine import Trainer
from metrics import (
    BaseMetric,
    CERMetric,
    OCRAccuracyMetric,
    OneMinusNEDMetric,
)
from metrics.ocr_metrics import (
    DEFAULT_OCR_MODEL,
    RapidOCRTextRecognizer,
    clear_ocr_caches,
    crop_text_region,
    edit_distance,
    normalize_text,
    rectified_corners,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())
TOOLS_TRAIN = REPO_ROOT / "tools" / "train.py"


def _ref_normalize_text(text: str, ignore_case: bool = False) -> str:
    text = "".join(unicodedata.normalize("NFKC", text).split())
    return text.lower() if ignore_case else text


def _ref_edit_distance(first: str, second: str) -> int:
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


def _ref_ocr_scores(pairs: list[tuple[str, str]]) -> dict[str, float]:
    distances = [_ref_edit_distance(text, label) for text, label in pairs]
    return {
        "OCR-A": sum(text == label for text, label in pairs) / len(pairs),
        "CER": sum(distances) / sum(len(label) for _, label in pairs),
        "1-NED": 1.0
        - sum(
            dist / max(len(pred_text), len(label_text))
            for dist, (pred_text, label_text) in zip(distances, pairs, strict=True)
        )
        / len(pairs),
    }


def _reset_accelerate_and_loggers() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    for name, obj in list(logging.root.manager.loggerDict.items()):
        if (name == "srtrainer" or name.startswith("srtrainer.")) and isinstance(
            obj, logging.Logger
        ):
            for handler in list(obj.handlers):
                obj.removeHandler(handler)
                handler.close()
            obj.disabled = False


@pytest.fixture(autouse=True)
def _cleanup_caches_and_state():
    clear_ocr_caches()
    _reset_accelerate_and_loggers()
    yield
    clear_ocr_caches()
    _reset_accelerate_and_loggers()


class _RecordingFakeRecognizer:
    """Test-only recognizer tracking init kwargs, call count, and BGR crop inputs."""

    instances: list[_RecordingFakeRecognizer] = []
    scripted_outputs: list[list[tuple[str, float]]] = []
    default_fn: Any = None

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = dict(kwargs)
        self.calls: list[list[np.ndarray]] = []
        type(self).instances.append(self)

    def __call__(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        self.calls.append([crop.copy() for crop in crops])
        cls = type(self)
        if cls.scripted_outputs:
            return cls.scripted_outputs.pop(0)
        if cls.default_fn is not None:
            return cls.default_fn(crops)
        return [("box0", 0.99) for _ in crops]


def _install_fake_recognizer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scripted_outputs: list[list[tuple[str, float]]] | None = None,
    default_fn: Any = None,
) -> type[_RecordingFakeRecognizer]:
    class _ScopedRecognizer(_RecordingFakeRecognizer):
        pass

    _ScopedRecognizer.instances = []
    _ScopedRecognizer.scripted_outputs = list(scripted_outputs or [])
    _ScopedRecognizer.default_fn = staticmethod(default_fn) if default_fn is not None else None
    monkeypatch.setattr(ocr_mod, "RapidOCRTextRecognizer", _ScopedRecognizer)
    return _ScopedRecognizer


def _install_mock_cv2_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Patch crop_text_region for tests running in environments without OpenCV."""
    recorded: list[tuple[np.ndarray, np.ndarray]] = []

    def _fake_crop(bgr_image: np.ndarray, points: Any) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        recorded.append((bgr_image.copy(), pts.copy()))
        x_min = max(0, int(np.floor(pts[:, 0].min())))
        y_min = max(0, int(np.floor(pts[:, 1].min())))
        x_max = min(bgr_image.shape[1], max(x_min + 1, int(np.ceil(pts[:, 0].max()))))
        y_max = min(bgr_image.shape[0], max(y_min + 1, int(np.ceil(pts[:, 1].max()))))
        crop = np.ascontiguousarray(bgr_image[y_min:y_max, x_min:x_max])
        if crop.shape[0] / crop.shape[1] >= 1.5:
            crop = np.ascontiguousarray(np.rot90(crop))
        return crop

    monkeypatch.setattr(ocr_mod, "crop_text_region", _fake_crop)
    return recorded


def _write_rgb_png(path: Path, height: int, width: int, rgb_value: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[..., 0] = rgb_value[0]
    arr[..., 1] = rgb_value[1]
    arr[..., 2] = rgb_value[2]
    Image.fromarray(arr, mode="RGB").save(path)


def _to_hydra_val(obj: Any) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        return json.dumps(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_to_hydra_val(item) for item in obj) + "]"
    if isinstance(obj, dict):
        return "{" + ",".join(f"{key}:{_to_hydra_val(val)}" for key, val in obj.items()) + "}"
    raise TypeError(f"Unsupported Hydra CLI value type: {type(obj).__name__}")


def _make_trainer_cfg(
    work_dir: Path,
    *,
    train_root: Path,
    train_json: Path,
    val_suites: dict[str, Any],
    primary_dataset: str | None,
    primary_metric: str | None,
    total_iters: int = 2,
    val_interval: int = 1,
    save_interval: int = 2,
) -> DictConfig:
    with hydra.initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=vosr2_onestep_text"],
        )
    OmegaConf.set_struct(cfg, False)
    cfg.exp_name = work_dir.name
    cfg.scale = 1
    cfg.seed = 42
    cfg.output_root = str(work_dir.parent)
    cfg.work_dir = str(work_dir)
    from tests.dummy_arch import get_dummy_vosr_arch_dict

    cfg.arch = OmegaConf.create(get_dummy_vosr_arch_dict())
    cfg.loss = OmegaConf.create(
        {
            "l1": {
                "weight": 1.0,
                "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
            }
        }
    )
    cfg.model.scale = 1
    cfg.model.ema.enabled = False
    cfg.model.trainable = OmegaConf.create({"vae": False, "dit": True, "dino": False})
    cfg.model.pretrain = None
    cfg.data.gt_size = 16
    cfg.data.scale = 1
    cfg.data.batch_size_per_gpu = 1
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.degradation = None
    cfg.data.sources = OmegaConf.create(
        [
            {
                "name": "paired_text",
                "root_dir": str(train_root),
                "json_path": str(train_json),
                "ratio": 1.0,
            }
        ]
    )
    cfg.optim.optimizer.lr = 1.0e-4
    cfg.optim.scheduler.T_max = max(1, total_iters)
    cfg.metric = OmegaConf.create(
        {
            "primary_dataset": primary_dataset,
            "primary_metric": primary_metric,
            "val_suites": val_suites,
        }
    )
    cfg.trainer.total_iters = total_iters
    cfg.trainer.log_interval = 1
    cfg.trainer.val_interval = val_interval
    cfg.trainer.save_interval = save_interval
    cfg.trainer.max_keep_states = 2
    cfg.trainer.mixed_precision = "no"
    cfg.trainer.gradient_accumulation_steps = 1
    cfg.trainer.log_with = None
    cfg.trainer.resume = None
    cfg.trainer.append_timestamp = False
    cfg.trainer.val_save_visuals = False
    cfg.trainer.val_use_ema = False
    cfg.trainer.val_tile.enabled = False
    cfg.trainer.val_tile.pad_multiple = 1
    return cfg


def test_normalize_text_and_edit_distance_parity() -> None:
    """Verify NFKC, whitespace removal, case handling, and Unicode Levenshtein parity."""
    samples = [
        ("  Hello\tWorld\n", False, "HelloWorld"),
        ("  Hello\tWorld\n", True, "helloworld"),
        ("ＡＢＣ\u3000１２３\u00a0ﬁ", False, "ABC123fi"),
        ("ＡＢＣ\u3000１２３\u00a0ﬁ", True, "abc123fi"),
        ("中文  测试\u2003文本", False, "中文测试文本"),
        ("   \t\r\n\u3000  ", False, ""),
        ("ÄÖÜ ß", True, "äöüß"),
    ]
    for raw_text, ignore_case, expected in samples:
        assert normalize_text(raw_text, ignore_case=ignore_case) == expected
        assert normalize_text(raw_text, ignore_case=ignore_case) == _ref_normalize_text(
            raw_text, ignore_case
        )

    with pytest.raises(TypeError):
        normalize_text(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        normalize_text("abc", ignore_case=1)  # type: ignore[arg-type]

    dist_cases = [
        ("", "", 0),
        ("", "ABC", 3),
        ("ABC", "", 3),
        ("ABC", "ABC", 0),
        ("ABC", "AXC", 1),
        ("ABC", "ABXC", 1),
        ("ABXC", "ABC", 1),
        ("kitten", "sitting", 3),
        ("中文测试", "中文评测", 2),
        ("A", "ABCDEF", 5),
        ("😀a", "😀b", 1),
    ]
    for left_text, right_text, expected_dist in dist_cases:
        assert edit_distance(left_text, right_text) == expected_dist
        assert edit_distance(left_text, right_text) == _ref_edit_distance(left_text, right_text)

    with pytest.raises(TypeError):
        edit_distance("a", None)  # type: ignore[arg-type]


def test_formula_parity_cer_gt_one_empty_pred_and_cross_image_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify exact cross-image weighted reduction, unclipped CER>1, empty pred, and case mode."""
    _install_mock_cv2_crop(monkeypatch)
    _install_fake_recognizer(
        monkeypatch,
        scripted_outputs=[
            [("", 0.0)],
            [("0123456789", 0.99), ("abcdefghij", 0.99), ("中文测试一二三四五六", 0.99)],
            [("ABCDEF", 0.8), ("Ａｂ\u3000ｃ", 0.9)],
        ],
    )

    acc_metric = OCRAccuracyMetric(ignore_case=False)
    cer_metric = CERMetric(ignore_case=False)
    ned_metric = OneMinusNEDMetric(ignore_case=False)
    assert acc_metric.higher_is_better is True
    assert cer_metric.higher_is_better is False
    assert ned_metric.higher_is_better is True

    img1 = torch.full((1, 3, 16, 32), 0.1)
    meta1 = {"label": "A"}
    img2 = torch.full((1, 3, 32, 64), 0.2)
    meta2 = {
        "text_lines": [
            {"transcript": "0123456789", "bbox": [0, 0, 20, 10]},
            {"transcription": "abcdefghij", "bbox": [0, 10, 20, 20]},
            {"text": "中文测试一二三四五六", "bbox": [0, 20, 20, 30]},
        ]
    }

    acc_1 = acc_metric.compute_stats(img1, meta=meta1)
    cer_1 = cer_metric.compute_stats(img1, meta=meta1)
    ned_1 = ned_metric.compute_stats(img1, meta=meta1)
    assert acc_1 == (0.0, 1.0)
    assert cer_1 == (1.0, 1.0)
    assert ned_1 == (0.0, 1.0)

    acc_2 = acc_metric.compute_stats(img2, meta=meta2)
    cer_2 = cer_metric.compute_stats(img2, meta=meta2)
    ned_2 = ned_metric.compute_stats(img2, meta=meta2)
    assert acc_2 == (3.0, 3.0)
    assert cer_2 == (0.0, 30.0)
    assert ned_2 == (3.0, 3.0)

    global_acc = (acc_1[0] + acc_2[0]) / (acc_1[1] + acc_2[1])
    global_cer = (cer_1[0] + cer_2[0]) / (cer_1[1] + cer_2[1])
    global_ned = (ned_1[0] + ned_2[0]) / (ned_1[1] + ned_2[1])
    ref = _ref_ocr_scores(
        [
            ("", "A"),
            ("0123456789", "0123456789"),
            ("abcdefghij", "abcdefghij"),
            ("中文测试一二三四五六", "中文测试一二三四五六"),
        ]
    )
    assert global_acc == pytest.approx(0.75, abs=1e-12)
    assert global_cer == pytest.approx(1.0 / 31.0, abs=1e-12)
    assert global_ned == pytest.approx(0.75, abs=1e-12)
    assert global_acc == pytest.approx(ref["OCR-A"], abs=1e-12)
    assert global_cer == pytest.approx(ref["CER"], abs=1e-12)
    assert global_ned == pytest.approx(ref["1-NED"], abs=1e-12)

    img3 = torch.full((1, 3, 32, 64), 0.3)
    meta3 = {
        "text_lines": [
            {"transcript": "A", "bbox": [0, 0, 20, 10]},
            {"transcript": "abc", "bbox": [0, 10, 20, 20]},
        ]
    }
    assert cer_metric.compute_stats(img3, meta=meta3) == (6.0, 4.0)
    assert cer_metric.compute(img3, meta=meta3) == pytest.approx(1.5, abs=1e-12)
    assert acc_metric.compute(img3, meta=meta3) == pytest.approx(0.0, abs=1e-12)
    assert ned_metric.compute(img3, meta=meta3) == pytest.approx(
        ((1.0 - 5.0 / 6.0) + (1.0 - 1.0 / 3.0)) / 2.0, abs=1e-12
    )

    acc_ci = OCRAccuracyMetric(ignore_case=True)
    cer_ci = CERMetric(ignore_case=True)
    ned_ci = OneMinusNEDMetric(ignore_case=True)
    assert acc_ci.compute(img3, meta=meta3) == pytest.approx(0.5, abs=1e-12)
    assert cer_ci.compute(img3, meta=meta3) == pytest.approx(5.0 / 4.0, abs=1e-12)
    assert ned_ci.compute(img3, meta=meta3) == pytest.approx(
        ((1.0 - 5.0 / 6.0) + 1.0) / 2.0, abs=1e-12
    )


def test_annotation_contract_ignore_rules_precedence_and_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify text_lines precedence, sentinel/flag ignore rules, full-width symbols, and scaling."""
    recorded_crops = _install_mock_cv2_crop(monkeypatch)
    fake_cls = _install_fake_recognizer(
        monkeypatch,
        default_fn=lambda crops: [("OK", 0.95) for _ in crops],
    )

    pred = torch.rand(1, 3, 40, 80, dtype=torch.float32)

    meta_empty_lines = {"label": "SHOULD_NOT_BE_USED", "text_lines": []}
    acc_metric = OCRAccuracyMetric()
    assert acc_metric.compute_stats(pred, target=None, meta=meta_empty_lines) == (0.0, 0.0)
    assert math.isnan(acc_metric.compute(pred, target=None, meta=meta_empty_lines))
    assert fake_cls.instances == []

    custom_metric = OCRAccuracyMetric(label_key="gt_text")
    assert custom_metric.compute_stats(pred, target=None, meta={"gt_text": "OK"}) == (1.0, 1.0)
    assert custom_metric.compute_stats(pred, target=None, meta={"gt_text": "###"}) == (0.0, 0.0)
    assert custom_metric.compute_stats(pred, target=None, meta={"gt_text": " * "}) == (0.0, 0.0)
    assert custom_metric.compute_stats(pred, target=None, meta={"gt_text": "  \u3000 "}) == (
        0.0,
        0.0,
    )
    assert custom_metric.compute_stats(
        pred, target=None, meta={"gt_text": "OK", "alignment_valid": False}
    ) == (0.0, 0.0)

    assert custom_metric.compute_stats(pred, target=None, meta={"gt_text": "＃＃＃"}) == (0.0, 1.0)
    assert custom_metric.compute_stats(pred, target=None, meta={"gt_text": "＊"}) == (0.0, 1.0)

    valid_box = [2, 2, 18, 12]
    meta_flags = {
        "text_lines": [
            {"transcript": "OK", "bbox": valid_box},
            {"transcript": "OK", "bbox": valid_box, "difficult": True},
            {"transcript": "OK", "bbox": valid_box, "difficult": True, "valid": False},
            {"transcript": "OK", "bbox": valid_box, "difficult": True, "ignore": True},
            {
                "transcript": "OK",
                "bbox": valid_box,
                "difficult": True,
                "alignment_valid": False,
            },
            {"transcript": "###", "bbox": valid_box, "difficult": True},
            {"transcript": "*", "bbox": valid_box, "difficult": True},
            {"transcript": "   ", "bbox": valid_box, "difficult": True},
        ]
    }
    assert OCRAccuracyMetric(include_difficult=False).compute_stats(pred, meta=meta_flags) == (
        1.0,
        1.0,
    )
    assert OCRAccuracyMetric(include_difficult=True).compute_stats(pred, meta=meta_flags) == (
        2.0,
        2.0,
    )

    meta_img_unaligned = {**copy.deepcopy(meta_flags), "alignment_valid": False}
    assert OCRAccuracyMetric(include_difficult=True).compute_stats(
        pred, meta=meta_img_unaligned
    ) == (0.0, 0.0)

    recorded_crops.clear()
    meta_geoms = {
        "annotation_size": [40, 20],
        "text_lines": [
            {"transcript": "OK", "bbox": [2, 3, 12, 8]},
            {"text": "OK", "box": [14, 3, 24, 8]},
            {"transcription": "OK", "quad": [2, 10, 12, 10, 12, 16, 2, 16]},
            {"transcript": "OK", "points": [[14, 10], [14, 16], [24, 16], [24, 10]]},
            {
                "transcript": "OK",
                "points": [[26, 4], [32, 2], [38, 4], [38, 12], [32, 10], [26, 12]],
            },
        ],
    }
    meta_snapshot = copy.deepcopy(meta_geoms)
    pred_grad = pred.clone().requires_grad_(True)
    pred_snapshot = pred.clone()

    stats = OCRAccuracyMetric().compute_stats(pred_grad, target=None, meta=meta_geoms)
    assert stats == (5.0, 5.0)
    assert meta_geoms == meta_snapshot
    assert torch.equal(pred_grad.detach(), pred_snapshot)
    assert pred_grad.grad is None
    assert len(recorded_crops) == 5

    np.testing.assert_allclose(
        recorded_crops[0][1],
        np.float32([[4, 6], [24, 6], [24, 16], [4, 16]]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        recorded_crops[3][1],
        np.float32([[28, 20], [48, 20], [48, 32], [28, 32]]),
        atol=1e-5,
    )
    assert recorded_crops[4][1].shape == (6, 2)
    np.testing.assert_allclose(
        recorded_crops[4][1],
        np.float32([[52, 8], [64, 4], [76, 8], [76, 24], [64, 20], [52, 24]]),
        atol=1e-5,
    )


def test_annotation_and_constructor_validation_errors() -> None:
    """Verify fail-fast rejection of malformed configs, tensors, metadata, and degenerate boxes."""
    pred = torch.rand(1, 3, 16, 32)
    metric = OCRAccuracyMetric()

    with pytest.raises(ValueError):
        OCRAccuracyMetric(ocr_model="")
    with pytest.raises(ValueError):
        OCRAccuracyMetric(ocr_model_dir="   ")
    with pytest.raises(ValueError):
        OCRAccuracyMetric(ocr_cls_model_dir="/tmp/cls")
    with pytest.raises(ValueError):
        OCRAccuracyMetric(ocr_device="")
    for bad_batch_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            OCRAccuracyMetric(ocr_batch_size=bad_batch_size)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OCRAccuracyMetric(ignore_case=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OCRAccuracyMetric(include_difficult="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OCRAccuracyMetric(label_key=" ")

    with pytest.raises(TypeError):
        metric.compute_stats(np.zeros((1, 3, 16, 32), dtype=np.float32), meta={"label": "A"})  # type: ignore[arg-type]
    for bad_pred in (
        torch.rand(3, 16, 32),
        torch.rand(2, 3, 16, 32),
        torch.rand(1, 1, 16, 32),
        torch.empty(1, 3, 0, 32),
        torch.zeros(1, 3, 16, 32, dtype=torch.uint8),
        torch.tensor([[[[float("nan")]]] * 3], dtype=torch.float32),
        torch.tensor([[[[float("inf")]]] * 3], dtype=torch.float32),
    ):
        with pytest.raises(ValueError):
            metric.compute_stats(bad_pred, meta={"label": "A"})

    for bad_meta in (
        None,
        "not_a_dict",
        {},
        {"label": None},
        {"label": 123},
        {"label": "A", "alignment_valid": 0},
        {"label": "A", "valid": "true"},
        {"label": "A", "ignore": 1},
        {"label": "A", "difficult": None},
        {"text_lines": None},
        {"text_lines": "bad"},
        {"text_lines": ["not_a_dict"]},
        {"text_lines": [{"bbox": [0, 0, 10, 10]}]},
        {"text_lines": [{"transcript": None, "bbox": [0, 0, 10, 10]}]},
        {"text_lines": [{"transcript": 42, "bbox": [0, 0, 10, 10]}]},
        {"text_lines": [{"transcript": "A"}]},
        {"text_lines": [{"transcript": "A", "bbox": [0, 0, 10, 10], "valid": 1}]},
        {"text_lines": [{"transcript": "A", "bbox": [0, 0, 10, 10], "ignore": "false"}]},
        {"text_lines": [{"transcript": "A", "bbox": [0, 0, 10, 10], "difficult": 0}]},
        {"text_lines": [{"transcript": "A", "bbox": [0, 0, 10, 10], "alignment_valid": "true"}]},
        {"label": "A", "annotation_size": [0, 10]},
        {"label": "A", "annotation_size": [10, -5]},
        {"label": "A", "annotation_size": [10]},
        {"label": "A", "annotation_size": [True, 10]},
        {"label": "A", "annotation_size": [float("nan"), 10]},
        {"label": "A", "annotation_size": "10x10"},
        {"text_lines": [{"transcript": "A", "bbox": [10, 0, 10, 10]}]},
        {"text_lines": [{"transcript": "A", "bbox": [10, 5, 5, 10]}]},
        {"text_lines": [{"transcript": "A", "bbox": [0, 0, float("inf"), 10]}]},
        {"text_lines": [{"transcript": "A", "bbox": [True, 0, 10, 10]}]},
        {"text_lines": [{"transcript": "A", "quad": [0, 0, 10, 10]}]},
        {"text_lines": [{"transcript": "A", "points": [[0, 0], [10, 10], [10, 0], [0, 10]]}]},
        {"text_lines": [{"transcript": "A", "points": [[0, 0], [4, 0], [8, 0], [12, 0]]}]},
        {"text_lines": [{"transcript": "A", "points": [[0, 0], [10, 0], [10, float("nan")], [0, 10]]}]},
        {"text_lines": [{"transcript": "A", "points": [[0, 0], [10, 0], [10, 10]]}]},
        {"text_lines": [{"transcript": "A", "points": [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]]}]},
        {
            "text_lines": [
                {"transcript": "A", "points": [[0, 0], [10, 0], [0, 10], [10, 10], [5, -2]]}
            ]
        },
    ):
        with pytest.raises(ValueError):
            metric.compute_stats(pred, meta=bad_meta)  # type: ignore[arg-type]


def test_rgb_to_bgr_uint8_clamping_vertical_rotation_and_device_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify RGB->BGR uint8 quantization, 1.5x vertical rotation, and safe .to() no-op."""
    fake_cls = _install_fake_recognizer(
        monkeypatch,
        default_fn=lambda crops: [("OK", 0.99) for _ in crops],
    )

    pred = torch.empty((1, 3, 20, 30), dtype=torch.float32)
    pred[:, 0] = -0.5
    pred[:, 1] = 128.0 / 255.0
    pred[:, 2] = 1.8

    metric = OCRAccuracyMetric(ocr_device="cpu")
    assert metric.to("cpu") is metric
    assert metric.to(torch.device("meta")) is metric
    assert metric.ocr_device == "cpu"
    assert fake_cls.instances == []

    assert metric.compute(pred, meta={"label": "OK"}) == 1.0
    assert len(fake_cls.instances) == 1
    assert fake_cls.instances[0].init_kwargs["device"] == "cpu"
    assert fake_cls.instances[0].init_kwargs["model_name"] == DEFAULT_OCR_MODEL
    assert fake_cls.instances[0].init_kwargs["batch_size"] == 1

    crop = fake_cls.instances[0].calls[0][0]
    assert crop.dtype == np.uint8
    assert crop.shape == (20, 30, 3)
    assert crop.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(crop[0, 0], np.array([255, 128, 0], dtype=np.uint8))

    vertical_pred = torch.zeros((1, 3, 30, 20), dtype=torch.float32)
    vertical_pred[0, 0, 0, -1] = 1.0
    assert metric.compute(vertical_pred, meta={"label": "OK"}) == 1.0
    rotated_crop = fake_cls.instances[0].calls[1][0]
    assert rotated_crop.shape == (20, 30, 3)
    np.testing.assert_array_equal(rotated_crop[0, 0], np.array([0, 0, 255], dtype=np.uint8))

    non_vertical_pred = torch.zeros((1, 3, 29, 20), dtype=torch.float32)
    assert metric.compute(non_vertical_pred, meta={"label": "OK"}) == 1.0
    assert fake_cls.instances[0].calls[2][0].shape == (29, 20, 3)


def test_crop_text_region_cv2_contract_and_real_cv2_if_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify INTER_CUBIC/BORDER_REPLICATE perspective crop contract and real OpenCV parity."""
    calls: list[dict[str, Any]] = []

    class _FakeCV2:
        INTER_CUBIC = 2
        BORDER_REPLICATE = 1

        @staticmethod
        def minAreaRect(pts: np.ndarray):
            xs, ys = pts[:, 0], pts[:, 1]
            cx, cy = float(xs.mean()), float(ys.mean())
            width, height = float(xs.max() - xs.min()), float(ys.max() - ys.min())
            calls.append({"op": "minAreaRect", "shape": pts.shape})
            return ((cx, cy), (width, height), 0.0)

        @staticmethod
        def boxPoints(rect: Any) -> np.ndarray:
            (cx, cy), (width, height), _ = rect
            return np.float32(
                [
                    [cx - width / 2, cy - height / 2],
                    [cx + width / 2, cy - height / 2],
                    [cx + width / 2, cy + height / 2],
                    [cx - width / 2, cy + height / 2],
                ]
            )

        @staticmethod
        def getPerspectiveTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
            calls.append({"op": "getPerspectiveTransform", "src": src.copy(), "dst": dst.copy()})
            return np.eye(3, dtype=np.float32)

        @staticmethod
        def warpPerspective(
            img: np.ndarray,
            mat: np.ndarray,
            dsize: tuple[int, int],
            *,
            flags: int,
            borderMode: int,
        ) -> np.ndarray:
            del img, mat
            calls.append(
                {"op": "warpPerspective", "dsize": dsize, "flags": flags, "borderMode": borderMode}
            )
            width, height = dsize
            return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)

    monkeypatch.setattr(ocr_mod, "_import_cv2", lambda: _FakeCV2)
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)

    ccw_quad = np.float32([[10, 10], [10, 40], [30, 40], [30, 10]])
    crop_vertical = crop_text_region(dummy_bgr, ccw_quad)
    assert crop_vertical.shape == (20, 30, 3)
    warp_call = next(call for call in calls if call["op"] == "warpPerspective")
    assert warp_call["flags"] == _FakeCV2.INTER_CUBIC
    assert warp_call["borderMode"] == _FakeCV2.BORDER_REPLICATE
    transform_call = next(call for call in calls if call["op"] == "getPerspectiveTransform")
    np.testing.assert_allclose(
        transform_call["src"], np.float32([[10, 10], [30, 10], [30, 40], [10, 40]])
    )

    calls.clear()
    poly6 = np.float32([[10, 10], [30, 10], [50, 10], [50, 26], [30, 26], [10, 26]])
    corners6 = rectified_corners(poly6)
    assert corners6.shape == (4, 2)
    assert any(call["op"] == "minAreaRect" for call in calls)
    crop6 = crop_text_region(dummy_bgr, poly6)
    assert crop6.shape == (16, 40, 3)

    monkeypatch.undo()
    try:
        import cv2 as real_cv2
    except ImportError:
        return

    img = np.zeros((80, 100, 3), dtype=np.uint8)
    img[10:30, 20:60] = np.array([10, 20, 30], dtype=np.uint8)
    img[10:14, 20:24] = np.array([200, 150, 100], dtype=np.uint8)
    quad_ccw = np.float32([[20, 10], [20, 30], [60, 30], [60, 10]])
    real_crop = crop_text_region(img, quad_ccw)
    assert real_crop.shape == (20, 40, 3)
    assert real_crop.dtype == np.uint8

    cw_corners = np.float32([[20, 10], [60, 10], [60, 30], [20, 30]])
    dst_corners = np.float32([[0, 0], [40, 0], [40, 20], [0, 20]])
    ref_crop = real_cv2.warpPerspective(
        img,
        real_cv2.getPerspectiveTransform(cw_corners, dst_corners),
        (40, 20),
        flags=real_cv2.INTER_CUBIC,
        borderMode=real_cv2.BORDER_REPLICATE,
    )
    np.testing.assert_array_equal(real_crop, ref_crop)

    vertical_real_crop = crop_text_region(
        img, np.float32([[20, 10], [40, 10], [40, 40], [20, 40]])
    )
    assert vertical_real_crop.shape == (20, 30, 3)


def test_rapidocr_text_recognizer_cls_180_rotation_local_paths_and_count_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec_dir = tmp_path / "rec_model"
    rec_dir.mkdir()
    rec_path = rec_dir / "inference.onnx"
    rec_path.touch()
    dict_path = rec_dir / "dict.txt"
    dict_path.touch()
    cls_path = tmp_path / "cls.onnx"
    cls_path.touch()

    init_calls: list[dict[str, Any]] = []
    rec_seen_crops: list[np.ndarray] = []
    counts = {"texts": 2, "scores": 2, "cls": 2}

    class _FakeRapidOCR:
        def __init__(self, *, params: dict[str, Any]) -> None:
            init_calls.append(params)

        def recognize_txt(self, crops: list[np.ndarray]) -> types.SimpleNamespace:
            rec_seen_crops[:] = [crop.copy() for crop in crops]
            return types.SimpleNamespace(
                txts=("t0", "", "extra")[: counts["texts"]],
                scores=(0.9, 0.0, 0.5)[: counts["scores"]],
            )

        def cls_and_rotate(
            self, crops: list[np.ndarray]
        ) -> tuple[list[np.ndarray], types.SimpleNamespace]:
            oriented = [crops[0], np.ascontiguousarray(crops[1][::-1, ::-1])]
            predictions = [("0", 0.9), ("180", 0.6), ("0", 0.5)][: counts["cls"]]
            return oriented, types.SimpleNamespace(cls_res=predictions)

    rapidocr = types.SimpleNamespace(
        RapidOCR=_FakeRapidOCR,
        OCRVersion=str,
        ModelType=str,
        EngineType=types.SimpleNamespace(ONNXRUNTIME="onnxruntime"),
    )
    onnxruntime = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(ocr_mod, "_import_rapidocr", lambda: (rapidocr, onnxruntime))

    with pytest.raises(FileNotFoundError, match="ONNX"):
        RapidOCRTextRecognizer(model_dir=tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="ONNX"):
        RapidOCRTextRecognizer(
            cls_model_name="PP-LCNet_x1_0_textline_ori", cls_model_dir=tmp_path / "missing_cls"
        )
    with pytest.raises(ValueError, match="Unknown OCR recognition model"):
        RapidOCRTextRecognizer(model_name="other_model")
    with pytest.raises(ValueError, match="requires CANNExecutionProvider"):
        RapidOCRTextRecognizer(device="npu:1")

    recognizer = RapidOCRTextRecognizer(
        model_dir=rec_dir,
        cls_model_name="PP-LCNet_x1_0_textline_ori",
        cls_model_dir=cls_path,
        device="cpu",
        batch_size=2,
    )
    (params,) = init_calls
    assert params["Global.use_det"] is False
    assert params["Global.use_cls"] is True
    assert params["Rec.engine_type"] == params["Cls.engine_type"] == "onnxruntime"
    assert params["Rec.ocr_version"] == params["Cls.ocr_version"] == "PP-OCRv5"
    assert params["Rec.model_type"] == params["Cls.model_type"] == "server"
    assert params["Rec.lang_type"] == "ch"
    assert params["Rec.model_path"] == str(rec_path)
    assert params["Rec.rec_keys_path"] == str(dict_path)
    assert params["Cls.model_path"] == str(cls_path)
    assert params["Rec.rec_batch_num"] == params["Cls.cls_batch_num"] == 2
    assert params["Cls.cls_thresh"] < 0.6

    crop0 = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    crop1 = np.arange(100, 100 + 2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    assert recognizer([]) == []
    assert recognizer([crop0, crop1]) == [("t0", 0.9), ("", 0.0)]
    np.testing.assert_array_equal(rec_seen_crops[0], crop0)
    np.testing.assert_array_equal(rec_seen_crops[1], crop1[::-1, ::-1])

    for output in counts:
        for count in (1, 3):
            counts[output] = count
            with pytest.raises(RuntimeError, match="returned"):
                recognizer([crop0, crop1])
        counts[output] = 2

    onnxruntime.get_available_providers = lambda: ["CUDAExecutionProvider"]
    RapidOCRTextRecognizer(device="gpu:2")
    assert init_calls[-1]["EngineConfig.onnxruntime.use_cuda"] is True
    assert init_calls[-1]["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] == 2
    assert init_calls[-1]["Global.use_cls"] is False
    assert init_calls[-1]["Rec.model_path"] is None


def test_missing_optional_dependencies_raise_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ordinary import/instantiation works without OCR dependencies and missing deps error clearly."""
    real_import = builtins.__import__

    def _block_optional(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"rapidocr", "onnxruntime"}:
            raise ImportError(f"No module named {name!r}")
        if name == "cv2" or name.startswith("cv2."):
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.setattr(builtins, "__import__", _block_optional)

    importlib.reload(metrics)
    acc_metric = OCRAccuracyMetric()
    assert isinstance(acc_metric, BaseMetric)
    assert acc_metric.to("cpu") is acc_metric

    pred = torch.rand(1, 3, 16, 32)
    with pytest.raises(ImportError, match="uv sync --extra ocr"):
        acc_metric.compute(pred, meta={"label": "HELLO"})

    with pytest.raises(ImportError, match="uv sync --extra ocr"):
        crop_text_region(
            np.zeros((16, 32, 3), dtype=np.uint8), [[0, 0], [10, 0], [10, 10], [0, 10]]
        )


def test_shared_model_single_recognition_invalidation_inference_mode_and_bounded_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 3-metric single pass, in-place & meta invalidation, inference_mode, and zero tensor leak."""
    _install_mock_cv2_crop(monkeypatch)
    call_counter = {"count": 0}

    def _dynamic_rec(crops: list[np.ndarray]) -> list[tuple[str, float]]:
        call_counter["count"] += 1
        return [(f"V{int(round(float(crop[..., 2].mean())))}", 0.99) for crop in crops]

    fake_cls = _install_fake_recognizer(monkeypatch, default_fn=_dynamic_rec)

    acc_metric = OCRAccuracyMetric()
    cer_metric = CERMetric()
    ned_metric = OneMinusNEDMetric()

    pred = torch.full((1, 3, 16, 32), 0.0, dtype=torch.float32)
    pred[:, 0] = 100.0 / 255.0
    meta = {"text_lines": [{"transcript": "V100", "bbox": [0, 0, 16, 12]}]}

    assert acc_metric.compute_stats(pred, meta=meta) == (1.0, 1.0)
    assert cer_metric.compute_stats(pred, meta=meta) == (0.0, 4.0)
    assert ned_metric.compute_stats(pred, meta=meta) == (1.0, 1.0)
    assert len(fake_cls.instances) == 1
    assert call_counter["count"] == 1

    pred[:, 0] = 200.0 / 255.0
    assert acc_metric.compute_stats(pred, meta=meta) == (0.0, 1.0)
    assert cer_metric.compute_stats(pred, meta=meta) == (1.0, 4.0)
    assert call_counter["count"] == 2

    meta_moved = {"text_lines": [{"transcript": "V200", "bbox": [2, 2, 18, 14]}]}
    assert acc_metric.compute_stats(pred, meta=meta_moved) == (1.0, 1.0)
    assert call_counter["count"] == 3

    meta_relabel = {"text_lines": [{"transcript": "DIFF", "bbox": [2, 2, 18, 14]}]}
    assert acc_metric.compute_stats(pred, meta=meta_relabel) == (0.0, 1.0)
    assert call_counter["count"] == 4

    with torch.inference_mode():
        inf_pred = torch.full((1, 3, 16, 32), 0.5, dtype=torch.float32)
        assert inf_pred.is_inference()
        count_before_inf = call_counter["count"]
        assert acc_metric.compute_stats(inf_pred, meta={"label": "V128"}) == (1.0, 1.0)
        assert cer_metric.compute_stats(inf_pred, meta={"label": "V128"}) == (0.0, 4.0)
        assert ned_metric.compute_stats(inf_pred, meta={"label": "V128"}) == (1.0, 1.0)
        assert call_counter["count"] == count_before_inf + 1

    pred_a = torch.full((1, 3, 16, 32), 0.1)
    pred_b = torch.full((1, 3, 16, 32), 0.9)
    meta_single = {"label": "V26"}
    acc_metric.compute_stats(pred_a, meta=meta_single)
    count_after_a1 = call_counter["count"]
    acc_metric.compute_stats(pred_b, meta=meta_single)
    assert call_counter["count"] == count_after_a1 + 1
    acc_metric.compute_stats(pred_a, meta=meta_single)
    assert call_counter["count"] == count_after_a1 + 2

    runtime = ocr_mod._get_shared_runtime(acc_metric._recognizer_config)
    assert runtime._last_key is not None
    assert runtime._last_texts is not None

    def _assert_no_tensors(obj: Any) -> None:
        assert not isinstance(obj, (torch.Tensor, np.ndarray))
        if isinstance(obj, (tuple, list)):
            for item in obj:
                _assert_no_tensors(item)

    _assert_no_tensors(runtime._last_key)
    _assert_no_tensors(runtime._last_texts)

    transient = torch.rand(1, 3, 16, 32, requires_grad=True)
    ref_transient = weakref.ref(transient)
    acc_metric.compute_stats(transient, meta={"label": "V0"})
    del transient
    gc.collect()
    assert ref_transient() is None

    other_metric = OCRAccuracyMetric(ocr_batch_size=4)
    other_metric.compute_stats(pred_a, meta=meta_single)
    assert len(fake_cls.instances) == 2


def test_hydra_text_ocr_config_compose_and_instantiate() -> None:
    """Verify configs/metric/text_ocr.yaml composes and instantiates lazily without OCR dependencies."""
    with hydra.initialize_config_dir(version_base=None, config_dir=CONFIGS_DIR):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=vosr2_onestep_text", "metric=text_ocr"],
        )
    assert cfg.metric.primary_dataset == "text_ocr"
    assert cfg.metric.primary_metric == "ocr_a"
    suite_metrics = cfg.metric.val_suites.text_ocr.metrics
    assert set(suite_metrics.keys()) == {"ocr_a", "cer", "one_minus_ned"}

    instantiated = {
        metric_key: instantiate(metric_cfg).to("cpu")
        for metric_key, metric_cfg in suite_metrics.items()
    }
    assert isinstance(instantiated["ocr_a"], OCRAccuracyMetric)
    assert isinstance(instantiated["cer"], CERMetric)
    assert isinstance(instantiated["one_minus_ned"], OneMinusNEDMetric)
    assert instantiated["ocr_a"].higher_is_better is True
    assert instantiated["cer"].higher_is_better is False
    assert instantiated["one_minus_ned"].higher_is_better is True
    assert instantiated["ocr_a"].ocr_device == "cpu"


class _CustomScalarMetric(BaseMetric):
    """Legacy custom BaseMetric implementing only compute() (verifies backward compatibility)."""

    higher_is_better: bool = True

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del pred, target
        assert meta is not None
        return float(meta.get("custom_score", 0.0))


def test_trainer_validate_mixed_metrics_weighted_ocr_cer_best_nan_and_no_stale_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify mixed PSNR/SSIM/custom/OCR suite, CER best direction, all-empty NaN, and PSNR +inf."""
    _install_mock_cv2_crop(monkeypatch)
    rec_calls = {"count": 0}
    scripted = [
        [("", 0.0)],
        [("0123456789", 0.99), ("abcdefghij", 0.99), ("中文测试一二三四五六", 0.99)],
        [("XX", 0.5)],
        [("X123456789", 0.9), ("Xbcdefghij", 0.9), ("X文测试一二三四五六", 0.9)],
        [("A", 0.99)],
        [("0123456789", 0.99), ("abcdefghij", 0.99), ("中文测试一二三四五六", 0.99)],
    ]

    def _scripted_rec(crops: list[np.ndarray]) -> list[tuple[str, float]]:
        del crops
        rec_calls["count"] += 1
        return scripted.pop(0)

    _install_fake_recognizer(monkeypatch, default_fn=_scripted_rec)

    data_root = tmp_path / "data"
    manifests = tmp_path / "manifests"
    manifests.mkdir()

    _write_rgb_png(data_root / "train/lq0.png", 16, 16, (100, 100, 100))
    _write_rgb_png(data_root / "train/gt0.png", 16, 16, (100, 100, 100))
    train_json = manifests / "train.json"
    train_json.write_text(
        json.dumps([{"lq": "train/lq0.png", "gt": "train/gt0.png"}]), encoding="utf-8"
    )

    _write_rgb_png(data_root / "val/img0.png", 16, 16, (50, 50, 50))
    _write_rgb_png(data_root / "val/img1.png", 16, 16, (100, 100, 100))
    _write_rgb_png(data_root / "val/img2.png", 16, 16, (150, 150, 150))
    mixed_val_json = manifests / "val_mixed.json"
    mixed_val_json.write_text(
        json.dumps(
            [
                {
                    "lq": "val/img0.png",
                    "gt": "val/img0.png",
                    "label": "A",
                    "custom_score": 10.0,
                },
                {
                    "lq": "val/img1.png",
                    "gt": "val/img1.png",
                    "custom_score": 20.0,
                    "text_lines": [
                        {"transcript": "0123456789", "bbox": [0, 0, 8, 4]},
                        {"transcript": "abcdefghij", "bbox": [0, 4, 8, 8]},
                        {"transcript": "中文测试一二三四五六", "bbox": [0, 8, 8, 12]},
                    ],
                },
                {
                    "lq": "val/img2.png",
                    "gt": "val/img2.png",
                    "custom_score": 30.0,
                    "text_lines": [],
                },
            ]
        ),
        encoding="utf-8",
    )

    empty_val_json = manifests / "val_empty_ocr.json"
    empty_val_json.write_text(
        json.dumps(
            [
                {"lq": "val/img0.png", "label": "###"},
                {
                    "lq": "val/img1.png",
                    "text_lines": [
                        {"transcript": "ignored", "bbox": [0, 0, 8, 8], "ignore": True}
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )

    val_suites = {
        "mixed": {
            "dataset": {"root_dir": str(data_root), "json_path": str(mixed_val_json)},
            "metrics": {
                "psnr": {"_target_": "metrics.PSNRMetric", "crop_border": 0, "test_y_channel": True},
                "ssim": {"_target_": "metrics.SSIMMetric", "crop_border": 0, "test_y_channel": True},
                "custom": {
                    "_target_": f"{_CustomScalarMetric.__module__}.{_CustomScalarMetric.__qualname__}"
                },
                "ocr_a": {"_target_": "metrics.OCRAccuracyMetric"},
                "cer": {"_target_": "metrics.CERMetric"},
                "one_minus_ned": {"_target_": "metrics.OneMinusNEDMetric"},
            },
        },
        "empty_ocr": {
            "dataset": {"root_dir": str(data_root), "json_path": str(empty_val_json)},
            "metrics": {
                "ocr_a": {"_target_": "metrics.OCRAccuracyMetric"},
                "cer": {"_target_": "metrics.CERMetric"},
                "one_minus_ned": {"_target_": "metrics.OneMinusNEDMetric"},
            },
        },
    }

    work_dir = tmp_path / "exp_cer_best"
    cfg = _make_trainer_cfg(
        work_dir,
        train_root=data_root,
        train_json=train_json,
        val_suites=val_suites,
        primary_dataset="mixed",
        primary_metric="cer",
        total_iters=1,
    )
    trainer = Trainer(cfg)

    res1 = trainer.validate(step=1)
    assert rec_calls["count"] == 2
    assert res1["mixed"]["custom"] == pytest.approx(20.0, abs=1e-12)
    assert math.isfinite(res1["mixed"]["psnr"]) or math.isinf(res1["mixed"]["psnr"])
    assert math.isfinite(res1["mixed"]["ssim"])
    assert res1["mixed"]["ocr_a"] == pytest.approx(0.75, abs=1e-12)
    assert res1["mixed"]["cer"] == pytest.approx(1.0 / 31.0, abs=1e-12)
    assert res1["mixed"]["one_minus_ned"] == pytest.approx(0.75, abs=1e-12)
    assert math.isnan(res1["empty_ocr"]["ocr_a"])
    assert math.isnan(res1["empty_ocr"]["cer"])
    assert math.isnan(res1["empty_ocr"]["one_minus_ned"])

    for comp in ("vae", "dit", "dino"):
        assert (work_dir / "weights" / f"best_mixed_cer_{comp}.safetensors").is_file()
    assert trainer.train_state.best_metrics["mixed/cer"] == pytest.approx(1.0 / 31.0, abs=1e-12)

    res2 = trainer.validate(step=2)
    assert rec_calls["count"] == 4
    assert res2["mixed"]["cer"] == pytest.approx(5.0 / 31.0, abs=1e-12)
    assert trainer.train_state.best_metrics["mixed/cer"] == pytest.approx(1.0 / 31.0, abs=1e-12)

    res3 = trainer.validate(step=3)
    assert rec_calls["count"] == 6
    assert res3["mixed"]["cer"] == pytest.approx(0.0, abs=1e-12)
    assert trainer.train_state.best_metrics["mixed/cer"] == pytest.approx(0.0, abs=1e-12)

    trainer.accelerator.end_training()  # type: ignore[union-attr]

    _reset_accelerate_and_loggers()
    work_empty = tmp_path / "exp_all_empty"
    cfg_empty = _make_trainer_cfg(
        work_empty,
        train_root=data_root,
        train_json=train_json,
        val_suites={"empty_ocr": val_suites["empty_ocr"]},
        primary_dataset="empty_ocr",
        primary_metric="ocr_a",
        total_iters=1,
    )
    trainer_empty = Trainer(cfg_empty)
    res_empty = trainer_empty.validate(step=1)
    assert math.isnan(res_empty["empty_ocr"]["ocr_a"])
    assert trainer_empty.train_state.best_metrics == {}
    assert list((work_empty / "weights").glob("best_*.safetensors")) == []
    trainer_empty.accelerator.end_training()  # type: ignore[union-attr]

    _reset_accelerate_and_loggers()
    work_inf = tmp_path / "exp_psnr_inf"
    cfg_inf = _make_trainer_cfg(
        work_inf,
        train_root=data_root,
        train_json=train_json,
        val_suites={
            "inf_suite": {
                "dataset": {"root_dir": str(data_root), "json_path": str(train_json)},
                "metrics": {
                    "psnr": {
                        "_target_": "metrics.PSNRMetric",
                        "crop_border": 0,
                        "test_y_channel": False,
                    }
                },
            }
        },
        primary_dataset="inf_suite",
        primary_metric="psnr",
        total_iters=1,
    )
    trainer_inf = Trainer(cfg_inf)
    monkeypatch.setattr(
        trainer_inf.model, "get_infer_network", lambda use_ema=True: torch.nn.Identity()
    )
    res_inf = trainer_inf.validate(step=1)
    assert math.isinf(res_inf["inf_suite"]["psnr"]) and res_inf["inf_suite"]["psnr"] > 0.0
    assert math.isinf(trainer_inf.train_state.best_metrics["inf_suite/psnr"])
    for comp in ("vae", "dit", "dino"):
        assert (work_inf / "weights" / f"best_inf_suite_psnr_{comp}.safetensors").is_file()
    trainer_inf.accelerator.end_training()  # type: ignore[union-attr]


def test_distributed_2proc_cpu_odd_samples_with_zero_box_exact_ocr_parity(
    tmp_path: Path,
) -> None:
    """Real 2-process CPU torchrun verifying [1, 2] gather_for_metrics & exact global OCR math.

    Uses 5 odd validation images (including a zero-box image) across 2 ranks so padded tail
    truncation is verified directly against independent hand calculation (OCR-A=4/7, CER=4/39,
    1-NED=29/42 vs padded CER=5/40).
    """
    data_root = tmp_path / "data"
    manifests = tmp_path / "manifests"
    manifests.mkdir()

    _write_rgb_png(data_root / "train/lq0.png", 16, 16, (128, 128, 128))
    _write_rgb_png(data_root / "train/gt0.png", 16, 16, (128, 128, 128))
    train_json = manifests / "train.json"
    train_json.write_text(
        json.dumps([{"lq": "train/lq0.png", "gt": "train/gt0.png"}]), encoding="utf-8"
    )

    val_specs = [
        ("val/img0.png", 16, 16),
        ("val/img1.png", 16, 20),
        ("val/img2.png", 16, 24),
        ("val/img3.png", 16, 28),
        ("val/img4.png", 16, 32),
    ]
    for rel_path, height, width in val_specs:
        _write_rgb_png(data_root / rel_path, height, width, (100, 120, 140))

    val_entries = [
        {"lq": "val/img0.png", "label": "A"},
        {
            "lq": "val/img1.png",
            "text_lines": [
                {"transcript": "0123456789", "bbox": [0, 0, 10, 8]},
                {"transcript": "abcdefghij", "bbox": [0, 0, 12, 8]},
                {"transcript": "中文测试一二三四五六", "bbox": [0, 0, 14, 8]},
            ],
        },
        {"lq": "val/img2.png", "text_lines": []},
        {
            "lq": "val/img3.png",
            "text_lines": [
                {"transcript": "AC", "bbox": [0, 0, 18, 8]},
                {"transcript": "X", "bbox": [0, 0, 22, 8]},
                {"transcript": "###", "bbox": [0, 0, 10, 8]},
            ],
        },
        {"lq": "val/img4.png", "label": "HELLO"},
    ]
    val_json = manifests / "val_odd5_ocr.json"
    val_json.write_text(json.dumps(val_entries), encoding="utf-8")

    stub_dir = tmp_path / "py_stubs"
    stub_dir.mkdir()
    (stub_dir / "sitecustomize.py").write_text(
        """
from importlib.machinery import ModuleSpec
import sys
import types
import numpy as np

WIDTH_TO_TEXT = {
    16: "",
    10: "0123456789",
    12: "abcdefghij",
    14: "中文测试一二三四五六",
    18: "AB",
    22: "XYZ",
    32: "HELLO",
}

class _DeterministicRapidOCR:
    def __init__(self, *, params):
        self.params = params

    def recognize_txt(self, crops):
        widths = [int(crop.shape[1]) for crop in crops]
        return types.SimpleNamespace(
            txts=tuple(WIDTH_TO_TEXT[width] for width in widths),
            scores=(0.99,) * len(crops),
        )

fake_rapidocr = types.ModuleType("rapidocr")
fake_rapidocr.RapidOCR = _DeterministicRapidOCR
fake_rapidocr.OCRVersion = str
fake_rapidocr.ModelType = str
fake_rapidocr.EngineType = types.SimpleNamespace(ONNXRUNTIME="onnxruntime")
sys.modules["rapidocr"] = fake_rapidocr
fake_onnxruntime = types.ModuleType("onnxruntime")
fake_onnxruntime.__spec__ = ModuleSpec("onnxruntime", loader=None)
sys.modules["onnxruntime"] = fake_onnxruntime

try:
    import cv2 as _cv2_check
    del _cv2_check
except ImportError:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.INTER_CUBIC = 2
    fake_cv2.BORDER_REPLICATE = 1

    def _get_persp(src, dst):
        del src, dst
        return np.eye(3, dtype=np.float32)

    def _warp(img, mat, dsize, flags=2, borderMode=1):
        del mat, flags, borderMode
        width, height = int(dsize[0]), int(dsize[1])
        return np.ascontiguousarray(img[:height, :width])

    fake_cv2.getPerspectiveTransform = _get_persp
    fake_cv2.warpPerspective = _warp
    sys.modules["cv2"] = fake_cv2
""",
        encoding="utf-8",
    )

    exp_dir = tmp_path / "ddp_ocr_exp"
    sources_override = _to_hydra_val(
        [
            {
                "name": "paired_text",
                "root_dir": str(data_root),
                "json_path": str(train_json),
                "ratio": 1.0,
            }
        ]
    )
    val_suites_override = _to_hydra_val(
        {
            "text_odd5": {
                "dataset": {"root_dir": str(data_root), "json_path": str(val_json)},
                "metrics": {
                    "ocr_a": {"_target_": "metrics.OCRAccuracyMetric"},
                    "cer": {"_target_": "metrics.CERMetric"},
                    "one_minus_ned": {"_target_": "metrics.OneMinusNEDMetric"},
                },
            }
        }
    )

    from tests.dummy_arch import get_dummy_vosr_arch_dict

    overrides = [
        "experiment=vosr2_onestep_text",
        f"exp_name={exp_dir.name}",
        f"output_root={exp_dir.parent}",
        f"work_dir={exp_dir}",
        "scale=1",
        "seed=42",
        "~arch/vae",
        "~arch/dit",
        "~arch/dino",
        f"+arch={_to_hydra_val(get_dummy_vosr_arch_dict())}",
        "~loss",
        "+loss={l1:{weight:1.0,criterion:{_target_:losses.L1Loss,reduction:mean}}}",
        "model.ema.enabled=false",
        "data.gt_size=16",
        "data.batch_size_per_gpu=1",
        "data.num_workers=0",
        "data.pin_memory=false",
        "data.persistent_workers=false",
        "data.degradation=null",
        f"data.sources={sources_override}",
        "metric.primary_dataset=text_odd5",
        "metric.primary_metric=cer",
        "~metric.val_suites",
        f"+metric.val_suites={val_suites_override}",
        "trainer.total_iters=1",
        "trainer.log_interval=1",
        "trainer.val_interval=1",
        "trainer.save_interval=1",
        "trainer.mixed_precision=no",
        "trainer.gradient_accumulation_steps=1",
        "trainer.log_with=null",
        "trainer.val_save_visuals=false",
        "trainer.val_use_ema=false",
        "trainer.val_tile.enabled=false",
        "trainer.val_tile.pad_multiple=1",
    ]

    env = os.environ.copy()
    env["ACCELERATE_USE_CPU"] = "true"
    env["OMP_NUM_THREADS"] = "1"
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in env:
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    extra_pp = f"{stub_dir}:{REPO_ROOT}:{REPO_ROOT / 'src'}"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{extra_pp}:{existing_pp}" if existing_pp else extra_pp

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = int(sock.getsockname()[1])

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={master_port}",
        str(TOOLS_TRAIN),
        *overrides,
    ]
    proc = subprocess.run(
        cmd,
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"2-process CPU OCR DDP failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    expected_ref = _ref_ocr_scores(
        [
            ("", "A"),
            ("0123456789", "0123456789"),
            ("abcdefghij", "abcdefghij"),
            ("中文测试一二三四五六", "中文测试一二三四五六"),
            ("AB", "AC"),
            ("XYZ", "X"),
            ("HELLO", "HELLO"),
        ]
    )
    expected_ocr_a = 4.0 / 7.0
    expected_cer = 4.0 / 39.0
    expected_ned = 29.0 / 42.0
    assert expected_ocr_a == pytest.approx(expected_ref["OCR-A"], abs=1e-12)
    assert expected_cer == pytest.approx(expected_ref["CER"], abs=1e-12)
    assert expected_ned == pytest.approx(expected_ref["1-NED"], abs=1e-12)

    state_dir = exp_dir / "checkpoints" / "state_iter_000001"
    ts_state = torch.load(
        state_dir / "custom_checkpoint_1.pkl", map_location="cpu", weights_only=True
    )
    actual_cer = float(ts_state["best_metrics"]["text_odd5/cer"])
    assert actual_cer == pytest.approx(expected_cer, abs=1e-6)
    assert abs(actual_cer - (5.0 / 40.0)) > 0.01

    log_text = (exp_dir / "train.log").read_text(encoding="utf-8")
    assert f"ocr_a: {expected_ocr_a:.4f}" in log_text, f"Missing exact ocr_a in log:\n{log_text}"
    assert f"cer: {expected_cer:.4f}" in log_text, f"Missing exact cer in log:\n{log_text}"
    assert (
        f"one_minus_ned: {expected_ned:.4f}" in log_text
    ), f"Missing exact 1-ned in log:\n{log_text}"
    for comp in ("vae", "dit", "dino"):
        assert (exp_dir / "weights" / f"best_text_odd5_cer_{comp}.safetensors").is_file()
