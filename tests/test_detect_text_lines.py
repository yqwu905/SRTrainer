"""Tests for offline multi-device RapidOCR text-line detection script (tools/detect_text_lines.py)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from data.manifest import load_train_manifest
from data.mixed_dataset import MixedSRDataset, train_collate_fn
from losses import OCRKLLoss
from tests.helpers.fake_backends import TinyRecognizer, patch_ppocr_loader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "detect_text_lines.py"


def _load_detect_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("detect_text_lines_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_rgb_png(path: Path, size: tuple[int, int] = (64, 48), color: tuple[int, int, int] = (128, 64, 32)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def test_parse_devices_and_rapidocr_params() -> None:
    mod = _load_detect_module()

    assert mod.parse_devices("cuda:0,cuda:2", workers_per_device=1) == ["cuda:0", "cuda:2"]
    assert mod.parse_devices("gpu:1,gpu:3", workers_per_device=2) == [
        "cuda:1",
        "cuda:1",
        "cuda:3",
        "cuda:3",
    ]
    assert mod.parse_devices("0,1,2", device_type="npu") == ["npu:0", "npu:1", "npu:2"]
    assert mod.parse_devices("cpu", workers_per_device=3) == ["cpu", "cpu", "cpu"]

    with pytest.raises(ValueError, match="Invalid device entry"):
        mod.parse_devices("tpu:0")
    with pytest.raises(ValueError, match="workers_per_device"):
        mod.parse_devices("cpu", workers_per_device=0)

    fake_rapidocr = types.SimpleNamespace(
        EngineType=types.SimpleNamespace(ONNXRUNTIME="onnxruntime"),
        OCRVersion=lambda v: f"ver:{v}",
        ModelType=lambda m: f"type:{m}",
        LangDet=lambda lang: f"lang:{lang}",
    )
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider", "CUDAExecutionProvider", "CANNExecutionProvider"]
    )

    cfg = mod.DetectConfig(det_only=False, use_cls=True, limit_side_len=1280, intra_threads=2)
    params, needs_download, _ = mod.build_rapidocr_params(cfg, "npu:3", fake_rapidocr, fake_ort)
    assert needs_download is True
    assert params["Global.use_det"] is True
    assert params["Global.use_rec"] is True
    assert params["Global.use_cls"] is True
    assert params["Det.limit_side_len"] == 1280
    assert params["EngineConfig.onnxruntime.use_cann"] is True
    assert params["EngineConfig.onnxruntime.cann_ep_cfg.device_id"] == 3
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 2

    cfg_det_only = mod.DetectConfig(det_only=True)
    params_det, _, _ = mod.build_rapidocr_params(cfg_det_only, "cuda:1", fake_rapidocr, fake_ort)
    assert params_det["Global.use_rec"] is False
    assert "Rec.model_path" not in params_det
    assert params_det["EngineConfig.onnxruntime.use_cuda"] is True
    assert params_det["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] == 1


def test_extract_valid_text_lines_clamps_and_drops_degenerate_boxes() -> None:
    mod = _load_detect_module()
    fake_result = types.SimpleNamespace(
        boxes=[
            # Slightly out-of-bounds box -> clamped to [0, 64] x [0, 48]
            [[-4.0, 2.0], [70.0, 2.0], [70.0, 20.0], [-4.0, 20.0]],
            # Degenerate collinear box -> dropped
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [40.0, 10.0]],
            # Tiny box (< min_box_size=3.0) -> dropped
            [[5.0, 5.0], [6.5, 5.0], [6.5, 6.5], [5.0, 6.5]],
            # Valid counter-clockwise box -> canonicalized to clockwise
            [[8.0, 24.0], [8.0, 40.0], [50.0, 40.0], [50.0, 24.0]],
        ],
        txts=["Hello", "collinear", "tiny", "世界"],
        scores=[0.98765, 0.5, 0.9, 0.91234],
    )
    lines, dropped = mod.extract_valid_text_lines(fake_result, (48, 64), min_box_size=3.0)
    assert dropped == 2
    assert len(lines) == 2
    assert lines[0]["transcript"] == "Hello"
    assert lines[0]["score"] == pytest.approx(0.9877)
    pts0 = np.asarray(lines[0]["quad"], dtype=np.float64)
    assert pts0[:, 0].min() == pytest.approx(0.0)
    assert pts0[:, 0].max() == pytest.approx(64.0)
    assert lines[1]["transcript"] == "世界"


def test_run_detect_manifest_overwrites_inserts_and_feeds_ocr_kl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_detect_module()

    _write_rgb_png(tmp_path / "hr/img0.png", (64, 64), color=(10, 20, 30))
    _write_rgb_png(tmp_path / "lr/img0.png", (16, 16), color=(10, 20, 30))
    _write_rgb_png(tmp_path / "hr/img1.png", (64, 64), color=(200, 100, 50))
    _write_rgb_png(tmp_path / "lr/img1.png", (16, 16), color=(200, 100, 50))

    input_json = tmp_path / "train.json"
    output_json = tmp_path / "train_detected.json"
    input_payload = [
        {
            "gt": "hr/img0.png",
            "lq": "lr/img0.png",
            "extra_meta": "keep_me",
            "text_lines": [{"transcript": "OLD_STALE", "bbox": [1, 1, 10, 10]}],
        },
        {
            "gt": "hr/img1.png",
            "lq": "lr/img1.png",
        },
    ]
    input_json.write_text(json.dumps(input_payload, ensure_ascii=False), encoding="utf-8")

    class _FakeEngine:
        def __call__(self, bgr: np.ndarray) -> Any:
            # Distinguish img0 (blue channel=30) vs img1 (blue channel=50)
            b_val = int(bgr[0, 0, 0])
            if b_val == 30:
                return types.SimpleNamespace(
                    boxes=[[[4.0, 6.0], [58.0, 6.0], [58.0, 22.0], [4.0, 22.0]]],
                    txts=["NEW_TEXT_0"],
                    scores=[0.95],
                )
            return types.SimpleNamespace(
                boxes=[
                    [[6.0, 8.0], [54.0, 8.0], [54.0, 24.0], [6.0, 24.0]],
                    [[6.0, 32.0], [48.0, 32.0], [48.0, 48.0], [6.0, 48.0]],
                ],
                txts=["LINE_A", "LINE_B"],
                scores=[0.92, 0.88],
            )

    summary = mod.run_detect_manifest(
        input_json=input_json,
        output_json=output_json,
        root_dir=tmp_path,
        devices="cpu",
        config=mod.DetectConfig(),
        engine_factory=lambda _cfg, _dev: _FakeEngine(),
    )
    assert summary.total_samples == 2
    assert summary.overwritten_samples == 1
    assert summary.added_samples == 1
    assert summary.total_lines == 3
    assert summary.empty_samples == 0
    assert output_json.is_file()

    # Original input file remains untouched
    assert json.loads(input_json.read_text(encoding="utf-8")) == input_payload

    out_records = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(out_records) == 2
    assert out_records[0]["extra_meta"] == "keep_me"
    assert [ln["transcript"] for ln in out_records[0]["text_lines"]] == ["NEW_TEXT_0"]
    assert [ln["transcript"] for ln in out_records[1]["text_lines"]] == ["LINE_A", "LINE_B"]

    # End-to-end: generated JSON loads cleanly into load_train_manifest -> MixedSRDataset -> OCRKLLoss
    samples, _ = load_train_manifest(output_json, tmp_path)
    assert len(samples) == 2
    assert samples[0].text_lines is not None and len(samples[0].text_lines) == 1
    assert samples[1].text_lines is not None and len(samples[1].text_lines) == 2

    ds = MixedSRDataset(
        sources=[
            {"name": "det_src", "root_dir": str(tmp_path), "json_path": str(output_json), "ratio": 1.0}
        ],
        gt_size=64,
        scale=4,
        augment={"hflip": False, "vflip": False, "rot90": False},
        has_degradation=False,
    )
    batch = train_collate_fn([ds[0], ds[1]])
    assert len(batch["text_lines"][0]) == 1
    assert len(batch["text_lines"][1]) == 2

    patch_ppocr_loader(monkeypatch, recognizer=TinyRecognizer(num_classes=9))
    ocr_kl = OCRKLLoss(
        use_text_lines=True,
        strip_batch_size=2,
        amp_dtype="bf16",
        use_checkpoint=True,
        sync_ddp_counts=False,
    )
    pred = (batch["gt"] + 0.1 * torch.randn_like(batch["gt"])).clamp(0.0, 1.0).requires_grad_(True)
    loss = ocr_kl(pred, batch["gt"], batch=batch)
    assert ocr_kl.last_counts[0] == 3
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_cli_multiprocess_spawn_preserves_order_with_stub_rapidocr(tmp_path: Path) -> None:
    """Exercise true multiprocessing ('spawn') CLI execution across 2 workers with stubbed rapidocr."""
    stub_dir = tmp_path / "stub_pkgs"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "onnxruntime.py").write_text(
        "def get_available_providers():\n    return ['CPUExecutionProvider']\n",
        encoding="utf-8",
    )
    (stub_dir / "rapidocr.py").write_text(
        """
from types import SimpleNamespace


class EngineType:
    ONNXRUNTIME = "onnxruntime"


class OCRVersion(str):
    pass


class ModelType(str):
    pass


class LangDet(str):
    pass


class RapidOCR:
    def __init__(self, params=None):
        self.params = params or {}

    def __call__(self, bgr):
        # Encode sample index from red channel (bgr[..., 2])
        idx = int(bgr[0, 0, 2])
        if idx % 2 == 1:
            return SimpleNamespace(boxes=[], txts=[], scores=[])
        return SimpleNamespace(
            boxes=[[[2.0, 4.0], [50.0, 4.0], [50.0, 20.0], [2.0, 20.0]]],
            txts=[f"sample_{idx}"],
            scores=[0.9],
        )
""".lstrip(),
        encoding="utf-8",
    )

    records = []
    for idx in range(6):
        rel = f"gt/img_{idx}.png"
        _write_rgb_png(tmp_path / rel, (64, 48), color=(idx, 10, 20))
        rec: dict[str, Any] = {"gt": rel, "sample_id": idx}
        if idx < 3:
            rec["text_lines"] = [{"transcript": "OLD", "bbox": [0, 0, 10, 10]}]
        records.append(rec)

    in_json = tmp_path / "in.json"
    out_json = tmp_path / "out.json"
    in_json.write_text(json.dumps(records), encoding="utf-8")

    # Point det/rec model dirs at dummy .onnx files so no download lock / network is touched
    dummy_det = tmp_path / "det.onnx"
    dummy_rec = tmp_path / "rec.onnx"
    dummy_det.write_bytes(b"stub")
    dummy_rec.write_bytes(b"stub")

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(stub_dir), str(REPO_ROOT / "src"), existing_pp] if p
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input-json",
            str(in_json),
            "--output-json",
            str(out_json),
            "--root-dir",
            str(tmp_path),
            "--devices",
            "cpu",
            "--workers-per-device",
            "2",
            "--chunk-size",
            "2",
            "--det-model-dir",
            str(dummy_det),
            "--rec-model-dir",
            str(dummy_rec),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    out = json.loads(out_json.read_text(encoding="utf-8"))
    assert [r["sample_id"] for r in out] == list(range(6))
    for idx, r in enumerate(out):
        if idx % 2 == 1:
            assert r["text_lines"] == []
        else:
            assert len(r["text_lines"]) == 1
            assert r["text_lines"][0]["transcript"] == f"sample_{idx}"
