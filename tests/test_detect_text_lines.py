"""Tests for shared RapidOCR utilities and offline resumable detection script (tools/detect_text_lines.py)."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
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
from utils.rapidocr import create_rapidocr_engine, strip_accelerator_device_params

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "detect_text_lines.py"


def _load_detect_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("detect_text_lines_mod", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_rgb_png(
    path: Path,
    size: tuple[int, int] = (64, 48),
    color: tuple[int, int, int] = (128, 64, 32),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _premature_zero_exit_worker() -> None:
    """Top-level spawnable target that exits with code 0 without sending completion."""
    sys.exit(0)


def test_shared_rapidocr_engine_caches_download_and_avoids_serialized_gpu_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "rapidocr_cache"
    lock_entries = 0
    init_calls: list[dict[str, Any]] = []
    eager_loads: list[str] = []

    class _FakeLock:
        def __init__(self, _path: Any) -> None:
            pass

        def __enter__(self) -> None:
            nonlocal lock_entries
            lock_entries += 1

        def __exit__(self, *_exc: Any) -> None:
            return None

    class _FakeRapidOCR:
        def __init__(self, *, params: dict[str, Any]) -> None:
            init_calls.append(dict(params))

        def _load_det_model(self) -> None:
            eager_loads.append("det")
            # Simulate RapidOCR 3.10 lazy download of PyTorch weights into model_root.
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "ch_PP-OCRv5_det_server.pth").write_bytes(b"fake-pth-det-bytes")

        def _load_rec_model(self) -> None:
            eager_loads.append("rec")
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "ch_PP-OCRv5_rec_server.pth").write_bytes(b"fake-pth-rec-bytes")
            (model_root / "ppocrv5_dict.txt").write_text("a\nb\n", encoding="utf-8")

    monkeypatch.setattr("huggingface_hub.utils.WeakFileLock", _FakeLock)
    fake_rapidocr = types.SimpleNamespace(RapidOCR=_FakeRapidOCR)

    gpu0_params = {
        "Global.use_det": True,
        "Global.use_rec": True,
        "Global.use_cls": False,
        "Global.model_root_dir": str(model_root),
        "EngineConfig.torch.use_cuda": True,
        "EngineConfig.torch.cuda_ep_cfg.device_id": 0,
    }
    gpu1_params = {
        "Global.use_det": True,
        "Global.use_rec": True,
        "Global.use_cls": False,
        "Global.model_root_dir": str(model_root),
        "EngineConfig.torch.use_cuda": True,
        "EngineConfig.torch.cuda_ep_cfg.device_id": 1,
    }
    assert "EngineConfig.torch.use_cuda" not in strip_accelerator_device_params(gpu0_params)
    assert "EngineConfig.torch.use_npu" not in strip_accelerator_device_params(
        {"EngineConfig.torch.use_npu": True, "EngineConfig.torch.npu_ep_cfg.device_id": 2}
    )
    assert "EngineConfig.torch.use_mps" not in strip_accelerator_device_params(
        {"EngineConfig.torch.use_mps": True}
    )

    # First cold-start call acquires WeakFileLock once for CPU download, then initializes GPU outside lock.
    create_rapidocr_engine(
        gpu0_params,
        rapidocr=fake_rapidocr,
        model_root=model_root,
        download_keys=("torch:det:PP-OCRv5:server:ch", "torch:rec:PP-OCRv5:server:ch"),
    )
    assert lock_entries == 1
    assert eager_loads == ["det", "rec"]
    assert len(init_calls) == 2
    assert "EngineConfig.torch.use_cuda" not in init_calls[0]
    assert init_calls[1]["EngineConfig.torch.cuda_ep_cfg.device_id"] == 0

    # Subsequent calls on other GPUs hit the ready cache and NEVER acquire WeakFileLock or run CPU init!
    create_rapidocr_engine(
        gpu1_params,
        rapidocr=fake_rapidocr,
        model_root=model_root,
        download_keys=("torch:det:PP-OCRv5:server:ch", "torch:rec:PP-OCRv5:server:ch"),
    )
    assert lock_entries == 1
    assert eager_loads == ["det", "rec"]
    assert len(init_calls) == 3
    assert init_calls[2]["EngineConfig.torch.cuda_ep_cfg.device_id"] == 1


def test_parse_devices_and_rapidocr_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_detect_module()
    # Point the project weights dir at an empty temporary tree so no local .pth is picked up.
    monkeypatch.setenv("SRTRAINER_WEIGHTS_DIR", str(tmp_path / "weights"))

    assert mod.parse_devices("cuda:0,cuda:2", workers_per_device=1) == ["cuda:0", "cuda:2"]
    assert mod.parse_devices("gpu:1,gpu:3", workers_per_device=2) == [
        "cuda:1",
        "cuda:1",
        "cuda:3",
        "cuda:3",
    ]
    assert mod.parse_devices("0,1,2", device_type="npu") == ["npu:0", "npu:1", "npu:2"]
    assert mod.parse_devices("mps,mps:0") == ["mps", "mps"]
    assert mod.parse_devices("cpu", workers_per_device=3) == ["cpu", "cpu", "cpu"]

    with pytest.raises(ValueError, match="Invalid device entry"):
        mod.parse_devices("tpu:0")
    with pytest.raises(ValueError, match="Invalid device entry"):
        mod.parse_devices("mps:1")
    with pytest.raises(ValueError, match="workers_per_device"):
        mod.parse_devices("cpu", workers_per_device=0)

    fake_rapidocr = types.SimpleNamespace(
        EngineType=types.SimpleNamespace(TORCH="torch"),
        OCRVersion=lambda v: f"ver:{v}",
        ModelType=lambda m: f"type:{m}",
        LangDet=lambda lang: f"lang:{lang}",
    )
    monkeypatch.setattr("utils.rapidocr._is_npu_available", lambda: (True, 8))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    cfg = mod.DetectConfig(det_only=False, use_cls=True, limit_side_len=1280, intra_threads=2)
    params, download_keys, _ = mod.build_rapidocr_params(cfg, "npu:3", fake_rapidocr)
    assert download_keys == (
        "torch:det:PP-OCRv5:server:ch",
        "torch:rec:PP-OCRv5:server:ch",
        "torch:cls:PP-OCRv4:mobile",
    )
    assert params["Global.use_det"] is True
    assert params["Global.use_rec"] is True
    assert params["Global.use_cls"] is True
    assert params["Det.engine_type"] == "torch"
    assert params["Rec.engine_type"] == "torch"
    assert params["Cls.engine_type"] == "torch"
    assert params["Cls.ocr_version"] == "ver:PP-OCRv4"
    assert params["Cls.model_type"] == "type:mobile"
    assert params["Det.limit_side_len"] == 1280
    assert params["EngineConfig.torch.use_npu"] is True
    assert params["EngineConfig.torch.npu_ep_cfg.device_id"] == 3

    cfg_det_only = mod.DetectConfig(det_only=True)
    params_det, det_keys, _ = mod.build_rapidocr_params(cfg_det_only, "cuda:1", fake_rapidocr)
    assert det_keys == ("torch:det:PP-OCRv5:server:ch",)
    assert params_det["Global.use_rec"] is False
    assert "Rec.model_path" not in params_det
    assert params_det["EngineConfig.torch.use_cuda"] is True
    assert params_det["EngineConfig.torch.cuda_ep_cfg.device_id"] == 1

    params_mps, _, _ = mod.build_rapidocr_params(cfg_det_only, "mps", fake_rapidocr)
    assert params_mps["EngineConfig.torch.use_mps"] is True

    thread_calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: thread_calls.append(n))
    monkeypatch.setattr(mod, "import_rapidocr", lambda **_kw: fake_rapidocr)
    monkeypatch.setattr(mod, "create_rapidocr_engine", lambda *a, **kw: "fake_engine")
    assert mod._create_rapidocr_engine(cfg, "cpu", skip_download_lock=True) == "fake_engine"
    assert thread_calls == [2]


def test_rapidocr_default_det_rec_paths_use_project_weights_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Det/rec weights default to <weights>/rapidocr instead of RapidOCR's default paths."""
    from utils import rapidocr as rapidocr_utils

    weights_dir = tmp_path / "weights"
    model_root = weights_dir / rapidocr_utils.RAPIDOCR_WEIGHTS_DIRNAME
    model_root.mkdir(parents=True)
    monkeypatch.setenv(rapidocr_utils.WEIGHTS_DIR_ENV_VAR, str(weights_dir))

    assert rapidocr_utils.get_weights_dir() == weights_dir
    assert rapidocr_utils.get_rapidocr_model_root() == model_root
    assert (
        rapidocr_utils.default_torch_model_path(rapidocr_utils.DEFAULT_DET_MODEL)
        == model_root / "ch_PP-OCRv5_det_server.pth"
    )
    assert (
        rapidocr_utils.default_torch_model_path(rapidocr_utils.DEFAULT_REC_MODEL)
        == model_root / "ch_PP-OCRv5_rec_server.pth"
    )

    det_path = model_root / "ch_PP-OCRv5_det_server.pth"
    rec_path = model_root / "ch_PP-OCRv5_rec_server.pth"
    det_path.write_bytes(b"stub-det")
    rec_path.write_bytes(b"stub-rec")
    dict_path = model_root / "ppocrv5_dict.txt"
    dict_path.write_text("a\nb\n", encoding="utf-8")

    fake_rapidocr = types.SimpleNamespace(
        EngineType=types.SimpleNamespace(TORCH="torch"),
        OCRVersion=lambda v: f"ver:{v}",
        ModelType=lambda m: f"type:{m}",
        LangDet=lambda lang: f"lang:{lang}",
    )
    mod = _load_detect_module()
    params, download_keys, resolved_root = mod.build_rapidocr_params(
        mod.DetectConfig(), "cpu", fake_rapidocr
    )
    assert resolved_root == model_root
    assert params["Global.model_root_dir"] == str(model_root)
    assert params["Det.model_path"] == str(det_path)
    assert params["Rec.model_path"] == str(rec_path)
    assert params["Rec.rec_keys_path"] == str(dict_path)
    # Local project weights are complete: nothing needs to be downloaded.
    assert download_keys == ()

    # RapidOCR's own default (site-packages/rapidocr/models) is never used as a fallback.
    assert "site-packages" not in str(model_root)

    # Without the override the project-local <repo>/experiments/weights directory wins.
    monkeypatch.delenv(rapidocr_utils.WEIGHTS_DIR_ENV_VAR)
    assert (
        rapidocr_utils.get_weights_dir()
        == rapidocr_utils.get_project_root() / "experiments" / "weights"
    )


def test_extract_valid_text_lines_clamps_and_drops_degenerate_boxes() -> None:
    mod = _load_detect_module()
    fake_result = types.SimpleNamespace(
        boxes=[
            [[-4.0, 2.0], [70.0, 2.0], [70.0, 20.0], [-4.0, 20.0]],
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [40.0, 10.0]],
            [[5.0, 5.0], [6.5, 5.0], [6.5, 6.5], [5.0, 6.5]],
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
    assert summary.resumed_samples == 0
    assert summary.processed_samples == 2
    assert summary.total_lines == 3
    assert summary.empty_samples == 0
    assert output_json.is_file()
    # Sidecar checkpoint cleaned up automatically after successful completion
    assert not mod.default_checkpoint_path(output_json).exists()

    assert json.loads(input_json.read_text(encoding="utf-8")) == input_payload
    out_records = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(out_records) == 2
    assert out_records[0]["extra_meta"] == "keep_me"
    assert [ln["transcript"] for ln in out_records[0]["text_lines"]] == ["NEW_TEXT_0"]
    assert [ln["transcript"] for ln in out_records[1]["text_lines"]] == ["LINE_A", "LINE_B"]

    # End-to-end: generated JSON loads cleanly into load_train_manifest -> MixedSRDataset -> OCRKLLoss
    samples, _ = load_train_manifest(output_json, tmp_path)
    assert len(samples) == 2
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


def test_detect_manifest_incremental_checkpoint_auto_resume_torn_tail_and_mismatch(
    tmp_path: Path,
) -> None:
    mod = _load_detect_module()

    records = []
    for idx in range(5):
        rel = f"gt/img_{idx}.png"
        _write_rgb_png(tmp_path / rel, (64, 48), color=(idx, 10, 20))
        records.append({"gt": rel, "sample_id": idx})

    input_json = tmp_path / "train.json"
    output_json = tmp_path / "out.json"
    ckpt_path = mod.default_checkpoint_path(output_json)
    input_json.write_text(json.dumps(records), encoding="utf-8")

    # Phase 1: Crash on sample index 2 -> first 2 samples must already be flushed to checkpoint
    calls_phase1: list[int] = []

    class _CrashingEngine:
        def __call__(self, bgr: np.ndarray) -> Any:
            idx = int(bgr[0, 0, 2])
            calls_phase1.append(idx)
            if idx == 2:
                raise RuntimeError("Simulated mid-run OOM / crash")
            return types.SimpleNamespace(
                boxes=[[[4.0, 6.0], [50.0, 6.0], [50.0, 22.0], [4.0, 22.0]]],
                txts=[f"txt_{idx}"],
                scores=[0.9],
            )

    with pytest.raises(RuntimeError, match="Simulated mid-run"):
        mod.run_detect_manifest(
            input_json=input_json,
            output_json=output_json,
            root_dir=tmp_path,
            devices="cpu",
            engine_factory=lambda _cfg, _dev: _CrashingEngine(),
        )
    assert calls_phase1 == [0, 1, 2]
    assert ckpt_path.is_file()
    assert not output_json.exists()

    # Append a torn trailing JSON line to simulate SIGKILL mid-write
    with ckpt_path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "result", "idx": 2, "gt": "gt/img_2.png", "text_lin')

    # Mismatched OCR config refuses silent stale resume unless resume=False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        mod.run_detect_manifest(
            input_json=input_json,
            output_json=output_json,
            root_dir=tmp_path,
            devices="cpu",
            config=mod.DetectConfig(det_only=True),
            engine_factory=lambda _cfg, _dev: _CrashingEngine(),
        )

    # Phase 2: Re-run with default resume=True -> auto-recovers 0 & 1, drops torn tail, runs only 2..4
    calls_phase2: list[int] = []

    class _HealthyEngine:
        def __call__(self, bgr: np.ndarray) -> Any:
            idx = int(bgr[0, 0, 2])
            calls_phase2.append(idx)
            return types.SimpleNamespace(
                boxes=[[[4.0, 6.0], [50.0, 6.0], [50.0, 22.0], [4.0, 22.0]]],
                txts=[f"txt_{idx}"],
                scores=[0.91],
            )

    summary = mod.run_detect_manifest(
        input_json=input_json,
        output_json=output_json,
        root_dir=tmp_path,
        devices="cpu",
        engine_factory=lambda _cfg, _dev: _HealthyEngine(),
    )
    assert calls_phase2 == [2, 3, 4]
    assert summary.resumed_samples == 2
    assert summary.processed_samples == 3
    assert summary.total_samples == 5
    assert not ckpt_path.exists()

    out = json.loads(output_json.read_text(encoding="utf-8"))
    assert [r["sample_id"] for r in out] == [0, 1, 2, 3, 4]
    assert [r["text_lines"][0]["transcript"] for r in out] == [
        "txt_0",
        "txt_1",
        "txt_2",
        "txt_3",
        "txt_4",
    ]


def test_drain_multiprocess_workers_fails_fast_on_premature_exitcode_zero() -> None:
    """Verify the main queue loop never hangs when a worker exits early with exitcode == 0."""
    mod = _load_detect_module()
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_premature_zero_exit_worker)
    proc.start()

    with pytest.raises(RuntimeError, match="exitcode=0"):
        mod._drain_multiprocess_workers(
            [proc],
            ["cpu"],
            result_q,
            total_chunks=2,
            on_chunk=lambda _chunk: None,
        )


def test_cli_multiprocess_spawn_auto_resume_and_order_with_stub_rapidocr(tmp_path: Path) -> None:
    """Exercise true multiprocessing ('spawn') CLI execution + auto-resume across 2 workers."""
    stub_dir = tmp_path / "stub_pkgs"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "rapidocr.py").write_text(
        """
import os
from types import SimpleNamespace


class EngineType:
    TORCH = "torch"


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
        idx = int(bgr[0, 0, 2])
        fail_idx = os.environ.get("STUB_FAIL_IDX")
        if fail_idx is not None and idx == int(fail_idx):
            raise RuntimeError(f"Injected worker failure on sample {idx}")
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

    dummy_det = tmp_path / "ch_PP-OCRv5_det_server.pth"
    dummy_rec = tmp_path / "ch_PP-OCRv5_rec_server.pth"
    dummy_dict = tmp_path / "ppocrv5_dict.txt"
    dummy_det.write_bytes(b"stub")
    dummy_rec.write_bytes(b"stub")
    dummy_dict.write_text("a\nb\n", encoding="utf-8")

    base_env = os.environ.copy()
    existing_pp = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = os.pathsep.join(
        p for p in [str(stub_dir), str(REPO_ROOT / "src"), existing_pp] if p
    )
    base_env["SRTRAINER_WEIGHTS_DIR"] = str(tmp_path / "weights")
    cmd = [
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
    ]

    # Run 1: Fail on sample 5 so earlier chunks flush to checkpoint
    env_fail = dict(base_env, STUB_FAIL_IDX="5")
    proc_fail = subprocess.run(cmd, env=env_fail, capture_output=True, text=True, timeout=60, check=False)
    assert proc_fail.returncode != 0
    ckpt_file = out_json.parent / f"{out_json.name}.ckpt.jsonl"
    assert ckpt_file.is_file()
    assert not out_json.exists()

    # Run 2: Re-run identical command without STUB_FAIL_IDX -> auto-resumes & completes
    proc_ok = subprocess.run(cmd, env=base_env, capture_output=True, text=True, timeout=60, check=False)
    assert proc_ok.returncode == 0, f"CLI failed:\nSTDOUT:\n{proc_ok.stdout}\nSTDERR:\n{proc_ok.stderr}"
    assert not ckpt_file.exists()

    out = json.loads(out_json.read_text(encoding="utf-8"))
    assert [r["sample_id"] for r in out] == list(range(6))
    for idx, r in enumerate(out):
        if idx % 2 == 1:
            assert r["text_lines"] == []
        else:
            assert len(r["text_lines"]) == 1
            assert r["text_lines"][0]["transcript"] == f"sample_{idx}"
