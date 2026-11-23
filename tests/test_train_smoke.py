"""Milestone 5 unit, integration, and 2-process CPU distributed training smoke tests (F41-F48)."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import socket
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from torch import nn

import engine as engine_pkg
import models as models_pkg
from engine import Checkpointer, TiledInferencer, Trainer, TrainState
from metrics.base import BaseMetric
from models import BaseModel, ModelEMA, SRModel
from models.ema import unwrap_model_for_ema
from tests.dummy_arch import DummySRNet
from utils.config import load_resolved_config
from utils.image import load_image
from utils.misc import load_weights, save_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO_ROOT / "configs")
TOOLS_TRAIN = REPO_ROOT / "tools" / "train.py"
TOOLS_INFER = REPO_ROOT / "tools" / "infer.py"


# ---------------------------------------------------------------------------
# Custom test metrics (importable via Hydra _target_)
# ---------------------------------------------------------------------------


class MockOCRAMetric(BaseMetric):
    """Auxiliary-label, GT-less validation metric verifying ``meta`` passthrough."""

    higher_is_better: bool = True

    def __init__(self, label_key: str = "label") -> None:
        self.label_key = label_key
        self.seen_records: list[dict[str, Any]] = []

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        assert target is None, f"Expected target=None for GT-less OCR suite, got {type(target)}"
        assert meta is not None and self.label_key in meta
        label = str(meta[self.label_key])
        self.seen_records.append(
            {
                "label": label,
                "tag": meta.get("tag"),
                "index": meta.get("index"),
                "pred_shape": tuple(pred.shape),
            }
        )
        return float(len(label)) / 10.0 + float(pred.mean().item()) * 0.01


class MetaScoreMetric(BaseMetric):
    """Deterministic metric returning ``float(meta[score_key])`` for exact DDP gather math."""

    higher_is_better: bool = True

    def __init__(self, score_key: str = "score") -> None:
        self.score_key = score_key

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        assert meta is not None and self.score_key in meta
        return float(meta[self.score_key])


class ScriptedSequenceMetric(BaseMetric):
    """Metric emitting a predefined sequence of values per call (for best/NaN testing)."""

    higher_is_better: bool = False

    def __init__(
        self,
        sequence: list[Any],
        higher_is_better: bool = False,
    ) -> None:
        self.higher_is_better = higher_is_better
        self.sequence: list[float] = [float(v) for v in sequence]
        self.call_idx: int = 0

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        idx = min(self.call_idx, len(self.sequence) - 1)
        self.call_idx += 1
        return self.sequence[idx]


class SentinelUninstantiatedLoss(nn.Module):
    """Loss that raises if ever instantiated (verifies weight <= 0 skips instantiation)."""

    def __init__(self) -> None:
        super().__init__()
        raise AssertionError("Zero/negative weight loss must NEVER be instantiated!")


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _to_hydra_val(obj: Any) -> str:
    """Format a Python value using Hydra CLI override grammar (unquoted dict keys)."""
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
        return "{" + ",".join(f"{k}:{_to_hydra_val(v)}" for k, v in obj.items()) + "}"
    raise TypeError(f"Unsupported Hydra CLI value type: {type(obj).__name__}")


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
def cleanup_state():
    _reset_accelerate_and_loggers()
    yield
    _reset_accelerate_and_loggers()


def _block_network_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def _deny(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Unexpected network access during offline smoke test!")

    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(urllib.request, "urlopen", _deny)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _deny)


def _write_rgb_png(path: Path, h: int, w: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(16, 240, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _create_synthetic_datasets(base_dir: Path, scale: int = 2) -> dict[str, Any]:
    """Create compact synthetic gt_only, paired, ref val, OCR val, odd-5 val, and 1-img val data."""
    data_root = base_dir / "datasets"
    manifests_dir = base_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Training source A: gt_only (4 images of 36x36)
    gt_only_entries: list[dict[str, Any]] = []
    for i in range(4):
        rel = f"gt_only/hr_{i:02d}.png"
        _write_rgb_png(data_root / rel, 36, 36, seed=100 + i)
        gt_only_entries.append({"gt": rel})
    gt_only_json = manifests_dir / "train_gt_only.json"
    gt_only_json.write_text(json.dumps(gt_only_entries), encoding="utf-8")

    # Training source B: paired (4 images gt=36x36, lq=18x18)
    paired_entries: list[dict[str, Any]] = []
    for i in range(4):
        gt_rel = f"paired/hr_{i:02d}.png"
        lq_rel = f"paired/lr_{i:02d}.png"
        _write_rgb_png(data_root / gt_rel, 36, 36, seed=200 + i)
        _write_rgb_png(data_root / lq_rel, 36 // scale, 36 // scale, seed=300 + i)
        paired_entries.append({"gt": gt_rel, "lq": lq_rel})
    paired_json = manifests_dir / "train_paired.json"
    paired_json.write_text(json.dumps(paired_entries), encoding="utf-8")

    # Validation suite 1: val_ref (3 images, variable sizes >= 32 HR, including duplicate stem!)
    val_ref_specs = [
        ("val_ref/sub_a/dup.png", "val_ref/sub_a_gt/dup.png", 16, 16, 401),
        ("val_ref/sub_b/dup.png", "val_ref/sub_b_gt/dup.png", 18, 20, 402),
        ("val_ref/unique_img.png", "val_ref/unique_gt.png", 16, 16, 403),
    ]
    val_ref_entries: list[dict[str, Any]] = []
    for lq_rel, gt_rel, lh, lw, sd in val_ref_specs:
        _write_rgb_png(data_root / lq_rel, lh, lw, seed=sd)
        _write_rgb_png(data_root / gt_rel, lh * scale, lw * scale, seed=sd + 50)
        val_ref_entries.append({"lq": lq_rel, "gt": gt_rel})
    val_ref_json = manifests_dir / "val_ref.json"
    val_ref_json.write_text(json.dumps(val_ref_entries), encoding="utf-8")

    # Validation suite 2: val_ocr_nogt (2 images, NO gt, extra label + nested tag)
    val_ocr_entries: list[dict[str, Any]] = []
    for i, label in enumerate(("HELLO", "WORLD")):
        lq_rel = f"val_ocr/text_{i:02d}.png"
        _write_rgb_png(data_root / lq_rel, 16, 18, seed=500 + i)
        val_ocr_entries.append(
            {"lq": lq_rel, "label": label, "tag": {"split": "hard", "id": i}}
        )
    val_ocr_json = manifests_dir / "val_ocr_nogt.json"
    val_ocr_json.write_text(json.dumps(val_ocr_entries), encoding="utf-8")

    # Validation suite 3: val_odd5 (5 images with distinct deterministic scores)
    odd5_scores = [10.0, 20.0, 30.0, 40.0, 100.0]  # Exact mean = 40.0; padded-6 mean = 35.0
    val_odd5_entries: list[dict[str, Any]] = []
    for i, sc in enumerate(odd5_scores):
        lq_rel = f"val_odd5/img_{i:02d}.png"
        gt_rel = f"val_odd5/gt_{i:02d}.png"
        lh = 16 if i % 2 == 0 else 18
        lw = 16 if i % 2 == 0 else 20
        _write_rgb_png(data_root / lq_rel, lh, lw, seed=600 + i)
        _write_rgb_png(data_root / gt_rel, lh * scale, lw * scale, seed=650 + i)
        val_odd5_entries.append({"lq": lq_rel, "gt": gt_rel, "score": sc})
    val_odd5_json = manifests_dir / "val_odd5.json"
    val_odd5_json.write_text(json.dumps(val_odd5_entries), encoding="utf-8")

    # Validation suite 4: val_tiny (1 image: N=1 < world_size=2)
    _write_rgb_png(data_root / "val_tiny/only.png", 16, 16, seed=700)
    _write_rgb_png(data_root / "val_tiny/only_gt.png", 16 * scale, 16 * scale, seed=750)
    val_tiny_json = manifests_dir / "val_tiny.json"
    val_tiny_json.write_text(
        json.dumps([{"lq": "val_tiny/only.png", "gt": "val_tiny/only_gt.png", "score": 77.5}]),
        encoding="utf-8",
    )

    return {
        "data_root": data_root,
        "gt_only_json": gt_only_json,
        "paired_json": paired_json,
        "val_ref_json": val_ref_json,
        "val_ocr_json": val_ocr_json,
        "val_odd5_json": val_odd5_json,
        "val_tiny_json": val_tiny_json,
        "odd5_scores": odd5_scores,
    }


def _make_base_train_cfg(
    work_dir: Path,
    ds: dict[str, Any],
    *,
    total_iters: int = 4,
    log_interval: int = 2,
    val_interval: int = 2,
    save_interval: int = 2,
    grad_accum: int = 1,
    max_keep_states: int = 2,
    mixed_precision: str = "no",
    log_with: Any = None,
    use_lpips_loss: bool = False,
    use_degradation: bool = True,
    val_suites: dict[str, Any] | None = None,
    primary_dataset: str | None = "val_ref",
    primary_metric: str | None = "psnr",
    val_use_ema: bool = True,
    val_save_visuals: bool = True,
    resume: Any = None,
    append_timestamp: bool = False,
    t_max: int | None = None,
    lr: float = 2.0e-4,
    ema_enabled: bool = True,
    ema_decay: float = 0.99,
    ema_warmup: int = 0,
    pretrain_path: str | None = None,
    pretrain_strict: bool = True,
    pretrain_key: str | None = None,
) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="train")

    OmegaConf.set_struct(cfg, False)
    cfg.exp_name = work_dir.name
    cfg.scale = 2
    cfg.seed = 42
    cfg.output_root = str(work_dir.parent)
    cfg.work_dir = str(work_dir)

    cfg.arch = OmegaConf.create(
        {
            "_target_": "tests.dummy_arch.DummySRNet",
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 8,
            "num_block": 1,
            "scale": 2,
        }
    )

    cfg.loss = OmegaConf.create(
        {
            "l1": {
                "weight": 1.0,
                "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
            }
        }
    )
    if use_lpips_loss:
        cfg.loss.lpips = OmegaConf.create(
            {
                "weight": 0.1,
                "criterion": {
                    "_target_": "losses.LPIPSLoss",
                    "net": "alex",
                    "pretrained": False,
                },
            }
        )

    cfg.model.ema.enabled = ema_enabled
    cfg.model.ema.decay = ema_decay
    cfg.model.ema.warmup_steps = ema_warmup
    cfg.model.pretrain_path = pretrain_path
    cfg.model.pretrain_strict = pretrain_strict
    cfg.model.pretrain_key = pretrain_key

    cfg.data.gt_size = 32
    cfg.data.scale = 2
    cfg.data.batch_size_per_gpu = 2
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.sources = OmegaConf.create(
        [
            {
                "name": "syn_gt_only",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["gt_only_json"]),
                "ratio": 0.6,
            },
            {
                "name": "syn_paired",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["paired_json"]),
                "ratio": 0.4,
            },
        ]
    )
    if use_degradation:
        cfg.data.degradation.scale = 2
        cfg.data.degradation.queue_size = 0
        cfg.data.degradation.kernel_cfg.kernel_size = 7
        cfg.data.degradation.kernel_cfg.kernel_range = [7]
    else:
        cfg.data.degradation = None

    cfg.optim.optimizer.lr = lr
    cfg.optim.scheduler.T_max = int(t_max if t_max is not None else total_iters)
    cfg.optim.scheduler.eta_min = 1.0e-6

    if val_suites is None:
        val_suites = {
            "val_ref": {
                "dataset": {
                    "root_dir": str(ds["data_root"]),
                    "json_path": str(ds["val_ref_json"]),
                },
                "metrics": {
                    "psnr": {
                        "_target_": "metrics.PSNRMetric",
                        "crop_border": 2,
                        "test_y_channel": True,
                    },
                    "ssim": {
                        "_target_": "metrics.SSIMMetric",
                        "crop_border": 2,
                        "test_y_channel": True,
                    },
                    "lpips": {
                        "_target_": "metrics.LPIPSMetric",
                        "net": "alex",
                        "pretrained": False,
                    },
                },
            }
        }
    cfg.metric = OmegaConf.create(
        {
            "primary_dataset": primary_dataset,
            "primary_metric": primary_metric,
            "val_suites": val_suites,
        }
    )

    cfg.trainer.total_iters = total_iters
    cfg.trainer.log_interval = log_interval
    cfg.trainer.val_interval = val_interval
    cfg.trainer.save_interval = save_interval
    cfg.trainer.max_keep_states = max_keep_states
    cfg.trainer.mixed_precision = mixed_precision
    cfg.trainer.gradient_accumulation_steps = grad_accum
    cfg.trainer.grad_clip_norm = 1.0
    cfg.trainer.compile = False
    cfg.trainer.log_with = log_with
    cfg.trainer.resume = resume
    cfg.trainer.append_timestamp = append_timestamp
    cfg.trainer.val_save_visuals = val_save_visuals
    cfg.trainer.val_max_save_images = 8
    cfg.trainer.val_use_ema = val_use_ema
    cfg.trainer.val_tile.enabled = True
    cfg.trainer.val_tile.size = "auto"
    cfg.trainer.val_tile.overlap = 4
    cfg.trainer.val_tile.batch_size = 2
    cfg.trainer.val_tile.blend_mode = "cosine"
    cfg.trainer.val_tile.pad_multiple = 4
    return cfg


def _run_torchrun_train(
    overrides: list[str],
    *,
    cwd: Path,
    nproc: int = 2,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ACCELERATE_USE_CPU"] = "true"
    env["OMP_NUM_THREADS"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    extra_pp = f"{REPO_ROOT}:{REPO_ROOT / 'src'}"
    env["PYTHONPATH"] = f"{extra_pp}:{existing_pp}" if existing_pp else extra_pp

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(TOOLS_TRAIN),
        *overrides,
    ]
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Test 1: Exports & Trainer Fail-Fast Validation
# ---------------------------------------------------------------------------


def test_engine_and_models_exports_and_trainer_failfast_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network_downloads(monkeypatch)

    assert engine_pkg.__all__ == ["Trainer", "TrainState", "Checkpointer", "TiledInferencer"]
    assert models_pkg.__all__ == ["BaseModel", "SRModel", "ModelEMA"]
    assert issubclass(SRModel, BaseModel)
    with pytest.raises(TypeError):
        BaseModel({})  # type: ignore[abstract]

    ds = _create_synthetic_datasets(tmp_path)

    # 1. Bad primary_dataset
    cfg_bad_ds = _make_base_train_cfg(
        tmp_path / "bad_ds", ds, primary_dataset="nonexistent", primary_metric="psnr"
    )
    with pytest.raises(ValueError, match="primary_dataset"):
        Trainer(cfg_bad_ds)
    assert not (tmp_path / "bad_ds").exists()

    # 2. Bad primary_metric
    cfg_bad_metric = _make_base_train_cfg(
        tmp_path / "bad_metric", ds, primary_dataset="val_ref", primary_metric="nonexistent"
    )
    with pytest.raises(ValueError, match="primary_metric"):
        Trainer(cfg_bad_metric)
    assert not (tmp_path / "bad_metric").exists()

    # 3. Mismatched primary pair
    cfg_half_primary = _make_base_train_cfg(
        tmp_path / "half_primary", ds, primary_dataset="val_ref", primary_metric=None
    )
    with pytest.raises(ValueError, match="must be set together"):
        Trainer(cfg_half_primary)

    # 4. Invalid total_iters / grad_accum / resume=True / nonexistent explicit resume
    cfg_zero_iters = _make_base_train_cfg(tmp_path / "zero_iters", ds, total_iters=0)
    with pytest.raises(ValueError, match="total_iters"):
        Trainer(cfg_zero_iters)

    cfg_zero_accum = _make_base_train_cfg(tmp_path / "zero_accum", ds, grad_accum=0)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        Trainer(cfg_zero_accum)

    cfg_resume_true = _make_base_train_cfg(tmp_path / "resume_true", ds, resume=True)
    with pytest.raises(ValueError, match="resume"):
        Trainer(cfg_resume_true)

    cfg_resume_missing = _make_base_train_cfg(
        tmp_path / "resume_missing", ds, resume=str(tmp_path / "does_not_exist")
    )
    with pytest.raises(FileNotFoundError):
        Trainer(cfg_resume_missing)

    # 5. Legacy 'mode' key in data.sources rejected fail-fast
    cfg_legacy_mode = _make_base_train_cfg(tmp_path / "legacy_mode", ds)
    cfg_legacy_mode.data.sources[0]["mode"] = "gt_only"
    with pytest.raises(ValueError, match="mode"):
        Trainer(cfg_legacy_mode)

    # 6. gt_only samples with data.degradation=null rejected fail-fast
    cfg_no_deg = _make_base_train_cfg(tmp_path / "no_deg", ds, use_degradation=False)
    with pytest.raises(ValueError, match="degradation"):
        Trainer(cfg_no_deg)

    # 7. Non-mapping val_suites and empty val_suites with non-null primary rejected fail-fast
    cfg_scalar_suites = _make_base_train_cfg(tmp_path / "scalar_suites", ds)
    cfg_scalar_suites.metric.val_suites = 123
    with pytest.raises(ValueError, match="val_suites"):
        Trainer(cfg_scalar_suites)

    cfg_empty_bad_primary = _make_base_train_cfg(
        tmp_path / "empty_bad_primary",
        ds,
        val_suites={},
        primary_dataset="val_ref",
        primary_metric="psnr",
    )
    with pytest.raises(ValueError, match="primary_dataset"):
        Trainer(cfg_empty_bad_primary)

    # 8. Empty val_suites with primary_dataset=None & primary_metric=None gracefully skips validation
    empty_dir = tmp_path / "empty_val_exp"
    cfg_empty_val = _make_base_train_cfg(
        empty_dir,
        ds,
        total_iters=2,
        log_interval=1,
        val_interval=1,
        save_interval=2,
        val_suites={},
        primary_dataset=None,
        primary_metric=None,
    )
    trainer = Trainer(cfg_empty_val)
    state = trainer.fit()
    assert state.global_step == 2
    assert trainer.primary_dataset is None
    assert trainer.primary_metric is None
    assert list((empty_dir / "weights").glob("best_*.safetensors")) == []
    assert (empty_dir / "weights" / "latest.safetensors").is_file()


# ---------------------------------------------------------------------------
# Test 2: ModelEMA, TrainState, Checkpointer Unit Tests
# ---------------------------------------------------------------------------


class _BNToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class _FakeOrigModWrapper(nn.Module):
    def __init__(self, mod: nn.Module) -> None:
        super().__init__()
        self._orig_mod = mod


def test_model_ema_train_state_and_checkpointer_units(tmp_path: Path) -> None:
    # --- ModelEMA ---
    for bad_decay in (0.0, -0.5, 1.1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ModelEMA(_BNToy(), decay=bad_decay)  # type: ignore[arg-type]
    for bad_warmup in (-1, True):
        with pytest.raises(ValueError):
            ModelEMA(_BNToy(), decay=0.99, warmup_steps=bad_warmup)  # type: ignore[arg-type]

    # FP32 shadow copy from BF16 model preserving int64 buffer dtype
    src = _BNToy().to(dtype=torch.bfloat16)
    src.bn.num_batches_tracked.fill_(7)
    ema = ModelEMA(src, decay=0.9, warmup_steps=10, device="cpu")
    assert not ema.module.training
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in ema.module.parameters())
    assert ema.module.bn.running_mean.dtype == torch.float32
    assert ema.module.bn.num_batches_tracked.dtype == torch.int64
    assert int(ema.module.bn.num_batches_tracked.item()) == 7

    # Exact warmup decay formula: min(decay, (1 + num_updates) / (warmup_steps + num_updates))
    assert ema.get_decay() == pytest.approx(min(0.9, (1 + 0) / (10 + 0)))
    assert ema.get_decay(num_updates=1) == pytest.approx(min(0.9, 2 / 11))
    assert ema.get_decay(num_updates=100) == pytest.approx(0.9)

    ema_no_warmup = ModelEMA(src, decay=0.8, warmup_steps=0)
    assert ema_no_warmup.get_decay() == pytest.approx(0.8)

    # Exact FP32 lerp update math + buffer copy + atomic mismatch check
    before_w = ema.module.conv.weight.detach().clone()
    with torch.no_grad():
        src.conv.weight.fill_(2.0)
        src.bn.num_batches_tracked.fill_(11)
    # Wrapped in simulated _orig_mod
    wrapped = _FakeOrigModWrapper(src)
    assert unwrap_model_for_ema(wrapped) is src
    ema.update(wrapped)
    assert ema.num_updates == 1
    d1 = min(0.9, (1 + 1) / (10 + 1))
    expected_w = before_w * d1 + torch.full_like(before_w, 2.0) * (1.0 - d1)
    assert torch.allclose(ema.module.conv.weight, expected_w, atol=1e-6)
    assert int(ema.module.bn.num_batches_tracked.item()) == 11

    # Parameter mismatch raises ValueError BEFORE mutating num_updates
    with pytest.raises(ValueError, match="mismatch"):
        ema.update(nn.Linear(3, 3))
    assert ema.num_updates == 1

    # weights_only=True safe state_dict / load_state_dict round-trip
    ema_sd_path = tmp_path / "ema_state.pt"
    torch.save(ema.state_dict(), ema_sd_path)
    loaded_ema_sd = torch.load(ema_sd_path, map_location="cpu", weights_only=True)
    ema2 = ModelEMA(_BNToy(), decay=0.5, warmup_steps=0)
    ema2.load_state_dict(loaded_ema_sd)
    assert ema2.num_updates == 1
    assert ema2.decay == pytest.approx(0.9)
    assert ema2.warmup_steps == 10
    assert torch.equal(ema2.module.conv.weight, ema.module.conv.weight)
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in ema2.module.parameters())

    # --- TrainState ---
    ts = TrainState()
    assert ts.global_step == 0 and ts.best_metrics == {}
    with pytest.raises(ValueError):
        TrainState(global_step=-1)
    with pytest.raises(ValueError):
        TrainState(global_step=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TrainState(best_metrics={"psnr": float("nan")})

    # +inf allowed (exact-match PSNR), NaN ignored in update_best
    assert ts.update_best("set5/psnr", 30.0, higher_is_better=True) is True
    assert ts.update_best("set5/psnr", float("nan"), higher_is_better=True) is False
    assert ts.best_metrics["set5/psnr"] == 30.0
    assert ts.update_best("set5/psnr", float("inf"), higher_is_better=True) is True
    assert ts.update_best("set5/psnr", float("inf"), higher_is_better=True) is False
    assert ts.update_best("set5/lpips", 0.2, higher_is_better=False) is True
    assert ts.update_best("set5/lpips", 0.3, higher_is_better=False) is False
    assert ts.update_best("set5/lpips", 0.1, higher_is_better=False) is True

    # Defensive copy + atomic rejection on invalid load_state_dict
    ts.global_step = 42
    snap = ts.state_dict()
    snap["best_metrics"]["set5/psnr"] = -999.0
    assert math.isinf(ts.best_metrics["set5/psnr"])

    ts_path = tmp_path / "train_state.pt"
    torch.save(ts.state_dict(), ts_path)
    raw_ts = torch.load(ts_path, map_location="cpu", weights_only=True)
    ts_restored = TrainState()
    ts_restored.load_state_dict(raw_ts)
    assert ts_restored.global_step == 42
    assert math.isinf(ts_restored.best_metrics["set5/psnr"])
    assert ts_restored.best_metrics["set5/lpips"] == pytest.approx(0.1)

    with pytest.raises(ValueError):
        ts_restored.load_state_dict({"global_step": 99, "best_metrics": {"x": float("nan")}})
    assert ts_restored.global_step == 42  # Unchanged on error!

    # --- Checkpointer ---
    ckpt_work = tmp_path / "ckpt_test"
    ckpt = Checkpointer(ckpt_work, max_keep_states=2)
    assert ckpt.find_latest_state() is None
    assert ckpt.resolve_resume_path(None) is None
    assert ckpt.resolve_resume_path("null") is None
    assert ckpt.resolve_resume_path("") is None
    assert ckpt.resolve_resume_path("auto") is None
    with pytest.raises(ValueError):
        ckpt.resolve_resume_path(True)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        ckpt.resolve_resume_path(tmp_path / "missing_dir")

    acc = Accelerator(mixed_precision="no")
    toy = DummySRNet(num_feat=8, num_block=1, scale=2)
    opt = torch.optim.AdamW(toy.parameters(), lr=1e-3)
    toy, opt = acc.prepare(toy, opt)
    acc.register_for_checkpointing(ts_restored)

    # Create stray file & non-matching subdir to prove pruning ignores them
    (ckpt_work / "checkpoints").mkdir(parents=True, exist_ok=True)
    stray_file = ckpt_work / "checkpoints" / "state_iter_9999999"
    stray_file.write_text("stray file", encoding="utf-8")
    stray_dir = ckpt_work / "checkpoints" / "backup_keep"
    stray_dir.mkdir()

    # Numerical pruning across digit widths (2 < 999999 < 1000000!)
    ckpt.save_state(acc, step=2)
    ckpt.save_state(acc, step=999999)
    ckpt.save_state(acc, step=1000000)

    remaining = sorted(p.name for p in (ckpt_work / "checkpoints").iterdir() if p.is_dir())
    assert remaining == ["backup_keep", "state_iter_1000000", "state_iter_999999"]
    assert stray_file.is_file()
    assert ckpt.find_latest_state() == ckpt_work / "checkpoints" / "state_iter_1000000"
    assert ckpt.resolve_resume_path("auto") == ckpt_work / "checkpoints" / "state_iter_1000000"
    with pytest.raises(ValueError):
        ckpt.resolve_resume_path(stray_file)

    # max_keep_states <= 0 or None disables pruning
    ckpt_all = Checkpointer(tmp_path / "ckpt_keep_all", max_keep_states=0)
    assert ckpt_all.max_keep_states is None
    ckpt_all.save_state(acc, step=1)
    ckpt_all.save_state(acc, step=2)
    ckpt_all.save_state(acc, step=3)
    assert len(list((tmp_path / "ckpt_keep_all" / "checkpoints").glob("state_iter_*"))) == 3

    # Track 2 atomic safetensors with prefix stripping & zero leftover .tmp files
    prefixed_sd = {f"module._orig_mod.{k}": v for k, v in toy.state_dict().items()}
    saved = ckpt.save_weights(acc, {"net_g": prefixed_sd, "net_g_ema": prefixed_sd}, step=10)
    assert (ckpt_work / "weights" / "iter_000010.safetensors").is_file()
    assert (ckpt_work / "weights" / "iter_000010_ema.safetensors").is_file()
    assert (ckpt_work / "weights" / "latest.safetensors").is_file()
    assert (ckpt_work / "weights" / "latest_ema.safetensors").is_file()
    assert not (ckpt_work / "weights" / "latest.safetensors").is_symlink()
    assert set(saved.keys()) == {"net_g", "latest", "net_g_ema", "latest_ema"}

    loaded_st = load_safetensors_file(str(ckpt_work / "weights" / "latest.safetensors"))
    assert all(not k.startswith(("module.", "_orig_mod.")) for k in loaded_st)

    best_path = ckpt.save_best(acc, prefixed_sd, "set5", "psnr", step=10, metric_value=31.5)
    assert best_path == ckpt_work / "weights" / "best_set5_psnr.safetensors"
    assert best_path.is_file()
    assert list((ckpt_work / "weights").glob("*.tmp")) == []

    acc.end_training()


# ---------------------------------------------------------------------------
# Test 3: SRModel 7-Step Setup, Pretrain, Losses, LR Scheduler & Skip Gating
# ---------------------------------------------------------------------------


def test_sr_model_7step_setup_losses_pretrain_and_scheduler_gating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network_downloads(monkeypatch)
    ds = _create_synthetic_datasets(tmp_path)

    # Save a distinctive pretrained checkpoint as .pth with nested key 'params_ema'
    ref_net = DummySRNet(num_feat=8, num_block=1, scale=2)
    with torch.no_grad():
        for p in ref_net.parameters():
            p.fill_(0.125)
    pth_pretrain = tmp_path / "pretrain_nested.pth"
    torch.save({"params_ema": ref_net.state_dict()}, pth_pretrain)

    cfg = _make_base_train_cfg(
        tmp_path / "sr_model_test",
        ds,
        use_lpips_loss=True,
        grad_accum=2,
        pretrain_path=str(pth_pretrain),
        pretrain_strict=True,
        pretrain_key="params_ema",
        ema_enabled=True,
        ema_decay=0.9,
    )
    # Add a zero-weight sentinel loss that MUST be skipped before instantiation!
    cfg.loss.zero_skip = OmegaConf.create(
        {
            "weight": 0.0,
            "criterion": {"_target_": "tests.test_train_smoke.SentinelUninstantiatedLoss"},
        }
    )

    import hydra

    acc = Accelerator(mixed_precision="no", gradient_accumulation_steps=2)
    model = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)
    assert isinstance(model, SRModel)
    model.setup(acc)

    # Verify double-setup rejected
    with pytest.raises(RuntimeError, match="already been called"):
        model.setup(acc)

    # Verify pretrain loaded BEFORE EMA clone so EMA inherits exact 0.125 weights
    assert model.ema is not None
    assert all(torch.allclose(p, torch.full_like(p, 0.125)) for p in model.ema.module.parameters())
    assert "zero_skip" not in model.losses
    assert set(model.losses.keys()) == {"l1", "lpips"}

    # Verify lr_scheduler is NOT in accelerator._schedulers and registered before ema
    assert len(acc._schedulers) == 0
    assert acc._custom_objects == [model.lr_scheduler, model.ema]

    # Single-group LR dict
    lr_dict = model.get_lr_dict()
    assert list(lr_dict.keys()) == ["lr"]

    # Simulate optimizer_step_was_skipped=True on sync step -> LR & EMA must NOT advance
    batch = {
        "lq": torch.rand(2, 3, 16, 16),
        "gt": torch.rand(2, 3, 32, 32),
        "need_degradation": torch.tensor([False, False]),
    }
    lr_before = model.get_lr_dict()["lr"]
    # Use property patch on Accelerator class for clean skip simulation
    monkeypatch.setattr(Accelerator, "optimizer_step_was_skipped", property(lambda self: True))

    s0, stepped0 = model.train_step(batch, global_step=0)
    assert stepped0 is False and s0 == {}
    s1, stepped1 = model.train_step(batch, global_step=0)
    assert stepped1 is True
    assert "loss/total" in s1 and "loss/l1" in s1 and "loss/lpips" in s1 and "grad_norm" in s1
    assert "lr" not in s1  # Never duplicate lr inside scalars!
    assert model.ema.num_updates == 0
    assert model.get_lr_dict()["lr"] == pytest.approx(lr_before)

    # Now allow step to apply -> LR & EMA advance exactly once
    monkeypatch.setattr(Accelerator, "optimizer_step_was_skipped", property(lambda self: False))
    s2, stepped2 = model.train_step(batch, global_step=1)
    assert stepped2 is False
    s3, stepped3 = model.train_step(batch, global_step=1)
    assert stepped3 is True
    assert model.ema.num_updates == 1
    assert model.get_lr_dict()["lr"] < lr_before

    # Verify export_weights returns isolated CPU clones without prefixes
    exported = model.export_weights(acc)
    assert set(exported.keys()) == {"net_g", "net_g_ema"}
    first_k = next(iter(exported["net_g"]))
    saved_val = exported["net_g"][first_k].clone()
    with torch.no_grad():
        next(model.get_infer_network(use_ema=False).parameters()).fill_(99.0)
    assert torch.equal(exported["net_g"][first_k], saved_val)

    # Verify multi-param-group get_lr_dict returns lr/group_i
    model.optimizer.add_param_group({"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 5e-5})
    multi_lr = model.get_lr_dict()
    assert list(multi_lr.keys()) == ["lr/group_0", "lr/group_1"]
    assert multi_lr["lr/group_1"] == pytest.approx(5e-5)

    acc.end_training()
    _reset_accelerate_and_loggers()

    # Verify pretrain .safetensors with strict=False & all-zero loss fail-fast
    st_path = tmp_path / "partial.safetensors"
    partial_sd = {k: v for i, (k, v) in enumerate(ref_net.state_dict().items()) if i > 0}
    partial_sd["extra_unused_weight"] = torch.zeros(1)
    save_safetensors(partial_sd, st_path)

    cfg_partial = _make_base_train_cfg(
        tmp_path / "sr_partial",
        ds,
        pretrain_path=str(st_path),
        pretrain_strict=False,
        pretrain_key=None,
    )
    acc2 = Accelerator(mixed_precision="no")
    model2 = SRModel(cfg_partial)
    model2.setup(acc2)
    assert model2.get_infer_network(use_ema=True) is model2.ema.module
    acc2.end_training()
    _reset_accelerate_and_loggers()

    cfg_all_zero = _make_base_train_cfg(tmp_path / "sr_zero_loss", ds)
    cfg_all_zero.loss.l1.weight = 0.0
    acc3 = Accelerator(mixed_precision="no")
    with pytest.raises(ValueError, match="at least one active loss"):
        SRModel(cfg_all_zero).setup(acc3)
    acc3.end_training()


# ---------------------------------------------------------------------------
# Test 4: Full E2E Smoke (Mixed + L1 + LPIPS + Multi-Suite + GradAccum=2 + TB)
# ---------------------------------------------------------------------------


def test_trainer_full_smoke_mixed_l1_lpips_multisuite_grad_accum_tb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network_downloads(monkeypatch)
    ds = _create_synthetic_datasets(tmp_path)
    work_dir = tmp_path / "full_smoke_exp"

    val_suites = {
        "val_ref": {
            "dataset": {
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["val_ref_json"]),
            },
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                },
                "ssim": {
                    "_target_": "metrics.SSIMMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                },
                "lpips": {
                    "_target_": "metrics.LPIPSMetric",
                    "net": "alex",
                    "pretrained": False,
                },
            },
        },
        "val_ocr_nogt": {
            "dataset": {
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["val_ocr_json"]),
            },
            "metrics": {
                "ocr_a": {
                    "_target_": "tests.test_train_smoke.MockOCRAMetric",
                    "label_key": "label",
                }
            },
        },
    }

    cfg = _make_base_train_cfg(
        work_dir,
        ds,
        total_iters=4,
        log_interval=2,
        val_interval=2,
        save_interval=2,
        grad_accum=2,
        max_keep_states=1,
        log_with=["tensorboard"],
        use_lpips_loss=True,
        val_suites=val_suites,
        primary_dataset="val_ref",
        primary_metric="psnr",
    )

    trainer = Trainer(cfg)
    # Count exact micro-steps to verify gradient_state stayed clean after mid-training validation
    micro_steps = 0
    orig_train_step = trainer.model.train_step

    def _counting_train_step(batch: dict[str, Any], global_step: int):
        nonlocal micro_steps
        micro_steps += 1
        return orig_train_step(batch, global_step)

    monkeypatch.setattr(trainer.model, "train_step", _counting_train_step)
    state = trainer.fit()

    assert state.global_step == 4
    assert micro_steps == 8  # Exact 4 steps * grad_accum=2 even with mid-training validation at step 2!

    # Directory layout & config_resolved.yaml without unresolved interpolations
    for sub in ("trackers", "checkpoints", "weights", "visuals"):
        assert (work_dir / sub).is_dir()
    resolved_yaml = (work_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    assert "${" not in resolved_yaml
    loaded_cfg = load_resolved_config(work_dir / "config_resolved.yaml")
    assert loaded_cfg.work_dir == str(work_dir)

    # Structured train.log check
    log_text = (work_dir / "train.log").read_text(encoding="utf-8")
    assert re.search(
        r"\[Iter 2/4\] lr: .* \| loss/total: .* \| loss/l1: .* \| loss/lpips: .* \| grad_norm: .* \| .* it/s \| mem: 0\.0 GB \| ETA: ",
        log_text,
    )
    assert "[Iter 4/4]" in log_text
    assert "[Val Iter 000002] val_ref |" in log_text
    assert "[Val Iter 000004] val_ocr_nogt | ocr_a:" in log_text

    # TensorBoard event file written under trackers/
    tb_events = list((work_dir / "trackers").rglob("events.out.tfevents.*"))
    assert len(tb_events) >= 1

    # Track 1 pruned to max_keep_states=1 (only state_iter_000004 remains)
    state_dirs = sorted(p.name for p in (work_dir / "checkpoints").glob("state_iter_*"))
    assert state_dirs == ["state_iter_000004"]

    # Track 2 weights exist and load strictly into ToyModel
    for fname in (
        "iter_000002.safetensors",
        "iter_000002_ema.safetensors",
        "iter_000004.safetensors",
        "iter_000004_ema.safetensors",
        "latest.safetensors",
        "latest_ema.safetensors",
        "best_val_ref_psnr.safetensors",
    ):
        fpath = work_dir / "weights" / fname
        assert fpath.is_file(), f"Missing expected weight file: {fname}"
        missing, unexpected = load_weights(DummySRNet(num_feat=8, num_block=1, scale=2), fpath, strict=True)
        assert missing == [] and unexpected == []

    # Visuals: 3-panel for val_ref (collision-safe duplicate stems!) and 2-panel for val_ocr_nogt
    ref_vis_dir = work_dir / "visuals" / "iter_000004" / "val_ref"
    ref_vis_files = sorted(p.name for p in ref_vis_dir.glob("*_cmp.png"))
    assert ref_vis_files == ["dup_0000_cmp.png", "dup_0001_cmp.png", "unique_img_cmp.png"]
    ref_img = load_image(ref_vis_dir / "unique_img_cmp.png")
    assert ref_img.shape == (3, 32, 32 * 3)  # [LQ_bicubic | SR | GT]

    ocr_vis_dir = work_dir / "visuals" / "iter_000004" / "val_ocr_nogt"
    ocr_vis_files = sorted(p.name for p in ocr_vis_dir.glob("*_cmp.png"))
    assert ocr_vis_files == ["text_00_cmp.png", "text_01_cmp.png"]
    ocr_img = load_image(ocr_vis_dir / "text_00_cmp.png")
    assert ocr_img.shape == (3, 32, 36 * 2)  # [LQ_bicubic | SR] (no GT)

    # Verify meta passthrough on the instantiated MockOCRAMetric
    ocr_metric = trainer.val_suites["val_ocr_nogt"].metrics["ocr_a"]
    labels_seen = [rec["label"] for rec in ocr_metric.seen_records]  # type: ignore[attr-defined]
    assert "HELLO" in labels_seen and "WORLD" in labels_seen
    assert any(
        rec["tag"] == {"split": "hard", "id": 0}
        for rec in ocr_metric.seen_records  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Test 5: log_with=null, lower-is-better best, val_use_ema=false, NaN safety, timestamp
# ---------------------------------------------------------------------------


def test_trainer_log_with_null_lower_is_better_best_and_nan_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network_downloads(monkeypatch)
    ds = _create_synthetic_datasets(tmp_path)

    # Sequence across 4 validations (steps 1, 2, 3, 4) on 1-image suite:
    # step 1: 0.50 (initial best -> saved)
    # step 2: 0.80 (worse for higher_is_better=False -> NOT saved)
    # step 3: NaN  (must NEVER overwrite best -> NOT saved)
    # step 4: 0.25 (strictly lower -> saved!)
    val_suites = {
        "val_tiny": {
            "dataset": {
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["val_tiny_json"]),
            },
            "metrics": {
                "lpips_like": {
                    "_target_": "tests.test_train_smoke.ScriptedSequenceMetric",
                    "sequence": [0.50, 0.80, "nan", 0.25],
                    "higher_is_better": False,
                }
            },
        }
    }

    base_work = tmp_path / "null_tracker_exp"
    cfg = _make_base_train_cfg(
        base_work,
        ds,
        total_iters=4,
        log_interval=2,
        val_interval=1,
        save_interval=4,
        log_with=None,
        val_suites=val_suites,
        primary_dataset="val_tiny",
        primary_metric="lpips_like",
        val_use_ema=False,
        val_save_visuals=False,
        append_timestamp=True,
        resume=None,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer = Trainer(cfg)
        state = trainer.fit()

    # No spurious Accelerator log_with warnings
    log_with_warnings = [w for w in caught if "log_with" in str(w.message)]
    assert log_with_warnings == []

    # Timestamped work_dir created and recorded in config_resolved.yaml
    assert trainer.work_dir != base_work
    assert trainer.work_dir.name.startswith("null_tracker_exp_")
    resolved = load_resolved_config(trainer.work_dir / "config_resolved.yaml")
    assert resolved.work_dir == str(trainer.work_dir)

    # Zero files under trackers/ when log_with=null
    assert list((trainer.work_dir / "trackers").rglob("*")) == []

    # Lower-is-better selected 0.25 (ignoring 0.80 and NaN)
    assert state.best_metrics["val_tiny/lpips_like"] == pytest.approx(0.25)

    # val_use_ema=False -> best_*.safetensors must equal net_g (iter_000004), NOT net_g_ema!
    best_sd = load_safetensors_file(
        str(trainer.work_dir / "weights" / "best_val_tiny_lpips_like.safetensors")
    )
    net_g_sd = load_safetensors_file(str(trainer.work_dir / "weights" / "iter_000004.safetensors"))
    ema_sd = load_safetensors_file(
        str(trainer.work_dir / "weights" / "iter_000004_ema.safetensors")
    )
    assert all(torch.equal(best_sd[k], net_g_sd[k]) for k in net_g_sd)
    assert any(not torch.equal(best_sd[k], ema_sd[k]) for k in ema_sd)

    # Verify resume="auto" on empty checkpoints logs WARNING and starts cleanly from step 0
    _reset_accelerate_and_loggers()
    fresh_work = tmp_path / "auto_empty_resume"
    cfg_auto_empty = _make_base_train_cfg(
        fresh_work,
        ds,
        total_iters=1,
        log_interval=1,
        val_interval=1,
        save_interval=1,
        val_suites={},
        primary_dataset=None,
        primary_metric=None,
        resume="auto",
    )
    trainer_auto = Trainer(cfg_auto_empty)
    st_auto = trainer_auto.fit()
    assert st_auto.global_step == 1
    auto_log = (fresh_work / "train.log").read_text(encoding="utf-8")
    assert "WARNING" in auto_log and "resume='auto'" in auto_log


# ---------------------------------------------------------------------------
# Test 6: No Duplicate Final Val/Save & Already-Finished Resume
# ---------------------------------------------------------------------------


def test_trainer_no_duplicate_final_val_save_and_already_finished_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network_downloads(monkeypatch)
    ds = _create_synthetic_datasets(tmp_path)

    # (a) total_iters=4, val_interval=2, save_interval=2 -> called at [2, 4] ONLY
    work_a = tmp_path / "no_dup_a"
    cfg_a = _make_base_train_cfg(
        work_a, ds, total_iters=4, val_interval=2, save_interval=2, val_save_visuals=False
    )
    trainer_a = Trainer(cfg_a)
    val_steps_a: list[int] = []
    save_steps_a: list[int] = []
    orig_val_a = trainer_a.validate
    orig_save_a = trainer_a._save_checkpoint

    def _spy_val_a(step: int):
        val_steps_a.append(step)
        return orig_val_a(step)

    def _spy_save_a(step: int):
        save_steps_a.append(step)
        return orig_save_a(step)

    monkeypatch.setattr(trainer_a, "validate", _spy_val_a)
    monkeypatch.setattr(trainer_a, "_save_checkpoint", _spy_save_a)
    trainer_a.fit()
    assert val_steps_a == [2, 4]
    assert save_steps_a == [2, 4]

    # (b) total_iters=3, val_interval=2, save_interval=2 -> final val & save run at step 3 (val before save!)
    _reset_accelerate_and_loggers()
    work_b = tmp_path / "final_step_b"
    cfg_b = _make_base_train_cfg(
        work_b, ds, total_iters=3, val_interval=2, save_interval=2, val_save_visuals=False
    )
    trainer_b = Trainer(cfg_b)
    events_b: list[tuple[str, int]] = []
    orig_val_b = trainer_b.validate
    orig_save_b = trainer_b._save_checkpoint

    def _spy_val_b(step: int):
        events_b.append(("val", step))
        return orig_val_b(step)

    def _spy_save_b(step: int):
        events_b.append(("save", step))
        return orig_save_b(step)

    monkeypatch.setattr(trainer_b, "validate", _spy_val_b)
    monkeypatch.setattr(trainer_b, "_save_checkpoint", _spy_save_b)
    trainer_b.fit()
    assert events_b == [("val", 2), ("save", 2), ("val", 3), ("save", 3)]

    # (c) Resuming with resume=auto when global_step == total_iters skips loop & does not overwrite ckpts
    _reset_accelerate_and_loggers()
    ckpt_mtime_before = (work_b / "weights" / "iter_000003.safetensors").stat().st_mtime_ns
    cfg_done = _make_base_train_cfg(
        work_b,
        ds,
        total_iters=3,
        val_interval=2,
        save_interval=2,
        val_save_visuals=False,
        resume="auto",
    )
    trainer_done = Trainer(cfg_done)
    state_done = trainer_done.fit()
    assert state_done.global_step == 3
    assert (work_b / "weights" / "iter_000003.safetensors").stat().st_mtime_ns == ckpt_mtime_before


# ---------------------------------------------------------------------------
# Test 7: Interrupt & resume=auto Bit-Exact Consistency
# ---------------------------------------------------------------------------


def test_trainer_interrupt_and_resume_auto_bit_exact_consistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network_downloads(monkeypatch)
    ds = _create_synthetic_datasets(tmp_path)

    # Run A: uninterrupted 6 steps (T_max=6 fixed)
    work_cont = tmp_path / "run_continuous"
    cfg_cont = _make_base_train_cfg(
        work_cont,
        ds,
        total_iters=6,
        log_interval=3,
        val_interval=3,
        save_interval=3,
        grad_accum=2,
        t_max=6,
        val_save_visuals=False,
    )
    trainer_cont = Trainer(cfg_cont)
    state_cont = trainer_cont.fit()
    lr_cont = trainer_cont.model.get_lr_dict()["lr"]
    net_cont = load_safetensors_file(str(work_cont / "weights" / "iter_000006.safetensors"))
    ema_cont = load_safetensors_file(str(work_cont / "weights" / "iter_000006_ema.safetensors"))

    # Run B1: stop at step 3 (same T_max=6)
    _reset_accelerate_and_loggers()
    work_res = tmp_path / "run_resumed"
    cfg_part1 = _make_base_train_cfg(
        work_res,
        ds,
        total_iters=3,
        log_interval=3,
        val_interval=3,
        save_interval=3,
        grad_accum=2,
        t_max=6,
        val_save_visuals=False,
    )
    trainer_p1 = Trainer(cfg_part1)
    state_p1 = trainer_p1.fit()
    assert state_p1.global_step == 3
    assert (work_res / "checkpoints" / "state_iter_000003").is_dir()

    # Run B2: resume=auto to step 6 (same T_max=6)
    _reset_accelerate_and_loggers()
    cfg_part2 = _make_base_train_cfg(
        work_res,
        ds,
        total_iters=6,
        log_interval=3,
        val_interval=3,
        save_interval=3,
        grad_accum=2,
        t_max=6,
        val_save_visuals=False,
        resume="auto",
    )
    trainer_p2 = Trainer(cfg_part2)
    state_p2 = trainer_p2.fit()
    lr_res = trainer_p2.model.get_lr_dict()["lr"]
    net_res = load_safetensors_file(str(work_res / "weights" / "iter_000006.safetensors"))
    ema_res = load_safetensors_file(str(work_res / "weights" / "iter_000006_ema.safetensors"))

    assert state_p2.global_step == state_cont.global_step == 6
    assert lr_res == pytest.approx(lr_cont, abs=0.0)
    assert state_p2.best_metrics == pytest.approx(state_cont.best_metrics, abs=1e-6)

    max_net_diff = max((net_cont[k] - net_res[k]).abs().max().item() for k in net_cont)
    max_ema_diff = max((ema_cont[k] - ema_res[k]).abs().max().item() for k in ema_cont)
    assert max_net_diff == 0.0, f"Expected bit-exact net_g resume, got max_diff={max_net_diff}"
    assert max_ema_diff == 0.0, f"Expected bit-exact ema resume, got max_diff={max_ema_diff}"


# ---------------------------------------------------------------------------
# Test 8: Train -> tools/infer.py exp_dir Auto-Restore Integration
# ---------------------------------------------------------------------------


def test_train_to_tools_infer_exp_dir_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _block_network_downloads(monkeypatch)
    ds = _create_synthetic_datasets(tmp_path)
    work_dir = tmp_path / "train_for_infer"

    cfg = _make_base_train_cfg(
        work_dir,
        ds,
        total_iters=2,
        log_interval=2,
        val_interval=2,
        save_interval=2,
        val_save_visuals=False,
    )

    # Exercise importable run_train from tools/train.py
    import importlib.util

    spec = importlib.util.spec_from_file_location("srtrainer_tools_train", TOOLS_TRAIN)
    assert spec is not None and spec.loader is not None
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    trainer = train_mod.run_train(cfg)
    assert trainer.train_state.global_step == 2
    best_weight = work_dir / "weights" / "best_val_ref_psnr.safetensors"
    assert best_weight.is_file()

    # Invoke tools/infer.py in Mode 1 (infer.exp_dir ONLY — no arch/scale/weights_path!)
    infer_out = tmp_path / "infer_outputs"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    extra_pp = f"{REPO_ROOT}:{REPO_ROOT / 'src'}"
    env["PYTHONPATH"] = f"{extra_pp}:{existing_pp}" if existing_pp else extra_pp
    proc = subprocess.run(
        [
            sys.executable,
            str(TOOLS_INFER),
            f"infer.exp_dir={work_dir}",
            f"infer.input={ds['val_ref_json']}",
            f"infer.input_root={ds['data_root']}",
            f"infer.output_dir={infer_out}",
            "infer.mixed_precision=no",
            "infer.tile.overlap=4",
            "infer.tile.batch_size=2",
        ],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"tools/infer.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    out_unique = infer_out / "val_ref" / "unique_img.png"
    assert out_unique.is_file()

    # Reference output using TiledInferencer directly with best_*.safetensors & tile_size=16 (32//2)
    ref_model = DummySRNet(num_feat=8, num_block=1, scale=2)
    load_weights(ref_model, best_weight, strict=True)
    ref_inf = TiledInferencer(
        model=ref_model,
        scale=2,
        tile_enabled=True,
        tile_size="auto",
        tile_overlap=4,
        tile_batch_size=2,
        blend_mode="cosine",
        pad_multiple=8,
        gt_size=32,
        mixed_precision="no",
        device="cpu",
    )
    assert ref_inf.tile_size == (16, 16)
    lq_tensor = load_image(ds["data_root"] / "val_ref" / "unique_img.png")
    expected_sr = ref_inf.infer_tensor(lq_tensor)
    actual_sr = load_image(out_unique)
    assert actual_sr.shape == expected_sr.shape
    max_diff = (actual_sr - expected_sr).abs().max().item()
    assert max_diff <= (1.0 / 255.0 + 1e-5), f"Mismatch between tools/infer.py and TiledInferencer: {max_diff}"


# ---------------------------------------------------------------------------
# Test 9: Real 2-Process CPU Distributed Training via torchrun
#         Verifies: exact 1x LR (NOT 2x!), odd-5 gather_for_metrics deduplication,
#         N=1 < world_size=2 suite, append_timestamp broadcast, clean exit code 0!
# ---------------------------------------------------------------------------


def test_train_cli_distributed_2proc_cpu_torchrun_smoke(tmp_path: Path) -> None:
    ds = _create_synthetic_datasets(tmp_path)
    base_work = tmp_path / "dist_2proc_exp"

    # Build Hydra overrides for real CLI invocation from cwd="/tmp"
    sources_override = _to_hydra_val(
        [
            {
                "name": "syn_gt_only",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["gt_only_json"]),
                "ratio": 0.5,
            },
            {
                "name": "syn_paired",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["paired_json"]),
                "ratio": 0.5,
            },
        ]
    )
    val_suites_override = _to_hydra_val(
        {
            "val_odd5": {
                "dataset": {
                    "root_dir": str(ds["data_root"]),
                    "json_path": str(ds["val_odd5_json"]),
                },
                "metrics": {
                    "meta_score": {
                        "_target_": "tests.test_train_smoke.MetaScoreMetric",
                        "score_key": "score",
                    },
                    "psnr": {
                        "_target_": "metrics.PSNRMetric",
                        "crop_border": 2,
                        "test_y_channel": True,
                    },
                },
            },
            "val_tiny": {
                "dataset": {
                    "root_dir": str(ds["data_root"]),
                    "json_path": str(ds["val_tiny_json"]),
                },
                "metrics": {
                    "meta_score": {
                        "_target_": "tests.test_train_smoke.MetaScoreMetric",
                        "score_key": "score",
                    }
                },
            },
        }
    )

    init_lr = 2.0e-4
    eta_min = 1.0e-6
    t_max = 10
    total_iters = 4

    overrides = [
        f"exp_name={base_work.name}",
        f"output_root={base_work.parent}",
        f"work_dir={base_work}",
        "scale=2",
        "seed=42",
        "~arch",
        "+arch={_target_:tests.dummy_arch.DummySRNet,num_in_ch:3,num_out_ch:3,num_feat:8,num_block:1,scale:2}",
        "data.gt_size=32",
        "data.batch_size_per_gpu=2",
        "data.num_workers=0",
        "data.pin_memory=false",
        "data.persistent_workers=false",
        f"data.sources={sources_override}",
        "data.degradation.kernel_cfg.kernel_size=7",
        "data.degradation.kernel_cfg.kernel_range=[7]",
        f"optim.optimizer.lr={init_lr}",
        f"optim.scheduler.T_max={t_max}",
        f"optim.scheduler.eta_min={eta_min}",
        "metric.primary_dataset=val_odd5",
        "metric.primary_metric=meta_score",
        "~metric.val_suites",
        f"+metric.val_suites={val_suites_override}",
        f"trainer.total_iters={total_iters}",
        "trainer.log_interval=2",
        "trainer.val_interval=2",
        "trainer.save_interval=2",
        "trainer.max_keep_states=2",
        "trainer.mixed_precision=bf16",
        "trainer.gradient_accumulation_steps=2",
        "trainer.log_with=null",
        "trainer.append_timestamp=true",
        "trainer.val_save_visuals=true",
        "trainer.val_max_save_images=8",
        "trainer.val_tile.overlap=4",
        "trainer.val_tile.batch_size=2",
    ]

    proc = _run_torchrun_train(overrides, cwd=Path("/tmp"), nproc=2, timeout=120)
    assert proc.returncode == 0, (
        f"2-process torchrun failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # append_timestamp=true broadcasted identical single timestamped dir across both ranks
    matching_dirs = [p for p in tmp_path.glob(f"{base_work.name}_*") if p.is_dir()]
    assert len(matching_dirs) == 1, f"Expected 1 shared timestamped work_dir, got {matching_dirs}"
    exp_dir = matching_dirs[0]

    assert (exp_dir / "config_resolved.yaml").is_file()
    assert (exp_dir / "weights" / "best_val_odd5_meta_score.safetensors").is_file()

    # Both rank RNG files exist in state_iter_000004
    state_dir = exp_dir / "checkpoints" / "state_iter_000004"
    assert (state_dir / "random_states_0.pkl").is_file()
    assert (state_dir / "random_states_1.pkl").is_file()

    # 1) EXACT SINGLE-STEP LR CHECK (NOT 2x!)
    # custom_checkpoint_0.pkl = lr_scheduler, custom_checkpoint_1.pkl = ema, custom_checkpoint_2.pkl = train_state
    sched_state = torch.load(
        state_dir / "custom_checkpoint_0.pkl", map_location="cpu", weights_only=False
    )
    expected_1x_lr = eta_min + 0.5 * (init_lr - eta_min) * (1.0 + math.cos(math.pi * total_iters / t_max))
    wrong_2x_lr = eta_min + 0.5 * (init_lr - eta_min) * (1.0 + math.cos(math.pi * (2 * total_iters) / t_max))
    assert sched_state["last_epoch"] == total_iters, (
        f"LR scheduler stepped {sched_state['last_epoch']} times instead of {total_iters} (2x bug!)"
    )
    actual_lr = float(sched_state["_last_lr"][0])
    assert actual_lr == pytest.approx(expected_1x_lr, rel=1e-6, abs=1e-12)
    assert abs(actual_lr - wrong_2x_lr) > 1e-5

    # 2) EXACT ODD-5 GATHER_FOR_METRICS DEDUPLICATION + N=1 < world_size=2
    # Exact 5-image unweighted mean = (10+20+30+40+100)/5 = 40.0
    # Padded 6-sample mean (duplicating img 0 = 10.0) would be 210/6 = 35.0!
    ts_state = torch.load(
        state_dir / "custom_checkpoint_2.pkl", map_location="cpu", weights_only=True
    )
    assert ts_state["global_step"] == total_iters
    odd5_mean = ts_state["best_metrics"]["val_odd5/meta_score"]
    assert odd5_mean == pytest.approx(40.0, abs=1e-6)
    assert abs(odd5_mean - 35.0) > 1.0

    # Exact Rank-0 single log stream (no duplicated lines from Rank 1)
    log_text = (exp_dir / "train.log").read_text(encoding="utf-8")
    assert log_text.count("[Iter 2/4]") == 1
    assert log_text.count("[Iter 4/4]") == 1
    assert "val_odd5 | meta_score: 40.0000" in log_text
    assert "val_tiny | meta_score: 77.5000" in log_text

    # Exact visual counts (padded tail excluded, no Rank-1 overwrite of index 0)
    odd5_vis = sorted(p.name for p in (exp_dir / "visuals" / "iter_000004" / "val_odd5").glob("*_cmp.png"))
    tiny_vis = sorted(p.name for p in (exp_dir / "visuals" / "iter_000004" / "val_tiny").glob("*_cmp.png"))
    assert len(odd5_vis) == 5
    assert len(tiny_vis) == 1


# ---------------------------------------------------------------------------
# Test 10: Real 2-Process CPU Distributed Resume (resume=auto)
# ---------------------------------------------------------------------------


def test_train_cli_distributed_2proc_resume_auto(tmp_path: Path) -> None:
    ds = _create_synthetic_datasets(tmp_path)
    exp_dir = tmp_path / "dist_2proc_resume"

    sources_override = _to_hydra_val(
        [
            {
                "name": "syn_gt_only",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["gt_only_json"]),
                "ratio": 0.5,
            },
            {
                "name": "syn_paired",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["paired_json"]),
                "ratio": 0.5,
            },
        ]
    )
    val_suites_override = _to_hydra_val(
        {
            "val_tiny": {
                "dataset": {
                    "root_dir": str(ds["data_root"]),
                    "json_path": str(ds["val_tiny_json"]),
                },
                "metrics": {
                    "psnr": {
                        "_target_": "metrics.PSNRMetric",
                        "crop_border": 2,
                        "test_y_channel": True,
                    }
                },
            }
        }
    )

    common = [
        f"exp_name={exp_dir.name}",
        f"output_root={exp_dir.parent}",
        f"work_dir={exp_dir}",
        "scale=2",
        "seed=42",
        "~arch",
        "+arch={_target_:tests.dummy_arch.DummySRNet,num_in_ch:3,num_out_ch:3,num_feat:8,num_block:1,scale:2}",
        "data.gt_size=32",
        "data.batch_size_per_gpu=2",
        "data.num_workers=0",
        "data.pin_memory=false",
        "data.persistent_workers=false",
        f"data.sources={sources_override}",
        "data.degradation.kernel_cfg.kernel_size=7",
        "data.degradation.kernel_cfg.kernel_range=[7]",
        "optim.scheduler.T_max=6",
        "metric.primary_dataset=val_tiny",
        "metric.primary_metric=psnr",
        "~metric.val_suites",
        f"+metric.val_suites={val_suites_override}",
        "trainer.log_interval=2",
        "trainer.val_interval=2",
        "trainer.save_interval=2",
        "trainer.mixed_precision=no",
        "trainer.gradient_accumulation_steps=1",
        "trainer.log_with=null",
        "trainer.val_save_visuals=false",
        "trainer.val_tile.overlap=4",
    ]

    # Phase 1: Train to step 2
    proc1 = _run_torchrun_train(
        [*common, "trainer.total_iters=2"], cwd=Path("/tmp"), nproc=2, timeout=120
    )
    assert proc1.returncode == 0, f"Phase 1 failed:\nSTDOUT:\n{proc1.stdout}\nSTDERR:\n{proc1.stderr}"
    assert (exp_dir / "checkpoints" / "state_iter_000002").is_dir()

    # Phase 2: Resume via trainer.resume=auto to step 4
    proc2 = _run_torchrun_train(
        [*common, "trainer.total_iters=4", "trainer.resume=auto"],
        cwd=Path("/tmp"),
        nproc=2,
        timeout=120,
    )
    assert proc2.returncode == 0, f"Phase 2 resume failed:\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}"

    log_text = (exp_dir / "train.log").read_text(encoding="utf-8")
    assert "[Resume]" in log_text
    assert "[Iter 4/4]" in log_text
    assert (exp_dir / "weights" / "iter_000004.safetensors").is_file()
    assert (exp_dir / "weights" / "iter_000004_ema.safetensors").is_file()
    state4 = exp_dir / "checkpoints" / "state_iter_000004"
    assert (state4 / "random_states_0.pkl").is_file()
    assert (state4 / "random_states_1.pkl").is_file()

    # Confirm LR scheduler still stepped exact 1x (last_epoch == 4, NOT 8!)
    sched_state = torch.load(
        state4 / "custom_checkpoint_0.pkl", map_location="cpu", weights_only=False
    )
    assert sched_state["last_epoch"] == 4
