"""Unit and integration tests for logging training lq, gt, and pred images to trackers."""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import hydra
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from accelerate.utils import ProjectConfiguration
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from engine import Trainer
from models import BaseModel
from tests.dummy_arch import get_dummy_vosr_arch_dict
from utils.logging import log_image_dict_to_trackers, log_images_to_trackers

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


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
def cleanup_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    _reset_accelerate_and_loggers()
    yield
    _reset_accelerate_and_loggers()


def _write_solid_rgb_png(path: Path, h: int, w: int, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty((h, w, 3), dtype=np.uint8)
    arr[..., 0] = rgb[0]
    arr[..., 1] = rgb[1]
    arr[..., 2] = rgb[2]
    Image.fromarray(arr, mode="RGB").save(path)


def _create_train_manifests(base_dir: Path, scale: int = 2) -> dict[str, Any]:
    data_root = base_dir / "datasets"
    manifests_dir = base_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    paired_entries: list[dict[str, str]] = []
    for idx in range(4):
        gt_rel = f"paired/gt_{idx:02d}.png"
        lq_rel = f"paired/lq_{idx:02d}.png"
        # Distinct solid colors so we can assert LQ vs GT panels cleanly.
        _write_solid_rgb_png(data_root / gt_rel, 32, 32, (20, 220, 40))
        _write_solid_rgb_png(data_root / lq_rel, 32 // scale, 32 // scale, (220, 30, 20))
        paired_entries.append({"gt": gt_rel, "lq": lq_rel})
    paired_json = manifests_dir / "paired.json"
    paired_json.write_text(json.dumps(paired_entries), encoding="utf-8")

    gt_only_entries: list[dict[str, str]] = []
    for idx in range(4):
        gt_rel = f"gt_only/gt_{idx:02d}.png"
        _write_solid_rgb_png(data_root / gt_rel, 32, 32, (60, 120, 220))
        gt_only_entries.append({"gt": gt_rel})
    gt_only_json = manifests_dir / "gt_only.json"
    gt_only_json.write_text(json.dumps(gt_only_entries), encoding="utf-8")

    return {
        "data_root": data_root,
        "paired_json": paired_json,
        "gt_only_json": gt_only_json,
    }


def _make_train_cfg(
    work_dir: Path,
    ds: dict[str, Any],
    *,
    scale: int = 2,
    gt_size: int = 32,
    batch_size: int = 2,
    total_iters: int = 4,
    log_interval: int = 2,
    val_interval: int = 0,
    save_interval: int = 0,
    grad_accum: int = 1,
    log_with: Any = None,
    use_degradation: bool = False,
    train_log_image_interval: Any = 2,
    train_max_log_images: Any = 2,
    train_log_image_mode: Any = "comparison",
    model_target: str = "models.VOSRModel",
) -> DictConfig:
    sources: list[dict[str, Any]]
    if use_degradation:
        sources = [
            {
                "name": "gt_only_src",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["gt_only_json"]),
                "ratio": 1.0,
            }
        ]
        degradation: Any = {
            "_target_": "data.degradation.RealESRGANDegrader",
            "scale": scale,
            "use_usm": False,
            "queue_size": 0,
            "kernel_cfg": {
                "kernel_size": 7,
                "kernel_range": [7],
            },
        }
    else:
        sources = [
            {
                "name": "paired_src",
                "root_dir": str(ds["data_root"]),
                "json_path": str(ds["paired_json"]),
                "ratio": 1.0,
            }
        ]
        degradation = None

    return OmegaConf.create(
        {
            "exp_name": work_dir.name,
            "scale": scale,
            "seed": 42,
            "output_root": str(work_dir.parent),
            "work_dir": str(work_dir),
            "arch": get_dummy_vosr_arch_dict(),
            "loss": {
                "l1": {
                    "weight": 1.0,
                    "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
                }
            },
            "model": {
                "_target_": model_target,
                "scale": scale,
                "trainable": {"vae": False, "dit": True, "dino": False},
                "ema": {"enabled": False, "decay": 0.99, "warmup_steps": 0},
                "pretrain": {
                    "vae": {"path": None, "strict": True, "key": None},
                    "dit": {"path": None, "strict": True, "key": None},
                    "dino": {"path": None, "strict": True, "key": None},
                },
            },
            "data": {
                "gt_size": gt_size,
                "scale": scale,
                "batch_size_per_gpu": batch_size,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "sources": sources,
                "augment": {"hflip": False, "vflip": False, "rot90": False},
                "degradation": degradation,
            },
            "optim": {
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "_partial_": True,
                    "lr": 1.0e-3,
                },
                "scheduler": None,
            },
            "metric": {
                "primary_dataset": None,
                "primary_metric": None,
                "val_suites": {},
            },
            "trainer": {
                "total_iters": total_iters,
                "log_interval": log_interval,
                "val_interval": val_interval,
                "save_interval": save_interval,
                "max_keep_states": 1,
                "mixed_precision": "no",
                "gradient_accumulation_steps": grad_accum,
                "grad_clip_norm": 1.0,
                "compile": False,
                "log_with": [] if log_with is None else log_with,
                "tracker_init_kwargs": {},
                "resume": None,
                "append_timestamp": False,
                "train_log_image_interval": train_log_image_interval,
                "train_max_log_images": train_max_log_images,
                "train_log_image_mode": train_log_image_mode,
                "val_save_visuals": False,
                "val_max_save_images": 0,
                "val_use_ema": False,
            },
        }
    )


def _read_tb_images(tb_dir: Path) -> dict[str, list[tuple[int, np.ndarray]]]:
    acc = EventAccumulator(str(tb_dir), size_guidance={"images": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, np.ndarray]]] = {}
    for tag in acc.Tags().get("images", []):
        entries: list[tuple[int, np.ndarray]] = []
        for ev in acc.Images(tag):
            with Image.open(io.BytesIO(ev.encoded_image_string)) as img:
                entries.append((int(ev.step), np.asarray(img.convert("RGB"))))
        out[tag] = entries
    return out


class _NoVisualsCustomModel(BaseModel):
    """Minimal BaseModel subclass that intentionally does not override visual capture hooks."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.net = nn.Conv2d(3, 3, kernel_size=1)
        self.opt: torch.optim.Optimizer | None = None

    def setup(self, accelerator: Accelerator) -> None:
        self.opt = torch.optim.SGD(self.net.parameters(), lr=1e-3)
        self.net, self.opt = accelerator.prepare(self.net, self.opt)
        self.accelerator = accelerator

    def train_step(self, batch: dict[str, Any], global_step: int) -> tuple[dict[str, float], bool]:
        assert self.accelerator is not None and self.opt is not None
        x = batch["gt"].to(self.accelerator.device)
        loss = (self.net(x) - x).abs().mean()
        self.accelerator.backward(loss)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return {"loss/total": float(loss.detach().item())}, True

    def get_infer_network(self, use_ema: bool = True) -> nn.Module:
        return self.net

    def export_weights(self, accelerator: Accelerator) -> dict[str, dict[str, torch.Tensor]]:
        return {"net_g": {k: v.detach().cpu() for k, v in self.net.state_dict().items()}}


# ---------------------------------------------------------------------------
# 1. Config defaults & fail-fast validation
# ---------------------------------------------------------------------------


def test_hydra_trainer_default_train_image_logging_keys() -> None:
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "experiment=vosr2_onestep_text",
                "trainer.train_log_image_interval=250",
                "trainer.train_max_log_images=2",
                "trainer.train_log_image_mode=separate",
            ],
        )
    assert cfg.trainer.train_log_image_interval == 250
    assert cfg.trainer.train_max_log_images == 2
    assert cfg.trainer.train_log_image_mode == "separate"

    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg_default = hydra.compose(
            config_name="train",
            overrides=["experiment=vosr2_onestep_text"],
        )
    assert cfg_default.trainer.train_log_image_interval == 500
    assert cfg_default.trainer.train_max_log_images == 4
    assert cfg_default.trainer.train_log_image_mode == "comparison"


def test_trainer_train_image_logging_config_failfast(tmp_path: Path) -> None:
    ds = _create_train_manifests(tmp_path, scale=2)

    bad_cases = [
        ({"train_log_image_interval": -1}, "train_log_image_interval"),
        ({"train_log_image_interval": True}, "train_log_image_interval"),
        ({"train_max_log_images": -1}, "train_max_log_images"),
        ({"train_max_log_images": False}, "train_max_log_images"),
        ({"train_log_image_mode": "invalid_mode"}, "train_log_image_mode"),
        ({"train_log_image_mode": 123}, "train_log_image_mode"),
    ]
    for idx, (override, match_str) in enumerate(bad_cases):
        work_dir = tmp_path / f"bad_cfg_{idx}"
        cfg = _make_train_cfg(work_dir, ds, **override)
        with pytest.raises(ValueError, match=match_str):
            Trainer(cfg)
        assert not work_dir.exists()


# ---------------------------------------------------------------------------
# 2. log_image_dict_to_trackers unit tests (TB, WandB, SwanLab, MLflow, NaN)
# ---------------------------------------------------------------------------


def test_log_image_dict_to_trackers_batching_and_nan_safety(tmp_path: Path) -> None:
    img_a = torch.tensor(
        [[[float("nan"), float("inf")], [float("-inf"), 0.5]]] * 3,
        dtype=torch.float32,
    )
    img_b = torch.full((1, 3, 4, 6), 0.25, dtype=torch.float32)

    # Real TensorBoard: single flush writes all tags with clamped finite values
    proj_cfg = ProjectConfiguration(
        project_dir=str(tmp_path),
        logging_dir=str(tmp_path / "trackers"),
        automatic_checkpoint_naming=False,
    )
    acc = Accelerator(
        mixed_precision="no",
        cpu=True,
        log_with="tensorboard",
        project_config=proj_cfg,
    )
    acc.init_trackers("batch_tb")
    log_image_dict_to_trackers(acc, {"train/a": img_a, "train/b": img_b}, step=7)
    log_images_to_trackers(acc, "train/single", img_b, step=8)
    acc.end_training()

    tb_imgs = _read_tb_images(tmp_path / "trackers" / "batch_tb")
    assert set(tb_imgs.keys()) == {"train/a", "train/b", "train/single"}
    assert tb_imgs["train/a"][0][0] == 7
    assert tb_imgs["train/a"][0][1].shape == (2, 2, 3)
    # nan -> 0, +inf -> 255, -inf -> 0
    assert int(tb_imgs["train/a"][0][1][0, 0, 0]) == 0
    assert int(tb_imgs["train/a"][0][1][0, 1, 0]) == 255
    assert int(tb_imgs["train/a"][0][1][1, 0, 0]) == 0
    assert tb_imgs["train/b"][0][1].shape == (4, 6, 3)
    assert tb_imgs["train/single"][0][0] == 8

    # WandB & SwanLab batch all images into a single .log(payload, step=step) call
    fake_wandb_run = MagicMock()
    fake_wandb = SimpleNamespace(Image=lambda arr: ("wandb", arr.shape, arr.dtype))
    fake_swanlab_run = MagicMock()
    fake_swanlab = SimpleNamespace(Image=lambda arr: ("swanlab", arr.shape, arr.dtype))
    fake_mlflow = MagicMock()

    orig_mods = {k: sys.modules.get(k) for k in ("wandb", "swanlab", "mlflow")}
    sys.modules["wandb"] = fake_wandb  # type: ignore[assignment]
    sys.modules["swanlab"] = fake_swanlab  # type: ignore[assignment]
    sys.modules["mlflow"] = fake_mlflow  # type: ignore[assignment]
    try:
        fake_acc = SimpleNamespace(
            is_main_process=True,
            trackers=[
                SimpleNamespace(name="wandb", tracker=fake_wandb_run),
                SimpleNamespace(name="swanlab", tracker=fake_swanlab_run),
                SimpleNamespace(name="mlflow", tracker=object()),
            ],
        )
        log_image_dict_to_trackers(fake_acc, {"t/k1": img_a, "t/k2": img_b}, step=12)
    finally:
        for k, v in orig_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    fake_wandb_run.log.assert_called_once()
    w_payload, w_kwargs = fake_wandb_run.log.call_args
    assert w_kwargs == {"step": 12}
    assert set(w_payload[0].keys()) == {"t/k1", "t/k2"}
    assert w_payload[0]["t/k1"] == ("wandb", (2, 2, 3), np.uint8)
    assert w_payload[0]["t/k2"] == ("wandb", (4, 6, 3), np.uint8)

    fake_swanlab_run.log.assert_called_once()
    s_payload, s_kwargs = fake_swanlab_run.log.call_args
    assert s_kwargs == {"step": 12}
    assert set(s_payload[0].keys()) == {"t/k1", "t/k2"}

    assert fake_mlflow.log_image.call_count == 2


# ---------------------------------------------------------------------------
# 3. VOSRModel visual capture unit tests (disabled, grad_accum, stale clear)
# ---------------------------------------------------------------------------


def test_vosr_model_train_visual_capture_and_grad_accum(tmp_path: Path) -> None:
    ds = _create_train_manifests(tmp_path, scale=2)
    cfg = _make_train_cfg(tmp_path / "vosr_unit", ds, scale=2, grad_accum=2)
    from config import load_train_config
    from models.base_model import build_model

    acc = Accelerator(mixed_precision="no", cpu=True, gradient_accumulation_steps=2)
    model = build_model(load_train_config(cfg))
    model.setup(acc)

    with pytest.raises(ValueError, match="enabled"):
        model.set_capture_train_visuals("yes", 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_images"):
        model.set_capture_train_visuals(True, -1)

    batch1 = {"lq": torch.full((1, 3, 8, 8), 0.2), "gt": torch.full((1, 3, 16, 16), 0.8)}
    batch2 = {"lq": torch.full((1, 3, 8, 8), 0.4), "gt": torch.full((1, 3, 16, 16), 0.6)}

    # Disabled by default -> nothing captured
    _, stepped0 = model.train_step(batch1, global_step=0)
    assert stepped0 is False
    assert model.pop_train_visuals() is None
    _, stepped1 = model.train_step(batch2, global_step=0)
    assert stepped1 is True
    assert model.pop_train_visuals() is None

    # Enabled with max_images=2 across 2 grad_accum micro-steps (batch_size=1 each)
    model.set_capture_train_visuals(True, max_images=2)
    _, stepped2 = model.train_step(batch1, global_step=1)
    assert stepped2 is False
    # Calling set_capture_train_visuals(True, max_images=2) again mid-accum preserves micro-step 1
    model.set_capture_train_visuals(True, max_images=2)
    _, stepped3 = model.train_step(batch2, global_step=1)
    assert stepped3 is True

    visuals = model.pop_train_visuals()
    assert visuals is not None
    assert set(visuals.keys()) == {"lq", "gt", "pred"}
    assert visuals["lq"].shape == (2, 3, 8, 8)
    assert visuals["gt"].shape == (2, 3, 16, 16)
    assert visuals["pred"].shape == (2, 3, 16, 16)
    assert all(
        t.device.type == "cpu" and t.dtype == torch.float32 and not t.requires_grad
        for t in visuals.values()
    )
    assert torch.allclose(visuals["lq"][0], torch.full((3, 8, 8), 0.2))
    assert torch.allclose(visuals["lq"][1], torch.full((3, 8, 8), 0.4))
    assert model.pop_train_visuals() is None  # Cleared after pop

    # Stale unpopped visuals are cleared when the next optimizer step starts
    _, _ = model.train_step(batch1, global_step=2)
    _, _ = model.train_step(batch2, global_step=2)
    # Do NOT pop; start step 3 with max_images=1 -> should contain only batch1 from step 3
    model.set_capture_train_visuals(True, max_images=1)
    _, _ = model.train_step(batch1, global_step=3)
    _, _ = model.train_step(batch2, global_step=3)
    vis3 = model.pop_train_visuals()
    assert vis3 is not None and vis3["lq"].shape == (1, 3, 8, 8)
    assert torch.allclose(vis3["lq"][0], torch.full((3, 8, 8), 0.2))

    acc.end_training()


# ---------------------------------------------------------------------------
# 4. Trainer end-to-end integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 2])
def test_trainer_logs_train_comparison_images_to_tensorboard(
    tmp_path: Path, scale: int
) -> None:
    ds = _create_train_manifests(tmp_path, scale=scale)
    work_dir = tmp_path / f"tb_train_cmp_x{scale}"
    cfg = _make_train_cfg(
        work_dir,
        ds,
        scale=scale,
        gt_size=32,
        batch_size=2,
        total_iters=5,
        log_interval=2,
        log_with=["tensorboard"],
        use_degradation=False,
        train_log_image_interval=2,
        train_max_log_images=2,
        train_log_image_mode="comparison",
    )

    state = Trainer(cfg).fit()
    assert state.global_step == 5

    tb_imgs = _read_tb_images(work_dir / "trackers" / work_dir.name)
    assert set(tb_imgs.keys()) == {
        "train/lq_pred_gt/sample_00",
        "train/lq_pred_gt/sample_01",
    }
    for tag in ("train/lq_pred_gt/sample_00", "train/lq_pred_gt/sample_01"):
        steps = [step for step, _ in tb_imgs[tag]]
        # Triggered on step % 2 == 0 (2, 4) and final step (5)
        assert steps == [2, 4, 5]
        _, arr = tb_imgs[tag][0]
        # Horizontal strip [LQ_bicubic | Pred | GT] -> (H=32, W=32*3=96)
        assert arr.shape == (32, 96, 3)
        # Left panel is solid red-ish LQ (220, 30, 20); right panel is solid green-ish GT (20, 220, 40)
        lq_panel = arr[:, :32, :]
        gt_panel = arr[:, 64:, :]
        assert np.abs(lq_panel.mean(axis=(0, 1)) - np.array([220, 30, 20])).max() < 2.0
        assert np.abs(gt_panel.mean(axis=(0, 1)) - np.array([20, 220, 40])).max() < 2.0


def test_trainer_logs_online_degraded_lq_gt_pred_with_grad_accum(tmp_path: Path) -> None:
    ds = _create_train_manifests(tmp_path, scale=2)
    work_dir = tmp_path / "tb_train_degraded_accum"
    cfg = _make_train_cfg(
        work_dir,
        ds,
        scale=2,
        gt_size=32,
        batch_size=1,
        total_iters=2,
        log_interval=1,
        grad_accum=2,
        log_with=["tensorboard"],
        use_degradation=True,
        train_log_image_interval=2,
        train_max_log_images=2,
        train_log_image_mode="comparison",
    )

    state = Trainer(cfg).fit()
    assert state.global_step == 2

    tb_imgs = _read_tb_images(work_dir / "trackers" / work_dir.name)
    # batch_size=1 with grad_accum=2 and train_max_log_images=2 collects 2 samples!
    assert set(tb_imgs.keys()) == {
        "train/lq_pred_gt/sample_00",
        "train/lq_pred_gt/sample_01",
    }
    _, arr = tb_imgs["train/lq_pred_gt/sample_00"][0]
    assert arr.shape == (32, 96, 3)
    # Post-degradation LQ must NOT be the zero placeholder from MixedSRDataset!
    lq_panel = arr[:, :32, :]
    assert float(lq_panel.mean()) > 10.0


def test_trainer_logs_separate_lq_gt_pred_images_to_mocked_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _create_train_manifests(tmp_path, scale=2)
    work_dir = tmp_path / "wandb_separate"
    cfg = _make_train_cfg(
        work_dir,
        ds,
        scale=2,
        gt_size=32,
        batch_size=2,
        total_iters=3,
        log_interval=0,
        log_with=["tensorboard"],
        train_log_image_interval=2,
        train_max_log_images=1,
        train_log_image_mode="separate",
    )

    trainer = Trainer(cfg)
    assert trainer.accelerator is not None
    # Replace active tracker list with a mocked WandB tracker to verify single-call batching.
    fake_run = MagicMock()
    fake_wandb = SimpleNamespace(Image=lambda arr: ("wandb_img", arr.shape, arr.dtype))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    trainer.accelerator.trackers = [
        SimpleNamespace(name="wandb", tracker=fake_run, finish=lambda: None)
    ]

    state = trainer.fit()
    assert state.global_step == 3

    # Logged at step 2 and final step 3 -> 2 calls total (1 batched call per step)
    assert fake_run.log.call_count == 2
    logged_steps = [call.kwargs["step"] for call in fake_run.log.call_args_list]
    assert logged_steps == [2, 3]
    for call in fake_run.log.call_args_list:
        payload = call.args[0]
        assert set(payload.keys()) == {
            "train/sample_00/lq",
            "train/sample_00/pred",
            "train/sample_00/gt",
        }
        # LQ is 16x16 (scale=2), Pred and GT are 32x32
        assert payload["train/sample_00/lq"] == ("wandb_img", (16, 16, 3), np.uint8)
        assert payload["train/sample_00/pred"] == ("wandb_img", (32, 32, 3), np.uint8)
        assert payload["train/sample_00/gt"] == ("wandb_img", (32, 32, 3), np.uint8)


def test_trainer_disabled_train_image_logging_and_custom_base_model(tmp_path: Path) -> None:
    ds = _create_train_manifests(tmp_path, scale=1)

    # Case 1: train_log_image_interval=0 disables training image logging even when TB is active
    work_disabled = tmp_path / "tb_disabled"
    cfg_disabled = _make_train_cfg(
        work_disabled,
        ds,
        scale=1,
        total_iters=2,
        log_interval=1,
        log_with=["tensorboard"],
        train_log_image_interval=0,
        train_max_log_images=4,
    )
    Trainer(cfg_disabled).fit()
    tb_imgs = _read_tb_images(work_disabled / "trackers" / work_disabled.name)
    assert tb_imgs == {}

    # Case 2: train_max_log_images=0 disables training image logging
    _reset_accelerate_and_loggers()
    work_zero_max = tmp_path / "tb_zero_max"
    cfg_zero_max = _make_train_cfg(
        work_zero_max,
        ds,
        scale=1,
        total_iters=2,
        log_interval=1,
        log_with=["tensorboard"],
        train_log_image_interval=1,
        train_max_log_images=0,
    )
    Trainer(cfg_zero_max).fit()
    assert _read_tb_images(work_zero_max / "trackers" / work_zero_max.name) == {}

    # Case 3: Custom BaseModel subclass without visual capture hooks works seamlessly
    _reset_accelerate_and_loggers()
    work_custom = tmp_path / "tb_custom_model"
    cfg_custom = _make_train_cfg(
        work_custom,
        ds,
        scale=1,
        total_iters=2,
        log_interval=1,
        log_with=["tensorboard"],
        train_log_image_interval=1,
        train_max_log_images=2,
        model_target="tests.test_train_visual_logging._NoVisualsCustomModel",
    )
    state = Trainer(cfg_custom).fit()
    assert state.global_step == 2
    assert _read_tb_images(work_custom / "trackers" / work_custom.name) == {}
