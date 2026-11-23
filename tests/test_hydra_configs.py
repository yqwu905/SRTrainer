"""Unit tests for Hydra config composition, inheritance, instantiation, and scope constraints."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

import archs
import losses
from archs import VOSR2OneStep
from losses import L1Loss, LPIPSLoss
from utils import normalize_mixed_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


def _compose(
    config_name: str,
    overrides: list[str] | None = None,
    return_hydra_config: bool = False,
):
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return hydra.compose(
            config_name=config_name,
            overrides=overrides or [],
            return_hydra_config=return_hydra_config,
        )


def test_train_default_compose_and_interpolations() -> None:
    cfg = _compose("train", return_hydra_config=True)

    assert cfg.exp_name == "debug_vosr2"
    assert cfg.scale == 1
    assert cfg.seed == 42
    assert cfg.output_root == "experiments"
    assert cfg.work_dir == "experiments/debug_vosr2"

    # Arch & Loss & Model
    assert cfg.arch._target_ == "archs.VOSR2OneStep"
    assert cfg.arch.scale == 1
    assert cfg.arch.hidden_size == 1536
    assert cfg.arch.depth == 36
    assert cfg.arch.num_heads == 24
    assert set(cfg.loss.keys()) == {"l1"}
    assert cfg.loss.l1.weight == 1.0
    assert cfg.loss.l1.criterion._target_ == "losses.L1Loss"
    assert cfg.model._target_ == "models.SRModel"
    assert cfg.model.ema.enabled is True
    assert cfg.model.pretrain_path is None

    # Data & Degradation
    assert cfg.data.gt_size == 256
    assert cfg.data.scale == 1
    assert cfg.data.degradation._target_ == "data.degradation.RealESRGANDegrader"
    assert cfg.data.degradation.scale == 1

    # Optim & Metric & Trainer
    assert cfg.optim.optimizer._partial_ is True
    assert cfg.optim.scheduler._partial_ is True
    assert cfg.optim.scheduler.T_max == 300000
    assert cfg.metric.primary_dataset == "set5"
    assert cfg.metric.primary_metric == "psnr"
    assert cfg.metric.val_suites.set5.metrics.psnr.crop_border == 1
    assert cfg.trainer.total_iters == 300000
    assert cfg.trainer.append_timestamp is False
    assert cfg.trainer.val_tile.size == "auto"
    assert cfg.trainer.mixed_precision == "bf16"

    # Hydra runtime safeguards
    assert cfg.hydra.job.chdir is False
    assert cfg.hydra.output_subdir is None
    assert cfg.hydra.run.dir == "experiments/debug_vosr2"
    assert cfg.hydra.job_logging.disable_existing_loggers is True


def test_vosr2_experiment_composition() -> None:
    cfg = _compose("train", overrides=["experiment=vosr2_onestep_text"])

    assert cfg.exp_name == "vosr2_onestep_text"
    assert cfg.work_dir == "experiments/vosr2_onestep_text"
    assert cfg.scale == 1
    assert cfg.arch._target_ == "archs.VOSR2OneStep"
    assert cfg.arch.scale == 1
    assert cfg.arch.hidden_size == 1536
    assert cfg.arch.depth == 36

    # Loss group override (l1 -> vosr2_text)
    assert set(cfg.loss.keys()) == {"l1", "lpips", "ocr_kl", "glyph"}
    assert cfg.loss.l1.criterion._target_ == "losses.L1Loss"
    assert cfg.loss.lpips.criterion._target_ == "losses.LPIPSLoss"
    assert cfg.loss.ocr_kl.criterion._target_ == "losses.OCRKLLoss"
    assert cfg.loss.glyph.criterion._target_ == "losses.GlyphLoss"

    # Localized overrides & cascading interpolations
    assert cfg.data.gt_size == 512
    assert cfg.data.scale == 1
    assert cfg.data.augment.hflip is False
    assert cfg.data.augment.vflip is False
    assert cfg.data.augment.rot90 is False
    assert cfg.optim.optimizer.lr == pytest.approx(5e-6)
    assert list(cfg.optim.optimizer.betas) == [0.9, 0.95]
    assert cfg.trainer.total_iters == 10000
    assert cfg.optim.scheduler.T_max == 10000
    assert cfg.trainer.val_interval == 1000
    assert cfg.trainer.save_interval == 1000
    assert cfg.trainer.mixed_precision == "bf16"
    assert cfg.trainer.val_tile.overlap == 64
    assert cfg.trainer.val_tile.pad_multiple == 16
    assert cfg.trainer.tracker_init_kwargs.wandb.name == "vosr2_onestep_text"


def test_cli_overrides_precedence() -> None:
    cfg = _compose(
        "train",
        overrides=[
            "experiment=vosr2_onestep_text",
            "optim.optimizer.lr=5e-5",
            "trainer.total_iters=50000",
            "trainer.log_with=[tensorboard,wandb]",
            "arch.depth=12",
            "scale=2",
            "trainer.mixed_precision=no",
            "trainer.resume=auto",
        ],
    )

    assert cfg.optim.optimizer.lr == 5e-5
    assert cfg.trainer.total_iters == 50000
    assert cfg.optim.scheduler.T_max == 50000
    assert list(cfg.trainer.log_with) == ["tensorboard", "wandb"]
    assert cfg.arch._target_ == "archs.VOSR2OneStep"
    assert cfg.arch.depth == 12
    assert cfg.arch.scale == 2
    assert cfg.data.scale == 2
    assert cfg.data.degradation.scale == 2
    assert cfg.metric.val_suites.set5.metrics.psnr.crop_border == 2
    assert normalize_mixed_precision(cfg.trainer.mixed_precision) == "no"
    assert cfg.trainer.resume == "auto"


def test_atomic_config_group_switches_and_disable_degradation() -> None:
    cfg = _compose("train", overrides=["arch=vosr2", "loss=l1_lpips", "optim=adam_multistep"])
    assert cfg.arch._target_ == "archs.VOSR2OneStep"
    assert set(cfg.loss.keys()) == {"l1", "lpips"}
    assert cfg.optim.optimizer._target_ == "torch.optim.Adam"
    assert cfg.optim.scheduler._target_ == "torch.optim.lr_scheduler.MultiStepLR"

    # Disable degradation via dot-key override (pure-paired training)
    cfg_no_deg = _compose("train", overrides=["data.degradation=null"])
    assert cfg_no_deg.data.degradation is None

    # Disable degradation via Hydra group deletion
    cfg_del_deg = _compose("train", overrides=["~data/degradation"])
    assert cfg_del_deg.data.get("degradation") is None


def test_infer_yaml_dual_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mode 1: exp_dir auto-restore defaults
    cfg_default = _compose("infer", return_hydra_config=True)
    assert cfg_default.arch is None
    assert cfg_default.scale == 1
    assert cfg_default.infer.exp_dir is None
    assert cfg_default.infer.use_ema is True
    assert cfg_default.infer.tile.enabled is True
    assert cfg_default.infer.tile.size == "auto"
    assert cfg_default.infer.tile.overlap == 16
    assert cfg_default.infer.tile.blend_mode == "cosine"
    assert cfg_default.hydra.job.chdir is False
    assert cfg_default.hydra.output_subdir is None
    assert cfg_default.hydra.run.dir == "outputs/infer"
    assert cfg_default.hydra.job_logging.disable_existing_loggers is True

    # Mode 2: standalone arch=vosr2 + weights override (tiny mocked backends for fast CPU check)
    from tests.helpers.fake_backends import patch_vosr2_backends

    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16)
    cfg_standalone = _compose(
        "infer",
        overrides=[
            "arch=vosr2",
            "scale=1",
            "infer.weights_path=/tmp/w.safetensors",
            "arch.input_size=8",
            "arch.in_channels=8",
            "arch.out_channels=4",
            "arch.hidden_size=32",
            "arch.depth=1",
            "arch.num_heads=4",
            "arch.mlp_ratio=2.0",
            "arch.z_dims=16",
            "arch.encdim_ratio=2",
            "arch.dino_size=28",
            "arch.dino_layer=0",
        ],
    )
    assert cfg_standalone.arch._target_ == "archs.VOSR2OneStep"
    assert cfg_standalone.arch.scale == 1
    assert cfg_standalone.infer.weights_path == "/tmp/w.safetensors"
    net = instantiate(cfg_standalone.arch).eval()
    assert isinstance(net, VOSR2OneStep)
    with torch.no_grad():
        out = net(torch.rand(1, 3, 16, 16))
    assert out.shape == (1, 3, 16, 16)


def test_hydra_instantiate_components(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline, patch_vosr2_backends

    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16)
    patch_lpips_offline(monkeypatch)
    cfg_exp = _compose(
        "train",
        overrides=[
            "experiment=vosr2_onestep_text",
            "loss=l1_lpips",
            "arch.input_size=8",
            "arch.in_channels=8",
            "arch.out_channels=4",
            "arch.hidden_size=32",
            "arch.depth=1",
            "arch.num_heads=4",
            "arch.mlp_ratio=2.0",
            "arch.z_dims=16",
            "arch.encdim_ratio=2",
            "arch.dino_size=28",
            "arch.dino_layer=0",
        ],
    )
    net = instantiate(cfg_exp.arch).eval()
    assert isinstance(net, VOSR2OneStep)
    assert net.scale == 1
    with torch.no_grad():
        assert net(torch.rand(1, 3, 16, 20)).shape == (1, 3, 16, 20)

    l1 = instantiate(cfg_exp.loss.l1.criterion)
    lpips_loss = instantiate(cfg_exp.loss.lpips.criterion)
    assert isinstance(l1, L1Loss)
    assert isinstance(lpips_loss, LPIPSLoss)

    dummy = nn.Linear(4, 4)
    opt_partial = instantiate(cfg_exp.optim.optimizer)
    opt = opt_partial(params=dummy.parameters())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-6)

    sched_partial = instantiate(cfg_exp.optim.scheduler)
    sched = sched_partial(optimizer=opt)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 10000

    cfg_multi = _compose("train", overrides=["optim=adam_multistep"])
    opt2 = instantiate(cfg_multi.optim.optimizer)(params=dummy.parameters())
    sched2 = instantiate(cfg_multi.optim.scheduler)(optimizer=opt2)
    assert isinstance(opt2, torch.optim.Adam)
    assert isinstance(sched2, torch.optim.lr_scheduler.MultiStepLR)


def test_narrowed_scope_and_mandatory_user_requirements() -> None:
    # Registered built-in archs and losses (placeholder ToyModel removed)
    assert archs.__all__ == ["VOSR2OneStep"]
    assert losses.__all__ == ["L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"]

    arch_py_files = sorted(p.name for p in (REPO_ROOT / "src" / "archs").glob("*.py"))
    assert arch_py_files == ["__init__.py", "textverifier.py"]
    assert (REPO_ROOT / "src" / "archs" / "vosr2" / "__init__.py").is_file()

    arch_yaml_files = sorted(p.name for p in (REPO_ROOT / "configs" / "arch").glob("*.yaml"))
    assert arch_yaml_files == ["vosr2.yaml"]

    loss_yaml_files = sorted(p.name for p in (REPO_ROOT / "configs" / "loss").glob("*.yaml"))
    assert loss_yaml_files == ["glyph.yaml", "l1.yaml", "l1_lpips.yaml", "ocr_kl.yaml", "vosr2_text.yaml"]

    exp_yaml_files = sorted(
        p.name for p in (REPO_ROOT / "configs" / "experiment").glob("*.yaml")
    )
    assert exp_yaml_files == ["vosr2_onestep_text.yaml"]

    cfg = _compose("train")

    # Requirement #2: NO `mode` key in data.sources or any YAML under configs/data/
    assert len(cfg.data.sources) >= 1
    for src in cfg.data.sources:
        assert "mode" not in src
        assert {"name", "root_dir", "json_path", "ratio"}.issubset(set(src.keys()))

    for yaml_path in (REPO_ROOT / "configs" / "data").glob("*.yaml"):
        raw = OmegaConf.load(yaml_path)
        for src in raw.get("sources", []):
            assert "mode" not in src

    # Requirement #3: Validation datasets ONLY in metric.val_suites, NEVER in data
    forbidden_data_keys = {"val", "val_datasets", "val_suites", "validation"}
    assert forbidden_data_keys.isdisjoint(set(cfg.data.keys()))
    assert "val_suites" in cfg.metric
    assert {"set5", "urban100"}.issubset(set(cfg.metric.val_suites.keys()))
    for suite_cfg in cfg.metric.val_suites.values():
        assert "dataset" in suite_cfg
        assert "root_dir" in suite_cfg.dataset and "json_path" in suite_cfg.dataset
        assert "metrics" in suite_cfg and len(suite_cfg.metrics) >= 1

    # Zero prohibited/removed placeholder symbols anywhere in src/, configs/, pyproject.toml
    prohibited_symbols = (
        "SRPredictor",
        "ToyModel",
        "RRDBNet",
        "SwinIR",
        "TinySRNet",
        "CharbonnierLoss",
        "FFTLoss",
        "_build_offline_textverifier",
        "_OFFLINE_TEST_CHARS",
        "_OfflineDinoStub",
        "DEFAULT_TEXTVERIFIER_REVISION",
    )
    scan_paths = [
        *(REPO_ROOT / "configs").rglob("*.yaml"),
        *(REPO_ROOT / "src").rglob("*.py"),
        REPO_ROOT / "pyproject.toml",
    ]
    for file_path in scan_paths:
        text = file_path.read_text(encoding="utf-8")
        for sym in prohibited_symbols:
            assert sym not in text, f"Prohibited symbol {sym!r} found in {file_path}"
