"""Unit tests for Hydra config composition, multi-level inheritance, instantiation, and scope constraints."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

import srtrainer
import srtrainer.archs
import srtrainer.losses
from srtrainer.archs import ToyModel
from srtrainer.losses import L1Loss, LPIPSLoss
from srtrainer.utils import normalize_mixed_precision

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

    assert cfg.exp_name == "debug_sr_x4"
    assert cfg.scale == 4
    assert cfg.seed == 42
    assert cfg.output_root == "experiments"
    assert cfg.work_dir == "experiments/debug_sr_x4"

    # Arch & Loss & Model
    assert cfg.arch._target_ == "srtrainer.archs.ToyModel"
    assert cfg.arch.num_in_ch == 3
    assert cfg.arch.num_out_ch == 3
    assert cfg.arch.num_feat == 32
    assert cfg.arch.num_block == 2
    assert cfg.arch.scale == 4
    assert set(cfg.loss.keys()) == {"l1"}
    assert cfg.loss.l1.weight == 1.0
    assert cfg.loss.l1.criterion._target_ == "srtrainer.losses.L1Loss"
    assert cfg.model._target_ == "srtrainer.models.SRModel"
    assert cfg.model.ema.enabled is True
    assert cfg.model.pretrain_path is None

    # Data & Degradation
    assert cfg.data.gt_size == 256
    assert cfg.data.scale == 4
    assert cfg.data.degradation._target_ == "srtrainer.data.degradation.RealESRGANDegrader"
    assert cfg.data.degradation.scale == 4

    # Optim & Metric & Trainer
    assert cfg.optim.optimizer._partial_ is True
    assert cfg.optim.scheduler._partial_ is True
    assert cfg.optim.scheduler.T_max == 300000
    assert cfg.metric.primary_dataset == "set5"
    assert cfg.metric.primary_metric == "psnr"
    assert cfg.metric.val_suites.set5.metrics.psnr.crop_border == 4
    assert cfg.trainer.total_iters == 300000
    assert cfg.trainer.append_timestamp is False
    assert cfg.trainer.val_tile.size == "auto"
    assert cfg.trainer.mixed_precision == "bf16"

    # Hydra runtime safeguards
    assert cfg.hydra.job.chdir is False
    assert cfg.hydra.output_subdir is None
    assert cfg.hydra.run.dir == "experiments/debug_sr_x4"
    assert cfg.hydra.job_logging.disable_existing_loggers is True


def test_base_sr_x4_experiment() -> None:
    cfg = _compose("train", overrides=["experiment=base_sr_x4"])

    assert cfg.scale == 4
    assert cfg.arch._target_ == "srtrainer.archs.ToyModel"
    assert cfg.arch.num_feat == 32
    assert cfg.arch.num_block == 2
    assert cfg.trainer.total_iters == 300000
    assert cfg.trainer.val_interval == 5000
    assert cfg.trainer.save_interval == 5000
    assert cfg.trainer.mixed_precision == "bf16"
    assert cfg.trainer.gradient_accumulation_steps == 1
    assert set(cfg.loss.keys()) == {"l1"}


def test_multi_level_experiment_inheritance() -> None:
    cfg = _compose("train", overrides=["experiment=exp01_toy_mixed_x4"])

    assert cfg.exp_name == "exp01_toy_l1_lpips_x4"
    assert cfg.work_dir == "experiments/exp01_toy_l1_lpips_x4"
    assert cfg.scale == 4
    assert cfg.arch._target_ == "srtrainer.archs.ToyModel"
    assert cfg.arch.num_feat == 16
    assert cfg.arch.num_block == 2
    assert cfg.arch.scale == 4

    # Group override (l1 -> l1_lpips) + localized weight override
    assert set(cfg.loss.keys()) == {"l1", "lpips"}
    assert cfg.loss.l1.weight == 1.0
    assert cfg.loss.lpips.weight == 0.05
    assert cfg.loss.lpips.criterion._target_ == "srtrainer.losses.LPIPSLoss"
    assert cfg.loss.lpips.criterion.pretrained is True

    # Localized overrides & inherited values from base_sr_x4
    assert cfg.optim.optimizer.lr == 1e-4
    assert cfg.trainer.total_iters == 200000
    assert cfg.optim.scheduler.T_max == 200000
    assert cfg.trainer.val_interval == 5000
    assert cfg.trainer.save_interval == 5000
    assert cfg.trainer.mixed_precision == "bf16"
    assert cfg.trainer.tracker_init_kwargs.wandb.name == "exp01_toy_l1_lpips_x4"


def test_cli_overrides_precedence() -> None:
    cfg = _compose(
        "train",
        overrides=[
            "experiment=exp01_toy_mixed_x4",
            "optim.optimizer.lr=5e-5",
            "trainer.total_iters=50000",
            "trainer.log_with=[tensorboard,wandb]",
            "arch.num_feat=8",
            "arch.num_block=1",
            "scale=2",
            "trainer.mixed_precision=no",
            "trainer.resume=auto",
        ],
    )

    assert cfg.optim.optimizer.lr == 5e-5
    assert cfg.trainer.total_iters == 50000
    assert cfg.optim.scheduler.T_max == 50000
    assert list(cfg.trainer.log_with) == ["tensorboard", "wandb"]
    assert cfg.arch._target_ == "srtrainer.archs.ToyModel"
    assert cfg.arch.num_feat == 8
    assert cfg.arch.num_block == 1
    assert cfg.arch.scale == 2
    assert cfg.data.scale == 2
    assert cfg.data.degradation.scale == 2
    assert cfg.metric.val_suites.set5.metrics.psnr.crop_border == 2
    assert normalize_mixed_precision(cfg.trainer.mixed_precision) == "no"
    assert cfg.trainer.resume == "auto"


def test_atomic_config_group_switches_and_disable_degradation() -> None:
    cfg = _compose("train", overrides=["arch=toy_x4", "loss=l1_lpips", "optim=adam_multistep"])
    assert cfg.arch._target_ == "srtrainer.archs.ToyModel"
    assert set(cfg.loss.keys()) == {"l1", "lpips"}
    assert cfg.optim.optimizer._target_ == "torch.optim.Adam"
    assert cfg.optim.scheduler._target_ == "torch.optim.lr_scheduler.MultiStepLR"

    # Disable degradation via dot-key override (pure-paired training)
    cfg_no_deg = _compose("train", overrides=["data.degradation=null"])
    assert cfg_no_deg.data.degradation is None

    # Disable degradation via Hydra group deletion
    cfg_del_deg = _compose("train", overrides=["~data/degradation"])
    assert cfg_del_deg.data.get("degradation") is None


def test_infer_yaml_dual_modes() -> None:
    # Mode 1: exp_dir auto-restore defaults
    cfg_default = _compose("infer", return_hydra_config=True)
    assert cfg_default.arch is None
    assert cfg_default.scale == 4
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

    # Mode 2: standalone arch=toy_x4 + weights override
    cfg_standalone = _compose(
        "infer",
        overrides=[
            "arch=toy_x4",
            "scale=4",
            "infer.weights_path=/tmp/w.safetensors",
            "arch.num_feat=16",
            "arch.num_block=1",
        ],
    )
    assert cfg_standalone.arch._target_ == "srtrainer.archs.ToyModel"
    assert cfg_standalone.arch.scale == 4
    assert cfg_standalone.arch.num_feat == 16
    assert cfg_standalone.arch.num_block == 1
    assert cfg_standalone.infer.weights_path == "/tmp/w.safetensors"
    net = instantiate(cfg_standalone.arch)
    assert isinstance(net, ToyModel)
    assert net(torch.randn(1, 3, 8, 8)).shape == (1, 3, 32, 32)


def test_hydra_instantiate_m1_components() -> None:
    # Default ToyModel instantiation
    cfg_default = _compose("train")
    toy_default = instantiate(cfg_default.arch)
    assert isinstance(toy_default, ToyModel)
    assert toy_default.num_feat == 32
    assert toy_default.num_block == 2
    assert toy_default.scale == 4
    assert toy_default(torch.randn(1, 3, 8, 8)).shape == (1, 3, 32, 32)

    # Multi-level inherited ToyModel (num_feat=16) + losses + partial optim/scheduler
    cfg_exp01 = _compose(
        "train",
        overrides=["experiment=exp01_toy_mixed_x4", "loss.lpips.criterion.pretrained=false"],
    )
    toy_exp01 = instantiate(cfg_exp01.arch)
    assert isinstance(toy_exp01, ToyModel)
    assert toy_exp01.num_feat == 16
    assert toy_exp01(torch.randn(1, 3, 6, 7)).shape == (1, 3, 24, 28)

    l1 = instantiate(cfg_exp01.loss.l1.criterion)
    lpips_loss = instantiate(cfg_exp01.loss.lpips.criterion)
    assert isinstance(l1, L1Loss)
    assert isinstance(lpips_loss, LPIPSLoss)
    assert lpips_loss.pretrained is False

    dummy = nn.Linear(4, 4)
    opt_partial = instantiate(cfg_exp01.optim.optimizer)
    opt = opt_partial(params=dummy.parameters())
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)

    sched_partial = instantiate(cfg_exp01.optim.scheduler)
    sched = sched_partial(optimizer=opt)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 200000

    cfg_multi = _compose("train", overrides=["optim=adam_multistep"])
    opt2 = instantiate(cfg_multi.optim.optimizer)(params=dummy.parameters())
    sched2 = instantiate(cfg_multi.optim.scheduler)(optimizer=opt2)
    assert isinstance(opt2, torch.optim.Adam)
    assert isinstance(sched2, torch.optim.lr_scheduler.MultiStepLR)


def test_narrowed_scope_and_mandatory_user_requirements() -> None:
    # Registered built-in archs and losses
    assert srtrainer.archs.__all__ == ["ToyModel", "VOSR2OneStep"]
    assert srtrainer.losses.__all__ == ["L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"]

    arch_py_files = sorted(
        p.name for p in (REPO_ROOT / "src" / "srtrainer" / "archs").glob("*.py")
    )
    assert arch_py_files == ["__init__.py", "toy_model.py"]
    assert (REPO_ROOT / "src" / "srtrainer" / "archs" / "vosr2" / "__init__.py").is_file()

    arch_yaml_files = sorted(p.name for p in (REPO_ROOT / "configs" / "arch").glob("*.yaml"))
    assert arch_yaml_files == ["toy_x4.yaml", "vosr2_x4.yaml"]

    loss_yaml_files = sorted(p.name for p in (REPO_ROOT / "configs" / "loss").glob("*.yaml"))
    assert loss_yaml_files == ["glyph.yaml", "l1.yaml", "l1_lpips.yaml", "ocr_kl.yaml", "vosr2_text.yaml"]

    exp_yaml_files = sorted(
        p.name for p in (REPO_ROOT / "configs" / "experiment").glob("*.yaml")
    )
    assert exp_yaml_files == ["base_sr_x4.yaml", "exp01_toy_mixed_x4.yaml", "exp02_vosr2_x4.yaml"]

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

    # Requirement #4 + Follow-up scope narrowing: ZERO prohibited symbols anywhere in src/, configs/, pyproject.toml
    assert not hasattr(srtrainer, "SRPredictor")
    prohibited_symbols = (
        "SRPredictor",
        "RRDBNet",
        "SwinIR",
        "TinySRNet",
        "CharbonnierLoss",
        "FFTLoss",
    )
    scan_paths = [
        *(REPO_ROOT / "configs").rglob("*.yaml"),
        *(REPO_ROOT / "src" / "srtrainer").rglob("*.py"),
        REPO_ROOT / "pyproject.toml",
    ]
    for file_path in scan_paths:
        text = file_path.read_text(encoding="utf-8")
        for sym in prohibited_symbols:
            assert sym not in text, f"Prohibited symbol {sym!r} found in {file_path}"
