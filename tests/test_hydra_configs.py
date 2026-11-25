"""Unit tests for Hydra config composition, inheritance, instantiation, and scope constraints."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
import torch
from hydra.errors import ConfigCompositionException
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

import archs
import losses
from archs import AutoencoderKLQwenImage2D, DINOv2Encoder, VOSR2DiT
from config import load_train_config
from losses import L1Loss, LPIPSLoss
from models import VOSRModel, VOSRPipeline
from models.base_model import build_model
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


def test_train_requires_mandatory_groups_and_explicit_composition() -> None:
    # train.yaml intentionally leaves arch/vae, arch/dit, loss, and experiment mandatory (???)
    with pytest.raises(ConfigCompositionException):
        _compose("train")

    cfg = _compose(
        "train",
        overrides=[
            "~experiment",
            "arch/vae=qwen_vae2d",
            "arch/dit=vosr2_dit",
            "arch/dino=dinov2_vitl14",
            "loss=l1",
        ],
        return_hydra_config=True,
    )

    assert cfg.exp_name == "debug_vosr2"
    assert cfg.scale == 1
    assert cfg.seed == 42
    assert cfg.output_root == "experiments"
    assert cfg.work_dir == "experiments/debug_vosr2"

    # Independent arch configs & VOSRModel
    assert cfg.arch.vae._target_ == "archs.AutoencoderKLQwenImage2D"
    assert cfg.arch.vae.z_dim == 16
    assert cfg.arch.dit._target_ == "archs.VOSR2DiT"
    assert cfg.arch.dit.hidden_size == 1536
    assert cfg.arch.dit.depth == 36
    assert cfg.arch.dit.num_heads == 24
    assert cfg.arch.dino._target_ == "archs.DINOv2Encoder"
    assert cfg.arch.dino.layer == 17

    assert set(cfg.loss.keys()) == {"l1"}
    assert cfg.loss.l1.weight == 1.0
    assert cfg.loss.l1.criterion._target_ == "losses.L1Loss"
    assert cfg.model._target_ == "models.VOSRModel"
    assert cfg.model.scale == 1
    assert dict(cfg.model.trainable) == {"vae": False, "dit": True, "dino": False}
    assert cfg.model.ema.enabled is True
    assert cfg.model.save_frozen_weights is True
    assert cfg.model.pretrain.dit.path is None

    # Data & Degradation
    assert cfg.data.scale == 1
    assert cfg.data.degradation._target_ == "data.degradation.RealESRGANDegrader"
    assert cfg.data.degradation.scale == 1

    # Optim & Metric & Trainer
    assert cfg.optim.optimizer._partial_ is True
    assert cfg.optim.scheduler._partial_ is True
    assert cfg.optim.scheduler.T_max == 300000
    assert cfg.trainer.total_iters == 300000
    assert cfg.trainer.append_timestamp is False
    assert cfg.trainer.val_tile.size is None
    assert cfg.trainer.mixed_precision == "bf16"

    # Hydra runtime safeguards
    assert cfg.hydra.job.chdir is False
    assert cfg.hydra.output_subdir is None
    assert cfg.hydra.run.dir == "experiments/debug_vosr2"
    assert cfg.hydra.job_logging.disable_existing_loggers is True


def test_vosr2_experiment_composition_and_local_inheritance() -> None:
    cfg = _compose("train", overrides=["experiment=vosr2_onestep_text"])

    assert cfg.exp_name == "vosr2_onestep_text"
    assert cfg.work_dir == "experiments/vosr2_onestep_text"
    assert cfg.scale == 1
    assert cfg.model._target_ == "models.VOSRModel"
    assert cfg.model.scale == 1
    assert cfg.arch.vae._target_ == "archs.AutoencoderKLQwenImage2D"
    assert cfg.arch.dit._target_ == "archs.VOSR2DiT"
    assert cfg.arch.dino._target_ == "archs.DINOv2Encoder"
    assert cfg.arch.dit.hidden_size == 1536
    assert cfg.arch.dit.depth == 36

    # Loss group override (vosr2_text)
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

    # Local experiment multi-level inheritance overrides per-arch checkpoint paths cleanly
    cfg_local = _compose("train", overrides=["experiment=vosr2_onestep_text_local"])
    assert cfg_local.exp_name == "vosr2_onestep_text_local"
    assert set(cfg_local.loss.keys()) == {"l1", "lpips"}
    assert cfg_local.arch.vae.pretrained_path is not None
    assert cfg_local.arch.vae.subfolder is None
    assert cfg_local.arch.dit.pretrained_path is not None
    assert cfg_local.arch.dino.pretrained_path is not None
    assert cfg_local.arch.dino.local_repo is not None


def test_cli_overrides_precedence_and_component_toggles() -> None:
    cfg = _compose(
        "train",
        overrides=[
            "experiment=vosr2_onestep_text",
            "optim.optimizer.lr=5e-5",
            "trainer.total_iters=50000",
            "trainer.log_with=[tensorboard,wandb]",
            "arch.dit.depth=12",
            "model.trainable.vae=true",
            "model.trainable.dino=true",
            "scale=2",
            "trainer.mixed_precision=no",
            "trainer.resume=auto",
        ],
    )

    assert cfg.optim.optimizer.lr == 5e-5
    assert cfg.trainer.total_iters == 50000
    assert cfg.optim.scheduler.T_max == 50000
    assert list(cfg.trainer.log_with) == ["tensorboard", "wandb"]
    assert cfg.arch.dit._target_ == "archs.VOSR2DiT"
    assert cfg.arch.dit.depth == 12
    assert cfg.model.scale == 2
    assert cfg.model.trainable.vae is True
    assert cfg.model.trainable.dit is True
    assert cfg.model.trainable.dino is True
    assert cfg.data.scale == 2
    assert cfg.data.degradation.scale == 2
    assert normalize_mixed_precision(cfg.trainer.mixed_precision) == "no"
    assert cfg.trainer.resume == "auto"


def test_atomic_config_group_switches_and_disable_degradation_and_dino() -> None:
    cfg = _compose(
        "train",
        overrides=[
            "~experiment",
            "arch/vae=qwen_vae2d",
            "arch/dit=vosr2_dit",
            "loss=l1_lpips",
            "optim=adam_multistep",
        ],
    )
    assert cfg.arch.vae._target_ == "archs.AutoencoderKLQwenImage2D"
    assert cfg.arch.dit._target_ == "archs.VOSR2DiT"
    assert cfg.arch.get("dino") is None
    assert set(cfg.loss.keys()) == {"l1", "lpips"}
    assert cfg.optim.optimizer._target_ == "torch.optim.Adam"
    assert cfg.optim.scheduler._target_ == "torch.optim.lr_scheduler.MultiStepLR"

    # Disable dino from an experiment via ~arch/dino or arch.dino=null
    cfg_no_dino = _compose(
        "train", overrides=["experiment=vosr2_onestep_text", "~arch/dino", "arch.dit.z_dims=null"]
    )
    assert cfg_no_dino.arch.get("dino") is None
    assert cfg_no_dino.arch.dit.z_dims is None

    # Disable degradation via dot-key override (pure-paired training)
    cfg_no_deg = _compose(
        "train", overrides=["experiment=vosr2_onestep_text", "data.degradation=null"]
    )
    assert cfg_no_deg.data.degradation is None

    # Disable degradation via Hydra group deletion
    cfg_del_deg = _compose(
        "train", overrides=["experiment=vosr2_onestep_text", "~data/degradation"]
    )
    assert cfg_del_deg.data.get("degradation") is None


def test_infer_yaml_dual_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mode 1: exp_dir auto-restore defaults
    cfg_default = _compose("infer", return_hydra_config=True)
    assert cfg_default.get("arch") is None
    assert cfg_default.scale == 1
    assert cfg_default.infer.exp_dir is None
    assert cfg_default.infer.use_ema is True
    assert cfg_default.infer.tile.enabled is True
    assert cfg_default.infer.tile.size is None
    assert cfg_default.infer.tile.overlap == 16
    assert cfg_default.infer.tile.blend_mode == "cosine"
    assert cfg_default.hydra.job.chdir is False
    assert cfg_default.hydra.output_subdir is None
    assert cfg_default.hydra.run.dir == "outputs/infer"
    assert cfg_default.hydra.job_logging.disable_existing_loggers is True

    # Mode 2: standalone multi-arch VOSR composition + instantiation
    from tests.helpers.fake_backends import patch_vosr2_backends

    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16)
    cfg_standalone = _compose(
        "infer",
        overrides=[
            "model=vosr",
            "arch/vae=qwen_vae2d",
            "arch/dit=vosr2_dit",
            "arch/dino=dinov2_vitl14",
            "scale=1",
            "arch.vae.base_dim=8",
            "arch.vae.z_dim=4",
            "arch.vae.dim_mult=[1,2,2,2]",
            "arch.vae.num_res_blocks=1",
            "arch.vae.latents_mean=[0.0,0.0,0.0,0.0]",
            "arch.vae.latents_std=[1.0,1.0,1.0,1.0]",
            "arch.vae.pretrained_path=null",
            "arch.dit.input_size=8",
            "arch.dit.in_channels=8",
            "arch.dit.out_channels=4",
            "arch.dit.hidden_size=32",
            "arch.dit.depth=1",
            "arch.dit.num_heads=4",
            "arch.dit.mlp_ratio=2.0",
            "arch.dit.z_dims=16",
            "arch.dit.encdim_ratio=2",
            "arch.dino.image_size=28",
            "arch.dino.layer=0",
            "arch.dino.load_hub_weights=false",
        ],
    )
    vae = instantiate(cfg_standalone.arch.vae).eval()
    dit = instantiate(cfg_standalone.arch.dit).eval()
    dino = instantiate(cfg_standalone.arch.dino).eval()
    assert isinstance(vae, AutoencoderKLQwenImage2D)
    assert isinstance(dit, VOSR2DiT)
    assert isinstance(dino, DINOv2Encoder)
    pipe = VOSRPipeline(vae=vae, dit=dit, dino=dino, scale=cfg_standalone.scale).eval()
    with torch.no_grad():
        out = pipe(torch.rand(1, 3, 16, 16))
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
            "arch.vae.base_dim=8",
            "arch.vae.z_dim=4",
            "arch.vae.dim_mult=[1,2,2,2]",
            "arch.vae.num_res_blocks=1",
            "arch.vae.latents_mean=[0.0,0.0,0.0,0.0]",
            "arch.vae.latents_std=[1.0,1.0,1.0,1.0]",
            "arch.vae.pretrained_path=null",
            "arch.dit.input_size=8",
            "arch.dit.in_channels=8",
            "arch.dit.out_channels=4",
            "arch.dit.hidden_size=32",
            "arch.dit.depth=1",
            "arch.dit.num_heads=4",
            "arch.dit.mlp_ratio=2.0",
            "arch.dit.z_dims=16",
            "arch.dit.encdim_ratio=2",
            "arch.dino.image_size=28",
            "arch.dino.layer=0",
            "arch.dino.load_hub_weights=false",
        ],
    )
    vae = instantiate(cfg_exp.arch.vae).eval()
    dit = instantiate(cfg_exp.arch.dit).eval()
    dino = instantiate(cfg_exp.arch.dino).eval()
    assert isinstance(vae, AutoencoderKLQwenImage2D)
    assert isinstance(dit, VOSR2DiT)
    assert isinstance(dino, DINOv2Encoder)
    pipe = VOSRPipeline(vae=vae, dit=dit, dino=dino, scale=cfg_exp.scale).eval()
    with torch.no_grad():
        assert pipe(torch.rand(1, 3, 16, 20)).shape == (1, 3, 16, 20)

    model_obj = build_model(load_train_config(cfg_exp, require_data=False))
    assert isinstance(model_obj, VOSRModel)

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

    cfg_multi = _compose(
        "train",
        overrides=[
            "~experiment",
            "arch/vae=qwen_vae2d",
            "arch/dit=vosr2_dit",
            "loss=l1",
            "optim=adam_multistep",
        ],
    )
    opt2 = instantiate(cfg_multi.optim.optimizer)(params=dummy.parameters())
    sched2 = instantiate(cfg_multi.optim.scheduler)(optimizer=opt2)
    assert isinstance(opt2, torch.optim.Adam)
    assert isinstance(sched2, torch.optim.lr_scheduler.MultiStepLR)


def test_narrowed_scope_and_mandatory_user_requirements() -> None:
    # Registered built-in independent archs and losses
    assert set(archs.__all__) == {"AutoencoderKLQwenImage2D", "VOSR2DiT", "DINOv2Encoder"}
    assert losses.__all__ == ["L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"]

    arch_py_files = sorted(p.name for p in (REPO_ROOT / "src" / "archs").glob("*.py"))
    assert arch_py_files == ["__init__.py", "textverifier.py"]
    vosr_arch_files = sorted(p.name for p in (REPO_ROOT / "src" / "archs" / "vosr2").glob("*.py"))
    assert vosr_arch_files == ["__init__.py", "dinov2.py", "dit.py", "qwen_vae2d.py"]

    arch_yaml_files = sorted(
        p.relative_to(REPO_ROOT / "configs" / "arch").as_posix()
        for p in (REPO_ROOT / "configs" / "arch").rglob("*.yaml")
    )
    assert arch_yaml_files == [
        "dino/dinov2_vitl14.yaml",
        "dit/vosr2_dit.yaml",
        "vae/qwen_vae2d.yaml",
    ]

    model_yaml_files = sorted(p.name for p in (REPO_ROOT / "configs" / "model").glob("*.yaml"))
    assert model_yaml_files == ["vosr.yaml"]

    loss_yaml_files = sorted(p.name for p in (REPO_ROOT / "configs" / "loss").glob("*.yaml"))
    assert loss_yaml_files == ["glyph.yaml", "l1.yaml", "l1_lpips.yaml", "ocr_kl.yaml", "vosr2_text.yaml"]

    exp_yaml_files = sorted(
        p.name for p in (REPO_ROOT / "configs" / "experiment").glob("*.yaml")
    )
    assert exp_yaml_files == ["vosr2_onestep_text.yaml", "vosr2_onestep_text_local.yaml"]

    cfg = _compose("train", overrides=["experiment=vosr2_onestep_text"])

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
    assert len(cfg.metric.val_suites) >= 1
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
        "VOSR2OneStep",
        "SRModel",
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
    import re

    for file_path in scan_paths:
        text = file_path.read_text(encoding="utf-8")
        for sym in prohibited_symbols:
            assert re.search(rf"\b{re.escape(sym)}\b", text) is None, (
                f"Prohibited symbol {sym!r} found in {file_path}"
            )
