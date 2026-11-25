"""Tests for DiT LoRA config parsing, VOSRModel hook integration, full export, resume, and infer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from tools.infer import resolve_infer_config, run_infer

from archs import VOSR2DiT
from engine import Trainer
from models import VOSRModel, VOSRPipeline
from models import lora as dit_lora
from models.lora import DiTLoRAConfig, initialize_dit_ffn_big_lora, parse_dit_lora_config
from tests.dummy_arch import get_dummy_vosr_arch_dict
from tests.helpers.fake_backends import patch_vosr2_backends
from utils.misc import save_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


@pytest.fixture(autouse=True)
def reset_accelerate_state():
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


# ---------------------------------------------------------------------------
# Minimal mock LoRA helpers (in-place and PeftModel-style wrapper)
# ---------------------------------------------------------------------------


class _MockLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int | float) -> None:
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.05)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out


def _mock_inplace_lora(
    dit: nn.Module,
    lora_rank: int,
    lora_alpha: int | float,
) -> tuple[nn.Module, list[str]]:
    dit.requires_grad_(False)
    targets: list[str] = []
    for name, module in list(dit.named_modules()):
        if isinstance(module, nn.Linear) and ("mlp" in name or "attn" in name or "proj" in name):
            parent_name, _, attr = name.rpartition(".")
            parent = dit.get_submodule(parent_name) if parent_name else dit
            setattr(parent, attr, _MockLoRALinear(module, lora_rank, lora_alpha))
            targets.append(name)
    return dit, targets


class _MockPeftWrapper(nn.Module):
    """Simulates a PEFT-style wrapper whose state_dict keys are prefixed with ``base_model.model.``."""

    def __init__(self, model: nn.Module, rank: int, alpha: int | float) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = model
        self.rank = rank
        self.alpha = alpha
        self.adapter_bias = nn.Parameter(torch.zeros(1, model.out_channels, 1, 1))

    def get_base_model(self) -> nn.Module:
        return self.base_model.model

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model.model, name)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.base_model.model(*args, **kwargs) + self.adapter_bias


def _mock_peft_wrapper_lora(
    dit: nn.Module,
    lora_rank: int,
    lora_alpha: int | float,
) -> tuple[nn.Module, list[str]]:
    dit, targets = _mock_inplace_lora(dit, lora_rank, lora_alpha)
    wrapped = _MockPeftWrapper(dit, lora_rank, lora_alpha)
    return wrapped, targets


def _write_rgb_png(path: Path, h: int, w: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(16, 240, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


# ---------------------------------------------------------------------------
# 1. Config parsing, Hydra composition & stub behavior
# ---------------------------------------------------------------------------


def test_parse_dit_lora_config_and_stub_raises() -> None:
    assert parse_dit_lora_config(None) == DiTLoRAConfig(enabled=False, rank=16, alpha=16)
    assert parse_dit_lora_config({"enabled": True, "rank": 32}) == DiTLoRAConfig(
        enabled=True, rank=32, alpha=32
    )
    assert parse_dit_lora_config({"enabled": True, "rank": 8, "alpha": 16.5}) == DiTLoRAConfig(
        enabled=True, rank=8, alpha=16.5
    )

    with pytest.raises(TypeError, match="mapping or None"):
        parse_dit_lora_config("bad")
    with pytest.raises(ValueError, match="Unknown keys"):
        parse_dit_lora_config({"enabled": True, "target_modules": ["w12"]})
    with pytest.raises(ValueError, match="model.lora.enabled"):
        parse_dit_lora_config({"enabled": "yes"})
    with pytest.raises(ValueError, match="model.lora.rank"):
        parse_dit_lora_config({"rank": 0})
    with pytest.raises(ValueError, match="model.lora.rank"):
        parse_dit_lora_config({"rank": True})
    with pytest.raises(ValueError, match="model.lora.alpha"):
        parse_dit_lora_config({"alpha": -1.0})
    with pytest.raises(ValueError, match="model.lora.alpha"):
        parse_dit_lora_config({"alpha": float("nan")})

    with pytest.raises(NotImplementedError, match="initialize_dit_ffn_big_lora"):
        initialize_dit_ffn_big_lora(nn.Linear(4, 4), 8, 8)


def test_hydra_model_lora_defaults_and_overrides() -> None:
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(
            config_name="train",
            overrides=[
                "experiment=vosr2_onestep_text",
                "model.lora.enabled=true",
                "model.lora.rank=32",
                "model.lora.alpha=64",
            ],
        )
    assert cfg.model.lora.enabled is True
    assert cfg.model.lora.rank == 32
    assert cfg.model.lora.alpha == 64

    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg_default = hydra.compose(
            config_name="train",
            overrides=["experiment=vosr2_onestep_text"],
        )
    assert cfg_default.model.lora.enabled is False
    assert cfg_default.model.lora.rank == 16
    assert cfg_default.model.lora.alpha == 16


# ---------------------------------------------------------------------------
# 2. VOSRModel fail-fast validation & LoRA setup (in-place + wrapper)
# ---------------------------------------------------------------------------


def test_vosr_model_lora_fail_fast_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    base_cfg = {
        "scale": 1,
        "arch": get_dummy_vosr_arch_dict(with_dino=False),
        "loss": {"l1": {"weight": 1.0, "criterion": {"_target_": "losses.L1Loss"}}},
        "model": {
            "_target_": "models.VOSRModel",
            "scale": 1,
            "trainable": {"vae": True, "dit": False, "dino": False},
            "lora": {"enabled": True, "rank": 8, "alpha": 16},
            "ema": {"enabled": False},
        },
        "optim": {
            "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1e-3},
            "scheduler": None,
        },
    }
    with pytest.raises(ValueError, match="model.trainable.dit is False"):
        VOSRModel(base_cfg)

    # Stub returning non-nn.Module fails fast
    cfg_ok = OmegaConf.create(base_cfg)
    cfg_ok.model.trainable.dit = True
    monkeypatch.setattr(dit_lora, "initialize_dit_ffn_big_lora", lambda dit, r, a: ("not_a_mod", []))
    acc = Accelerator(cpu=True, mixed_precision="no")
    with pytest.raises(TypeError, match="initialize_dit_ffn_big_lora must return"):
        VOSRModel(cfg_ok).setup(acc)
    acc.end_training()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    # Stub freezing everything (0 trainable params) fails fast
    def _freeze_all(dit: nn.Module, r: int, a: int | float) -> tuple[nn.Module, list[str]]:
        dit.requires_grad_(False)
        return dit, []

    monkeypatch.setattr(dit_lora, "initialize_dit_ffn_big_lora", _freeze_all)
    acc2 = Accelerator(cpu=True, mixed_precision="no")
    with pytest.raises(ValueError, match="no parameters with requires_grad=True"):
        VOSRModel(cfg_ok).setup(acc2)
    acc2.end_training()


@pytest.mark.parametrize("injector_kind", ["inplace", "peft_wrapper"])
def test_vosr_model_lora_pretrain_train_ema_and_full_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injector_kind: str,
) -> None:
    torch.manual_seed(23)
    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16, dino_num_blocks=2)

    # Save a base (non-LoRA) VOSR2DiT checkpoint to prove pretrain loads BEFORE LoRA injection.
    base_dit = VOSR2DiT(
        input_size=4,
        patch_size=2,
        in_channels=8,
        out_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        z_dims=16,
        encdim_ratio=2,
        use_checkpoint=True,
    )
    with torch.no_grad():
        nn.init.normal_(base_dit.final_layer.linear.weight, std=0.1)
    base_dit_ckpt = save_safetensors(base_dit.state_dict(), tmp_path / "base_dit.safetensors")

    seen_calls: list[tuple[int, int | float]] = []
    injector = _mock_inplace_lora if injector_kind == "inplace" else _mock_peft_wrapper_lora

    def _spy_injector(dit: nn.Module, lora_rank: int, lora_alpha: int | float):
        seen_calls.append((lora_rank, lora_alpha))
        # Base pretrained weights must already be loaded before LoRA injection.
        assert torch.equal(dit.final_layer.linear.weight, base_dit.final_layer.linear.weight)
        return injector(dit, lora_rank, lora_alpha)

    monkeypatch.setattr(dit_lora, "initialize_dit_ffn_big_lora", _spy_injector)

    cfg = OmegaConf.create(
        {
            "scale": 1,
            "arch": {
                "vae": {
                    "_target_": "archs.AutoencoderKLQwenImage2D",
                    "base_dim": 8,
                    "z_dim": 4,
                    "dim_mult": [1, 2, 2, 2],
                    "num_res_blocks": 1,
                    "latents_mean": [0.0, 0.0, 0.0, 0.0],
                    "latents_std": [1.0, 1.0, 1.0, 1.0],
                    "gradient_checkpointing": False,
                    "pretrained_path": None,
                },
                "dit": {
                    "_target_": "archs.VOSR2DiT",
                    "input_size": 4,
                    "patch_size": 2,
                    "in_channels": 8,
                    "out_channels": 4,
                    "hidden_size": 32,
                    "depth": 2,
                    "num_heads": 4,
                    "mlp_ratio": 2.0,
                    "z_dims": 16,
                    "encdim_ratio": 2,
                    "use_checkpoint": True,
                    "pretrained_path": None,
                },
                "dino": {
                    "_target_": "archs.DINOv2Encoder",
                    "model_name": "dinov2_vitl14",
                    "image_size": 28,
                    "layer": 0,
                    "load_hub_weights": False,
                    "pretrained_path": None,
                },
            },
            "loss": {
                "l1": {
                    "weight": 1.0,
                    "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
                }
            },
            "model": {
                "_target_": "models.VOSRModel",
                "scale": 1,
                "trainable": {"vae": False, "dit": True, "dino": False},
                "lora": {"enabled": True, "rank": 4, "alpha": 8},
                "noise_input": "gaussian",
                "posterior_mode": True,
                "clamp_decode_in_train": False,
                "eval_seed": 42,
                "ema": {"enabled": True, "decay": 0.9, "warmup_steps": 0},
                "save_frozen_weights": True,
                "pretrain": {
                    "vae": {"path": None, "strict": True, "key": None},
                    "dit": {"path": str(base_dit_ckpt), "strict": True, "key": None},
                    "dino": {"path": None, "strict": True, "key": None},
                },
            },
            "optim": {
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "_partial_": True,
                    "lr": 1e-2,
                },
                "scheduler": None,
            },
            "trainer": {"grad_clip_norm": 1.0, "compile": False},
        }
    )

    acc = Accelerator(cpu=True, mixed_precision="no")
    model = VOSRModel(cfg)
    model.setup(acc)

    assert seen_calls == [(4, 8)]
    assert model.dit_lora_target_modules is not None and len(model.dit_lora_target_modules) > 0
    assert set(model.emas.keys()) == {"dit"}

    # Verify ONLY LoRA params are trainable (base weights stay frozen)
    dit_named = dict(model.dit.named_parameters())
    lora_names = [k for k, p in dit_named.items() if p.requires_grad]
    frozen_names = [k for k, p in dit_named.items() if not p.requires_grad]
    assert lora_names and frozen_names
    assert all(("lora_" in k or "adapter_bias" in k) for k in lora_names)
    assert all("lora_" not in k for k in frozen_names)

    init_snap = {k: v.detach().cpu().clone() for k, v in model.dit.state_dict().items()}

    # Two steps so lora_B moves away from zero and lora_A receives gradients too
    batch = {"lq": torch.rand(1, 3, 16, 16), "gt": torch.rand(1, 3, 16, 16)}
    for step in range(2):
        scalars, stepped = model.train_step(batch, global_step=step)
        assert stepped is True
        assert "loss/total" in scalars

    assert model.emas["dit"].num_updates == 2
    after_sd = model.dit.state_dict()
    for k in frozen_names:
        assert torch.equal(after_sd[k].detach().cpu(), init_snap[k]), (
            f"Frozen base weight {k!r} unexpectedly mutated!"
        )
    assert any(not torch.equal(after_sd[k].detach().cpu(), init_snap[k]) for k in lora_names)

    # Full LoRA weights are exported directly (no merge, no separate dit_lora key)
    exported = model.export_weights(acc)
    assert set(exported.keys()) == {"vae", "dit", "dit_ema", "dino"}
    assert set(exported["dit"].keys()) == set(model.dit.state_dict().keys())
    assert set(exported["dit_ema"].keys()) == set(model.emas["dit"].module.state_dict().keys())
    assert any("lora_A" in k for k in exported["dit"])
    assert any("lora_B" in k for k in exported["dit"])

    # EMA infer network runs forward cleanly
    infer_pipe = model.get_infer_network(use_ema=True)
    assert isinstance(infer_pipe, VOSRPipeline)
    with torch.no_grad():
        out = infer_pipe(torch.rand(1, 3, 16, 16))
    assert out.shape == (1, 3, 16, 16)
    acc.end_training()


# ---------------------------------------------------------------------------
# 3. End-to-end Trainer (train -> save -> resume) + tools/infer.py (Mode 1 & Mode 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("injector_kind", ["inplace", "peft_wrapper"])
def test_dit_lora_e2e_train_resume_and_infer_both_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injector_kind: str,
) -> None:
    torch.manual_seed(31)
    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16, dino_num_blocks=2)

    injector = _mock_inplace_lora if injector_kind == "inplace" else _mock_peft_wrapper_lora
    monkeypatch.setattr(dit_lora, "initialize_dit_ffn_big_lora", injector)

    # Build tiny synthetic paired dataset & inference input
    data_root = tmp_path / "data"
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(2):
        gt_rel = f"hr_{i}.png"
        lq_rel = f"lr_{i}.png"
        _write_rgb_png(data_root / gt_rel, 16, 16, seed=10 + i)
        _write_rgb_png(data_root / lq_rel, 16, 16, seed=20 + i)
        entries.append({"gt": gt_rel, "lq": lq_rel})
    train_json = manifests_dir / "train.json"
    train_json.write_text(json.dumps(entries), encoding="utf-8")

    exp_dir = tmp_path / f"exp_{injector_kind}"
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["experiment=vosr2_onestep_text"])

    OmegaConf.set_struct(cfg, False)
    cfg.exp_name = exp_dir.name
    cfg.scale = 1
    cfg.seed = 42
    cfg.output_root = str(exp_dir.parent)
    cfg.work_dir = str(exp_dir)
    cfg.arch = OmegaConf.create(
        {
            "vae": {
                "_target_": "archs.AutoencoderKLQwenImage2D",
                "base_dim": 8,
                "z_dim": 4,
                "dim_mult": [1, 2, 2, 2],
                "num_res_blocks": 1,
                "latents_mean": [0.0, 0.0, 0.0, 0.0],
                "latents_std": [1.0, 1.0, 1.0, 1.0],
                "gradient_checkpointing": False,
                "pretrained_path": None,
            },
            "dit": {
                "_target_": "archs.VOSR2DiT",
                "input_size": 4,
                "patch_size": 2,
                "in_channels": 8,
                "out_channels": 4,
                "hidden_size": 32,
                "depth": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "z_dims": 16,
                "encdim_ratio": 2,
                "use_checkpoint": False,
                "pretrained_path": None,
            },
            "dino": {
                "_target_": "archs.DINOv2Encoder",
                "model_name": "dinov2_vitl14",
                "image_size": 28,
                "layer": 0,
                "load_hub_weights": False,
                "pretrained_path": None,
            },
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
    cfg.model.scale = 1
    cfg.model.trainable = OmegaConf.create({"vae": False, "dit": True, "dino": False})
    cfg.model.lora = OmegaConf.create({"enabled": True, "rank": 4, "alpha": 8})
    cfg.model.ema = OmegaConf.create({"enabled": True, "decay": 0.9, "warmup_steps": 0})
    cfg.model.save_frozen_weights = True
    cfg.data.gt_size = 16
    cfg.data.scale = 1
    cfg.data.batch_size_per_gpu = 1
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.degradation = None
    cfg.data.sources = OmegaConf.create(
        [
            {
                "name": "paired",
                "root_dir": str(data_root),
                "json_path": str(train_json),
                "ratio": 1.0,
            }
        ]
    )
    cfg.metric = OmegaConf.create(
        {"primary_dataset": None, "primary_metric": None, "val_suites": {}}
    )
    cfg.optim.optimizer.lr = 1e-2
    cfg.optim.scheduler = None
    cfg.trainer.total_iters = 2
    cfg.trainer.log_interval = 1
    cfg.trainer.val_interval = 0
    cfg.trainer.save_interval = 2
    cfg.trainer.mixed_precision = "no"
    cfg.trainer.log_with = None
    cfg.trainer.train_log_image_interval = 0

    # Phase 1: Train 2 steps and save full state + full LoRA weights
    trainer1 = Trainer(cfg)
    state1 = trainer1.fit()
    assert state1.global_step == 2
    assert (exp_dir / "checkpoints" / "state_iter_000002").is_dir()
    dit_ema_path = exp_dir / "weights" / "latest_dit_ema.safetensors"
    assert dit_ema_path.is_file()
    saved_dit_sd = load_safetensors_file(str(dit_ema_path))
    assert any("lora_A" in k for k in saved_dit_sd)

    # Phase 2: Resume for 1 more step
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    cfg_resume = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_resume.trainer.total_iters = 3
    cfg_resume.trainer.save_interval = 1
    cfg_resume.trainer.resume = "auto"
    trainer2 = Trainer(cfg_resume)
    state2 = trainer2.fit()
    assert state2.global_step == 3

    # Phase 3: Mode 1 inference (exp_dir auto-restore: attaches LoRA then loads full LoRA weights)
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    out_dir_m1 = tmp_path / f"infer_m1_{injector_kind}"
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        infer_cfg_m1 = hydra.compose(config_name="infer")
    OmegaConf.set_struct(infer_cfg_m1, False)
    infer_cfg_m1.infer.exp_dir = str(exp_dir)
    infer_cfg_m1.infer.input = str(data_root / "lr_0.png")
    infer_cfg_m1.infer.output_dir = str(out_dir_m1)
    infer_cfg_m1.infer.mixed_precision = "no"
    infer_cfg_m1.infer.tile.enabled = False
    infer_cfg_m1.infer.tile.pad_multiple = 16

    resolved_m1 = resolve_infer_config(infer_cfg_m1)
    assert resolved_m1.dit_lora_cfg == DiTLoRAConfig(enabled=True, rank=4, alpha=8)

    summary_m1 = run_infer(infer_cfg_m1)
    assert summary_m1.total_images == 1
    assert (out_dir_m1 / "lr_0.png").is_file()

    # Phase 4: Mode 2 inference (standalone with model.lora enabled)
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    out_dir_m2 = tmp_path / f"infer_m2_{injector_kind}"
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        infer_cfg_m2 = hydra.compose(config_name="infer", overrides=["model=vosr"])
    OmegaConf.set_struct(infer_cfg_m2, False)
    infer_cfg_m2.scale = 1
    infer_cfg_m2.arch = OmegaConf.create(OmegaConf.to_container(cfg.arch, resolve=True))
    infer_cfg_m2.model.lora = OmegaConf.create({"enabled": True, "rank": 4, "alpha": 8})
    infer_cfg_m2.infer.exp_dir = None
    infer_cfg_m2.infer.weights_path = OmegaConf.create(
        {
            "vae": str(exp_dir / "weights" / "latest_vae.safetensors"),
            "dit": str(exp_dir / "weights" / "latest_dit_ema.safetensors"),
            "dino": str(exp_dir / "weights" / "latest_dino.safetensors"),
        }
    )
    infer_cfg_m2.infer.input = str(data_root / "lr_0.png")
    infer_cfg_m2.infer.output_dir = str(out_dir_m2)
    infer_cfg_m2.infer.mixed_precision = "no"
    infer_cfg_m2.infer.tile.enabled = False
    infer_cfg_m2.infer.tile.pad_multiple = 16

    summary_m2 = run_infer(infer_cfg_m2)
    assert summary_m2.total_images == 1
    assert (out_dir_m2 / "lr_0.png").is_file()

    # Mode 1 and Mode 2 must produce identical output images (deterministic eval_seed + same weights)
    img1 = np.asarray(Image.open(out_dir_m1 / "lr_0.png"))
    img2 = np.asarray(Image.open(out_dir_m2 / "lr_0.png"))
    assert np.array_equal(img1, img2)
