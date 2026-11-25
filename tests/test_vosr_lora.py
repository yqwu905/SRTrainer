"""Unit and integration tests for VOSR DiT LoRA config, training, checkpointing, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState
from PIL import Image
from tools.infer import run_infer
from torch import nn

import models.vosr_model as vosr_mod
from config import LoRAConfig, load_train_config
from models import VOSRModel, VOSRPipeline
from models.base_model import build_model
from models.vosr_model import attach_dit_lora, should_attach_dit_lora
from tests.dummy_arch import get_dummy_vosr_arch_dict
from utils.config import save_resolved_config
from utils.misc import save_safetensors


@pytest.fixture
def reset_accelerate_state():
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


class _TinyLoRALinear(nn.Module):
    """Minimal LoRA adapter wrapping an existing ``nn.Linear`` for testing."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.1)
        nn.init.normal_(self.lora_B.weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def _inject_tiny_lora_inplace(dit: nn.Module, lora_cfg: LoRAConfig) -> nn.Module:
    """Test LoRA injector replacing ``t_proj`` (and ``z_proj`` if present) in-place."""
    if hasattr(dit, "t_proj") and isinstance(dit.t_proj, nn.Linear):
        dit.t_proj = _TinyLoRALinear(dit.t_proj, rank=lora_cfg.lora_rank, alpha=lora_cfg.lora_alpha)
    if hasattr(dit, "z_proj") and isinstance(dit.z_proj, nn.Linear):
        dit.z_proj = _TinyLoRALinear(dit.z_proj, rank=lora_cfg.lora_rank, alpha=lora_cfg.lora_alpha)
    return dit


class _TinyLoRAWrapper(nn.Module):
    """Test LoRA wrapper mimicking PEFT-style outer wrapper modules."""

    def __init__(self, base_dit: nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        self.base_dit = base_dit
        self.out_channels = int(base_dit.out_channels)  # type: ignore[arg-type]
        self.in_channels = int(base_dit.in_channels)  # type: ignore[arg-type]
        self.patch_size = int(base_dit.patch_size)  # type: ignore[arg-type]
        self.z_dims = base_dit.z_dims
        self.lora_down = nn.Conv2d(self.out_channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(rank, self.out_channels, kernel_size=1, bias=False)
        nn.init.normal_(self.lora_down.weight, std=0.1)
        nn.init.normal_(self.lora_up.weight, std=0.1)
        self.scaling = float(alpha) / float(rank)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        out = self.base_dit(x, t, z=z)
        return out + self.lora_up(self.lora_down(out)) * self.scaling


def _make_vosr_train_dict(
    *,
    trainable: dict[str, bool] | None = None,
    lora: dict[str, Any] | None = None,
    pretrain: dict[str, Any] | None = None,
    ema_enabled: bool = True,
    with_dino: bool = True,
) -> dict[str, Any]:
    model_dict: dict[str, Any] = {
        "_target_": "models.VOSRModel",
        "scale": 1,
        "trainable": trainable or {"vae": False, "dit": True, "dino": False},
        "lora": lora
        if lora is not None
        else {"enable": False, "lora_rank": 32, "lora_alpha": 32, "checkpoint": None},
        "noise_input": "gaussian",
        "posterior_mode": True,
        "clamp_decode_in_train": False,
        "eval_seed": 42,
        "ema": {"enabled": ema_enabled, "decay": 0.9, "warmup_steps": 0},
        "save_frozen_weights": True,
    }
    if pretrain is not None:
        model_dict["pretrain"] = pretrain
    return {
        "scale": 1,
        "arch": get_dummy_vosr_arch_dict(with_dino=with_dino),
        "loss": {
            "l1": {
                "weight": 1.0,
                "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
            }
        },
        "model": model_dict,
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


def _write_rgb_png(path: Path, hw: tuple[int, int] = (16, 16), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(hw[0], hw[1], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def test_attach_dit_lora_stub_and_unimplemented_fail_fast(reset_accelerate_state) -> None:
    dummy = nn.Linear(4, 4)
    lora_cfg = LoRAConfig(enable=True, lora_rank=8, lora_alpha=16)
    assert attach_dit_lora(dummy, lora_cfg) is dummy

    cfg = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": False, "dit": True, "dino": False},
            lora={"enable": True, "lora_rank": 8, "lora_alpha": 16, "checkpoint": None},
        ),
        require_data=False,
        require_trainer_full=False,
    )
    assert should_attach_dit_lora(cfg.model) is True

    acc = Accelerator(cpu=True, mixed_precision="no")
    model = build_model(cfg)
    assert isinstance(model, VOSRModel)
    assert model.dit_lora_enabled is True
    with pytest.raises(ValueError, match="attach_dit_lora"):
        model.setup(acc)
    acc.end_training()


def test_dit_lora_trains_only_lora_params_and_exports_full_weights(
    monkeypatch: pytest.MonkeyPatch,
    reset_accelerate_state,
) -> None:
    torch.manual_seed(13)
    calls: list[LoRAConfig] = []

    def _spy_attach(dit: nn.Module, lora_cfg: LoRAConfig) -> nn.Module:
        calls.append(lora_cfg)
        return _inject_tiny_lora_inplace(dit, lora_cfg)

    monkeypatch.setattr(vosr_mod, "attach_dit_lora", _spy_attach)

    cfg = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": False, "dit": True, "dino": False},
            lora={"enable": True, "lora_rank": 4, "lora_alpha": 8, "checkpoint": None},
            ema_enabled=True,
        ),
        require_data=False,
        require_trainer_full=False,
    )
    acc = Accelerator(cpu=True, mixed_precision="no")
    model = build_model(cfg)
    assert isinstance(model, VOSRModel)
    model.setup(acc)

    assert len(calls) == 1
    assert calls[0] == LoRAConfig(enable=True, lora_rank=4, lora_alpha=8.0, checkpoint=None)
    assert model.trainable_components == ("dit",)
    assert set(model.emas.keys()) == {"dit"}

    # Only LoRA params on DiT have requires_grad=True; all base DiT params & VAE/DINO are frozen
    dit_named = dict(model.dit.named_parameters())
    lora_names = {k for k in dit_named if "lora" in k.lower()}
    base_names = set(dit_named.keys()) - lora_names
    assert lora_names, "Expected injected LoRA parameters on DiT"
    assert base_names, "Expected base parameters on DiT"
    assert all(dit_named[k].requires_grad for k in lora_names)
    assert all(not dit_named[k].requires_grad for k in base_names)
    assert all(not p.requires_grad for p in model.vae.parameters())
    assert model.dino is not None and all(not p.requires_grad for p in model.dino.parameters())

    # Optimizer must only hold the LoRA parameters
    opt_param_ids = {
        id(p) for group in model.optimizer.param_groups for p in group["params"]  # type: ignore[union-attr]
    }
    assert opt_param_ids == {id(dit_named[k]) for k in lora_names}

    init_dit = {k: v.detach().cpu().clone() for k, v in model.dit.state_dict().items()}
    init_vae = {k: v.detach().cpu().clone() for k, v in model.vae.state_dict().items()}

    batch = {"lq": torch.rand(1, 3, 16, 16), "gt": torch.rand(1, 3, 16, 16)}
    scalars, stepped = model.train_step(batch, global_step=0)
    assert stepped is True
    assert "loss/total" in scalars
    assert model.emas["dit"].num_updates == 1

    # LoRA params updated; base DiT params & VAE completely unchanged
    cur_dit = model.dit.state_dict()
    assert any(not torch.equal(cur_dit[k].detach().cpu(), init_dit[k]) for k in lora_names)
    assert all(torch.equal(cur_dit[k].detach().cpu(), init_dit[k]) for k in base_names)
    assert all(
        torch.equal(v.detach().cpu(), init_vae[k]) for k, v in model.vae.state_dict().items()
    )

    # Full weight export (base + LoRA together, no separate lora key and no merging)
    exported = model.export_weights(acc)
    assert set(exported.keys()) == {"vae", "dit", "dino", "dit_ema"}
    assert set(exported["dit"].keys()) == set(model.dit.state_dict().keys())
    assert set(exported["dit_ema"].keys()) == set(model.dit.state_dict().keys())
    assert lora_names.issubset(exported["dit"].keys())
    assert base_names.issubset(exported["dit"].keys())

    exported_infer = model.export_infer_weights(acc, use_ema=True)
    assert set(exported_infer.keys()) == {"vae", "dit", "dino"}
    assert set(exported_infer["dit"].keys()) == set(model.dit.state_dict().keys())
    acc.end_training()


def test_dit_not_trainable_ignores_lora_and_freezes_dit(
    monkeypatch: pytest.MonkeyPatch,
    reset_accelerate_state,
) -> None:
    torch.manual_seed(19)
    calls: list[LoRAConfig] = []

    def _spy_attach(dit: nn.Module, lora_cfg: LoRAConfig) -> nn.Module:
        calls.append(lora_cfg)
        return _inject_tiny_lora_inplace(dit, lora_cfg)

    monkeypatch.setattr(vosr_mod, "attach_dit_lora", _spy_attach)

    cfg = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": True, "dit": False, "dino": False},
            lora={
                "enable": True,
                "lora_rank": 16,
                "lora_alpha": 16,
                "checkpoint": "/nonexistent/ignored_when_dit_frozen.safetensors",
            },
            ema_enabled=True,
        ),
        require_data=False,
        require_trainer_full=False,
    )
    assert should_attach_dit_lora(cfg.model) is False

    acc = Accelerator(cpu=True, mixed_precision="no")
    model = build_model(cfg)
    assert isinstance(model, VOSRModel)
    assert model.dit_lora_enabled is False
    model.setup(acc)

    assert calls == [], "attach_dit_lora must NOT be called when trainable.dit=False"
    assert model.trainable_components == ("vae",)
    assert set(model.emas.keys()) == {"vae"}
    assert all(not p.requires_grad for p in model.dit.parameters())
    assert model.dit.training is False
    assert not any("lora" in k.lower() for k in model.dit.state_dict())

    init_dit = {k: v.detach().cpu().clone() for k, v in model.dit.state_dict().items()}
    init_vae = {k: v.detach().cpu().clone() for k, v in model.vae.state_dict().items()}

    batch = {"lq": torch.rand(1, 3, 16, 16), "gt": torch.rand(1, 3, 16, 16)}
    _, stepped = model.train_step(batch, global_step=0)
    assert stepped is True

    assert all(
        torch.equal(v.detach().cpu(), init_dit[k]) for k, v in model.dit.state_dict().items()
    )
    assert any(
        not torch.equal(v.detach().cpu(), init_vae[k]) for k, v in model.vae.state_dict().items()
    )
    acc.end_training()


def test_dit_lora_checkpoint_loads_after_attach_for_partial_and_full_ckpts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_accelerate_state,
) -> None:
    torch.manual_seed(23)
    events: list[str] = []

    def _recording_attach(dit: nn.Module, lora_cfg: LoRAConfig) -> nn.Module:
        events.append(f"attach:{lora_cfg.lora_rank}:{lora_cfg.lora_alpha}")
        return _inject_tiny_lora_inplace(dit, lora_cfg)

    monkeypatch.setattr(vosr_mod, "attach_dit_lora", _recording_attach)

    # Build a reference base DiT and a reference LoRA-attached DiT to create checkpoints
    base_cfg = load_train_config(
        _make_vosr_train_dict(trainable={"vae": False, "dit": True, "dino": False}),
        require_data=False,
        require_trainer_full=False,
    )
    acc0 = Accelerator(cpu=True, mixed_precision="no")
    ref_base_model = build_model(base_cfg)
    assert isinstance(ref_base_model, VOSRModel)
    ref_base_model.setup(acc0)
    base_dit_sd = {
        k: torch.full_like(v, 0.25) if v.is_floating_point() else v.clone()
        for k, v in ref_base_model.dit.state_dict().items()
    }
    base_dit_path = save_safetensors(base_dit_sd, tmp_path / "base_dit.safetensors")
    acc0.end_training()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    # Build a LoRA-attached reference DiT with distinct known weights
    lora_ref_cfg = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": False, "dit": True, "dino": False},
            lora={"enable": True, "lora_rank": 4, "lora_alpha": 8, "checkpoint": None},
        ),
        require_data=False,
        require_trainer_full=False,
    )
    acc1 = Accelerator(cpu=True, mixed_precision="no")
    ref_lora_model = build_model(lora_ref_cfg)
    assert isinstance(ref_lora_model, VOSRModel)
    ref_lora_model.setup(acc1)

    lora_only_sd = {
        k: torch.full_like(v, 0.75)
        for k, v in ref_lora_model.dit.state_dict().items()
        if "lora" in k.lower()
    }
    full_lora_sd = {
        k: (torch.full_like(v, 0.9) if "lora" in k.lower() else torch.full_like(v, 0.1))
        if v.is_floating_point()
        else v.clone()
        for k, v in ref_lora_model.dit.state_dict().items()
    }
    lora_only_path = save_safetensors(lora_only_sd, tmp_path / "lora_only.safetensors")
    full_lora_path = save_safetensors(full_lora_sd, tmp_path / "full_lora.safetensors")
    acc1.end_training()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    # Case A: Base pretrain loaded BEFORE attach, LoRA-only checkpoint loaded AFTER attach
    events.clear()
    cfg_partial = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": False, "dit": True, "dino": False},
            pretrain={"dit": {"path": str(base_dit_path), "strict": True, "key": None}},
            lora={
                "enable": True,
                "lora_rank": 4,
                "lora_alpha": 8,
                "checkpoint": str(lora_only_path),
            },
        ),
        require_data=False,
        require_trainer_full=False,
    )
    acc2 = Accelerator(cpu=True, mixed_precision="no")
    model_partial = build_model(cfg_partial)
    assert isinstance(model_partial, VOSRModel)
    model_partial.setup(acc2)
    assert events == ["attach:4:8.0"]
    for k, v in model_partial.dit.state_dict().items():
        if not v.is_floating_point():
            continue
        expected = 0.75 if "lora" in k.lower() else 0.25
        assert torch.allclose(v.detach().cpu(), torch.full_like(v.detach().cpu(), expected)), k
    # EMA cloned after lora.checkpoint load must also hold those weights
    for k, v in model_partial.emas["dit"].module.state_dict().items():
        if not v.is_floating_point():
            continue
        expected = 0.75 if "lora" in k.lower() else 0.25
        assert torch.allclose(v.detach().cpu(), torch.full_like(v.detach().cpu(), expected)), k
    acc2.end_training()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    # Case B: Full checkpoint (base + LoRA) loaded AFTER attach
    cfg_full = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": False, "dit": True, "dino": False},
            lora={
                "enable": True,
                "lora_rank": 4,
                "lora_alpha": 8,
                "checkpoint": str(full_lora_path),
            },
        ),
        require_data=False,
        require_trainer_full=False,
    )
    acc3 = Accelerator(cpu=True, mixed_precision="no")
    model_full = build_model(cfg_full)
    assert isinstance(model_full, VOSRModel)
    model_full.setup(acc3)
    for k, v in model_full.dit.state_dict().items():
        if not v.is_floating_point():
            continue
        expected = 0.9 if "lora" in k.lower() else 0.1
        assert torch.allclose(v.detach().cpu(), torch.full_like(v.detach().cpu(), expected)), k
    acc3.end_training()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    # Case C: Missing lora.checkpoint raises FileNotFoundError
    cfg_missing = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": False, "dit": True, "dino": False},
            lora={
                "enable": True,
                "lora_rank": 4,
                "lora_alpha": 8,
                "checkpoint": str(tmp_path / "does_not_exist.safetensors"),
            },
        ),
        require_data=False,
        require_trainer_full=False,
    )
    acc4 = Accelerator(cpu=True, mixed_precision="no")
    model_missing = build_model(cfg_missing)
    assert isinstance(model_missing, VOSRModel)
    with pytest.raises(FileNotFoundError):
        model_missing.setup(acc4)
    acc4.end_training()


@pytest.mark.parametrize("use_wrapper", [False, True])
def test_dit_lora_inference_exp_dir_and_standalone_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_accelerate_state,
    use_wrapper: bool,
) -> None:
    torch.manual_seed(31)
    attach_calls: list[LoRAConfig] = []

    def _custom_attach(dit: nn.Module, lora_cfg: LoRAConfig) -> nn.Module:
        attach_calls.append(lora_cfg)
        if use_wrapper:
            return _TinyLoRAWrapper(dit, rank=lora_cfg.lora_rank, alpha=lora_cfg.lora_alpha)
        return _inject_tiny_lora_inplace(dit, lora_cfg)

    monkeypatch.setattr(vosr_mod, "attach_dit_lora", _custom_attach)

    # 1. Train 1 step with LoRA and export full weights into an exp_dir
    train_dict = _make_vosr_train_dict(
        trainable={"vae": False, "dit": True, "dino": False},
        lora={"enable": True, "lora_rank": 4, "lora_alpha": 16, "checkpoint": None},
        ema_enabled=True,
    )
    train_cfg = load_train_config(train_dict, require_data=False, require_trainer_full=False)

    exp_dir = tmp_path / "exp_lora"
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(train_cfg, exp_dir / "config_resolved.yaml")

    acc = Accelerator(cpu=True, mixed_precision="no")
    model = build_model(train_cfg)
    assert isinstance(model, VOSRModel)
    model.setup(acc)

    batch = {"lq": torch.rand(1, 3, 16, 16), "gt": torch.rand(1, 3, 16, 16)}
    _, stepped = model.train_step(batch, global_step=0)
    assert stepped is True

    exported = model.export_weights(acc)
    save_safetensors(exported["vae"], weights_dir / "latest_vae.safetensors")
    save_safetensors(exported["dit"], weights_dir / "latest_dit.safetensors")
    save_safetensors(exported["dit_ema"], weights_dir / "latest_dit_ema.safetensors")
    save_safetensors(exported["dino"], weights_dir / "latest_dino.safetensors")

    # Capture reference output from trained EMA pipeline
    inp_img_path = _write_rgb_png(tmp_path / "inputs" / "sample.png", hw=(16, 16), seed=7)
    from utils.image import load_image

    lq_tensor = load_image(inp_img_path).unsqueeze(0)
    infer_pipe = model.get_infer_network(use_ema=True)
    assert isinstance(infer_pipe, VOSRPipeline)
    with torch.no_grad():
        expected_sr = infer_pipe(lq_tensor).clamp(0.0, 1.0)
    acc.end_training()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    # 2. Mode 1 (exp_dir) inference: restores lora.enable=True & trainable.dit=True from config_resolved.yaml
    attach_calls.clear()
    out_exp_dir = tmp_path / "out_exp"
    summary_exp = run_infer(
        {
            "infer": {
                "exp_dir": str(exp_dir),
                "use_ema": True,
                "input": str(inp_img_path),
                "output_dir": str(out_exp_dir),
                "mixed_precision": "no",
                "tile": {"enabled": False},
            }
        }
    )
    assert summary_exp.total_images == 1
    assert len(attach_calls) == 1
    assert attach_calls[0] == LoRAConfig(enable=True, lora_rank=4, lora_alpha=16.0, checkpoint=None)
    exp_out = load_image(out_exp_dir / "sample.png").unsqueeze(0)
    assert torch.allclose(exp_out, expected_sr, atol=1.0 / 255.0 + 1e-4)

    # 3. Mode 2 (standalone) inference with model.lora.enable=True & trainable.dit=True
    attach_calls.clear()
    out_standalone = tmp_path / "out_standalone"
    summary_standalone = run_infer(
        {
            "scale": 1,
            "arch": get_dummy_vosr_arch_dict(with_dino=True),
            "model": {
                "_target_": "models.VOSRModel",
                "scale": 1,
                "trainable": {"vae": False, "dit": True, "dino": False},
                "lora": {"enable": True, "lora_rank": 4, "lora_alpha": 16, "checkpoint": None},
                "eval_seed": 42,
            },
            "infer": {
                "exp_dir": None,
                "component_weights": {
                    "vae": str(weights_dir / "latest_vae.safetensors"),
                    "dit": str(weights_dir / "latest_dit_ema.safetensors"),
                    "dino": str(weights_dir / "latest_dino.safetensors"),
                },
                "input": str(inp_img_path),
                "output_dir": str(out_standalone),
                "mixed_precision": "no",
                "tile": {"enabled": False},
            },
        }
    )
    assert summary_standalone.total_images == 1
    assert len(attach_calls) == 1
    standalone_out = load_image(out_standalone / "sample.png").unsqueeze(0)
    assert torch.equal(standalone_out, exp_out)

    # 4. Mode 2 (standalone) with lora.enable=True but trainable.dit=False -> must NOT attach LoRA
    # Export a clean non-LoRA DiT checkpoint to verify it loads without attaching LoRA.
    attach_calls.clear()
    non_lora_cfg = load_train_config(
        _make_vosr_train_dict(
            trainable={"vae": True, "dit": False, "dino": False},
            lora={"enable": True, "lora_rank": 4, "lora_alpha": 16, "checkpoint": None},
        ),
        require_data=False,
        require_trainer_full=False,
    )
    acc_nl = Accelerator(cpu=True, mixed_precision="no")
    nl_model = build_model(non_lora_cfg)
    assert isinstance(nl_model, VOSRModel)
    nl_model.setup(acc_nl)
    nl_exported = nl_model.export_weights(acc_nl)
    nl_dir = tmp_path / "non_lora_weights"
    nl_vae = save_safetensors(nl_exported["vae"], nl_dir / "vae.safetensors")
    nl_dit = save_safetensors(nl_exported["dit"], nl_dir / "dit.safetensors")
    nl_dino = save_safetensors(nl_exported["dino"], nl_dir / "dino.safetensors")
    acc_nl.end_training()
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    out_frozen = tmp_path / "out_frozen_dit"
    summary_frozen = run_infer(
        {
            "scale": 1,
            "arch": get_dummy_vosr_arch_dict(with_dino=True),
            "model": {
                "_target_": "models.VOSRModel",
                "scale": 1,
                "trainable": {"vae": True, "dit": False, "dino": False},
                "lora": {"enable": True, "lora_rank": 4, "lora_alpha": 16, "checkpoint": None},
                "eval_seed": 42,
            },
            "infer": {
                "exp_dir": None,
                "component_weights": {
                    "vae": str(nl_vae),
                    "dit": str(nl_dit),
                    "dino": str(nl_dino),
                },
                "input": str(inp_img_path),
                "output_dir": str(out_frozen),
                "mixed_precision": "no",
                "tile": {"enabled": False},
            },
        }
    )
    assert summary_frozen.total_images == 1
    assert attach_calls == [], "Inference must NOT attach LoRA when trainable.dit=False"
