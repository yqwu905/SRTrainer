"""Unit and integration tests for independent VOSR2 archs, VOSRModel, OCRKLLoss, GlyphLoss, and glyph plumbing."""

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
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file

from archs import AutoencoderKLQwenImage2D, DINOv2Encoder, VOSR2DiT
from data.mixed_dataset import MixedSRDataset, train_collate_fn
from engine import TiledInferencer, Trainer
from losses import GlyphLoss, OCRKLLoss
from losses.glyph_loss import canonical_quad, rectify_text_roi
from losses.ocr_kl_loss import (
    PPOCRv5ServerRec,
    load_ppocr_recognizer,
)
from models import ModelEMA, VOSRModel, VOSRPipeline
from tests.dummy_arch import DummyDino, DummyDiT, DummyVAE, get_dummy_vosr_arch_dict
from tests.helpers.fake_backends import (
    TinyRecognizer,
    patch_ppocr_loader,
    patch_textverifier_loader,
    patch_vosr2_backends,
)
from utils.misc import load_weights, save_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


@pytest.fixture
def reset_accelerate_state():
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


def _make_tiny_vosr_components(
    monkeypatch: pytest.MonkeyPatch,
    *,
    out_channels: int = 4,
    z_dims: int | None = 16,
    rope_mode: str = "interpolate",
    use_checkpoint: bool = False,
    vae_gradient_checkpointing: bool = False,
    dino_layer: int = 0,
    dino_num_blocks: int = 2,
) -> tuple[AutoencoderKLQwenImage2D, VOSR2DiT, DINOv2Encoder | None]:
    patch_vosr2_backends(
        monkeypatch,
        z_dim=out_channels,
        dino_dim=z_dims or 16,
        vae_base_dim=8,
        vae_dim_mult=(1, 2, 2, 2),
        vae_num_res_blocks=1,
        dino_num_blocks=dino_num_blocks,
    )
    vae = AutoencoderKLQwenImage2D(
        base_dim=8,
        z_dim=out_channels,
        dim_mult=(1, 2, 2, 2),
        num_res_blocks=1,
        latents_mean=[0.0] * out_channels,
        latents_std=[1.0] * out_channels,
        gradient_checkpointing=vae_gradient_checkpointing,
    )
    dit = VOSR2DiT(
        input_size=8,
        patch_size=2,
        in_channels=out_channels * 2,
        out_channels=out_channels,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        z_dims=z_dims,
        encdim_ratio=2,
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_checkpoint=use_checkpoint,
        rope_mode=rope_mode,
    )
    dino = (
        DINOv2Encoder(
            model_name="dinov2_vitl14",
            image_size=28,
            layer=dino_layer,
            load_hub_weights=False,
        )
        if z_dims is not None
        else None
    )
    return vae, dit, dino


def _make_tiny_vosr_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    scale: int = 4,
    rope_mode: str = "interpolate",
    use_checkpoint: bool = False,
    vae_gradient_checkpointing: bool = False,
    clamp_decode_in_train: bool = True,
    trainable: dict[str, bool] | None = None,
    z_dims: int | None = 16,
    dino_layer: int = 0,
    dino_num_blocks: int = 2,
) -> VOSRPipeline:
    vae, dit, dino = _make_tiny_vosr_components(
        monkeypatch,
        z_dims=z_dims,
        rope_mode=rope_mode,
        use_checkpoint=use_checkpoint,
        vae_gradient_checkpointing=vae_gradient_checkpointing,
        dino_layer=dino_layer,
        dino_num_blocks=dino_num_blocks,
    )
    return VOSRPipeline(
        vae=vae,
        dit=dit,
        dino=dino,
        scale=scale,
        noise_input="gaussian",
        posterior_mode=True,
        clamp_decode_in_train=clamp_decode_in_train,
        eval_seed=42,
        trainable=trainable,
    )


# ---------------------------------------------------------------------------
# 1. Independent VOSR2 Archs, Official Key Parity, VOSRPipeline & Gradient Flow
# ---------------------------------------------------------------------------


def test_vosr2_official_meta_key_counts_and_no_vosr1_r_embedder() -> None:
    with torch.device("meta"):
        dit = VOSR2DiT()
        vae = AutoencoderKLQwenImage2D()

    dit_keys = set(dit.state_dict().keys())
    vae_keys = set(vae.state_dict().keys())

    # Exact VOSR2 1.4B single-step checkpoint parity (no VOSR1 r_embedder)
    assert len(dit_keys) == 849
    assert not any(k.startswith("r_embedder") for k in dit_keys)
    assert dit.state_dict()["x_embedder.proj.weight"].shape == (1536, 32, 2, 2)
    assert dit.state_dict()["final_layer.linear.weight"].shape == (64, 1536)
    assert dit.state_dict()["blocks.0.mlp.w12.weight"].shape == (8192, 1536)
    assert dit.state_dict()["blocks.0.mlp.w3.weight"].shape == (1536, 4096)
    assert dit.state_dict()["feat_rope.freqs_cos"].shape == (1024, 64)

    # Exact Qwen-Image 2D VAE parity (186 tensors, non-persistent mean/std buffers)
    assert len(vae_keys) == 186
    assert "_latents_mean" not in vae_keys
    assert "_latents_std_inv" not in vae_keys


def test_independent_archs_save_load_roundtrip_and_dinov2_raw_vs_wrapped_ckpt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    vae, dit, dino = _make_tiny_vosr_components(monkeypatch, dino_layer=0, dino_num_blocks=2)
    assert dino is not None

    # Independent safetensors round-trip for VAE, DiT, and DINOv2Encoder
    vae_path = save_safetensors(vae.state_dict(), tmp_path / "vae.safetensors")
    dit_path = save_safetensors(dit.state_dict(), tmp_path / "dit.safetensors")
    dino_path = save_safetensors(dino.state_dict(), tmp_path / "dino.safetensors")
    dino_raw_pth = tmp_path / "dinov2_raw_pretrain.pth"
    torch.save(dino.encoder.state_dict(), dino_raw_pth)

    vae2 = AutoencoderKLQwenImage2D(
        base_dim=8,
        z_dim=4,
        dim_mult=(1, 2, 2, 2),
        num_res_blocks=1,
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        pretrained_path=vae_path,
    )
    dit2 = VOSR2DiT(
        input_size=8,
        patch_size=2,
        in_channels=8,
        out_channels=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        z_dims=16,
        encdim_ratio=2,
        pretrained_path=dit_path,
    )
    dino_from_wrapped = DINOv2Encoder(
        image_size=28, layer=0, pretrained_path=dino_path, load_hub_weights=False
    )
    dino_from_raw = DINOv2Encoder(
        image_size=28, layer=0, pretrained_path=dino_raw_pth, load_hub_weights=False
    )

    for k, v in vae.state_dict().items():
        assert torch.equal(vae2.state_dict()[k], v)
    for k, v in dit.state_dict().items():
        assert torch.equal(dit2.state_dict()[k], v)
    for k, v in dino.state_dict().items():
        assert torch.equal(dino_from_wrapped.state_dict()[k], v)
        assert torch.equal(dino_from_raw.state_dict()[k], v)

    # DINOv2Encoder.freeze_unused_parameters freezes post-layer blocks, norm, and mask_token
    dino_from_raw.requires_grad_(True)
    dino_from_raw.freeze_unused_parameters()
    assert all(p.requires_grad for p in dino_from_raw.encoder.blocks[0].parameters())
    assert all(not p.requires_grad for p in dino_from_raw.encoder.blocks[1].parameters())
    assert all(not p.requires_grad for p in dino_from_raw.encoder.norm.parameters())
    assert dino_from_raw.encoder.mask_token.requires_grad is False


@pytest.mark.parametrize("rope_mode", ["interpolate", "extrapolate"])
@pytest.mark.parametrize("in_hw", [(8, 8), (12, 16), (7, 9)])
def test_vosr_pipeline_forward_shapes_rectangular_and_deterministic_eval(
    monkeypatch: pytest.MonkeyPatch,
    rope_mode: str,
    in_hw: tuple[int, int],
) -> None:
    torch.manual_seed(0)
    pipeline = _make_tiny_vosr_pipeline(monkeypatch, rope_mode=rope_mode).eval()
    lq = torch.rand(1, 3, *in_hw)

    with torch.no_grad():
        y1 = pipeline(lq)
        y2 = pipeline(lq)

    assert y1.shape == (1, 3, in_hw[0] * 4, in_hw[1] * 4)
    assert torch.isfinite(y1).all()
    assert y1.min().item() >= 0.0 and y1.max().item() <= 1.0
    assert torch.equal(y1, y2)


def test_vosr_pipeline_tiled_infer_and_replaceable_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1)
    pipeline = _make_tiny_vosr_pipeline(monkeypatch).eval()

    inferencer = TiledInferencer(
        model=pipeline,
        scale=4,
        tile_enabled=True,
        tile_size=8,
        tile_overlap=2,
        tile_batch_size=2,
        blend_mode="cosine",
        mixed_precision="no",
    )
    out = inferencer.infer_tensor(torch.rand(1, 3, 12, 14))
    assert out.shape == (1, 3, 48, 56)
    assert torch.isfinite(out).all()

    # Replaceable arch components: swap Qwen VAE + VOSR2DiT + DINOv2 with completely different modules
    swapped = VOSRPipeline(
        vae=DummyVAE(z_dim=4, spatial_compression_ratio=2),
        dit=DummyDiT(in_channels=8, out_channels=4, patch_size=2, z_dims=8),
        dino=DummyDino(embed_dim=8, image_size=16, patch_size=4),
        scale=2,
    ).eval()
    with torch.no_grad():
        swapped_out = swapped(torch.rand(1, 3, 10, 12))
    assert swapped_out.shape == (1, 3, 20, 24)
    assert torch.isfinite(swapped_out).all()

    # Also supports dino=None when dit.z_dims is None
    no_dino = VOSRPipeline(
        vae=DummyVAE(z_dim=4, spatial_compression_ratio=2),
        dit=DummyDiT(in_channels=8, out_channels=4, patch_size=2, z_dims=None),
        dino=None,
        scale=1,
    ).eval()
    with torch.no_grad():
        no_dino_out = no_dino(torch.rand(1, 3, 11, 13))
    assert no_dino_out.shape == (1, 3, 11, 13)
    assert torch.isfinite(no_dino_out).all()


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_vosr_gradient_flow_through_frozen_vae_decoder_into_all_dit_blocks(
    monkeypatch: pytest.MonkeyPatch,
    use_checkpoint: bool,
) -> None:
    torch.manual_seed(7)
    pipeline = _make_tiny_vosr_pipeline(
        monkeypatch,
        use_checkpoint=use_checkpoint,
        vae_gradient_checkpointing=use_checkpoint,
        clamp_decode_in_train=False,
        trainable={"vae": False, "dit": True, "dino": False},
    )
    pipeline.vae.requires_grad_(False).eval()
    assert pipeline.dino is not None
    pipeline.dino.requires_grad_(False).eval()
    pipeline.dit.requires_grad_(True)
    pipeline.train(True)

    # Frozen components stay eval even after pipeline.train(True)
    assert pipeline.dit.training is True
    assert pipeline.vae.training is False
    assert pipeline.dino.training is False

    opt = torch.optim.AdamW(pipeline.dit.parameters(), lr=1e-2)
    lq = torch.rand(1, 3, 8, 8)
    gt = torch.rand(1, 3, 32, 32)

    # Step 1: DiT zero-initializes final_layer.linear, so step 1 updates final_layer.
    pred1 = pipeline(lq)
    loss1 = F.l1_loss(pred1, gt)
    loss1.backward()
    assert pipeline.dit.final_layer.linear.weight.grad is not None
    assert pipeline.dit.final_layer.linear.weight.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in pipeline.vae.parameters())
    assert all(p.grad is None for p in pipeline.dino.parameters())
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Step 2: Now final_layer is non-zero -> gradients flow through frozen VAE decoder
    # all the way to x_embedder, t_embedder, mlp_ca, blocks[0], and final_layer.
    pred2 = pipeline(lq)
    loss2 = F.l1_loss(pred2, gt)
    loss2.backward()

    dit_params = dict(pipeline.dit.named_parameters())
    for key in (
        "x_embedder.proj.weight",
        "t_embedder.mlp.0.weight",
        "mlp_ca.fc1.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.cross_attn.q_linear.weight",
        "blocks.0.mlp.w12.weight",
        "final_layer.linear.weight",
    ):
        param = dit_params[key]
        assert param.grad is not None, f"Missing gradient on {key}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {key}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient on {key}"

    assert all(p.grad is None for p in pipeline.vae.parameters())
    assert all(p.grad is None for p in pipeline.dino.parameters())


@pytest.mark.parametrize(
    "trainable_cfg",
    [
        {"vae": False, "dit": True, "dino": False},
        {"vae": True, "dit": False, "dino": False},
        {"vae": False, "dit": False, "dino": True},
        {"vae": True, "dit": True, "dino": False},
        {"vae": True, "dit": True, "dino": True},
    ],
)
def test_vosr_model_per_arch_trainability_combinations_ema_and_separate_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_accelerate_state,
    trainable_cfg: dict[str, bool],
) -> None:
    """Verify every subset of trainable archs updates ONLY selected archs, tracks EMA, and exports separately."""
    torch.manual_seed(17)
    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16, dino_num_blocks=2)

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
                "trainable": trainable_cfg,
                "noise_input": "gaussian",
                "posterior_mode": True,
                "clamp_decode_in_train": False,
                "eval_seed": 42,
                "ema": {"enabled": True, "decay": 0.9, "warmup_steps": 0},
                "save_frozen_weights": True,
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

    from config import load_train_config
    from models.base_model import build_model

    acc = Accelerator(cpu=True, mixed_precision="no")
    model = build_model(load_train_config(cfg, require_data=False, require_trainer_full=False))
    model.setup(acc)

    # Non-zero final_layer so step 1 propagates gradients through frozen DiT into VAE encoder / DINO too!
    with torch.no_grad():
        nn.init.normal_(model.dit.final_layer.linear.weight, std=0.1)

    expected_trainable = {k for k, v in trainable_cfg.items() if v}
    assert set(model.emas.keys()) == expected_trainable
    assert set(model.trainable_components) == expected_trainable

    init_snap = {
        comp: {k: v.detach().cpu().clone() for k, v in model.get_component(comp).state_dict().items()}
        for comp in ("vae", "dit", "dino")
    }

    batch = {"lq": torch.rand(1, 3, 16, 16), "gt": torch.rand(1, 3, 16, 16)}
    scalars, stepped = model.train_step(batch, global_step=0)
    assert stepped is True
    assert "loss/total" in scalars and "loss/l1" in scalars

    for comp in ("vae", "dit", "dino"):
        cur_sd = model.get_component(comp).state_dict()
        changed = any(
            not torch.equal(cur_sd[k].detach().cpu(), init_snap[comp][k])
            for k in cur_sd
            if torch.is_floating_point(cur_sd[k])
        )
        if trainable_cfg[comp]:
            assert changed, f"Trainable component {comp!r} did not update!"
            assert model.emas[comp].num_updates == 1
        else:
            assert not changed, f"Frozen component {comp!r} unexpectedly mutated!"
            assert model.get_component(comp).training is False

    # Verify separate per-arch export keys and strict reloadability into each arch
    exported = model.export_weights(acc)
    expected_keys = {"vae", "dit", "dino"} | {f"{c}_ema" for c in expected_trainable}
    assert set(exported.keys()) == expected_keys
    assert len(exported["vae"]) == len(model.vae.state_dict())
    assert len(exported["dit"]) == len(model.dit.state_dict())
    assert len(exported["dino"]) == len(model.dino.state_dict())

    # Verify EMA infer network shares frozen components without copying while using EMA for trainable ones
    infer_ema = model.get_infer_network(use_ema=True)
    assert isinstance(infer_ema, VOSRPipeline)
    for comp in ("vae", "dit", "dino"):
        if trainable_cfg[comp]:
            assert infer_ema.get_component(comp) is model.emas[comp].module
        else:
            assert infer_ema.get_component(comp) is model.get_component(comp)

    # Verify save_frozen_weights=False exports ONLY trainable archs + their EMAs
    model.save_frozen_weights = False
    exported_train_only = model.export_weights(acc)
    assert set(exported_train_only.keys()) == expected_trainable | {
        f"{c}_ema" for c in expected_trainable
    }
    acc.end_training()


def test_vosr_model_invalid_trainability_and_missing_dino_fail_fast(
    reset_accelerate_state,
) -> None:
    base_cfg = {
        "scale": 1,
        "arch": get_dummy_vosr_arch_dict(with_dino=False),
        "loss": {
            "l1": {"weight": 1.0, "criterion": {"_target_": "losses.L1Loss"}},
        },
        "model": {
            "_target_": "models.VOSRModel",
            "scale": 1,
            "trainable": {"vae": False, "dit": False, "dino": False},
            "ema": {"enabled": False},
        },
        "optim": {
            "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1e-3},
            "scheduler": None,
        },
    }

    from config import load_train_config

    # All components frozen -> fail fast
    with pytest.raises(ValueError, match="At least one active VOSR arch"):
        load_train_config(base_cfg, require_data=False, require_trainer_full=False)

    # trainable.dino=True when arch.dino is None -> fail fast
    cfg_bad_dino = OmegaConf.create(base_cfg)
    cfg_bad_dino.model.trainable.dino = True
    with pytest.raises(ValueError, match="trainable.dino"):
        load_train_config(cfg_bad_dino, require_data=False, require_trainer_full=False)

    # Unknown component in trainable -> fail fast (Structured Config rejects unknown keys)
    cfg_unknown = OmegaConf.create(base_cfg)
    cfg_unknown.model.trainable.unknown_arch = True
    with pytest.raises(ValueError, match="unknown_arch"):
        load_train_config(cfg_unknown, require_data=False, require_trainer_full=False)


# ---------------------------------------------------------------------------
# 2. OCR-KL Loss (PPOCRv5ServerRec + OCRKLLoss)
# ---------------------------------------------------------------------------


def test_ppocr_layout_strict_load(tmp_path: Path) -> None:
    model = PPOCRv5ServerRec(num_classes=11).eval()
    keys = [k for k in model.state_dict() if not k.endswith("num_batches_tracked")]
    assert len(keys) == 458

    state = {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "pp_ocrv5_server_rec"}), encoding="utf-8"
    )
    loaded = load_ppocr_recognizer(tmp_path).eval()
    probe = torch.rand(1, 3, 48, 32) * 2.0 - 1.0
    with torch.no_grad():
        assert torch.equal(loaded(probe), model(probe))


def test_ocr_kl_loss_rescaling_frozen_eval_and_skips_unannotated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    calls = 0

    class _CountingRecognizer(TinyRecognizer):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return super().forward(x)

    patch_ppocr_loader(monkeypatch, recognizer=_CountingRecognizer())
    criterion = OCRKLLoss(
        temperature=2.0,
        input_range="0_1",
        strip_batch_size=2,
        amp_dtype="bf16",
        use_checkpoint=True,
        sync_ddp_counts=False,
    )

    criterion.train(True)
    assert criterion.training is False
    assert criterion.recognizer.training is False
    assert all(not p.requires_grad for p in criterion.parameters())

    # Legacy strip-mode kwargs are removed and must raise TypeError
    for removed_kwarg in (
        {"strip_height": 64},
        {"strip_stride": 32},
        {"use_text_lines": False},
        {"fallback_to_strips": True},
    ):
        with pytest.raises(TypeError):
            OCRKLLoss(**removed_kwarg)  # type: ignore[arg-type]

    target = torch.rand(2, 3, 40, 48)
    pred = (target + 0.25 * torch.randn_like(target)).clamp(0.0, 1.0).requires_grad_(True)

    # Unannotated batch (batch=None or no 'text_lines' key) -> gracefully skipped (0 loss, 0 grad)
    val_no_batch = criterion(pred, target)
    assert val_no_batch.item() == 0.0
    assert criterion.last_counts == (0, 0)
    assert calls == 0
    val_no_batch.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0

    pred.grad = None
    val_missing_key = criterion(pred, target, batch={"gt": target})
    assert val_missing_key.item() == 0.0
    assert criterion.last_counts == (0, 0)
    assert calls == 0
    val_missing_key.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0

    # Annotated batch -> exact zero on identical inputs, positive finite loss + grads on mismatch
    pred.grad = None
    lines = [
        [{"bbox": [2, 4, 44, 20]}],
        [{"bbox": [4, 18, 42, 34]}],
    ]
    assert criterion(target, target, text_lines=lines).item() == pytest.approx(0.0, abs=1e-6)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        val = criterion(pred, target, batch={"text_lines": lines})
    assert val.ndim == 0 and val.item() > 1e-4
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in criterion.parameters())


def test_ocr_kl_loss_chunk_and_checkpoint_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(7)
    rec = TinyRecognizer(num_classes=11)
    patch_ppocr_loader(monkeypatch, recognizer=rec)

    target = torch.rand(2, 3, 64, 96)
    pred_base = (target + 0.3 * torch.randn_like(target)).clamp(0.0, 1.0)
    # Same-width boxes across both images so chunk size does not change padding width
    lines = [
        [
            {"bbox": [4, 4, 52, 20]},
            {"bbox": [4, 24, 52, 40]},
            {"bbox": [4, 44, 52, 60]},
        ],
        [
            {"bbox": [10, 8, 58, 24]},
            {"bbox": [10, 32, 58, 48]},
        ],
    ]

    loss_ref = OCRKLLoss(
        temperature=1.5,
        strip_batch_size=16,
        width_bucket_step=16,
        amp_dtype="fp32",
        use_checkpoint=False,
        sync_ddp_counts=False,
    )
    loss_chunked = OCRKLLoss(
        temperature=1.5,
        strip_batch_size=2,
        width_bucket_step=16,
        amp_dtype="fp32",
        use_checkpoint=True,
        sync_ddp_counts=False,
    )

    p1 = pred_base.clone().requires_grad_(True)
    p2 = pred_base.clone().requires_grad_(True)
    v1 = loss_ref(p1, target, batch={"text_lines": lines})
    v2 = loss_chunked(p2, target, batch={"text_lines": lines})
    torch.testing.assert_close(v1, v2)
    v1.backward()
    v2.backward()
    assert p1.grad is not None and p2.grad is not None
    torch.testing.assert_close(p1.grad, p2.grad)


def test_ocr_kl_loss_text_lines_chunked_bf16_vertical_mixed_and_max_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(11)
    observed_batch_sizes: list[int] = []
    observed_autocast: list[tuple[bool, torch.dtype]] = []

    class _SpyRecognizer(TinyRecognizer):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            observed_batch_sizes.append(int(x.shape[0]))
            observed_autocast.append(
                (torch.is_autocast_enabled(x.device.type), torch.get_autocast_dtype(x.device.type))
            )
            return super().forward(x)

    patch_ppocr_loader(monkeypatch, recognizer=_SpyRecognizer(num_classes=9))
    criterion = OCRKLLoss(
        temperature=1.0,
        input_range="0_1",
        strip_batch_size=2,
        max_lines_per_image=3,
        max_strip_width=128,
        width_bucket_step=16,
        amp_dtype="bf16",
        use_checkpoint=True,
        sync_ddp_counts=False,
    )
    assert criterion.needs_batch is True

    target = torch.rand(3, 3, 64, 96)
    pred = (target + 0.25 * torch.randn_like(target)).clamp(0.0, 1.0).requires_grad_(True)

    # Mixed batch:
    # - Image 0 has 4 valid lines (capped to 3 by max_lines_per_image), incl. one vertical line
    # - Image 1 has 1 line in the top-left corner only
    # - Image 2 has NO text_lines (unannotated sample in a mixed batch) -> must get 0 grad
    batch_lines: list[list[dict[str, Any]] | None] = [
        [
            {"transcript": "line0", "bbox": [2, 2, 50, 18]},
            {"transcript": "line1", "bbox": [2, 20, 74, 36]},
            {"transcript": "vert", "bbox": [76, 4, 90, 58]},  # height/width > 1.5 -> auto-rotated
            {"transcript": "line3", "bbox": [2, 40, 42, 56]},
            {"transcript": "###", "bbox": [2, 2, 20, 18]},  # ignored sentinel
            {"transcript": "bad", "bbox": [2, 2, 20, 18], "valid": False},
        ],
        [
            {"transcript": "img1_line", "bbox": [0, 0, 32, 16]},
        ],
        None,
    ]

    val = criterion(pred, target, batch={"text_lines": batch_lines})
    assert criterion.last_counts[0] == 4  # 3 (capped from 4) + 1 + 0
    assert criterion.last_counts[1] > 0
    assert val.ndim == 0 and val.item() > 0.0
    assert observed_batch_sizes and max(observed_batch_sizes) <= 2
    assert all(enabled and dtype == torch.bfloat16 for enabled, dtype in observed_autocast)

    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad[0].abs().sum().item() > 0.0
    assert pred.grad[1, :, :18, :34].abs().sum().item() > 0.0
    # Region far outside image 1's single text box must receive zero gradient.
    assert torch.count_nonzero(pred.grad[1, :, 32:, 48:]) == 0
    # Unannotated image 2 must receive strictly zero gradient.
    assert torch.count_nonzero(pred.grad[2]) == 0

    # Empty text_lines batch -> connected zero loss and zero grad without calling recognizer
    pred.grad = None
    observed_batch_sizes.clear()
    empty_val = criterion(pred, target, batch={"text_lines": [[], [], []]})
    assert criterion.last_counts == (0, 0)
    assert empty_val.item() == 0.0
    assert observed_batch_sizes == []
    empty_val.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0

    # Identical pred & target produces exact zero KL
    assert criterion(target, target, batch={"text_lines": batch_lines}).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_mixed_dataset_preserves_line_quads_without_glyphs_and_clips_boundary_lines(
    tmp_path: Path,
) -> None:
    hr_dir = tmp_path / "hr"
    lr_dir = tmp_path / "lr"
    _write_dummy_rgb_image(hr_dir / "page.png", (64, 64), seed=5)
    _write_dummy_rgb_image(lr_dir / "page.png", (16, 16), seed=6)

    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "gt": "hr/page.png",
                    "lq": "lr/page.png",
                    "text_lines": [
                        {"transcript": "hello", "bbox": [4, 8, 56, 24]},
                        {"transcript": "world", "quad": [[6, 30], [50, 30], [50, 48], [6, 48]]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    ds = MixedSRDataset(
        sources=[
            {"name": "lines_only", "root_dir": str(tmp_path), "json_path": str(manifest), "ratio": 1.0}
        ],
        gt_size=64,
        scale=4,
        augment={"hflip": False, "vflip": False, "rot90": False},
        has_degradation=False,
    )
    assert ds.has_text_annotations is True
    item = ds[0]
    assert len(item["text_lines"]) == 2
    assert "quad" in item["text_lines"][0] and "quad" in item["text_lines"][1]
    assert "glyphs" not in item["text_lines"][0]

    # Verify boundary clipping keeps horizontally truncated line and drops out-of-bounds line
    from data.mixed_dataset import _crop_text_lines

    cropped = _crop_text_lines(
        (
            {"valid": True, "alignment_valid": True, "quad": [[0.0, 10.0], [100.0, 10.0], [100.0, 30.0], [0.0, 30.0]], "glyphs": []},
            {"valid": True, "alignment_valid": True, "quad": [[70.0, 70.0], [95.0, 70.0], [95.0, 90.0], [70.0, 90.0]], "glyphs": []},
        ),
        crop_left=0,
        crop_top=0,
        crop_size=(48, 64),
        orig_h=128,
        orig_w=128,
    )
    assert len(cropped) == 1
    pts = np.asarray(cropped[0]["quad"], dtype=np.float64)
    assert pts[:, 0].min() == pytest.approx(0.0)
    assert pts[:, 0].max() == pytest.approx(64.0)
    assert pts[:, 1].min() == pytest.approx(10.0)
    assert pts[:, 1].max() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 3. Glyph Loss (TextVerifier Stage-3 + Geometry + Batch Dispatch)
# ---------------------------------------------------------------------------


def test_glyph_loss_exact_score_parity_region_gradients_and_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(5)
    encoder = nn.Sequential(nn.Conv2d(3, 3, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    prototypes = torch.randn(2, 3, 3)
    patch_textverifier_loader(
        monkeypatch,
        encoder=encoder,
        prototypes=prototypes,
        char_to_index={"中": 0, "文": 1},
    )
    loss_fn = GlyphLoss(
        batch_size=2,
        input_range="0_1",
        sync_ddp_counts=False,
    )

    assert loss_fn.needs_batch is True
    loss_fn.train(True)
    assert loss_fn.training is False
    assert loss_fn.encoder.training is False
    assert all(not p.requires_grad for p in loss_fn.parameters())

    pred = torch.rand(2, 3, 16, 16, requires_grad=True)
    boxes = [
        {"char": "中", "bbox": [0, 0, 8, 8]},
        {"char": "文", "quad": canonical_quad([8, 0, 16, 8]).tolist()},
        {"char": "?", "bbox": [0, 8, 8, 16]},
    ]
    lines = [[{"glyphs": boxes}], [{"glyphs": [boxes[1]]}]]

    loss = loss_fn(pred, None, batch={"text_lines": lines})
    assert loss_fn.last_counts == (3, 1)

    with torch.no_grad():
        pred_m11 = (pred * 2.0 - 1.0).clamp(-1.0, 1.0)
        crops = torch.stack(
            [
                F.interpolate(
                    rectify_text_roi(pred_m11[0:1], [0, 0, 8, 8], (8, 8)),
                    (128, 128),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0],
                F.interpolate(
                    rectify_text_roi(pred_m11[0:1], [8, 0, 16, 8], (8, 8)),
                    (128, 128),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0],
                F.interpolate(
                    rectify_text_roi(pred_m11[1:2], [8, 0, 16, 8], (8, 8)),
                    (128, 128),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0],
            ]
        )
        feats = F.normalize(encoder(crops), dim=-1)
        selected = F.normalize(prototypes, dim=-1)[[0, 1, 1]]
        expected = (1.0 - (feats[:, None] * selected).sum(-1).max(-1).values).mean()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert torch.count_nonzero(pred.grad[:, :, 8:]) == 0
    assert all(p.grad is None for p in loss_fn.parameters())

    pred.grad = None
    empty = loss_fn(pred, None, text_lines=[[{"glyphs": [boxes[-1]]}], []])
    assert loss_fn.last_counts == (0, 1)
    assert empty.item() == 0.0
    empty.backward()
    assert pred.grad is not None and torch.count_nonzero(pred.grad) == 0


# ---------------------------------------------------------------------------
# 4. Data Pipeline (Inline & External glyphs_json) + End-to-End Trainer Smoke
# ---------------------------------------------------------------------------


def _write_dummy_rgb_image(path: Path, size: tuple[int, int], seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 225, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def test_mixed_dataset_inline_and_external_glyphs_and_collate(tmp_path: Path) -> None:
    hr_dir = tmp_path / "hr"
    lr_dir = tmp_path / "lr"
    _write_dummy_rgb_image(hr_dir / "page01.png", (64, 64), seed=1)
    _write_dummy_rgb_image(lr_dir / "page01.png", (16, 16), seed=2)
    _write_dummy_rgb_image(hr_dir / "page02.png", (64, 64), seed=3)
    _write_dummy_rgb_image(lr_dir / "page02.png", (16, 16), seed=4)

    manifest1 = tmp_path / "m1.json"
    manifest1.write_text(
        json.dumps(
            [
                {
                    "gt": "hr/page01.png",
                    "lq": "lr/page01.png",
                    "glyphs": [
                        {"char": "中", "bbox": [4, 4, 24, 24]},
                        {"char": "文", "bbox": [28, 4, 48, 24]},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    manifest2 = tmp_path / "m2.json"
    manifest2.write_text(
        json.dumps([{"gt": "hr/page02.png", "lq": "lr/page02.png"}]),
        encoding="utf-8",
    )
    glyphs_json = tmp_path / "glyphs.json"
    glyphs_json.write_text(
        json.dumps(
            {
                "page02": [
                    {
                        "transcript": "春",
                        "bbox": [8, 8, 40, 40],
                        "glyphs": [{"char": "春", "bbox": [10, 10, 34, 34]}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    ds = MixedSRDataset(
        sources=[
            {"name": "s1", "root_dir": str(tmp_path), "json_path": str(manifest1), "ratio": 0.5},
            {
                "name": "s2",
                "root_dir": str(tmp_path),
                "json_path": str(manifest2),
                "glyphs_json": str(glyphs_json),
                "ratio": 0.5,
            },
        ],
        gt_size=64,
        scale=4,
        augment={"hflip": False, "vflip": False, "rot90": False},
        has_degradation=False,
    )
    assert ds.has_text_annotations is True
    batch = train_collate_fn([ds[0], ds[1]])
    assert batch["gt"].shape == (2, 3, 64, 64)
    assert batch["lq"].shape == (2, 3, 16, 16)
    assert len(batch["text_lines"]) == 2
    assert [g["char"] for g in batch["text_lines"][0][0]["glyphs"]] == ["中", "文"]
    assert [g["char"] for g in batch["text_lines"][1][0]["glyphs"]] == ["春"]


def test_hydra_configs_vosr2_ocr_kl_and_glyph(monkeypatch: pytest.MonkeyPatch) -> None:
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["experiment=vosr2_onestep_text"])

    assert cfg.exp_name == "vosr2_onestep_text"
    assert cfg.scale == 1
    assert cfg.model._target_ == "models.VOSRModel"
    assert cfg.model.scale == 1
    assert dict(cfg.model.trainable) == {"vae": False, "dit": True, "dino": False}
    assert cfg.arch.vae._target_ == "archs.AutoencoderKLQwenImage2D"
    assert cfg.arch.dit._target_ == "archs.VOSR2DiT"
    assert cfg.arch.dino._target_ == "archs.DINOv2Encoder"
    assert cfg.arch.dit.hidden_size == 1536
    assert cfg.arch.dit.depth == 36
    assert cfg.arch.dit.num_heads == 24
    assert cfg.arch.dit.input_size == 64
    assert cfg.arch.dit.z_dims == 1024
    assert set(cfg.loss.keys()) == {"l1", "lpips", "ocr_kl", "glyph"}
    assert cfg.loss.ocr_kl.criterion._target_ == "losses.OCRKLLoss"
    assert cfg.loss.glyph.criterion._target_ == "losses.GlyphLoss"
    assert "revision" not in cfg.loss.glyph.criterion
    assert "pretrained" not in cfg.loss.glyph.criterion
    assert cfg.data.augment.hflip is False
    assert cfg.data.augment.vflip is False
    assert cfg.data.augment.rot90 is False
    assert cfg.trainer.val_tile.overlap == 64
    assert cfg.trainer.val_tile.pad_multiple == 16

    # Default scale=1 (no upscale) produces same-resolution output
    pipe_x1 = _make_tiny_vosr_pipeline(monkeypatch, scale=1).eval()
    assert pipe_x1.scale == 1
    with torch.no_grad():
        out_x1 = pipe_x1(torch.rand(1, 3, 19, 23))
    assert out_x1.shape == (1, 3, 19, 23)


def test_end_to_end_trainer_vosr2_with_ocr_kl_and_glyph_losses_and_per_arch_saving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reset_accelerate_state,
) -> None:
    """Mocked-backend smoke: VOSRModel + QwenVAE/VOSR2DiT/DINOv2 + L1/OCR-KL/Glyph + per-arch saving."""
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16)
    patch_ppocr_loader(monkeypatch, recognizer=TinyRecognizer(num_classes=16))
    patch_textverifier_loader(monkeypatch)
    hr_dir = tmp_path / "hr"
    lr_dir = tmp_path / "lr"
    for i in range(2):
        _write_dummy_rgb_image(hr_dir / f"img_{i}.png", (32, 32), seed=10 + i)
        _write_dummy_rgb_image(lr_dir / f"img_{i}.png", (8, 8), seed=20 + i)

    train_manifest = tmp_path / "train.json"
    train_manifest.write_text(
        json.dumps(
            [
                {
                    "gt": f"hr/img_{i}.png",
                    "lq": f"lr/img_{i}.png",
                    "text_lines": [
                        {
                            "transcript": "中文",
                            "bbox": [2, 2, 30, 30],
                            "glyphs": [
                                {"char": "中", "bbox": [2, 4, 14, 18]},
                                {"char": "文", "bbox": [16, 4, 28, 18]},
                            ],
                        }
                    ],
                }
                for i in range(2)
            ]
        ),
        encoding="utf-8",
    )
    val_manifest = tmp_path / "val.json"
    val_manifest.write_text(
        json.dumps([{"lq": "lr/img_0.png", "gt": "hr/img_0.png"}]),
        encoding="utf-8",
    )

    work_dir = tmp_path / "exp_vosr2_smoke"
    cfg = OmegaConf.create(
        {
            "exp_name": "vosr2_smoke",
            "scale": 4,
            "seed": 42,
            "work_dir": str(work_dir),
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
                    "input_size": 8,
                    "patch_size": 2,
                    "in_channels": 8,
                    "out_channels": 4,
                    "hidden_size": 32,
                    "depth": 2,
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
            },
            "loss": {
                "l1": {
                    "weight": 1.0,
                    "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
                },
                "ocr_kl": {
                    "weight": 0.1,
                    "criterion": {
                        "_target_": "losses.OCRKLLoss",
                        "temperature": 1.0,
                        "strip_batch_size": 2,
                        "amp_dtype": "bf16",
                    },
                },
                "glyph": {
                    "weight": 1.0,
                    "criterion": {
                        "_target_": "losses.GlyphLoss",
                        "batch_size": 4,
                    },
                },
            },
            "model": {
                "_target_": "models.VOSRModel",
                "scale": 4,
                "trainable": {"vae": False, "dit": True, "dino": False},
                "noise_input": "gaussian",
                "posterior_mode": True,
                "clamp_decode_in_train": False,
                "eval_seed": 42,
                "ema": {"enabled": True, "decay": 0.99, "warmup_steps": 0},
                "save_frozen_weights": True,
            },
            "data": {
                "gt_size": 32,
                "scale": 4,
                "batch_size_per_gpu": 2,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "text_crop_prob": 0.5,
                "sources": [
                    {
                        "name": "paired_text",
                        "root_dir": str(tmp_path),
                        "json_path": str(train_manifest),
                        "ratio": 1.0,
                    }
                ],
                "augment": {"hflip": False, "vflip": False, "rot90": False},
                "degradation": None,
            },
            "optim": {
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "_partial_": True,
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                },
                "scheduler": None,
            },
            "metric": {
                "primary_dataset": "val_text",
                "primary_metric": "psnr",
                "val_suites": {
                    "val_text": {
                        "dataset": {
                            "root_dir": str(tmp_path),
                            "json_path": str(val_manifest),
                        },
                        "metrics": {
                            "psnr": {
                                "_target_": "metrics.PSNRMetric",
                                "crop_border": 0,
                                "test_y_channel": False,
                            }
                        },
                    }
                },
            },
            "trainer": {
                "total_iters": 2,
                "val_interval": 2,
                "save_interval": 2,
                "log_interval": 1,
                "max_keep_states": 1,
                "mixed_precision": "no",
                "gradient_accumulation_steps": 1,
                "grad_clip_norm": 1.0,
                "compile": False,
                "log_with": [],
                "resume": None,
                "append_timestamp": False,
                "val_save_visuals": False,
                "val_max_save_images": 0,
                "val_use_ema": True,
                "val_tile": {"enabled": False, "pad_multiple": 4},
            },
        }
    )

    trainer = Trainer(cfg)
    assert isinstance(trainer.model, VOSRModel)
    raw_pipe = trainer.model.get_infer_network(use_ema=False)
    assert isinstance(raw_pipe, VOSRPipeline)
    assert isinstance(raw_pipe.vae, AutoencoderKLQwenImage2D)
    assert isinstance(raw_pipe.dit, VOSR2DiT)
    assert isinstance(raw_pipe.dino, DINOv2Encoder)

    init_final_w = raw_pipe.dit.final_layer.linear.weight.detach().clone()
    init_block0_w = raw_pipe.dit.blocks[0].attn.qkv.weight.detach().clone()
    init_vae_w = raw_pipe.vae.decoder.conv_out.weight.detach().clone()
    init_dino_w = raw_pipe.dino.encoder.proj.weight.detach().clone()

    trainer.fit()

    # Step 1 updates final_layer; Step 2 propagates gradients through VAE decoder into blocks[0].
    assert not torch.equal(raw_pipe.dit.final_layer.linear.weight.detach(), init_final_w)
    assert not torch.equal(raw_pipe.dit.blocks[0].attn.qkv.weight.detach(), init_block0_w)
    assert torch.equal(raw_pipe.vae.decoder.conv_out.weight.detach(), init_vae_w)
    assert torch.equal(raw_pipe.dino.encoder.proj.weight.detach(), init_dino_w)

    # Every arch saved separately (iter, latest, ema for trainable dit, and best)
    weights_dir = work_dir / "weights"
    for fname in (
        "iter_000002_vae.safetensors",
        "iter_000002_dit.safetensors",
        "iter_000002_dit_ema.safetensors",
        "iter_000002_dino.safetensors",
        "latest_vae.safetensors",
        "latest_dit.safetensors",
        "latest_dit_ema.safetensors",
        "latest_dino.safetensors",
        "best_val_text_psnr_vae.safetensors",
        "best_val_text_psnr_dit.safetensors",
        "best_val_text_psnr_dino.safetensors",
    ):
        assert (weights_dir / fname).is_file(), f"Missing expected per-arch weight file: {fname}"

    # Verify best_val_text_psnr_dit equals latest_dit_ema (val_use_ema=True) and loads cleanly
    best_dit_sd = load_safetensors_file(str(weights_dir / "best_val_text_psnr_dit.safetensors"))
    ema_dit_sd = load_safetensors_file(str(weights_dir / "latest_dit_ema.safetensors"))
    assert all(torch.equal(best_dit_sd[k], ema_dit_sd[k]) for k in ema_dit_sd)

    vae_reloaded, dit_reloaded, dino_reloaded = _make_tiny_vosr_components(monkeypatch)
    assert dino_reloaded is not None
    assert load_weights(vae_reloaded, weights_dir / "latest_vae.safetensors", strict=True) == ([], [])
    assert load_weights(dit_reloaded, weights_dir / "latest_dit.safetensors", strict=True) == ([], [])
    assert load_weights(dino_reloaded, weights_dir / "latest_dino.safetensors", strict=True) == ([], [])

    # Verify ModelEMA unit check still works cleanly on standalone VOSR2DiT
    ema = ModelEMA(dit_reloaded, decay=0.99, device="cpu")
    assert ema.module.x_embedder.proj.weight is not dit_reloaded.x_embedder.proj.weight
