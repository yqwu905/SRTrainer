"""Unit and integration tests for VOSR2OneStep, OCRKLLoss, GlyphLoss, and glyph dataset plumbing."""

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
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import save_file

from archs import VOSR2OneStep
from archs.vosr2 import AutoencoderKLQwenImage2D, VOSR2DiT
from data.mixed_dataset import MixedSRDataset, train_collate_fn
from engine import TiledInferencer, Trainer
from losses import GlyphLoss, OCRKLLoss
from losses.glyph_loss import canonical_quad, rectify_text_roi
from losses.ocr_kl_loss import (
    PPOCRv5ServerRec,
    extract_text_strips,
    load_ppocr_recognizer,
)
from models import ModelEMA
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


def _make_tiny_vosr2(**overrides: Any) -> VOSR2OneStep:
    kwargs: dict[str, Any] = dict(
        scale=4,
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
        use_qknorm=True,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_checkpoint=False,
        noise_input="gaussian",
        eval_seed=42,
        vae_pretrained=False,
        vae_base_dim=8,
        vae_dim_mult=(1, 2, 2, 2),
        vae_num_res_blocks=1,
        dino_size=28,
        dino_layer=0,
        dino_pretrained=False,
    )
    kwargs.update(overrides)
    return VOSR2OneStep(**kwargs)


class _TinyRecognizer(nn.Module):
    """Lightweight (N, 3, 16, W) -> (N, W // 4, 7) recognizer for fast CPU OCR-KL unit tests."""

    input_height = 16

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 7, (16, 4), stride=(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).squeeze(2).transpose(1, 2) * 8.0


# ---------------------------------------------------------------------------
# 1. VOSR2 Architecture, Official Key Parity, EMA Sharing & Gradient Flow
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


@pytest.mark.parametrize("rope_mode", ["interpolate", "extrapolate"])
@pytest.mark.parametrize("in_hw", [(8, 8), (12, 16), (7, 9)])
def test_vosr2_onestep_forward_shapes_rectangular_and_deterministic_eval(
    rope_mode: str,
    in_hw: tuple[int, int],
) -> None:
    torch.manual_seed(0)
    model = _make_tiny_vosr2(rope_mode=rope_mode).eval()
    lq = torch.rand(1, 3, *in_hw)

    with torch.no_grad():
        y1 = model(lq)
        y2 = model(lq)

    assert y1.shape == (1, 3, in_hw[0] * 4, in_hw[1] * 4)
    assert torch.isfinite(y1).all()
    assert y1.min().item() >= 0.0 and y1.max().item() <= 1.0
    assert torch.equal(y1, y2)


def test_vosr2_state_dict_exclusivity_ema_backend_sharing_and_tiled_infer(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = _make_tiny_vosr2()

    # state_dict() must contain ONLY DiT weights (never vae.* or dino.*)
    sd = model.state_dict()
    assert len(sd) > 0
    assert not any(k.startswith(("vae.", "dino.", "_backends.")) for k in sd)

    # Strict safetensors save/load round-trip via SRTrainer's load_weights
    w_path = save_safetensors(sd, tmp_path / "vosr2_tiny.safetensors")
    model2 = _make_tiny_vosr2()
    missing, unexpected = load_weights(model2, w_path, strict=True)
    assert missing == [] and unexpected == []

    # ModelEMA shares frozen VAE & DINO backends by reference without copying VRAM,
    # while deep-copying DiT parameters independently.
    ema = ModelEMA(model, decay=0.99, device="cpu")
    assert ema.module._backends is model._backends
    assert ema.module.vae is model.vae
    assert ema.module.dino is model.dino
    assert ema.module.x_embedder.proj.weight is not model.x_embedder.proj.weight

    # Frozen backends remain locked in eval & requires_grad=False even after model.train(True)
    model.train(True)
    assert model.training is True
    assert model.vae.training is False
    assert model.dino is not None and model.dino.training is False
    assert all(not p.requires_grad for p in model.vae.parameters())
    assert all(not p.requires_grad for p in model.dino.parameters())

    # TiledInferencer seamlessly runs VOSR2OneStep
    inferencer = TiledInferencer(
        model=model.eval(),
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


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_vosr2_gradient_flow_through_frozen_vae_decoder_into_all_dit_blocks(
    use_checkpoint: bool,
) -> None:
    torch.manual_seed(7)
    model = _make_tiny_vosr2(
        use_checkpoint=use_checkpoint,
        vae_gradient_checkpointing=use_checkpoint,
        clamp_decode_in_train=False,
    ).train()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    lq = torch.rand(1, 3, 8, 8)
    gt = torch.rand(1, 3, 32, 32)

    # Step 1: DiT zero-initializes final_layer.linear, so step 1 updates final_layer.
    pred1 = model(lq)
    loss1 = F.l1_loss(pred1, gt)
    loss1.backward()
    assert model.final_layer.linear.weight.grad is not None
    assert model.final_layer.linear.weight.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in model.vae.parameters())
    assert model.dino is not None and all(p.grad is None for p in model.dino.parameters())
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Step 2: Now final_layer is non-zero -> gradients must flow through frozen VAE decoder
    # all the way to x_embedder, t_embedder, mlp_ca, blocks[0], and final_layer.
    pred2 = model(lq)
    loss2 = F.l1_loss(pred2, gt)
    loss2.backward()

    for key in (
        "x_embedder.proj.weight",
        "t_embedder.mlp.0.weight",
        "mlp_ca.fc1.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.cross_attn.q_linear.weight",
        "blocks.0.mlp.w12.weight",
        "final_layer.linear.weight",
    ):
        param = dict(model.named_parameters())[key]
        assert param.grad is not None, f"Missing gradient on {key}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {key}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient on {key}"

    assert all(p.grad is None for p in model.vae.parameters())
    assert all(p.grad is None for p in model.dino.parameters())


# ---------------------------------------------------------------------------
# 2. OCR-KL Loss (PPOCRv5ServerRec + OCRKLLoss)
# ---------------------------------------------------------------------------


def test_ppocr_layout_strict_load_and_text_strips(tmp_path: Path) -> None:
    rows = torch.arange(100.0).view(1, 1, 100, 1).expand(2, 3, 100, 5)
    strips = extract_text_strips(rows, 64, 32)
    assert strips.shape == (6, 3, 64, 5)
    assert strips[:, 0, :, 0].unique().tolist() == list(range(100))

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


def test_ocr_kl_loss_rescaling_frozen_eval_and_gradients() -> None:
    torch.manual_seed(0)
    criterion = OCRKLLoss(
        recognizer=_TinyRecognizer(),
        input_height=16,
        strip_height=16,
        strip_stride=8,
        temperature=2.0,
        input_range="0_1",
    )

    # Permanently frozen and eval-locked
    criterion.train(True)
    assert criterion.training is False
    assert criterion.recognizer.training is False
    assert all(not p.requires_grad for p in criterion.parameters())

    target = torch.rand(2, 3, 40, 48)
    assert criterion(target, target).item() == pytest.approx(0.0, abs=1e-6)

    pred = (target + 0.25 * torch.randn_like(target)).clamp(0.0, 1.0).requires_grad_(True)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        val = criterion(pred, target)
    assert val.ndim == 0 and val.item() > 1e-4
    val.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in criterion.parameters())


# ---------------------------------------------------------------------------
# 3. Glyph Loss (TextVerifier Stage-3 + Geometry + Batch Dispatch)
# ---------------------------------------------------------------------------


def test_glyph_loss_exact_score_parity_region_gradients_and_empty_batch() -> None:
    torch.manual_seed(5)
    encoder = nn.Sequential(nn.Conv2d(3, 3, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    prototypes = torch.randn(2, 3, 3)
    loss_fn = GlyphLoss(
        encoder=encoder,
        prototypes=prototypes,
        char_to_index={"中": 0, "文": 1},
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
    # Lower half (rows 8..16) contains only unsupported '?' -> strictly zero gradient!
    assert torch.count_nonzero(pred.grad[:, :, 8:]) == 0
    assert all(p.grad is None for p in loss_fn.parameters())

    # Empty / unsupported glyph batch still produces connected zero loss
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

    # Source 1: inline glyphs in manifest JSON
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

    # Source 2: external VOSR annotate_glyphs.py glyphs.json
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


def test_hydra_configs_vosr2_ocr_kl_and_glyph() -> None:
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["experiment=exp02_vosr2_x4"])

    assert cfg.exp_name == "exp02_vosr2_onestep_text_x4"
    assert cfg.arch._target_ == "archs.VOSR2OneStep"
    assert cfg.arch.hidden_size == 1536
    assert cfg.arch.depth == 36
    assert cfg.arch.num_heads == 24
    assert cfg.arch.input_size == 64
    assert cfg.arch.z_dims == 1024
    assert set(cfg.loss.keys()) == {"l1", "lpips", "ocr_kl", "glyph"}
    assert cfg.loss.ocr_kl.criterion._target_ == "losses.OCRKLLoss"
    assert cfg.loss.glyph.criterion._target_ == "losses.GlyphLoss"
    assert cfg.data.augment.hflip is False
    assert cfg.data.augment.vflip is False
    assert cfg.data.augment.rot90 is False


def test_end_to_end_trainer_vosr2_with_ocr_kl_and_glyph_losses(
    tmp_path: Path,
    reset_accelerate_state,
) -> None:
    """Full Trainer smoke test: VOSR2OneStep + L1 + OCRKLLoss + GlyphLoss + EMA + Val + Save."""
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
                "_target_": "archs.VOSR2OneStep",
                "scale": 4,
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
                "clamp_decode_in_train": False,
                "vae_pretrained": False,
                "vae_base_dim": 8,
                "vae_dim_mult": [1, 2, 2, 2],
                "vae_num_res_blocks": 1,
                "dino_size": 28,
                "dino_layer": 0,
                "dino_pretrained": False,
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
                        "pretrained": False,
                        "num_classes": 16,
                        "strip_height": 16,
                        "strip_stride": 16,
                        "temperature": 1.0,
                    },
                },
                "glyph": {
                    "weight": 1.0,
                    "criterion": {
                        "_target_": "losses.GlyphLoss",
                        "pretrained": False,
                        "batch_size": 4,
                    },
                },
            },
            "model": {
                "_target_": "models.SRModel",
                "ema": {"enabled": True, "decay": 0.99, "warmup_steps": 0},
                "pretrain_path": None,
                "pretrain_strict": True,
                "pretrain_key": None,
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
                "log_with": None,
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
    raw_net = trainer.model.get_infer_network(use_ema=False)
    assert isinstance(raw_net, VOSR2OneStep)
    init_final_w = raw_net.final_layer.linear.weight.detach().clone()
    init_block0_w = raw_net.blocks[0].attn.qkv.weight.detach().clone()
    init_vae_w = raw_net.vae.decoder.conv_out.weight.detach().clone()

    trainer.fit()

    # Step 1 updates final_layer; Step 2 propagates gradients through VAE decoder into blocks[0].
    assert not torch.equal(raw_net.final_layer.linear.weight.detach(), init_final_w)
    assert not torch.equal(raw_net.blocks[0].attn.qkv.weight.detach(), init_block0_w)
    assert torch.equal(raw_net.vae.decoder.conv_out.weight.detach(), init_vae_w)
    assert (work_dir / "weights" / "iter_000002.safetensors").is_file()
    assert (work_dir / "weights" / "iter_000002_ema.safetensors").is_file()
    assert (work_dir / "weights" / "best_val_text_psnr.safetensors").is_file()
