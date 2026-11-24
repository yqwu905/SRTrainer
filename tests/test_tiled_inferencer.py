"""Unit, integration, and 2-process CPU distributed tests for M4 (TiledInferencer & tools/infer.py)."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import hydra
import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn
from torch.nn import functional as F

from engine import inferencer as inferencer_mod
from engine.inferencer import (
    TiledInferencer,
    build_blend_window,
    compute_tile_starts,
)
from tests.dummy_arch import DummySRNet
from utils.config import save_resolved_config
from utils.image import load_image, save_image
from utils.misc import save_safetensors

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())
INFER_SCRIPT = REPO_ROOT / "tools" / "infer.py"

_INFER_MODULE_CACHE: ModuleType | None = None


def _load_infer_module() -> ModuleType:
    global _INFER_MODULE_CACHE
    if _INFER_MODULE_CACHE is None:
        spec = importlib.util.spec_from_file_location("srtrainer_tools_infer", INFER_SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _INFER_MODULE_CACHE = module
    return _INFER_MODULE_CACHE


def _compose_infer(overrides: list[str]) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return hydra.compose(config_name="infer", overrides=overrides)


def _reset_srtrainer_loggers() -> None:
    root_sr = logging.getLogger("srtrainer")
    for handler in list(root_sr.handlers):
        root_sr.removeHandler(handler)
        handler.close()
    root_sr.disabled = False
    root_sr.propagate = True
    root_sr.setLevel(logging.NOTSET)

    for name, obj in list(logging.root.manager.loggerDict.items()):
        if name.startswith("srtrainer.") and isinstance(obj, logging.Logger):
            obj.disabled = False


@pytest.fixture(autouse=True)
def reset_accelerate_state():
    """Ensure Accelerate singletons and srtrainer loggers are clean before and after every test."""
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    _reset_srtrainer_loggers()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    _reset_srtrainer_loggers()


# ---------------------------------------------------------------------------
# Lightweight deterministic dummy networks & test data factories
# ---------------------------------------------------------------------------


class ConstantUpsampler(nn.Module):
    def __init__(self, scale: int, value: float = 0.73) -> None:
        super().__init__()
        self.scale = scale
        self.value = value
        self.observed_training: list[bool] = []
        self.observed_grad_enabled: list[bool] = []
        self.observed_dtypes: list[torch.dtype] = []
        self.observed_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observed_training.append(self.training)
        self.observed_grad_enabled.append(torch.is_grad_enabled())
        self.observed_dtypes.append(x.dtype)
        self.observed_shapes.append(tuple(x.shape))
        b, c, h, w = x.shape
        return torch.full(
            (b, c, h * self.scale, w * self.scale),
            self.value,
            dtype=x.dtype,
            device=x.device,
        )


class NearestUpsampler(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.scale, mode="nearest")


class OutOfRangeUpsampler(nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return up * 5.0 - 2.0


class FailingUpsampler(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("intentional forward failure")


def _make_gradient_tensor(h: int, w: int, batch: int = 1) -> torch.Tensor:
    y = torch.linspace(0.05, 0.95, steps=h, dtype=torch.float32)[:, None].expand(h, w)
    x = torch.linspace(0.10, 0.90, steps=w, dtype=torch.float32)[None, :].expand(h, w)
    z = 0.5 * (y + x)
    img = torch.stack([y, x, z], dim=0)
    if batch == 1:
        return img.unsqueeze(0).contiguous()
    return torch.stack(
        [(img + 0.03 * b).clamp(0.0, 1.0) for b in range(batch)],
        dim=0,
    ).contiguous()


def _make_rgb_image(
    path: Path,
    h: int,
    w: int,
    rgb: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    if rgb is None:
        tensor = _make_gradient_tensor(h, w, batch=1).squeeze(0)
    else:
        tensor = torch.empty((3, h, w), dtype=torch.float32)
        tensor[0].fill_(rgb[0])
        tensor[1].fill_(rgb[1])
        tensor[2].fill_(rgb[2])
    save_image(tensor, path)
    return tensor


def _make_constant_toy_state_dict(
    scale: int,
    fill_value: float,
    num_feat: int = 8,
    num_block: int = 1,
) -> dict[str, torch.Tensor]:
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=num_feat, num_block=num_block, scale=scale)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        assert model.conv_last.bias is not None
        model.conv_last.bias.fill_(fill_value)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _make_exp_dir(
    exp_dir: Path,
    *,
    scale: int = 2,
    gt_size: int | list[int] | None = 32,
    num_feat: int = 8,
    num_block: int = 1,
    primary_dataset: str | None = "set5",
    primary_metric: str | None = "psnr",
    val_use_ema: bool | None = True,
    weights_spec: dict[str, float] | None = None,
    write_config: bool = True,
    create_weights_dir: bool = True,
) -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    if write_config:
        cfg_dict: dict[str, Any] = {
            "scale": scale,
            "arch": {
                "_target_": "tests.dummy_arch.DummySRNet",
                "num_in_ch": 3,
                "num_out_ch": 3,
                "num_feat": num_feat,
                "num_block": num_block,
                "scale": scale,
            },
            "data": {"gt_size": gt_size} if gt_size is not None else {},
            "metric": {
                "primary_dataset": primary_dataset,
                "primary_metric": primary_metric,
            },
            "trainer": {} if val_use_ema is None else {"val_use_ema": val_use_ema},
        }
        save_resolved_config(OmegaConf.create(cfg_dict), exp_dir / "config_resolved.yaml")

    if create_weights_dir:
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        for filename, fill_value in (weights_spec or {}).items():
            sd = _make_constant_toy_state_dict(
                scale=scale,
                fill_value=fill_value,
                num_feat=num_feat,
                num_block=num_block,
            )
            save_safetensors(sd, weights_dir / filename)

    return exp_dir


# ---------------------------------------------------------------------------
# Part A: TiledInferencer core math, feathering, edge sizes & guards
# ---------------------------------------------------------------------------


def test_tiled_inferencer_tile_size_modes_and_validation(caplog: pytest.LogCaptureFixture) -> None:
    model = NearestUpsampler(scale=4)

    # tile_size="auto" with gt_size -> gt_size // scale
    inf_auto = TiledInferencer(model=model, scale=4, tile_size="auto", tile_overlap=4, gt_size=64)
    assert inf_auto.tile_enabled is True
    assert inf_auto.tile_size == (16, 16)
    assert inf_auto.weight_window is not None
    assert tuple(inf_auto.weight_window.shape) == (1, 1, 64, 64)

    rectangular = TiledInferencer(
        model=model, scale=4, tile_size="auto", tile_overlap=4, gt_size=[48, 80]
    )
    assert rectangular.tile_size == (12, 20)
    image = torch.rand(1, 3, 29, 43)
    torch.testing.assert_close(rectangular.infer_tensor(image), model(image))

    # tile_size="auto" without gt_size -> warning + untiled fallback
    with caplog.at_level(logging.WARNING, logger="srtrainer"):
        inf_fallback = TiledInferencer(
            model=model,
            scale=4,
            tile_size="auto",
            tile_overlap=16,
            gt_size=None,
            pad_multiple=8,
        )
    assert inf_fallback.tile_enabled is False
    assert inf_fallback.tile_size is None
    assert inf_fallback.weight_window is None
    assert any("auto" in rec.getMessage() and "gt_size" in rec.getMessage() for rec in caplog.records)

    # tile_enabled=False ignores default overlap/auto without warning
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="srtrainer"):
        inf_disabled = TiledInferencer(
            model=model,
            scale=4,
            tile_enabled=False,
            tile_size="auto",
            tile_overlap=16,
            gt_size=None,
        )
    assert inf_disabled.tile_enabled is False
    assert inf_disabled.tile_size is None
    assert inf_disabled.weight_window is None
    assert len(caplog.records) == 0

    # gt_size not divisible by scale
    with pytest.raises(ValueError, match="divisible"):
        TiledInferencer(model=model, scale=4, tile_size="auto", tile_overlap=4, gt_size=65)

    # int, list, and OmegaConf ListConfig
    inf_int = TiledInferencer(model=model, scale=4, tile_size=32, tile_overlap=8)
    assert inf_int.tile_size == (32, 32)

    inf_seq = TiledInferencer(model=model, scale=4, tile_size=[24, 32], tile_overlap=6)
    assert inf_seq.tile_size == (24, 32)

    inf_lc = TiledInferencer(model=model, scale=4, tile_size=ListConfig([24, 32]), tile_overlap=6)
    assert inf_lc.tile_size == (24, 32)

    # Type & value validation
    with pytest.raises(TypeError):
        TiledInferencer(model="not_a_module", scale=4)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TiledInferencer(model=model, scale=4, tile_enabled=False, device=123)  # type: ignore[arg-type]

    for bad_size in (0, -8, True, False, 16.5, [16], [16, 0], [True, 16], "bad", b"16", None):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=4, tile_size=bad_size, tile_overlap=0)  # type: ignore[arg-type]

    for bad_scale in (0, -2, True, 2.5):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=bad_scale, tile_size=16, tile_overlap=0)  # type: ignore[arg-type]

    for bad_enabled in (1, "true", None):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=4, tile_enabled=bad_enabled)  # type: ignore[arg-type]

    for bad_bs in (0, -1, True):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=4, tile_size=16, tile_overlap=0, tile_batch_size=bad_bs)  # type: ignore[arg-type]

    for bad_pm in (0, -4, True):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=4, tile_size=16, tile_overlap=0, pad_multiple=bad_pm)  # type: ignore[arg-type]

    for bad_mode in ("gaussian", "", 123):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=4, tile_size=16, tile_overlap=0, blend_mode=bad_mode)  # type: ignore[arg-type]

    for bad_gt in (0, -16, True, [64], [64, 0], [65, 64], [64, 65], [64, True]):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=4, tile_size=16, tile_overlap=0, gt_size=bad_gt)  # type: ignore[arg-type]

    # from_config with Hydra DictConfig (infer.tile and trainer.val_tile), plain dict, and None
    infer_cfg = _compose_infer([])
    inf_from_infer = TiledInferencer.from_config(model=model, scale=4, tile_cfg=infer_cfg.infer.tile, gt_size=256)
    assert inf_from_infer.tile_enabled is True
    assert inf_from_infer.tile_size == (64, 64)
    assert inf_from_infer.tile_overlap == 16
    assert inf_from_infer.blend_mode == "cosine"

    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        train_cfg = hydra.compose(
            config_name="train",
            overrides=[
                "~experiment", "arch/vae=qwen_vae2d", "arch/dit=vosr2_dit", "loss=l1",
                "scale=4", "data.gt_size=[256,512]", "trainer.val_tile.overlap=16",
            ],
        )
    inf_from_train = TiledInferencer.from_config(
        model=model,
        scale=4,
        tile_cfg=train_cfg.trainer.val_tile,
        gt_size=train_cfg.data.gt_size,
    )
    assert inf_from_train.tile_size == (64, 128)

    inf_from_dict = TiledInferencer.from_config(
        model=model,
        scale=4,
        tile_cfg={"enabled": True, "size": [16, 20], "overlap": 4, "batch_size": 2, "blend_mode": "linear"},
    )
    assert inf_from_dict.tile_size == (16, 20)
    assert inf_from_dict.blend_mode == "linear"

    inf_from_none = TiledInferencer.from_config(model=model, scale=4, tile_cfg=None, gt_size=256)
    assert inf_from_none.tile_size == (64, 64)

    with pytest.raises(TypeError):
        TiledInferencer.from_config(model=model, scale=4, tile_cfg=[16, 16])  # type: ignore[arg-type]


def test_tiled_inferencer_overlap_validation_and_zero_overlap() -> None:
    model = NearestUpsampler(scale=2)

    # Invalid overlaps
    for size, overlap in (
        (32, -1),
        (32, True),
        (32, 16),
        ([20, 40], 10),
        (17, 8),
        (1, 1),
    ):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=2, tile_size=size, tile_overlap=overlap)  # type: ignore[arg-type]

    # Max valid boundary and tiny overlap=0 boundary
    inf_max = TiledInferencer(model=model, scale=2, tile_size=32, tile_overlap=15)
    assert inf_max.tile_overlap == 15
    inf_tiny = TiledInferencer(model=model, scale=2, tile_size=1, tile_overlap=0)
    assert inf_tiny.tile_size == (1, 1)

    # Sliding window coordinates
    assert compute_tile_starts(32, 64, 8) == [0]
    assert compute_tile_starts(64, 64, 8) == [0]
    assert compute_tile_starts(112, 64, 16) == [0, 48]
    assert compute_tile_starts(100, 64, 0) == [0, 36]
    assert compute_tile_starts(65, 64, 16) == [0, 1]

    # overlap=0 hard stitch on non-divisible 100x100
    lq = _make_gradient_tensor(100, 100, batch=1)
    inf_zero = TiledInferencer(model=model, scale=2, tile_size=64, tile_overlap=0, blend_mode="cosine")
    out = inf_zero.infer_tensor(lq)
    ref = model(lq)
    assert tuple(out.shape) == (1, 3, 200, 200)
    assert torch.isfinite(out).all()
    assert (out - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize("blend_mode", ["cosine", "linear", "mean"])
@pytest.mark.parametrize("scale", [2, 4])
def test_tiled_inferencer_feathering_seamless_all_blend_modes(blend_mode: str, scale: int) -> None:
    # 1. Window invariants (including ramp=1 edge safety)
    win = build_blend_window(tile_h=32, tile_w=24, scale=scale, overlap=6, blend_mode=blend_mode)
    assert tuple(win.shape) == (1, 1, 32 * scale, 24 * scale)
    assert win.dtype == torch.float32
    assert win.is_contiguous()
    assert torch.isfinite(win).all()
    assert win.min().item() >= 1e-6 - 1e-9
    assert torch.allclose(win, win.flip(-1), atol=1e-6)
    assert torch.allclose(win, win.flip(-2), atol=1e-6)

    ramp = 6 * scale
    interior = win[0, 0, ramp:-ramp, ramp:-ramp]
    assert torch.allclose(interior, torch.ones_like(interior), atol=1e-6)
    if blend_mode in {"cosine", "linear"}:
        # 1D opposing ramp sum is strictly 1 + eps
        w_row = win[0, 0, ramp, :]
        assert abs(w_row[0].item() - 1e-3) < 1e-6
        assert torch.allclose(w_row[:ramp] + w_row[-ramp:], torch.full((ramp,), 1.001), atol=1e-5)

    win_ramp1 = build_blend_window(tile_h=4, tile_w=4, scale=1, overlap=1, blend_mode=blend_mode)
    assert torch.isfinite(win_ramp1).all()
    assert win_ramp1.min().item() > 0.0

    # 2. Constant upsampler seamlessness (< 1e-5)
    lq = _make_gradient_tensor(50, 70, batch=1)
    const_model = ConstantUpsampler(scale=scale, value=0.73)
    inf_const = TiledInferencer(
        model=const_model,
        scale=scale,
        tile_size=32,
        tile_overlap=8,
        blend_mode=blend_mode,
        mixed_precision="no",
    )
    out_const = inf_const.infer_tensor(lq)
    assert tuple(out_const.shape) == (1, 3, 50 * scale, 70 * scale)
    assert (out_const - 0.73).abs().max().item() < 1e-5

    # 3. Shift-invariant upsampler seamlessness on square & rectangular tiles (< 1e-5)
    nn_model = NearestUpsampler(scale=scale)
    ref = nn_model(lq)
    for tile_size, overlap in ((32, 8), ([24, 32], 6)):
        inf_nn = TiledInferencer(
            model=nn_model,
            scale=scale,
            tile_size=tile_size,
            tile_overlap=overlap,
            blend_mode=blend_mode,
            mixed_precision="no",
        )
        out_nn = inf_nn.infer_tensor(lq)
        assert tuple(out_nn.shape) == tuple(ref.shape)
        assert (out_nn - ref).abs().max().item() < 1e-5


def test_tiled_inferencer_batch_size_equivalence_and_chunking() -> None:
    torch.manual_seed(42)
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    lq = _make_gradient_tensor(55, 67, batch=1)

    outs = {}
    for bs in (1, 4, 16):
        inf = TiledInferencer(
            model=model,
            scale=2,
            tile_size=24,
            tile_overlap=6,
            tile_batch_size=bs,
            blend_mode="cosine",
            mixed_precision="no",
        )
        outs[bs] = inf.infer_tensor(lq)

    assert torch.allclose(outs[1], outs[4], atol=1e-5)
    assert torch.allclose(outs[1], outs[16], atol=1e-5)

    # Hook verifies exact tile batch chunking:
    # 55x67 with tile=24, overlap=6 -> 3x4 = 12 tiles -> [4, 4, 4]
    # 55x57 with tile=24, overlap=6 -> 3x3 = 9 tiles -> [4, 4, 1] (B=1) and [4, 4, 4, 4, 2] (B=2)
    observed_batches: list[int] = []
    handle = model.register_forward_pre_hook(
        lambda _m, args: observed_batches.append(int(args[0].shape[0]))
    )
    try:
        inf4 = TiledInferencer(model=model, scale=2, tile_size=24, tile_overlap=6, tile_batch_size=4)
        inf4.infer_tensor(lq)
        assert observed_batches == [4, 4, 4]

        observed_batches.clear()
        lq_9 = _make_gradient_tensor(55, 57, batch=1)
        inf4.infer_tensor(lq_9)
        assert observed_batches == [4, 4, 1]

        observed_batches.clear()
        lq_b2 = _make_gradient_tensor(55, 57, batch=2)
        inf4.infer_tensor(lq_b2)
        assert observed_batches == [4, 4, 4, 4, 2]
        assert all(n <= 4 for n in observed_batches)
    finally:
        handle.remove()


def test_tiled_inferencer_edge_sizes_small_strip_odd_and_untiled() -> None:
    model = NearestUpsampler(scale=2)

    # Tiny images (pad >= dim -> reflect falls back to replicate), strips, and prime/odd dims
    cases = [
        ((3, 5), 32, 4),
        ((1, 1), 32, 4),
        ((32, 150), 64, 8),
        ((150, 32), 64, 8),
        ((37, 53), 24, 4),
    ]
    for (h, w), tile_size, overlap in cases:
        lq = _make_gradient_tensor(h, w, batch=1)
        inf = TiledInferencer(model=model, scale=2, tile_size=tile_size, tile_overlap=overlap)
        out = inf.infer_tensor(lq)
        assert tuple(out.shape) == (1, 3, h * 2, w * 2)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert out.min().item() >= 0.0 and out.max().item() <= 1.0
        # Exact content match for shift-invariant nearest upsampler inside unpadded region
        assert (out - model(lq)).abs().max().item() < 1e-5

    # Untiled mode with pad_multiple=8 on normal (13x19) and tiny replicate-fallback (3x5)
    spy = ConstantUpsampler(scale=2, value=0.5)
    inf_untiled = TiledInferencer(model=spy, scale=2, tile_enabled=False, pad_multiple=8)
    for (h, w), expected_pad in (((13, 19), (16, 24)), ((3, 5), (8, 8))):
        spy.observed_shapes.clear()
        lq = _make_gradient_tensor(h, w, batch=1)
        out = inf_untiled.infer_tensor(lq)
        assert spy.observed_shapes == [(1, 3, expected_pad[0], expected_pad[1])]
        assert tuple(out.shape) == (1, 3, h * 2, w * 2)


def test_tiled_inferencer_shapes_clamp_training_preservation_and_cpu_mp(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nn_model = NearestUpsampler(scale=2)
    inf = TiledInferencer(model=nn_model, scale=2, tile_size=16, tile_overlap=4)

    # [3, H, W], [1, 3, H, W], [2, 3, H, W] batch-rank preservation
    lq_3d = _make_gradient_tensor(25, 29, batch=1).squeeze(0)
    out_3d = inf.infer_tensor(lq_3d)
    assert tuple(out_3d.shape) == (3, 50, 58)
    assert out_3d.dtype == torch.float32 and out_3d.is_contiguous()

    out_4d_1 = inf.infer_tensor(lq_3d.unsqueeze(0))
    assert tuple(out_4d_1.shape) == (1, 3, 50, 58)
    assert torch.allclose(out_3d, out_4d_1.squeeze(0), atol=1e-6)

    # B > 1 pixel accuracy verifying no shared-weight_acc bug
    b0 = torch.full((3, 25, 29), 0.2, dtype=torch.float32)
    b1 = torch.full((3, 25, 29), 0.8, dtype=torch.float32)
    out_4d_2 = inf.infer_tensor(torch.stack([b0, b1], dim=0))
    assert tuple(out_4d_2.shape) == (2, 3, 50, 58)
    assert (out_4d_2[0] - 0.2).abs().max().item() < 1e-5
    assert (out_4d_2[1] - 0.8).abs().max().item() < 1e-5

    # Invalid input types & shapes
    with pytest.raises(TypeError):
        inf.infer_tensor("not_a_tensor")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        inf.infer_tensor(torch.zeros((1, 3, 16, 16), dtype=torch.uint8))
    for bad_shape in ((16, 16), (1, 1, 3, 16, 16), (0, 3, 16, 16), (1, 3, 0, 16)):
        with pytest.raises(ValueError):
            inf.infer_tensor(torch.zeros(bad_shape, dtype=torch.float32))

    # Wrong channel count rejected by DummySRNet
    toy = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    inf_toy = TiledInferencer(model=toy, scale=2, tile_size=16, tile_overlap=4)
    with pytest.raises(ValueError, match="channels"):
        inf_toy.infer_tensor(torch.zeros((1, 4, 16, 16), dtype=torch.float32))

    # Strict [0, 1] clamping on out-of-range outputs
    inf_clamp = TiledInferencer(model=OutOfRangeUpsampler(scale=2), scale=2, tile_size=16, tile_overlap=4)
    clamped = inf_clamp.infer_tensor(_make_gradient_tensor(20, 20, batch=1))
    assert clamped.min().item() == 0.0
    assert clamped.max().item() == 1.0

    # model.training preservation on normal path & exception path
    spy = ConstantUpsampler(scale=2, value=0.4)
    spy.train(True)
    inf_spy = TiledInferencer(model=spy, scale=2, tile_size=16, tile_overlap=4)
    _ = inf_spy.infer_tensor(_make_gradient_tensor(20, 20, batch=1))
    assert spy.observed_training and all(flag is False for flag in spy.observed_training)
    assert spy.observed_grad_enabled and all(flag is False for flag in spy.observed_grad_enabled)
    assert spy.training is True

    spy.eval()
    _ = inf_spy.infer_tensor(_make_gradient_tensor(20, 20, batch=1))
    assert spy.training is False

    failing = FailingUpsampler()
    failing.train(True)
    inf_fail = TiledInferencer(model=failing, scale=2, tile_size=16, tile_overlap=4)
    with pytest.raises(RuntimeError, match="intentional forward failure"):
        inf_fail.infer_tensor(_make_gradient_tensor(20, 20, batch=1))
    assert failing.training is True

    # Mismatched model output spatial scale raises RuntimeError in both tiled and untiled modes
    wrong_scale_model = NearestUpsampler(scale=4)
    with pytest.raises(RuntimeError, match="Expected model tile output shape"):
        TiledInferencer(model=wrong_scale_model, scale=2, tile_size=16, tile_overlap=4).infer_tensor(
            _make_gradient_tensor(20, 20, batch=1)
        )
    with pytest.raises(RuntimeError, match="Expected model output shape"):
        TiledInferencer(model=wrong_scale_model, scale=2, tile_enabled=False).infer_tensor(
            _make_gradient_tensor(20, 20, batch=1)
        )

    # ToyModel integration with CPU mixed precision modes (including fp16 safe disable)
    lq_mp = _make_gradient_tensor(20, 24, batch=1)
    for mp in ("no", "bf16", "fp16", False, None):
        inf_mp = TiledInferencer(
            model=toy,
            scale=2,
            tile_size=16,
            tile_overlap=4,
            mixed_precision=mp,  # type: ignore[arg-type]
            device="cpu",
        )
        out_mp = inf_mp.infer_tensor(lq_mp)
        assert tuple(out_mp.shape) == (1, 3, 40, 48)
        assert out_mp.dtype == torch.float32
        assert torch.isfinite(out_mp).all()
        assert out_mp.min().item() >= 0.0 and out_mp.max().item() <= 1.0

    # Verify fp16 on CPU logs one-time warning and passes FP32 into model
    caplog.clear()
    spy_fp16 = ConstantUpsampler(scale=2, value=0.5)
    inf_fp16 = TiledInferencer(
        model=spy_fp16,
        scale=2,
        tile_size=16,
        tile_overlap=4,
        mixed_precision="fp16",
        device="cpu",
    )
    with caplog.at_level(logging.WARNING, logger="srtrainer"):
        _ = inf_fp16.infer_tensor(_make_gradient_tensor(24, 24, batch=1))
        _ = inf_fp16.infer_tensor(_make_gradient_tensor(24, 24, batch=1))
    assert all(dt == torch.float32 for dt in spy_fp16.observed_dtypes)
    fp16_warnings = [r for r in caplog.records if "fp16" in r.getMessage()]
    assert len(fp16_warnings) == 1


# ---------------------------------------------------------------------------
# Part B: tools/infer.py Mode 1, Mode 2, I/O & error handling (in-process)
# ---------------------------------------------------------------------------


def test_infer_mode1_exp_dir_auto_restore_and_weight_priority(tmp_path: Path) -> None:
    infer_mod = _load_infer_module()
    in_img = tmp_path / "input.png"
    _make_rgb_image(in_img, h=14, w=18, rgb=(0.5, 0.5, 0.5))

    exp_dir = _make_exp_dir(
        tmp_path / "exp_val_ema_true",
        scale=2,
        gt_size=[24, 32],
        primary_dataset="set5",
        primary_metric="psnr",
        val_use_ema=True,
        weights_spec={
            "best_set5_psnr.safetensors": 0.20,
            "best_other_ssim.safetensors": 0.35,
            "latest_ema.safetensors": 0.55,
            "latest.safetensors": 0.75,
        },
    )

    def _run_and_read_mean(exp: Path, out_name: str, extra_overrides: list[str] | None = None) -> tuple[Any, float, tuple[int, ...]]:
        out_dir = tmp_path / out_name
        overrides = [
            f"infer.exp_dir={exp}",
            f"infer.input={in_img}",
            f"infer.output_dir={out_dir}",
            "infer.mixed_precision=no",
            "infer.tile.overlap=4",
        ]
        if extra_overrides:
            overrides.extend(extra_overrides)
        cfg = _compose_infer(overrides)
        summary = infer_mod.run_infer(cfg)
        out_tensor = load_image(out_dir / "input.png")
        return summary, out_tensor.mean().item(), tuple(out_tensor.shape)

    # 1. All present + use_ema=True -> best_{primary_dataset}_{primary_metric}; auto-restores scale=2
    summary, mean_val, shape = _run_and_read_mean(exp_dir, "out_1")
    assert summary.mode == "exp_dir"
    assert summary.weights_path.name == "best_set5_psnr.safetensors"
    assert shape == (3, 28, 36)  # scale=2 auto-restored (not default 4)
    assert abs(mean_val - 0.20) < 0.01

    # 2. Remove primary best -> falls back to remaining best_* (newest mtime if multiple)
    (exp_dir / "weights" / "best_set5_psnr.safetensors").unlink()
    older_best = exp_dir / "weights" / "best_aaa_psnr.safetensors"
    save_safetensors(_make_constant_toy_state_dict(scale=2, fill_value=0.28), older_best)
    other_best = exp_dir / "weights" / "best_other_ssim.safetensors"
    os.utime(older_best, (1_000_000, 1_000_000))
    os.utime(other_best, (2_000_000, 2_000_000))
    summary, mean_val, _ = _run_and_read_mean(exp_dir, "out_2")
    assert summary.weights_path.name == "best_other_ssim.safetensors"
    assert abs(mean_val - 0.35) < 0.01

    # 3. Remove all best_* + use_ema=True -> latest_ema
    older_best.unlink()
    other_best.unlink()
    summary, mean_val, _ = _run_and_read_mean(exp_dir, "out_3")
    assert summary.weights_path.name == "latest_ema.safetensors"
    assert abs(mean_val - 0.55) < 0.01

    # 4. Remove latest_ema + use_ema=True -> latest
    (exp_dir / "weights" / "latest_ema.safetensors").unlink()
    summary, mean_val, _ = _run_and_read_mean(exp_dir, "out_4")
    assert summary.weights_path.name == "latest.safetensors"
    assert abs(mean_val - 0.75) < 0.01

    # 5. use_ema=False with val_use_ema=True -> skips EMA best_* & latest_ema, selects latest
    exp_ema_skip = _make_exp_dir(
        tmp_path / "exp_ema_skip",
        scale=2,
        gt_size=32,
        val_use_ema=True,
        weights_spec={
            "best_set5_psnr.safetensors": 0.20,
            "latest_ema.safetensors": 0.55,
            "latest.safetensors": 0.75,
        },
    )
    summary, mean_val, _ = _run_and_read_mean(exp_ema_skip, "out_5", ["infer.use_ema=false"])
    assert summary.weights_path.name == "latest.safetensors"
    assert abs(mean_val - 0.75) < 0.01

    # 6. use_ema=False with val_use_ema=True and NO latest -> raises FileNotFoundError
    (exp_ema_skip / "weights" / "latest.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        _run_and_read_mean(exp_ema_skip, "out_6_fail", ["infer.use_ema=false"])

    # 7. val_use_ema=False: both use_ema=False AND use_ema=True pick best_* (Survey 3 A3)
    exp_val_no_ema = _make_exp_dir(
        tmp_path / "exp_val_no_ema",
        scale=2,
        gt_size=32,
        val_use_ema=False,
        weights_spec={
            "best_set5_psnr.safetensors": 0.22,
            "latest_ema.safetensors": 0.55,
            "latest.safetensors": 0.75,
        },
    )
    summary, mean_val, _ = _run_and_read_mean(exp_val_no_ema, "out_7a", ["infer.use_ema=false"])
    assert summary.weights_path.name == "best_set5_psnr.safetensors"
    assert abs(mean_val - 0.22) < 0.01

    summary, mean_val, _ = _run_and_read_mean(exp_val_no_ema, "out_7b", ["infer.use_ema=true"])
    assert summary.weights_path.name == "best_set5_psnr.safetensors"
    assert abs(mean_val - 0.22) < 0.01

    # 7c. val_use_ema=False with NO best_* and use_ema=False -> skips latest_ema, selects latest
    (exp_val_no_ema / "weights" / "best_set5_psnr.safetensors").unlink()
    summary, mean_val, _ = _run_and_read_mean(exp_val_no_ema, "out_7c", ["infer.use_ema=false"])
    assert summary.weights_path.name == "latest.safetensors"
    assert abs(mean_val - 0.75) < 0.01

    # 7d. Passing non-null CLI arch alongside exp_dir logs a warning and still restores exp_dir arch
    warn_logger = logging.getLogger("srtrainer")
    warn_records: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            warn_records.append(record.getMessage())

    handler = _ListHandler(level=logging.WARNING)
    warn_logger.addHandler(handler)
    try:
        cfg_with_arch = _compose_infer(
            [f"infer.exp_dir={exp_val_no_ema}", "arch/vae=qwen_vae2d", "arch/dit=vosr2_dit"]
        )
        resolved_with_arch = infer_mod.resolve_infer_config(cfg_with_arch, logger=warn_logger)
    finally:
        warn_logger.removeHandler(handler)
    assert resolved_with_arch.scale == 2
    assert any("ignoring CLI arch override" in msg for msg in warn_records)

    # 8. Explicit weights_path override (.pth + weights_key) beats auto-selection
    override_pth = tmp_path / "override.pth"
    torch.save(
        {
            "params_ema": _make_constant_toy_state_dict(scale=2, fill_value=0.90),
            "params": _make_constant_toy_state_dict(scale=2, fill_value=0.10),
        },
        override_pth,
    )
    summary, mean_val, _ = _run_and_read_mean(
        exp_val_no_ema,
        "out_8",
        [f"infer.weights_path={override_pth}", "infer.weights_key=params_ema"],
    )
    assert summary.weights_path == override_pth
    assert abs(mean_val - 0.90) < 0.01

    # 9. Mode 1 error cases: missing exp_dir, missing config_resolved.yaml, missing/empty weights
    with pytest.raises(FileNotFoundError):
        infer_mod.run_infer(
            _compose_infer(
                [
                    f"infer.exp_dir={tmp_path / 'does_not_exist'}",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'err1'}",
                ]
            )
        )

    no_cfg_exp = _make_exp_dir(tmp_path / "no_cfg_exp", write_config=False)
    with pytest.raises(FileNotFoundError):
        infer_mod.run_infer(
            _compose_infer(
                [
                    f"infer.exp_dir={no_cfg_exp}",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'err2'}",
                ]
            )
        )

    empty_weights_exp = _make_exp_dir(tmp_path / "empty_weights_exp", weights_spec={})
    with pytest.raises(FileNotFoundError):
        infer_mod.run_infer(
            _compose_infer(
                [
                    f"infer.exp_dir={empty_weights_exp}",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'err3'}",
                ]
            )
        )


def test_infer_mode2_standalone_safetensors_pth_and_errors(tmp_path: Path) -> None:
    infer_mod = _load_infer_module()
    in_img = tmp_path / "standalone_in.png"
    lq_tensor = _make_rgb_image(in_img, h=15, w=19)

    torch.manual_seed(7)
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    st_path = tmp_path / "toy_x2.safetensors"
    save_safetensors(model.state_dict(), st_path)

    dummy_arch_ov = (
        "++arch={_target_:tests.dummy_arch.DummySRNet,"
        "num_in_ch:3,num_out_ch:3,num_feat:8,num_block:1,scale:2}"
    )

    # Standalone with safetensors + tile.size="auto" (no gt_size -> untiled fallback)
    out_st = tmp_path / "out_st"
    cfg_st = _compose_infer(
        [
            dummy_arch_ov,
            "scale=2",
            f"infer.weights_path={st_path}",
            f"infer.input={in_img}",
            f"infer.output_dir={out_st}",
            "infer.mixed_precision=no",
            "infer.tile.size=auto",
        ]
    )
    summary_st = infer_mod.run_infer(cfg_st)
    assert summary_st.mode == "standalone"
    assert summary_st.total_images == 1
    out_img = load_image(out_st / "standalone_in.png")
    ref_inf = TiledInferencer(model=model, scale=2, tile_enabled=False, pad_multiple=8, mixed_precision="no")
    ref_img = ref_inf.infer_tensor(lq_tensor)
    assert tuple(out_img.shape) == (3, 30, 38)
    assert (out_img - ref_img).abs().max().item() <= (1.0 / 255.0) + 1e-5

    # Nested .pth with weights_key="params_ema"
    pth_path = tmp_path / "toy_nested.pth"
    torch.save(
        {
            "params_ema": _make_constant_toy_state_dict(scale=2, fill_value=0.64),
            "params": _make_constant_toy_state_dict(scale=2, fill_value=0.12),
        },
        pth_path,
    )
    out_pth = tmp_path / "out_pth"
    cfg_pth = _compose_infer(
        [
            dummy_arch_ov,
            "scale=2",
            f"infer.weights_path={pth_path}",
            "infer.weights_key=params_ema",
            f"infer.input={in_img}",
            f"infer.output_dir={out_pth}",
            "infer.mixed_precision=no",
            "infer.tile.enabled=false",
        ]
    )
    infer_mod.run_infer(cfg_pth)
    pth_out_img = load_image(out_pth / "standalone_in.png")
    assert abs(pth_out_img.mean().item() - 0.64) < 0.01

    # Error cases: missing arch, missing weights_path, missing file, nested .pth w/o key, bad key
    with pytest.raises(ValueError, match="arch"):
        infer_mod.run_infer(
            _compose_infer(
                [
                    f"infer.weights_path={st_path}",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'e_no_arch'}",
                ]
            )
        )

    with pytest.raises(ValueError, match="weights_path"):
        infer_mod.run_infer(
            _compose_infer(
                [
                    "arch/vae=qwen_vae2d",
                    "arch/dit=vosr2_dit",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'e_no_w'}",
                ]
            )
        )

    with pytest.raises(FileNotFoundError):
        infer_mod.run_infer(
            _compose_infer(
                [
                    "arch/vae=qwen_vae2d",
                    "arch/dit=vosr2_dit",
                    f"++infer.weights_path={{dit:{tmp_path / 'missing.safetensors'}}}",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'e_missing_w'}",
                ]
            )
        )

    with pytest.raises(ValueError, match="nested"):
        infer_mod.run_infer(
            _compose_infer(
                [
                    dummy_arch_ov,
                    "scale=2",
                    f"infer.weights_path={pth_path}",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'e_no_key'}",
                    "infer.mixed_precision=no",
                ]
            )
        )

    with pytest.raises(KeyError):
        infer_mod.run_infer(
            _compose_infer(
                [
                    dummy_arch_ov,
                    "scale=2",
                    f"infer.weights_path={st_path}",
                    "infer.weights_key=params_ema",
                    f"infer.input={in_img}",
                    f"infer.output_dir={tmp_path / 'e_st_key'}",
                    "infer.mixed_precision=no",
                ]
            )
        )


def test_infer_vosr_multi_arch_exp_dir_and_standalone(tmp_path: Path) -> None:
    """Verify Mode 1 (exp_dir) and Mode 2 (standalone) multi-arch VOSR weight resolution."""
    from models import VOSRPipeline
    from tests.dummy_arch import DummyDino, DummyDiT, DummyVAE, get_dummy_vosr_arch_dict

    infer_mod = _load_infer_module()
    in_img = tmp_path / "vosr_in.png"
    lq_tensor = _make_rgb_image(in_img, h=16, w=16, rgb=(0.5, 0.5, 0.5))

    def _make_vosr_state_dicts(seed: int) -> tuple[VOSRPipeline, dict[str, dict[str, torch.Tensor]]]:
        torch.manual_seed(seed)
        vae = DummyVAE(z_dim=4, spatial_compression_ratio=2).eval()
        dit = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8).eval()
        dino = DummyDino(embed_dim=8, image_size=16, patch_size=4).eval()
        pipe = VOSRPipeline(vae=vae, dit=dit, dino=dino, scale=2, eval_seed=42).eval()
        sds = {
            "vae": {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()},
            "dit": {k: v.detach().cpu().clone() for k, v in dit.state_dict().items()},
            "dino": {k: v.detach().cpu().clone() for k, v in dino.state_dict().items()},
        }
        return pipe, sds

    pipe_best, sd_best = _make_vosr_state_dicts(101)
    pipe_ema, sd_ema = _make_vosr_state_dicts(202)
    pipe_latest, sd_latest = _make_vosr_state_dicts(303)

    exp_dir = tmp_path / "exp_vosr_multi"
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(
        OmegaConf.create(
            {
                "scale": 2,
                "arch": get_dummy_vosr_arch_dict(),
                "model": {
                    "_target_": "models.VOSRModel",
                    "scale": 2,
                    "trainable": {"vae": False, "dit": True, "dino": False},
                    "noise_input": "gaussian",
                    "posterior_mode": True,
                    "eval_seed": 42,
                },
                "data": {"gt_size": 32},
                "metric": {"primary_dataset": "val_text", "primary_metric": "psnr"},
                "trainer": {"val_use_ema": True},
            }
        ),
        exp_dir / "config_resolved.yaml",
    )
    for comp in ("vae", "dit", "dino"):
        save_safetensors(sd_best[comp], weights_dir / f"best_val_text_psnr_{comp}.safetensors")
        save_safetensors(sd_latest[comp], weights_dir / f"latest_{comp}.safetensors")
    save_safetensors(sd_ema["dit"], weights_dir / "latest_dit_ema.safetensors")
    # Also ensure latest_vae and latest_dino match pipe_ema when testing latest_ema fallback
    ref_inf_best = TiledInferencer(
        model=pipe_best, scale=2, tile_size="auto", tile_overlap=4, gt_size=32, mixed_precision="no"
    )
    expected_best = ref_inf_best.infer_tensor(lq_tensor)

    # 1. Mode 1 default (use_ema=True): selects consistent best_val_text_psnr_{vae,dit,dino}
    out_best = tmp_path / "vosr_out_best"
    sum_best = infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_dir}",
                f"infer.input={in_img}",
                f"infer.output_dir={out_best}",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert sum_best.mode == "exp_dir"
    assert sum_best.weights_path.name == "best_val_text_psnr_dit.safetensors"
    actual_best = load_image(out_best / "vosr_in.png")
    assert tuple(actual_best.shape) == (3, 32, 32)
    assert (actual_best - expected_best).abs().max().item() <= (1.0 / 255.0) + 1e-5

    # 2. Mode 1 use_ema=False (val_use_ema=True): skips best_* & latest_dit_ema, selects latest_{comp}
    out_no_ema = tmp_path / "vosr_out_no_ema"
    sum_no_ema = infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_dir}",
                f"infer.input={in_img}",
                f"infer.output_dir={out_no_ema}",
                "infer.use_ema=false",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert sum_no_ema.weights_path.name == "latest_dit.safetensors"
    ref_inf_latest = TiledInferencer(
        model=pipe_latest, scale=2, tile_size="auto", tile_overlap=4, gt_size=32, mixed_precision="no"
    )
    assert (load_image(out_no_ema / "vosr_in.png") - ref_inf_latest.infer_tensor(lq_tensor)).abs().max().item() <= (1.0 / 255.0) + 1e-5

    # 3. Remove best_* -> Mode 1 use_ema=True picks latest_dit_ema + latest_{vae,dino}
    for comp in ("vae", "dit", "dino"):
        (weights_dir / f"best_val_text_psnr_{comp}.safetensors").unlink()
    out_ema = tmp_path / "vosr_out_ema"
    sum_ema = infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_dir}",
                f"infer.input={in_img}",
                f"infer.output_dir={out_ema}",
                "infer.use_ema=true",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert sum_ema.weights_path.name == "latest_dit_ema.safetensors"
    pipe_mixed = VOSRPipeline(
        vae=DummyVAE(z_dim=4, spatial_compression_ratio=2).eval(),
        dit=DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8).eval(),
        dino=DummyDino(embed_dim=8, image_size=16, patch_size=4).eval(),
        scale=2,
        eval_seed=42,
    ).eval()
    pipe_mixed.vae.load_state_dict(sd_latest["vae"])
    pipe_mixed.dit.load_state_dict(sd_ema["dit"])
    assert pipe_mixed.dino is not None
    pipe_mixed.dino.load_state_dict(sd_latest["dino"])
    ref_inf_mixed = TiledInferencer(
        model=pipe_mixed, scale=2, tile_size="auto", tile_overlap=4, gt_size=32, mixed_precision="no"
    )
    assert (load_image(out_ema / "vosr_in.png") - ref_inf_mixed.infer_tensor(lq_tensor)).abs().max().item() <= (1.0 / 255.0) + 1e-5

    # 4. Mode 1 save_frozen_weights=False fallback to model.pretrain.{vae,dino}.path
    pretrain_dir = tmp_path / "frozen_pretrains"
    pretrain_dir.mkdir()
    vae_pt = save_safetensors(sd_best["vae"], pretrain_dir / "vae_base.safetensors")
    dino_pt = save_safetensors(sd_best["dino"], pretrain_dir / "dino_base.safetensors")
    exp_frozen_fallback = tmp_path / "exp_vosr_frozen_fallback"
    (exp_frozen_fallback / "weights").mkdir(parents=True)
    save_safetensors(sd_best["dit"], exp_frozen_fallback / "weights" / "best_val_text_psnr_dit.safetensors")
    save_resolved_config(
        OmegaConf.create(
            {
                "scale": 2,
                "arch": get_dummy_vosr_arch_dict(),
                "model": {
                    "_target_": "models.VOSRModel",
                    "scale": 2,
                    "trainable": {"vae": False, "dit": True, "dino": False},
                    "save_frozen_weights": False,
                    "pretrain": {
                        "vae": {"path": str(vae_pt), "strict": True, "key": None},
                        "dit": {"path": None, "strict": True, "key": None},
                        "dino": {"path": str(dino_pt), "strict": True, "key": None},
                    },
                    "eval_seed": 42,
                },
                "data": {"gt_size": 32},
                "metric": {"primary_dataset": "val_text", "primary_metric": "psnr"},
                "trainer": {"val_use_ema": True},
            }
        ),
        exp_frozen_fallback / "config_resolved.yaml",
    )
    out_fb = tmp_path / "vosr_out_fb"
    infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_frozen_fallback}",
                f"infer.input={in_img}",
                f"infer.output_dir={out_fb}",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert (load_image(out_fb / "vosr_in.png") - expected_best).abs().max().item() <= (1.0 / 255.0) + 1e-5

    # 5. Mode 2 standalone multi-arch VOSR without DINO (dino=null)
    torch.manual_seed(404)
    vae_nd = DummyVAE(z_dim=4, spatial_compression_ratio=2).eval()
    dit_nd = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=None).eval()
    vae_nd_path = save_safetensors(vae_nd.state_dict(), tmp_path / "standalone_vae.safetensors")
    dit_nd_path = save_safetensors(dit_nd.state_dict(), tmp_path / "standalone_dit.safetensors")
    pipe_nd = VOSRPipeline(vae=vae_nd, dit=dit_nd, dino=None, scale=2, eval_seed=42).eval()
    ref_nd = TiledInferencer(
        model=pipe_nd, scale=2, tile_enabled=False, pad_multiple=4, mixed_precision="no"
    ).infer_tensor(lq_tensor)

    out_standalone = tmp_path / "vosr_out_standalone"
    cfg_standalone = _compose_infer(
        [
            "scale=2",
            "++arch={vae:{_target_:tests.dummy_arch.DummyVAE,z_dim:4,spatial_compression_ratio:2},"
            "dit:{_target_:tests.dummy_arch.DummyDiT,in_channels:8,out_channels:4,patch_size:2,hidden_ch:8,z_dims:null},"
            "dino:null}",
            f"++infer.weights_path={{vae:{vae_nd_path},dit:{dit_nd_path}}}",
            f"infer.input={in_img}",
            f"infer.output_dir={out_standalone}",
            "infer.mixed_precision=no",
            "infer.tile.enabled=false",
            "infer.tile.pad_multiple=4",
        ]
    )
    sum_nd = infer_mod.run_infer(cfg_standalone)
    assert sum_nd.mode == "standalone"
    assert sum_nd.weights_path == dit_nd_path
    assert (load_image(out_standalone / "vosr_in.png") - ref_nd).abs().max().item() <= (1.0 / 255.0) + 1e-5


def test_infer_input_output_modes_single_dir_recursive_and_json(tmp_path: Path) -> None:
    infer_mod = _load_infer_module()
    # Use a ToyModel that actually depends on input pixels so distinct inputs give distinct outputs
    torch.manual_seed(123)
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    exp_dir = _make_exp_dir(tmp_path / "exp_io", scale=2, gt_size=32, weights_spec={})
    save_safetensors(model.state_dict(), exp_dir / "weights" / "latest.safetensors")

    # 1. Single image: png and jpg + suffix
    single_in = tmp_path / "single.png"
    _make_rgb_image(single_in, h=12, w=16, rgb=(0.2, 0.6, 0.8))
    for fmt in ("png", "jpg"):
        out_single = tmp_path / f"out_single_{fmt}"
        infer_mod.run_infer(
            _compose_infer(
                [
                    f"infer.exp_dir={exp_dir}",
                    f"infer.input={single_in}",
                    f"infer.output_dir={out_single}",
                    "infer.suffix=_x2",
                    f"infer.save_format={fmt}",
                    "infer.mixed_precision=no",
                    "infer.tile.overlap=4",
                ]
            )
        )
        saved = out_single / f"single_x2.{fmt}"
        assert saved.is_file()
        assert tuple(load_image(saved).shape) == (3, 24, 32)

    # 2. Directory: non-recursive vs recursive preserving subdirs & preventing collisions
    dir_in = tmp_path / "dir_in"
    _make_rgb_image(dir_in / "top1.png", 10, 12, rgb=(0.1, 0.2, 0.3))
    _make_rgb_image(dir_in / "top2.webp", 10, 12, rgb=(0.4, 0.5, 0.6))
    (dir_in / "ignore.txt").write_text("not an image", encoding="utf-8")
    _make_rgb_image(dir_in / "sub1" / "same.png", 10, 12, rgb=(0.9, 0.1, 0.1))
    _make_rgb_image(dir_in / "sub2" / "same.png", 10, 12, rgb=(0.1, 0.1, 0.9))

    out_flat = tmp_path / "out_flat"
    sum_flat = infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_dir}",
                f"infer.input={dir_in}",
                f"infer.output_dir={out_flat}",
                "infer.recursive=false",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert sum_flat.total_images == 2
    assert sorted(p.relative_to(out_flat).as_posix() for p in out_flat.rglob("*.png")) == [
        "top1.png",
        "top2.png",
    ]

    out_rec = tmp_path / "out_rec"
    sum_rec = infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_dir}",
                f"infer.input={dir_in}",
                f"infer.output_dir={out_rec}",
                "infer.recursive=true",
                "infer.suffix=_sr",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert sum_rec.total_images == 4
    rec_files = sorted(p.relative_to(out_rec).as_posix() for p in out_rec.rglob("*.png"))
    assert rec_files == [
        "sub1/same_sr.png",
        "sub2/same_sr.png",
        "top1_sr.png",
        "top2_sr.png",
    ]
    # Verify sub1/same and sub2/same were NOT overwritten and produce distinct outputs
    im1 = load_image(out_rec / "sub1" / "same_sr.png")
    im2 = load_image(out_rec / "sub2" / "same_sr.png")
    assert not torch.allclose(im1, im2, atol=1e-3)

    # 3. JSON manifest with input_root (lq + gt-fallback + empty-lq fallback)
    data_root = tmp_path / "manifest_data"
    _make_rgb_image(data_root / "lr" / "img1.png", 10, 10, rgb=(0.8, 0.2, 0.2))
    _make_rgb_image(data_root / "hr" / "img1.png", 20, 20, rgb=(0.2, 0.8, 0.2))
    _make_rgb_image(data_root / "hr" / "img2.png", 12, 14, rgb=(0.2, 0.2, 0.8))
    _make_rgb_image(data_root / "hr" / "img3.png", 14, 10, rgb=(0.7, 0.7, 0.1))

    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {"lq": "lr/img1.png", "gt": "hr/img1.png"},
                {"gt": "hr/img2.png"},
                {"lq": "", "gt": "hr/img3.png"},
            ]
        ),
        encoding="utf-8",
    )
    out_json_root = tmp_path / "out_json_root"
    sum_jr = infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_dir}",
                f"infer.input={manifest_path}",
                f"infer.input_root={data_root}",
                f"infer.output_dir={out_json_root}",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert sum_jr.total_images == 3
    # img1 used lq (10x10 -> 20x20), NOT gt (20x20 -> 40x40)
    assert tuple(load_image(out_json_root / "lr" / "img1.png").shape) == (3, 20, 20)
    assert tuple(load_image(out_json_root / "hr" / "img2.png").shape) == (3, 24, 28)
    assert tuple(load_image(out_json_root / "hr" / "img3.png").shape) == (3, 28, 20)

    # 4. JSON manifest with input_root=null (resolves relative to JSON parent dir)
    rel_manifest = data_root / "rel_manifest.json"
    rel_manifest.write_text(json.dumps([{"gt": "hr/img2.png"}]), encoding="utf-8")
    out_json_noroot = tmp_path / "out_json_noroot"
    sum_jnr = infer_mod.run_infer(
        _compose_infer(
            [
                f"infer.exp_dir={exp_dir}",
                f"infer.input={rel_manifest}",
                f"infer.output_dir={out_json_noroot}",
                "infer.mixed_precision=no",
                "infer.tile.overlap=4",
            ]
        )
    )
    assert sum_jnr.total_images == 1
    assert (out_json_noroot / "hr" / "img2.png").is_file()

    # 5. Collision disambiguation & path-traversal safety in collect_input_items
    abs_a = tmp_path / "abs_a" / "dup.png"
    abs_b = tmp_path / "abs_b" / "dup.png"
    abs_b_jpg = tmp_path / "abs_b" / "dup.jpg"
    _make_rgb_image(abs_a, 8, 8)
    _make_rgb_image(abs_b, 8, 8)
    _make_rgb_image(abs_b_jpg, 8, 8)
    coll_manifest = tmp_path / "sub_m" / "coll.json"
    coll_manifest.parent.mkdir(parents=True, exist_ok=True)
    coll_manifest.write_text(
        json.dumps(
            [
                {"lq": str(abs_a)},
                {"lq": str(abs_b)},
                {"lq": str(abs_b_jpg)},
                {"gt": "../abs_a/dup.png"},
            ]
        ),
        encoding="utf-8",
    )
    safe_out = tmp_path / "safe_out"
    items = infer_mod.collect_input_items(
        input_spec=coll_manifest,
        output_dir=safe_out,
        save_format="png",
    )
    assert len(items) == 4
    out_paths = [it.output_path.resolve() for it in items]
    assert len(set(out_paths)) == 4
    assert all(safe_out.resolve() in p.parents for p in out_paths)

    # 6. Error handling matrix
    with pytest.raises(ValueError):
        infer_mod.collect_input_items(None, tmp_path / "o")
    with pytest.raises(ValueError):
        infer_mod.collect_input_items("", tmp_path / "o")
    with pytest.raises(FileNotFoundError):
        infer_mod.collect_input_items(tmp_path / "nonexistent.png", tmp_path / "o")
    with pytest.raises(ValueError):
        infer_mod.collect_input_items(dir_in / "ignore.txt", tmp_path / "o")

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        infer_mod.collect_input_items(empty_dir, tmp_path / "o")

    for bad_name, content in (
        ("bad_syntax.json", "{not valid json"),
        ("not_list.json", json.dumps({"lq": "a.png"})),
        ("empty_list.json", "[]"),
        ("non_dict_item.json", json.dumps(["a.png"])),
        ("no_lq_or_gt.json", json.dumps([{"label": "ABC"}])),
        ("empty_lq_gt.json", json.dumps([{"lq": "", "gt": None}])),
        ("non_str_lq.json", json.dumps([{"lq": 123}])),
    ):
        p = tmp_path / bad_name
        p.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            infer_mod.collect_input_items(p, tmp_path / "o")

    missing_img_manifest = tmp_path / "missing_img.json"
    missing_img_manifest.write_text(json.dumps([{"lq": "nope.png"}]), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        infer_mod.collect_input_items(missing_img_manifest, tmp_path / "o")
    with pytest.raises(FileNotFoundError):
        infer_mod.collect_input_items(
            missing_img_manifest,
            tmp_path / "o",
            input_root=tmp_path / "missing_root",
        )

    with pytest.raises(ValueError):
        infer_mod.collect_input_items(single_in, tmp_path / "o", save_format="bmp")
    with pytest.raises(ValueError):
        infer_mod.collect_input_items(single_in, tmp_path / "o", suffix="../escape")
    with pytest.raises(ValueError):
        infer_mod.collect_input_items(single_in, tmp_path / "o", suffix="sub/bad")
    with pytest.raises(ValueError):
        infer_mod.collect_input_items(single_in, "")


# ---------------------------------------------------------------------------
# Part C: Real CLI & 2-process CPU torchrun distributed tests (cwd="/tmp")
# ---------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_subprocess(cmd: list[str], cwd: str = "/tmp") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ACCELERATE_USE_CPU"] = "true"
    env["OMP_NUM_THREADS"] = "1"
    if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in env:
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    existing_pp = env.get("PYTHONPATH", "")
    extra_pp = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    env["PYTHONPATH"] = f"{extra_pp}:{existing_pp}" if existing_pp else extra_pp
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def test_infer_cli_distributed_2proc_cpu_5_images_from_tmp(tmp_path: Path) -> None:
    torch.manual_seed(99)
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    exp_dir = _make_exp_dir(tmp_path / "exp_dist5", scale=2, gt_size=32, weights_spec={})
    save_safetensors(model.state_dict(), exp_dir / "weights" / "best_set5_psnr.safetensors")

    in_dir = tmp_path / "dist_in"
    sizes = {
        "img_0.png": (16, 16),
        "img_1.png": (17, 23),
        "img_2.png": (20, 12),
        "img_3.png": (3, 5),
        "img_4.png": (24, 24),
    }
    in_tensors: dict[str, torch.Tensor] = {}
    for name, (h, w) in sizes.items():
        in_tensors[name] = _make_rgb_image(in_dir / name, h=h, w=w)

    out_dir = tmp_path / "dist_out_5"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={_free_port()}",
        str(INFER_SCRIPT),
        f"infer.exp_dir={exp_dir}",
        f"infer.input={in_dir}",
        f"infer.output_dir={out_dir}",
        "infer.mixed_precision=no",
        "infer.tile.size=auto",
        "infer.tile.overlap=4",
        "infer.tile.batch_size=2",
    ]
    proc = _run_subprocess(cmd, cwd="/tmp")
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    saved_files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert saved_files == sorted(sizes.keys())

    # Verify pixel accuracy against single-process TiledInferencer reference
    ref_inf = TiledInferencer(
        model=model,
        scale=2,
        tile_size="auto",
        tile_overlap=4,
        tile_batch_size=2,
        gt_size=32,
        mixed_precision="no",
    )
    for name, (h, w) in sizes.items():
        actual = load_image(out_dir / name)
        expected = ref_inf.infer_tensor(in_tensors[name])
        assert tuple(actual.shape) == (3, h * 2, w * 2)
        assert torch.isfinite(actual).all()
        assert (actual - expected).abs().max().item() <= (1.0 / 255.0) + 1e-5

    # Verify Rank 0 summary printed exactly once with total count, throughput, and output_dir
    done_lines = [line for line in proc.stdout.splitlines() if "[Infer] Done:" in line]
    assert len(done_lines) == 1, f"Expected 1 summary line, got:\n{proc.stdout}"
    assert "5 images" in done_lines[0]
    assert "img/s" in done_lines[0]
    assert str(out_dir) in done_lines[0]


def test_infer_cli_distributed_2proc_cpu_single_image_lt_world_size(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    exp_dir = _make_exp_dir(tmp_path / "exp_dist1", scale=2, gt_size=32, weights_spec={})
    save_safetensors(model.state_dict(), exp_dir / "weights" / "latest.safetensors")

    in_img = tmp_path / "only_one.png"
    _make_rgb_image(in_img, h=15, w=17)
    out_dir = tmp_path / "dist_out_1"

    # Uses default mixed_precision=bf16 on CPU across 2 ranks with only 1 image
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={_free_port()}",
        str(INFER_SCRIPT),
        f"infer.exp_dir={exp_dir}",
        f"infer.input={in_img}",
        f"infer.output_dir={out_dir}",
        "infer.tile.overlap=4",
    ]
    proc = _run_subprocess(cmd, cwd="/tmp")
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    saved_files = [p.name for p in out_dir.iterdir() if p.is_file()]
    assert saved_files == ["only_one.png"]
    out_img = load_image(out_dir / "only_one.png")
    assert tuple(out_img.shape) == (3, 30, 34)
    assert torch.isfinite(out_img).all()

    done_lines = [line for line in proc.stdout.splitlines() if "[Infer] Done:" in line]
    assert len(done_lines) == 1, f"Expected 1 summary line, got:\n{proc.stdout}"
    assert "1 images" in done_lines[0]


def test_infer_cli_single_proc_exp_dir_from_tmp(tmp_path: Path) -> None:
    exp_dir = _make_exp_dir(
        tmp_path / "exp_cli_single",
        scale=2,
        gt_size=32,
        weights_spec={"best_set5_psnr.safetensors": 0.42},
    )
    in_img = tmp_path / "cli_in.png"
    _make_rgb_image(in_img, h=12, w=14)
    out_dir = tmp_path / "cli_out_single"

    cmd = [
        sys.executable,
        str(INFER_SCRIPT),
        f"infer.exp_dir={exp_dir}",
        f"infer.input={in_img}",
        f"infer.output_dir={out_dir}",
        "infer.mixed_precision=no",
        "infer.tile.overlap=4",
    ]
    proc = _run_subprocess(cmd, cwd="/tmp")
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    out_file = out_dir / "cli_in.png"
    assert out_file.is_file()
    out_img = load_image(out_file)
    assert tuple(out_img.shape) == (3, 24, 28)
    assert abs(out_img.mean().item() - 0.42) < 0.01
    assert "[Infer] Done:" in proc.stdout


# ---------------------------------------------------------------------------
# Part D: Mandatory User Requirement #4 Enforcement (F35 — NO SRPredictor)
# ---------------------------------------------------------------------------


def test_no_srpredictor_anywhere() -> None:
    assert not hasattr(inferencer_mod, "SRPredictor")

    scan_targets = [
        *(REPO_ROOT / "src").rglob("*.py"),
        *(REPO_ROOT / "tools").rglob("*.py"),
        *(REPO_ROOT / "configs").rglob("*.yaml"),
    ]
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in scan_targets
        if "SRPredictor" in path.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"Forbidden symbol 'SRPredictor' found in: {offenders}"
