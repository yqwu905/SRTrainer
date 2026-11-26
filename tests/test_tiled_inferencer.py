"""TiledInferencer core math and standalone model and image-directory inference."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import ListConfig
from tools.infer import InferItem, collect_input_items, run_infer
from torch import nn
from torch.nn import functional as F

from config import load_infer_config
from engine import inferencer as inferencer_mod
from engine.inferencer import (
    TiledInferencer,
    build_blend_window,
    compute_tile_starts,
)
from models import VOSRPipeline, build_infer_model
from tests.dummy_arch import DummySRNet, get_dummy_vosr_arch_dict
from utils.image import load_image, save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str(REPO_ROOT / "configs")


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
def reset_loggers():
    _reset_srtrainer_loggers()
    yield
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


# ---------------------------------------------------------------------------
# Part A: TiledInferencer core math, feathering, edge sizes & guards
# ---------------------------------------------------------------------------


def test_tiled_inferencer_tile_size_modes_and_validation(caplog: pytest.LogCaptureFixture) -> None:
    model = NearestUpsampler(scale=4)

    # tile_size=None (auto) with gt_size -> gt_size // scale
    inf_auto = TiledInferencer(model=model, scale=4, tile_size=None, tile_overlap=4, gt_size=64)
    assert inf_auto.tile_enabled is True
    assert inf_auto.tile_size == (16, 16)
    assert inf_auto.weight_window is not None
    assert tuple(inf_auto.weight_window.shape) == (1, 1, 64, 64)

    # tile_size=None (auto) without gt_size -> warning + untiled fallback
    with caplog.at_level(logging.WARNING, logger="srtrainer"):
        inf_fallback = TiledInferencer(
            model=model,
            scale=4,
            tile_size=None,
            tile_overlap=16,
            gt_size=None,
            pad_multiple=8,
        )
    assert inf_fallback.tile_enabled is False
    assert inf_fallback.tile_size is None
    assert inf_fallback.weight_window is None
    assert any("gt_size" in rec.getMessage() for rec in caplog.records)

    # tile_enabled=False ignores default overlap/auto without warning
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="srtrainer"):
        inf_disabled = TiledInferencer(
            model=model,
            scale=4,
            tile_enabled=False,
            tile_size=None,
            tile_overlap=16,
            gt_size=None,
        )
    assert inf_disabled.tile_enabled is False
    assert inf_disabled.tile_size is None
    assert inf_disabled.weight_window is None
    assert len(caplog.records) == 0

    # gt_size not divisible by scale
    with pytest.raises(ValueError, match="divisible"):
        TiledInferencer(model=model, scale=4, tile_size=None, tile_overlap=4, gt_size=65)

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

    for bad_size in (0, -8, True, False, 16.5, [16], [16, 0], [True, 16], "auto", "bad", b"16"):
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

    for bad_gt in (0, -16, True):
        with pytest.raises(ValueError):
            TiledInferencer(model=model, scale=4, tile_size=16, tile_overlap=0, gt_size=bad_gt)  # type: ignore[arg-type]

    # from_config with Hydra DictConfig (infer.tile and trainer.val_tile), plain dict, and None
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        infer_cfg = hydra.compose(config_name="infer")
    inf_from_infer = TiledInferencer.from_config(model=model, scale=4, tile_cfg=infer_cfg.infer.tile, gt_size=256)
    assert inf_from_infer.tile_enabled is True
    assert inf_from_infer.tile_size == (64, 64)
    assert inf_from_infer.tile_overlap == 16
    assert inf_from_infer.blend_mode == "cosine"

    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        train_cfg = hydra.compose(
            config_name="train",
            overrides=["~experiment", "arch/vae=qwen_vae2d", "arch/dit=vosr2_dit", "loss=l1"],
        )
    inf_from_train = TiledInferencer.from_config(
        model=model,
        scale=4,
        tile_cfg=train_cfg.trainer.val_tile,
        gt_size=train_cfg.data.gt_size,
    )
    expected_tile = int(train_cfg.data.gt_size) // 4
    assert inf_from_train.tile_size == (expected_tile, expected_tile)

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
# Standalone inference
# ---------------------------------------------------------------------------


def test_collect_directory_images(tmp_path: Path) -> None:
    for name in ("b.JPG", "a.PnG", "skip.jpeg", "skip.webp"):
        (tmp_path / name).touch()
    nested = tmp_path / "nested.png"
    nested.mkdir()
    (nested / "child.png").touch()
    output = tmp_path / "out"
    items = collect_input_items(tmp_path, output, suffix="_sr", save_format="jpg")
    assert items == [
        InferItem(tmp_path / "a.PnG", output / "a_sr.jpg"),
        InferItem(tmp_path / "b.JPG", output / "b_sr.jpg"),
    ]


def test_collect_input_errors(tmp_path: Path) -> None:
    file = tmp_path / "manifest.json"
    file.touch()
    for path in (None, "", tmp_path / "missing", file):
        with pytest.raises(ValueError, match="directory"):
            collect_input_items(path, tmp_path, suffix="", save_format="png")
    with pytest.raises(ValueError, match="No .png or .jpg"):
        collect_input_items(tmp_path, tmp_path, suffix="", save_format="png")


@pytest.mark.parametrize("model", [None, {}])
def test_model_target_required(model) -> None:
    with pytest.raises(ValueError, match="model._target_"):
        build_infer_model(load_infer_config({"model": model}))


@pytest.mark.parametrize("with_dino", [True, False])
def test_vosr_model_directory_inference(tmp_path: Path, with_dino: bool) -> None:
    input_dir = tmp_path / "images"
    _make_rgb_image(input_dir / "odd.png", 23, 29)
    cfg = load_infer_config({
        "scale": 2,
        "model": {"_target_": "models.VOSRModel", "scale": 2},
        "arch": get_dummy_vosr_arch_dict(with_dino=with_dino),
        "infer": {
            "input": str(input_dir), "output_dir": str(tmp_path / "output"),
            "device": "cpu", "mixed_precision": "no",
            "tile": {"size": [16, 20], "overlap": 4},
        },
    })
    model = build_infer_model(cfg, device="cpu")
    assert isinstance(model, VOSRPipeline) and not model.training
    assert (model.dino is not None) == with_dino
    summary = run_infer(cfg)
    assert summary.total_images == 1 and summary.elapsed_sec > 0
    assert summary.images_per_sec == pytest.approx(1 / summary.elapsed_sec)
    assert load_image(summary.output_dir / "odd.png").shape == (3, 46, 58)


def _dummy_sr_factory(num_feat: int = 8) -> DummySRNet:
    return DummySRNet(num_feat=num_feat)


def test_direct_module_with_custom_params() -> None:
    cfg = load_infer_config({
        "model": {"_target_": "tests.dummy_arch.DummySRNet", "num_feat": 17},
        "arch": {"custom": {"anything": 1}},
    })
    model = build_infer_model(cfg, device="cpu")
    assert isinstance(model, DummySRNet) and model.num_feat == 17
    assert not model.training and next(model.parameters()).device.type == "cpu"

    factory_cfg = load_infer_config({
        "model": {"_target_": f"{__name__}._dummy_sr_factory", "num_feat": 11},
    })
    factory_model = build_infer_model(factory_cfg, device="cpu")
    assert isinstance(factory_model, DummySRNet) and factory_model.num_feat == 11
    assert not factory_model.training


def test_vosr_setup_infer_internal_guards() -> None:
    with pytest.raises(ValueError, match=r"model\.scale"):
        build_infer_model(
            load_infer_config({
                "scale": 2,
                "model": {"_target_": "models.VOSRModel", "scale": 1},
                "arch": get_dummy_vosr_arch_dict(with_dino=False),
            }),
            device="cpu",
        )

    with pytest.raises(ValueError, match=r"pretrain\.dino\.path"):
        build_infer_model(
            load_infer_config({
                "scale": 1,
                "model": {
                    "_target_": "models.VOSRModel",
                    "scale": 1,
                    "pretrain": {"dino": {"path": "/nonexistent/dino.safetensors"}},
                },
                "arch": get_dummy_vosr_arch_dict(with_dino=False),
            }),
            device="cpu",
        )


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
