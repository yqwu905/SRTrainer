"""Tier 2 E2E tests — Area F: Shared Tiled Inference & Single-Device CLI Boundaries.

Covers boundary, edge, and negative contracts for F35-F40.
Imports ``from engine.inferencer import TiledInferencer`` inside test
functions so Area F executes cleanly on M4 before M5 populates ``engine/__init__.py``.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def test_t2_f01_tiled_inferencer_rejects_invalid_tile_overlap() -> None:
    """F36: TiledInferencer rejects overlap >= tile_size // 2 or < 0, accepts tight valid overlap."""
    import torch
    from tests.dummy_arch import DummySRNet

    from engine.inferencer import TiledInferencer

    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    lq = torch.rand(1, 3, 40, 40, dtype=torch.float32)

    invalid_pairs: list[tuple[int | list[int], int]] = [
        (16, 8),
        (16, 10),
        (16, -1),
        ([16, 32], 8),
        ([32, 16], 8),
        (15, 7),
    ]
    for tile_size, overlap in invalid_pairs:
        with pytest.raises(ValueError, match="."):
            inf = TiledInferencer(
                model=model,
                scale=2,
                tile_enabled=True,
                tile_size=tile_size,
                tile_overlap=overlap,
                mixed_precision="no",
                device="cpu",
            )
            inf.infer_tensor(lq)

    for valid_size in (16, [16, 32]):
        inf_ok = TiledInferencer(
            model=model,
            scale=2,
            tile_enabled=True,
            tile_size=valid_size,
            tile_overlap=7,
            mixed_precision="no",
            device="cpu",
        )
        out = inf_ok.infer_tensor(lq)
        assert out.shape == (1, 3, 80, 80)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_t2_f02_tiled_inferencer_rejects_invalid_tile_size_blend_batch_and_auto_indivisible() -> (
    None
):
    """F36: TiledInferencer rejects bad tile_size, non-positive auto tile_size, blend, batch_size."""
    import torch
    from tests.dummy_arch import DummySRNet

    from engine.inferencer import TiledInferencer

    model_x2 = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    model_x4 = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=4).eval()
    lq = torch.rand(1, 3, 32, 32, dtype=torch.float32)

    invalid_configs: list[dict[str, Any]] = [
        {"model": model_x2, "scale": 2, "tile_size": 0, "tile_overlap": 0},
        {"model": model_x2, "scale": 2, "tile_size": -4, "tile_overlap": 0},
        {"model": model_x2, "scale": 2, "tile_size": [16, 16, 16], "tile_overlap": 0},
        {"model": model_x2, "scale": 2, "tile_size": [16, 0], "tile_overlap": 0},
        # tile_size=None (auto) derives gt_size // scale; reject cases where gt_size // scale <= 0.
        {"model": model_x4, "scale": 4, "tile_size": None, "gt_size": 2, "tile_overlap": 0},
        {"model": model_x4, "scale": 4, "tile_size": None, "gt_size": 0, "tile_overlap": 0},
        {"model": model_x2, "scale": 2, "tile_size": 16, "tile_overlap": 4, "blend_mode": "cubic"},
        {"model": model_x2, "scale": 2, "tile_size": 16, "tile_overlap": 4, "tile_batch_size": 0},
        {"model": model_x2, "scale": 2, "tile_size": 16, "tile_overlap": 4, "tile_batch_size": -2},
    ]
    for kwargs in invalid_configs:
        with pytest.raises(ValueError, match="."):
            inf = TiledInferencer(
                tile_enabled=True,
                mixed_precision="no",
                device="cpu",
                **kwargs,
            )
            inf.infer_tensor(lq)


def test_t2_f03_tiled_inferencer_auto_without_gt_size_falls_back_to_untiled() -> None:
    """F36: tile_size='auto' with gt_size=None falls back to single untiled padded forward."""
    import torch
    import torch.nn as nn
    from tests.dummy_arch import DummySRNet

    from engine.inferencer import TiledInferencer

    class _CallShapeRecorder(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.inner = inner
            self.input_shapes: list[tuple[int, ...]] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.input_shapes.append(tuple(x.shape))
            return self.inner(x)

    torch.manual_seed(31)
    toy = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    probe = _CallShapeRecorder(toy)
    lq = torch.rand(1, 3, 19, 23, dtype=torch.float32)

    inf_auto = TiledInferencer(
        model=probe,
        scale=2,
        tile_enabled=True,
        tile_size=None,
        tile_overlap=4,
        gt_size=None,
        pad_multiple=8,
        mixed_precision="no",
        device="cpu",
    )
    out = inf_auto.infer_tensor(lq)

    assert out.shape == (1, 3, 38, 46)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert probe.input_shapes == [(1, 3, 24, 24)]

    out_ref = TiledInferencer(
        model=toy,
        scale=2,
        tile_enabled=False,
        pad_multiple=8,
        mixed_precision="no",
        device="cpu",
    ).infer_tensor(lq)
    assert torch.allclose(out, out_ref, atol=1e-6)


def test_t2_f04_tiled_inferencer_tiny_replicate_fallback_and_strip_shapes() -> None:
    """F36: TiledInferencer handles ultra-tiny (replicate fallback) and skinny strip tensors."""
    import torch
    from tests.dummy_arch import DummySRNet

    from engine.inferencer import TiledInferencer

    torch.manual_seed(34)
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    inf_tiled = TiledInferencer(
        model=model,
        scale=2,
        tile_enabled=True,
        tile_size=16,
        tile_overlap=4,
        tile_batch_size=2,
        blend_mode="cosine",
        pad_multiple=8,
        mixed_precision="no",
        device="cpu",
    )

    for h, w in ((3, 5), (1, 2), (10, 50), (50, 10)):
        lq = torch.rand(1, 3, h, w, dtype=torch.float32)
        out = inf_tiled.infer_tensor(lq)
        assert out.shape == (1, 3, h * 2, w * 2)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    inf_untiled = TiledInferencer(
        model=model,
        scale=2,
        tile_enabled=False,
        pad_multiple=8,
        mixed_precision="no",
        device="cpu",
    )
    out_tiny_untiled = inf_untiled.infer_tensor(torch.rand(1, 3, 1, 2, dtype=torch.float32))
    assert out_tiny_untiled.shape == (1, 3, 2, 4)
    assert out_tiny_untiled.dtype == torch.float32
    assert torch.isfinite(out_tiny_untiled).all()
    assert float(out_tiny_untiled.min()) >= 0.0 and float(out_tiny_untiled.max()) <= 1.0


def test_t2_f05_tiled_inferencer_zero_overlap_hard_stitch_non_divisible() -> None:
    """F36: Zero-overlap hard stitch achieves exact coverage on non-divisible dims across all blends."""
    import torch
    import torch.nn as nn
    from tests.dummy_arch import DummySRNet

    from engine.inferencer import TiledInferencer

    torch.manual_seed(35)
    lq = torch.rand(1, 3, 37, 43, dtype=torch.float32)
    upsampler = nn.Upsample(scale_factor=2, mode="nearest")
    ref = upsampler(lq)

    for blend_mode in ("cosine", "linear", "mean"):
        inf = TiledInferencer(
            model=upsampler,
            scale=2,
            tile_enabled=True,
            tile_size=16,
            tile_overlap=0,
            tile_batch_size=3,
            blend_mode=blend_mode,
            pad_multiple=8,
            mixed_precision="no",
            device="cpu",
        )
        out = inf.infer_tensor(lq)
        assert out.shape == (1, 3, 74, 86)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert (out - ref).abs().max().item() < 1e-5, (
            f"Hard stitch mismatch under blend_mode={blend_mode!r}"
        )

    toy = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    out_toy = TiledInferencer(
        model=toy,
        scale=2,
        tile_enabled=True,
        tile_size=16,
        tile_overlap=0,
        tile_batch_size=4,
        blend_mode="cosine",
        mixed_precision="no",
        device="cpu",
    ).infer_tensor(lq)
    assert out_toy.shape == (1, 3, 74, 86)
    assert out_toy.dtype == torch.float32
    assert torch.isfinite(out_toy).all()
    assert float(out_toy.min()) >= 0.0 and float(out_toy.max()) <= 1.0


def test_t2_f06_tiled_inferencer_cpu_fp16_autocast_guard() -> None:
    """F36: TiledInferencer safely guards unsupported CPU fp16 autocast to FP32 output."""
    import torch
    from tests.dummy_arch import DummySRNet

    from engine.inferencer import TiledInferencer

    torch.manual_seed(36)
    model = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    lq = torch.rand(1, 3, 24, 28, dtype=torch.float32)

    out_no = TiledInferencer(
        model=model,
        scale=2,
        tile_enabled=True,
        tile_size=16,
        tile_overlap=4,
        tile_batch_size=2,
        blend_mode="cosine",
        mixed_precision="no",
        device="cpu",
    ).infer_tensor(lq)

    out_fp16_tiled = TiledInferencer(
        model=model,
        scale=2,
        tile_enabled=True,
        tile_size=16,
        tile_overlap=4,
        tile_batch_size=2,
        blend_mode="cosine",
        mixed_precision="fp16",
        device="cpu",
    ).infer_tensor(lq)

    assert out_fp16_tiled.shape == (1, 3, 48, 56)
    assert out_fp16_tiled.dtype == torch.float32
    assert torch.isfinite(out_fp16_tiled).all()
    assert float(out_fp16_tiled.min()) >= 0.0 and float(out_fp16_tiled.max()) <= 1.0
    assert torch.allclose(out_fp16_tiled, out_no, atol=1e-5)

    out_fp16_untiled = TiledInferencer(
        model=model,
        scale=2,
        tile_enabled=False,
        pad_multiple=8,
        mixed_precision="fp16",
        device="cpu",
    ).infer_tensor(lq)
    assert out_fp16_untiled.shape == (1, 3, 48, 56)
    assert out_fp16_untiled.dtype == torch.float32
    assert torch.isfinite(out_fp16_untiled).all()
    assert float(out_fp16_untiled.min()) >= 0.0 and float(out_fp16_untiled.max()) <= 1.0


@pytest.mark.parametrize(("overrides", "error"), [
    ([], "model._target_"),
    (["model=vosr", "++arch.vae={}"], "arch.dit"),
    (["++model._target_=torch.nn.Identity", "infer.tile.size=[16,16]"], "overlap"),
])
def test_t2_tools_infer_rejects_invalid_configs(
    tmp_path: Path, synthetic_image_factory: Callable[..., Path],
    infer_cli_runner: Callable[..., Any], overrides: list[str], error: str,
) -> None:
    inputs = tmp_path / "images"
    synthetic_image_factory(inputs / "sample.png", width=8, height=8)
    proc = infer_cli_runner([
        f"infer.input={inputs}", f"infer.output_dir={tmp_path / 'out'}", *overrides,
    ], check=False)
    assert proc.returncode != 0 and error in proc.stderr
