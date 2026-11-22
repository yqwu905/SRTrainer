"""Tier 2 E2E tests — Area F: Shared Tiled Inference & Distributed CLI Boundaries.

Covers boundary, edge, and negative contracts for F35-F40.
Imports ``from engine.inferencer import TiledInferencer`` inside test
functions so Area F executes cleanly on M4 before M5 populates ``engine/__init__.py``.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def test_t2_f01_tiled_inferencer_rejects_invalid_tile_overlap() -> None:
    """F36: TiledInferencer rejects overlap >= tile_size // 2 or < 0, accepts tight valid overlap."""
    import torch

    from archs import ToyModel
    from engine.inferencer import TiledInferencer

    model = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
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

    from archs import ToyModel
    from engine.inferencer import TiledInferencer

    model_x2 = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    model_x4 = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=4).eval()
    lq = torch.rand(1, 3, 32, 32, dtype=torch.float32)

    invalid_configs: list[dict[str, Any]] = [
        {"model": model_x2, "scale": 2, "tile_size": 0, "tile_overlap": 0},
        {"model": model_x2, "scale": 2, "tile_size": -4, "tile_overlap": 0},
        {"model": model_x2, "scale": 2, "tile_size": [16, 16, 16], "tile_overlap": 0},
        {"model": model_x2, "scale": 2, "tile_size": [16, 0], "tile_overlap": 0},
        # tile_size="auto" derives gt_size // scale; reject cases where gt_size // scale <= 0.
        {"model": model_x4, "scale": 4, "tile_size": "auto", "gt_size": 2, "tile_overlap": 0},
        {"model": model_x4, "scale": 4, "tile_size": "auto", "gt_size": 0, "tile_overlap": 0},
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

    from archs import ToyModel
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
    toy = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
    probe = _CallShapeRecorder(toy)
    lq = torch.rand(1, 3, 19, 23, dtype=torch.float32)

    inf_auto = TiledInferencer(
        model=probe,
        scale=2,
        tile_enabled=True,
        tile_size="auto",
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

    from archs import ToyModel
    from engine.inferencer import TiledInferencer

    torch.manual_seed(34)
    model = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
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

    from archs import ToyModel
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

    toy = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
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

    from archs import ToyModel
    from engine.inferencer import TiledInferencer

    torch.manual_seed(36)
    model = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2).eval()
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


def test_t2_f07_tools_infer_cli_fail_fast_invalid_configs(
    e2e_workdir: dict[str, Path],
    synthetic_image_factory: Callable[..., Path],
    bias_tagged_exp_dir_factory: Callable[..., dict[str, Any]],
    standalone_infer_arch_overrides_factory: Callable[..., list[str]],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F37, F38: tools/infer.py exits non-zero on missing inputs, arch, weights, config, or format."""
    from archs import ToyModel
    from utils.misc import save_safetensors

    fmt = hydra_value_formatter
    sample_img = synthetic_image_factory(
        e2e_workdir["data"] / "sample.png", width=16, height=16, seed=1
    )
    exp_valid = bias_tagged_exp_dir_factory(e2e_workdir["work"] / "exp_ok", scale=2, gt_size=32)

    toy = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    valid_st = save_safetensors(toy.state_dict(), e2e_workdir["work"] / "standalone_toy.safetensors")

    # (a) infer.input=null
    proc_a = infer_cli_runner(
        [
            f"infer.exp_dir={fmt(str(exp_valid['exp_dir']))}",
            "infer.input=null",
            f"infer.output_dir={fmt(str(e2e_workdir['outputs'] / 'out_a'))}",
            "infer.mixed_precision='no'",
            "infer.tile.overlap=4",
        ],
        check=False,
    )
    assert proc_a.returncode != 0

    # (b1) Standalone missing arch
    proc_b1 = infer_cli_runner(
        [
            "infer.exp_dir=null",
            f"infer.weights_path={fmt(str(valid_st))}",
            f"infer.input={fmt(str(sample_img))}",
            f"infer.output_dir={fmt(str(e2e_workdir['outputs'] / 'out_b1'))}",
            "infer.mixed_precision='no'",
        ],
        check=False,
    )
    assert proc_b1.returncode != 0

    # (b2) Standalone missing weights_path
    arch_ovs = standalone_infer_arch_overrides_factory(
        {
            "_target_": "archs.ToyModel",
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 8,
            "num_block": 1,
            "scale": 2,
        },
        scale=2,
    )
    proc_b2 = infer_cli_runner(
        [
            *arch_ovs,
            "infer.exp_dir=null",
            "infer.weights_path=null",
            f"infer.input={fmt(str(sample_img))}",
            f"infer.output_dir={fmt(str(e2e_workdir['outputs'] / 'out_b2'))}",
            "infer.mixed_precision='no'",
        ],
        check=False,
    )
    assert proc_b2.returncode != 0

    # (c1) Mode 1 exp_dir missing config_resolved.yaml
    exp_no_cfg = e2e_workdir["work"] / "exp_no_cfg"
    save_safetensors(toy.state_dict(), exp_no_cfg / "weights" / "latest.safetensors")
    proc_c1 = infer_cli_runner(
        [
            f"infer.exp_dir={fmt(str(exp_no_cfg))}",
            f"infer.input={fmt(str(sample_img))}",
            f"infer.output_dir={fmt(str(e2e_workdir['outputs'] / 'out_c1'))}",
            "infer.mixed_precision='no'",
        ],
        check=False,
    )
    assert proc_c1.returncode != 0

    # (c2) Mode 1 exp_dir missing all weights
    exp_no_w = bias_tagged_exp_dir_factory(e2e_workdir["work"] / "exp_no_w", scale=2, gt_size=32)
    for weight_file in list(exp_no_w["weights_dir"].iterdir()):
        if weight_file.is_file():
            weight_file.unlink()
    proc_c2 = infer_cli_runner(
        [
            f"infer.exp_dir={fmt(str(exp_no_w['exp_dir']))}",
            "infer.weights_path=null",
            f"infer.input={fmt(str(sample_img))}",
            f"infer.output_dir={fmt(str(e2e_workdir['outputs'] / 'out_c2'))}",
            "infer.mixed_precision='no'",
        ],
        check=False,
    )
    assert proc_c2.returncode != 0

    # (d) Invalid save_format=bmp
    proc_d = infer_cli_runner(
        [
            f"infer.exp_dir={fmt(str(exp_valid['exp_dir']))}",
            f"infer.input={fmt(str(sample_img))}",
            f"infer.output_dir={fmt(str(e2e_workdir['outputs'] / 'out_d'))}",
            "infer.save_format=bmp",
            "infer.mixed_precision='no'",
            "infer.tile.overlap=4",
        ],
        check=False,
    )
    assert proc_d.returncode != 0


def test_t2_f08_tools_infer_bad_json_entry_and_non_recursive_dir_scan(
    e2e_workdir: dict[str, Path],
    bias_tagged_exp_dir_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F38: tools/infer.py rejects JSON entries without lq/gt and honors recursive=false."""
    from utils.image import load_image

    fmt = hydra_value_formatter
    exp_info = bias_tagged_exp_dir_factory(
        e2e_workdir["work"] / "exp_io_edge", scale=2, gt_size=32
    )
    common = [
        f"infer.exp_dir={fmt(str(exp_info['exp_dir']))}",
        "infer.mixed_precision='no'",
        "infer.tile.enabled=true",
        "infer.tile.size=auto",
        "infer.tile.overlap=4",
    ]

    # Part A: JSON entry lacking both 'lq' and 'gt' exits non-zero
    bad_json = e2e_workdir["data"] / "bad_infer.json"
    bad_json.write_text(json.dumps([{"label": "no_img"}]), encoding="utf-8")
    proc_bad = infer_cli_runner(
        [
            *common,
            f"infer.input={fmt(str(bad_json))}",
            f"infer.output_dir={fmt(str(e2e_workdir['outputs'] / 'bad_json'))}",
        ],
        check=False,
    )
    assert proc_bad.returncode != 0

    # Part B: Non-recursive directory scan ignores nested subdirs and non-image files
    scan_dir = e2e_workdir["data"] / "nonrec_dir"
    synthetic_image_factory(scan_dir / "top_a.png", width=16, height=18, seed=1)
    synthetic_image_factory(scan_dir / "top_b.jpg", width=18, height=16, seed=2)
    (scan_dir / "notes.txt").write_text("not an image", encoding="utf-8")
    synthetic_image_factory(scan_dir / "nested/sub_a.png", width=16, height=16, seed=3)
    synthetic_image_factory(scan_dir / "nested/deep/sub_b.png", width=16, height=16, seed=4)

    out_nonrec = e2e_workdir["outputs"] / "nonrec_out"
    infer_cli_runner(
        [
            *common,
            f"infer.input={fmt(str(scan_dir))}",
            f"infer.output_dir={fmt(str(out_nonrec))}",
            "infer.recursive=false",
            "infer.save_format=png",
        ],
        check=True,
    )

    out_rel = sorted(p.relative_to(out_nonrec).as_posix() for p in out_nonrec.rglob("*") if p.is_file())
    assert out_rel == ["top_a.png", "top_b.png"]
    assert not (out_nonrec / "nested").exists()
    assert load_image(out_nonrec / "top_a.png").shape == (3, 36, 32)
    assert load_image(out_nonrec / "top_b.png").shape == (3, 32, 36)


def test_t2_f09_tools_infer_2proc_cpu_single_image_idle_rank(
    repo_root: Path,
    e2e_workdir: dict[str, Path],
    bias_tagged_exp_dir_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    torchrun_cpu_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F39, F40: 2-proc CPU distributed tools/infer.py handles 1 image (idle Rank 1) with zero deadlock."""
    from utils.image import load_image

    fmt = hydra_value_formatter
    exp_info = bias_tagged_exp_dir_factory(
        e2e_workdir["work"] / "exp_dist_1img",
        scale=2,
        gt_size=32,
        include_best=True,
        include_ema=True,
    )
    in_dir = e2e_workdir["data"] / "single_dist"
    synthetic_image_factory(
        in_dir / "only_one.png",
        width=20,
        height=18,
        pattern="solid",
        color=(0, 0, 0),
    )
    out_dir = e2e_workdir["outputs"] / "dist_1img"

    proc = torchrun_cpu_runner(
        repo_root / "tools" / "infer.py",
        [
            f"infer.exp_dir={fmt(str(exp_info['exp_dir']))}",
            f"infer.input={fmt(str(in_dir))}",
            f"infer.output_dir={fmt(str(out_dir))}",
            "infer.mixed_precision='no'",
            "infer.tile.enabled=true",
            "infer.tile.size=auto",
            "infer.tile.overlap=4",
        ],
        nproc_per_node=2,
        timeout=120,
        check=True,
    )
    assert proc.returncode == 0

    out_pngs = sorted(out_dir.rglob("*.png"))
    assert [p.name for p in out_pngs] == ["only_one.png"]
    out_t = load_image(out_pngs[0])
    assert out_t.shape == (3, 36, 40)
    assert out_t.mean().item() == pytest.approx(exp_info["expected_means"]["best"], abs=0.02)

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert "img/s" in combined, f"Expected throughput summary with 'img/s':\n{combined}"
    assert re.search(r"(?<!\d)1(?!\d)", combined), (
        f"Expected image count '1' in throughput output:\n{combined}"
    )
