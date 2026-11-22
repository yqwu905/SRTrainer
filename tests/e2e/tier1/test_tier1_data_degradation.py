"""Tier 1 E2E tests — Area C: Mixed Data Pipeline & GPU Masked Real-ESRGAN Degradation.

Covers features F02, F21-F23, F25-F29 and User Requirements #1 & #2.
All non-stdlib imports (besides ``pytest``) are deferred inside test functions.
"""

from __future__ import annotations

import inspect
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import pytest


def _make_kernel_generator(kernel_size: int = 9, *, disable_sinc: bool = False) -> Any:
    """Opaque-box helper instantiating KernelGenerator across signature variants."""
    from data.degradation import KernelGenerator

    params = inspect.signature(KernelGenerator.__init__).parameters
    kernel_range = [k for k in (7, kernel_size) if k <= kernel_size] or [kernel_size]
    cfg: dict[str, Any] = {
        "kernel_size": kernel_size,
        "kernel_range": kernel_range,
    }
    if disable_sinc:
        cfg.update({"sinc_prob": 0.0, "sinc_prob2": 0.0, "final_sinc_prob": 0.0})

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if "kernel_size" in params:
        kwargs = dict(cfg) if has_var_kw else {k: v for k, v in cfg.items() if k in params}
        return KernelGenerator(**kwargs)
    if "kernel_cfg" in params:
        return KernelGenerator(kernel_cfg=cfg)
    if "cfg" in params:
        return KernelGenerator(cfg=cfg)
    if has_var_kw:
        return KernelGenerator(**cfg)
    return KernelGenerator()


def _sample_kernels(gen: Any) -> tuple[Any, Any, Any]:
    """Invoke KernelGenerator and unpack (kernel1, kernel2, sinc_kernel)."""
    if callable(gen):
        res = gen()
    elif hasattr(gen, "generate"):
        res = gen.generate()
    else:
        res = gen.sample()
    if isinstance(res, Mapping):
        return res["kernel1"], res["kernel2"], res["sinc_kernel"]
    k1, k2, sinc = res
    return k1, k2, sinc


def test_t1_c01_data_package_reexports_and_manifest_autodetect(
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_single_json_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
    validate_sources_caller: Callable[..., Any],
) -> None:
    """F02, F21, Reqs #1/#2: Short-path re-exports and per-sample gt_only/paired auto-detection."""
    import data as data_pkg
    import data.degradation as deg_pkg

    expected_data = (
        "MixedSRDataset",
        "ValImageDataset",
        "val_collate_fn",
        "DistributedWeightedInfiniteSampler",
        "load_train_manifest",
        "load_val_manifest",
        "validate_sources_config",
        "TrainSample",
        "ValSample",
        "SourceStats",
    )
    for sym in expected_data:
        assert hasattr(data_pkg, sym), f"Missing data.{sym}"

    expected_deg = (
        "RealESRGANDegrader",
        "DiffJPEG",
        "USMSharp",
        "TrainingPool",
        "KernelGenerator",
        "circular_lowpass_kernel",
        "random_mixed_kernels",
        "filter2d",
    )
    for sym in expected_deg:
        assert hasattr(deg_pkg, sym), f"Missing data.degradation.{sym}"

    gt_src = gt_only_train_source_factory(name="pure_gt", n_samples=2, gt_h=32, gt_w=32, ratio=0.5)
    pair_src = paired_train_source_factory(
        name="pure_pair", n_samples=2, lq_h=16, lq_w=16, scale=2, ratio=0.5
    )
    mixed_info = mixed_single_json_train_source_factory(
        name="single_mixed", lq_h=16, lq_w=16, scale=2, ratio=1.0
    )

    # User Req #2: Zero `mode` key in any source config.
    for src in (gt_src, pair_src, mixed_info["source"]):
        assert "mode" not in src
    validate_sources_caller([gt_src, pair_src, mixed_info["source"]], has_degradation=True)

    # Pure gt_only manifest -> all need_degradation == True.
    ds_gt = mixed_dataset_factory([gt_src], gt_size=32, scale=2, has_degradation=True)
    assert len(ds_gt) == 2
    assert [bool(ds_gt[i]["need_degradation"].item()) for i in range(len(ds_gt))] == [True, True]

    # Pure paired manifest -> all need_degradation == False.
    ds_pair = mixed_dataset_factory([pair_src], gt_size=32, scale=2, has_degradation=True)
    assert len(ds_pair) == 2
    assert [bool(ds_pair[i]["need_degradation"].item()) for i in range(len(ds_pair))] == [
        False,
        False,
    ]

    # Single JSON mixing paired + absent/null/"" lq + .png/.jpg/.webp.
    ds_mixed = mixed_dataset_factory(
        [mixed_info["source"]], gt_size=32, scale=2, has_degradation=True
    )
    assert len(ds_mixed) == 6
    observed = [bool(ds_mixed[i]["need_degradation"].item()) for i in range(len(ds_mixed))]
    assert observed == mixed_info["expected_need_degradation"]


def test_t1_c02_mixed_dataset_indexing_placeholders_and_default_collate(
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
) -> None:
    """F21, F23: Tuple & flat indexing, zero placeholders, normalized kernels, default collate."""
    import torch
    from torch.utils.data import DataLoader

    gt_src = gt_only_train_source_factory(name="gt_src", n_samples=2, gt_h=48, gt_w=48, ratio=0.5)
    pair_src = paired_train_source_factory(
        name="pair_src", n_samples=2, lq_h=24, lq_w=24, scale=2, ratio=0.5
    )

    ds = mixed_dataset_factory([gt_src, pair_src], gt_size=32, scale=2, has_degradation=True)
    assert len(ds) == 4

    # Tuple indexing: (0, 0) is gt_only, (1, 0) is paired.
    item_gt = ds[(0, 0)]
    assert item_gt["gt"].shape == (3, 32, 32) and item_gt["gt"].dtype == torch.float32
    assert float(item_gt["gt"].min()) >= 0.0 and float(item_gt["gt"].max()) <= 1.0
    assert item_gt["lq"].shape == (3, 16, 16) and item_gt["lq"].dtype == torch.float32
    assert torch.equal(item_gt["lq"], torch.zeros_like(item_gt["lq"]))
    assert item_gt["need_degradation"].dtype == torch.bool and item_gt["need_degradation"].ndim == 0
    assert bool(item_gt["need_degradation"].item()) is True
    assert item_gt["source_id"].dtype == torch.int64 and int(item_gt["source_id"].item()) == 0
    for kname in ("kernel1", "kernel2", "sinc_kernel"):
        k = item_gt[kname]
        assert k.ndim == 2 and k.shape[0] == k.shape[1] and k.dtype == torch.float32
        assert torch.isfinite(k).all()
        assert k.sum().item() == pytest.approx(1.0, abs=1e-4)

    item_pair = ds[(1, 0)]
    assert item_pair["gt"].shape == (3, 32, 32) and item_pair["lq"].shape == (3, 16, 16)
    assert item_pair["lq"].abs().sum().item() > 0.0
    assert bool(item_pair["need_degradation"].item()) is False
    assert int(item_pair["source_id"].item()) == 1
    for kname in ("kernel1", "kernel2", "sinc_kernel"):
        k = item_pair[kname]
        assert k.shape == item_gt[kname].shape and k.dtype == torch.float32
        assert torch.equal(k, torch.zeros_like(k))

    # Flat integer indexing across sources.
    assert int(ds[0]["source_id"].item()) == 0
    assert int(ds[2]["source_id"].item()) == 1

    # Default PyTorch DataLoader collate handles mixed batch cleanly.
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["gt"].shape == (4, 3, 32, 32)
    assert batch["lq"].shape == (4, 3, 16, 16)
    assert batch["need_degradation"].tolist() == [True, True, False, False]
    assert batch["source_id"].tolist() == [0, 0, 1, 1]
    assert batch["kernel1"].shape[0] == 4


def test_t1_c03_mixed_dataset_pure_paired_without_degradation(
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
) -> None:
    """F21, F23: Pure-paired MixedSRDataset runs cleanly with degradation disabled."""
    import torch
    from torch.utils.data import DataLoader

    pair_src = paired_train_source_factory(
        name="paired_only", n_samples=4, lq_h=24, lq_w=24, scale=2, ratio=1.0
    )
    ds = mixed_dataset_factory([pair_src], gt_size=32, scale=2, has_degradation=False)
    assert len(ds) == 4

    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["gt"].shape == (2, 3, 32, 32) and batch["gt"].dtype == torch.float32
    assert batch["lq"].shape == (2, 3, 16, 16) and batch["lq"].dtype == torch.float32
    assert batch["need_degradation"].tolist() == [False, False]
    assert float(batch["lq"].min()) >= 0.0 and float(batch["lq"].max()) <= 1.0
    assert batch["lq"].abs().sum().item() > 0.0


def test_t1_c04_paired_aligned_crop_and_synchronized_augmentations(
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
) -> None:
    """F22: Paired random crop aligns strictly by scale and synchronizes hflip/vflip/rot90."""
    import torch

    from utils.image import load_image

    scale = 2
    src = paired_train_source_factory(
        name="aligned_aug", n_samples=1, lq_h=24, lq_w=28, scale=scale, ratio=1.0
    )
    ds_aug = mixed_dataset_factory(
        [src],
        gt_size=32,
        scale=scale,
        augment={"hflip": True, "vflip": True, "rot90": True},
        has_degradation=False,
    )

    unique_signatures: set[tuple[float, float, float, float]] = set()
    for _ in range(32):
        sample = ds_aug[(0, 0)]
        gt = sample["gt"]
        lq = sample["lq"]
        assert gt.shape == (3, 32, 32)
        assert lq.shape == (3, 16, 16)
        upsampled_lq = lq.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
        max_diff = (gt - upsampled_lq).abs().max().item()
        assert max_diff < 1e-5, f"Paired crop/augment misalignment detected: max_diff={max_diff}"
        sig = (
            round(float(lq[0, 0, 0].item()), 4),
            round(float(lq[0, 0, -1].item()), 4),
            round(float(lq[0, -1, 0].item()), 4),
            round(float(lq[0, -1, -1].item()), 4),
        )
        unique_signatures.add(sig)

    assert len(unique_signatures) >= 2, "Expected spatial/orientation diversity across 32 draws"

    # With all augmentations disabled on exact-size image, pixels must remain unchanged.
    exact_src = paired_train_source_factory(
        name="exact_noaug", n_samples=1, lq_h=16, lq_w=16, scale=scale, ratio=1.0
    )
    ds_noaug = mixed_dataset_factory(
        [exact_src],
        gt_size=32,
        scale=scale,
        augment={"hflip": False, "vflip": False, "rot90": False},
        has_degradation=False,
    )
    item = ds_noaug[(0, 0)]
    root = exact_src["root_dir"]
    orig_gt = load_image(f"{root}/hr/0000.png")
    orig_lq = load_image(f"{root}/lr/0000.png")
    assert torch.allclose(item["gt"], orig_gt, atol=1e-5)
    assert torch.allclose(item["lq"], orig_lq, atol=1e-5)


def test_t1_c05_distributed_weighted_sampler_ratio_decoupled_from_size(
    weighted_sampler_factory: Callable[..., Any],
) -> None:
    """F25: Sampler normalizes ratios decoupled from dataset sizes and streams infinitely."""
    source_sizes = [1000, 2]
    sampler = weighted_sampler_factory(
        source_sizes=source_sizes,
        ratios=[3.0, 7.0],
        batch_size_per_gpu=10,
        world_size=1,
        rank=0,
        base_seed=42,
    )

    it = iter(sampler)
    n_draws = 10_000
    draws = [next(it) for _ in range(n_draws)]

    count_s1 = 0
    for sid, idx in draws:
        assert 0 <= sid < len(source_sizes)
        assert 0 <= idx < source_sizes[sid]
        if sid == 1:
            count_s1 += 1

    freq_s1 = count_s1 / n_draws
    assert abs(freq_s1 - 0.70) <= 0.025, (
        f"Expected source 1 frequency ~0.70 decoupled from sizes {source_sizes}, got {freq_s1:.4f}"
    )

    # Equivalent normalized ratios [0.3, 0.7] must reproduce identical stream.
    sampler_norm = weighted_sampler_factory(
        source_sizes=source_sizes,
        ratios=[0.3, 0.7],
        batch_size_per_gpu=10,
        world_size=1,
        rank=0,
        base_seed=42,
    )
    it_norm = iter(sampler_norm)
    draws_norm = [next(it_norm) for _ in range(100)]
    assert draws_norm == draws[:100]


def test_t1_c06_distributed_weighted_sampler_determinism_sharding_and_o1_resume(
    weighted_sampler_factory: Callable[..., Any],
) -> None:
    """F25: Seed determinism, disjoint multi-rank sharding, and O(1) set_start_micro_step."""
    sizes = [50, 20]
    ratios = [0.6, 0.4]

    # Determinism across identical seeds vs divergence on different seed.
    s_a = weighted_sampler_factory(sizes, ratios, batch_size_per_gpu=4, world_size=1, base_seed=77)
    s_b = weighted_sampler_factory(sizes, ratios, batch_size_per_gpu=4, world_size=1, base_seed=77)
    s_c = weighted_sampler_factory(sizes, ratios, batch_size_per_gpu=4, world_size=1, base_seed=78)
    it_a, it_b, it_c = iter(s_a), iter(s_b), iter(s_c)
    seq_a = [next(it_a) for _ in range(24)]
    seq_b = [next(it_b) for _ in range(24)]
    seq_c = [next(it_c) for _ in range(24)]
    assert seq_a == seq_b
    assert seq_a != seq_c

    # Multi-rank sharding: interleaving rank 0 & rank 1 matches world_size=1 draw of size 2*B.
    s_global = weighted_sampler_factory(
        sizes, ratios, batch_size_per_gpu=8, world_size=1, rank=0, base_seed=123
    )
    s_r0 = weighted_sampler_factory(
        sizes, ratios, batch_size_per_gpu=4, world_size=2, rank=0, base_seed=123
    )
    s_r1 = weighted_sampler_factory(
        sizes, ratios, batch_size_per_gpu=4, world_size=2, rank=1, base_seed=123
    )
    it_g, it_r0, it_r1 = iter(s_global), iter(s_r0), iter(s_r1)
    for _ in range(5):
        g_step = [next(it_g) for _ in range(8)]
        r0_step = [next(it_r0) for _ in range(4)]
        r1_step = [next(it_r1) for _ in range(4)]
        interleaved: list[tuple[int, int]] = []
        for a, b in zip(r0_step, r1_step, strict=True):
            interleaved.extend([a, b])
        assert interleaved == g_step

    # Exact O(1) resume matching uninterrupted continuation.
    resume_micro_step = 15 * 2  # global_step=15, grad_accum=2
    batch_sz = 4
    s_cont = weighted_sampler_factory(
        sizes, ratios, batch_size_per_gpu=batch_sz, world_size=2, rank=1, base_seed=999
    )
    it_cont = iter(s_cont)
    for _ in range(resume_micro_step * batch_sz):
        next(it_cont)
    expected_tail = [next(it_cont) for _ in range(10 * batch_sz)]

    s_resumed = weighted_sampler_factory(
        sizes, ratios, batch_size_per_gpu=batch_sz, world_size=2, rank=1, base_seed=999
    )
    s_resumed.set_start_micro_step(resume_micro_step)
    it_resumed = iter(s_resumed)
    resumed_tail = [next(it_resumed) for _ in range(10 * batch_sz)]
    assert resumed_tail == expected_tail

    # O(1) seek time at step 1_000_000_000 (< 0.5s).
    t0 = time.perf_counter()
    s_huge = weighted_sampler_factory(
        sizes, ratios, batch_size_per_gpu=4, world_size=1, rank=0, base_seed=42
    )
    s_huge.set_start_micro_step(1_000_000_000)
    it_huge = iter(s_huge)
    _ = [next(it_huge) for _ in range(4)]
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"set_start_micro_step(1e9) took {elapsed:.3f}s (must be O(1))"


def test_t1_c07_kernels_all_six_types_sinc_identity_and_filter2d() -> None:
    """F26: circular_lowpass_kernel, all 6 blur types, KernelGenerator, and batched filter2d."""
    import torch

    from data.degradation import (
        circular_lowpass_kernel,
        filter2d,
        random_mixed_kernels,
    )

    # circular_lowpass_kernel: symmetry, finite center, pad_to, normalized sum.
    sinc = torch.as_tensor(
        circular_lowpass_kernel(cutoff=math.pi / 3.0, kernel_size=9, pad_to=11),
        dtype=torch.float32,
    )
    assert sinc.shape == (11, 11)
    assert torch.isfinite(sinc).all()
    assert sinc.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(sinc, sinc.T, atol=1e-5)
    assert torch.allclose(sinc, torch.flip(sinc, dims=[0, 1]), atol=1e-5)

    # All 6 Real-ESRGAN blur kernel types via random_mixed_kernels.
    rmk_params = inspect.signature(random_mixed_kernels).parameters
    all_six = (
        "iso",
        "aniso",
        "generalized_iso",
        "generalized_aniso",
        "plateau_iso",
        "plateau_aniso",
    )
    for ktype in all_six:
        candidates: dict[str, Any] = {
            "kernel_list": [ktype],
            "kernel_prob": [1.0],
            "kernel_size": 9,
            "sigma_x_range": (0.5, 2.0),
            "sigma_y_range": (0.5, 2.0),
            "rotation_range": (-math.pi, math.pi),
            "betag_range": (0.5, 4.0),
            "betap_range": (1.0, 2.0),
            "noise_range": None,
        }
        kwargs = {k: v for k, v in candidates.items() if k in rmk_params}
        kern = torch.as_tensor(random_mixed_kernels(**kwargs), dtype=torch.float32)
        assert kern.shape == (9, 9)
        assert torch.isfinite(kern).all()
        assert kern.sum().item() == pytest.approx(1.0, abs=1e-4), f"Unnormalized {ktype}"

    # KernelGenerator: uniform [K, K] normalized outputs + identity pulse when sinc disabled.
    gen = _make_kernel_generator(kernel_size=9, disable_sinc=True)
    k1, k2, ksinc = _sample_kernels(gen)
    k1_t = torch.as_tensor(k1, dtype=torch.float32)
    k2_t = torch.as_tensor(k2, dtype=torch.float32)
    ksinc_t = torch.as_tensor(ksinc, dtype=torch.float32)
    assert k1_t.shape == k2_t.shape == ksinc_t.shape
    for kt in (k1_t, k2_t, ksinc_t):
        assert kt.ndim == 2 and kt.shape[0] == kt.shape[1]
        assert torch.isfinite(kt).all()
        assert kt.sum().item() == pytest.approx(1.0, abs=1e-4)

    center = ksinc_t.shape[0] // 2
    expected_pulse = torch.zeros_like(ksinc_t)
    expected_pulse[center, center] = 1.0
    assert torch.allclose(ksinc_t, expected_pulse, atol=1e-5)

    # filter2d: identity invariance, constant DC preservation, and per-sample independence.
    torch.manual_seed(5)
    img = torch.rand(2, 3, 16, 16, dtype=torch.float32)
    id_batch = expected_pulse.unsqueeze(0).repeat(2, 1, 1)
    out_id = filter2d(img, id_batch)
    assert out_id.shape == img.shape
    assert torch.allclose(out_id, img, atol=1e-5)

    const_img = torch.full((2, 3, 16, 16), 0.42, dtype=torch.float32)
    blur_k = torch.full((2, 5, 5), 1.0 / 25.0, dtype=torch.float32)
    out_dc = filter2d(const_img, blur_k)
    assert torch.allclose(out_dc, const_img, atol=1e-4)

    # Per-sample batch independence: sample 0 uses identity pulse, sample 1 uses uniform blur.
    id_5 = torch.zeros((5, 5), dtype=torch.float32)
    id_5[2, 2] = 1.0
    mixed_k = torch.stack([id_5, torch.full((5, 5), 1.0 / 25.0)], dim=0)
    twin_img = img[0:1].repeat(2, 1, 1, 1)
    out_mixed = filter2d(twin_img, mixed_k)
    assert torch.allclose(out_mixed[0], twin_img[0], atol=1e-5)
    assert (out_mixed[1] - twin_img[1]).abs().mean().item() > 0.02


def test_t1_c08_diffjpeg_batched_fp32_quality_monotonicity() -> None:
    """F27: Batched FP32 DiffJPEG preserves shape/range and degrades more at lower quality."""
    import torch

    from data.degradation import DiffJPEG

    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    checker = ((yy // 2 + xx // 2) % 2).float()
    r = 0.15 + 0.70 * checker
    g = 0.85 - 0.65 * checker
    b = 0.25 + 0.50 * ((yy // 3) % 2).float()
    textured = torch.stack([r, g, b], dim=0).unsqueeze(0).repeat(2, 1, 1, 1)

    jpeg = DiffJPEG()
    out = jpeg(textured, quality=torch.tensor([95.0, 20.0], dtype=torch.float32))
    assert out.shape == textured.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    mse_high = ((out[0] - textured[0]) ** 2).mean().item()
    mse_low = ((out[1] - textured[1]) ** 2).mean().item()
    assert mse_low > 1e-5
    assert mse_high < mse_low, f"Expected MSE(q=95)={mse_high} < MSE(q=20)={mse_low}"

    const_img = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    out_const = jpeg(const_img, quality=torch.tensor([75.0], dtype=torch.float32))
    assert torch.allclose(out_const, const_img, atol=2e-2)


def test_t1_c09_realesrgan_degrader_masked_execution_and_usm() -> None:
    """F28: Masked RealESRGANDegrader leaves paired (~mask) bit-exact and applies USM only on mask."""
    import torch

    from data.degradation import RealESRGANDegrader, USMSharp

    torch.manual_seed(11)
    # Textured 64x64 GT so BasicSR default USMSharp(radius=50) reflect pad (50 < 64) succeeds
    # and USM sharpening produces measurable contrast shift.
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    wave = 0.5 + 0.35 * torch.sin(xx.float() * 0.6) * torch.cos(yy.float() * 0.6)
    gt = wave.unsqueeze(0).unsqueeze(0).repeat(4, 3, 1, 1).clamp(0.0, 1.0).contiguous()
    gt = (gt + 0.05 * torch.rand_like(gt)).clamp(0.0, 1.0)

    mask = torch.tensor([True, False, True, False], dtype=torch.bool)
    lq = torch.zeros(4, 3, 32, 32, dtype=torch.float32)
    lq[~mask] = torch.rand(2, 3, 32, 32, dtype=torch.float32)

    pulse = torch.zeros(7, 7, dtype=torch.float32)
    pulse[3, 3] = 1.0
    k1 = torch.zeros(4, 7, 7, dtype=torch.float32)
    k2 = torch.zeros(4, 7, 7, dtype=torch.float32)
    sinc = torch.zeros(4, 7, 7, dtype=torch.float32)
    k1[mask] = torch.full((2, 7, 7), 1.0 / 49.0)
    k2[mask] = pulse.unsqueeze(0).repeat(2, 1, 1)
    sinc[mask] = pulse.unsqueeze(0).repeat(2, 1, 1)

    batch_no_usm = {
        "gt": gt.clone(),
        "lq": lq.clone(),
        "need_degradation": mask.clone(),
        "kernel1": k1.clone(),
        "kernel2": k2.clone(),
        "sinc_kernel": sinc.clone(),
    }
    # Clone BEFORE calling degrader so in-place scatter checks on ~mask are genuine.
    orig_lq = batch_no_usm["lq"].clone()
    orig_gt = batch_no_usm["gt"].clone()

    degrader = RealESRGANDegrader(scale=2, use_usm=False, queue_size=0)
    out = degrader(batch_no_usm)
    assert out["lq"].shape == (4, 3, 32, 32)
    assert out["gt"].shape == (4, 3, 64, 64)
    assert out["lq"].dtype == torch.float32
    assert not out["lq"].requires_grad and not out["gt"].requires_grad
    assert torch.equal(out["lq"][~mask], orig_lq[~mask])
    assert torch.equal(out["gt"][~mask], orig_gt[~mask])
    assert torch.isfinite(out["lq"][mask]).all()
    assert float(out["lq"][mask].min()) >= 0.0 and float(out["lq"][mask].max()) <= 1.0
    assert out["lq"][mask].abs().sum().item() > 0.0

    # With use_usm=True: USM applies ONLY to gt[mask], leaving paired (~mask) bit-exact.
    batch_usm = {
        "gt": gt.clone(),
        "lq": lq.clone(),
        "need_degradation": mask.clone(),
        "kernel1": k1.clone(),
        "kernel2": k2.clone(),
        "sinc_kernel": sinc.clone(),
    }
    orig_lq_u = batch_usm["lq"].clone()
    orig_gt_u = batch_usm["gt"].clone()

    degrader_usm = RealESRGANDegrader(scale=2, use_usm=True, queue_size=0)
    out_usm = degrader_usm(batch_usm)
    assert torch.equal(out_usm["lq"][~mask], orig_lq_u[~mask])
    assert torch.equal(out_usm["gt"][~mask], orig_gt_u[~mask])
    assert float(out_usm["gt"][mask].min()) >= 0.0 and float(out_usm["gt"][mask].max()) <= 1.0
    assert not torch.equal(out_usm["gt"][mask], orig_gt_u[mask])

    # Standalone USMSharp contract.
    usm = USMSharp()
    const_x = torch.full((1, 3, 64, 64), 0.5, dtype=torch.float32)
    assert torch.allclose(usm(const_x), const_x, atol=1e-5)
    sharp_x = usm(gt[:1])
    assert sharp_x.shape == gt[:1].shape
    assert float(sharp_x.min()) >= 0.0 and float(sharp_x.max()) <= 1.0
    assert (sharp_x - gt[:1]).abs().max().item() > 1e-4


def test_t1_c10_realesrgan_degrader_training_pool_pairing_and_mixing() -> None:
    """F28, F29: TrainingPool preserves exact (lq, gt) pairing while mixing across steps."""
    import torch

    from data.degradation import RealESRGANDegrader, TrainingPool

    torch.manual_seed(17)
    pool = TrainingPool(queue_size=8)

    def _call_pool(p: Any, lq_t: Any, gt_t: Any) -> tuple[Any, Any]:
        if callable(p):
            return p(lq_t, gt_t)
        return p.dequeue_and_enqueue(lq_t, gt_t)

    mixed_observed = False
    for step in range(10):
        t0 = 0.05 + 0.04 * (2 * step)
        t1 = 0.05 + 0.04 * (2 * step + 1)
        lq_in = torch.stack(
            [torch.full((3, 16, 16), t0), torch.full((3, 16, 16), t1)], dim=0
        )
        gt_in = torch.stack(
            [torch.full((3, 32, 32), t0), torch.full((3, 32, 32), t1)], dim=0
        )
        lq_out, gt_out = _call_pool(pool, lq_in, gt_in)
        assert lq_out.shape == (2, 3, 16, 16)
        assert gt_out.shape == (2, 3, 32, 32)
        for i in range(2):
            assert abs(lq_out[i].mean().item() - gt_out[i].mean().item()) < 1e-5
        if step >= 4 and not torch.allclose(lq_out, lq_in, atol=1e-5):
            mixed_observed = True
    assert mixed_observed, "TrainingPool did not mix pairs across steps after filling"

    # End-to-end post-degradation TrainingPool inside RealESRGANDegrader.
    degrader_pool = RealESRGANDegrader(scale=2, use_usm=False, queue_size=8)
    deg_mixed = False
    for step in range(10):
        t0 = 0.06 + 0.04 * (2 * step)
        t1 = 0.06 + 0.04 * (2 * step + 1)
        lq_in = torch.stack(
            [torch.full((3, 16, 16), t0), torch.full((3, 16, 16), t1)], dim=0
        )
        gt_in = torch.stack(
            [torch.full((3, 32, 32), t0), torch.full((3, 32, 32), t1)], dim=0
        )
        batch = {
            "lq": lq_in.clone(),
            "gt": gt_in.clone(),
            "need_degradation": torch.tensor([False, False], dtype=torch.bool),
            "kernel1": torch.zeros(2, 7, 7),
            "kernel2": torch.zeros(2, 7, 7),
            "sinc_kernel": torch.zeros(2, 7, 7),
        }
        out = degrader_pool(batch)
        assert out["lq"].shape == (2, 3, 16, 16)
        assert out["gt"].shape == (2, 3, 32, 32)
        for i in range(2):
            assert abs(out["lq"][i].mean().item() - out["gt"][i].mean().item()) < 1e-5
        if step >= 4 and not torch.allclose(out["lq"], lq_in, atol=1e-5):
            deg_mixed = True
    assert deg_mixed, "RealESRGANDegrader(queue_size=8) did not mix pairs across steps"
