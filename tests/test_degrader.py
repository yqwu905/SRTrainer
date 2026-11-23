"""Unit tests for GPU masked Real-ESRGAN degradation pipeline: kernels, DiffJPEG, filter2d, USMSharp, TrainingPool, RealESRGANDegrader."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import data.degradation as deg_pkg
import data.degradation.realesrgan as realesrgan_mod
from data.degradation import (
    DiffJPEG,
    KernelGenerator,
    RealESRGANDegrader,
    TrainingPool,
    USMSharp,
    circular_lowpass_kernel,
    filter2d,
    random_mixed_kernels,
)
from data.degradation.kernels import VALID_KERNEL_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


def _make_textured_image(b: int = 1, h: int = 64, w: int = 64, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0.0, 1.0, h).view(1, 1, h, 1)
    xs = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w)
    pattern = 0.5 + 0.35 * torch.sin(2.0 * math.pi * 6.0 * xs) * torch.cos(
        2.0 * math.pi * 5.0 * ys
    )
    noise = 0.1 * torch.randn((b, 3, h, w), generator=gen)
    img = pattern.expand(b, 3, h, w) + noise
    return img.clamp(0.0, 1.0).contiguous()


def _make_batch(
    b: int = 4,
    gt_h: int = 64,
    gt_w: int = 64,
    scale: int = 4,
    mask: list[bool] | None = None,
    k_size: int = 21,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    if mask is None:
        mask = [True] * b
    gt = _make_textured_image(b=b, h=gt_h, w=gt_w, seed=seed)
    lq = _make_textured_image(b=b, h=gt_h // scale, w=gt_w // scale, seed=seed + 100)
    mask_t = torch.tensor(mask, dtype=torch.bool)
    lq[mask_t] = 0.0

    kgen = KernelGenerator({"kernel_size": k_size})
    rng = np.random.default_rng(seed + 7)
    k1_list, k2_list, sk_list = [], [], []
    for need in mask:
        if need:
            k1, k2, sk = kgen.sample(rng=rng)
        else:
            k1 = torch.zeros((k_size, k_size), dtype=torch.float32)
            k2 = torch.zeros((k_size, k_size), dtype=torch.float32)
            sk = torch.zeros((k_size, k_size), dtype=torch.float32)
        k1_list.append(k1)
        k2_list.append(k2)
        sk_list.append(sk)

    return {
        "gt": gt,
        "lq": lq,
        "need_degradation": mask_t,
        "kernel1": torch.stack(k1_list, dim=0),
        "kernel2": torch.stack(k2_list, dim=0),
        "sinc_kernel": torch.stack(sk_list, dim=0),
    }


def test_degradation_package_all_exports() -> None:
    expected = {
        "RealESRGANDegrader",
        "DiffJPEG",
        "USMSharp",
        "TrainingPool",
        "KernelGenerator",
        "circular_lowpass_kernel",
        "random_mixed_kernels",
        "filter2d",
    }
    assert set(deg_pkg.__all__) == expected
    assert len(deg_pkg.__all__) == 8
    for name in expected:
        assert hasattr(deg_pkg, name)


@pytest.mark.parametrize("k_size", [7, 13, 21])
@pytest.mark.parametrize("cutoff", [math.pi / 5.0, math.pi / 3.0, math.pi])
def test_circular_lowpass_kernel_math_and_padding(k_size: int, cutoff: float) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        k = circular_lowpass_kernel(cutoff=cutoff, kernel_size=k_size, pad_to=0)
    assert len(caught) == 0
    assert k.shape == (k_size, k_size)
    assert k.dtype == np.float64
    assert np.isfinite(k).all()
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-12)
    # 4-way symmetry and peak at center
    assert np.allclose(k, k[::-1, :])
    assert np.allclose(k, k[:, ::-1])
    assert np.allclose(k, k.T)
    c = k_size // 2
    assert k[c, c] == pytest.approx(float(k.max()), abs=1e-12)

    # Padded to 21
    k_pad = circular_lowpass_kernel(cutoff=cutoff, kernel_size=k_size, pad_to=21)
    assert k_pad.shape == (21, 21)
    assert float(k_pad.sum()) == pytest.approx(1.0, abs=1e-12)
    if k_size < 21:
        pad = (21 - k_size) // 2
        assert np.all(k_pad[:pad, :] == 0.0)
        assert np.all(k_pad[-pad:, :] == 0.0)
        assert np.all(k_pad[:, :pad] == 0.0)
        assert np.all(k_pad[:, -pad:] == 0.0)


def test_circular_lowpass_kernel_invalid_args() -> None:
    with pytest.raises(ValueError):
        circular_lowpass_kernel(math.pi / 3, kernel_size=8)
    with pytest.raises(ValueError):
        circular_lowpass_kernel(0.0, kernel_size=7)
    with pytest.raises(ValueError):
        circular_lowpass_kernel(-1.0, kernel_size=7)
    with pytest.raises(ValueError):
        circular_lowpass_kernel(math.pi / 3, kernel_size=7, pad_to=10)
    with pytest.raises(ValueError):
        circular_lowpass_kernel(math.pi / 3, kernel_size=11, pad_to=7)


@pytest.mark.parametrize("kernel_type", VALID_KERNEL_TYPES)
def test_random_mixed_kernels_all_six_types(kernel_type: str) -> None:
    rng = np.random.default_rng(42)
    k = random_mixed_kernels(
        kernel_list=[kernel_type],
        kernel_prob=[1.0],
        kernel_size=15,
        sigma_x_range=(0.5, 2.5),
        sigma_y_range=(0.5, 2.5),
        betag_range=(0.5, 4.0),
        betap_range=(1.0, 2.0),
        rng=rng,
    )
    assert k.shape == (15, 15)
    assert np.isfinite(k).all()
    assert float(k.min()) >= 0.0
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-12)
    if kernel_type.endswith("_iso") or kernel_type == "iso":
        assert np.allclose(k, k.T, atol=1e-12)
        assert np.allclose(k, k[::-1, :], atol=1e-12)

    # With multiplicative noise
    k_noisy = random_mixed_kernels(
        kernel_list=[kernel_type],
        kernel_prob=[1.0],
        kernel_size=15,
        noise_range=(0.9, 1.1),
        rng=rng,
    )
    assert k_noisy.shape == (15, 15)
    assert np.isfinite(k_noisy).all()
    assert float(k_noisy.sum()) == pytest.approx(1.0, abs=1e-12)


def test_random_mixed_kernels_invalid_args() -> None:
    with pytest.raises(ValueError):
        random_mixed_kernels(["unknown_type"], [1.0], kernel_size=15)
    with pytest.raises(ValueError):
        random_mixed_kernels(["iso"], [1.0], kernel_size=14)
    with pytest.raises(ValueError):
        random_mixed_kernels(["iso", "aniso"], [1.0], kernel_size=15)
    with pytest.raises(ValueError):
        random_mixed_kernels(["iso", "aniso"], [0.5, 0.8], kernel_size=15)
    with pytest.raises(ValueError):
        random_mixed_kernels(["iso"], [1.0], kernel_size=15, sigma_x_range=(0.0, 2.0))


@pytest.mark.parametrize(
    "kernel_type",
    ("generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso"),
)
def test_random_mixed_kernels_extreme_sigma_beta_no_warnings(kernel_type: str) -> None:
    rng = np.random.default_rng(123)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        k = random_mixed_kernels(
            kernel_list=[kernel_type],
            kernel_prob=[1.0],
            kernel_size=21,
            sigma_x_range=(0.01, 0.01),
            sigma_y_range=(0.01, 0.01),
            betag_range=(50.0, 50.0),
            betap_range=(50.0, 50.0),
            rng=rng,
        )
    assert k.shape == (21, 21)
    assert np.isfinite(k).all()
    assert float(k.min()) >= 0.0
    assert float(k.sum()) == pytest.approx(1.0, abs=1e-12)


def test_kernel_generator_variants_and_hydra_cfg() -> None:
    # Default (None)
    gen_default = KernelGenerator(None)
    assert gen_default.kernel_size == 21
    k1, k2, sk = gen_default.sample(rng=np.random.default_rng(1))
    for kt in (k1, k2, sk):
        assert kt.shape == (21, 21) and kt.dtype == torch.float32 and kt.is_contiguous()
        assert torch.isfinite(kt).all()
        assert kt.sum().item() == pytest.approx(1.0, abs=1e-5)

    # Partial config + callable interface + np.random global path (rng=None)
    gen_small = KernelGenerator({"kernel_size": 15, "final_sinc_prob": 0.0})
    assert gen_small.kernel_size == 15
    assert gen_small.kernel_range == (7, 9, 11, 13, 15)
    _, _, pulse = gen_small()
    expected_pulse = torch.zeros((15, 15), dtype=torch.float32)
    expected_pulse[7, 7] = 1.0
    assert torch.equal(pulse, expected_pulse)

    # final_sinc_prob=1.0 produces genuine non-pulse sinc kernel
    gen_sinc = KernelGenerator({"kernel_size": 15, "sinc_prob": 1.0, "sinc_prob2": 1.0, "final_sinc_prob": 1.0, "kernel_range": [7]})
    sk1, sk2, skf = gen_sinc.sample(rng=np.random.default_rng(2))
    assert not torch.equal(skf, expected_pulse)
    for kt in (sk1, sk2, skf):
        assert kt.sum().item() == pytest.approx(1.0, abs=1e-5)

    # kernel_range >= 13 branch
    gen_sinc_15 = KernelGenerator({"kernel_size": 15, "sinc_prob": 1.0, "kernel_range": [15]})
    k1_15, _, _ = gen_sinc_15.sample(rng=np.random.default_rng(3))
    assert k1_15.shape == (15, 15) and k1_15.sum().item() == pytest.approx(1.0, abs=1e-5)

    # Full Hydra-composed kernel_cfg
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(config_name="train")
    gen_hydra = KernelGenerator(cfg.data.degradation.kernel_cfg)
    assert gen_hydra.kernel_size == 21
    hk1, hk2, hsk = gen_hydra.sample(rng=np.random.default_rng(4))
    assert hk1.shape == hk2.shape == hsk.shape == (21, 21)

    # Invalid kernel_range & OmegaConf missing
    with pytest.raises(ValueError):
        KernelGenerator({"kernel_size": 15, "kernel_range": [7, 17]})
    with pytest.raises(ValueError):
        KernelGenerator({"kernel_size": 15, "kernel_range": [8]})
    with pytest.raises(ValueError):
        KernelGenerator(OmegaConf.create({"kernel_size": "???"}))


def test_diffjpeg_quality_shapes_boundaries_and_amp() -> None:
    jpeger = DiffJPEG(differentiable=False)
    assert sum(p.numel() for p in jpeger.parameters()) == 0

    # (a) Quality monotonicity on textured 64x64 image
    img = _make_textured_image(b=1, h=64, w=64, seed=11)
    q10 = jpeger(img, quality=10.0)
    q50 = jpeger(img, quality=50.0)
    q95 = jpeger(img, quality=95.0)
    mse10 = float(torch.mean((img - q10) ** 2))
    mse50 = float(torch.mean((img - q50) ** 2))
    mse95 = float(torch.mean((img - q95) ** 2))
    assert mse10 > mse50 > mse95

    # (b) Per-sample [B] quality vector matches individual calls
    img2 = _make_textured_image(b=2, h=64, w=64, seed=12)
    batched = jpeger(img2, quality=torch.tensor([20.0, 90.0]))
    single0 = jpeger(img2[0:1], quality=20.0)
    single1 = jpeger(img2[1:2], quality=90.0)
    assert torch.allclose(batched[0:1], single0, atol=1e-6)
    assert torch.allclose(batched[1:2], single1, atol=1e-6)

    # (c) Boundary clamping (including quality==100 division-by-zero protection) & out-of-range inputs
    oor_img = img * 1.4 - 0.2
    for q in (100.0, 150.0, 0.0, -5.0, 1.0):
        out = jpeger(oor_img, quality=q)
        assert out.shape == oor_img.shape
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    # (d) Non-multiple-of-16 and tiny shapes down to 1x1
    for h, w in ((73, 119), (17, 31), (5, 7), (1, 1)):
        x = _make_textured_image(b=2, h=h, w=w, seed=h + w)
        out = jpeger(x, quality=75.0)
        assert out.shape == (2, 3, h, w)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    # (e) FP32 output even inside outer bfloat16 autocast
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_amp = jpeger(img.to(torch.bfloat16), quality=50.0)
    assert out_amp.dtype == torch.float32
    assert torch.isfinite(out_amp).all()

    # (f) Differentiable mode produces gradients
    jpeger_diff = DiffJPEG(differentiable=True)
    x_grad = img.clone().requires_grad_(True)
    loss = jpeger_diff(x_grad, quality=50.0).mean()
    loss.backward()
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()

    # (g) Invalid inputs raise ValueError
    with pytest.raises(ValueError):
        jpeger(torch.zeros(1, 1, 32, 32), quality=50.0)
    with pytest.raises(ValueError):
        jpeger(torch.zeros(3, 32, 32), quality=50.0)
    with pytest.raises(ValueError):
        jpeger(img2, quality=torch.tensor([30.0, 50.0, 70.0]))
    with pytest.raises(ValueError):
        jpeger(img2, quality=True)  # type: ignore[arg-type]

    # (h) Non-contiguous tensors (channels_last / transposed) match contiguous output bit-exact
    img_nhwc = img2.contiguous(memory_format=torch.channels_last)
    assert not img_nhwc.is_contiguous()
    assert torch.equal(jpeger(img_nhwc, quality=65.0), jpeger(img2, quality=65.0))
    ch_contig = torch.rand(2, 1, 16, 16)
    ch_noncontig = ch_contig.transpose(2, 3)
    assert not ch_noncontig.is_contiguous()
    qtable = jpeger.y_table.view(1, 1, 8, 8)
    out_nc = jpeger._dct_quant_idct_channel(ch_noncontig, qtable, jpeger.dct_matrix)
    out_c = jpeger._dct_quant_idct_channel(ch_noncontig.contiguous(), qtable, jpeger.dct_matrix)
    assert torch.equal(out_nc, out_c)


def test_filter2d_identity_blur_batch_and_tiny() -> None:
    img = _make_textured_image(b=2, h=32, w=32, seed=21)
    pulse = torch.zeros((2, 9, 9), dtype=torch.float32)
    pulse[:, 4, 4] = 1.0
    out_id = filter2d(img, pulse)
    assert torch.allclose(out_id, img, atol=1e-6)

    # Box blur strictly reduces variance on checkerboard
    cb = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    cb[:, :, ::2, ::2] = 1.0
    cb[:, :, 1::2, 1::2] = 1.0
    box = torch.full((1, 5, 5), 1.0 / 25.0, dtype=torch.float32)
    blurred = filter2d(cb, box)
    assert float(blurred.var()) < float(cb.var()) * 0.25

    # Per-sample distinct kernels in same batch
    mixed_k = torch.zeros((2, 5, 5), dtype=torch.float32)
    mixed_k[0, 2, 2] = 1.0
    mixed_k[1] = 1.0 / 25.0
    mixed_out = filter2d(img, mixed_k)
    assert torch.allclose(mixed_out[0], img[0], atol=1e-6)
    assert not torch.allclose(mixed_out[1], img[1], atol=1e-2)

    # Tiny images smaller than pad (replicate fallback)
    big_k = torch.full((1, 21, 21), 1.0 / 441.0, dtype=torch.float32)
    for h, w in ((4, 4), (1, 1)):
        tiny = torch.rand(1, 3, h, w)
        tiny_out = filter2d(tiny, big_k)
        assert tiny_out.shape == (1, 3, h, w) and torch.isfinite(tiny_out).all()


def test_usm_sharp_behavior_and_small_image() -> None:
    usm = USMSharp(radius=50, sigma=0.0)

    # Constant image is invariant
    const = torch.full((1, 3, 32, 32), 0.42, dtype=torch.float32)
    out_const = usm(const)
    assert torch.allclose(out_const, const, atol=1e-6)

    # Step-edge image gets sharper while staying strictly in [0, 1] FP32
    step = torch.full((1, 3, 64, 64), 0.25, dtype=torch.float32)
    step[:, :, :, 32:] = 0.75
    out_step = usm(step, weight=0.5, threshold=10)
    assert out_step.dtype == torch.float32
    assert float(out_step.min()) >= 0.0 and float(out_step.max()) <= 1.0
    orig_grad = float((step[:, :, :, 32] - step[:, :, :, 31]).abs().mean())
    sharp_grad = float((out_step[:, :, :, 32] - out_step[:, :, :, 31]).abs().mean())
    assert sharp_grad > orig_grad

    # Small image (16x16 < radius=50) succeeds via replicate fallback
    small = _make_textured_image(b=1, h=16, w=16, seed=31)
    out_small = usm(small)
    assert out_small.shape == (1, 3, 16, 16) and torch.isfinite(out_small).all()

    with pytest.raises(ValueError):
        USMSharp(radius=0)
    with pytest.raises(ValueError):
        USMSharp(radius=10, sigma=-1.0)
    # Zero-channel / zero-dim inputs raise ValueError cleanly
    for bad_shape in ((1, 0, 16, 16), (0, 3, 16, 16), (1, 3, 0, 16), (1, 3, 16, 0)):
        with pytest.raises(ValueError):
            USMSharp(radius=5)(torch.zeros(bad_shape))


def test_training_pool_warmup_cycling_and_pair_correspondence() -> None:
    with pytest.raises(ValueError):
        TrainingPool(queue_size=0)
    with pytest.raises(ValueError):
        TrainingPool(queue_size=True)  # type: ignore[arg-type]

    pool = TrainingPool(queue_size=8)
    with pytest.raises(ValueError, match="divisible"):
        pool.dequeue_and_enqueue(torch.zeros(3, 3, 8, 8), torch.zeros(3, 3, 32, 32))

    # Tag each sample with a unique constant ID in both lq and gt to verify pair integrity
    def _tagged_batch(start_id: int, b: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        ids = torch.arange(start_id, start_id + b, dtype=torch.float32)
        lq = ids.view(b, 1, 1, 1).expand(b, 3, 8, 8).clone()
        gt = (ids * 10.0).view(b, 1, 1, 1).expand(b, 3, 32, 32).clone()
        return lq, gt

    # Warm-up calls 1 & 2 return live batches verbatim
    lq1, gt1 = _tagged_batch(1)
    out_lq1, out_gt1 = pool.dequeue_and_enqueue(lq1, gt1)
    assert torch.equal(out_lq1, lq1) and torch.equal(out_gt1, gt1)

    lq2, gt2 = _tagged_batch(5)
    out_lq2, out_gt2 = pool.dequeue_and_enqueue(lq2, gt2)
    assert torch.equal(out_lq2, lq2) and torch.equal(out_gt2, gt2)

    # Cycling calls 3..6 return shuffled stored historical pairs with exact (lq, gt) pairing
    seen_ids: set[int] = set()
    for step_idx in range(4):
        in_lq, in_gt = _tagged_batch(9 + step_idx * 4)
        out_lq, out_gt = pool.dequeue_and_enqueue(in_lq, in_gt)
        assert out_lq.shape == in_lq.shape and out_gt.shape == in_gt.shape
        # Returned tensors must not alias pool buffers
        assert pool.queue_lr is not None and out_lq.data_ptr() != pool.queue_lr.data_ptr()
        for sample_i in range(4):
            lq_id = float(out_lq[sample_i, 0, 0, 0].item())
            gt_id = float(out_gt[sample_i, 0, 0, 0].item())
            assert gt_id == pytest.approx(lq_id * 10.0, abs=1e-5)
            seen_ids.add(int(round(lq_id)))
    assert len(seen_ids) >= 6


def test_training_pool_warmup_overflow_and_aliasing_consistency() -> None:
    # (a) Variable batch-size divisor overflow during warm-up: B=4 -> B=8 on queue_size=8
    pool_a = TrainingPool(queue_size=8)
    pool_a.dequeue_and_enqueue(torch.ones(4, 3, 8, 8), torch.ones(4, 3, 32, 32))
    with pytest.raises(ValueError, match="remaining warm-up capacity"):
        pool_a.dequeue_and_enqueue(torch.ones(8, 3, 8, 8), torch.ones(8, 3, 32, 32))
    assert pool_a.queue_ptr == 4

    # (b) Variable batch-size divisor overflow during warm-up: B=4 -> B=2 -> B=4 on queue_size=8
    pool_b = TrainingPool(queue_size=8)
    pool_b.dequeue_and_enqueue(torch.ones(4, 3, 8, 8), torch.ones(4, 3, 32, 32))
    pool_b.dequeue_and_enqueue(torch.ones(2, 3, 8, 8), torch.ones(2, 3, 32, 32))
    assert pool_b.queue_ptr == 6
    with pytest.raises(ValueError, match="remaining warm-up capacity"):
        pool_b.dequeue_and_enqueue(torch.ones(4, 3, 8, 8), torch.ones(4, 3, 32, 32))
    assert pool_b.queue_ptr == 6
    # Pool state remains uncorrupted and accepts exact remaining capacity (B=2)
    out_fill_lq, out_fill_gt = pool_b.dequeue_and_enqueue(
        torch.full((2, 3, 8, 8), 7.0), torch.full((2, 3, 32, 32), 70.0)
    )
    assert pool_b.queue_ptr == 8
    assert out_fill_lq.shape == (2, 3, 8, 8) and out_fill_gt.shape == (2, 3, 32, 32)

    # (c) Warm-up returns detached clones with requires_grad=False, aliasing neither inputs nor pool
    pool_c = TrainingPool(queue_size=4)
    in_lq = torch.full((2, 3, 4, 4), 1.5, requires_grad=True)
    in_gt = torch.full((2, 3, 16, 16), 3.5, requires_grad=True)
    w_lq, w_gt = pool_c.dequeue_and_enqueue(in_lq, in_gt)
    assert w_lq.requires_grad is False
    assert w_gt.requires_grad is False
    assert w_lq.data_ptr() != in_lq.data_ptr()
    assert w_gt.data_ptr() != in_gt.data_ptr()
    assert pool_c.queue_lr is not None and w_lq.data_ptr() != pool_c.queue_lr.data_ptr()
    assert pool_c.queue_gt is not None and w_gt.data_ptr() != pool_c.queue_gt.data_ptr()

    # Mutating returned warm-up tensors mutates neither caller inputs nor pool buffers
    w_lq.fill_(-9.0)
    w_gt.fill_(-99.0)
    assert torch.all(in_lq.detach() == 1.5)
    assert torch.all(in_gt.detach() == 3.5)
    assert torch.all(pool_c.queue_lr[:2] == 1.5)
    assert torch.all(pool_c.queue_gt[:2] == 3.5)

    # Mutating caller inputs (detached) also does not affect internal pool buffers
    with torch.no_grad():
        in_lq.fill_(4.0)
        in_gt.fill_(8.0)
    assert torch.all(pool_c.queue_lr[:2] == 1.5)
    assert torch.all(pool_c.queue_gt[:2] == 3.5)

    # Fill remainder and confirm cycling also has requires_grad=False and no aliasing
    pool_c.dequeue_and_enqueue(
        torch.full((2, 3, 4, 4), 2.5, requires_grad=True),
        torch.full((2, 3, 16, 16), 5.5, requires_grad=True),
    )
    c_lq, c_gt = pool_c.dequeue_and_enqueue(
        torch.full((2, 3, 4, 4), 9.0, requires_grad=True),
        torch.full((2, 3, 16, 16), 18.0, requires_grad=True),
    )
    assert c_lq.requires_grad is False
    assert c_gt.requires_grad is False


def test_realesrgan_degrader_hydra_instantiation() -> None:
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(config_name="train")
    deg = instantiate(cfg.data.degradation)
    assert isinstance(deg, RealESRGANDegrader)
    assert deg.scale == 4
    assert deg.use_usm is False
    assert deg.queue_size == 0
    assert deg.jpeg_range == (30.0, 95.0)
    assert deg.jpeg_range2 == (30.0, 95.0)
    assert deg.kernel_cfg is not None and deg.kernel_cfg["kernel_size"] == 21


def test_realesrgan_degrader_mask_modes_and_usm(monkeypatch: pytest.MonkeyPatch) -> None:
    # (a) All-False passthrough: bit-for-bit identical, zero compute, works without kernel keys
    deg_pass = RealESRGANDegrader(scale=4, use_usm=True, queue_size=0)
    batch_all_false = _make_batch(b=4, mask=[False, False, False, False], seed=41)
    # Remove kernel keys to prove passthrough doesn't touch kernels
    batch_no_kernels = {
        "gt": batch_all_false["gt"],
        "lq": batch_all_false["lq"],
        "need_degradation": batch_all_false["need_degradation"],
    }

    def _forbid(*_args, **_kwargs):
        raise AssertionError("Degradation operator should NOT be called on all-False batch!")

    monkeypatch.setattr(realesrgan_mod, "filter2d", _forbid)
    monkeypatch.setattr(deg_pass.jpeger, "forward", _forbid)
    assert deg_pass.usm_sharpener is not None
    monkeypatch.setattr(deg_pass.usm_sharpener, "forward", _forbid)

    out_pass = deg_pass(batch_no_kernels)
    assert torch.equal(out_pass["lq"], batch_no_kernels["lq"])
    assert torch.equal(out_pass["gt"], batch_no_kernels["gt"])
    monkeypatch.undo()

    # (b) All-True batch
    deg = RealESRGANDegrader(scale=4, use_usm=False, queue_size=0)
    batch_all_true = _make_batch(b=4, mask=[True, True, True, True], seed=42)
    out_true = deg(batch_all_true)
    assert out_true["lq"].shape == (4, 3, 16, 16)
    assert out_true["lq"].dtype == torch.float32
    assert torch.isfinite(out_true["lq"]).all()
    assert float(out_true["lq"].min()) >= 0.0 and float(out_true["lq"].max()) <= 1.0
    # 8-bit quantization exactness: lq * 255 is exact integer
    scaled_255 = out_true["lq"] * 255.0
    assert torch.allclose(scaled_255, scaled_255.round(), atol=1e-5)
    assert not torch.all(out_true["lq"] == 0.0)
    # Original input dict tensors untouched
    assert torch.all(batch_all_true["lq"] == 0.0)

    # (c) Mixed batch [True, False, True, False]: ~mask bit-exact, mask degraded
    batch_mixed = _make_batch(b=4, mask=[True, False, True, False], seed=43)
    orig_lq_clone = batch_mixed["lq"].clone()
    orig_gt_clone = batch_mixed["gt"].clone()
    mask = batch_mixed["need_degradation"]

    out_mixed = deg(batch_mixed)
    assert torch.equal(out_mixed["lq"][~mask], orig_lq_clone[~mask])
    assert torch.equal(out_mixed["gt"][~mask], orig_gt_clone[~mask])
    assert torch.equal(out_mixed["gt"][mask], orig_gt_clone[mask])
    assert not torch.all(out_mixed["lq"][mask] == 0.0)
    assert torch.allclose(
        out_mixed["lq"][mask] * 255.0,
        (out_mixed["lq"][mask] * 255.0).round(),
        atol=1e-5,
    )

    # (d) use_usm=True on mixed batch: sharpens ONLY gt[mask], leaves gt[~mask] bit-exact
    deg_usm = RealESRGANDegrader(scale=4, use_usm=True, queue_size=0, usm_radius=15)
    # Inject crisp step edge in all samples so USM visibly modifies sharp edges
    batch_usm = _make_batch(b=4, mask=[True, False, True, False], seed=44)
    batch_usm["gt"][:, :, :, :32] = 0.2
    batch_usm["gt"][:, :, :, 32:] = 0.8
    orig_gt_usm = batch_usm["gt"].clone()
    orig_lq_usm = batch_usm["lq"].clone()
    mask_u = batch_usm["need_degradation"]

    out_usm = deg_usm(batch_usm)
    assert torch.equal(out_usm["gt"][~mask_u], orig_gt_usm[~mask_u])
    assert torch.equal(out_usm["lq"][~mask_u], orig_lq_usm[~mask_u])
    assert not torch.equal(out_usm["gt"][mask_u], orig_gt_usm[mask_u])
    assert float(out_usm["gt"][mask_u].min()) >= 0.0
    assert float(out_usm["gt"][mask_u].max()) <= 1.0


def test_realesrgan_degrader_strict_mask_validation() -> None:
    deg = RealESRGANDegrader(scale=4)
    base_b4 = _make_batch(b=4, mask=[True, False, True, False], seed=81)

    # Malformed shapes for B=4 MUST raise ValueError
    for bad_shape_mask in (
        torch.tensor([[True, False], [True, False]], dtype=torch.bool),
        torch.tensor([[True, False, True, False]], dtype=torch.bool),
        torch.tensor([[[True, False]], [[True, False]]], dtype=torch.bool),
        torch.tensor([True, False, True], dtype=torch.bool),
        torch.tensor(True, dtype=torch.bool),
    ):
        with pytest.raises(ValueError, match="need_degradation"):
            deg({**base_b4, "need_degradation": bad_shape_mask})

    # Non-boolean dtypes (including float with NaN and int) MUST raise ValueError
    for bad_dtype_mask in (
        torch.tensor([0.0, 1.0, 0.5, float("nan")], dtype=torch.float32),
        torch.tensor([0, 1, 2, -1], dtype=torch.int64),
        torch.tensor([0, 1, 0, 1], dtype=torch.uint8),
    ):
        with pytest.raises(ValueError, match="torch.bool"):
            deg({**base_b4, "need_degradation": bad_dtype_mask})

    # Non-tensor mask raises TypeError
    with pytest.raises(TypeError, match="need_degradation"):
        deg({**base_b4, "need_degradation": [True, False, True, False]})

    # Valid shapes for B=4: (4,) and (4, 1)
    for valid_mask_b4 in (
        torch.tensor([True, False, True, False], dtype=torch.bool),
        torch.tensor([[True], [False], [True], [False]], dtype=torch.bool),
    ):
        orig_lq = base_b4["lq"].clone()
        out = deg({**base_b4, "need_degradation": valid_mask_b4})
        flat_m = valid_mask_b4.reshape(4)
        assert torch.equal(out["lq"][~flat_m], orig_lq[~flat_m])
        assert not torch.all(out["lq"][flat_m] == 0.0)

    # Valid shapes for B=1: (1,), (1, 1), and scalar ()
    base_b1 = _make_batch(b=1, mask=[True], seed=82)
    for valid_mask_b1 in (
        torch.tensor([True], dtype=torch.bool),
        torch.tensor([[True]], dtype=torch.bool),
        torch.tensor(True, dtype=torch.bool),
    ):
        out_b1 = deg({**base_b1, "need_degradation": valid_mask_b1})
        assert out_b1["lq"].shape == (1, 3, 16, 16)
        assert not torch.all(out_b1["lq"] == 0.0)

    # Scalar False for B=1 hits all-False fast path cleanly
    base_b1_false = _make_batch(b=1, mask=[False], seed=83)
    out_b1_f = deg({**base_b1_false, "need_degradation": torch.tensor(False, dtype=torch.bool)})
    assert torch.equal(out_b1_f["lq"], base_b1_false["lq"])
    assert out_b1_f["lq"].data_ptr() == base_b1_false["lq"].data_ptr()

    # 3D shape (1, 1, 1) for B=1 MUST still be rejected
    with pytest.raises(ValueError, match="need_degradation"):
        deg({**base_b1, "need_degradation": torch.tensor([[[True]]], dtype=torch.bool)})


def test_slice_kernel_cross_device_mask_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    # (a) Direct CPU check: slices exact rows and casts to FP32
    kernels_f64 = torch.arange(4 * 3 * 3, dtype=torch.float64).reshape(4, 3, 3)
    mask_cpu = torch.tensor([True, False, True, False], dtype=torch.bool)
    sliced = realesrgan_mod._slice_kernel(kernels_f64, mask_cpu, torch.device("cpu"))
    assert sliced.dtype == torch.float32
    assert sliced.shape == (2, 3, 3)
    assert torch.equal(sliced, kernels_f64[mask_cpu].float())

    # (b) Simulated cross-device alignment (CPU kernel tensor + CUDA mask/target_device):
    # enforces PyTorch's CUDA rule that __getitem__ requires mask.device == tensor.device
    # and transfers only the sliced subset (b_sub=2) to target_device in FP32.
    cpu_dev = torch.device("cpu")
    fake_cuda = torch.device("cuda", 0)
    events: list[tuple[str, object]] = []

    class _FakeMask:
        def __init__(self, data: torch.Tensor, device: torch.device) -> None:
            self.data = data
            self.device = device

        def to(self, *, device: torch.device) -> _FakeMask:
            events.append(("mask.to", device))
            return _FakeMask(self.data, device)

    class _FakeSliced:
        def __init__(self, data: torch.Tensor) -> None:
            self.data = data

        def to(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            events.append(("sliced.to", (device, dtype, tuple(self.data.shape))))
            return self.data.to(dtype=dtype)

    class _StrictDeviceKernel:
        def __init__(self, data: torch.Tensor, device: torch.device) -> None:
            self.data = data
            self.device = device

        def __getitem__(self, idx: _FakeMask) -> _FakeSliced:
            if idx.device != self.device:
                raise RuntimeError(
                    f"Cross-device boolean indexing: mask on {idx.device}, tensor on {self.device}"
                )
            events.append(("kernel.__getitem__", idx.device))
            return _FakeSliced(self.data[idx.data])

    res = realesrgan_mod._slice_kernel(
        _StrictDeviceKernel(kernels_f64, cpu_dev),  # type: ignore[arg-type]
        _FakeMask(mask_cpu, fake_cuda),  # type: ignore[arg-type]
        fake_cuda,
    )
    assert events == [
        ("mask.to", cpu_dev),
        ("kernel.__getitem__", cpu_dev),
        ("sliced.to", (fake_cuda, torch.float32, (2, 3, 3))),
    ]
    assert res.dtype == torch.float32
    assert torch.equal(res, kernels_f64[mask_cpu].float())

    # Same-device fast path: mask.to MUST NOT be invoked when mask.device == tensor.device
    events.clear()
    res_same = realesrgan_mod._slice_kernel(
        _StrictDeviceKernel(kernels_f64, fake_cuda),  # type: ignore[arg-type]
        _FakeMask(mask_cpu, fake_cuda),  # type: ignore[arg-type]
        fake_cuda,
    )
    assert events == [
        ("kernel.__getitem__", fake_cuda),
        ("sliced.to", (fake_cuda, torch.float32, (2, 3, 3))),
    ]
    assert torch.equal(res_same, kernels_f64[mask_cpu].float())

    # (c) Verify RealESRGANDegrader.forward delegates all 3 kernels to _slice_kernel
    seen_targets: list[torch.device] = []
    orig_slice = realesrgan_mod._slice_kernel

    def _spy_slice(
        tensor: torch.Tensor, mask: torch.Tensor, target_device: torch.device
    ) -> torch.Tensor:
        seen_targets.append(target_device)
        return orig_slice(tensor, mask, target_device)

    monkeypatch.setattr(realesrgan_mod, "_slice_kernel", _spy_slice)
    deg = RealESRGANDegrader(scale=4)
    batch = _make_batch(b=4, mask=[True, False, True, False], seed=84)
    _ = deg(batch)
    assert seen_targets == [batch["gt"].device] * 3


def test_realesrgan_degrader_amp_noise_variants_and_pool() -> None:
    # (a) Outer bfloat16 autocast + bf16 inputs -> outputs strictly FP32, no_grad, finite in [0, 1]
    deg = RealESRGANDegrader(scale=4)
    batch = _make_batch(b=4, mask=[True, False, True, False], seed=50)
    batch_bf16 = {
        **batch,
        "gt": batch["gt"].to(torch.bfloat16).requires_grad_(True),
        "lq": batch["lq"].to(torch.bfloat16),
    }
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_amp = deg(batch_bf16)
    assert out_amp["lq"].dtype == torch.float32
    assert out_amp["gt"].dtype == torch.float32
    assert out_amp["lq"].requires_grad is False
    assert out_amp["gt"].requires_grad is False
    assert torch.isfinite(out_amp["lq"]).all()
    assert float(out_amp["lq"].min()) >= 0.0 and float(out_amp["lq"].max()) <= 1.0

    # (b) Force pure Gaussian, pure Poisson, both second_blur_prob branches, non-square, scale=2, tiny
    for g_prob, sb_prob in ((1.0, 1.0), (0.0, 0.0), (0.0, 1.0)):
        deg_var = RealESRGANDegrader(
            scale=2,
            gaussian_noise_prob=g_prob,
            gaussian_noise_prob2=g_prob,
            second_blur_prob=sb_prob,
        )
        b_rect = _make_batch(b=2, gt_h=48, gt_w=64, scale=2, mask=[True, True], seed=51)
        out_rect = deg_var(b_rect)
        assert out_rect["lq"].shape == (2, 3, 24, 32)
        assert torch.isfinite(out_rect["lq"]).all()

    # Tiny GT with extreme downscale range -> guarded by max(1, ...)
    deg_tiny = RealESRGANDegrader(
        scale=4,
        resize_range=(0.1, 0.2),
        resize_range2=(0.1, 0.2),
        gaussian_noise_prob=0.0,
        gaussian_noise_prob2=0.0,
    )
    b_tiny = _make_batch(b=2, gt_h=16, gt_w=16, scale=4, mask=[True, True], k_size=15, seed=52)
    out_tiny = deg_tiny(b_tiny)
    assert out_tiny["lq"].shape == (2, 3, 4, 4)
    assert torch.isfinite(out_tiny["lq"]).all()
    assert float(out_tiny["lq"].min()) >= 0.0 and float(out_tiny["lq"].max()) <= 1.0

    # (c) Integrated TrainingPool: queue_size=8 with B=4 cycles cleanly; queue_size=6 raises
    deg_pool = RealESRGANDegrader(scale=4, queue_size=8)
    for step in range(4):
        b_step = _make_batch(b=4, mask=[True, False, True, False], seed=60 + step)
        out_p = deg_pool(b_step)
        assert out_p["lq"].shape == (4, 3, 16, 16)
        assert out_p["gt"].shape == (4, 3, 64, 64)
        assert torch.isfinite(out_p["lq"]).all()

    deg_bad_pool = RealESRGANDegrader(scale=4, queue_size=6)
    with pytest.raises(ValueError, match="divisible"):
        deg_bad_pool(_make_batch(b=4, mask=[True, False, True, False], seed=70))


def test_end_to_end_mixed_dataset_loader_and_degrader(tmp_path: Path) -> None:
    """Verify full M2 pipeline: manifests -> MixedSRDataset -> multi-worker DataLoader -> RealESRGANDegrader."""
    import json

    from PIL import Image
    from torch.utils.data import DataLoader

    from data import DistributedWeightedInfiniteSampler, MixedSRDataset

    gt_root = tmp_path / "gt_only"
    paired_root = tmp_path / "paired"
    gt_root.mkdir()
    paired_root.mkdir()

    gt_entries = []
    paired_entries = []
    rng = np.random.default_rng(77)
    for idx in range(4):
        arr_hr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        arr_lr = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr_hr, mode="RGB").save(gt_root / f"g_{idx}.png")
        Image.fromarray(arr_hr, mode="RGB").save(paired_root / f"p_hr_{idx}.png")
        Image.fromarray(arr_lr, mode="RGB").save(paired_root / f"p_lr_{idx}.png")
        gt_entries.append({"gt": f"g_{idx}.png"})
        paired_entries.append({"gt": f"p_hr_{idx}.png", "lq": f"p_lr_{idx}.png"})

    gt_json = tmp_path / "gt.json"
    paired_json = tmp_path / "paired.json"
    gt_json.write_text(json.dumps(gt_entries), encoding="utf-8")
    paired_json.write_text(json.dumps(paired_entries), encoding="utf-8")

    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(config_name="train")
    degrader = instantiate(cfg.data.degradation)

    ds = MixedSRDataset(
        sources=[
            {"name": "gt_src", "root_dir": str(gt_root), "json_path": str(gt_json), "ratio": 0.6},
            {"name": "p_src", "root_dir": str(paired_root), "json_path": str(paired_json), "ratio": 0.4},
        ],
        gt_size=64,
        scale=4,
        augment={"hflip": True, "vflip": True, "rot90": True},
        has_degradation=True,
        kernel_cfg=cfg.data.degradation.kernel_cfg,
        is_main_process=False,
    )
    sampler = DistributedWeightedInfiniteSampler(
        source_sizes=ds.source_sizes,
        ratios=ds.source_ratios,
        batch_size_per_gpu=6,
        base_seed=1234,
    )
    loader = DataLoader(ds, batch_size=6, sampler=sampler, num_workers=2)

    it = iter(loader)
    unique_k1_hashes: set[bytes] = set()
    saw_mixed = False
    for _ in range(4):
        raw_batch = next(it)
        mask = raw_batch["need_degradation"]
        if mask.any() and (~mask).any():
            saw_mixed = True
        for k1 in raw_batch["kernel1"][mask]:
            unique_k1_hashes.add(k1.numpy().tobytes())
        orig_paired_lq = raw_batch["lq"][~mask].clone()
        out = degrader(raw_batch)
        assert out["lq"].shape == (6, 3, 16, 16)
        assert out["gt"].shape == (6, 3, 64, 64)
        assert torch.isfinite(out["lq"]).all()
        assert torch.equal(out["lq"][~mask], orig_paired_lq)
        if mask.any():
            assert not torch.all(out["lq"][mask] == 0.0)

    assert saw_mixed is True
    # Multi-worker reseeding yields distinct kernels across samples/workers
    assert len(unique_k1_hashes) >= 4
