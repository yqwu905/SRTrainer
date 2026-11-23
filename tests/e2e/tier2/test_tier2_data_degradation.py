"""Tier 2 E2E tests — Area C: Mixed Data Pipeline & GPU Masked Real-ESRGAN Degradation.

Covers boundary, edge, and negative contracts for F21-F23, F25-F29 and
User Requirements #1-#2. All non-stdlib imports (besides ``pytest``) are
strictly deferred inside test functions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def test_t2_c01_legacy_mode_key_rejected_fail_fast(
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
    validate_sources_caller: Callable[..., Any],
) -> None:
    """F21, Req #2: Legacy 'mode' key in data.sources is rejected fail-fast with ValueError."""
    gt_src = gt_only_train_source_factory(name="c01_gt", n_samples=2, gt_h=48, gt_w=48)
    pair_src = paired_train_source_factory(name="c01_pair", n_samples=2, lq_h=24, lq_w=24, scale=2)

    bad_gt = {**gt_src, "mode": "gt_only"}
    bad_pair = {**pair_src, "mode": "paired"}
    bad_none = {**pair_src, "mode": None}

    for bad_src in (bad_gt, bad_pair, bad_none):
        with pytest.raises(ValueError, match="."):
            validate_sources_caller([bad_src], has_degradation=True)

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([bad_gt], gt_size=32, scale=2, has_degradation=True)

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([bad_pair], gt_size=32, scale=2, has_degradation=False)


def test_t2_c02_gt_only_without_degradation_fail_fast(
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_single_json_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
) -> None:
    """F21, F23: Any gt_only sample with has_degradation=False fails fast with ValueError."""
    gt_src = gt_only_train_source_factory(name="c02_gt", n_samples=2, gt_h=48, gt_w=48)
    pair_src = paired_train_source_factory(name="c02_pair", n_samples=2, lq_h=24, lq_w=24, scale=2)
    mixed_info = mixed_single_json_train_source_factory(
        name="c02_mixed", lq_h=24, lq_w=24, scale=2
    )

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([gt_src], gt_size=32, scale=2, has_degradation=False)

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([pair_src, gt_src], gt_size=32, scale=2, has_degradation=False)

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([mixed_info["source"]], gt_size=32, scale=2, has_degradation=False)


def test_t2_c03_invalid_train_manifests_and_sources_fail_fast(
    tmp_path: Path,
    synthetic_image_factory: Callable[..., Path],
    mixed_dataset_factory: Callable[..., Any],
    validate_sources_caller: Callable[..., Any],
) -> None:
    """F21, Req #1: Malformed JSON manifests and invalid source definitions fail fast."""
    root = tmp_path / "invalid_manifests"
    synthetic_image_factory(root / "hr/0000.png", width=32, height=32, seed=1)

    def _make_src(json_path: Path, ratio: float = 1.0) -> dict[str, Any]:
        return {
            "name": json_path.stem,
            "root_dir": str(root),
            "json_path": str(json_path),
            "ratio": ratio,
        }

    # 1. Nonexistent manifest file
    with pytest.raises((FileNotFoundError, ValueError)):
        mixed_dataset_factory(
            [_make_src(root / "no_such_manifest.json")],
            gt_size=32,
            scale=2,
            has_degradation=True,
        )

    # 2. Top-level dict JSON instead of list-of-dicts
    dict_json = root / "dict_top.json"
    dict_json.write_text(json.dumps({"gt": "hr/0000.png"}), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        mixed_dataset_factory([_make_src(dict_json)], gt_size=32, scale=2, has_degradation=True)

    # 3. Empty list JSON
    empty_json = root / "empty_list.json"
    empty_json.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([_make_src(empty_json)], gt_size=32, scale=2, has_degradation=True)

    # 4. Legacy list-of-strings JSON (violates Req #1)
    str_list_json = root / "str_list.json"
    str_list_json.write_text(json.dumps(["hr/0000.png"]), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        mixed_dataset_factory([_make_src(str_list_json)], gt_size=32, scale=2, has_degradation=True)

    # 5. Dict missing 'gt', empty 'gt', or null 'gt'
    for idx, payload in enumerate(
        (
            [{"lq": "lr/0000.png"}],
            [{"gt": ""}],
            [{"gt": None}],
        )
    ):
        bad_gt_json = root / f"bad_gt_{idx}.json"
        bad_gt_json.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="."):
            mixed_dataset_factory(
                [_make_src(bad_gt_json)], gt_size=32, scale=2, has_degradation=True
            )

    # 6. Missing image file referenced in JSON (covers eager or lazy existence check)
    missing_img_json = root / "missing_img.json"
    missing_img_json.write_text(json.dumps([{"gt": "hr/missing.png"}]), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        ds = mixed_dataset_factory(
            [_make_src(missing_img_json)], gt_size=32, scale=2, has_degradation=True
        )
        _ = ds[0]

    # 7. Empty sources list
    with pytest.raises(ValueError, match="."):
        validate_sources_caller([], has_degradation=True)

    # 8. Zero and negative source ratios
    valid_json = root / "valid.json"
    valid_json.write_text(json.dumps([{"gt": "hr/0000.png"}]), encoding="utf-8")
    for bad_ratio in (0.0, -0.5):
        bad_ratio_src = _make_src(valid_json, ratio=bad_ratio)
        with pytest.raises(ValueError, match="."):
            validate_sources_caller([bad_ratio_src], has_degradation=True)
        with pytest.raises(ValueError, match="."):
            mixed_dataset_factory([bad_ratio_src], gt_size=32, scale=2, has_degradation=True)

    # 9. Source dict missing required json_path or root_dir
    with pytest.raises((ValueError, KeyError)):
        validate_sources_caller(
            [{"name": "no_json", "root_dir": str(root), "ratio": 1.0}],
            has_degradation=True,
        )
    with pytest.raises((ValueError, KeyError)):
        validate_sources_caller(
            [{"name": "no_root", "json_path": str(valid_json), "ratio": 1.0}],
            has_degradation=True,
        )


def test_t2_c04_paired_scale_mismatch_and_gt_size_indivisible_rejected(
    tmp_path: Path,
    synthetic_image_factory: Callable[..., Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
) -> None:
    """F22, F23: Indivisible gt_size, non-positive dims, and paired scale mismatches raise ValueError."""
    valid_x4 = paired_train_source_factory(name="c04_x4", n_samples=2, lq_h=16, lq_w=16, scale=4)

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([valid_x4], gt_size=30, scale=4, has_degradation=False)

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([valid_x4], gt_size=0, scale=4, has_degradation=False)

    with pytest.raises(ValueError, match="."):
        mixed_dataset_factory([valid_x4], gt_size=32, scale=0, has_degradation=False)

    # Paired height mismatch: lq=16x16, gt=30x32 at scale=2 (gt_h != lq_h * 2)
    root_bad_h = tmp_path / "pair_bad_h"
    synthetic_image_factory(root_bad_h / "lr/0.png", width=16, height=16, seed=1)
    synthetic_image_factory(root_bad_h / "hr/0.png", width=32, height=30, seed=2)
    man_h = root_bad_h / "manifest.json"
    man_h.write_text(json.dumps([{"gt": "hr/0.png", "lq": "lr/0.png"}]), encoding="utf-8")
    src_h = {
        "name": "bad_h",
        "root_dir": str(root_bad_h),
        "json_path": str(man_h),
        "ratio": 1.0,
    }
    with pytest.raises(ValueError, match="."):
        ds_h = mixed_dataset_factory([src_h], gt_size=16, scale=2, has_degradation=False)
        _ = ds_h[(0, 0)]

    # Swapped H/W mismatch: lq=(H=16, W=20), gt=(H=40, W=32) at scale=2
    root_swap = tmp_path / "pair_swap_hw"
    synthetic_image_factory(root_swap / "lr/0.png", width=20, height=16, seed=3)
    synthetic_image_factory(root_swap / "hr/0.png", width=32, height=40, seed=4)
    man_swap = root_swap / "manifest.json"
    man_swap.write_text(json.dumps([{"gt": "hr/0.png", "lq": "lr/0.png"}]), encoding="utf-8")
    src_swap = {
        "name": "bad_swap",
        "root_dir": str(root_swap),
        "json_path": str(man_swap),
        "ratio": 1.0,
    }
    with pytest.raises(ValueError, match="."):
        ds_swap = mixed_dataset_factory([src_swap], gt_size=16, scale=2, has_degradation=False)
        _ = ds_swap[(0, 0)]


def test_t2_c05_single_gt_smaller_than_gt_size_reflect_and_replicate_fallback(
    tmp_path: Path,
    synthetic_image_factory: Callable[..., Path],
    mixed_dataset_factory: Callable[..., Any],
) -> None:
    """F22, F23: Small single-GT images pad cleanly via reflect and replicate fallback (non-zero)."""
    import torch

    root = tmp_path / "small_gt_pad"
    rgb1 = (120, 180, 60)
    rgb2 = (90, 150, 210)

    # Entry 0: 20x24 -> pad_h=12 < 20, pad_w=8 < 24 (standard reflect)
    synthetic_image_factory(
        root / "hr/0000.png", width=24, height=20, pattern="gradient", seed=11
    )
    # Entry 1: 4x6 -> pad_h=28 >= 4, pad_w=26 >= 6 (reflect -> replicate fallback)
    synthetic_image_factory(
        root / "hr/0001.png", width=6, height=4, pattern="solid", color=rgb1
    )
    # Entry 2: 1x3 -> pad_h=31 >= 1, pad_w=29 >= 3 (ultra-tiny replicate fallback)
    synthetic_image_factory(
        root / "hr/0002.png", width=3, height=1, pattern="solid", color=rgb2
    )

    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps([{"gt": "hr/0000.png"}, {"gt": "hr/0001.png"}, {"gt": "hr/0002.png"}]),
        encoding="utf-8",
    )
    src = {
        "name": "small_gt",
        "root_dir": str(root),
        "json_path": str(manifest),
        "ratio": 1.0,
    }
    ds = mixed_dataset_factory(
        [src],
        gt_size=32,
        scale=2,
        augment={"hflip": False, "vflip": False, "rot90": False},
        has_degradation=True,
        kernel_cfg={"kernel_size": 7, "kernel_range": [7]},
    )

    for idx in range(3):
        sample = ds[(0, idx)]
        gt = sample["gt"]
        lq = sample["lq"]
        assert gt.shape == (3, 32, 32)
        assert gt.dtype == torch.float32
        assert torch.isfinite(gt).all()
        assert float(gt.min()) >= 0.0 and float(gt.max()) <= 1.0
        assert lq.shape == (3, 16, 16)
        assert bool(sample["need_degradation"].item()) is True

    # Non-zero-pad oracle: solid RGB images remain uniformly filled with the exact color.
    for idx, rgb in ((1, rgb1), (2, rgb2)):
        gt_solid = ds[(0, idx)]["gt"]
        expected = torch.tensor([c / 255.0 for c in rgb], dtype=torch.float32).view(3, 1, 1)
        assert torch.allclose(gt_solid, expected.expand_as(gt_solid), atol=1e-2), (
            f"Small GT entry {idx} appears zero-padded instead of reflect/replicate padded"
        )


def test_t2_c06_weighted_sampler_invalid_arguments_rejected(
    weighted_sampler_factory: Callable[..., Any],
) -> None:
    """F25: DistributedWeightedInfiniteSampler rejects invalid sizes, ratios, ranks, and step seeks."""
    invalid_cases: list[tuple[list[int], list[float], dict[str, Any]]] = [
        ([], [], {}),
        ([10, 20], [1.0], {}),
        ([10, 0], [0.5, 0.5], {}),
        ([-1], [1.0], {}),
        ([10, 10], [-0.2, 1.2], {}),
        ([10, 10], [0.0, 0.0], {}),
        ([10], [1.0], {"batch_size_per_gpu": 0}),
        ([10], [1.0], {"batch_size_per_gpu": -2}),
        ([10], [1.0], {"world_size": 0}),
        ([10], [1.0], {"world_size": -1}),
        ([10], [1.0], {"world_size": 2, "rank": -1}),
        ([10], [1.0], {"world_size": 2, "rank": 2}),
    ]
    for sizes, ratios, kwargs in invalid_cases:
        with pytest.raises(ValueError, match="."):
            weighted_sampler_factory(sizes, ratios, **kwargs)

    sampler = weighted_sampler_factory([10, 10], [0.5, 0.5], batch_size_per_gpu=2)
    with pytest.raises(ValueError, match="."):
        sampler.set_start_micro_step(-1)


def test_t2_c07_diffjpeg_non_multiple_of_16_and_sub16_auto_pad_unpad() -> None:
    """F27: DiffJPEG auto-pads/unpads non-multiples of 16, sub-16 dims, and boundary qualities."""
    import torch

    from data.degradation import DiffJPEG

    torch.manual_seed(27)
    jpeg = DiffJPEG()

    for h, w in ((17, 19), (33, 25), (7, 9), (3, 5)):
        x = torch.rand(2, 3, h, w, dtype=torch.float32)
        out = jpeg(x, quality=torch.tensor([90.0, 35.0], dtype=torch.float32))
        assert out.shape == (2, 3, h, w)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    x_bound = torch.rand(2, 3, 17, 19, dtype=torch.float32)
    out_bound = jpeg(x_bound, quality=torch.tensor([1.0, 100.0], dtype=torch.float32))
    assert out_bound.shape == (2, 3, 17, 19)
    assert out_bound.dtype == torch.float32
    assert torch.isfinite(out_bound).all()
    assert float(out_bound.min()) >= 0.0 and float(out_bound.max()) <= 1.0


def test_t2_c08_realesrgan_degrader_all_false_passthrough_with_usm_and_all_true_batch() -> None:
    """F28: RealESRGANDegrader handles all-False passthrough (even with USM) and all-True batches."""
    import torch

    from data.degradation import RealESRGANDegrader

    torch.manual_seed(28)
    ys = torch.linspace(0.0, 6.0 * 3.14159, 64, dtype=torch.float32)
    xs = torch.linspace(0.0, 6.0 * 3.14159, 64, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    wave = (0.5 + 0.35 * torch.sin(xx) * torch.cos(yy)).unsqueeze(0).repeat(3, 1, 1)
    gt = (wave.unsqueeze(0).repeat(3, 1, 1, 1) + 0.08 * torch.randn(3, 3, 64, 64)).clamp(0.0, 1.0)

    # Sub-case 1: All-False mask with use_usm=True -> bit-exact passthrough for both lq and gt.
    lq_paired = torch.rand(3, 3, 32, 32, dtype=torch.float32)
    mask_none = torch.tensor([False, False, False], dtype=torch.bool)
    batch_none = {
        "gt": gt.clone(),
        "lq": lq_paired.clone(),
        "need_degradation": mask_none.clone(),
        "kernel1": torch.zeros(3, 7, 7, dtype=torch.float32),
        "kernel2": torch.zeros(3, 7, 7, dtype=torch.float32),
        "sinc_kernel": torch.zeros(3, 7, 7, dtype=torch.float32),
    }
    orig_gt = batch_none["gt"].clone()
    orig_lq = batch_none["lq"].clone()
    out_none = RealESRGANDegrader(scale=2, use_usm=True, queue_size=0)(batch_none)
    assert torch.equal(out_none["lq"], orig_lq)
    assert torch.equal(out_none["gt"], orig_gt)

    # Sub-case 2: All-True batch with normalized 7x7 kernels (use_usm=False and use_usm=True).
    pulse = torch.zeros(7, 7, dtype=torch.float32)
    pulse[3, 3] = 1.0
    k1 = torch.full((3, 7, 7), 1.0 / 49.0, dtype=torch.float32)
    k2 = pulse.unsqueeze(0).repeat(3, 1, 1)
    sinc = pulse.unsqueeze(0).repeat(3, 1, 1)
    mask_all = torch.tensor([True, True, True], dtype=torch.bool)

    batch_all_no_usm = {
        "gt": gt.clone(),
        "lq": torch.zeros(3, 3, 32, 32, dtype=torch.float32),
        "need_degradation": mask_all.clone(),
        "kernel1": k1.clone(),
        "kernel2": k2.clone(),
        "sinc_kernel": sinc.clone(),
    }
    out_no_usm = RealESRGANDegrader(scale=2, use_usm=False, queue_size=0)(batch_all_no_usm)
    assert out_no_usm["lq"].shape == (3, 3, 32, 32)
    assert out_no_usm["lq"].dtype == torch.float32
    assert torch.isfinite(out_no_usm["lq"]).all()
    assert float(out_no_usm["lq"].min()) >= 0.0 and float(out_no_usm["lq"].max()) <= 1.0
    assert out_no_usm["lq"].abs().sum().item() > 0.0
    assert torch.equal(out_no_usm["gt"], gt)

    batch_all_usm = {
        "gt": gt.clone(),
        "lq": torch.zeros(3, 3, 32, 32, dtype=torch.float32),
        "need_degradation": mask_all.clone(),
        "kernel1": k1.clone(),
        "kernel2": k2.clone(),
        "sinc_kernel": sinc.clone(),
    }
    out_usm = RealESRGANDegrader(scale=2, use_usm=True, queue_size=0)(batch_all_usm)
    assert torch.isfinite(out_usm["lq"]).all()
    assert float(out_usm["lq"].min()) >= 0.0 and float(out_usm["lq"].max()) <= 1.0
    assert torch.isfinite(out_usm["gt"]).all()
    assert float(out_usm["gt"].min()) >= 0.0 and float(out_usm["gt"].max()) <= 1.0
    assert not torch.equal(out_usm["gt"], gt)


def test_t2_c09_realesrgan_degrader_no_grad_and_fp32_under_cpu_bf16_autocast() -> None:
    """F28, Guardrail #4: RealESRGANDegrader forces torch.no_grad() and FP32 under bf16 autocast."""
    import torch

    from data.degradation import RealESRGANDegrader

    torch.manual_seed(29)
    gt = torch.rand(2, 3, 64, 64, dtype=torch.float32, requires_grad=True)
    lq = torch.zeros(2, 3, 32, 32, dtype=torch.float32)
    lq[1] = torch.rand(3, 32, 32, dtype=torch.float32)
    orig_lq_paired = lq[1].clone()
    mask = torch.tensor([True, False], dtype=torch.bool)

    pulse = torch.zeros(7, 7, dtype=torch.float32)
    pulse[3, 3] = 1.0
    k1 = torch.zeros(2, 7, 7, dtype=torch.float32)
    k2 = torch.zeros(2, 7, 7, dtype=torch.float32)
    sinc = torch.zeros(2, 7, 7, dtype=torch.float32)
    k1[0] = torch.full((7, 7), 1.0 / 49.0, dtype=torch.float32)
    k2[0] = pulse
    sinc[0] = pulse

    batch = {
        "gt": gt,
        "lq": lq,
        "need_degradation": mask,
        "kernel1": k1,
        "kernel2": k2,
        "sinc_kernel": sinc,
    }
    degrader = RealESRGANDegrader(scale=2, use_usm=True, queue_size=0)
    with (
        torch.enable_grad(),
        torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True),
    ):
        out = degrader(batch)

    assert out["lq"].dtype == torch.float32
    assert out["gt"].dtype == torch.float32
    assert out["lq"].requires_grad is False
    assert out["lq"].grad_fn is None
    assert torch.isfinite(out["lq"]).all()
    assert float(out["lq"].min()) >= 0.0 and float(out["lq"].max()) <= 1.0
    assert torch.equal(out["lq"][1], orig_lq_paired)


def test_t2_c10_training_pool_invalid_queue_size_and_non_divisible_batch_rejected() -> None:
    """F28: TrainingPool rejects negative/indivisible queue sizes and works at queue_size == B."""
    import torch

    from data.degradation import RealESRGANDegrader, TrainingPool

    def _call_pool(pool: Any, lq_t: Any, gt_t: Any) -> tuple[Any, Any]:
        if callable(pool):
            return pool(lq_t, gt_t)
        return pool.dequeue_and_enqueue(lq_t, gt_t)

    with pytest.raises(ValueError, match="."):
        TrainingPool(queue_size=-4)

    with pytest.raises(ValueError, match="."):
        RealESRGANDegrader(scale=2, queue_size=-4)

    # Non-divisible queue_size=5 with B=2
    pool_bad = TrainingPool(queue_size=5)
    with pytest.raises(ValueError, match="."):
        _call_pool(pool_bad, torch.rand(2, 3, 16, 16), torch.rand(2, 3, 32, 32))

    deg_bad = RealESRGANDegrader(scale=2, use_usm=False, queue_size=5)
    bad_batch = {
        "lq": torch.rand(2, 3, 16, 16, dtype=torch.float32),
        "gt": torch.rand(2, 3, 32, 32, dtype=torch.float32),
        "need_degradation": torch.tensor([False, False], dtype=torch.bool),
        "kernel1": torch.zeros(2, 7, 7, dtype=torch.float32),
        "kernel2": torch.zeros(2, 7, 7, dtype=torch.float32),
        "sinc_kernel": torch.zeros(2, 7, 7, dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="."):
        deg_bad(bad_batch)

    # queue_size < B (2 % 4 != 0)
    pool_small = TrainingPool(queue_size=2)
    with pytest.raises(ValueError, match="."):
        _call_pool(pool_small, torch.rand(4, 3, 16, 16), torch.rand(4, 3, 32, 32))

    # Minimal valid boundary: queue_size == B == 2 preserves exact (lq[i], gt[i]) pairing
    pool_min = TrainingPool(queue_size=2)
    for step in range(4):
        v0 = 0.1 + 0.1 * (2 * step)
        v1 = 0.1 + 0.1 * (2 * step + 1)
        lq_in = torch.stack([torch.full((3, 16, 16), v0), torch.full((3, 16, 16), v1)], dim=0)
        gt_in = torch.stack([torch.full((3, 32, 32), v0), torch.full((3, 32, 32), v1)], dim=0)
        lq_out, gt_out = _call_pool(pool_min, lq_in, gt_in)
        assert lq_out.shape == (2, 3, 16, 16)
        assert gt_out.shape == (2, 3, 32, 32)
        for i in range(2):
            assert abs(lq_out[i].mean().item() - gt_out[i].mean().item()) < 1e-5
