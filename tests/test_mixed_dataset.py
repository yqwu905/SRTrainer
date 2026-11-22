"""Unit tests for M2 data pipeline: manifests, transforms, MixedSRDataset, ValImageDataset, sampler."""

from __future__ import annotations

import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

import srtrainer.data as data_pkg
from srtrainer.data import (
    DistributedWeightedInfiniteSampler,
    MixedSRDataset,
    SourceStats,
    TrainSample,
    ValImageDataset,
    ValSample,
    load_train_manifest,
    load_val_manifest,
    val_collate_fn,
    validate_sources_config,
)
from srtrainer.data.manifest import is_valid_rel_path
from srtrainer.data.transforms import augment_tensors, paired_random_crop, random_crop

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


def _write_rgb_image(path: Path, height: int, width: int, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _build_synthetic_sources(tmp_path: Path, scale: int = 4) -> dict[str, Any]:
    """Create 3 small synthetic sources: pure gt_only, pure paired, and mixed single-JSON."""
    gt_root = tmp_path / "gt_only_src"
    gt_entries = []
    # Include normal and small images (smaller than gt_size=32) to verify pad fallback
    sizes = [(48, 48), (32, 40), (12, 16), (4, 6)]
    for idx, (h, w) in enumerate(sizes):
        rel = f"hr/img_{idx:02d}.png"
        _write_rgb_image(gt_root / rel, h, w, seed=100 + idx)
        gt_entries.append({"gt": rel})
    gt_json = _write_json(tmp_path / "manifests" / "gt_only.json", gt_entries)

    paired_root = tmp_path / "paired_src"
    paired_entries = []
    for idx in range(4):
        lq_h, lq_w = 10 + idx, 12 + idx
        hr_rel = f"hr/p_{idx:02d}.png"
        lr_rel = f"lr/p_{idx:02d}.png"
        _write_rgb_image(paired_root / hr_rel, lq_h * scale, lq_w * scale, seed=200 + idx)
        _write_rgb_image(paired_root / lr_rel, lq_h, lq_w, seed=300 + idx)
        paired_entries.append({"gt": hr_rel, "lq": lr_rel})
    paired_json = _write_json(tmp_path / "manifests" / "paired.json", paired_entries)

    mixed_root = tmp_path / "mixed_src"
    _write_rgb_image(mixed_root / "hr/m0.png", 40, 40, seed=400)
    _write_rgb_image(mixed_root / "hr/m1.png", 40, 40, seed=401)
    _write_rgb_image(mixed_root / "hr/m2.png", 40, 40, seed=402)
    _write_rgb_image(mixed_root / "hr/m3.png", 40, 40, seed=403)
    _write_rgb_image(mixed_root / "hr/m4.png", 40, 40, seed=404)
    _write_rgb_image(mixed_root / "lr/m4.png", 40 // scale, 40 // scale, seed=405)
    mixed_entries = [
        {"gt": "hr/m0.png"},
        {"gt": "hr/m1.png", "lq": None},
        {"gt": "hr/m2.png", "lq": ""},
        {"gt": "hr/m3.png", "lq": "   "},
        {"gt": "hr/m4.png", "lq": "lr/m4.png"},
    ]
    mixed_json = _write_json(tmp_path / "manifests" / "mixed.json", mixed_entries)

    return {
        "gt_only": {
            "name": "src_gt_only",
            "root_dir": str(gt_root),
            "json_path": str(gt_json),
            "ratio": 0.5,
        },
        "paired": {
            "name": "src_paired",
            "root_dir": str(paired_root),
            "json_path": str(paired_json),
            "ratio": 0.3,
        },
        "mixed": {
            "name": "src_mixed",
            "root_dir": str(mixed_root),
            "json_path": str(mixed_json),
            "ratio": 0.2,
        },
    }


def test_data_package_all_exports() -> None:
    expected = {
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
    }
    assert set(data_pkg.__all__) == expected
    assert len(data_pkg.__all__) == 10
    for name in expected:
        assert hasattr(data_pkg, name)


def test_train_manifest_pure_and_mixed_autodetect(tmp_path: Path) -> None:
    assert is_valid_rel_path("a/b.png") is True
    for invalid in (None, "", "   ", "\t\n", 0, 123, True, False, [], {}):
        assert is_valid_rel_path(invalid) is False

    root = tmp_path / "data_root"
    gt_json = _write_json(
        tmp_path / "gt.json",
        [{"gt": "hr/01.png"}, {"gt": "hr/02.png"}],
    )
    samples, stats = load_train_manifest(gt_json, root, source_name="df2k")
    assert len(samples) == 2
    assert all(isinstance(s, TrainSample) for s in samples)
    assert all(s.need_degradation is True and s.lq_path is None for s in samples)
    assert samples[0].gt_path == root / "hr/01.png"
    assert stats == SourceStats(name="df2k", total=2, gt_only=2, paired=0, ratio=1.0)
    assert stats.format_summary() == "[Data] df2k: 2 samples (gt_only: 2, paired: 0)"

    paired_json = _write_json(
        tmp_path / "paired.json",
        [{"gt": "hr/01.png", "lq": "lr/01.png"}],
    )
    p_samples, p_stats = load_train_manifest(paired_json, root)
    assert p_stats.name == "paired"
    assert p_stats == SourceStats(name="paired", total=1, gt_only=0, paired=1, ratio=1.0)
    assert p_samples[0].need_degradation is False
    assert p_samples[0].lq_path == root / "lr/01.png"

    mixed_json = _write_json(
        tmp_path / "mixed.json",
        [
            {"gt": "hr/1.png"},
            {"gt": "hr/2.png", "lq": None},
            {"gt": "hr/3.png", "lq": ""},
            {"gt": "hr/4.png", "lq": "   "},
            {"gt": "hr/5.png", "lq": "lr/5.png"},
        ],
    )
    m_samples, m_stats = load_train_manifest(mixed_json, root, source_name="mix")
    assert [s.need_degradation for s in m_samples] == [True, True, True, True, False]
    assert [s.lq_path is not None for s in m_samples] == [False, False, False, False, True]
    assert m_stats == SourceStats(name="mix", total=5, gt_only=4, paired=1, ratio=1.0)


def test_validate_sources_rank0_logging(tmp_path: Path) -> None:
    sources_map = _build_synthetic_sources(tmp_path)
    sources = [sources_map["gt_only"], sources_map["paired"], sources_map["mixed"]]

    class _Recorder:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def info(self, msg: str) -> None:
            self.messages.append(msg)

    rec_rank0 = _Recorder()
    all_samples, all_stats = validate_sources_config(
        sources=sources,
        has_degradation=True,
        is_main_process=True,
        logger=rec_rank0,  # type: ignore[arg-type]
    )
    assert len(all_samples) == 3
    assert len(all_stats) == 3
    assert [s.ratio for s in all_stats] == [0.5, 0.3, 0.2]
    assert rec_rank0.messages == [
        "[Data] src_gt_only: 4 samples (gt_only: 4, paired: 0)",
        "[Data] src_paired: 4 samples (gt_only: 0, paired: 4)",
        "[Data] src_mixed: 5 samples (gt_only: 4, paired: 1)",
    ]

    # Non-main rank MUST emit zero logs even when a logger is supplied.
    rec_rank1 = _Recorder()
    validate_sources_config(
        sources=sources,
        has_degradation=True,
        is_main_process=False,
        logger=rec_rank1,  # type: ignore[arg-type]
    )
    assert rec_rank1.messages == []

    # Default logger ("srtrainer") works when logger=None.
    sr_logger = logging.getLogger("srtrainer")
    captured: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _ListHandler()
    prev_level = sr_logger.level
    sr_logger.setLevel(logging.INFO)
    sr_logger.addHandler(handler)
    try:
        validate_sources_config(sources=[sources_map["paired"]], has_degradation=False)
    finally:
        sr_logger.removeHandler(handler)
        sr_logger.setLevel(prev_level)
    assert "[Data] src_paired: 4 samples (gt_only: 0, paired: 4)" in captured


def test_manifest_and_sources_fail_fast(tmp_path: Path) -> None:
    sources_map = _build_synthetic_sources(tmp_path)

    # Missing JSON file -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_train_manifest(tmp_path / "does_not_exist.json", tmp_path)

    # Empty, non-list, bad JSON, non-dict entry, missing/empty gt, invalid lq type -> ValueError
    empty_json = _write_json(tmp_path / "empty.json", [])
    with pytest.raises(ValueError, match="empty"):
        load_train_manifest(empty_json, tmp_path)

    dict_json = _write_json(tmp_path / "dict.json", {"gt": "a.png"})
    with pytest.raises(ValueError, match="list of dicts"):
        load_train_manifest(dict_json, tmp_path)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="parse"):
        load_train_manifest(bad_json, tmp_path)

    non_dict_entry = _write_json(tmp_path / "non_dict.json", ["hr/01.png"])
    with pytest.raises(ValueError, match="JSON object"):
        load_train_manifest(non_dict_entry, tmp_path)

    for bad_gt in ({"lq": "lr/1.png"}, {"gt": None}, {"gt": ""}, {"gt": "   "}, {"gt": 123}):
        p = _write_json(tmp_path / "bad_gt.json", [bad_gt])
        with pytest.raises(ValueError, match="gt"):
            load_train_manifest(p, tmp_path)

    bad_lq_type = _write_json(tmp_path / "bad_lq.json", [{"gt": "hr/1.png", "lq": 123}])
    with pytest.raises(ValueError, match="lq"):
        load_train_manifest(bad_lq_type, tmp_path)

    # Legacy 'mode' key MUST fail-fast even when mode is None or empty string
    for legacy_mode in ("gt_only", "paired", None, ""):
        bad_src = {**sources_map["paired"], "mode": legacy_mode}
        with pytest.raises(ValueError, match="mode"):
            validate_sources_config([bad_src], has_degradation=True)

    # has_degradation=False with any gt_only sample (pure or inside mixed JSON) -> ValueError
    with pytest.raises(ValueError, match="gt_only"):
        validate_sources_config([sources_map["gt_only"]], has_degradation=False)

    with pytest.raises(ValueError, match="gt_only"):
        validate_sources_config(
            [sources_map["paired"], sources_map["mixed"]],
            has_degradation=False,
        )

    # Non-bool has_degradation MUST raise TypeError (fail-fast, never coerce truthiness)
    for bad_has_deg in ("false", "true", 1, 0, None):
        with pytest.raises(TypeError, match="has_degradation"):
            validate_sources_config(
                [sources_map["gt_only"]],
                has_degradation=bad_has_deg,  # type: ignore[arg-type]
            )

    # Empty sources / invalid ratios / OmegaConf missing ???
    with pytest.raises(ValueError):
        validate_sources_config([], has_degradation=True)

    for bad_ratio in (-0.1, True, False, float("nan"), float("inf"), "0.5", None):
        bad_src = {**sources_map["paired"], "ratio": bad_ratio}
        with pytest.raises(ValueError, match="ratio"):
            validate_sources_config([bad_src], has_degradation=False)

    zero_src = {**sources_map["paired"], "ratio": 0.0}
    with pytest.raises(ValueError, match="ratios"):
        validate_sources_config([zero_src], has_degradation=False)

    missing_cfg = OmegaConf.create(
        [{"name": "s", "root_dir": "???", "json_path": sources_map["paired"]["json_path"], "ratio": 1.0}]
    )
    with pytest.raises(ValueError):
        validate_sources_config(missing_cfg, has_degradation=False)


def test_paired_random_crop_exact_alignment_and_errors() -> None:
    scale = 4
    lq_h, lq_w = 24, 28
    gt_size = 32  # lq_size = 8

    # Encode unique (y, x, c) values into LQ and expand block-wise to GT
    ys = torch.arange(lq_h, dtype=torch.float32).view(1, lq_h, 1).expand(1, lq_h, lq_w) / lq_h
    xs = torch.arange(lq_w, dtype=torch.float32).view(1, 1, lq_w).expand(1, lq_h, lq_w) / lq_w
    diag = (ys + xs) * 0.5
    lq = torch.cat([ys, xs, diag], dim=0).contiguous()
    gt = lq.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2).contiguous()

    for _ in range(50):
        gt_c, lq_c = paired_random_crop(gt, lq, gt_size=gt_size, scale=scale)
        assert gt_c.shape == (3, gt_size, gt_size)
        assert lq_c.shape == (3, gt_size // scale, gt_size // scale)
        assert gt_c.is_contiguous() and lq_c.is_contiguous()
        expected_gt = lq_c.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
        assert torch.equal(gt_c, expected_gt)

    # Exact fit crop (lq_h == lq_size, lq_w == lq_size)
    lq_fit = lq[:, :8, :8].contiguous()
    gt_fit = gt[:, :32, :32].contiguous()
    gt_c, lq_c = paired_random_crop(gt_fit, lq_fit, gt_size=32, scale=4)
    assert torch.equal(gt_c, gt_fit) and torch.equal(lq_c, lq_fit)

    # Error checks
    with pytest.raises(ValueError, match="divisible"):
        paired_random_crop(gt, lq, gt_size=30, scale=4)
    with pytest.raises(ValueError, match="mismatch"):
        paired_random_crop(torch.zeros(3, 64, 64), torch.zeros(3, 17, 16), gt_size=32, scale=4)
    with pytest.raises(ValueError, match="smaller"):
        paired_random_crop(torch.zeros(3, 16, 16), torch.zeros(3, 4, 4), gt_size=32, scale=4)
    with pytest.raises(ValueError):
        paired_random_crop(gt, lq, gt_size=True, scale=4)  # type: ignore[arg-type]


def test_random_crop_normal_small_and_tiny_fallback() -> None:
    gt_large = torch.rand(3, 64, 80)
    out_large = random_crop(gt_large, gt_size=32)
    assert out_large.shape == (3, 32, 32) and out_large.is_contiguous()

    # Small image (reflect pad) and tiny image (reflect->replicate fallback)
    for shape in ((3, 12, 20), (3, 4, 4), (3, 1, 1), (3, 32, 32)):
        img = torch.rand(shape)
        cropped = random_crop(img, gt_size=32, pad_if_smaller=True)
        assert cropped.shape == (3, 32, 32)
        assert cropped.dtype == torch.float32
        assert cropped.is_contiguous()
        assert torch.isfinite(cropped).all()
        assert float(cropped.min()) >= 0.0 and float(cropped.max()) <= 1.0

    with pytest.raises(ValueError, match="smaller"):
        random_crop(torch.rand(3, 16, 16), gt_size=32, pad_if_smaller=False)
    with pytest.raises(ValueError, match="positive"):
        random_crop(torch.empty(3, 0, 16), gt_size=32)


def test_augment_tensors_synchronized_and_flags() -> None:
    scale = 4
    lq = torch.arange(3 * 8 * 8, dtype=torch.float32).view(3, 8, 8)
    gt = lq.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)

    # All-False -> exact identity, contiguous, no mutation
    gt_id, lq_id = augment_tensors([gt, lq], hflip=False, vflip=False, rot90=False)
    assert torch.equal(gt_id, gt) and torch.equal(lq_id, lq)
    assert gt_id.is_contiguous() and lq_id.is_contiguous()

    # Synchronized across [gt, lq] over 50 random draws; covers multiple symmetries
    seen_hashes: set[bytes] = set()
    for _ in range(50):
        gt_a, lq_a = augment_tensors([gt, lq], hflip=True, vflip=True, rot90=True)
        assert gt_a.is_contiguous() and lq_a.is_contiguous()
        assert torch.equal(
            gt_a,
            lq_a.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2),
        )
        seen_hashes.add(lq_a.numpy().tobytes())
    assert len(seen_hashes) >= 4
    assert augment_tensors([], hflip=True, vflip=True, rot90=True) == []


def test_mixed_dataset_tuple_and_flat_indexing(tmp_path: Path) -> None:
    sources_map = _build_synthetic_sources(tmp_path, scale=4)
    sources_cfg = OmegaConf.create(
        [sources_map["gt_only"], sources_map["paired"], sources_map["mixed"]]
    )
    augment_cfg = OmegaConf.create({"hflip": True, "vflip": True, "rot90": True})

    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        train_cfg = hydra.compose(config_name="train")
        kernel_cfg = train_cfg.data.degradation.kernel_cfg

    ds = MixedSRDataset(
        sources=sources_cfg,
        gt_size=32,
        scale=4,
        augment=augment_cfg,
        has_degradation=True,
        kernel_cfg=kernel_cfg,
        is_main_process=False,
    )

    assert len(ds) == 4 + 4 + 5
    assert ds.source_sizes == [4, 4, 5]
    assert ds.source_ratios == [0.5, 0.3, 0.2]
    assert len(ds.source_stats) == 3

    # Defensive copy checks
    sizes_copy = ds.source_sizes
    sizes_copy[0] = 999
    assert ds.source_sizes == [4, 4, 5]

    # Tuple indexing: gt_only (including tiny 4x6 image at (0, 3))
    item_gt = ds[(0, 3)]
    assert set(item_gt.keys()) == {
        "gt",
        "lq",
        "need_degradation",
        "source_id",
        "kernel1",
        "kernel2",
        "sinc_kernel",
    }
    assert item_gt["gt"].shape == (3, 32, 32) and item_gt["gt"].dtype == torch.float32
    assert item_gt["lq"].shape == (3, 8, 8) and torch.all(item_gt["lq"] == 0.0)
    assert item_gt["need_degradation"].dtype == torch.bool
    assert bool(item_gt["need_degradation"].item()) is True
    assert int(item_gt["source_id"].item()) == 0
    for k_name in ("kernel1", "kernel2", "sinc_kernel"):
        kt = item_gt[k_name]
        assert kt.shape == (21, 21) and kt.dtype == torch.float32
        assert kt.sum().item() == pytest.approx(1.0, abs=1e-5)

    # Tuple indexing: paired
    item_paired = ds[(1, 0)]
    assert item_paired["gt"].shape == (3, 32, 32)
    assert item_paired["lq"].shape == (3, 8, 8)
    assert bool(item_paired["need_degradation"].item()) is False
    assert int(item_paired["source_id"].item()) == 1
    assert torch.all(item_paired["kernel1"] == 0.0)
    assert torch.all(item_paired["kernel2"] == 0.0)
    assert torch.all(item_paired["sinc_kernel"] == 0.0)
    # Distinct zero tensors (not aliased)
    assert item_paired["kernel1"].data_ptr() != item_paired["kernel2"].data_ptr()
    assert item_paired["kernel2"].data_ptr() != item_paired["sinc_kernel"].data_ptr()

    # Flat indexing including negative indexing (-1 is last item of mixed source, which is paired)
    item_last = ds[-1]
    assert int(item_last["source_id"].item()) == 2
    assert bool(item_last["need_degradation"].item()) is False

    item_first = ds[-len(ds)]
    assert int(item_first["source_id"].item()) == 0
    assert bool(item_first["need_degradation"].item()) is True


def test_mixed_dataset_degradation_modes_and_errors(tmp_path: Path) -> None:
    sources_map = _build_synthetic_sources(tmp_path, scale=4)

    # (a) has_degradation=True, kernel_cfg=None -> default 21x21 kernels
    ds_default_k = MixedSRDataset(
        sources=[sources_map["gt_only"]],
        gt_size=32,
        scale=4,
        has_degradation=True,
        kernel_cfg=None,
        is_main_process=False,
    )
    assert ds_default_k.kernel_size == 21
    assert ds_default_k[0]["kernel1"].shape == (21, 21)

    # (b) has_degradation=False on pure-paired -> omits kernel keys cleanly
    ds_paired_only = MixedSRDataset(
        sources=[sources_map["paired"]],
        gt_size=32,
        scale=4,
        has_degradation=False,
        is_main_process=False,
    )
    assert ds_paired_only.kernel_generator is None
    assert ds_paired_only.kernel_size is None
    item = ds_paired_only[0]
    assert set(item.keys()) == {"gt", "lq", "need_degradation", "source_id"}
    assert bool(item["need_degradation"].item()) is False

    # (c) has_degradation=False with gt_only -> ValueError
    with pytest.raises(ValueError, match="gt_only"):
        MixedSRDataset(
            sources=[sources_map["gt_only"]],
            gt_size=32,
            scale=4,
            has_degradation=False,
            is_main_process=False,
        )

    # (d) Index and constructor errors
    with pytest.raises(IndexError):
        _ = ds_paired_only[len(ds_paired_only)]
    with pytest.raises(IndexError):
        _ = ds_paired_only[(-1, 0)]
    with pytest.raises(IndexError):
        _ = ds_paired_only[(0, -1)]
    with pytest.raises(IndexError):
        _ = ds_paired_only[(0, 0, 0)]  # type: ignore[index]
    with pytest.raises(TypeError):
        _ = ds_paired_only[True]  # type: ignore[index]
    with pytest.raises(TypeError):
        _ = ds_paired_only["0"]  # type: ignore[index]

    with pytest.raises(ValueError, match="divisible"):
        MixedSRDataset(sources=[sources_map["paired"]], gt_size=30, scale=4, has_degradation=False)
    with pytest.raises(TypeError, match="has_degradation"):
        MixedSRDataset(
            sources=[sources_map["paired"]],
            gt_size=32,
            scale=4,
            has_degradation="false",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="augment"):
        MixedSRDataset(
            sources=[sources_map["paired"]],
            gt_size=32,
            scale=4,
            augment={"hflip": "true"},
            has_degradation=False,
        )

    # Paired dimension mismatch detected inside __getitem__
    bad_root = tmp_path / "bad_paired"
    _write_rgb_image(bad_root / "hr.png", 40, 40, seed=1)
    _write_rgb_image(bad_root / "lr.png", 11, 10, seed=2)
    bad_json = _write_json(tmp_path / "bad_paired.json", [{"gt": "hr.png", "lq": "lr.png"}])
    ds_bad = MixedSRDataset(
        sources=[{"name": "bad", "root_dir": str(bad_root), "json_path": str(bad_json), "ratio": 1.0}],
        gt_size=32,
        scale=4,
        has_degradation=False,
        is_main_process=False,
    )
    with pytest.raises(ValueError, match="mismatch"):
        _ = ds_bad[0]


def test_sampler_ratio_decoupled_from_dataset_size() -> None:
    for ratios in ([0.7, 0.3], [70.0, 30.0]):
        sampler = DistributedWeightedInfiniteSampler(
            source_sizes=[10, 100_000],
            ratios=ratios,
            batch_size_per_gpu=10,
            rank=0,
            world_size=1,
            base_seed=123,
        )
        draws = list(itertools.islice(iter(sampler), 10_000))
        src0 = [idx for sid, idx in draws if sid == 0]
        src1 = [idx for sid, idx in draws if sid == 1]
        emp_rate = len(src0) / len(draws)
        assert abs(emp_rate - 0.70) < 0.02
        assert set(src0) == set(range(10))
        assert all(0 <= idx < 100_000 for idx in src1)

    # Zero-weight source must never be sampled
    sampler_zero = DistributedWeightedInfiniteSampler(
        source_sizes=[10, 20, 30],
        ratios=[0.0, 1.0, 3.0],
        batch_size_per_gpu=8,
        base_seed=7,
    )
    draws_z = list(itertools.islice(iter(sampler_zero), 2000))
    assert all(sid != 0 for sid, _ in draws_z)
    rate_src2 = sum(1 for sid, _ in draws_z if sid == 2) / len(draws_z)
    assert abs(rate_src2 - 0.75) < 0.03


def test_sampler_multi_rank_disjoint_and_partition() -> None:
    world_size = 4
    batch_per_gpu = 8
    ref_sampler = DistributedWeightedInfiniteSampler(
        source_sizes=[100, 100],
        ratios=[0.6, 0.4],
        batch_size_per_gpu=world_size * batch_per_gpu,
        rank=0,
        world_size=1,
        base_seed=42,
    )
    rank_samplers = [
        DistributedWeightedInfiniteSampler(
            source_sizes=[100, 100],
            ratios=[0.6, 0.4],
            batch_size_per_gpu=batch_per_gpu,
            rank=r,
            world_size=world_size,
            base_seed=42,
        )
        for r in range(world_size)
    ]

    ref_it = iter(ref_sampler)
    rank_its = [iter(s) for s in rank_samplers]

    for _ in range(20):
        ref_step = list(itertools.islice(ref_it, world_size * batch_per_gpu))
        per_rank = [list(itertools.islice(it, batch_per_gpu)) for it in rank_its]

        # Reconstruct interleaved global batch and check bit-for-bit equality
        reconstructed: list[tuple[int, int]] = []
        for slot in range(batch_per_gpu):
            for r in range(world_size):
                reconstructed.append(per_rank[r][slot])
        assert reconstructed == ref_step

        # Pairwise rank sets are 100% disjoint
        for i in range(world_size):
            for j in range(i + 1, world_size):
                assert set(per_rank[i]).isdisjoint(set(per_rank[j]))

    # Small source_sizes (< global_batch) still partitions cleanly without crash
    small_ref = DistributedWeightedInfiniteSampler(
        source_sizes=[5, 5],
        ratios=[0.5, 0.5],
        batch_size_per_gpu=32,
        rank=0,
        world_size=1,
        base_seed=99,
    )
    small_ranks = [
        DistributedWeightedInfiniteSampler(
            source_sizes=[5, 5],
            ratios=[0.5, 0.5],
            batch_size_per_gpu=8,
            rank=r,
            world_size=4,
            base_seed=99,
        )
        for r in range(4)
    ]
    ref_small_step = list(itertools.islice(iter(small_ref), 32))
    ranks_small_step = [list(itertools.islice(iter(s), 8)) for s in small_ranks]
    rebuilt = [ranks_small_step[r][slot] for slot in range(8) for r in range(4)]
    assert rebuilt == ref_small_step
    assert all(0 <= idx < 5 for _, idx in rebuilt)


def test_sampler_o1_resume_exact_match() -> None:
    cont = DistributedWeightedInfiniteSampler(
        source_sizes=[50, 80],
        ratios=[0.7, 0.3],
        batch_size_per_gpu=4,
        rank=1,
        world_size=2,
        base_seed=2026,
    )
    stream_30 = list(itertools.islice(iter(cont), 30 * 4))
    tail_expected = stream_30[20 * 4 :]

    # Resume via set_start_micro_step
    resumed = DistributedWeightedInfiniteSampler(
        source_sizes=[50, 80],
        ratios=[0.7, 0.3],
        batch_size_per_gpu=4,
        rank=1,
        world_size=2,
        base_seed=2026,
    )
    resumed.set_start_micro_step(20)
    assert list(itertools.islice(iter(resumed), 10 * 4)) == tail_expected

    # Calling iter(resumed) twice replays identical stream
    assert list(itertools.islice(iter(resumed), 10 * 4)) == tail_expected

    # Direct start_micro_step in __init__
    direct = DistributedWeightedInfiniteSampler(
        source_sizes=[50, 80],
        ratios=[0.7, 0.3],
        batch_size_per_gpu=4,
        rank=1,
        world_size=2,
        base_seed=2026,
        start_micro_step=20,
    )
    assert list(itertools.islice(iter(direct), 10 * 4)) == tail_expected

    # O(1) speed check at 10,000,000 micro_step
    t0 = time.perf_counter()
    resumed.set_start_micro_step(10_000_000)
    first_batch = list(itertools.islice(iter(resumed), 4))
    elapsed = time.perf_counter() - t0
    assert len(first_batch) == 4
    assert elapsed < 0.2

    # Infinite sampler does NOT implement __len__
    with pytest.raises(TypeError):
        _ = len(resumed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_sizes": [], "ratios": [], "batch_size_per_gpu": 2},
        {"source_sizes": [10, 20], "ratios": [1.0], "batch_size_per_gpu": 2},
        {"source_sizes": [0, 10], "ratios": [0.5, 0.5], "batch_size_per_gpu": 2},
        {"source_sizes": [True], "ratios": [1.0], "batch_size_per_gpu": 2},
        {"source_sizes": [10], "ratios": [True], "batch_size_per_gpu": 2},
        {"source_sizes": [10], "ratios": [float("nan")], "batch_size_per_gpu": 2},
        {"source_sizes": [10], "ratios": [float("inf")], "batch_size_per_gpu": 2},
        {"source_sizes": [10], "ratios": [-0.5], "batch_size_per_gpu": 2},
        {"source_sizes": [10, 20], "ratios": [0.0, 0.0], "batch_size_per_gpu": 2},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": 0},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": True},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": 2, "world_size": 0},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": 2, "rank": -1},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": 2, "rank": 2, "world_size": 2},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": 2, "base_seed": -1},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": 2, "start_micro_step": -1},
        {"source_sizes": [10], "ratios": [1.0], "batch_size_per_gpu": 2, "start_micro_step": True},
    ],
)
def test_sampler_invalid_args_fail_fast(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DistributedWeightedInfiniteSampler(**kwargs)


@pytest.mark.parametrize(
    ("num_workers", "persistent_workers"),
    [(0, False), (2, False), (2, True)],
)
def test_dataloader_mixed_batch_collation_and_resume(
    tmp_path: Path,
    num_workers: int,
    persistent_workers: bool,
) -> None:
    sources_map = _build_synthetic_sources(tmp_path, scale=4)

    # Non-picklable local logger verifies MixedSRDataset does NOT store logger on self
    logged: list[str] = []

    class _LocalLogger:
        def info(self, msg: str) -> None:
            logged.append(msg)

    ds = MixedSRDataset(
        sources=[sources_map["gt_only"], sources_map["paired"]],
        gt_size=32,
        scale=4,
        augment={"hflip": True, "vflip": True, "rot90": True},
        has_degradation=True,
        kernel_cfg={"kernel_size": 15},
        is_main_process=True,
        logger=_LocalLogger(),
    )
    assert len(logged) == 2

    sampler = DistributedWeightedInfiniteSampler(
        source_sizes=ds.source_sizes,
        ratios=ds.source_ratios,
        batch_size_per_gpu=6,
        rank=0,
        world_size=1,
        base_seed=42,
    )
    loader = DataLoader(
        ds,
        batch_size=6,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    seen_both_modes = False
    step5_source_ids: list[int] | None = None
    loader_iter = iter(loader)
    for step in range(8):
        batch = next(loader_iter)
        assert batch["gt"].shape == (6, 3, 32, 32)
        assert batch["lq"].shape == (6, 3, 8, 8)
        assert batch["need_degradation"].shape == (6,)
        assert batch["need_degradation"].dtype == torch.bool
        assert batch["source_id"].shape == (6,)
        assert batch["kernel1"].shape == (6, 15, 15)

        mask = batch["need_degradation"]
        if mask.any() and (~mask).any():
            seen_both_modes = True
        if mask.any():
            assert torch.all(batch["lq"][mask] == 0.0)
        if (~mask).any():
            assert torch.all(batch["kernel1"][~mask] == 0.0)
            assert torch.all(batch["kernel2"][~mask] == 0.0)
            assert torch.all(batch["sinc_kernel"][~mask] == 0.0)
        if step == 5:
            step5_source_ids = batch["source_id"].tolist()

    assert seen_both_modes is True
    assert step5_source_ids is not None

    # Resume at micro_step=5 on same loader and verify exact source_id sequence
    sampler.set_start_micro_step(5)
    resumed_batch = next(iter(loader))
    assert resumed_batch["source_id"].tolist() == step5_source_ids


def test_val_dataset_and_collate_meta_passthrough(tmp_path: Path) -> None:
    root = tmp_path / "val_root"
    _write_rgb_image(root / "lr/001.png", 12, 16, seed=1)
    _write_rgb_image(root / "hr/001.png", 48, 64, seed=2)
    _write_rgb_image(root / "lr/002.png", 10, 14, seed=3)
    _write_rgb_image(root / "lr/003.png", 14, 14, seed=4)
    _write_rgb_image(root / "lr/004.png", 8, 12, seed=5)

    val_json = _write_json(
        tmp_path / "val.json",
        [
            {
                "lq": "lr/001.png",
                "gt": "hr/001.png",
                "label": "HELLO 2026",
                "tags": ["ocr", "sharp"],
                "nested": {"score": 1.5, "empty": None},
                # Reserved keys in raw JSON must be overwritten by resolved framework fields
                "index": 999,
                "lq_path": "fake",
            },
            {"lq": "lr/002.png", "label": "NO_GT"},
            {"lq": "lr/003.png", "gt": None, "label": "NULL_GT"},
            {"lq": "lr/004.png", "gt": "", "label": "EMPTY_GT"},
        ],
    )

    samples = load_val_manifest(json_path=val_json, root_dir=root)
    assert len(samples) == 4
    assert all(isinstance(s, ValSample) for s in samples)
    assert samples[0].gt_path == root / "hr/001.png"
    assert samples[1].gt_path is None
    assert samples[2].gt_path is None
    assert samples[3].gt_path is None
    assert samples[0].meta["index"] == 0
    assert samples[0].meta["lq_path"] == str(root / "lr/001.png")

    ds = ValImageDataset(root_dir=root, json_path=val_json)
    assert len(ds) == 4

    # Mutation isolation across accesses
    first = ds[0]
    first["meta"]["nested"]["score"] = -999.0
    first["meta"]["label"] = "MUTATED"
    again = ds[0]
    assert again["meta"]["nested"]["score"] == 1.5
    assert again["meta"]["label"] == "HELLO 2026"

    # Negative indexing & out-of-range IndexError
    last = ds[-1]
    assert last["gt"] is None
    assert last["meta"]["index"] == 3
    with pytest.raises(IndexError):
        _ = ds[4]

    # DataLoader with val_collate_fn (num_workers=0 and num_workers=2)
    for nw in (0, 2):
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=nw, collate_fn=val_collate_fn)
        batches = list(loader)
        assert len(batches) == 4

        b0 = batches[0]
        assert b0["lq"].shape == (1, 3, 12, 16)
        assert b0["gt"] is not None and b0["gt"].shape == (1, 3, 48, 64)
        assert isinstance(b0["meta"], dict)
        assert b0["meta"]["label"] == "HELLO 2026"
        assert b0["meta"]["tags"] == ["ocr", "sharp"]
        assert b0["meta"]["nested"] == {"score": 1.5, "empty": None}
        assert b0["meta"]["lq_path"] == str(root / "lr/001.png")
        assert b0["meta"]["gt_path"] == str(root / "hr/001.png")
        assert b0["meta"]["index"] == 0

        for idx in (1, 2, 3):
            assert batches[idx]["gt"] is None
            assert batches[idx]["meta"]["gt_path"] is None
            assert batches[idx]["meta"]["index"] == idx

    # Fail-fast checks
    with pytest.raises(ValueError, match="batch_size=1"):
        val_collate_fn([])
    with pytest.raises(ValueError, match="batch_size=1"):
        val_collate_fn([ds[0], ds[1]])

    bad_val_json = _write_json(tmp_path / "bad_val.json", [{"gt": "hr/001.png"}])
    with pytest.raises(ValueError, match="lq"):
        ValImageDataset(root_dir=root, json_path=bad_val_json)

    with pytest.raises(FileNotFoundError):
        ValImageDataset(root_dir=root, json_path=tmp_path / "missing_val.json")

    missing_img_json = _write_json(tmp_path / "missing_img.json", [{"lq": "lr/nope.png"}])
    ds_missing_img = ValImageDataset(root_dir=root, json_path=missing_img_json)
    with pytest.raises(FileNotFoundError):
        _ = ds_missing_img[0]

    # root_dir=None resolves relative to manifest JSON parent directory
    rel_dir = tmp_path / "rel_val"
    _write_rgb_image(rel_dir / "img_lq.png", 8, 8, seed=9)
    rel_json = _write_json(rel_dir / "manifest.json", [{"lq": "img_lq.png"}])
    ds_rel = ValImageDataset(root_dir=None, json_path=rel_json)
    assert ds_rel[0]["lq"].shape == (3, 8, 8)
