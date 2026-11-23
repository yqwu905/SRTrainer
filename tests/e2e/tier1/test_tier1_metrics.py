"""Tier 1 E2E tests — Area D: Extensible Metrics & Validation Suites.

Covers features F02, F09, F24, F30-F34 and User Requirement #3.
All non-stdlib imports (besides ``pytest``) are deferred inside test functions.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable
from typing import Any

import pytest


def _reference_basicsr_ssim(
    pred: Any,
    target: Any,
    *,
    crop_border: int = 0,
    test_y_channel: bool = False,
) -> float:
    """Independent pure-PyTorch float64 BasicSR/MATLAB 11x11 sigma=1.5 valid-conv SSIM oracle."""
    import torch
    import torch.nn.functional as F

    p = pred.detach().to(dtype=torch.float64)
    t = target.detach().to(dtype=torch.float64)

    if test_y_channel:
        # BasicSR / MATLAB ITU-R BT.601 on [0, 1] RGB -> [16, 235] Y channel.
        coeffs = torch.tensor([65.481, 128.553, 24.966], dtype=torch.float64).view(1, 3, 1, 1)
        p = 16.0 + (p * coeffs).sum(dim=1, keepdim=True)
        t = 16.0 + (t * coeffs).sum(dim=1, keepdim=True)
    else:
        p = p * 255.0
        t = t * 255.0

    if crop_border > 0:
        p = p[:, :, crop_border:-crop_border, crop_border:-crop_border]
        t = t[:, :, crop_border:-crop_border, crop_border:-crop_border]

    coords = torch.arange(11, dtype=torch.float64) - 5.0
    g1d = torch.exp(-(coords**2) / (2.0 * (1.5**2)))
    g1d = g1d / g1d.sum()
    window = torch.outer(g1d, g1d).view(1, 1, 11, 11)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    channels = p.shape[1]
    w_ch = window.repeat(channels, 1, 1, 1)
    mu1 = F.conv2d(p, w_ch, padding=0, groups=channels)
    mu2 = F.conv2d(t, w_ch, padding=0, groups=channels)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(p * p, w_ch, padding=0, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(t * t, w_ch, padding=0, groups=channels) - mu2_sq
    sigma12 = F.conv2d(p * t, w_ch, padding=0, groups=channels) - mu1_mu2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


def test_t1_d01_metrics_reexports_and_base_metric_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F02, F30: metrics re-exports, ABC protocol, higher_is_better flags, .to()."""
    from tests.helpers.fake_backends import patch_lpips_offline

    from metrics import BaseMetric, LPIPSMetric, PSNRMetric, SSIMMetric

    patch_lpips_offline(monkeypatch)
    assert issubclass(BaseMetric, abc.ABC)
    assert BaseMetric.higher_is_better is True

    psnr = PSNRMetric()
    ssim = SSIMMetric()
    lpips_m = LPIPSMetric(net="alex")

    for metric, expected_higher in ((psnr, True), (ssim, True), (lpips_m, False)):
        assert isinstance(metric, BaseMetric)
        assert metric.higher_is_better is expected_higher
        assert metric.to("cpu") is metric


def test_t1_d02_val_dataset_and_collate_with_gt_and_meta_passthrough(
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
) -> None:
    """F24, Req #3: ValImageDataset + val_collate_fn handle variable sizes, GT, and raw meta."""
    import torch
    from torch.utils.data import DataLoader

    from data import ValImageDataset, val_collate_fn

    suite = val_suite_gt_label_factory(name="paired_meta", scale=2)
    ds = ValImageDataset(root_dir=suite["root_dir"], json_path=suite["json_path"])
    assert len(ds) == 3

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=val_collate_fn)
    seen_shapes: list[tuple[int, int]] = []
    for idx, batch in enumerate(loader):
        lq_h, lq_w = suite["lq_sizes"][idx]
        seen_shapes.append((lq_h, lq_w))
        assert batch["lq"].shape == (1, 3, lq_h, lq_w)
        assert batch["lq"].dtype == torch.float32
        assert float(batch["lq"].min()) >= 0.0 and float(batch["lq"].max()) <= 1.0

        assert batch["gt"] is not None
        assert batch["gt"].shape == (1, 3, lq_h * 2, lq_w * 2)
        assert batch["gt"].dtype == torch.float32

        meta = batch["meta"]
        assert isinstance(meta, dict), "val_collate_fn must preserve raw Python dict for meta"
        expected = suite["entries"][idx]
        assert isinstance(meta["label"], str) and meta["label"] == expected["label"]
        assert meta["nested"] == expected["nested"]
        assert isinstance(meta["lq_path"], str) and meta["lq_path"]
        assert isinstance(meta["gt_path"], str) and meta["gt_path"]
        assert meta["index"] == idx

    assert len(set(seen_shapes)) == 3, "Variable resolutions must be preserved without forced crop"


def test_t1_d03_val_dataset_and_collate_gt_less_autodetect(
    val_suite_gt_less_meta_factory: Callable[..., dict[str, Any]],
) -> None:
    """F24, Req #3: ValImageDataset auto-detects absent/null/'' gt as None while preserving meta."""
    import torch
    from torch.utils.data import DataLoader

    from data import ValImageDataset, val_collate_fn

    suite = val_suite_gt_less_meta_factory(name="nogt_meta")
    ds = ValImageDataset(root_dir=suite["root_dir"], json_path=suite["json_path"])
    assert len(ds) == 3

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=val_collate_fn)
    batches = list(loader)
    assert len(batches) == 3

    for idx, batch in enumerate(batches):
        lq_h, lq_w = suite["lq_sizes"][idx]
        assert batch["lq"].shape == (1, 3, lq_h, lq_w)
        assert batch["lq"].dtype == torch.float32
        assert batch["gt"] is None
        meta = batch["meta"]
        assert isinstance(meta, dict)
        assert meta["gt_path"] is None
        assert meta["label"] == suite["entries"][idx]["label"]
        assert meta["index"] == idx

    # Verify native Python types (bool, int, None, empty containers, Unicode) survived verbatim.
    meta0 = batches[0]["meta"]
    assert meta0["flag"] is False
    assert meta0["count"] == 0
    assert meta0["empty_list"] == []
    assert meta0["empty_dict"] == {}
    assert meta0["nullable"] is None
    assert batches[1]["meta"]["label"] == "中文标签2026"
    assert batches[1]["meta"]["nested"] == {"domain": "doc", "scores": [1, 2.5]}


def test_t1_d04_psnr_metric_rgb_and_basicsr_bt601_y_channel_exact_oracle() -> None:
    """F31: PSNRMetric exact parity with BasicSR/MATLAB RGB and ITU-R BT.601 Y-channel formulas."""
    import torch

    from metrics import PSNRMetric

    psnr_rgb = PSNRMetric(crop_border=0, test_y_channel=False)
    psnr_y = PSNRMetric(crop_border=0, test_y_channel=True)

    base = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)

    # Identical tensors -> +inf
    val_inf_rgb = psnr_rgb.compute(base, base)
    val_inf_y = psnr_y.compute(base, base)
    assert isinstance(val_inf_rgb, float) and math.isinf(val_inf_rgb) and val_inf_rgb > 0
    assert isinstance(val_inf_y, float) and math.isinf(val_inf_y) and val_inf_y > 0

    # RGB: dR=0.1 -> mse = 25.5^2 / 3 -> PSNR = 10 * log10(300) = 24.77121255 dB
    pert_r = base.clone()
    pert_r[:, 0] = 0.6
    assert psnr_rgb.compute(pert_r, base) == pytest.approx(10.0 * math.log10(300.0), abs=1e-3)

    # BT.601 Y-channel: Y = 16.0 + 65.481*R + 128.553*G + 24.966*B
    expected_y = {
        0: 10.0 * math.log10(255.0**2 / (6.5481**2)),
        1: 10.0 * math.log10(255.0**2 / (12.8553**2)),
        2: 10.0 * math.log10(255.0**2 / (2.4966**2)),
    }
    for ch, exp_db in expected_y.items():
        pert = base.clone()
        pert[:, ch] = 0.6
        got = psnr_y.compute(pert, base)
        assert got == pytest.approx(exp_db, abs=1e-3), (
            f"BT.601 Y-channel PSNR mismatch on channel {ch}: got {got}, expected {exp_db}"
        )

    # crop_border=4 ignores outer 4-pixel corruption.
    corrupted = base.clone()
    corrupted[:, :, :4, :] = 0.0
    corrupted[:, :, -4:, :] = 1.0
    corrupted[:, :, :, :4] = 0.0
    corrupted[:, :, :, -4:] = 1.0
    assert psnr_rgb.compute(corrupted, base) < 40.0
    cropped_psnr = PSNRMetric(crop_border=4, test_y_channel=False).compute(corrupted, base)
    assert math.isinf(cropped_psnr) and cropped_psnr > 0


def test_t1_d05_ssim_metric_rgb_and_basicsr_bt601_y_channel_exact_oracle() -> None:
    """F32: SSIMMetric exact BasicSR/MATLAB BT.601 Y & RGB analytical + reference parity."""
    import torch

    from metrics import SSIMMetric

    ssim_rgb = SSIMMetric(crop_border=0, test_y_channel=False)
    ssim_y = SSIMMetric(crop_border=0, test_y_channel=True)

    torch.manual_seed(3)
    img = torch.rand(1, 3, 32, 32, dtype=torch.float32)

    # Identical -> 1.0
    assert ssim_rgb.compute(img, img) == pytest.approx(1.0, abs=1e-6)
    assert ssim_y.compute(img, img) == pytest.approx(1.0, abs=1e-6)

    # Analytical constant (0,0,0) vs (1,1,1): verifies exact +16.0 BT.601 offset & C1.
    zeros = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    ones = torch.ones(1, 3, 32, 32, dtype=torch.float32)
    c1 = (0.01 * 255.0) ** 2
    expected_rgb_const = c1 / (255.0**2 + c1)
    expected_y_const = (2.0 * 16.0 * 235.0 + c1) / (16.0**2 + 235.0**2 + c1)
    assert ssim_rgb.compute(zeros, ones) == pytest.approx(expected_rgb_const, abs=1e-5)
    assert ssim_y.compute(zeros, ones) == pytest.approx(expected_y_const, abs=1e-4)

    # Parity against independent 11x11 sigma=1.5 valid-conv reference on noisy textured image.
    noisy_small = (img + 0.03 * torch.randn_like(img)).clamp(0.0, 1.0)
    noisy_large = (img + 0.20 * torch.randn_like(img)).clamp(0.0, 1.0)

    ref_rgb = _reference_basicsr_ssim(noisy_small, img, crop_border=0, test_y_channel=False)
    ref_y_crop = _reference_basicsr_ssim(noisy_small, img, crop_border=4, test_y_channel=True)
    assert ssim_rgb.compute(noisy_small, img) == pytest.approx(ref_rgb, abs=1e-4)
    assert SSIMMetric(crop_border=4, test_y_channel=True).compute(
        noisy_small, img
    ) == pytest.approx(ref_y_crop, abs=1e-4)

    # Symmetry & monotonicity.
    assert ssim_rgb.compute(noisy_small, img) == pytest.approx(
        ssim_rgb.compute(img, noisy_small), abs=1e-6
    )
    assert ssim_rgb.compute(noisy_small, img) > ssim_rgb.compute(noisy_large, img)

    # crop_border=4 ignores outer border corruption.
    border_bad = img.clone()
    border_bad[:, :, :4, :] = 0.0
    border_bad[:, :, -4:, :] = 1.0
    border_bad[:, :, :, :4] = 0.0
    border_bad[:, :, :, -4:] = 1.0
    assert SSIMMetric(crop_border=4, test_y_channel=False).compute(
        border_bad, img
    ) == pytest.approx(1.0, abs=1e-6)


def test_t1_d06_lpips_metric_offline_cached_and_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F33: LPIPSMetric is lower-is-better, monotonic, cached (offline mocked factory)."""
    import torch
    from tests.helpers.fake_backends import patch_lpips_offline

    from metrics import LPIPSMetric

    patch_lpips_offline(monkeypatch)
    metric = LPIPSMetric(net="alex").to("cpu")
    assert metric.higher_is_better is False

    torch.manual_seed(9)
    base = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    small_pert = (base + 0.03 * torch.randn_like(base)).clamp(0.0, 1.0)
    large_pert = (base + 0.30 * torch.randn_like(base)).clamp(0.0, 1.0)

    same_score = metric.compute(base, base)
    assert isinstance(same_score, float) and abs(same_score) < 1e-5

    score_small = metric.compute(small_pert, base)
    score_large = metric.compute(large_pert, base)
    assert 0.0 < score_small < score_large

    # Symmetry & model-weight caching (repeated calls on pretrained=False give identical float).
    score_rev = metric.compute(base, small_pert)
    assert abs(score_small - score_rev) < 1e-5
    assert metric.compute(small_pert, base) == score_small


def test_t1_d07_custom_base_metric_via_hydra_target_with_meta_and_no_gt(
    val_suite_gt_less_meta_factory: Callable[..., dict[str, Any]],
) -> None:
    """F09, F24, F30, F34, Req #3: Custom BaseMetric via Hydra consumes meta without GT."""
    import hydra.utils
    from torch.utils.data import DataLoader

    from data import ValImageDataset, val_collate_fn
    from metrics import BaseMetric

    metric = hydra.utils.instantiate(
        {
            "_target_": "tests.e2e.custom_metrics.MockOCRAMetric",
            "label_key": "label",
            "strict": True,
        }
    )
    assert isinstance(metric, BaseMetric)
    assert metric.higher_is_better is True
    assert metric.to("cpu") is metric

    suite = val_suite_gt_less_meta_factory(name="ocr_nogt")
    ds = ValImageDataset(root_dir=suite["root_dir"], json_path=suite["json_path"])
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=val_collate_fn)

    scores: list[float] = []
    for batch in loader:
        assert batch["gt"] is None
        s1 = metric.compute(batch["lq"], target=None, meta=batch["meta"])
        s2 = metric.compute(batch["lq"], target=None, meta=batch["meta"])
        assert isinstance(s1, float) and 0.0 <= s1 <= 1.0
        assert s1 == pytest.approx(s2, abs=1e-7)
        scores.append(s1)

    assert len(scores) == 3
    seen_labels = [m["label"] for m in metric.seen_metas if m is not None]
    assert "中文标签2026" in seen_labels


def test_t1_d08_hydra_val_suites_config_structure_and_co_location(
    hydra_compose: Callable[..., Any],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    val_suite_gt_less_meta_factory: Callable[..., dict[str, Any]],
) -> None:
    """F09, Req #3: metric.val_suites co-locates datasets & heterogeneous metrics cleanly."""
    import hydra.utils
    from omegaconf import OmegaConf

    from metrics import BaseMetric

    cfg = hydra_compose("train")
    assert "val_suites" in cfg.metric
    assert not ({"val", "val_datasets", "val_suites"} & set(cfg.data.keys()))

    gt_suite = val_suite_gt_label_factory(name="gt_bench", scale=2)
    nogt_suite = val_suite_gt_less_meta_factory(name="ocr_bench")

    multi_metric_cfg = OmegaConf.create(
        {
            "primary_dataset": "ocr_bench",
            "primary_metric": "ocr_a",
            "val_suites": {
                "gt_bench": {
                    "dataset": {
                        "root_dir": gt_suite["root_dir"],
                        "json_path": gt_suite["json_path"],
                    },
                    "metrics": {
                        "psnr": {
                            "_target_": "metrics.PSNRMetric",
                            "crop_border": 2,
                            "test_y_channel": True,
                        },
                        "ssim": {
                            "_target_": "metrics.SSIMMetric",
                            "crop_border": 2,
                            "test_y_channel": True,
                        },
                    },
                },
                "ocr_bench": {
                    "dataset": {
                        "root_dir": nogt_suite["root_dir"],
                        "json_path": nogt_suite["json_path"],
                    },
                    "metrics": {
                        "ocr_a": {
                            "_target_": "tests.e2e.custom_metrics.MockOCRAMetric",
                            "label_key": "label",
                        },
                    },
                },
            },
        }
    )

    assert multi_metric_cfg.primary_dataset in multi_metric_cfg.val_suites
    for suite_cfg in multi_metric_cfg.val_suites.values():
        assert "dataset" in suite_cfg and "metrics" in suite_cfg
        for m_cfg in suite_cfg.metrics.values():
            inst = hydra.utils.instantiate(m_cfg)
            assert isinstance(inst, BaseMetric)


def test_t1_d09_custom_lower_is_better_metric_via_hydra_target() -> None:
    """F30, F34: Custom lower-is-better BaseMetric instantiates via Hydra and computes scores."""
    import hydra.utils
    import torch

    from metrics import BaseMetric

    metric = hydra.utils.instantiate(
        {
            "_target_": "tests.e2e.custom_metrics.MockLowerIsBetterMetric",
            "target_mean": 0.5,
            "weight": 2.0,
        }
    )
    assert isinstance(metric, BaseMetric)
    assert metric.higher_is_better is False
    assert metric.to("cpu") is metric

    pred = torch.full((1, 3, 16, 16), 0.7, dtype=torch.float32)
    val = metric.compute(pred, target=None, meta={"id": 1})
    assert isinstance(val, float)
    assert val == pytest.approx(0.4, abs=1e-5)

    # Sequence mode returns deterministic ordered values.
    seq_metric = hydra.utils.instantiate(
        {
            "_target_": "tests.e2e.custom_metrics.MockLowerIsBetterMetric",
            "sequence": [0.5, 0.2, 0.4],
        }
    )
    assert isinstance(seq_metric, BaseMetric) and seq_metric.higher_is_better is False
    got = [seq_metric.compute(pred) for _ in range(3)]
    assert got == [pytest.approx(0.5), pytest.approx(0.2), pytest.approx(0.4)]
