"""Tier 2 E2E tests — Area D: Extensible Metrics & Validation Suites Boundaries.

Covers boundary, edge, and negative contracts for F09, F24, and F30-F34.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

import abc
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def test_t2_d01_base_metric_abstract_enforcement() -> None:
    """F30: BaseMetric ABC enforces abstract compute() and provides default higher_is_better/to()."""
    import torch

    from metrics import BaseMetric

    assert issubclass(BaseMetric, abc.ABC)
    assert "compute" in BaseMetric.__abstractmethods__

    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]

    class _MissingComputeMetric(BaseMetric):
        higher_is_better = False

    with pytest.raises(TypeError):
        _MissingComputeMetric()  # type: ignore[abstract]

    class _MinimalConcreteMetric(BaseMetric):
        def compute(
            self,
            pred: Any,
            target: Any | None = None,
            meta: dict[str, Any] | None = None,
        ) -> float:
            del target, meta
            return float(pred.detach().float().mean().item())

    metric = _MinimalConcreteMetric()
    assert isinstance(metric, BaseMetric)
    assert metric.higher_is_better is True
    assert metric.to("cpu") is metric

    val = metric.compute(torch.zeros(1, 3, 8, 8, dtype=torch.float32), target=None, meta={"k": "v"})
    assert isinstance(val, float) and val == pytest.approx(0.0, abs=1e-7)


def test_t2_d02_val_collate_fn_rejects_non_unit_batch_sizes(
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
) -> None:
    """F24: val_collate_fn rejects batch_size != 1 directly and via DataLoader."""
    import torch
    from torch.utils.data import DataLoader

    from data import ValImageDataset, val_collate_fn

    with pytest.raises(ValueError, match="."):
        val_collate_fn([])

    sample = {
        "lq": torch.zeros(3, 16, 16, dtype=torch.float32),
        "gt": torch.zeros(3, 32, 32, dtype=torch.float32),
        "meta": {"lq_path": "lr/0.png", "gt_path": "hr/0.png", "index": 0},
    }
    with pytest.raises(ValueError, match="."):
        val_collate_fn([sample, sample])

    suite = val_suite_gt_label_factory(name="collate_b2", scale=2)
    ds = ValImageDataset(root_dir=suite["root_dir"], json_path=suite["json_path"])
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=val_collate_fn)
    with pytest.raises(ValueError, match="."):
        next(iter(loader))

    out = val_collate_fn([sample])
    assert out["lq"].shape == (1, 3, 16, 16)
    assert out["gt"].shape == (1, 3, 32, 32)
    assert isinstance(out["meta"], dict)
    assert out["meta"] == sample["meta"]


def test_t2_d03_val_dataset_rejects_invalid_and_missing_manifests(
    e2e_workdir: dict[str, Path],
    synthetic_image_factory: Callable[..., Path],
) -> None:
    """F24: ValImageDataset fails fast on invalid JSON entries and missing manifest/image files."""
    from data import ValImageDataset

    data_dir = e2e_workdir["data"] / "val_invalid"
    synthetic_image_factory(data_dir / "lr/ok.png", width=16, height=16, seed=1)
    synthetic_image_factory(data_dir / "hr/ok.png", width=32, height=32, seed=2)

    invalid_lq_payloads: dict[str, Any] = {
        "missing_lq.json": [{"gt": "hr/ok.png", "label": "no_lq"}],
        "null_lq.json": [{"lq": None, "gt": "hr/ok.png"}],
        "empty_lq.json": [{"lq": "", "gt": "hr/ok.png"}],
    }
    for filename, payload in invalid_lq_payloads.items():
        json_path = data_dir / filename
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="."):
            ds = ValImageDataset(root_dir=data_dir, json_path=json_path)
            _ = ds[0]

    non_list_json = data_dir / "non_list_obj.json"
    non_list_json.write_text(
        json.dumps({"lq": "lr/ok.png", "gt": "hr/ok.png"}),
        encoding="utf-8",
    )
    with pytest.raises((ValueError, TypeError)):
        ds = ValImageDataset(root_dir=data_dir, json_path=non_list_json)
        _ = ds[0]

    with pytest.raises(FileNotFoundError):
        ValImageDataset(root_dir=data_dir, json_path=data_dir / "does_not_exist.json")

    missing_lq_json = data_dir / "missing_lq_img.json"
    missing_lq_json.write_text(json.dumps([{"lq": "lr/missing_lq.png"}]), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ds = ValImageDataset(root_dir=data_dir, json_path=missing_lq_json)
        _ = ds[0]

    missing_gt_json = data_dir / "missing_gt_img.json"
    missing_gt_json.write_text(
        json.dumps([{"lq": "lr/ok.png", "gt": "hr/missing_gt.png"}]),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        ds = ValImageDataset(root_dir=data_dir, json_path=missing_gt_json)
        _ = ds[0]


def test_t2_d04_full_reference_metrics_reject_none_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F31, F32, F33: Full-reference PSNR, SSIM, and LPIPS metrics reject target=None."""
    import torch
    from tests.helpers.fake_backends import patch_lpips_offline

    from metrics import LPIPSMetric, PSNRMetric, SSIMMetric

    patch_lpips_offline(monkeypatch)
    pred = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    metrics = [
        PSNRMetric(crop_border=0, test_y_channel=False),
        PSNRMetric(crop_border=0, test_y_channel=True),
        SSIMMetric(crop_border=0, test_y_channel=False),
        SSIMMetric(crop_border=0, test_y_channel=True),
        LPIPSMetric(net="alex").to("cpu"),
    ]
    for metric in metrics:
        with pytest.raises(ValueError, match="."):
            metric.compute(pred, target=None)
        with pytest.raises(ValueError, match="."):
            metric.compute(pred, target=None, meta={"label": "ignored"})


def test_t2_d05_full_reference_metrics_reject_shape_and_channel_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F31, F32, F33: Full-reference metrics reject spatial and channel shape mismatches."""
    import torch
    from tests.helpers.fake_backends import patch_lpips_offline

    from metrics import LPIPSMetric, PSNRMetric, SSIMMetric

    patch_lpips_offline(monkeypatch)
    metrics = [
        PSNRMetric(crop_border=0, test_y_channel=False),
        PSNRMetric(crop_border=0, test_y_channel=True),
        SSIMMetric(crop_border=0, test_y_channel=False),
        SSIMMetric(crop_border=0, test_y_channel=True),
        LPIPSMetric(net="alex").to("cpu"),
    ]
    mismatched_pairs = [
        (torch.rand(1, 3, 64, 64, dtype=torch.float32), torch.rand(1, 3, 64, 68, dtype=torch.float32)),
        (torch.rand(1, 3, 68, 64, dtype=torch.float32), torch.rand(1, 3, 64, 64, dtype=torch.float32)),
        (torch.rand(1, 3, 64, 64, dtype=torch.float32), torch.rand(1, 1, 64, 64, dtype=torch.float32)),
        (torch.rand(1, 3, 64, 64, dtype=torch.float32), torch.rand(1, 4, 64, 64, dtype=torch.float32)),
    ]
    for metric in metrics:
        for pred, target in mismatched_pairs:
            with pytest.raises(ValueError, match="."):
                metric.compute(pred, target)


def test_t2_d06_psnr_metric_boundaries_and_crop_border() -> None:
    """F31: PSNRMetric matches analytical boundaries, rejects invalid crop_border, handles tight crop."""
    import torch

    from metrics import PSNRMetric

    zeros = torch.zeros(1, 3, 16, 16, dtype=torch.float32)
    ones = torch.ones(1, 3, 16, 16, dtype=torch.float32)

    assert PSNRMetric(crop_border=0, test_y_channel=False).compute(zeros, ones) == pytest.approx(
        0.0, abs=1e-5
    )
    expected_y_max_diff = 10.0 * math.log10((255.0 / 219.0) ** 2)
    assert PSNRMetric(crop_border=0, test_y_channel=True).compute(zeros, ones) == pytest.approx(
        expected_y_max_diff, abs=1e-3
    )

    for test_y in (False, True):
        with pytest.raises(ValueError, match="."):
            m_neg = PSNRMetric(crop_border=-1, test_y_channel=test_y)
            m_neg.compute(zeros, ones)

        for border, (h, w) in ((8, (16, 16)), (9, (16, 16)), (8, (16, 32))):
            a = torch.zeros(1, 3, h, w, dtype=torch.float32)
            with pytest.raises(ValueError, match="."):
                PSNRMetric(crop_border=border, test_y_channel=test_y).compute(a, a)

    # Tight valid crop_border=7 on 16x16 leaves exact 2x2 interior [7:9, 7:9].
    base = torch.full((1, 3, 16, 16), 0.5, dtype=torch.float32)
    pert = torch.zeros_like(base)
    pert[:, :, 7:9, 7:9] = 0.5
    inf_val = PSNRMetric(crop_border=7, test_y_channel=False).compute(pert, base)
    assert math.isinf(inf_val) and inf_val > 0.0

    pert[:, 0, 7:9, 7:9] = 0.6
    assert PSNRMetric(crop_border=7, test_y_channel=False).compute(pert, base) == pytest.approx(
        10.0 * math.log10(300.0), abs=1e-3
    )
    assert PSNRMetric(crop_border=7, test_y_channel=True).compute(pert, base) == pytest.approx(
        10.0 * math.log10((255.0**2) / (6.5481**2)), abs=1e-3
    )


def test_t2_d07_ssim_metric_11x11_window_and_crop_border_boundaries() -> None:
    """F32: SSIMMetric enforces 11x11 post-crop spatial floor and rejects negative crop_border."""
    import torch

    from metrics import SSIMMetric

    img16 = torch.rand(1, 3, 16, 16, dtype=torch.float32)
    for test_y in (False, True):
        with pytest.raises(ValueError, match="."):
            m_neg = SSIMMetric(crop_border=-1, test_y_channel=test_y)
            m_neg.compute(img16, img16)

        too_small_cases = [
            (0, 10, 10),
            (0, 10, 20),
            (0, 20, 10),
            (4, 18, 18),
            (8, 16, 16),
        ]
        for border, h, w in too_small_cases:
            t = torch.rand(1, 3, h, w, dtype=torch.float32)
            with pytest.raises(ValueError, match="."):
                SSIMMetric(crop_border=border, test_y_channel=test_y).compute(t, t)

    torch.manual_seed(7)
    img11 = torch.rand(1, 3, 11, 11, dtype=torch.float32)
    img11_pert = (img11 + 0.05 * torch.randn_like(img11)).clamp(0.0, 1.0)
    img19 = torch.rand(1, 3, 19, 19, dtype=torch.float32)
    img19_bad = torch.zeros_like(img19)
    img19_bad[:, :, 4:15, 4:15] = img19[:, :, 4:15, 4:15]

    for test_y in (False, True):
        m0 = SSIMMetric(crop_border=0, test_y_channel=test_y)
        assert m0.compute(img11, img11) == pytest.approx(1.0, abs=1e-6)
        v_pert = m0.compute(img11_pert, img11)
        assert type(v_pert) is float and math.isfinite(v_pert) and 0.0 < v_pert < 1.0

        m4 = SSIMMetric(crop_border=4, test_y_channel=test_y)
        assert m4.compute(img19_bad, img19) == pytest.approx(1.0, abs=1e-6)


def test_t2_d08_lpips_metric_no_grad_fp32_under_bf16_autocast_and_invalid_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F33: LPIPSMetric enforces FP32+no_grad under CPU bf16 autocast and rejects invalid net."""
    import torch
    from tests.helpers.fake_backends import patch_lpips_offline

    from metrics import LPIPSMetric

    patch_lpips_offline(monkeypatch)
    metric = LPIPSMetric(net="alex").to("cpu")
    torch.manual_seed(17)
    pred_ref = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    target_ref = (pred_ref + 0.1 * torch.randn_like(pred_ref)).clamp(0.0, 1.0)

    ref_val = metric.compute(pred_ref, target_ref)
    assert type(ref_val) is float and math.isfinite(ref_val) and ref_val > 0.0

    pred_grad = pred_ref.clone().requires_grad_(True)
    target_grad = target_ref.clone().requires_grad_(True)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        val_bf16 = metric.compute(pred_grad, target_grad)

    assert type(val_bf16) is float and math.isfinite(val_bf16)
    assert val_bf16 == pytest.approx(ref_val, abs=1e-5)
    assert pred_grad.grad is None and target_grad.grad is None
    assert torch.is_grad_enabled() is True

    with pytest.raises((ValueError, KeyError)):
        bad_metric = LPIPSMetric(net="unknown_net").to("cpu")
        bad_metric.compute(pred_ref, target_ref)


def test_t2_d09_custom_metric_missing_meta_key_and_edge_meta_types(
    e2e_workdir: dict[str, Path],
    synthetic_image_factory: Callable[..., Path],
) -> None:
    """F09, F24, F30, F34: Custom metric strict/lenient missing meta & ValImageDataset edge meta."""
    import hydra.utils
    import torch
    from torch.utils.data import DataLoader

    from data import ValImageDataset, val_collate_fn

    strict_m = hydra.utils.instantiate(
        {"_target_": "tests.e2e.custom_metrics.MockOCRAMetric", "label_key": "label", "strict": True}
    )
    lenient_m = hydra.utils.instantiate(
        {
            "_target_": "tests.e2e.custom_metrics.MockOCRAMetric",
            "label_key": "label",
            "strict": False,
        }
    )

    pred = torch.full((1, 3, 16, 16), 0.5, dtype=torch.float32)
    with pytest.raises(KeyError):
        strict_m.compute(pred, target=None, meta=None)
    with pytest.raises(KeyError):
        strict_m.compute(pred, target=None, meta={"other": "abc"})
    assert lenient_m.compute(pred, target=None, meta=None) == pytest.approx(0.25, abs=1e-5)
    assert lenient_m.compute(pred, target=None, meta={"other": "abc"}) == pytest.approx(
        0.25, abs=1e-5
    )

    root = e2e_workdir["data"] / "val_edge_meta"
    synthetic_image_factory(root / "lr/0000.png", width=16, height=16, seed=10)
    synthetic_image_factory(root / "lr/0001.png", width=16, height=16, seed=11)

    entries: list[dict[str, Any]] = [
        {
            "lq": "lr/0000.png",
            "nullable_val": None,
            "bool_false": False,
            "zero_int": 0,
            "empty_list": [],
            "empty_dict": {},
            "label": "中文标签2026",
            "nested": {"mix": [None, False, 0, "嵌套"], "inner": {}},
        },
        {
            "lq": "lr/0001.png",
            "gt": None,
            "tag": "no_label_entry",
        },
    ]
    manifest = root / "edge_meta.json"
    manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    loader = DataLoader(
        ValImageDataset(root_dir=root, json_path=manifest),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=val_collate_fn,
    )
    batches = list(loader)
    assert len(batches) == 2

    b0, b1 = batches[0], batches[1]
    meta0 = b0["meta"]
    assert meta0["nullable_val"] is None
    assert meta0["bool_false"] is False and type(meta0["bool_false"]) is bool
    assert meta0["zero_int"] == 0 and type(meta0["zero_int"]) is int
    assert meta0["empty_list"] == [] and isinstance(meta0["empty_list"], list)
    assert meta0["empty_dict"] == {} and isinstance(meta0["empty_dict"], dict)
    assert meta0["label"] == "中文标签2026"
    assert meta0["nested"] == {"mix": [None, False, 0, "嵌套"], "inner": {}}

    score0 = strict_m.compute(b0["lq"], target=b0["gt"], meta=meta0)
    assert isinstance(score0, float) and math.isfinite(score0) and 0.0 <= score0 <= 1.0

    assert b1["gt"] is None
    assert "label" not in b1["meta"]
    with pytest.raises(KeyError):
        strict_m.compute(b1["lq"], target=b1["gt"], meta=b1["meta"])
    score1 = lenient_m.compute(b1["lq"], target=b1["gt"], meta=b1["meta"])
    assert isinstance(score1, float) and math.isfinite(score1) and 0.0 <= score1 <= 1.0
