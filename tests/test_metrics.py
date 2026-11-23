"""Unit tests for the Milestone 3 extensible metrics suite (BaseMetric, PSNR, SSIM, LPIPS)."""

from __future__ import annotations

import copy
import math
import socket
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pytest
import scipy.signal
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import metrics
from metrics import BaseMetric, LPIPSMetric, PSNRMetric, SSIMMetric
from metrics.psnr_ssim import rgb_to_y_bt601

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


# ---------------------------------------------------------------------------
# Independent NumPy / SciPy reference oracles & Hydra / custom metric helpers
# ---------------------------------------------------------------------------


def _compose(config_name: str, overrides: list[str] | None = None):
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return hydra.compose(config_name=config_name, overrides=overrides or [])


def _ref_rgb_to_y_bt601_np(img_nchw: np.ndarray) -> np.ndarray:
    img64 = img_nchw.astype(np.float64)
    return (
        65.481 * img64[:, 0:1, :, :]
        + 128.553 * img64[:, 1:2, :, :]
        + 24.966 * img64[:, 2:3, :, :]
        + 16.0
    )


def _ref_psnr_np(pred: np.ndarray, target: np.ndarray, crop_border: int, y_only: bool) -> float:
    if crop_border > 0:
        pred = pred[:, :, crop_border:-crop_border, crop_border:-crop_border]
        target = target[:, :, crop_border:-crop_border, crop_border:-crop_border]
    x1 = _ref_rgb_to_y_bt601_np(pred) if y_only else pred.astype(np.float64) * 255.0
    x2 = _ref_rgb_to_y_bt601_np(target) if y_only else target.astype(np.float64) * 255.0
    scores: list[float] = []
    for idx in range(x1.shape[0]):
        mse = float(np.mean((x1[idx] - x2[idx]) ** 2))
        scores.append(float("inf") if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse))
    return float(np.mean(scores))


def _ref_ssim_scipy(pred: np.ndarray, target: np.ndarray, crop_border: int, y_only: bool) -> float:
    if crop_border > 0:
        pred = pred[:, :, crop_border:-crop_border, crop_border:-crop_border]
        target = target[:, :, crop_border:-crop_border, crop_border:-crop_border]
    x1 = _ref_rgb_to_y_bt601_np(pred) if y_only else pred.astype(np.float64) * 255.0
    x2 = _ref_rgb_to_y_bt601_np(target) if y_only else target.astype(np.float64) * 255.0

    g1d = scipy.signal.windows.gaussian(11, std=1.5)
    g1d = g1d / g1d.sum()
    kernel = np.outer(g1d, g1d)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    vals: list[float] = []
    for n in range(x1.shape[0]):
        for c in range(x1.shape[1]):
            i1 = x1[n, c]
            i2 = x2[n, c]
            mu1 = scipy.signal.correlate2d(i1, kernel, mode="valid")
            mu2 = scipy.signal.correlate2d(i2, kernel, mode="valid")
            s1 = scipy.signal.correlate2d(i1 * i1, kernel, mode="valid") - mu1 * mu1
            s2 = scipy.signal.correlate2d(i2 * i2, kernel, mode="valid") - mu2 * mu2
            s12 = scipy.signal.correlate2d(i1 * i2, kernel, mode="valid") - mu1 * mu2
            smap = ((2.0 * mu1 * mu2 + c1) * (2.0 * s12 + c2)) / (
                (mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)
            )
            vals.append(float(smap.mean()))
    return float(np.mean(vals))


def _block_network_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def _deny(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"Unexpected network access during offline test: {args!r}, {kwargs!r}")

    monkeypatch.setattr(socket, "create_connection", _deny)
    monkeypatch.setattr(urllib.request, "urlopen", _deny)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", _deny)


class _NoRefMeanLuminanceMetric(BaseMetric):
    """Minimal no-reference custom metric inheriting BaseMetric defaults."""

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del target, meta
        return float(pred.float().mean().item())


class _MockOCRAMetric(BaseMetric):
    """Custom auxiliary-label metric (OCR-A mock) consuming ``meta["label"]`` with ``target=None``.

    Decodes uppercase ASCII character codes stored in the top-left pixels of ``pred``
    (normalized to ``[0, 1]`` via ``ord(ch) / 255.0``) and compares against ``meta[label_key]``.
    """

    higher_is_better: bool = True

    def __init__(self, label_key: str = "label", case_sensitive: bool = True) -> None:
        self.label_key = label_key
        self.case_sensitive = case_sensitive

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del target
        if meta is None or self.label_key not in meta:
            raise ValueError(f"Expected metadata key {self.label_key!r} in meta.")
        label = meta[self.label_key]
        if not isinstance(label, str) or len(label) == 0:
            raise ValueError(f"Expected non-empty string label, got {label!r}.")

        codes = torch.round(pred[0, 0, 0, : len(label)].detach().float() * 255.0).to(torch.int64)
        recognized = "".join(chr(int(c)) for c in codes.tolist())
        expected = label
        if not self.case_sensitive:
            recognized = recognized.lower()
            expected = expected.lower()
        matches = sum(a == b for a, b in zip(recognized, expected, strict=True))
        return float(matches / len(expected))


# ---------------------------------------------------------------------------
# Package exports & BaseMetric extensibility
# ---------------------------------------------------------------------------


def test_metrics_package_exports() -> None:
    assert metrics.__all__ == ["BaseMetric", "PSNRMetric", "SSIMMetric", "LPIPSMetric"]
    assert metrics.BaseMetric is BaseMetric
    assert metrics.PSNRMetric is PSNRMetric
    assert metrics.SSIMMetric is SSIMMetric
    assert metrics.LPIPSMetric is LPIPSMetric
    assert issubclass(PSNRMetric, BaseMetric)
    assert issubclass(SSIMMetric, BaseMetric)
    assert issubclass(LPIPSMetric, BaseMetric)


def test_base_metric_abc_defaults_and_custom_ocr_extensibility() -> None:
    # ABC cannot be instantiated directly or without implementing compute()
    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]

    class _IncompleteMetric(BaseMetric):
        pass

    with pytest.raises(TypeError):
        _IncompleteMetric()  # type: ignore[abstract]

    # Default higher_is_better=True & chainable no-op to(device)
    no_ref = _NoRefMeanLuminanceMetric()
    assert no_ref.higher_is_better is True
    assert no_ref.to("cpu") is no_ref
    assert no_ref.to(torch.device("cpu")) is no_ref
    score = no_ref.compute(torch.full((1, 3, 8, 8), 0.25))
    assert type(score) is float
    assert score == pytest.approx(0.25, abs=1e-7)

    # Custom OCR-A mock metric consuming meta["label"] with target=None (direct + Hydra instantiate)
    target_text = "SRTRAINER"
    pred_exact = torch.zeros(1, 3, 16, 16, dtype=torch.float32)
    for idx, ch in enumerate(target_text):
        pred_exact[0, 0, 0, idx] = ord(ch) / 255.0

    pred_partial = pred_exact.clone()
    pred_partial[0, 0, 0, 0] = ord("X") / 255.0

    meta = {"label": target_text, "lq_path": "lr/001.png", "gt_path": None, "index": 0}
    meta_snapshot = copy.deepcopy(meta)

    ocr_direct = _MockOCRAMetric(label_key="label")
    assert ocr_direct.to("cpu") is ocr_direct
    exact_score = ocr_direct.compute(pred_exact, target=None, meta=meta)
    partial_score = ocr_direct.compute(pred_partial, target=None, meta=meta)
    assert type(exact_score) is float
    assert exact_score == pytest.approx(1.0, abs=1e-12)
    assert partial_score == pytest.approx((len(target_text) - 1) / len(target_text), abs=1e-12)
    assert meta == meta_snapshot

    # Missing meta or missing label key raises ValueError
    with pytest.raises(ValueError):
        ocr_direct.compute(pred_exact, target=None, meta=None)
    with pytest.raises(ValueError):
        ocr_direct.compute(pred_exact, target=None, meta={"lq_path": "lr/001.png"})

    # Hydra _target_ instantiation of custom BaseMetric subclass
    custom_cfg = OmegaConf.create(
        {
            "_target_": f"{_MockOCRAMetric.__module__}.{_MockOCRAMetric.__qualname__}",
            "label_key": "text",
            "case_sensitive": False,
        }
    )
    ocr_hydra = instantiate(custom_cfg)
    assert isinstance(ocr_hydra, BaseMetric)
    assert isinstance(ocr_hydra, _MockOCRAMetric)
    assert ocr_hydra.compute(pred_exact, target=None, meta={"text": target_text.lower()}) == 1.0


# ---------------------------------------------------------------------------
# rgb_to_y_bt601 tests
# ---------------------------------------------------------------------------


def test_rgb_to_y_bt601_analytical_extremes_primaries_and_no_clamping() -> None:
    cases = [
        ((0.0, 0.0, 0.0), 16.0),
        ((1.0, 1.0, 1.0), 235.0),
        ((1.0, 0.0, 0.0), 81.481),
        ((0.0, 1.0, 0.0), 144.553),
        ((0.0, 0.0, 1.0), 40.966),
        # Verify linear behavior outside [0, 1] (no clamping)
        ((1.5, 1.5, 1.5), 16.0 + 219.0 * 1.5),
        ((-0.1, 0.0, 0.0), 16.0 - 6.5481),
    ]
    for (r, g, b), expected_y in cases:
        img = (
            torch.tensor([r, g, b], dtype=torch.float64).view(1, 3, 1, 1).expand(1, 3, 4, 5).clone()
        )
        y = rgb_to_y_bt601(img)
        assert y.shape == (1, 1, 4, 5)
        assert y.dtype == torch.float64
        assert y.device == img.device
        assert torch.allclose(y, torch.full_like(y, expected_y), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rgb_to_y_bt601_random_parity_dtypes_autocast_and_non_mutation(dtype: torch.dtype) -> None:
    gen = torch.Generator().manual_seed(2026)
    img = torch.rand(2, 3, 29, 37, generator=gen, dtype=torch.float32).to(dtype=dtype)
    img_before = img.clone()

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = rgb_to_y_bt601(img)

    expected = _ref_rgb_to_y_bt601_np(img.to(dtype=torch.float64).numpy())
    assert y.shape == (2, 1, 29, 37)
    assert y.dtype == torch.float64
    assert y.is_contiguous()
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-12, rtol=0.0)
    assert torch.equal(img, img_before)


def test_rgb_to_y_bt601_validation_errors() -> None:
    for bad_obj in (np.zeros((1, 3, 8, 8), dtype=np.float32), None, [[[[0.0]]]]):
        with pytest.raises(TypeError):
            rgb_to_y_bt601(bad_obj)  # type: ignore[arg-type]

    invalid_tensors = [
        torch.rand(3, 8, 8),  # 3D
        torch.rand(1, 1, 3, 8, 8),  # 5D
        torch.rand(1, 1, 8, 8),  # C=1
        torch.rand(1, 4, 8, 8),  # C=4
        torch.empty(0, 3, 8, 8),  # N=0
        torch.empty(1, 3, 0, 8),  # H=0
        torch.empty(1, 3, 8, 0),  # W=0
        torch.zeros(1, 3, 8, 8, dtype=torch.uint8),
        torch.zeros(1, 3, 8, 8, dtype=torch.int32),
        torch.zeros(1, 3, 8, 8, dtype=torch.int64),
        torch.zeros(1, 3, 8, 8, dtype=torch.bool),
        torch.zeros(1, 3, 8, 8, dtype=torch.complex64),
    ]
    for tensor in invalid_tensors:
        with pytest.raises(ValueError):
            rgb_to_y_bt601(tensor)


# ---------------------------------------------------------------------------
# PSNRMetric & SSIMMetric accuracy, identities, edge cases & validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("crop_border", [0, 2, 4])
@pytest.mark.parametrize("test_y_channel", [False, True])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_psnr_and_ssim_oracle_parity(
    crop_border: int,
    test_y_channel: bool,
    batch_size: int,
) -> None:
    gen = torch.Generator().manual_seed(100 + crop_border * 10 + int(test_y_channel) + batch_size)
    target = torch.rand(batch_size, 3, 37, 43, generator=gen, dtype=torch.float32)
    noise = 0.08 * torch.randn(batch_size, 3, 37, 43, generator=gen, dtype=torch.float32)
    pred = (target + noise).clamp(0.0, 1.0)

    psnr_metric = PSNRMetric(crop_border=crop_border, test_y_channel=test_y_channel)
    ssim_metric = SSIMMetric(crop_border=crop_border, test_y_channel=test_y_channel)
    assert psnr_metric.higher_is_better is True
    assert ssim_metric.higher_is_better is True

    psnr_val = psnr_metric.compute(pred, target)
    ssim_val = ssim_metric.compute(pred, target)

    expected_psnr = _ref_psnr_np(pred.numpy(), target.numpy(), crop_border, test_y_channel)
    expected_ssim = _ref_ssim_scipy(pred.numpy(), target.numpy(), crop_border, test_y_channel)

    assert type(psnr_val) is float
    assert type(ssim_val) is float
    assert psnr_val == pytest.approx(expected_psnr, abs=1e-12)
    assert ssim_val == pytest.approx(expected_ssim, abs=1e-12)


def test_psnr_and_ssim_analytical_bt601_studio_swing_identities() -> None:
    # 1. Constant RGB 0.0 vs 15/255 -> exact 20 * log10(17) dB
    zeros = torch.zeros(1, 3, 24, 24, dtype=torch.float64)
    step = torch.full((1, 3, 24, 24), 15.0 / 255.0, dtype=torch.float64)
    assert PSNRMetric(crop_border=0, test_y_channel=False).compute(zeros, step) == pytest.approx(
        20.0 * math.log10(17.0), abs=1e-12
    )

    # 2. Black (0, 0, 0) vs White (1, 1, 1): distinguishes BT.601 studio swing [16, 235] from [0, 255]
    black = torch.zeros(1, 3, 24, 24, dtype=torch.float64)
    white = torch.ones(1, 3, 24, 24, dtype=torch.float64)
    c1 = (0.01 * 255.0) ** 2

    assert PSNRMetric(test_y_channel=False).compute(black, white) == pytest.approx(0.0, abs=1e-12)
    assert PSNRMetric(test_y_channel=True).compute(black, white) == pytest.approx(
        20.0 * math.log10(255.0 / 219.0), abs=1e-12
    )
    assert SSIMMetric(test_y_channel=False).compute(black, white) == pytest.approx(
        c1 / (255.0**2 + c1), abs=1e-12
    )
    assert SSIMMetric(test_y_channel=True).compute(black, white) == pytest.approx(
        (2.0 * 16.0 * 235.0 + c1) / (16.0**2 + 235.0**2 + c1), abs=1e-12
    )

    # 3. Pure Red (1, 0, 0) vs Pure Blue (0, 0, 1)
    red = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).view(1, 3, 1, 1).expand(1, 3, 24, 24)
    blue = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).view(1, 3, 1, 1).expand(1, 3, 24, 24)
    assert PSNRMetric(test_y_channel=False).compute(red, blue) == pytest.approx(
        10.0 * math.log10(1.5), abs=1e-12
    )
    assert PSNRMetric(test_y_channel=True).compute(red, blue) == pytest.approx(
        20.0 * math.log10(255.0 / (65.481 - 24.966)), abs=1e-12
    )


@pytest.mark.parametrize("test_y_channel", [False, True])
def test_psnr_and_ssim_crop_border_exact_four_edges_and_corners(test_y_channel: bool) -> None:
    gen = torch.Generator().manual_seed(77)
    height, width, border = 32, 36, 4
    base = torch.rand(1, 3, height, width, generator=gen, dtype=torch.float64)

    # Corrupt ONLY outer `border` pixels on all 4 edges
    outer_corrupted = base.clone()
    outer_corrupted[:, :, :border, :] = 1.0 - outer_corrupted[:, :, :border, :]
    outer_corrupted[:, :, -border:, :] = 1.0 - outer_corrupted[:, :, -border:, :]
    outer_corrupted[:, :, :, :border] = 1.0 - outer_corrupted[:, :, :, :border]
    outer_corrupted[:, :, :, -border:] = 1.0 - outer_corrupted[:, :, :, -border:]

    assert math.isfinite(
        PSNRMetric(crop_border=0, test_y_channel=test_y_channel).compute(outer_corrupted, base)
    )
    assert (
        SSIMMetric(crop_border=0, test_y_channel=test_y_channel).compute(outer_corrupted, base)
        < 1.0
    )
    assert math.isinf(
        PSNRMetric(crop_border=border, test_y_channel=test_y_channel).compute(outer_corrupted, base)
    )
    assert SSIMMetric(crop_border=border, test_y_channel=test_y_channel).compute(
        outer_corrupted, base
    ) == pytest.approx(1.0, abs=1e-12)

    # Corrupt a single pixel at each of the 4 interior boundary corners:
    # crop_border=b MUST detect it; crop_border=b+1 MUST exclude it.
    corners = [
        (border, border),
        (border, width - border - 1),
        (height - border - 1, border),
        (height - border - 1, width - border - 1),
    ]
    psnr_b = PSNRMetric(crop_border=border, test_y_channel=test_y_channel)
    psnr_b1 = PSNRMetric(crop_border=border + 1, test_y_channel=test_y_channel)
    ssim_b = SSIMMetric(crop_border=border, test_y_channel=test_y_channel)
    ssim_b1 = SSIMMetric(crop_border=border + 1, test_y_channel=test_y_channel)

    for row, col in corners:
        corrupted = base.clone()
        corrupted[:, :, row, col] = (corrupted[:, :, row, col] + 0.5) % 1.0
        assert math.isfinite(psnr_b.compute(corrupted, base))
        assert ssim_b.compute(corrupted, base) < 1.0
        assert math.isinf(psnr_b1.compute(corrupted, base))
        assert ssim_b1.compute(corrupted, base) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("test_y_channel", [False, True])
def test_psnr_and_ssim_identical_zero_warnings_symmetry_monotonicity_and_negative_ssim(
    test_y_channel: bool,
) -> None:
    psnr_metric = PSNRMetric(crop_border=0, test_y_channel=test_y_channel)
    ssim_metric = SSIMMetric(crop_border=0, test_y_channel=test_y_channel)

    # Identical constant & random inputs -> inf PSNR (zero warnings) and 1.0 SSIM
    gen = torch.Generator().manual_seed(42)
    constant_imgs = [
        torch.full((1, 3, 24, 24), fill, dtype=torch.float32) for fill in (0.0, 1.0 / 3.0, 0.5, 1.0)
    ]
    random_img = torch.rand(1, 3, 24, 24, generator=gen, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for img in [*constant_imgs, random_img]:
            p_val = psnr_metric.compute(img, img)
            s_val = ssim_metric.compute(img, img)
            assert type(p_val) is float
            assert type(s_val) is float
            assert math.isinf(p_val) and p_val > 0.0
            assert s_val == pytest.approx(1.0, abs=1e-12)

        for const_img in constant_imgs:
            assert ssim_metric.compute(const_img, const_img) == 1.0

    # Symmetry & noise monotonicity
    noise = torch.randn(1, 3, 32, 32, generator=gen, dtype=torch.float32)
    base = torch.rand(1, 3, 32, 32, generator=gen, dtype=torch.float32)
    small_pert = (base + 0.02 * noise).clamp(0.0, 1.0)
    large_pert = (base + 0.20 * noise).clamp(0.0, 1.0)

    assert psnr_metric.compute(small_pert, base) == pytest.approx(
        psnr_metric.compute(base, small_pert), abs=1e-12
    )
    assert ssim_metric.compute(small_pert, base) == pytest.approx(
        ssim_metric.compute(base, small_pert), abs=1e-12
    )
    assert psnr_metric.compute(small_pert, base) > psnr_metric.compute(large_pert, base)
    assert ssim_metric.compute(small_pert, base) > ssim_metric.compute(large_pert, base)

    # Anti-correlated stripe pair produces negative SSIM in [-1, 0)
    stripes = (
        (torch.arange(32).view(1, 1, 1, 32) % 2).to(torch.float32).expand(1, 3, 32, 32).clone()
    )
    inv_stripes = 1.0 - stripes
    neg_ssim = ssim_metric.compute(stripes, inv_stripes)
    assert -1.0 <= neg_ssim < 0.0


def test_psnr_and_ssim_nan_batch_dtypes_noncontiguous_autocast_and_meta() -> None:
    gen = torch.Generator().manual_seed(999)
    psnr_rgb = PSNRMetric(crop_border=2, test_y_channel=False)
    ssim_rgb = SSIMMetric(crop_border=2, test_y_channel=False)
    psnr_y = PSNRMetric(crop_border=2, test_y_channel=True)
    ssim_y = SSIMMetric(crop_border=2, test_y_channel=True)

    # NaN propagation (PSNR must return NaN, NEVER +inf)
    target = torch.rand(1, 3, 28, 28, generator=gen, dtype=torch.float32)
    pred_nan = target.clone()
    pred_nan[0, 0, 14, 14] = float("nan")
    for metric in (psnr_rgb, ssim_rgb, psnr_y, ssim_y):
        val = metric.compute(pred_nan, target)
        assert type(val) is float
        assert math.isnan(val)

    # Batch consistency: N=3 equals mean of single-image scores
    t_batch = torch.rand(3, 3, 28, 30, generator=gen, dtype=torch.float32)
    p_batch = (t_batch + 0.05 * torch.randn(3, 3, 28, 30, generator=gen)).clamp(0.0, 1.0)
    for metric in (psnr_rgb, ssim_rgb, psnr_y, ssim_y):
        batch_score = metric.compute(p_batch, t_batch)
        per_image = [metric.compute(p_batch[i : i + 1], t_batch[i : i + 1]) for i in range(3)]
        assert batch_score == pytest.approx(float(np.mean(per_image)), abs=1e-12)

    # Batch with one identical pair -> PSNR returns float("inf")
    p_mixed = p_batch[:2].clone()
    p_mixed[1] = t_batch[1].clone()
    assert math.isinf(psnr_y.compute(p_mixed, t_batch[:2]))

    # Grayscale C=1 supported when test_y_channel=False
    g_target = torch.rand(1, 1, 28, 28, generator=gen, dtype=torch.float32)
    g_pred = (g_target + 0.05 * torch.randn(1, 1, 28, 28, generator=gen)).clamp(0.0, 1.0)
    assert psnr_rgb.compute(g_pred, g_target) == pytest.approx(
        _ref_psnr_np(g_pred.numpy(), g_target.numpy(), 2, False), abs=1e-12
    )
    assert ssim_rgb.compute(g_pred, g_target) == pytest.approx(
        _ref_ssim_scipy(g_pred.numpy(), g_target.numpy(), 2, False), abs=1e-12
    )

    # Mixed float dtypes (bfloat16 / float16 pred vs float32 target)
    for mixed_dtype in (torch.bfloat16, torch.float16):
        p_low = p_batch[:1].to(dtype=mixed_dtype)
        t_fp32 = t_batch[:1]
        assert psnr_y.compute(p_low, t_fp32) == pytest.approx(
            _ref_psnr_np(
                p_low.to(torch.float64).numpy(), t_fp32.to(torch.float64).numpy(), 2, True
            ),
            abs=1e-12,
        )
        assert ssim_y.compute(p_low, t_fp32) == pytest.approx(
            _ref_ssim_scipy(
                p_low.to(torch.float64).numpy(), t_fp32.to(torch.float64).numpy(), 2, True
            ),
            abs=1e-12,
        )

    # Non-contiguous transpose + requires_grad=True + outer CPU bfloat16 autocast + meta passthrough
    p_ref = p_batch[:1].transpose(2, 3).contiguous()
    t_ref = t_batch[:1].transpose(2, 3).contiguous()
    p_leaf = p_batch[:1].clone().requires_grad_(True)
    t_leaf = t_batch[:1].clone()
    p_nc = p_leaf.transpose(2, 3)
    t_nc = t_leaf.transpose(2, 3)
    assert not p_nc.is_contiguous()
    assert not t_nc.is_contiguous()

    meta = {"label": "HELLO", "nested": {"k": [1, 2]}, "index": 7}
    meta_snapshot = copy.deepcopy(meta)

    for metric in (psnr_y, ssim_y):
        ref_val = metric.compute(p_ref, t_ref)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            nc_val = metric.compute(p_nc, t_nc, meta=meta)
        assert nc_val == ref_val

    assert p_leaf.grad is None
    assert torch.equal(p_leaf.detach(), p_batch[:1])
    assert torch.equal(t_leaf, t_batch[:1])
    assert meta == meta_snapshot


def test_ssim_window_cache_reuse_and_inference_mode_safety() -> None:
    metric = SSIMMetric(crop_border=0, test_y_channel=True)
    assert metric.to("cpu") is metric
    cached_after_to = metric._window_cache[(torch.device("cpu"), 1)]
    assert metric.to("cpu:0") is metric
    assert metric._window_cache[(torch.device("cpu"), 1)] is cached_after_to

    x = torch.rand(1, 3, 24, 24)
    y = torch.rand(1, 3, 24, 24)
    s1 = metric.compute(x, y)
    s2 = metric.compute(x, y)
    assert s1 == s2
    assert metric._window_cache[(torch.device("cpu"), 1)] is cached_after_to

    # First call inside torch.inference_mode() must NOT cache an inference tensor
    fresh = SSIMMetric(crop_border=0, test_y_channel=False)
    with torch.inference_mode():
        score_inf = fresh.compute(x, y)
    cached_win = fresh._window_cache[(torch.device("cpu"), 3)]
    assert not cached_win.is_inference()

    score_std = fresh.compute(x, y)
    assert score_std == score_inf
    assert fresh._window_cache[(torch.device("cpu"), 3)] is cached_win


def test_psnr_and_ssim_minimal_boundary_sizes_and_validation_matrix() -> None:
    # Minimal valid boundary sizes
    assert math.isinf(
        PSNRMetric(crop_border=0).compute(torch.zeros(1, 3, 1, 1), torch.zeros(1, 3, 1, 1))
    )
    assert math.isinf(
        PSNRMetric(crop_border=4).compute(torch.zeros(1, 3, 9, 9), torch.zeros(1, 3, 9, 9))
    )
    assert (
        SSIMMetric(crop_border=0).compute(torch.zeros(1, 3, 11, 11), torch.zeros(1, 3, 11, 11))
        == 1.0
    )
    assert (
        SSIMMetric(crop_border=4).compute(torch.zeros(1, 3, 19, 19), torch.zeros(1, 3, 19, 19))
        == 1.0
    )

    # Constructor validation
    for cls in (PSNRMetric, SSIMMetric):
        for bad_crop in (-1, -4, True, False, 0.0, 4.0, "4", None):
            with pytest.raises(ValueError):
                cls(crop_border=bad_crop)  # type: ignore[arg-type]
        for bad_y in (0, 1, "true", "false", None, 1.0):
            with pytest.raises(ValueError):
                cls(test_y_channel=bad_y)  # type: ignore[arg-type]

    # Input validation error matrix
    for cls in (PSNRMetric, SSIMMetric):
        metric = cls(crop_border=0, test_y_channel=False)
        valid = torch.rand(1, 3, 24, 24)

        # 1. target is None -> ValueError (MUST precede TypeError check)
        with pytest.raises(ValueError):
            metric.compute(valid, target=None)
        with pytest.raises(ValueError):
            metric.compute("not_a_tensor", target=None)  # type: ignore[arg-type]

        # 2. Non-Tensor pred or target -> TypeError
        with pytest.raises(TypeError):
            metric.compute(np.zeros((1, 3, 24, 24), dtype=np.float32), valid)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            metric.compute(valid, np.zeros((1, 3, 24, 24), dtype=np.float32))  # type: ignore[arg-type]

        # 3. Non-4D
        with pytest.raises(ValueError):
            metric.compute(torch.rand(3, 24, 24), torch.rand(3, 24, 24))
        with pytest.raises(ValueError):
            metric.compute(torch.rand(1, 1, 3, 24, 24), torch.rand(1, 1, 3, 24, 24))

        # 4. Shape mismatch
        with pytest.raises(ValueError):
            metric.compute(torch.rand(1, 3, 24, 24), torch.rand(2, 3, 24, 24))
        with pytest.raises(ValueError):
            metric.compute(torch.rand(1, 3, 24, 24), torch.rand(1, 3, 24, 25))

        # 5. Zero dimensions
        for shape in ((0, 3, 24, 24), (1, 0, 24, 24), (1, 3, 0, 24), (1, 3, 24, 0)):
            with pytest.raises(ValueError):
                metric.compute(torch.empty(*shape), torch.empty(*shape))

        # 6. Non-floating dtypes
        for bad_dtype in (torch.uint8, torch.int64, torch.bool, torch.complex64):
            bad_t = torch.zeros(1, 3, 24, 24, dtype=bad_dtype)
            with pytest.raises(ValueError):
                metric.compute(bad_t, valid)
            with pytest.raises(ValueError):
                metric.compute(valid, bad_t)

        # 7. Device mismatch (cpu vs meta)
        meta_t = torch.empty(1, 3, 24, 24, device="meta")
        with pytest.raises(ValueError):
            metric.compute(valid, meta_t)

        # 8. test_y_channel=True with C != 3
        y_metric = cls(crop_border=0, test_y_channel=True)
        for bad_c in (1, 4):
            c_t = torch.rand(1, bad_c, 24, 24)
            with pytest.raises(ValueError):
                y_metric.compute(c_t, c_t)

        # 9. Spatial dims <= 2 * crop_border
        crop_metric = cls(crop_border=4, test_y_channel=False)
        for bad_hw in ((8, 24), (24, 8), (6, 6)):
            small = torch.rand(1, 3, *bad_hw)
            with pytest.raises(ValueError):
                crop_metric.compute(small, small)

    # 10. SSIM-specific cropped < 11 check
    for crop_border, bad_hw in [(0, (10, 11)), (0, (11, 10)), (4, (18, 19)), (4, (19, 18))]:
        ssim_m = SSIMMetric(crop_border=crop_border, test_y_channel=False)
        small = torch.rand(1, 3, *bad_hw)
        with pytest.raises(ValueError):
            ssim_m.compute(small, small)


# ---------------------------------------------------------------------------
# LPIPSMetric offline tests, rescaling, caching, autocast & validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("net", ["alex", "vgg", "squeeze"])
@pytest.mark.parametrize("spatial", [False, True])
def test_lpips_metric_offline_zero_download_rescaling_and_parity(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    net: str,
    spatial: bool,
) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    _block_network_downloads(monkeypatch)
    patch_lpips_offline(monkeypatch)
    torch.manual_seed(1234)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metric = LPIPSMetric(net=net, spatial=spatial, version="0.1")

    captured_io = capsys.readouterr()
    assert captured_io.out == ""
    assert captured_io.err == ""

    assert isinstance(metric, BaseMetric)
    assert metric.higher_is_better is False
    assert metric.device == torch.device("cpu")

    # All params frozen, eval mode locked
    params = list(metric.lpips.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)
    assert metric.lpips.training is False
    assert all(not m.training for m in metric.lpips.modules())

    # Pre-hook verifies exact [0, 1] -> [-1, 1] rescaling
    captured_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _hook(_mod: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured_inputs.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

    handle = metric.lpips.register_forward_pre_hook(_hook)
    try:
        zeros = torch.zeros(1, 3, 64, 64)
        ones = torch.ones(1, 3, 64, 64)
        d_zero_one = metric.compute(zeros, ones)
    finally:
        handle.remove()

    assert len(captured_inputs) == 1
    assert torch.allclose(captured_inputs[0][0], torch.full((1, 3, 64, 64), -1.0))
    assert torch.allclose(captured_inputs[0][1], torch.full((1, 3, 64, 64), 1.0))
    assert type(d_zero_one) is float and math.isfinite(d_zero_one) and d_zero_one >= 0.0

    # Identical inputs -> exact 0.0
    gen = torch.Generator().manual_seed(42)
    x = torch.rand(1, 3, 64, 64, generator=gen)
    y = torch.rand(1, 3, 64, 64, generator=gen)
    assert metric.compute(zeros, zeros) == 0.0
    assert metric.compute(ones, ones) == 0.0
    assert metric.compute(x, x) == 0.0

    # Distinct inputs -> strictly > 0.0, deterministic, direct parity with underlying lpips.LPIPS
    d1 = metric.compute(x, y)
    d2 = metric.compute(x, y)
    assert type(d1) is float
    assert d1 > 0.0
    assert d1 == d2
    with torch.no_grad():
        direct = float(metric.lpips(2.0 * x - 1.0, 2.0 * y - 1.0).mean().item())
    assert d1 == pytest.approx(direct, abs=1e-7)

    # Batch N > 1 parity
    bx = torch.rand(2, 3, 64, 64, generator=gen)
    by = torch.rand(2, 3, 64, 64, generator=gen)
    d_batch = metric.compute(bx, by)
    with torch.no_grad():
        direct_batch = float(metric.lpips(2.0 * bx - 1.0, 2.0 * by - 1.0).mean().item())
    assert type(d_batch) is float
    assert d_batch > 0.0
    assert d_batch == pytest.approx(direct_batch, abs=1e-7)


def test_lpips_metric_version_0_0_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    _block_network_downloads(monkeypatch)
    patch_lpips_offline(monkeypatch)
    torch.manual_seed(7)
    metric = LPIPSMetric(net="alex", spatial=False, version="0.0")
    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)
    assert metric.compute(x, x) == 0.0
    val = metric.compute(x, y)
    assert type(val) is float and math.isfinite(val) and val > 0.0


def test_lpips_metric_single_load_reuse_eval_recovery_and_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    _block_network_downloads(monkeypatch)
    patch_lpips_offline(monkeypatch)
    torch.manual_seed(99)
    metric = LPIPSMetric(net="alex")
    orig_lpips = metric.lpips

    assert metric.to("cpu") is metric
    assert metric.lpips is orig_lpips
    assert metric.to("cpu:0") is metric
    assert metric.lpips is orig_lpips
    assert metric.device == torch.device("cpu")

    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)
    meta = {"label": "TEXT", "index": 3}
    meta_snapshot = copy.deepcopy(meta)

    baseline = metric.compute(x, y, meta=meta)
    assert metric.lpips is orig_lpips
    assert meta == meta_snapshot

    # External tampering with train mode (root + submodule) is recovered on compute()
    metric.lpips.train(True)
    assert metric.lpips.training is True
    recovered_score = metric.compute(x, y)
    assert metric.lpips.training is False
    assert all(not m.training for m in metric.lpips.modules())
    assert recovered_score == baseline
    assert metric.lpips is orig_lpips

    metric.lpips.lins.train(True)
    assert metric.lpips.lins.training is True
    assert metric.compute(x, y) == baseline
    assert metric.lpips.lins.training is False


def test_lpips_metric_no_grad_and_bf16_autocast_fp32_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    _block_network_downloads(monkeypatch)
    patch_lpips_offline(monkeypatch)
    torch.manual_seed(55)
    metric = LPIPSMetric(net="alex")

    x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    y = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    standalone_fp32 = metric.compute(x, y)

    observed: list[dict[str, Any]] = []

    def _hook(_mod: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        observed.append(
            {
                "in0_dtype": inputs[0].dtype,
                "in1_dtype": inputs[1].dtype,
                "grad_enabled": torch.is_grad_enabled(),
                "autocast_cpu": torch.is_autocast_enabled("cpu"),
            }
        )

    pred_grad = x.clone().requires_grad_(True)
    handle = metric.lpips.register_forward_pre_hook(_hook)
    try:
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            autocast_score = metric.compute(pred_grad, y)
    finally:
        handle.remove()

    assert len(observed) == 1
    assert observed[0]["in0_dtype"] == torch.float32
    assert observed[0]["in1_dtype"] == torch.float32
    assert observed[0]["grad_enabled"] is False
    assert observed[0]["autocast_cpu"] is False
    assert autocast_score == standalone_fp32
    assert pred_grad.grad is None
    assert all(p.grad is None for p in metric.lpips.parameters())

    # Non-FP32 floating dtypes execute cleanly as finite non-negative floats
    for dtype in (torch.bfloat16, torch.float16, torch.float64):
        val = metric.compute(x.to(dtype=dtype), y.to(dtype=dtype))
        assert type(val) is float and math.isfinite(val) and val >= 0.0


def test_lpips_metric_validation_and_min_spatial_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    _block_network_downloads(monkeypatch)
    patch_lpips_offline(monkeypatch)

    # Constructor validation
    for bad_net in ("resnet", "vgg16", 123, None):
        with pytest.raises(ValueError):
            LPIPSMetric(net=bad_net)  # type: ignore[arg-type]
    for bad_version in ("9.9", "v0.1", 0.1, None):
        with pytest.raises(ValueError):
            LPIPSMetric(version=bad_version)  # type: ignore[arg-type]
    for bad_bool in (0, 1, "false", "true", None):
        with pytest.raises(ValueError):
            LPIPSMetric(spatial=bad_bool)  # type: ignore[arg-type]

    # Input validation
    metric = LPIPSMetric(net="alex")
    valid = torch.rand(1, 3, 64, 64)

    with pytest.raises(ValueError):
        metric.compute(valid, target=None)
    with pytest.raises(ValueError):
        metric.compute("not_a_tensor", target=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        metric.compute(np.zeros((1, 3, 64, 64), dtype=np.float32), valid)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        metric.compute(valid, np.zeros((1, 3, 64, 64), dtype=np.float32))  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        metric.compute(torch.rand(3, 64, 64), torch.rand(3, 64, 64))
    with pytest.raises(ValueError):
        metric.compute(torch.rand(1, 1, 3, 64, 64), torch.rand(1, 1, 3, 64, 64))
    with pytest.raises(ValueError):
        metric.compute(torch.rand(1, 3, 64, 64), torch.rand(2, 3, 64, 64))
    with pytest.raises(ValueError):
        metric.compute(torch.empty(0, 3, 64, 64), torch.empty(0, 3, 64, 64))
    for bad_c in (1, 4):
        c_t = torch.rand(1, bad_c, 64, 64)
        with pytest.raises(ValueError):
            metric.compute(c_t, c_t)
    for bad_dtype in (torch.uint8, torch.int64, torch.bool, torch.complex64):
        bad_t = torch.zeros(1, 3, 64, 64, dtype=bad_dtype)
        with pytest.raises(ValueError):
            metric.compute(bad_t, valid)
        with pytest.raises(ValueError):
            metric.compute(valid, bad_t)
    with pytest.raises(ValueError):
        metric.compute(valid, torch.empty(1, 3, 64, 64, device="meta"))

    # Backbone-specific minimum spatial boundaries
    for net_name, min_sz in (("alex", 31), ("vgg", 16), ("squeeze", 17)):
        m = LPIPSMetric(net=net_name)
        for bad_hw in ((min_sz - 1, min_sz), (min_sz, min_sz - 1), (0, min_sz)):
            small = torch.rand(1, 3, *bad_hw)
            with pytest.raises(ValueError):
                m.compute(small, small)
        boundary = torch.rand(1, 3, min_sz, min_sz)
        assert m.compute(boundary, boundary) == 0.0


# ---------------------------------------------------------------------------
# Hydra configs/metric/sr_default.yaml integration
# ---------------------------------------------------------------------------


def test_hydra_sr_default_metric_instantiation_and_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    _block_network_downloads(monkeypatch)
    patch_lpips_offline(monkeypatch)
    torch.manual_seed(2026)

    pred = torch.rand(1, 3, 64, 64)
    target = (pred + 0.05 * torch.randn(1, 3, 64, 64)).clamp(0.0, 1.0)

    # Default scale=1 composition
    cfg = _compose("train")

    set5_metrics = {
        name: instantiate(m_cfg).to("cpu")
        for name, m_cfg in cfg.metric.val_suites.set5.metrics.items()
    }
    assert set(set5_metrics.keys()) == {"psnr", "ssim", "lpips"}
    assert isinstance(set5_metrics["psnr"], PSNRMetric)
    assert isinstance(set5_metrics["ssim"], SSIMMetric)
    assert isinstance(set5_metrics["lpips"], LPIPSMetric)
    assert set5_metrics["psnr"].crop_border == 1
    assert set5_metrics["psnr"].test_y_channel is True
    assert set5_metrics["psnr"].higher_is_better is True
    assert set5_metrics["ssim"].crop_border == 1
    assert set5_metrics["ssim"].test_y_channel is True
    assert set5_metrics["ssim"].higher_is_better is True
    assert set5_metrics["lpips"].net == "alex"
    assert set5_metrics["lpips"].higher_is_better is False

    urban_metrics = {
        name: instantiate(m_cfg).to("cpu")
        for name, m_cfg in cfg.metric.val_suites.urban100.metrics.items()
    }
    assert set(urban_metrics.keys()) == {"psnr", "ssim"}
    assert isinstance(urban_metrics["psnr"], PSNRMetric)
    assert isinstance(urban_metrics["ssim"], SSIMMetric)
    assert urban_metrics["psnr"].crop_border == 1
    assert urban_metrics["ssim"].crop_border == 1

    for suite_metrics in (set5_metrics, urban_metrics):
        for name, metric in suite_metrics.items():
            val = metric.compute(pred, target)
            assert type(val) is float, f"Expected float for {name}, got {type(val)}"
            assert math.isfinite(val), f"Expected finite score for {name}, got {val}"

    # Verify scale=2 interpolation propagates to crop_border across both suites
    cfg_x2 = _compose("train", overrides=["scale=2"])
    for suite_name in ("set5", "urban100"):
        suite_cfg = cfg_x2.metric.val_suites[suite_name].metrics
        psnr_m = instantiate(suite_cfg.psnr)
        ssim_m = instantiate(suite_cfg.ssim)
        assert isinstance(psnr_m, PSNRMetric) and psnr_m.crop_border == 2
        assert isinstance(ssim_m, SSIMMetric) and ssim_m.crop_border == 2
        assert math.isfinite(psnr_m.compute(pred, target))
        assert math.isfinite(ssim_m.compute(pred, target))
