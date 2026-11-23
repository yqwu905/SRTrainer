"""Pure-PyTorch BasicSR/MATLAB-exact PSNR, SSIM, and ITU-R BT.601 Y-channel metrics."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .base import BaseMetric

__all__ = ["rgb_to_y_bt601", "PSNRMetric", "SSIMMetric"]

_SSIM_WINDOW_SIZE: int = 11
_SSIM_SIGMA: float = 1.5
_SSIM_C1: float = (0.01 * 255.0) ** 2
_SSIM_C2: float = (0.03 * 255.0) ** 2
_NO_FLOAT64_DEVICE_TYPES: frozenset[str] = frozenset({"npu", "mps"})


def _metric_compute_dtype(device: torch.device) -> torch.dtype:
    """Select ``float32`` on backends without native ``float64`` ops (NPU/MPS), else ``float64``."""
    return torch.float32 if device.type in _NO_FLOAT64_DEVICE_TYPES else torch.float64


def _validate_metric_init(crop_border: int, test_y_channel: bool) -> None:
    if isinstance(crop_border, bool) or not isinstance(crop_border, int) or crop_border < 0:
        raise ValueError(f"crop_border must be a non-negative integer, got {crop_border!r}.")
    if not isinstance(test_y_channel, bool):
        raise ValueError(f"test_y_channel must be a bool, got {test_y_channel!r}.")


def _validate_image_pair(
    pred: torch.Tensor,
    target: torch.Tensor | None,
    *,
    metric_name: str,
    crop_border: int,
    test_y_channel: bool,
    min_cropped_size: int = 1,
) -> torch.Tensor:
    """Validate full-reference image pair inputs in strict fail-fast order."""
    if target is None:
        raise ValueError(f"{metric_name} requires a non-None target tensor.")
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"Expected pred to be a torch.Tensor, got {type(pred).__name__}.")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected target to be a torch.Tensor, got {type(target).__name__}.")
    if pred.ndim != 4 or target.ndim != 4:
        raise ValueError(
            f"{metric_name} expects 4D tensors [N, C, H, W], "
            f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}."
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred has shape {tuple(pred.shape)}, "
            f"target has shape {tuple(target.shape)}."
        )
    if pred.shape[0] == 0 or pred.shape[1] == 0 or pred.shape[2] == 0 or pred.shape[3] == 0:
        raise ValueError(
            f"{metric_name} expects non-empty dimensions (N, C, H, W >= 1), "
            f"got shape {tuple(pred.shape)}."
        )
    if not pred.is_floating_point() or not target.is_floating_point():
        raise ValueError(
            f"{metric_name} expects floating-point tensors in [0, 1], "
            f"got pred.dtype={pred.dtype}, target.dtype={target.dtype}."
        )
    if pred.device != target.device:
        raise ValueError(
            f"Device mismatch: pred is on {pred.device}, target is on {target.device}."
        )
    if test_y_channel and pred.shape[1] != 3:
        raise ValueError(
            f"{metric_name} with test_y_channel=True expects 3-channel RGB inputs, "
            f"got {pred.shape[1]} channels."
        )

    cropped_h = pred.shape[2] - 2 * crop_border
    cropped_w = pred.shape[3] - 2 * crop_border
    if cropped_h <= 0 or cropped_w <= 0:
        raise ValueError(
            f"{metric_name} crop_border={crop_border} leaves non-positive spatial size "
            f"({cropped_h}, {cropped_w}) for input shape {tuple(pred.shape)}."
        )
    if cropped_h < min_cropped_size or cropped_w < min_cropped_size:
        raise ValueError(
            f"{metric_name} requires cropped spatial dimensions >= {min_cropped_size}, "
            f"got ({cropped_h}, {cropped_w}) after crop_border={crop_border} "
            f"from input shape {tuple(pred.shape)}."
        )
    return target


def rgb_to_y_bt601(img: torch.Tensor) -> torch.Tensor:
    """Convert a 4D floating-point RGB tensor in ``[0, 1]`` to studio-swing ITU-R BT.601 Y.

    Computes the exact MATLAB ``rgb2ycbcr`` / BasicSR luminance formula in ``float64``
    (or ``float32`` on backends without native ``float64`` ops such as NPU/MPS):
    ``Y = 65.481 * R + 128.553 * G + 24.966 * B + 16.0`` (nominal range ``[16.0, 235.0]``).

    Args:
        img: 4D floating-point RGB tensor of shape ``[N, 3, H, W]``.

    Returns:
        4D luminance tensor of shape ``[N, 1, H, W]`` on ``img.device`` in ``float64``
        (or ``float32`` on NPU/MPS).
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Expected img to be a torch.Tensor, got {type(img).__name__}.")
    if img.ndim != 4:
        raise ValueError(
            f"rgb_to_y_bt601 expects a 4D tensor [N, 3, H, W], got shape {tuple(img.shape)}."
        )
    if img.shape[1] != 3:
        raise ValueError(f"rgb_to_y_bt601 expects 3 RGB channels, got {img.shape[1]} channels.")
    if img.shape[0] == 0 or img.shape[2] == 0 or img.shape[3] == 0:
        raise ValueError(
            f"rgb_to_y_bt601 expects non-empty dimensions (N, H, W >= 1), "
            f"got shape {tuple(img.shape)}."
        )
    if not img.is_floating_point():
        raise ValueError(
            f"rgb_to_y_bt601 expects a floating-point tensor in [0, 1], got dtype {img.dtype}."
        )

    compute_dtype = _metric_compute_dtype(img.device)
    with torch.amp.autocast(device_type=img.device.type, enabled=False):
        x = img.to(dtype=compute_dtype)
        return (
            65.481 * x[:, 0:1, :, :] + 128.553 * x[:, 1:2, :, :] + 24.966 * x[:, 2:3, :, :] + 16.0
        )


def _create_gaussian_window(
    window_size: int = _SSIM_WINDOW_SIZE, sigma: float = _SSIM_SIGMA
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float64) - (window_size - 1) / 2.0
    gauss_1d = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
    gauss_1d = gauss_1d / gauss_1d.sum()
    window_2d = torch.outer(gauss_1d, gauss_1d)
    return window_2d.view(1, 1, window_size, window_size).contiguous()


_BASE_SSIM_WINDOW: torch.Tensor = _create_gaussian_window(_SSIM_WINDOW_SIZE, _SSIM_SIGMA)


class PSNRMetric(BaseMetric):
    """Peak Signal-to-Noise Ratio (PSNR) aligned with BasicSR / MATLAB BT.601 conventions.

    Args:
        crop_border: Number of border pixels to crop from each spatial edge before evaluation.
        test_y_channel: If ``True``, evaluate on ITU-R BT.601 studio-swing Y channel.
    """

    higher_is_better: bool = True

    def __init__(self, crop_border: int = 0, test_y_channel: bool = False) -> None:
        _validate_metric_init(crop_border, test_y_channel)
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del meta
        target_tensor = _validate_image_pair(
            pred,
            target,
            metric_name="PSNRMetric",
            crop_border=self.crop_border,
            test_y_channel=self.test_y_channel,
            min_cropped_size=1,
        )

        compute_dtype = _metric_compute_dtype(pred.device)
        with torch.no_grad(), torch.amp.autocast(device_type=pred.device.type, enabled=False):
            p = pred
            t = target_tensor
            if self.crop_border > 0:
                b = self.crop_border
                p = p[:, :, b:-b, b:-b]
                t = t[:, :, b:-b, b:-b]

            if self.test_y_channel:
                img1 = rgb_to_y_bt601(p)
                img2 = rgb_to_y_bt601(t)
            else:
                img1 = p.to(dtype=compute_dtype) * 255.0
                img2 = t.to(dtype=compute_dtype) * 255.0

            mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3))
            psnr = torch.where(
                mse == 0.0,
                torch.full_like(mse, float("inf")),
                10.0 * torch.log10((255.0**2) / mse),
            )
            return float(psnr.mean().item())


class SSIMMetric(BaseMetric):
    """Structural Similarity Index (SSIM) aligned with BasicSR / MATLAB BT.601 conventions.

    Uses an 11x11 ``float64`` Gaussian window (``sigma=1.5``), valid convolution, and
    ``C1=(0.01*255)^2``, ``C2=(0.03*255)^2``.

    Args:
        crop_border: Number of border pixels to crop from each spatial edge before evaluation.
        test_y_channel: If ``True``, evaluate on ITU-R BT.601 studio-swing Y channel.
    """

    higher_is_better: bool = True

    def __init__(self, crop_border: int = 0, test_y_channel: bool = False) -> None:
        _validate_metric_init(crop_border, test_y_channel)
        self.crop_border = crop_border
        self.test_y_channel = test_y_channel
        self._window_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def _get_window(
        self,
        device: torch.device,
        channels: int,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        canonical_device = torch.device("cpu") if device.type == "cpu" else device
        default_dtype = _metric_compute_dtype(canonical_device)
        target_dtype = default_dtype if dtype is None else dtype
        key: tuple[Any, ...] = (
            (canonical_device, channels)
            if target_dtype == default_dtype
            else (canonical_device, channels, target_dtype)
        )
        window = self._window_cache.get(key)
        if window is not None and window.dtype == target_dtype:
            return window

        with torch.inference_mode(False), torch.no_grad():
            window = (
                _BASE_SSIM_WINDOW.to(dtype=target_dtype)
                .to(device=canonical_device)
                .expand(channels, 1, _SSIM_WINDOW_SIZE, _SSIM_WINDOW_SIZE)
                .contiguous()
            )

        resolved_key: tuple[Any, ...] = (
            (window.device, channels)
            if target_dtype == default_dtype
            else (window.device, channels, target_dtype)
        )
        self._window_cache[resolved_key] = window
        self._window_cache[key] = window
        return window

    def to(self, device: torch.device | str) -> SSIMMetric:
        """Pre-stage the Gaussian SSIM window on ``device`` and return ``self``."""
        target_device = torch.device(device)
        default_channels = 1 if self.test_y_channel else 3
        self._get_window(target_device, default_channels)
        return self

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del meta
        target_tensor = _validate_image_pair(
            pred,
            target,
            metric_name="SSIMMetric",
            crop_border=self.crop_border,
            test_y_channel=self.test_y_channel,
            min_cropped_size=_SSIM_WINDOW_SIZE,
        )

        compute_dtype = _metric_compute_dtype(pred.device)
        with torch.no_grad(), torch.amp.autocast(device_type=pred.device.type, enabled=False):
            p = pred
            t = target_tensor
            if self.crop_border > 0:
                b = self.crop_border
                p = p[:, :, b:-b, b:-b]
                t = t[:, :, b:-b, b:-b]

            if self.test_y_channel:
                img1 = rgb_to_y_bt601(p)
                img2 = rgb_to_y_bt601(t)
            else:
                img1 = p.to(dtype=compute_dtype) * 255.0
                img2 = t.to(dtype=compute_dtype) * 255.0

            if img2.dtype != img1.dtype:
                img2 = img2.to(dtype=img1.dtype)

            channels = img1.shape[1]
            window = self._get_window(img1.device, channels, dtype=img1.dtype)

            mu1 = F.conv2d(img1, window, padding=0, groups=channels)
            mu2 = F.conv2d(img2, window, padding=0, groups=channels)
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channels) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channels) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channels) - mu1_mu2

            numerator = (2.0 * mu1_mu2 + _SSIM_C1) * (2.0 * sigma12 + _SSIM_C2)
            denominator = (mu1_sq + mu2_sq + _SSIM_C1) * (sigma1_sq + sigma2_sq + _SSIM_C2)
            ssim_map = numerator / denominator
            return float(ssim_map.mean(dim=(1, 2, 3)).mean().item())
