"""Device-cached LPIPS perceptual evaluation metric."""

from __future__ import annotations

import warnings
from typing import Any

import lpips
import torch

from .base import BaseMetric

__all__ = ["LPIPSMetric"]

_VALID_NETS: tuple[str, ...] = ("alex", "vgg", "squeeze")
_VALID_VERSIONS: tuple[str, ...] = ("0.1", "0.0")
_MIN_SPATIAL_SIZE: dict[str, int] = {"alex": 31, "vgg": 16, "squeeze": 17}


class LPIPSMetric(BaseMetric):
    """Learned Perceptual Image Patch Similarity (LPIPS) evaluation metric for RGB images in ``[0, 1]``.

    Loads the LPIPS backbone once at initialization, freezes all parameters, locks ``eval()``
    mode, reuses weights in-place across ``to(device)`` and repeated ``compute()`` calls,
    rescales inputs from ``[0, 1]`` to ``[-1, 1]``, and evaluates strictly in FP32 under
    ``torch.no_grad()``. When ``pretrained=False``, sets ``pnet_rand=True`` (zero network
    download) and constrains random linear weights to be non-negative.

    Args:
        net: Backbone architecture (``"alex"``, ``"vgg"``, or ``"squeeze"``).
        spatial: If ``True``, compute a spatial LPIPS map before averaging to a scalar.
        version: LPIPS calibration version (``"0.1"`` or ``"0.0"``).
        pretrained: If ``True``, load pretrained ImageNet backbone and calibrated LPIPS weights.
            If ``False``, initialize randomly offline without downloading any weights.
    """

    higher_is_better: bool = False

    def __init__(
        self,
        net: str = "alex",
        spatial: bool = False,
        version: str = "0.1",
        pretrained: bool = True,
    ) -> None:
        if not isinstance(net, str) or net not in _VALID_NETS:
            raise ValueError(f"Unsupported LPIPS net={net!r}; expected one of {_VALID_NETS}.")
        if not isinstance(version, str) or version not in _VALID_VERSIONS:
            raise ValueError(
                f"Unsupported LPIPS version={version!r}; expected one of {_VALID_VERSIONS}."
            )
        if not isinstance(spatial, bool):
            raise ValueError(f"spatial must be a bool, got {spatial!r}.")
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be a bool, got {pretrained!r}.")

        self.net = net
        self.spatial = spatial
        self.version = version
        self.pretrained = pretrained

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The parameter 'pretrained' is deprecated",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Arguments other than a weight enum or `None` for 'weights'",
                category=UserWarning,
            )
            self.lpips = lpips.LPIPS(
                net=net,
                spatial=spatial,
                version=version,
                pretrained=pretrained,
                pnet_rand=not pretrained,
                pnet_tune=False,
                eval_mode=True,
                verbose=False,
            )

        if not self.pretrained and hasattr(self.lpips, "lins"):
            with torch.no_grad():
                for param in self.lpips.lins.parameters():
                    param.abs_()

        for param in self.lpips.parameters():
            param.requires_grad = False

        self.lpips.eval()
        self._device: torch.device = next(self.lpips.parameters()).device

    @property
    def device(self) -> torch.device:
        """Canonical device on which the cached LPIPS model currently resides."""
        return self._device

    def to(self, device: torch.device | str) -> LPIPSMetric:
        """Move the cached LPIPS model to ``device`` in-place without re-instantiating."""
        target_device = torch.device(device)
        if target_device.type == "cpu":
            target_device = torch.device("cpu")
        if self._device != target_device:
            with torch.inference_mode(False), torch.no_grad():
                self.lpips.to(target_device)
            self._device = next(self.lpips.parameters()).device
        self.lpips.eval()
        return self

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
    ) -> float:
        del meta
        if target is None:
            raise ValueError("LPIPSMetric requires a non-None target tensor.")
        if not isinstance(pred, torch.Tensor):
            raise TypeError(f"Expected pred to be a torch.Tensor, got {type(pred).__name__}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"Expected target to be a torch.Tensor, got {type(target).__name__}.")
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                f"LPIPSMetric expects 4D inputs [N, 3, H, W], "
                f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}."
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred has shape {tuple(pred.shape)}, "
                f"target has shape {tuple(target.shape)}."
            )
        if pred.shape[0] == 0:
            raise ValueError(
                f"LPIPSMetric expects batch size N >= 1, got shape {tuple(pred.shape)}."
            )
        if pred.shape[1] != 3:
            raise ValueError(
                f"LPIPSMetric expects 3-channel RGB inputs, got {pred.shape[1]} channels."
            )
        min_size = _MIN_SPATIAL_SIZE[self.net]
        if pred.shape[2] < min_size or pred.shape[3] < min_size:
            raise ValueError(
                f"LPIPSMetric(net={self.net!r}) requires spatial dimensions >= {min_size}, "
                f"got shape {tuple(pred.shape)}."
            )
        if not pred.is_floating_point() or not target.is_floating_point():
            raise ValueError(
                f"LPIPSMetric expects floating-point tensors in [0, 1], "
                f"got pred.dtype={pred.dtype}, target.dtype={target.dtype}."
            )
        if pred.device != target.device:
            raise ValueError(
                f"Device mismatch: pred is on {pred.device}, target is on {target.device}."
            )

        if self._device != pred.device:
            self.to(pred.device)
        self.lpips.eval()

        with torch.no_grad(), torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred_in = pred.float() * 2.0 - 1.0
            target_in = target.float() * 2.0 - 1.0
            score = self.lpips(pred_in, target_in)

        return float(score.mean().item())
