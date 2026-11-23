"""Perceptual loss modules backed by ``lpips``."""

from __future__ import annotations

import warnings

import lpips
import torch
from torch import nn

_VALID_NETS = ("alex", "vgg", "squeeze")
_VALID_VERSIONS = ("0.1", "0.0")


class LPIPSLoss(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss for RGB images in ``[0, 1]``.

    Internally rescales inputs from ``[0, 1]`` to ``[-1, 1]``, freezes all LPIPS parameters,
    and permanently locks the module in ``eval()`` mode so Dropout remains disabled.
    When ``pretrained=False``, passes ``pnet_rand=True`` (zero network download) and constrains
    random linear weights to be non-negative so distance semantics hold.
    """

    def __init__(
        self,
        net: str = "alex",
        version: str = "0.1",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if net not in _VALID_NETS:
            raise ValueError(f"Unsupported LPIPS net={net!r}; expected one of {_VALID_NETS}.")
        if version not in _VALID_VERSIONS:
            raise ValueError(
                f"Unsupported LPIPS version={version!r}; expected one of {_VALID_VERSIONS}."
            )

        self.net = net
        self.version = version
        self.pretrained = bool(pretrained)

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
                version=version,
                pretrained=self.pretrained,
                pnet_rand=not self.pretrained,
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

        self.eval()

    def train(self, mode: bool = True) -> LPIPSLoss:
        """Keep LPIPS backbone and linear dropout layers permanently locked in eval mode."""
        return super().train(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                f"LPIPSLoss expects 4D inputs (B, 3, H, W), "
                f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}."
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: pred has shape {tuple(pred.shape)}, "
                f"target has shape {tuple(target.shape)}."
            )
        if pred.shape[1] != 3:
            raise ValueError(
                f"LPIPSLoss expects 3-channel RGB inputs, got {pred.shape[1]} channels."
            )

        with torch.amp.autocast(device_type=pred.device.type, enabled=False):
            pred_in = pred.float() * 2.0 - 1.0
            target_in = target.float() * 2.0 - 1.0
            return self.lpips(pred_in, target_in).mean()
