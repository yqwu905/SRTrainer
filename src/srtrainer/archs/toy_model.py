"""Lightweight placeholder super-resolution network for framework development and testing."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class _ToyResBlock(nn.Module):
    """Minimal residual conv block with scaled second conv for stable convergence."""

    def __init__(self, num_feat: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        with torch.no_grad():
            self.conv2.weight.mul_(0.1)
            if self.conv2.bias is not None:
                self.conv2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class ToyModel(nn.Module):
    """Lightweight placeholder residual-conv + PixelShuffle SR network.

    Accepts ``[B, num_in_ch, H, W]`` float tensors and returns unbounded
    ``[B, num_out_ch, H * scale, W * scale]`` float tensors. Supports any positive
    integer ``scale >= 1``, ``num_block >= 0``, and arbitrary ``H, W >= 1``.
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 32,
        num_block: int = 2,
        scale: int = 4,
    ) -> None:
        super().__init__()
        if not _is_positive_int(scale):
            raise ValueError(f"scale must be a positive integer, got {scale!r}.")
        for name, val in (
            ("num_in_ch", num_in_ch),
            ("num_out_ch", num_out_ch),
            ("num_feat", num_feat),
        ):
            if not _is_positive_int(val):
                raise ValueError(f"{name} must be a positive integer, got {val!r}.")
        if not _is_non_negative_int(num_block):
            raise ValueError(f"num_block must be a non-negative integer, got {num_block!r}.")

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_block = num_block
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_ToyResBlock(num_feat) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch * (scale * scale), 3, 1, 1)
        self.upsample: nn.Module = nn.PixelShuffle(scale) if scale > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {tuple(x.shape)}.")
        if x.shape[1] != self.num_in_ch:
            raise ValueError(
                f"Expected {self.num_in_ch} input channels, got {x.shape[1]} channels."
            )
        if x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError(
                f"Expected positive spatial dimensions (H >= 1, W >= 1), got shape {tuple(x.shape)}."
            )

        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        return self.upsample(self.conv_last(feat))
