"""Pure-PyTorch FP32 batched JPEG compression simulator (DiffJPEG) with auto pad/unpad."""

from __future__ import annotations

import math
import numbers

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from srtrainer.utils.misc import pad_to_multiple, unpad_tensor

__all__ = ["DiffJPEG"]


def _build_dct_matrix() -> torch.Tensor:
    """Build orthonormal 8x8 1D DCT-II transform matrix in float64, cast to float32."""
    u = np.arange(8, dtype=np.float64)[:, None]
    x = np.arange(8, dtype=np.float64)[None, :]
    alpha = np.full((8, 1), 0.5, dtype=np.float64)
    alpha[0, 0] = 1.0 / math.sqrt(8.0)
    dct = alpha * np.cos(((2.0 * x + 1.0) * u * math.pi) / 16.0)
    return torch.tensor(dct, dtype=torch.float32)


class DiffJPEG(nn.Module):
    """Batched FP32 RGB<->YCbCr 4:2:0 + 8x8 DCT JPEG simulator with auto pad/unpad to multiple of 16."""

    rgb_to_ycbcr_matrix: torch.Tensor
    rgb_to_ycbcr_shift: torch.Tensor
    ycbcr_to_rgb_matrix: torch.Tensor
    ycbcr_to_rgb_shift: torch.Tensor
    dct_matrix: torch.Tensor
    y_table: torch.Tensor
    c_table: torch.Tensor

    def __init__(self, differentiable: bool = False) -> None:
        super().__init__()
        if not isinstance(differentiable, bool):
            raise TypeError(
                f"differentiable must be a bool, got {type(differentiable).__name__}."
            )
        self.differentiable = differentiable

        self.register_buffer(
            "rgb_to_ycbcr_matrix",
            torch.tensor(
                [
                    [0.299000, 0.587000, 0.114000],
                    [-0.168736, -0.331264, 0.500000],
                    [0.500000, -0.418688, -0.081312],
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "rgb_to_ycbcr_shift",
            torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "ycbcr_to_rgb_matrix",
            torch.tensor(
                [
                    [1.0, 0.000000, 1.402000],
                    [1.0, -0.344136, -0.714136],
                    [1.0, 1.772000, 0.000000],
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "ycbcr_to_rgb_shift",
            torch.tensor([0.0, -128.0, -128.0], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer("dct_matrix", _build_dct_matrix(), persistent=True)

        # ITU-T T.81 Annex K.1 luminance & K.2 chrominance quantization tables
        self.register_buffer(
            "y_table",
            torch.tensor(
                [
                    [16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99],
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "c_table",
            torch.tensor(
                [
                    [17, 18, 24, 47, 99, 99, 99, 99],
                    [18, 21, 26, 66, 99, 99, 99, 99],
                    [24, 26, 56, 99, 99, 99, 99, 99],
                    [47, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                ],
                dtype=torch.float32,
            ),
            persistent=True,
        )

    def _round(self, z: torch.Tensor) -> torch.Tensor:
        if self.differentiable:
            rounded = torch.round(z)
            return rounded + (z - rounded) ** 3
        return torch.round(z)

    @staticmethod
    def _normalize_quality(
        quality: torch.Tensor | float | int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(quality, bool):
            raise ValueError(f"quality must be a numeric scalar or 1D tensor, got {quality!r}.")
        if isinstance(quality, numbers.Real):
            q_val = float(quality)
            if not math.isfinite(q_val):
                raise ValueError(f"quality must be finite, got {quality!r}.")
            return torch.full((batch_size,), q_val, device=device, dtype=torch.float32)
        if isinstance(quality, torch.Tensor):
            if quality.dtype == torch.bool:
                raise ValueError("quality tensor must be numeric, not bool.")
            q_t = quality.to(device=device, dtype=torch.float32)
            if q_t.ndim == 0:
                q_t = q_t.expand(batch_size)
            elif q_t.ndim == 1 and q_t.shape[0] == 1:
                q_t = q_t.expand(batch_size)
            elif q_t.ndim != 1 or q_t.shape[0] != batch_size:
                raise ValueError(
                    f"Expected quality tensor of shape (), (1,), or ({batch_size},), "
                    f"got {tuple(quality.shape)}."
                )
            if not bool(torch.isfinite(q_t).all()):
                raise ValueError("quality tensor contains non-finite values.")
            return q_t
        raise TypeError(
            f"quality must be a float, int, or torch.Tensor, got {type(quality).__name__}."
        )

    def _dct_quant_idct_channel(
        self,
        ch: torch.Tensor,
        qtable: torch.Tensor,
        dct_m: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = ch.shape
        nh, nw = h // 8, w // 8
        blocks = ch.reshape(b, nh, 8, nw, 8).permute(0, 1, 3, 2, 4).reshape(b, nh * nw, 8, 8)
        blocks = blocks - 128.0
        dct = torch.matmul(torch.matmul(dct_m, blocks), dct_m.T)
        dequant = self._round(dct / qtable) * qtable
        rec = torch.matmul(torch.matmul(dct_m.T, dequant), dct_m) + 128.0
        return rec.reshape(b, nh, nw, 8, 8).permute(0, 1, 3, 2, 4).reshape(b, 1, h, w)

    def forward(self, x: torch.Tensor, quality: torch.Tensor | float | int) -> torch.Tensor:
        """Apply batched JPEG compression to a ``[B, 3, H, W]`` RGB tensor in ``[0, 1]``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected 4D RGB tensor [B, 3, H, W], got shape {tuple(x.shape)}.")
        b, _, h_in, w_in = x.shape
        if b < 1 or h_in < 1 or w_in < 1:
            raise ValueError(f"Expected non-empty tensor dimensions, got shape {tuple(x.shape)}.")

        device = x.device
        with torch.amp.autocast(device_type=device.type, enabled=False):
            q = self._normalize_quality(quality, batch_size=b, device=device).clamp(1.0, 100.0)
            factor = torch.where(q < 50.0, 5000.0 / q, 200.0 - 2.0 * q) / 100.0
            q_scale = factor.view(b, 1, 1, 1)

            x_fp32 = x.float().clamp(0.0, 1.0)
            x_pad, (orig_h, orig_w) = pad_to_multiple(x_fp32, multiple=16, mode="reflect")
            x_255 = x_pad * 255.0

            rgb2ycbcr = self.rgb_to_ycbcr_matrix.to(device=device, dtype=torch.float32)
            rgb2ycbcr_shift = self.rgb_to_ycbcr_shift.to(device=device, dtype=torch.float32)
            ycbcr2rgb = self.ycbcr_to_rgb_matrix.to(device=device, dtype=torch.float32)
            ycbcr2rgb_shift = self.ycbcr_to_rgb_shift.to(device=device, dtype=torch.float32)
            dct_m = self.dct_matrix.to(device=device, dtype=torch.float32)
            y_base = self.y_table.to(device=device, dtype=torch.float32)
            c_base = self.c_table.to(device=device, dtype=torch.float32)

            # Clamp scaled quantization tables to >= 1.0 so quality==100 never divides by 0.
            y_qtable = torch.clamp(y_base.view(1, 1, 8, 8) * q_scale, min=1.0)
            c_qtable = torch.clamp(c_base.view(1, 1, 8, 8) * q_scale, min=1.0)

            ycbcr = torch.einsum("oi,bihw->bohw", rgb2ycbcr, x_255) + rgb2ycbcr_shift
            y = ycbcr[:, 0:1]
            cb_sub = F.avg_pool2d(ycbcr[:, 1:2], kernel_size=2, stride=2)
            cr_sub = F.avg_pool2d(ycbcr[:, 2:3], kernel_size=2, stride=2)

            y_rec = self._dct_quant_idct_channel(y, y_qtable, dct_m)
            cb_rec = self._dct_quant_idct_channel(cb_sub, c_qtable, dct_m)
            cr_rec = self._dct_quant_idct_channel(cr_sub, c_qtable, dct_m)

            cb_up = F.interpolate(cb_rec, scale_factor=2, mode="nearest")
            cr_up = F.interpolate(cr_rec, scale_factor=2, mode="nearest")

            ycbcr_rec = torch.cat((y_rec, cb_up, cr_up), dim=1) + ycbcr2rgb_shift
            rgb_rec = torch.einsum("oi,bihw->bohw", ycbcr2rgb, ycbcr_rec)
            rgb_out = rgb_rec.clamp(0.0, 255.0) / 255.0
            return unpad_tensor(rgb_out, orig_h, orig_w, scale=1)
