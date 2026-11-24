"""Lightweight test-only dummy networks for fast CPU unit & E2E tests."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class _DummyResBlock(nn.Module):
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


class DummySRNet(nn.Module):
    """Minimal test-only residual-conv + PixelShuffle network for standalone/TiledInferencer tests."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 8,
        num_block: int = 1,
        scale: int = 1,
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
        self.body = nn.Sequential(*[_DummyResBlock(num_feat) for _ in range(num_block)])
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


class DummyVAE(nn.Module):
    """Lightweight replaceable test-only VAE implementing VOSR's encode/decode contract."""

    def __init__(
        self,
        z_dim: int = 4,
        spatial_compression_ratio: int = 2,
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.spatial_compression_ratio = int(spatial_compression_ratio)
        s = self.spatial_compression_ratio
        self.enc_conv = nn.Conv2d(3, self.z_dim, kernel_size=s, stride=s)
        self.dec_conv = nn.Conv2d(self.z_dim, 3 * s * s, kernel_size=3, padding=1)
        self.dec_up: nn.Module = nn.PixelShuffle(s) if s > 1 else nn.Identity()

    def encode_normalized(
        self,
        x: torch.Tensor,
        posterior_mode: bool = True,
    ) -> torch.Tensor:
        del posterior_mode
        return self.enc_conv(x)

    def decode_normalized(
        self,
        z_norm: torch.Tensor,
        clamp_output: bool = True,
    ) -> torch.Tensor:
        out = self.dec_up(self.dec_conv(z_norm))
        return out.clamp(-1.0, 1.0) if clamp_output else out

    def forward(
        self,
        sample: torch.Tensor,
        posterior_mode: bool = True,
        clamp_output: bool = True,
    ) -> torch.Tensor:
        return self.decode_normalized(
            self.encode_normalized(sample, posterior_mode=posterior_mode),
            clamp_output=clamp_output,
        )


class DummyDiT(nn.Module):
    """Lightweight replaceable test-only latent denoiser implementing VOSR's DiT contract."""

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        patch_size: int = 2,
        hidden_ch: int = 8,
        z_dims: int | None = 8,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.z_dims = z_dims
        self.conv_in = nn.Conv2d(self.in_channels, hidden_ch, kernel_size=3, padding=1)
        self.t_proj = nn.Linear(1, hidden_ch)
        self.z_proj: nn.Module | None = (
            nn.Linear(int(z_dims), hidden_ch) if z_dims is not None else None
        )
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(hidden_ch, self.out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        h = self.conv_in(x)
        t_emb = self.t_proj(t.float().view(-1, 1)).to(dtype=h.dtype)
        h = h + t_emb[:, :, None, None]
        if self.z_proj is not None:
            if z is None:
                raise ValueError("DummyDiT requires conditioning tensor z when z_dims is set.")
            z_tensor = z[0] if isinstance(z, (list, tuple)) else z
            z_emb = self.z_proj(z_tensor.mean(dim=1).to(dtype=h.dtype))
            h = h + z_emb[:, :, None, None]
        return self.conv_out(self.act(h))


class DummyDino(nn.Module):
    """Lightweight replaceable test-only vision encoder implementing VOSR's DINO contract."""

    def __init__(
        self,
        embed_dim: int = 8,
        image_size: int = 16,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            x.float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.proj(resized).flatten(2).transpose(1, 2)


def get_dummy_vosr_arch_dict(
    *,
    z_dim: int = 4,
    dino_dim: int = 8,
    with_dino: bool = True,
    spatial_compression_ratio: int = 2,
    patch_size: int = 2,
    hidden_ch: int = 8,
) -> dict[str, Any]:
    """Build a compact Hydra arch dict for fast CPU ``VOSRModel`` tests."""
    arch: dict[str, Any] = {
        "vae": {
            "_target_": "tests.dummy_arch.DummyVAE",
            "z_dim": z_dim,
            "spatial_compression_ratio": spatial_compression_ratio,
        },
        "dit": {
            "_target_": "tests.dummy_arch.DummyDiT",
            "in_channels": z_dim * 2,
            "out_channels": z_dim,
            "patch_size": patch_size,
            "hidden_ch": hidden_ch,
            "z_dims": dino_dim if with_dino else None,
        },
    }
    if with_dino:
        arch["dino"] = {
            "_target_": "tests.dummy_arch.DummyDino",
            "embed_dim": dino_dim,
            "image_size": 16,
            "patch_size": 4,
        }
    else:
        arch["dino"] = None
    return arch
