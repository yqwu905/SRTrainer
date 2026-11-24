"""Self-contained pure-PyTorch 2D Qwen-Image VAE matching official VOSR2 weights."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.checkpoint import checkpoint

from utils.misc import load_weights

__all__ = [
    "AutoencoderKLQwenImage2D",
    "DiagonalGaussianDistribution",
]

DEFAULT_QWEN_LATENTS_MEAN: tuple[float, ...] = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
DEFAULT_QWEN_LATENTS_STD: tuple[float, ...] = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


def _get_activation(name: str) -> nn.Module:
    norm = name.strip().lower()
    if norm in ("silu", "swish"):
        return nn.SiLU()
    if norm == "gelu":
        return nn.GELU()
    if norm == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported Qwen VAE activation {name!r}.")


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior matching diffusers' VAE posterior API."""

    def __init__(self, parameters: torch.Tensor) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def mode(self) -> torch.Tensor:
        return self.mean

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        return self.mean + self.std * noise


class RMSNorm2D(nn.Module):
    """Channel-first 2D RMS normalization used throughout Qwen-Image VAE."""

    def __init__(self, dim: int, channel_first: bool = True, bias: bool = False) -> None:
        super().__init__()
        if not channel_first:
            raise ValueError("Only channel_first=True is supported for RMSNorm2D.")
        self.channel_first = True
        self.scale = dim**0.5
        shape = (dim, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        if bias:
            self.bias: nn.Parameter | float = nn.Parameter(torch.zeros(shape))
        else:
            self.bias = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


class ResidualBlock2D(nn.Module):
    """2D residual block with RMSNorm2D and SiLU activations."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = _get_activation(non_linearity)

        self.norm1 = RMSNorm2D(in_dim, channel_first=True)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.norm2 = RMSNorm2D(out_dim, channel_first=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.conv_shortcut: nn.Module = (
            nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + h


class AttentionBlock2D(nn.Module):
    """Single-head spatial self-attention using 4D SDPA layout for O(N) memory kernels."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b, 3, c, h * w).permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.unbind(dim=1)
        x_attn = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x_attn = x_attn.permute(0, 2, 1).reshape(b, c, h, w)
        x_attn = self.proj(x_attn)
        return x_attn + identity


class Resample2D(nn.Module):
    """2x spatial upsampling or strided downsampling block."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample: nn.Module = nn.Sequential(
                nn.Upsample(scale_factor=2.0, mode="nearest"),
                nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            )
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            )
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class MidBlock2D(nn.Module):
    """Bottleneck residual + attention block."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        resnets = [ResidualBlock2D(dim, dim, dropout, non_linearity)]
        attentions: list[nn.Module] = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock2D(dim))
            resnets.append(ResidualBlock2D(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = attn(x)
            x = resnet(x)
        return x


class Encoder2D(nn.Module):
    """2D Qwen-Image VAE encoder (8x spatial compression) with optional activation checkpointing."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 32,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        dropout: float = 0.0,
        non_linearity: str = "silu",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.nonlinearity = _get_activation(non_linearity)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        dims = [dim * u for u in [1] + self.dim_mult]
        scale = 1.0
        self.conv_in = nn.Conv2d(3, dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([])
        out_dim = dims[-1]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock2D(in_dim, out_dim, dropout, non_linearity))
                if scale in self.attn_scales:
                    self.down_blocks.append(AttentionBlock2D(out_dim))
                in_dim = out_dim
            if i != len(self.dim_mult) - 1:
                self.down_blocks.append(Resample2D(out_dim, mode="downsample2d"))
                scale /= 2.0

        self.mid_block = MidBlock2D(out_dim, dropout, non_linearity, num_layers=1)
        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, z_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled() and (
            x.requires_grad or self.training
        )
        x = self.conv_in(x)
        if use_ckpt:
            for layer in self.down_blocks:
                x = checkpoint(layer, x, use_reentrant=False)
            x = checkpoint(self.mid_block, x, use_reentrant=False)
        else:
            for layer in self.down_blocks:
                x = layer(x)
            x = self.mid_block(x)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        return self.conv_out(x)


class UpBlock2D(nn.Module):
    """Decoder upsampling stage with ``num_res_blocks + 1`` residual blocks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        resnets: list[nn.Module] = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(ResidualBlock2D(cur, out_dim, dropout, non_linearity))
            cur = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers: nn.ModuleList | None = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([Resample2D(out_dim, mode=upsample_mode)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Decoder2D(nn.Module):
    """2D Qwen-Image VAE decoder (8x spatial upsampling) with optional activation checkpointing."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        dropout: float = 0.0,
        non_linearity: str = "silu",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.nonlinearity = _get_activation(non_linearity)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        dims = [dim * u for u in [self.dim_mult[-1]] + self.dim_mult[::-1]]
        self.conv_in = nn.Conv2d(z_dim, dims[0], kernel_size=3, padding=1)
        self.mid_block = MidBlock2D(dims[0], dropout, non_linearity, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        out_dim = dims[-1]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            if i > 0:
                in_dim = in_dim // 2
            upsample_mode = "upsample2d" if i != len(self.dim_mult) - 1 else None
            self.up_blocks.append(
                UpBlock2D(in_dim, out_dim, num_res_blocks, dropout, upsample_mode, non_linearity)
            )

        self.norm_out = RMSNorm2D(out_dim)
        self.conv_out = nn.Conv2d(out_dim, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled() and (
            x.requires_grad or self.training
        )
        x = self.conv_in(x)
        if use_ckpt:
            x = checkpoint(self.mid_block, x, use_reentrant=False)
            for up_block in self.up_blocks:
                x = checkpoint(up_block, x, use_reentrant=False)
        else:
            x = self.mid_block(x)
            for up_block in self.up_blocks:
                x = up_block(x)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        return self.conv_out(x)


def _resolve_qwen_vae_files(
    pretrained_path_or_repo: str | Path,
    *,
    subfolder: str | None = None,
) -> tuple[Path | None, Path]:
    """Resolve ``(config_path, weights_path)`` for a local file/dir or HuggingFace Hub repo."""
    raw_path = Path(pretrained_path_or_repo)
    if subfolder is not None and raw_path.exists():
        candidate = raw_path / subfolder
        if candidate.exists():
            raw_path = candidate

    config_path: Path | None = None
    weights_path: Path | None = None

    if raw_path.is_file():
        weights_path = raw_path
        sibling_cfg = raw_path.parent / "config.json"
        if sibling_cfg.is_file():
            config_path = sibling_cfg
    elif raw_path.is_dir():
        cfg_candidate = raw_path / "config.json"
        if cfg_candidate.is_file():
            config_path = cfg_candidate
        for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
            w_candidate = raw_path / name
            if w_candidate.is_file():
                weights_path = w_candidate
                break
        if weights_path is None:
            raise FileNotFoundError(
                f"No safetensors weights found in Qwen VAE directory: {raw_path}"
            )
    else:
        repo_str = str(pretrained_path_or_repo)
        if raw_path.is_absolute() or repo_str.count("/") != 1:
            raise FileNotFoundError(
                f"Qwen VAE checkpoint path not found: {pretrained_path_or_repo}"
            )
        from huggingface_hub import snapshot_download

        prefix = f"{subfolder.strip('/')}/" if subfolder else ""
        snapshot_dir = Path(
            snapshot_download(
                repo_id=repo_str,
                allow_patterns=[
                    f"{prefix}config.json",
                    f"{prefix}diffusion_pytorch_model.safetensors",
                ],
            )
        )
        sub_dir = snapshot_dir / subfolder if subfolder else snapshot_dir
        config_path = sub_dir / "config.json"
        weights_path = sub_dir / "diffusion_pytorch_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Incomplete Qwen VAE snapshot under {sub_dir} from {repo_str}."
            )

    assert weights_path is not None
    return config_path, weights_path


class AutoencoderKLQwenImage2D(nn.Module):
    """2D Qwen-Image VAE (8x spatial compression, 16-channel normalized latents).

    State-dict keys match ``CSWRY/VOSR/Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors``
    (186 tensors) with ``strict=True``.
    """

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        dropout: float = 0.0,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        non_linearity: str = "silu",
        gradient_checkpointing: bool = False,
        pretrained_path: str | Path | None = None,
        subfolder: str | None = None,
        pretrained_strict: bool = True,
        pretrained_key: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(pretrained_strict, bool):
            raise ValueError(
                f"pretrained_strict must be a bool, got {type(pretrained_strict).__name__}: {pretrained_strict!r}."
            )

        self.base_dim = int(base_dim)
        self.z_dim = int(z_dim)
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.temperal_downsample = list(temperal_downsample)
        self.spatial_compression_ratio = 2 ** (len(self.dim_mult) - 1)
        self._explicit_latents_mean = latents_mean is not None
        self._explicit_latents_std = latents_std is not None

        self.encoder = Encoder2D(
            dim=self.base_dim,
            z_dim=self.z_dim * 2,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            attn_scales=attn_scales,
            dropout=dropout,
            non_linearity=non_linearity,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.quant_conv = nn.Conv2d(self.z_dim * 2, self.z_dim * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(self.z_dim, self.z_dim, kernel_size=1)
        self.decoder = Decoder2D(
            dim=self.base_dim,
            z_dim=self.z_dim,
            dim_mult=self.dim_mult,
            num_res_blocks=self.num_res_blocks,
            attn_scales=attn_scales,
            dropout=dropout,
            non_linearity=non_linearity,
            gradient_checkpointing=gradient_checkpointing,
        )

        if latents_mean is None:
            if self.z_dim != len(DEFAULT_QWEN_LATENTS_MEAN):
                raise ValueError(
                    f"latents_mean must be provided explicitly when z_dim={self.z_dim} != "
                    f"{len(DEFAULT_QWEN_LATENTS_MEAN)}."
                )
            mean_vals = list(DEFAULT_QWEN_LATENTS_MEAN)
        else:
            mean_vals = [float(v) for v in latents_mean]
        if latents_std is None:
            if self.z_dim != len(DEFAULT_QWEN_LATENTS_STD):
                raise ValueError(
                    f"latents_std must be provided explicitly when z_dim={self.z_dim} != "
                    f"{len(DEFAULT_QWEN_LATENTS_STD)}."
                )
            std_vals = list(DEFAULT_QWEN_LATENTS_STD)
        else:
            std_vals = [float(v) for v in latents_std]

        self._set_latent_stats(mean_vals, std_vals)

        if pretrained_path is not None:
            self.load_pretrained(
                pretrained_path,
                subfolder=subfolder,
                strict=pretrained_strict,
                key=pretrained_key,
            )

    def _set_latent_stats(self, mean_vals: Sequence[float], std_vals: Sequence[float]) -> None:
        means = [float(v) for v in mean_vals]
        stds = [float(v) for v in std_vals]
        if len(means) != self.z_dim or len(stds) != self.z_dim:
            raise ValueError(
                f"latents_mean ({len(means)}) and latents_std ({len(stds)}) "
                f"must have length z_dim={self.z_dim}."
            )
        if any(s <= 0.0 for s in stds):
            raise ValueError("All latents_std entries must be strictly positive.")

        self.latents_mean = means
        self.latents_std = stds
        # Non-persistent buffers keep state_dict() strictly identical to the 186-key official file.
        mean_t = torch.tensor(means, dtype=torch.float32).view(1, self.z_dim, 1, 1)
        std_inv_t = (1.0 / torch.tensor(stds, dtype=torch.float32)).view(1, self.z_dim, 1, 1)
        if "_latents_mean" in self._buffers and self._buffers["_latents_mean"] is not None:
            dev = self._buffers["_latents_mean"].device
            self._buffers["_latents_mean"] = mean_t.to(device=dev)
            self._buffers["_latents_std_inv"] = std_inv_t.to(device=dev)
        else:
            self.register_buffer("_latents_mean", mean_t, persistent=False)
            self.register_buffer("_latents_std_inv", std_inv_t, persistent=False)

    def load_pretrained(
        self,
        pretrained_path_or_repo: str | Path,
        *,
        subfolder: str | None = None,
        strict: bool = True,
        key: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Load Qwen VAE weights from a local ``.safetensors``/``.pth`` file, directory, or HF repo."""
        raw_path = Path(pretrained_path_or_repo)
        if raw_path.is_file():
            return load_weights(self, raw_path, strict=strict, key=key)

        config_path, weights_path = _resolve_qwen_vae_files(
            pretrained_path_or_repo,
            subfolder=subfolder,
        )
        if config_path is not None and config_path.is_file():
            raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            cfg_mean = raw_cfg.get("latents_mean")
            cfg_std = raw_cfg.get("latents_std")
            new_mean = (
                cfg_mean
                if (not self._explicit_latents_mean and isinstance(cfg_mean, list) and len(cfg_mean) == self.z_dim)
                else self.latents_mean
            )
            new_std = (
                cfg_std
                if (not self._explicit_latents_std and isinstance(cfg_std, list) and len(cfg_std) == self.z_dim)
                else self.latents_std
            )
            self._set_latent_stats(new_mean, new_std)

        return load_weights(self, weights_path, strict=strict, key=key)

    def encode_posterior(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.quant_conv(self.encoder(x))
        return DiagonalGaussianDistribution(h)

    def encode_normalized(
        self,
        x: torch.Tensor,
        posterior_mode: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Encode ``[-1, 1]`` RGB images into channel-normalized 16-ch VOSR2 latents."""
        posterior = self.encode_posterior(x)
        z = posterior.mode() if posterior_mode else posterior.sample(generator=generator)
        mean = self._latents_mean.to(device=z.device, dtype=z.dtype)
        std_inv = self._latents_std_inv.to(device=z.device, dtype=z.dtype)
        return (z - mean) * std_inv

    def decode_normalized(
        self,
        z_norm: torch.Tensor,
        clamp_output: bool = True,
    ) -> torch.Tensor:
        """Unnormalize VOSR2 latents and decode to RGB in ``[-1, 1]``."""
        mean = self._latents_mean.to(device=z_norm.device, dtype=z_norm.dtype)
        std_inv = self._latents_std_inv.to(device=z_norm.device, dtype=z_norm.dtype)
        z = z_norm / std_inv + mean
        dec = self.decoder(self.post_quant_conv(z))
        return dec.clamp(-1.0, 1.0) if clamp_output else dec

    def forward(
        self,
        sample: torch.Tensor,
        posterior_mode: bool = True,
        clamp_output: bool = True,
    ) -> torch.Tensor:
        z_norm = self.encode_normalized(sample, posterior_mode=posterior_mode)
        return self.decode_normalized(z_norm, clamp_output=clamp_output)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path_or_repo: str | Path,
        *,
        subfolder: str | None = None,
        gradient_checkpointing: bool = False,
    ) -> AutoencoderKLQwenImage2D:
        """Load ``AutoencoderKLQwenImage2D`` from a local directory/file or HuggingFace Hub repo."""
        config_path, weights_path = _resolve_qwen_vae_files(
            pretrained_path_or_repo,
            subfolder=subfolder,
        )

        kwargs: dict[str, Any] = {}
        if config_path is not None:
            raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            allowed_keys = (
                "base_dim",
                "z_dim",
                "dim_mult",
                "num_res_blocks",
                "attn_scales",
                "temperal_downsample",
                "dropout",
                "latents_mean",
                "latents_std",
                "non_linearity",
            )
            for key in allowed_keys:
                if key in raw_cfg:
                    kwargs[key] = raw_cfg[key]

        kwargs["gradient_checkpointing"] = gradient_checkpointing
        model = cls(**kwargs)
        state = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state, strict=True)
        return model
