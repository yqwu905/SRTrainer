"""VOSR2 single-step LightningDiT backbone (no VOSR1 r_embedder / multi-step paths)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "RMSNorm",
    "SwiGLUFFN",
    "VisionRotaryEmbeddingFast",
    "VOSR2DiT",
]

_VALID_ROPE_MODES: frozenset[str] = frozenset({"interpolate", "extrapolate"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved 2D rotary half-turn matching EVA-02 / VOSR ``pos_embed.rotate_half``."""
    x_pair = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_pair.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def _build_2d_rope_cos_sin(
    gh: int,
    gw: int,
    half_head_dim: int,
    pt_seq_len: int,
    rope_mode: str,
    device: torch.device,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build axial 2D RoPE ``(cos, sin)`` tables of shape ``(gh * gw, half_head_dim * 2)``."""
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, half_head_dim, 2, device=device)[: (half_head_dim // 2)].float()
            / half_head_dim
        )
    )
    if rope_mode == "interpolate":
        pos_h = torch.arange(gh, device=device, dtype=torch.float32) / gh * pt_seq_len
        pos_w = torch.arange(gw, device=device, dtype=torch.float32) / gw * pt_seq_len
    elif rope_mode == "extrapolate":
        pos_h = torch.arange(gh, device=device, dtype=torch.float32)
        pos_w = torch.arange(gw, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported rope_mode={rope_mode!r}; expected {sorted(_VALID_ROPE_MODES)}.")

    freqs_h = torch.einsum("..., f -> ... f", pos_h, freqs).repeat_interleave(2, dim=-1)
    freqs_w = torch.einsum("..., f -> ... f", pos_w, freqs).repeat_interleave(2, dim=-1)
    grid_h = freqs_h[:, None, :].expand(gh, gw, -1)
    grid_w = freqs_w[None, :, :].expand(gh, gw, -1)
    freqs_2d = torch.cat((grid_h, grid_w), dim=-1).reshape(gh * gw, -1)
    return freqs_2d.cos(), freqs_2d.sin()


def _modulate_adasin(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor,
) -> torch.Tensor:
    if shift is None:
        return x * (1.0 + scale.unsqueeze(1))
    return x * (1.0 + scale) + shift


def _modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor,
) -> torch.Tensor:
    if shift is None:
        return x * (1.0 + scale.unsqueeze(1))
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (pure PyTorch, matching VOSR ``models/rmsnorm.py``)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block matching VOSR ``models/swiglu_ffn.py`` state keys."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class _Mlp(nn.Module):
    """Two-layer MLP with ``fc1`` / ``fc2`` keys matching ``timm`` ``Mlp``."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class _PatchEmbed(nn.Module):
    """Patchify Conv2d matching ``timm`` ``PatchEmbed`` ``proj`` key and flexible H/W support."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionRotaryEmbeddingFast(nn.Module):
    """Precomputed reference-grid 2D axial RoPE matching VOSR ``feat_rope`` buffers."""

    def __init__(self, dim: int, pt_seq_len: int = 32, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.pt_seq_len = int(pt_seq_len)
        self.theta = float(theta)
        freqs_cos, freqs_sin = _build_2d_rope_cos_sin(
            gh=self.pt_seq_len,
            gw=self.pt_seq_len,
            half_head_dim=self.dim,
            pt_seq_len=self.pt_seq_len,
            rope_mode="interpolate",
            device=torch.device("cpu"),
            theta=self.theta,
        )
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        cos = self.freqs_cos.to(device=t.device, dtype=t.dtype)
        sin = self.freqs_sin.to(device=t.device, dtype=t.dtype)
        return (t * cos + _rotate_half(t) * sin).contiguous()


class _TableRoPE:
    """Lightweight dynamic RoPE callable for non-reference or rectangular grids."""

    def __init__(self, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> None:
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        cos = self.freqs_cos.to(device=t.device, dtype=t.dtype)
        sin = self.freqs_sin.to(device=t.device, dtype=t.dtype)
        return (t * cos + _rotate_half(t) * sin).contiguous()


class MultiHeadCrossAttention(nn.Module):
    """DINOv2 conditioning cross-attention matching VOSR ``blocks.*.cross_attn`` keys."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm: nn.Module = RMSNorm(self.head_dim)
            self.k_norm: nn.Module = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        b_cond, n_cond, _ = cond.shape

        q = self.q_linear(x).view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_linear(cond).view(b_cond, n_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_linear(cond).view(b_cond, n_cond, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = self.attn_drop(attn.softmax(dim=-1))
            out = attn @ v

        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        return self.proj_drop(self.proj(out))


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm and 2D RoPE matching VOSR ``blocks.*.attn`` keys."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 24,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        fused_attn: bool = True,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        norm_layer: type[nn.Module] = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm: nn.Module = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm: nn.Module = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: Any = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = self.attn_drop(attn.softmax(dim=-1))
            out = attn @ v

        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar timestep embedder matching DiT / VOSR."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class VOSR2DiTBlock(nn.Module):
    """Single VOSR2 LightningDiT block (AdaLN-single + Self-Attn + Cross-Attn + SwiGLU FFN)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        z_dims: int | None = 1024,
    ) -> None:
        super().__init__()
        self.wo_shift = bool(wo_shift)
        if use_rmsnorm:
            self.norm1: nn.Module = RMSNorm(hidden_size)
            self.norm2: nn.Module = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = SelfAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
        )

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_swiglu:
            self.mlp: nn.Module = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        else:
            self.mlp = _Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0.0)

        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.z_dims = z_dims
        if self.z_dims is not None:
            self.cross_attn = MultiHeadCrossAttention(
                d_model=hidden_size,
                num_heads=num_heads,
                qk_norm=use_qknorm,
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        z: torch.Tensor | None = None,
        feat_rope: Any = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + c.reshape(b, 6, -1)
        ).chunk(6, dim=1)

        if self.wo_shift:
            shift_msa = None
            shift_mlp = None

        x = x + gate_msa * self.attn(
            _modulate_adasin(self.norm1(x), shift_msa, scale_msa),
            rope=feat_rope,
        )
        if self.z_dims is not None and z is not None:
            x = x + self.cross_attn(x, z)
        x = x + gate_mlp * self.mlp(_modulate_adasin(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Output projection layer with AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = True,
    ) -> None:
        super().__init__()
        if use_rmsnorm:
            self.norm_final: nn.Module = RMSNorm(hidden_size)
        else:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class VOSR2DiT(nn.Module):
    """VOSR2 single-step LightningDiT backbone (849 exact official checkpoint keys).

    Strictly targets VOSR2 one-step (``auxiliary_time_cond=False``, single ``t_embedder``).
    Supports arbitrary square and rectangular latent grids divisible by ``patch_size``
    via dynamic 2D RoPE (``interpolate`` or ``extrapolate``).
    """

    def __init__(
        self,
        input_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 32,
        out_channels: int = 16,
        hidden_size: int = 1536,
        depth: int = 36,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        z_dims: int | None = 1024,
        encdim_ratio: int = 3,
        use_qknorm: bool = True,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_checkpoint: bool = False,
        rope_mode: str = "interpolate",
    ) -> None:
        super().__init__()
        for name, val in (
            ("input_size", input_size),
            ("patch_size", patch_size),
            ("in_channels", in_channels),
            ("out_channels", out_channels),
            ("hidden_size", hidden_size),
            ("depth", depth),
            ("num_heads", num_heads),
            ("encdim_ratio", encdim_ratio),
        ):
            if not _is_positive_int(val):
                raise ValueError(f"{name} must be a positive integer, got {val!r}.")
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by patch_size ({patch_size})."
            )
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        if use_rope and (hidden_size // num_heads) % 4 != 0:
            raise ValueError(
                f"head_dim ({hidden_size // num_heads}) must be divisible by 4 when use_rope=True."
            )
        if z_dims is not None and not _is_positive_int(z_dims):
            raise ValueError(f"z_dims must be a positive integer or None, got {z_dims!r}.")
        if rope_mode not in _VALID_ROPE_MODES:
            raise ValueError(
                f"Unsupported rope_mode={rope_mode!r}; expected one of {sorted(_VALID_ROPE_MODES)}."
            )

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = float(mlp_ratio)
        self.z_dims = z_dims
        self.encdim_ratio = encdim_ratio
        self.use_qknorm = bool(use_qknorm)
        self.use_swiglu = bool(use_swiglu)
        self.use_rope = bool(use_rope)
        self.use_rmsnorm = bool(use_rmsnorm)
        self.wo_shift = bool(wo_shift)
        self.use_checkpoint = bool(use_checkpoint)
        self.rope_mode = rope_mode

        self.x_embedder = _PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope: VisionRotaryEmbeddingFast | None = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        self.blocks = nn.ModuleList(
            [
                VOSR2DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=self.mlp_ratio,
                    use_qknorm=self.use_qknorm,
                    use_swiglu=self.use_swiglu,
                    use_rmsnorm=self.use_rmsnorm,
                    wo_shift=self.wo_shift,
                    z_dims=z_dims,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
            use_rmsnorm=self.use_rmsnorm,
        )

        if self.z_dims is not None:
            self.layer_norm = nn.LayerNorm(self.z_dims)
            self.mlp_ca = _Mlp(
                in_features=self.z_dims,
                hidden_features=hidden_size * encdim_ratio,
                out_features=hidden_size,
                drop=0.0,
            )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Apply official DiT / LightningDiT weight initialization."""

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0.0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_layer.linear.weight, 0.0)
        nn.init.constant_(self.final_layer.linear.bias, 0.0)

    def unpatchify(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """Unpatchify ``(N, gh * gw, patch_size**2 * C)`` -> ``(N, C, gh * p, gw * p)``."""
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, gh * p, gw * p)

    def _resolve_rope(self, gh: int, gw: int, device: torch.device) -> Any:
        if not self.use_rope:
            return None
        pt_seq_len = self.input_size // self.patch_size
        if gh == pt_seq_len and gw == pt_seq_len and self.feat_rope is not None:
            return self.feat_rope
        half_head_dim = self.hidden_size // self.num_heads // 2
        cos, sin = _build_2d_rope_cos_sin(
            gh=gh,
            gw=gw,
            half_head_dim=half_head_dim,
            pt_seq_len=pt_seq_len,
            rope_mode=self.rope_mode,
            device=device,
        )
        return _TableRoPE(cos, sin)

    def forward_dit(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Predict one-step latent velocity from ``cat([lq_latent, noise], dim=1)`` at timestep ``t``."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D latent input (B, C, H, W), got shape {tuple(x.shape)}.")
        _, c_in, h, w = x.shape
        if c_in != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {c_in}.")
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Latent spatial size {(h, w)} must be divisible by patch_size={self.patch_size}."
            )

        gh = h // self.patch_size
        gw = w // self.patch_size

        tokens = self.x_embedder(x)
        c = self.t_embedder(t)
        c0 = self.t_block(c)

        cond: torch.Tensor | None = None
        if self.z_dims is not None:
            if z is None:
                raise ValueError("DINOv2 conditioning features `z` are required when z_dims is set.")
            z_tensor = z[0] if isinstance(z, (list, tuple)) else z
            cond = self.mlp_ca(self.layer_norm(z_tensor))

        feat_rope = self._resolve_rope(gh, gw, tokens.device)
        use_ckpt = self.use_checkpoint and self.training and torch.is_grad_enabled()

        for block in self.blocks:
            if use_ckpt:
                tokens = checkpoint(block, tokens, c0, cond, feat_rope, use_reentrant=False)
            else:
                tokens = block(tokens, c0, cond, feat_rope)

        out = self.final_layer(tokens, c)
        return self.unpatchify(out, gh, gw)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        return self.forward_dit(x, t, z=z)
