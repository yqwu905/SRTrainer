"""Pure-PyTorch TextVerifier Stage-3 vision tower (ViT-L/16 @ 128px + ImageProjector).

Exact 298-key state_dict parity with the image branch of ``ATH-MaaS/TextVerifier-CJK-CLIP``
(``textverifier_stage3.pth``):
- ``image_encoder.backbone.*``: 294 tensors (ViT-L/16, 128x128 input, 64 patch tokens + 1 CLS
  token, 24 pre-norm blocks with ``LayerNorm(eps=1e-6)``, patch-mean pooling before ``fc_norm``).
- ``image_projector.*``: 4 tensors (``Linear(1024, 1024, bias=False) -> LayerNorm(1024, eps=1e-5)
  -> GELU -> Linear(1024, 1024, bias=False)``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TEXTVERIFIER_IMAGE_SIZE",
    "TEXTVERIFIER_PATCH_SIZE",
    "TEXTVERIFIER_EMBED_DIM",
    "TEXTVERIFIER_DEPTH",
    "TEXTVERIFIER_NUM_HEADS",
    "TEXTVERIFIER_MLP_RATIO",
    "TEXTVERIFIER_BACKBONE_NORM_EPS",
    "TEXTVERIFIER_PROJECTOR_NORM_EPS",
    "TEXTVERIFIER_VISION_NUM_KEYS",
    "PatchEmbed",
    "Attention",
    "Mlp",
    "Block",
    "TextVerifierViTBackbone",
    "TextVerifierImageEncoder",
    "ImageProjector",
    "TextVerifierVision",
]

TEXTVERIFIER_IMAGE_SIZE: int = 128
TEXTVERIFIER_PATCH_SIZE: int = 16
TEXTVERIFIER_EMBED_DIM: int = 1024
TEXTVERIFIER_DEPTH: int = 24
TEXTVERIFIER_NUM_HEADS: int = 16
TEXTVERIFIER_MLP_RATIO: float = 4.0
TEXTVERIFIER_BACKBONE_NORM_EPS: float = 1e-6
TEXTVERIFIER_PROJECTOR_NORM_EPS: float = 1e-5
TEXTVERIFIER_VISION_NUM_KEYS: int = 298


class PatchEmbed(nn.Module):
    """2D patch embedding matching ViT-L/16 (``patch_embed.proj``)."""

    def __init__(
        self,
        img_size: int = TEXTVERIFIER_IMAGE_SIZE,
        patch_size: int = TEXTVERIFIER_PATCH_SIZE,
        in_chans: int = 3,
        embed_dim: int = TEXTVERIFIER_EMBED_DIM,
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
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention matching ViT-L/16 (``attn.qkv`` and ``attn.proj``)."""

    def __init__(
        self,
        dim: int = TEXTVERIFIER_EMBED_DIM,
        num_heads: int = TEXTVERIFIER_NUM_HEADS,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # SDPA applies head_dim**-0.5 scaling internally; do not pre-scale q.
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class Mlp(nn.Module):
    """Feed-forward MLP matching ViT-L/16 (``mlp.fc1``, exact GELU, ``mlp.fc2``)."""

    def __init__(
        self,
        in_features: int = TEXTVERIFIER_EMBED_DIM,
        hidden_features: int = int(TEXTVERIFIER_EMBED_DIM * TEXTVERIFIER_MLP_RATIO),
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm ViT transformer block with ``LayerNorm(eps=1e-6)``."""

    def __init__(
        self,
        dim: int = TEXTVERIFIER_EMBED_DIM,
        num_heads: int = TEXTVERIFIER_NUM_HEADS,
        mlp_ratio: float = TEXTVERIFIER_MLP_RATIO,
        norm_eps: float = TEXTVERIFIER_BACKBONE_NORM_EPS,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TextVerifierViTBackbone(nn.Module):
    """Fixed ViT-L/16 @ 128x128 backbone with patch-average pooling before ``fc_norm``."""

    def __init__(self) -> None:
        super().__init__()
        self.img_size = TEXTVERIFIER_IMAGE_SIZE
        self.patch_size = TEXTVERIFIER_PATCH_SIZE
        self.embed_dim = TEXTVERIFIER_EMBED_DIM
        self.depth = TEXTVERIFIER_DEPTH
        self.num_heads = TEXTVERIFIER_NUM_HEADS

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=3,
            embed_dim=self.embed_dim,
        )
        num_tokens = self.patch_embed.num_patches + 1  # 64 patches + 1 CLS token = 65
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=TEXTVERIFIER_MLP_RATIO,
                    norm_eps=TEXTVERIFIER_BACKBONE_NORM_EPS,
                )
                for _ in range(self.depth)
            ]
        )
        self.fc_norm = nn.LayerNorm(self.embed_dim, eps=TEXTVERIFIER_BACKBONE_NORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, self.img_size, self.img_size):
            raise ValueError(
                f"TextVerifierViTBackbone expects (B, 3, {self.img_size}, {self.img_size}) RGB input, "
                f"got {tuple(x.shape)}."
            )
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        # Pool the 64 patch tokens first (excluding CLS), then apply fc_norm.
        pooled = tokens[:, 1:].mean(dim=1)
        return self.fc_norm(pooled)


class TextVerifierImageEncoder(nn.Module):
    """Wrapper preserving the official ``image_encoder.backbone.*`` state_dict hierarchy."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = TextVerifierViTBackbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ImageProjector(nn.Module):
    """Stage-3 image projection head (``Linear -> LayerNorm(1e-5) -> GELU -> Linear``)."""

    def __init__(
        self,
        dim: int = TEXTVERIFIER_EMBED_DIM,
        norm_eps: float = TEXTVERIFIER_PROJECTOR_NORM_EPS,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim, eps=norm_eps)
        self.act = nn.GELU(approximate="none")
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.ln(self.fc1(x))))


class TextVerifierVision(nn.Module):
    """Fixed TextVerifier Stage-3 vision encoder matching the official 298 image keys.

    - ``forward(images)`` returns unnormalized 1024-d projector embeddings.
    - ``encode_image(images)`` returns L2-normalized FP32 1024-d embeddings matching official
      ``TextVerifier.encode_image``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = TextVerifierImageEncoder()
        self.image_projector = ImageProjector(
            dim=TEXTVERIFIER_EMBED_DIM,
            norm_eps=TEXTVERIFIER_PROJECTOR_NORM_EPS,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_projector(self.image_encoder(images))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward(images).float(), dim=-1)
