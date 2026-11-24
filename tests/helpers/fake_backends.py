"""Test-only fakes and loader monkeypatches keeping ``src/`` 100% production-pure."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lpips
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.textverifier import TEXTVERIFIER_EMBED_DIM
from archs.vosr2.qwen_vae2d import AutoencoderKLQwenImage2D
from losses.textverifier import TextVerifierAssets, TextVerifierFiles

TEST_GLYPH_CHARS: tuple[str, ...] = (
    "中",
    "文",
    "春",
    "夏",
    "秋",
    "冬",
    "山",
    "河",
    "日",
    "月",
    "A",
    "B",
    "C",
    "0",
    "1",
    "2",
)


class FakeTextVerifierEncoder(nn.Module):
    """Lightweight deterministic encoder matching TextVerifierVision input/output contract."""

    def __init__(self, embed_dim: int = TEXTVERIFIER_EMBED_DIM, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        conv = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4, bias=False)
        with torch.no_grad():
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=gen) * 0.1)
        self.net = nn.Sequential(
            conv,
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_fake_textverifier_assets(
    *,
    encoder: nn.Module | None = None,
    prototypes: torch.Tensor | None = None,
    char_to_index: Mapping[str, int] | None = None,
    embed_dim: int = TEXTVERIFIER_EMBED_DIM,
    num_prototypes: int = 4,
    seed: int = 0,
) -> TextVerifierAssets:
    """Build a test-only ``TextVerifierAssets`` instance without touching the network or disk."""
    gen = torch.Generator().manual_seed(seed)
    enc = encoder if encoder is not None else FakeTextVerifierEncoder(embed_dim=embed_dim, seed=seed)
    enc = enc.eval().requires_grad_(False)
    mapping = (
        dict(char_to_index)
        if char_to_index is not None
        else {ch: idx for idx, ch in enumerate(TEST_GLYPH_CHARS)}
    )
    if prototypes is None:
        raw_protos = torch.randn((len(mapping), num_prototypes, embed_dim), generator=gen)
    else:
        raw_protos = prototypes
    norm_protos = F.normalize(raw_protos.detach().float(), dim=-1)

    dummy_root = Path("/tmp/test-textverifier-fake")
    files = TextVerifierFiles(
        root_dir=dummy_root,
        stage3_path=dummy_root / "textverifier_stage3.pth",
        prototypes_path=dummy_root / "textverifier_k10_q90.pt",
        metadata_path=dummy_root / "prototype_metadata.json",
        manifest_path=dummy_root / "manifest.sha256",
        source="test:fake",
    )
    metadata = {
        "schema_version": 1,
        "image_size": 128,
        "embedding_dim": int(norm_protos.shape[-1]),
        "num_characters": len(mapping),
        "K": int(norm_protos.shape[1]),
        "prototype_shape": list(norm_protos.shape),
        "threshold": 0.02152947708964348,
    }
    return TextVerifierAssets(
        encoder=enc,  # type: ignore[arg-type]
        prototypes=norm_protos,
        char_to_index=mapping,
        metadata=metadata,
        files=files,
        sha256_digests={},
        threshold=float(metadata["threshold"]),
    )


def patch_textverifier_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    assets: TextVerifierAssets | None = None,
    encoder: nn.Module | None = None,
    prototypes: torch.Tensor | None = None,
    char_to_index: Mapping[str, int] | None = None,
    embed_dim: int = TEXTVERIFIER_EMBED_DIM,
    num_prototypes: int = 4,
    seed: int = 0,
) -> TextVerifierAssets:
    """Monkeypatch ``losses.glyph_loss.load_textverifier_assets`` for offline unit tests."""
    resolved = assets or make_fake_textverifier_assets(
        encoder=encoder,
        prototypes=prototypes,
        char_to_index=char_to_index,
        embed_dim=embed_dim,
        num_prototypes=num_prototypes,
        seed=seed,
    )
    monkeypatch.setattr(
        "losses.glyph_loss.load_textverifier_assets",
        lambda **_kwargs: resolved,
    )
    return resolved


class _FakeDinoBlock(nn.Module):
    """Tiny token-mixing block so CLS and patch tokens both receive gradients when trainable."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.fc(x + x.mean(dim=1, keepdim=True))


class FakeDinoV2(nn.Module):
    """Lightweight DINOv2 test double implementing the exact real DINOv2 extraction protocol."""

    def __init__(self, embed_dim: int = 16, patch_size: int = 14, num_blocks: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList([_FakeDinoBlock(embed_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)

    def prepare_tokens_with_masks(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat((cls, patches), dim=1)


def patch_vosr2_backends(
    monkeypatch: pytest.MonkeyPatch,
    *,
    z_dim: int = 4,
    dino_dim: int = 16,
    vae_base_dim: int = 8,
    vae_dim_mult: Sequence[int] = (1, 2, 2, 2),
    vae_num_res_blocks: int = 1,
    dino_num_blocks: int = 2,
) -> None:
    """Monkeypatch Qwen VAE HF loader and DINOv2 torch.hub loader for offline CPU tests."""
    orig_load_pretrained = AutoencoderKLQwenImage2D.load_pretrained

    def _fake_vae_from_pretrained(
        cls: type[AutoencoderKLQwenImage2D],
        pretrained_path_or_repo: str | Path,
        *,
        subfolder: str | None = None,
        gradient_checkpointing: bool = False,
    ) -> AutoencoderKLQwenImage2D:
        del pretrained_path_or_repo, subfolder
        return cls(
            base_dim=vae_base_dim,
            z_dim=z_dim,
            dim_mult=vae_dim_mult,
            num_res_blocks=vae_num_res_blocks,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
            gradient_checkpointing=gradient_checkpointing,
        )

    def _fake_vae_load_pretrained(
        self: AutoencoderKLQwenImage2D,
        pretrained_path_or_repo: str | Path,
        *,
        subfolder: str | None = None,
        strict: bool = True,
        key: str | None = None,
    ) -> tuple[list[str], list[str]]:
        if Path(pretrained_path_or_repo).exists():
            return orig_load_pretrained(
                self,
                pretrained_path_or_repo,
                subfolder=subfolder,
                strict=strict,
                key=key,
            )
        return [], []

    def _fake_load_dinov2_backbone(**_kwargs: Any) -> nn.Module:
        return FakeDinoV2(embed_dim=dino_dim, num_blocks=dino_num_blocks)

    monkeypatch.setattr(
        "archs.vosr2.qwen_vae2d.AutoencoderKLQwenImage2D.from_pretrained",
        classmethod(_fake_vae_from_pretrained),
    )
    monkeypatch.setattr(
        "archs.vosr2.qwen_vae2d.AutoencoderKLQwenImage2D.load_pretrained",
        _fake_vae_load_pretrained,
    )
    monkeypatch.setattr(
        "archs.vosr2.dinov2._load_dinov2_backbone",
        _fake_load_dinov2_backbone,
    )


class TinyRecognizer(nn.Module):
    """Lightweight (N, 3, 16, W) -> (N, W // 4, C) recognizer for fast CPU OCR-KL unit tests."""

    input_height = 16

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, (16, 4), stride=(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).squeeze(2).transpose(1, 2) * 8.0


def patch_ppocr_loader(
    monkeypatch: pytest.MonkeyPatch,
    recognizer: nn.Module | None = None,
) -> nn.Module:
    """Monkeypatch ``losses.ocr_kl_loss.load_ppocr_recognizer`` for offline unit tests."""
    rec = recognizer if recognizer is not None else TinyRecognizer()
    monkeypatch.setattr(
        "losses.ocr_kl_loss.load_ppocr_recognizer",
        lambda _checkpoint=None: rec,
    )
    return rec


def patch_lpips_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ``lpips.LPIPS`` to zero-download random weights ONLY inside offline unit tests."""
    orig_lpips_cls = lpips.LPIPS

    def _offline_lpips_factory(*args: Any, **kwargs: Any) -> nn.Module:
        kwargs["pretrained"] = False
        kwargs["pnet_rand"] = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = orig_lpips_cls(*args, **kwargs)
        if hasattr(model, "lins"):
            with torch.no_grad():
                for param in model.lins.parameters():
                    param.abs_()
        return model

    monkeypatch.setattr(lpips, "LPIPS", _offline_lpips_factory)


class DummyLPIPSLoss(nn.Module):
    """Test-only zero-download perceptual loss target for subprocess CLI E2E tests."""

    def __init__(self, net: str = "alex", version: str = "0.1") -> None:
        super().__init__()
        self.net = net
        self.version = version

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred.float(), target.float())
