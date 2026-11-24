"""Independent DINOv2 vision encoder architecture for VOSR conditioning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import strip_state_dict_prefix

__all__ = ["DINOv2Encoder"]

_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
_TORCH_WEIGHT_EXTENSIONS: frozenset[str] = frozenset({".pth", ".pt", ".bin", ".ckpt"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _load_dinov2_backbone(
    *,
    model_name: str = "dinov2_vitl14",
    cache_dir: str | Path | None = None,
    local_repo: str | Path | None = None,
    pretrained: bool = True,
) -> nn.Module:
    """Load DINOv2 vision backbone from a local torch.hub repository or GitHub."""
    if cache_dir is not None:
        torch.hub.set_dir(str(cache_dir))

    if local_repo is not None:
        encoder = torch.hub.load(
            str(local_repo),
            model_name,
            source="local",
            pretrained=pretrained,
        )
    else:
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            trust_repo=True,
            pretrained=pretrained,
        )

    if hasattr(encoder, "head"):
        encoder.head = nn.Identity()

    return encoder


def _read_state_dict_file(path: Path, key: str | None = None) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        if key is not None:
            raise KeyError(
                f"Key {key!r} is not supported for flat safetensors file {path}; pass key=None."
            )
        raw_sd: Mapping[str, torch.Tensor] = load_file(str(path), device="cpu")
    elif suffix in _TORCH_WEIGHT_EXTENSIONS:
        raw_obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw_obj, Mapping):
            raise ValueError(
                f"Expected mapping checkpoint in {path}, got {type(raw_obj).__name__}."
            )
        if key is not None:
            if key not in raw_obj:
                raise KeyError(
                    f"Key {key!r} not found in {path}. "
                    f"Available keys: {sorted(map(str, raw_obj.keys()))}"
                )
            candidate = raw_obj[key]
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"Expected mapping under key {key!r} in {path}, got {type(candidate).__name__}."
                )
            raw_sd = candidate
        else:
            if any(not isinstance(v, torch.Tensor) for v in raw_obj.values()):
                raise ValueError(
                    f"Checkpoint {path} is nested "
                    f"(top-level keys: {sorted(map(str, raw_obj.keys()))}); "
                    "specify `key`."
                )
            raw_sd = raw_obj
    else:
        raise ValueError(
            f"Unsupported DINOv2 weight file format {suffix!r} for {path}; "
            "expected '.safetensors', '.pth', or '.pt'."
        )

    return strip_state_dict_prefix(raw_sd)


class DINOv2Encoder(nn.Module):
    """Independent DINOv2 vision encoder producing patch tokens for VOSR cross-attention.

    Supports both frozen feature extraction and fine-tuning (automatically freezing unused
    post-``layer`` blocks and ``mask_token`` via :meth:`freeze_unused_parameters` so DDP
    stays free of unused-parameter errors while preserving full checkpoint key parity).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        image_size: int = 448,
        layer: int = 17,
        cache_dir: str | Path | None = None,
        local_repo: str | Path | None = None,
        pretrained_path: str | Path | None = None,
        load_hub_weights: bool = True,
        pretrained_strict: bool = True,
        pretrained_key: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(f"model_name must be a non-empty string, got {model_name!r}.")
        if not _is_positive_int(image_size) or image_size % 14 != 0:
            raise ValueError(
                f"image_size must be a positive multiple of 14 (DINOv2 patch size), got {image_size!r}."
            )
        if not _is_non_negative_int(layer):
            raise ValueError(f"layer must be a non-negative integer, got {layer!r}.")
        if not isinstance(load_hub_weights, bool):
            raise ValueError(
                f"load_hub_weights must be a bool, got {type(load_hub_weights).__name__}: {load_hub_weights!r}."
            )
        if not isinstance(pretrained_strict, bool):
            raise ValueError(
                f"pretrained_strict must be a bool, got {type(pretrained_strict).__name__}: {pretrained_strict!r}."
            )

        self.model_name: str = model_name.strip()
        self.image_size: int = int(image_size)
        self.layer: int = int(layer)

        use_hub_weights = bool(load_hub_weights and pretrained_path is None)
        encoder = _load_dinov2_backbone(
            model_name=self.model_name,
            cache_dir=cache_dir,
            local_repo=local_repo,
            pretrained=use_hub_weights,
        )
        if not isinstance(encoder, nn.Module):
            raise TypeError(
                f"Loaded DINOv2 backbone must be an nn.Module, got {type(encoder).__name__}."
            )
        if not hasattr(encoder, "blocks") or not hasattr(encoder, "prepare_tokens_with_masks"):
            raise TypeError(
                "Loaded DINOv2 backbone must expose 'blocks' and 'prepare_tokens_with_masks'."
            )
        num_blocks = len(encoder.blocks)
        if not (0 <= self.layer < num_blocks):
            raise ValueError(
                f"layer={self.layer} is out of range for DINOv2 {self.model_name!r} "
                f"with {num_blocks} blocks."
            )

        self.encoder: nn.Module = encoder
        # Non-persistent normalization constants keep state_dict() strictly weights-only.
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if pretrained_path is not None:
            self.load_pretrained(
                pretrained_path,
                strict=pretrained_strict,
                key=pretrained_key,
            )

    @property
    def embed_dim(self) -> int | None:
        """Return token embedding width when exposed by the underlying DINOv2 backbone."""
        dim = getattr(self.encoder, "embed_dim", None)
        if isinstance(dim, int):
            return dim
        dim = getattr(self.encoder, "num_features", None)
        return dim if isinstance(dim, int) else None

    def load_pretrained(
        self,
        checkpoint_path: str | Path,
        *,
        strict: bool = True,
        key: str | None = None,
    ) -> tuple[list[str], list[str]]:
        """Load either SRTrainer-exported ``DINOv2Encoder`` weights or raw DINOv2 ``.pth`` weights."""
        cleaned = _read_state_dict_file(Path(checkpoint_path), key=key)
        self_keys = set(self.state_dict().keys())
        enc_keys = set(self.encoder.state_dict().keys())
        loaded_keys = set(cleaned.keys())

        overlap_self = len(loaded_keys & self_keys)
        overlap_enc = len(loaded_keys & enc_keys)

        if overlap_self > 0 and overlap_self >= overlap_enc:
            incompatible = self.load_state_dict(cleaned, strict=strict)
        elif overlap_enc > 0:
            incompatible = self.encoder.load_state_dict(cleaned, strict=strict)
        else:
            incompatible = self.load_state_dict(cleaned, strict=strict)

        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    def freeze_unused_parameters(self) -> None:
        """Freeze parameters bypassed by ``forward`` (post-``layer`` blocks, unused norm/mask_token)."""
        num_blocks = len(self.encoder.blocks)
        for idx in range(self.layer + 1, num_blocks):
            self.encoder.blocks[idx].requires_grad_(False)
        if self.layer < num_blocks - 1 and hasattr(self.encoder, "norm"):
            norm_mod = self.encoder.norm
            if isinstance(norm_mod, nn.Module):
                norm_mod.requires_grad_(False)
        mask_token = getattr(self.encoder, "mask_token", None)
        if isinstance(mask_token, nn.Parameter):
            mask_token.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate patch tokens ``[B, N_patches, D]`` from RGB ``[B, 3, H, W]`` in ``[0, 1]``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected 4D RGB input (B, 3, H, W), got shape {tuple(x.shape)}.")
        if x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError(
                f"Expected positive spatial dimensions (H >= 1, W >= 1), got shape {tuple(x.shape)}."
            )

        resized = F.interpolate(
            x.float(),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        mean = self._imagenet_mean.to(device=resized.device, dtype=resized.dtype)
        std = self._imagenet_std.to(device=resized.device, dtype=resized.dtype)
        normed = (resized - mean) / std

        tokens = self.encoder.prepare_tokens_with_masks(normed)
        num_blocks = len(self.encoder.blocks)
        for idx, block in enumerate(self.encoder.blocks):
            tokens = block(tokens)
            if idx == self.layer:
                out = self.encoder.norm(tokens) if idx == num_blocks - 1 else tokens
                return out[:, 1:]

        raise RuntimeError(f"Requested DINOv2 layer {self.layer} was not produced.")
