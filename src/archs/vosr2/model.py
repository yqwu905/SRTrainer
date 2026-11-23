"""VOSR2 single-step end-to-end super-resolution architecture for SRTrainer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from archs.vosr2.dit import VOSR2DiT
from archs.vosr2.qwen_vae2d import AutoencoderKLQwenImage2D
from utils.misc import pad_to_multiple, strip_state_dict_prefix, unpad_tensor

__all__ = ["VOSR2OneStep"]

_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
_VALID_NOISE_INPUTS: frozenset[str] = frozenset({"gaussian", "zero"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _load_dinov2(
    *,
    cache_dir: str | Path | None = None,
    local_repo: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> nn.Module:
    """Load frozen real DINOv2-ViT-L/14 vision encoder."""
    if cache_dir is not None:
        torch.hub.set_dir(str(cache_dir))

    use_hub_weights = checkpoint_path is None
    if local_repo is not None:
        encoder = torch.hub.load(
            str(local_repo),
            "dinov2_vitl14",
            source="local",
            pretrained=use_hub_weights,
        )
    else:
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14",
            trust_repo=True,
            pretrained=use_hub_weights,
        )

    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint file not found: {ckpt_file}")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state, strict=True)

    if hasattr(encoder, "head"):
        encoder.head = nn.Identity()

    return encoder.eval().requires_grad_(False)


class _VOSR2FrozenBackends:
    """Plain Python container holding frozen Qwen-Image 2D VAE and DINOv2 backends.

    Intentionally NOT an ``nn.Module`` submodule of ``VOSR2OneStep`` so that:
    - ``state_dict()`` exposes ONLY the 849 official VOSR2 DiT keys (strict checkpoint parity).
    - ``parameters()`` exposes ONLY trainable DiT parameters (clean optimizer / DDP / grad clip).
    - ``ModelEMA`` shares the immutable frozen backends by reference without duplicating VRAM.
    """

    def __init__(self, vae: AutoencoderKLQwenImage2D, dino: nn.Module | None) -> None:
        self.vae: AutoencoderKLQwenImage2D = vae.eval().requires_grad_(False)
        self.dino: nn.Module | None = dino.eval().requires_grad_(False) if dino is not None else None
        self.device: torch.device = torch.device("cpu")

    def ensure_device(self, device: torch.device) -> None:
        target = torch.device(device)
        if self.device == target:
            return
        self.vae.to(device=target)
        if self.dino is not None:
            self.dino.to(device=target)
        self.device = target

    def lock_eval(self) -> None:
        self.vae.eval()
        if self.dino is not None:
            self.dino.eval()

    @torch.no_grad()
    def encode_latent(self, lq_m11: torch.Tensor, posterior_mode: bool = True) -> torch.Tensor:
        with torch.amp.autocast(device_type=lq_m11.device.type, enabled=False):
            return self.vae.encode_normalized(lq_m11.float(), posterior_mode=posterior_mode)

    def decode_latent(self, sr_latent: torch.Tensor, clamp_output: bool = True) -> torch.Tensor:
        # Intentionally allows autograd through frozen decoder weights into sr_latent -> DiT.
        return self.vae.decode_normalized(sr_latent, clamp_output=clamp_output)

    @torch.no_grad()
    def extract_dino(
        self,
        lq_01: torch.Tensor,
        dino_size: int,
        dino_layer: int,
    ) -> list[torch.Tensor]:
        if self.dino is None:
            raise RuntimeError("DINOv2 backend is not initialized (z_dims is None).")
        x = F.interpolate(
            lq_01.float(),
            size=(dino_size, dino_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean) / std

        tokens = self.dino.prepare_tokens_with_masks(x)
        num_blocks = len(self.dino.blocks)
        if not (0 <= dino_layer < num_blocks):
            raise ValueError(
                f"dino_layer={dino_layer} is out of range for DINOv2 with {num_blocks} blocks."
            )
        for idx, block in enumerate(self.dino.blocks):
            tokens = block(tokens)
            if idx == dino_layer:
                out = self.dino.norm(tokens) if idx == num_blocks - 1 else tokens
                return [out[:, 1:]]
        raise RuntimeError(f"Requested DINOv2 layer {dino_layer} was not produced.")


def _resolve_dit_checkpoint_file(
    checkpoint_path_or_repo: str | Path,
    filename: str = "VOSR2/checkpoints/ema_model.safetensors",
) -> Path:
    path = Path(checkpoint_path_or_repo)
    if path.is_file():
        return path
    if path.is_dir():
        for sub in ("checkpoints", "clean_weights", ""):
            for name in ("ema_model.safetensors", "model.safetensors"):
                candidate = path / sub / name if sub else path / name
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"No VOSR2 .safetensors checkpoint found under directory: {path}")
    repo_str = str(checkpoint_path_or_repo)
    if path.is_absolute() or repo_str.count("/") != 1:
        raise FileNotFoundError(f"VOSR2 checkpoint path not found: {checkpoint_path_or_repo}")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_str, filename=filename))


class VOSR2OneStep(VOSR2DiT):
    """End-to-end VOSR2 one-step SR network adapted for SRTrainer's ``[0, 1]`` RGB interface.

    Pipeline per ``forward(lq)`` (``lq`` shape ``[B, 3, H, W]`` in ``[0, 1]``):
    1. Bicubic upsample ``lq`` by ``scale`` to ``(H * scale, W * scale)`` and pad to a multiple
       of ``vae_factor * patch_size`` if needed.
    2. Frozen Qwen-Image 2D VAE encodes ``lq`` (FP32, ``no_grad``) to 16-ch normalized latents.
    3. Frozen DINOv2-ViT-L/14 extracts layer-17 patch tokens (``no_grad``) for cross-attention.
    4. Trainable VOSR2 ``LightningDiT`` predicts one-step velocity at ``t = 1`` from
       ``cat([lq_latent, noise], dim=1)``; ``sr_latent = noise - velocity``.
    5. Frozen Qwen-Image 2D VAE decodes ``sr_latent`` WITH autograd enabled so pixel-space
       losses (L1, LPIPS, OCR-KL, Glyph) backpropagate cleanly into all trainable DiT weights.
    6. Returns RGB tensor ``[B, 3, H * scale, W * scale]`` in ``[0, 1]``.

    ``state_dict()`` and ``parameters()`` expose strictly the VOSR2 DiT weights (exact 849-key
    parity with official ``CSWRY/VOSR/VOSR2/checkpoints/ema_model.safetensors``).
    """

    def __init__(
        self,
        scale: int = 1,
        # VOSR2 DiT backbone defaults (exact VOSR2/args.json 1.4B configuration)
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
        # One-step sampling & decode options
        noise_input: str = "gaussian",
        posterior_mode: bool = True,
        clamp_decode_in_train: bool = True,
        eval_seed: int | None = 42,
        # Frozen Qwen-Image 2D VAE options
        vae_path: str | Path = "CSWRY/VOSR",
        vae_subfolder: str | None = "Qwen-Image-vae-2d",
        vae_gradient_checkpointing: bool = False,
        # Frozen DINOv2-ViT-L/14 options
        dino_size: int = 448,
        dino_layer: int = 17,
        dino_cache_dir: str | Path | None = None,
        dino_local_repo: str | Path | None = None,
        dino_checkpoint: str | Path | None = None,
        # Optional direct DiT checkpoint loading (in addition to model.pretrain_path)
        dit_checkpoint: str | Path | None = None,
        dit_filename: str = "VOSR2/checkpoints/ema_model.safetensors",
    ) -> None:
        if not _is_positive_int(scale):
            raise ValueError(f"scale must be a positive integer, got {scale!r}.")
        if in_channels != 2 * out_channels:
            raise ValueError(
                f"VOSR2OneStep requires in_channels == 2 * out_channels "
                f"(lq_latent + noise), got in_channels={in_channels}, out_channels={out_channels}."
            )
        if noise_input not in _VALID_NOISE_INPUTS:
            raise ValueError(
                f"Unsupported noise_input={noise_input!r}; expected one of {sorted(_VALID_NOISE_INPUTS)}."
            )
        if not _is_positive_int(dino_size) or dino_size % 14 != 0:
            raise ValueError(
                f"dino_size must be a positive multiple of 14 (DINOv2 patch size), got {dino_size!r}."
            )
        if not _is_non_negative_int(dino_layer):
            raise ValueError(f"dino_layer must be a non-negative integer, got {dino_layer!r}.")
        if eval_seed is not None and (isinstance(eval_seed, bool) or not isinstance(eval_seed, int)):
            raise ValueError(f"eval_seed must be an int or None, got {eval_seed!r}.")

        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            z_dims=z_dims,
            encdim_ratio=encdim_ratio,
            use_qknorm=use_qknorm,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            use_rmsnorm=use_rmsnorm,
            wo_shift=wo_shift,
            use_checkpoint=use_checkpoint,
            rope_mode=rope_mode,
        )

        self.scale = scale
        self.noise_input = noise_input
        self.posterior_mode = bool(posterior_mode)
        self.clamp_decode_in_train = bool(clamp_decode_in_train)
        self.eval_seed = eval_seed
        self.dino_size = dino_size
        self.dino_layer = dino_layer

        if vae_path is None or (isinstance(vae_path, str) and not vae_path.strip()):
            raise ValueError("vae_path must be a non-empty local path or HuggingFace repo id.")
        vae = AutoencoderKLQwenImage2D.from_pretrained(
            vae_path,
            subfolder=vae_subfolder,
            gradient_checkpointing=vae_gradient_checkpointing,
        )

        if vae.z_dim != out_channels:
            raise ValueError(
                f"Loaded Qwen VAE z_dim ({vae.z_dim}) does not match VOSR2 out_channels ({out_channels})."
            )

        dino: nn.Module | None = None
        if z_dims is not None:
            dino = _load_dinov2(
                cache_dir=dino_cache_dir,
                local_repo=dino_local_repo,
                checkpoint_path=dino_checkpoint,
            )

        self._backends = _VOSR2FrozenBackends(vae=vae, dino=dino)
        self._spatial_quantum: int = vae.spatial_compression_ratio * self.patch_size

        if dit_checkpoint is not None:
            self.load_dit_weights(dit_checkpoint, filename=dit_filename, strict=True)

    @property
    def vae(self) -> AutoencoderKLQwenImage2D:
        """Read-only access to the frozen Qwen-Image 2D VAE backend."""
        return self._backends.vae

    @property
    def dino(self) -> nn.Module | None:
        """Read-only access to the frozen DINOv2 vision encoder backend."""
        return self._backends.dino

    def load_dit_weights(
        self,
        checkpoint_path_or_repo: str | Path,
        *,
        filename: str = "VOSR2/checkpoints/ema_model.safetensors",
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load VOSR2 DiT weights from a local file/dir or HuggingFace repository."""
        ckpt_file = _resolve_dit_checkpoint_file(checkpoint_path_or_repo, filename=filename)
        state = load_file(str(ckpt_file), device="cpu")
        cleaned = strip_state_dict_prefix(state)
        incompatible = self.load_state_dict(cleaned, strict=strict)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    def __deepcopy__(self, memo: dict[int, Any]) -> VOSR2OneStep:
        # Share immutable frozen VAE & DINOv2 across EMA shadow copies to save ~1.5 GB VRAM.
        memo[id(self._backends)] = self._backends
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def train(self, mode: bool = True) -> VOSR2OneStep:
        super().train(mode)
        self._backends.lock_eval()
        return self

    def to(self, *args: Any, **kwargs: Any) -> VOSR2OneStep:
        device, _dtype, _non_blocking, _convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        out = super().to(*args, **kwargs)
        if device is not None:
            self._backends.ensure_device(device)
        return out

    def cuda(self, device: int | torch.device | None = None) -> VOSR2OneStep:
        out = super().cuda(device)
        target = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        self._backends.ensure_device(target)
        return out

    def cpu(self) -> VOSR2OneStep:
        out = super().cpu()
        self._backends.ensure_device(torch.device("cpu"))
        return out

    def _sample_noise(self, lq_latent: torch.Tensor) -> torch.Tensor:
        if self.noise_input == "zero":
            return torch.zeros_like(lq_latent)
        if not self.training and self.eval_seed is not None:
            gen = torch.Generator(device=lq_latent.device)
            gen.manual_seed(self.eval_seed)
            return torch.randn(
                lq_latent.shape,
                generator=gen,
                device=lq_latent.device,
                dtype=lq_latent.dtype,
            )
        return torch.randn_like(lq_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run one-step VOSR2 SR on RGB ``[B, 3, H, W]`` in ``[0, 1]`` -> ``[B, 3, H*s, W*s]``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, 3, H, W), got shape {tuple(x.shape)}.")
        if x.shape[1] != 3:
            raise ValueError(f"VOSR2OneStep expects 3-channel RGB inputs, got {x.shape[1]} channels.")
        if x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError(
                f"Expected positive spatial dimensions (H >= 1, W >= 1), got shape {tuple(x.shape)}."
            )

        self._backends.ensure_device(x.device)
        self._backends.lock_eval()

        b, _, h_lq, w_lq = x.shape
        target_h = h_lq * self.scale
        target_w = w_lq * self.scale

        if self.scale > 1:
            lq_up = F.interpolate(
                x.float(),
                size=(target_h, target_w),
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)
        else:
            lq_up = x.float().clamp(0.0, 1.0)

        lq_pad, (orig_h, orig_w) = pad_to_multiple(lq_up, self._spatial_quantum, mode="reflect")
        lq_m11 = lq_pad * 2.0 - 1.0

        with torch.no_grad():
            lq_latent = self._backends.encode_latent(lq_m11, posterior_mode=self.posterior_mode)
            z_fea = (
                self._backends.extract_dino(
                    lq_pad,
                    dino_size=self.dino_size,
                    dino_layer=self.dino_layer,
                )
                if self.z_dims is not None
                else None
            )

        noise = self._sample_noise(lq_latent)
        dit_inp = torch.cat((lq_latent, noise), dim=1)
        t = torch.ones((b,), device=lq_latent.device, dtype=lq_latent.dtype)

        velocity = self.forward_dit(dit_inp, t, z=z_fea)
        sr_latent = noise - velocity.float()

        clamp_decode = (not self.training) or self.clamp_decode_in_train
        sr_m11 = self._backends.decode_latent(sr_latent, clamp_output=clamp_decode)
        sr_01 = (sr_m11 + 1.0) * 0.5
        if sr_01.shape[-2] != orig_h or sr_01.shape[-1] != orig_w:
            sr_01 = unpad_tensor(sr_01, orig_h, orig_w, scale=1)
        return sr_01
