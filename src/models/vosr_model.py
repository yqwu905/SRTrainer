"""Multi-architecture VOSR training paradigm (``VOSRModel``) and inference pipeline (``VOSRPipeline``)."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from config.schema import (
    VOSR_COMPONENTS,
    NoiseInput,
    TrainConfig,
    VOSRModelConfig,
    VOSRPretrainConfig,
    VOSRTrainableConfig,
)
from models.base_model import BaseModel
from models.ema import ModelEMA, unwrap_model_for_ema
from utils.misc import load_weights, pad_to_multiple, strip_state_dict_prefix, unpad_tensor

if TYPE_CHECKING:
    from accelerate import Accelerator

__all__ = ["VOSRModel", "VOSRPipeline"]


def _move_to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        return obj.to(device=device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device, non_blocking=non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device, non_blocking=non_blocking) for v in obj)
    return obj


def _module_accepts_kwarg(module: nn.Module, kwarg_name: str) -> bool:
    try:
        sig = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == kwarg_name:
            return True
    return False


def _clone_arch_cfg_skip_pretrained(arch_cfg: DictConfig | Mapping[str, Any]) -> DictConfig:
    """Clone an arch config node and disable its built-in pretrained download before instantiation."""
    container = (
        OmegaConf.to_container(arch_cfg, resolve=True)
        if isinstance(arch_cfg, DictConfig)
        else dict(arch_cfg)
    )
    if not isinstance(container, dict):
        raise TypeError(f"Expected mapping arch config, got {type(container).__name__}.")
    if "pretrained_path" in container:
        container["pretrained_path"] = None
    if "load_hub_weights" in container:
        container["load_hub_weights"] = False
    cloned = OmegaConf.create(container)
    assert isinstance(cloned, DictConfig)
    return cloned


def _load_component_weights(
    module: nn.Module,
    path: Path,
    *,
    strict: bool = True,
    key: str | None = None,
) -> tuple[list[str], list[str]]:
    loader = getattr(module, "load_pretrained", None)
    if callable(loader):
        return loader(path, strict=strict, key=key)
    return load_weights(module, path, strict=strict, key=key)


class VOSRPipeline(nn.Module):
    """Composable end-to-end VOSR pipeline wiring independent ``vae``, ``dit``, and optional ``dino``.

    Exposes standard ``nn.Module`` RGB ``[0, 1] -> [0, 1]`` ``forward(lq)`` semantics for
    ``TiledInferencer`` validation/inference and single-pass DDP training across any subset of
    trainable components.
    """

    def __init__(
        self,
        vae: nn.Module,
        dit: nn.Module,
        dino: nn.Module | None = None,
        *,
        scale: int = 1,
        noise_input: NoiseInput | str = NoiseInput.gaussian,
        posterior_mode: bool = True,
        clamp_decode_in_train: bool = True,
        eval_seed: int | None = 42,
        trainable: VOSRTrainableConfig | Mapping[str, bool] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(vae, nn.Module):
            raise TypeError(f"vae must be an nn.Module, got {type(vae).__name__}.")
        if not isinstance(dit, nn.Module):
            raise TypeError(f"dit must be an nn.Module, got {type(dit).__name__}.")
        if dino is not None and not isinstance(dino, nn.Module):
            raise TypeError(f"dino must be an nn.Module or None, got {type(dino).__name__}.")

        if scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale!r}.")
        self.scale: int = int(scale)
        self.noise_input: str = NoiseInput(noise_input).value
        self.posterior_mode: bool = bool(posterior_mode)
        self.clamp_decode_in_train: bool = bool(clamp_decode_in_train)
        self.eval_seed: int | None = eval_seed

        raw_vae = unwrap_model_for_ema(vae)
        raw_dit = unwrap_model_for_ema(dit)
        raw_dino = unwrap_model_for_ema(dino) if dino is not None else None

        vae_z_dim = getattr(raw_vae, "z_dim", getattr(raw_vae, "latent_channels", None))
        dit_out_ch = getattr(raw_dit, "out_channels", None)
        dit_in_ch = getattr(raw_dit, "in_channels", None)
        dit_z_dims = getattr(raw_dit, "z_dims", None)

        if isinstance(dit_in_ch, int) and isinstance(dit_out_ch, int) and dit_in_ch != 2 * dit_out_ch:
            raise ValueError(
                f"VOSR requires dit.in_channels == 2 * dit.out_channels "
                f"(lq_latent + noise), got in_channels={dit_in_ch}, out_channels={dit_out_ch}."
            )
        if isinstance(vae_z_dim, int) and isinstance(dit_out_ch, int) and vae_z_dim != dit_out_ch:
            raise ValueError(
                f"VAE latent channels ({vae_z_dim}) does not match DiT out_channels ({dit_out_ch})."
            )
        if dit_z_dims is not None and raw_dino is None:
            raise ValueError(
                f"DiT expects conditioning features (z_dims={dit_z_dims}), but arch.dino is None."
            )

        vae_factor = int(getattr(raw_vae, "spatial_compression_ratio", 1))
        dit_patch = int(getattr(raw_dit, "patch_size", 1))
        if vae_factor < 1 or dit_patch < 1:
            raise ValueError(
                f"vae.spatial_compression_ratio ({vae_factor}) and dit.patch_size ({dit_patch}) must be >= 1."
            )
        self._spatial_quantum: int = vae_factor * dit_patch
        self._dit_accepts_z: bool = _module_accepts_kwarg(raw_dit, "z")
        self._dit_uses_z: bool = raw_dino is not None and (
            dit_z_dims is not None or not hasattr(raw_dit, "z_dims")
        )

        self.vae: nn.Module = vae
        self.dit: nn.Module = dit
        self.dino: nn.Module | None = dino
        if isinstance(trainable, VOSRTrainableConfig):
            self.trainable: dict[str, bool] = trainable.as_dict()
        elif trainable is not None:
            self.trainable = {name: bool(trainable.get(name, False)) for name in VOSR_COMPONENTS}
        else:
            self.trainable = {name: False for name in VOSR_COMPONENTS}

    @classmethod
    def from_model_config(
        cls,
        vae: nn.Module,
        dit: nn.Module,
        dino: nn.Module | None,
        model_cfg: VOSRModelConfig,
        *,
        scale: int | None = None,
        trainable: VOSRTrainableConfig | None = None,
    ) -> VOSRPipeline:
        """Build a ``VOSRPipeline`` directly from a typed ``VOSRModelConfig``."""
        return cls(
            vae=vae,
            dit=dit,
            dino=dino,
            scale=model_cfg.scale if scale is None else scale,
            noise_input=model_cfg.noise_input,
            posterior_mode=model_cfg.posterior_mode,
            clamp_decode_in_train=model_cfg.clamp_decode_in_train,
            eval_seed=model_cfg.eval_seed,
            trainable=trainable,
        )

    def iter_components(self) -> Iterator[tuple[str, nn.Module]]:
        """Yield ``(name, module)`` for active components in canonical order ``(vae, dit, dino)``."""
        for name in VOSR_COMPONENTS:
            mod = getattr(self, name)
            if mod is not None:
                yield name, mod

    def get_component(self, name: str) -> nn.Module | None:
        if name not in VOSR_COMPONENTS:
            raise KeyError(f"Unknown VOSR component {name!r}; expected one of {VOSR_COMPONENTS}.")
        return getattr(self, name)

    def train(self, mode: bool = True) -> VOSRPipeline:
        super().train(mode)
        if mode:
            for name, mod in self.iter_components():
                if not self.trainable.get(name, False):
                    mod.eval()
        return self

    def _call_vae_encode(self, lq_m11: torch.Tensor) -> torch.Tensor:
        enc_fn = getattr(self.vae, "encode_normalized", None)
        if callable(enc_fn):
            return enc_fn(lq_m11, posterior_mode=self.posterior_mode)
        enc_alt = getattr(self.vae, "encode", None)
        if callable(enc_alt):
            return enc_alt(lq_m11)
        raise TypeError(
            f"VAE arch {type(self.vae).__name__} must implement 'encode_normalized' or 'encode'."
        )

    def _call_vae_decode(self, sr_latent: torch.Tensor, clamp_output: bool) -> torch.Tensor:
        dec_fn = getattr(self.vae, "decode_normalized", None)
        if callable(dec_fn):
            return dec_fn(sr_latent, clamp_output=clamp_output)
        dec_alt = getattr(self.vae, "decode", None)
        if callable(dec_alt):
            out = dec_alt(sr_latent)
            return out.clamp(-1.0, 1.0) if clamp_output else out
        raise TypeError(
            f"VAE arch {type(self.vae).__name__} must implement 'decode_normalized' or 'decode'."
        )

    def _encode_lq(self, lq_m11: torch.Tensor) -> torch.Tensor:
        if self.training and self.trainable.get("vae", False):
            return self._call_vae_encode(lq_m11)
        with torch.no_grad():
            with torch.amp.autocast(device_type=lq_m11.device.type, enabled=False):
                return self._call_vae_encode(lq_m11.float())

    def _extract_dino(self, lq_pad: torch.Tensor) -> torch.Tensor | None:
        if self.dino is None or not self._dit_uses_z:
            return None
        if self.training and self.trainable.get("dino", False):
            return self.dino(lq_pad)
        with torch.no_grad():
            return self.dino(lq_pad)

    def _sample_noise(self, lq_latent: torch.Tensor) -> torch.Tensor:
        if self.noise_input == "zero":
            return torch.zeros_like(lq_latent)
        if not self.training and self.eval_seed is not None:
            # Generate eval noise on CPU for cross-device (CPU/CUDA/MPS/NPU) determinism.
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.eval_seed)
            return torch.randn(
                lq_latent.shape,
                generator=gen,
                device="cpu",
                dtype=torch.float32,
            ).to(device=lq_latent.device, dtype=lq_latent.dtype)
        return torch.randn_like(lq_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one-step VOSR SR on RGB ``[B, 3, H, W]`` in ``[0, 1]`` -> ``[B, 3, H*s, W*s]``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B, 3, H, W), got shape {tuple(x.shape)}.")
        if x.shape[1] != 3:
            raise ValueError(f"VOSRPipeline expects 3-channel RGB inputs, got {x.shape[1]} channels.")
        if x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError(
                f"Expected positive spatial dimensions (H >= 1, W >= 1), got shape {tuple(x.shape)}."
            )

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

        lq_latent = self._encode_lq(lq_m11)
        z_fea = self._extract_dino(lq_pad)

        noise = self._sample_noise(lq_latent)
        dit_inp = torch.cat((lq_latent, noise), dim=1)
        t = torch.ones((b,), device=lq_latent.device, dtype=lq_latent.dtype)

        if self._dit_accepts_z:
            velocity = self.dit(dit_inp, t, z=z_fea)
        else:
            velocity = self.dit(dit_inp, t)

        sr_latent = noise - velocity.float()
        clamp_decode = (not self.training) or self.clamp_decode_in_train
        sr_m11 = self._call_vae_decode(sr_latent, clamp_output=clamp_decode)
        sr_01 = (sr_m11 + 1.0) * 0.5
        if sr_01.shape[-2] != orig_h or sr_01.shape[-1] != orig_w:
            sr_01 = unpad_tensor(sr_01, orig_h, orig_w, scale=1)
        return sr_01


class VOSRModel(BaseModel):
    """Composable multi-arch VOSR training paradigm receiving typed ``TrainConfig`` directly."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        m: VOSRModelConfig = cfg.model

        self.model_cfg: VOSRModelConfig = m
        self.scale: int = m.scale
        self.trainable: dict[str, bool] = m.trainable.as_dict()
        self.noise_input: str = m.noise_input.value
        self.posterior_mode: bool = m.posterior_mode
        self.clamp_decode_in_train: bool = m.clamp_decode_in_train
        self.eval_seed: int | None = m.eval_seed
        self.ema_enabled: bool = m.ema.enabled
        self.ema_decay: float = m.ema.decay
        self.ema_warmup_steps: int = m.ema.warmup_steps
        self.save_frozen_weights: bool = m.save_frozen_weights
        self.pretrain: VOSRPretrainConfig = m.pretrain

        self.pipeline: nn.Module | None = None
        self.emas: dict[str, ModelEMA] = {}
        self.losses: dict[str, tuple[float, Callable[..., torch.Tensor]]] = {}
        self.degrader: nn.Module | Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: Any | None = None
        self.grad_clip_norm: float | None = None
        self._active_components: tuple[str, ...] = ()

        self._is_setup: bool = False
        self._accum_steps: int = 0
        self._accum_total: float = 0.0
        self._accum_losses: dict[str, float] = {}
        self._capture_visuals_max: int = 0
        self._captured_lq: list[torch.Tensor] = []
        self._captured_gt: list[torch.Tensor] = []
        self._captured_pred: list[torch.Tensor] = []
        self._captured_count: int = 0

    def _require_setup(self) -> Accelerator:
        if (
            not self._is_setup
            or self.accelerator is None
            or self.pipeline is None
            or self.optimizer is None
        ):
            raise RuntimeError("VOSRModel.setup(accelerator) must be called before using the model.")
        return self.accelerator

    @property
    def active_components(self) -> tuple[str, ...]:
        return self._active_components

    @property
    def trainable_components(self) -> tuple[str, ...]:
        return tuple(name for name in self._active_components if self.trainable.get(name, False))

    def get_component(self, name: str, *, use_ema: bool = False) -> nn.Module | None:
        """Return an unwrapped component arch (``vae`` / ``dit`` / ``dino``), optionally EMA."""
        self._require_setup()
        assert self.pipeline is not None
        if name not in VOSR_COMPONENTS:
            raise KeyError(f"Unknown VOSR component {name!r}; expected one of {VOSR_COMPONENTS}.")
        if use_ema and name in self.emas:
            return self.emas[name].module
        raw_pipe = unwrap_model_for_ema(self.pipeline)
        assert isinstance(raw_pipe, VOSRPipeline)
        comp = raw_pipe.get_component(name)
        return unwrap_model_for_ema(comp) if comp is not None else None

    @property
    def vae(self) -> nn.Module:
        comp = self.get_component("vae", use_ema=False)
        assert comp is not None
        return comp

    @property
    def dit(self) -> nn.Module:
        comp = self.get_component("dit", use_ema=False)
        assert comp is not None
        return comp

    @property
    def dino(self) -> nn.Module | None:
        return self.get_component("dino", use_ema=False)

    def _reset_loss_accumulators(self) -> None:
        self._accum_steps = 0
        self._accum_total = 0.0
        self._accum_losses.clear()

    def _clear_train_visuals(self) -> None:
        self._captured_lq.clear()
        self._captured_gt.clear()
        self._captured_pred.clear()
        self._captured_count = 0

    def set_capture_train_visuals(self, enabled: bool, max_images: int = 0) -> None:
        """Enable or disable CPU snapshotting of training ``(lq, gt, pred)`` tensors."""
        if not isinstance(enabled, bool):
            raise ValueError(f"enabled must be a bool, got {type(enabled).__name__}: {enabled!r}.")
        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images < 0:
            raise ValueError(
                f"max_images must be a non-negative integer, got {max_images!r}."
            )
        target_max = max_images if enabled else 0
        if target_max == 0:
            self._capture_visuals_max = 0
            self._clear_train_visuals()
        else:
            self._capture_visuals_max = target_max

    def pop_train_visuals(self) -> dict[str, torch.Tensor] | None:
        """Return and clear the latest captured training ``lq``, ``gt``, and ``pred`` CPU tensors."""
        if self._captured_count <= 0:
            self._clear_train_visuals()
            return None
        visuals = {
            "lq": torch.cat(self._captured_lq, dim=0).contiguous(),
            "gt": torch.cat(self._captured_gt, dim=0).contiguous(),
            "pred": torch.cat(self._captured_pred, dim=0).contiguous(),
        }
        self._clear_train_visuals()
        return visuals

    def setup(self, accelerator: Accelerator) -> None:
        """Execute strict VOSR model setup lifecycle on ``accelerator``."""
        if self._is_setup:
            raise RuntimeError("VOSRModel.setup() has already been called.")

        grad_clip_norm = self.cfg.trainer.grad_clip_norm
        use_compile = self.cfg.trainer.compile
        logger = logging.getLogger("srtrainer")

        # Step 1: Instantiate independent arch components (vae, dit, optional dino).
        raw_components: dict[str, nn.Module] = {}
        for comp_name in VOSR_COMPONENTS:
            comp_cfg = self.cfg.arch.get(comp_name)
            if comp_cfg is None:
                continue
            spec = self.pretrain.get(comp_name)
            inst_cfg = (
                _clone_arch_cfg_skip_pretrained(comp_cfg)
                if (spec.path is not None and spec.strict)
                else comp_cfg
            )
            with accelerator.main_process_first():
                comp_mod = hydra.utils.instantiate(inst_cfg)
            if not isinstance(comp_mod, nn.Module):
                raise TypeError(
                    f"Instantiated cfg.arch.{comp_name} must be an nn.Module, "
                    f"got {type(comp_mod).__name__}."
                )
            raw_components[comp_name] = comp_mod

        active_components = tuple(name for name in VOSR_COMPONENTS if name in raw_components)

        # Instantiate active losses and optional degrader.
        active_losses: dict[str, tuple[float, Callable[..., torch.Tensor]]] = {}
        for loss_name, entry in self.cfg.loss.items():
            weight_val = float(entry.weight)
            if weight_val <= 0.0:
                if weight_val < 0.0 and accelerator.is_main_process:
                    logger.warning(
                        "[Model] Skipping loss %r with non-positive weight=%.6g.",
                        loss_name,
                        weight_val,
                    )
                continue
            with accelerator.main_process_first():
                criterion = hydra.utils.instantiate(entry.criterion)
            if not callable(criterion):
                raise TypeError(
                    f"Instantiated cfg.loss[{loss_name!r}].criterion must be callable, "
                    f"got {type(criterion).__name__}."
                )
            if isinstance(criterion, nn.Module):
                criterion = criterion.to(accelerator.device)
            active_losses[loss_name] = (weight_val, criterion)

        deg_cfg = self.cfg.data.degradation
        degrader: Any = None
        if deg_cfg is not None:
            deg_node = OmegaConf.structured(deg_cfg) if is_dataclass(deg_cfg) else deg_cfg
            degrader = hydra.utils.instantiate(deg_node)
            if not callable(degrader):
                raise TypeError(
                    f"Instantiated cfg.data.degradation must be callable, got {type(degrader).__name__}."
                )
            if isinstance(degrader, nn.Module):
                degrader = degrader.to(accelerator.device)
                degrader.eval()

        # Step 2: Optional per-arch pretrain loading onto raw modules BEFORE EMA / compile / DDP.
        for comp_name in active_components:
            spec = self.pretrain.get(comp_name)
            if spec.path is None:
                continue
            missing_keys, unexpected_keys = _load_component_weights(
                raw_components[comp_name],
                Path(spec.path),
                strict=spec.strict,
                key=spec.key,
            )
            if accelerator.is_main_process:
                logger.info(
                    "[Model] Loaded pretrained %s weights from %s (strict=%s, key=%r).",
                    comp_name,
                    spec.path,
                    spec.strict,
                    spec.key,
                )
                if not spec.strict and (missing_keys or unexpected_keys):
                    logger.warning(
                        "[Model] Pretrain %s strict=False load summary: missing=%d %s, unexpected=%d %s.",
                        comp_name,
                        len(missing_keys),
                        missing_keys,
                        len(unexpected_keys),
                        unexpected_keys,
                    )

        # Step 3: Apply per-arch trainability switches BEFORE EMA / compile / DDP.
        trainable_names: list[str] = []
        trainable_params: list[nn.Parameter] = []
        for comp_name in active_components:
            comp_mod = raw_components[comp_name]
            if self.trainable[comp_name]:
                comp_mod.requires_grad_(True)
                freeze_unused = getattr(comp_mod, "freeze_unused_parameters", None)
                if callable(freeze_unused):
                    freeze_unused()
                comp_params = [p for p in comp_mod.parameters() if p.requires_grad]
                if not comp_params:
                    raise ValueError(
                        f"Component {comp_name!r} is marked trainable=True but has no parameters "
                        "with requires_grad=True."
                    )
                trainable_names.append(comp_name)
                trainable_params.extend(comp_params)
            else:
                comp_mod.requires_grad_(False)
                comp_mod.eval()

        # Step 4: Clone FP32 EMA shadow ONLY for trainable components BEFORE compile and prepare().
        emas: dict[str, ModelEMA] = {}
        if self.ema_enabled:
            for comp_name in trainable_names:
                emas[comp_name] = ModelEMA(
                    raw_components[comp_name],
                    decay=self.ema_decay,
                    warmup_steps=self.ema_warmup_steps,
                    device=accelerator.device,
                )

        # Step 5: Optional torch.compile on trainable components.
        if use_compile:
            for comp_name in trainable_names:
                raw_components[comp_name] = torch.compile(raw_components[comp_name])

        # Assemble composable pipeline (validates cross-component channel / conditioning contracts).
        raw_pipeline = VOSRPipeline.from_model_config(
            vae=raw_components["vae"],
            dit=raw_components["dit"],
            dino=raw_components.get("dino"),
            model_cfg=self.model_cfg,
            trainable=self.model_cfg.trainable,
        )

        # Step 6: Instantiate partial optimizer and optional partial lr_scheduler over trainable params.
        opt_params = [p for p in raw_pipeline.parameters() if p.requires_grad]
        opt_factory = hydra.utils.instantiate(self.cfg.optim.optimizer, _convert_="all")
        optimizer = opt_factory(opt_params) if callable(opt_factory) else opt_factory
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Instantiated cfg.optim.optimizer must produce a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}."
            )

        sched_cfg = self.cfg.optim.scheduler
        lr_scheduler: Any = None
        if sched_cfg is not None:
            sched_factory = hydra.utils.instantiate(sched_cfg, _convert_="all")
            lr_scheduler = sched_factory(optimizer) if callable(sched_factory) else sched_factory
            for required_attr in ("step", "state_dict", "load_state_dict"):
                if not callable(getattr(lr_scheduler, required_attr, None)):
                    raise TypeError(
                        f"Instantiated cfg.optim.scheduler must expose callable {required_attr!r}."
                    )

        # Step 7: Prepare ONLY (pipeline, optimizer) — NEVER pass lr_scheduler to prepare().
        prepared_pipeline, prepared_optimizer = accelerator.prepare(raw_pipeline, optimizer)
        prepared_pipeline.train()

        # Deterministic registration for Accelerate checkpointing (lr_scheduler -> per-arch EMAs).
        if lr_scheduler is not None:
            accelerator.register_for_checkpointing(lr_scheduler)
        for comp_name in VOSR_COMPONENTS:
            if comp_name in emas:
                accelerator.register_for_checkpointing(emas[comp_name])

        if accelerator.is_main_process:
            frozen_names = [n for n in active_components if n not in trainable_names]
            logger.info(
                "[Model] VOSRModel ready | active=%s | trainable=%s | frozen=%s | ema=%s",
                list(active_components),
                trainable_names,
                frozen_names,
                list(emas.keys()),
            )

        self.accelerator = accelerator
        self.grad_clip_norm = grad_clip_norm
        self.losses = active_losses
        self.degrader = degrader
        self.pipeline = prepared_pipeline
        self.emas = emas
        self.optimizer = prepared_optimizer
        self.lr_scheduler = lr_scheduler
        self._active_components = active_components
        self._reset_loss_accumulators()
        self._clear_train_visuals()
        self._is_setup = True

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move unprepared train_loader batch tensors to ``accelerator.device`` and apply degrader."""
        accelerator = self._require_setup()
        if not isinstance(batch, Mapping):
            raise TypeError(f"Expected batch mapping, got {type(batch).__name__}.")

        moved = _move_to_device(dict(batch), accelerator.device, non_blocking=True)
        if self.degrader is not None:
            moved = self.degrader(moved)
        return moved

    def train_step(
        self,
        batch: dict[str, Any],
        global_step: int,
    ) -> tuple[dict[str, float], bool]:
        """Run one micro-step forward/backward, updating optimizer/LR/EMAs only on synced steps."""
        accelerator = self._require_setup()
        assert self.pipeline is not None and self.optimizer is not None

        if not isinstance(batch, Mapping) or "lq" not in batch or "gt" not in batch:
            raise ValueError("Training batch must be a mapping containing 'lq' and 'gt' tensors.")
        lq = batch["lq"]
        gt = batch["gt"]
        if not isinstance(lq, torch.Tensor) or not isinstance(gt, torch.Tensor):
            raise TypeError("Training batch 'lq' and 'gt' must be torch.Tensor instances.")

        if lq.device != accelerator.device:
            lq = lq.to(device=accelerator.device, non_blocking=True)
        if gt.device != accelerator.device:
            gt = gt.to(device=accelerator.device, non_blocking=True)

        if self._accum_steps == 0:
            self._clear_train_visuals()

        self.pipeline.train()
        try:
            with accelerator.accumulate(self.pipeline):
                with accelerator.autocast():
                    pred = self.pipeline(lq)
                    total_loss: torch.Tensor | None = None
                    step_raw_losses: dict[str, float] = {}
                    for name, (weight, criterion) in self.losses.items():
                        if getattr(criterion, "needs_batch", False):
                            raw_loss = criterion(pred, gt, batch=batch)
                        else:
                            raw_loss = criterion(pred, gt)
                        if not isinstance(raw_loss, torch.Tensor):
                            raise TypeError(
                                f"Loss {name!r} must return a torch.Tensor, got {type(raw_loss).__name__}."
                            )
                        if raw_loss.ndim > 0:
                            raw_loss = raw_loss.mean()
                        step_raw_losses[name] = float(raw_loss.detach().item())
                        weighted = raw_loss * weight
                        total_loss = weighted if total_loss is None else total_loss + weighted

                    assert total_loss is not None
                    step_total = float(total_loss.detach().item())

                accelerator.backward(total_loss)

                if (
                    self._capture_visuals_max > 0
                    and self._captured_count < self._capture_visuals_max
                    and lq.ndim == 4
                    and gt.ndim == 4
                    and pred.ndim == 4
                    and lq.shape[1] == 3
                    and gt.shape[1] == 3
                    and pred.shape[1] == 3
                ):
                    shapes_compatible = (
                        not self._captured_lq
                        or (
                            lq.shape[2:] == self._captured_lq[0].shape[2:]
                            and gt.shape[2:] == self._captured_gt[0].shape[2:]
                            and pred.shape[2:] == self._captured_pred[0].shape[2:]
                        )
                    )
                    if shapes_compatible:
                        need = min(
                            self._capture_visuals_max - self._captured_count,
                            int(lq.shape[0]),
                            int(gt.shape[0]),
                            int(pred.shape[0]),
                        )
                        if need > 0:
                            with torch.no_grad():
                                self._captured_lq.append(
                                    lq[:need].detach().to(device="cpu", dtype=torch.float32)
                                )
                                self._captured_gt.append(
                                    gt[:need].detach().to(device="cpu", dtype=torch.float32)
                                )
                                self._captured_pred.append(
                                    pred[:need].detach().to(device="cpu", dtype=torch.float32)
                                )
                            self._captured_count += need

                self._accum_steps += 1
                self._accum_total += step_total
                for name, raw_val in step_raw_losses.items():
                    self._accum_losses[name] = self._accum_losses.get(name, 0.0) + raw_val

                stepped = bool(accelerator.sync_gradients)
                if not stepped:
                    return {}, False

                grad_norm_val: float | None = None
                if self.grad_clip_norm is not None:
                    clip_params = [p for p in self.pipeline.parameters() if p.requires_grad]
                    gn = accelerator.clip_grad_norm_(clip_params, self.grad_clip_norm)
                    if gn is not None:
                        grad_norm_val = (
                            float(gn.detach().item())
                            if isinstance(gn, torch.Tensor)
                            else float(gn)
                        )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if not accelerator.optimizer_step_was_skipped:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.emas:
                        raw_pipe = unwrap_model_for_ema(self.pipeline)
                        assert isinstance(raw_pipe, VOSRPipeline)
                        for comp_name, ema_obj in self.emas.items():
                            comp = raw_pipe.get_component(comp_name)
                            assert comp is not None
                            ema_obj.update(comp)

                denom = max(1, self._accum_steps)
                scalars: dict[str, float] = {"loss/total": self._accum_total / denom}
                for name in self.losses:
                    scalars[f"loss/{name}"] = self._accum_losses.get(name, 0.0) / denom
                if grad_norm_val is not None:
                    scalars["grad_norm"] = grad_norm_val

                self._reset_loss_accumulators()
                return scalars, True
        except BaseException:
            self._reset_loss_accumulators()
            self._clear_train_visuals()
            raise

    def get_infer_network(self, use_ema: bool = True) -> nn.Module:
        """Return an unwrapped ``VOSRPipeline`` using EMA shadows for trainable archs when requested."""
        self._require_setup()
        assert self.pipeline is not None
        raw_pipe = unwrap_model_for_ema(self.pipeline)
        assert isinstance(raw_pipe, VOSRPipeline)

        if not use_ema or not self.emas:
            return raw_pipe

        vae_mod = self.emas["vae"].module if "vae" in self.emas else unwrap_model_for_ema(raw_pipe.vae)
        dit_mod = self.emas["dit"].module if "dit" in self.emas else unwrap_model_for_ema(raw_pipe.dit)
        dino_mod = (
            self.emas["dino"].module
            if "dino" in self.emas
            else (unwrap_model_for_ema(raw_pipe.dino) if raw_pipe.dino is not None else None)
        )
        infer_pipe = VOSRPipeline.from_model_config(
            vae=vae_mod,
            dit=dit_mod,
            dino=dino_mod,
            model_cfg=self.model_cfg,
            trainable=None,
        )
        infer_pipe.eval()
        return infer_pipe

    def _split_pipeline_state_dicts(
        self,
        accelerator: Accelerator,
    ) -> dict[str, dict[str, torch.Tensor]]:
        assert self.pipeline is not None
        raw_full = strip_state_dict_prefix(accelerator.get_state_dict(self.pipeline, unwrap=True))
        per_comp: dict[str, dict[str, torch.Tensor]] = {name: {} for name in self._active_components}
        for key, tensor in raw_full.items():
            for comp_name in self._active_components:
                prefix = f"{comp_name}."
                if key.startswith(prefix):
                    per_comp[comp_name][key[len(prefix) :]] = tensor
                    break

        cleaned: dict[str, dict[str, torch.Tensor]] = {}
        for comp_name, raw_sd in per_comp.items():
            cleaned[comp_name] = {
                k: v.detach().cpu().contiguous().clone()
                for k, v in strip_state_dict_prefix(raw_sd).items()
            }
        return cleaned

    def export_weights(
        self,
        accelerator: Accelerator,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Export separate, prefix-free CPU weight snapshots for each arch (and ``<arch>_ema``)."""
        self._require_setup()
        live_sds = self._split_pipeline_state_dicts(accelerator)
        exported: dict[str, dict[str, torch.Tensor]] = {}

        for comp_name in self._active_components:
            if self.trainable[comp_name] or self.save_frozen_weights:
                if live_sds[comp_name]:
                    exported[comp_name] = live_sds[comp_name]
            if comp_name in self.emas:
                ema_sd = {
                    k: v.detach().cpu().contiguous().clone()
                    for k, v in strip_state_dict_prefix(
                        self.emas[comp_name].module.state_dict()
                    ).items()
                }
                if ema_sd:
                    exported[f"{comp_name}_ema"] = ema_sd

        return exported

    def export_infer_weights(
        self,
        accelerator: Accelerator,
        use_ema: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Export separate per-arch inference snapshots (EMA where enabled and requested)."""
        self._require_setup()
        live_sds = self._split_pipeline_state_dicts(accelerator)
        exported: dict[str, dict[str, torch.Tensor]] = {}

        for comp_name in self._active_components:
            if not (self.trainable[comp_name] or self.save_frozen_weights):
                continue
            if use_ema and comp_name in self.emas:
                sd = {
                    k: v.detach().cpu().contiguous().clone()
                    for k, v in strip_state_dict_prefix(
                        self.emas[comp_name].module.state_dict()
                    ).items()
                }
            else:
                sd = live_sds[comp_name]
            if sd:
                exported[comp_name] = sd

        return exported

    def get_lr_dict(self) -> dict[str, float]:
        """Return ``{'lr': ...}`` for single param group or ``{'lr/group_i': ...}`` for multiple."""
        if self.optimizer is None or not self.optimizer.param_groups:
            return {}
        if len(self.optimizer.param_groups) == 1:
            return {"lr": float(self.optimizer.param_groups[0]["lr"])}
        return {
            f"lr/group_{idx}": float(group["lr"])
            for idx, group in enumerate(self.optimizer.param_groups)
        }
