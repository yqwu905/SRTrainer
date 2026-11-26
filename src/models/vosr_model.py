"""Multi-architecture VOSR training paradigm (``VOSRModel``) and inference pipeline (``VOSRPipeline``)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from config.schema import (
    LoRAConfig,
    NoiseInput,
    PretrainEntryConfig,
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

__all__ = ["VOSRModel", "VOSRPipeline", "attach_dit_lora", "should_attach_dit_lora"]


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


def _build_arch(
    config: dict[str, Any], pretrain: PretrainEntryConfig, accelerator: Accelerator,
) -> nn.Module:
    if pretrain.path is not None and pretrain.strict:
        init_config = _clone_arch_cfg_skip_pretrained(config)
    else:
        init_config = config
    with accelerator.main_process_first():
        module = hydra.utils.instantiate(init_config)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected architecture to instantiate an nn.Module, got {type(module).__name__}.")
    if pretrain.path is not None:
        missing, unexpected = _load_component_weights(
            module, Path(pretrain.path), strict=pretrain.strict, key=pretrain.key,
        )
        if accelerator.is_main_process:
            logger = logging.getLogger("srtrainer")
            logger.info("[Model] Loaded %s from %s.", type(module).__name__, pretrain.path)
            if missing or unexpected:
                logger.warning("[Model] Pretrain missing=%s, unexpected=%s.", missing, unexpected)
    return module


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    module.requires_grad_(trainable)
    if not trainable:
        module.eval()
        return
    freeze_unused = getattr(module, "freeze_unused_parameters", None)
    if callable(freeze_unused):
        freeze_unused()
    if not any(p.requires_grad for p in module.parameters()):
        raise ValueError(f"{type(module).__name__} is marked trainable but has no trainable parameters.")


def attach_dit_lora(
    dit: nn.Module,
    lora_cfg: LoRAConfig,
) -> nn.Module:
    """Attach LoRA adapters to DiT (stub placeholder for custom LoRA injection).

    Args:
        dit: Instantiated DiT module (with base pretrained weights already loaded).
        lora_cfg: Typed ``LoRAConfig`` specifying ``lora_rank``, ``lora_alpha``, etc.

    Returns:
        The DiT module with LoRA layers attached (modified in-place or wrapped).
    """
    # TODO: Implement custom DiT LoRA injection logic here.
    del lora_cfg
    return dit


def should_attach_dit_lora(model_cfg: VOSRModelConfig) -> bool:
    """Return ``True`` iff DiT is trainable and ``model.lora.enable`` is ``True``."""
    return bool(model_cfg.trainable.dit and model_cfg.lora.enable)


def _apply_dit_lora(dit: nn.Module, lora_cfg: LoRAConfig) -> nn.Module:
    """Invoke ``attach_dit_lora`` and validate/normalize its return value."""
    attached = attach_dit_lora(dit, lora_cfg)
    if attached is None:
        return dit
    if not isinstance(attached, nn.Module):
        raise TypeError(
            f"attach_dit_lora() must return an nn.Module or None, got {type(attached).__name__}."
        )
    return attached


def _set_dit_lora_trainable_params(
    dit: nn.Module,
    base_param_ids: set[int],
) -> list[nn.Parameter]:
    """Freeze all non-LoRA parameters on ``dit`` and unfreeze only LoRA parameters."""
    named_params = list(dit.named_parameters())
    has_named_lora = any("lora" in name.lower() for name, _ in named_params)

    trainable_params: list[nn.Parameter] = []
    for name, param in named_params:
        is_lora_param = (
            ("lora" in name.lower())
            if has_named_lora
            else (id(param) not in base_param_ids)
        )
        param.requires_grad_(is_lora_param)
        if is_lora_param:
            trainable_params.append(param)
    return trainable_params


class VOSRPipeline(nn.Module):
    """RGB-to-RGB VOSR forward pass shared by distributed training and tiled inference."""

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
        trainable: VOSRTrainableConfig | None = None,
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

        vae_z_dim = vae.z_dim
        dit_out_ch = dit.out_channels
        dit_in_ch = dit.in_channels
        dit_z_dims = dit.z_dims

        if dit_in_ch != 2 * dit_out_ch:
            raise ValueError(
                f"VOSR requires dit.in_channels == 2 * dit.out_channels "
                f"(lq_latent + noise), got in_channels={dit_in_ch}, out_channels={dit_out_ch}."
            )
        if vae_z_dim != dit_out_ch:
            raise ValueError(
                f"VAE latent channels ({vae_z_dim}) does not match DiT out_channels ({dit_out_ch})."
            )
        if dit_z_dims is not None and dino is None:
            raise ValueError(
                f"DiT expects conditioning features (z_dims={dit_z_dims}), but arch.dino is None."
            )

        vae_factor = vae.spatial_compression_ratio
        dit_patch = dit.patch_size
        if vae_factor < 1 or dit_patch < 1:
            raise ValueError(
                f"vae.spatial_compression_ratio ({vae_factor}) and dit.patch_size ({dit_patch}) must be >= 1."
            )
        self._spatial_quantum: int = vae_factor * dit_patch

        self.vae: nn.Module = vae
        self.dit: nn.Module = dit
        self.dino: nn.Module | None = dino
        self.train_vae: bool = trainable.vae if trainable is not None else False
        self.train_dit: bool = trainable.dit if trainable is not None else False
        self.train_dino: bool = trainable.dino if trainable is not None else False

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

    def train(self, mode: bool = True) -> VOSRPipeline:
        super().train(mode)
        if mode:
            if not self.train_vae:
                self.vae.eval()
            if not self.train_dit:
                self.dit.eval()
            if self.dino is not None and not self.train_dino:
                self.dino.eval()
        return self

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

        if self.training and self.train_vae:
            lq_latent = self.vae.encode_normalized(lq_m11, posterior_mode=self.posterior_mode)
        else:
            with torch.no_grad(), torch.amp.autocast(device_type=lq_m11.device.type, enabled=False):
                lq_latent = self.vae.encode_normalized(lq_m11.float(), posterior_mode=self.posterior_mode)

        z_fea = None
        if self.dino is not None and self.dit.z_dims is not None:
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.training and self.train_dino):
                z_fea = self.dino(lq_pad)

        noise = self._sample_noise(lq_latent)
        dit_inp = torch.cat((lq_latent, noise), dim=1)
        t = torch.ones((b,), device=lq_latent.device, dtype=lq_latent.dtype)

        velocity = self.dit(dit_inp, t, z=z_fea)
        sr_latent = noise - velocity.float()
        clamp_decode = (not self.training) or self.clamp_decode_in_train
        sr_m11 = self.vae.decode_normalized(sr_latent, clamp_output=clamp_decode)
        sr_01 = (sr_m11 + 1.0) * 0.5
        if sr_01.shape[-2] != orig_h or sr_01.shape[-1] != orig_w:
            sr_01 = unpad_tensor(sr_01, orig_h, orig_w, scale=1)
        return sr_01


class VOSRModel(BaseModel):
    """Train a VAE and DiT with optional DINO conditioning."""

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(cfg)
        m: VOSRModelConfig = cfg.model

        self.model_cfg: VOSRModelConfig = m
        self.scale: int = m.scale
        self.lora: LoRAConfig = m.lora
        self.ema_enabled: bool = m.ema.enabled
        self.ema_decay: float = m.ema.decay
        self.ema_warmup_steps: int = m.ema.warmup_steps
        self.save_frozen_weights: bool = m.save_frozen_weights
        self.pretrain: VOSRPretrainConfig = m.pretrain

        self.pipeline: nn.Module | None = None
        self.vae: nn.Module | None = None
        self.dit: nn.Module | None = None
        self.dino: nn.Module | None = None
        self.vae_ema: ModelEMA | None = None
        self.dit_ema: ModelEMA | None = None
        self.dino_ema: ModelEMA | None = None
        self.losses: dict[str, tuple[float, Callable[..., torch.Tensor]]] = {}
        self.degrader: nn.Module | Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: Any | None = None
        self.grad_clip_norm: float | None = None

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
    def dit_lora_enabled(self) -> bool:
        return should_attach_dit_lora(self.model_cfg)

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

        arch = self.cfg.arch
        assert arch.vae is not None and arch.dit is not None
        vae = _build_arch(arch.vae, self.pretrain.vae, accelerator)
        dit = _build_arch(arch.dit, self.pretrain.dit, accelerator)
        dino = _build_arch(arch.dino, self.pretrain.dino, accelerator) if arch.dino is not None else None

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

        trainable = self.model_cfg.trainable
        _set_trainable(vae, trainable.vae)
        if dino is not None:
            _set_trainable(dino, trainable.dino)

        if self.dit_lora_enabled:
            dit_base_param_ids = {id(p) for p in dit.parameters()}
            dit.requires_grad_(False)
            dit = _apply_dit_lora(dit, self.lora)
            if self.lora.checkpoint is not None:
                missing_keys, unexpected_keys = load_weights(
                    dit,
                    Path(self.lora.checkpoint),
                    strict=False,
                )
                if accelerator.is_main_process:
                    logger.info(
                        "[Model] Loaded DiT LoRA checkpoint from %s (rank=%d, alpha=%g).",
                        self.lora.checkpoint,
                        self.lora.lora_rank,
                        self.lora.lora_alpha,
                    )
                    if missing_keys or unexpected_keys:
                        logger.warning(
                            "[Model] DiT LoRA checkpoint load summary: missing=%d %s, unexpected=%d %s.",
                            len(missing_keys),
                            missing_keys,
                            len(unexpected_keys),
                            unexpected_keys,
                        )

            if not _set_dit_lora_trainable_params(dit, dit_base_param_ids):
                raise ValueError(
                    "No trainable LoRA parameters were found after attach_dit_lora(). "
                    "Implement attach_dit_lora() to inject LoRA parameters into DiT."
                )
        else:
            _set_trainable(dit, trainable.dit)

        def make_ema(module: nn.Module) -> ModelEMA:
            return ModelEMA(
                module,
                decay=self.ema_decay,
                warmup_steps=self.ema_warmup_steps,
                device=accelerator.device,
            )

        vae_ema = make_ema(vae) if self.ema_enabled and trainable.vae else None
        dit_ema = make_ema(dit) if self.ema_enabled and trainable.dit else None
        dino_ema = make_ema(dino) if self.ema_enabled and trainable.dino and dino is not None else None

        raw_pipeline = VOSRPipeline.from_model_config(
            vae=torch.compile(vae) if use_compile and trainable.vae else vae,
            dit=torch.compile(dit) if use_compile and trainable.dit else dit,
            dino=torch.compile(dino) if use_compile and trainable.dino and dino is not None else dino,
            model_cfg=self.model_cfg,
            trainable=trainable,
        )

        # Build the optimizer from the parameters used by the training pipeline.
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

        # The scheduler advances only on successful optimizer updates.
        prepared_pipeline, prepared_optimizer = accelerator.prepare(raw_pipeline, optimizer)
        prepared_pipeline.train()

        # Deterministic registration for Accelerate checkpointing (lr_scheduler -> per-arch EMAs).
        if lr_scheduler is not None:
            accelerator.register_for_checkpointing(lr_scheduler)
        if vae_ema is not None:
            accelerator.register_for_checkpointing(vae_ema)
        if dit_ema is not None:
            accelerator.register_for_checkpointing(dit_ema)
        if dino_ema is not None:
            accelerator.register_for_checkpointing(dino_ema)

        prepared = unwrap_model_for_ema(prepared_pipeline)
        self.vae = unwrap_model_for_ema(prepared.vae)
        self.dit = unwrap_model_for_ema(prepared.dit)
        self.dino = unwrap_model_for_ema(prepared.dino) if prepared.dino is not None else None
        self.vae_ema = vae_ema
        self.dit_ema = dit_ema
        self.dino_ema = dino_ema
        self.accelerator = accelerator
        self.grad_clip_norm = grad_clip_norm
        self.losses = active_losses
        self.degrader = degrader
        self.pipeline = prepared_pipeline
        self.optimizer = prepared_optimizer
        self.lr_scheduler = lr_scheduler
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
                    if self.vae_ema is not None:
                        assert self.vae is not None
                        self.vae_ema.update(self.vae)
                    if self.dit_ema is not None:
                        assert self.dit is not None
                        self.dit_ema.update(self.dit)
                    if self.dino_ema is not None:
                        assert self.dino is not None
                        self.dino_ema.update(self.dino)

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
        if not use_ema or (self.vae_ema is None and self.dit_ema is None and self.dino_ema is None):
            return unwrap_model_for_ema(self.pipeline)

        assert self.vae is not None and self.dit is not None
        return VOSRPipeline.from_model_config(
            vae=self.vae_ema.module if self.vae_ema else self.vae,
            dit=self.dit_ema.module if self.dit_ema else self.dit,
            dino=self.dino_ema.module if self.dino_ema else self.dino,
            model_cfg=self.model_cfg,
        ).eval()

    def _export_weights(
        self, accelerator: Accelerator, *, inference: bool, use_ema: bool,
    ) -> dict[str, dict[str, torch.Tensor]]:
        self._require_setup()
        assert self.pipeline is not None
        full = strip_state_dict_prefix(accelerator.get_state_dict(self.pipeline, unwrap=True))
        exported: dict[str, dict[str, torch.Tensor]] = {}

        def snapshot(key: str, state: dict[str, torch.Tensor]) -> None:
            if state:
                exported[key] = {
                    k: v.detach().cpu().contiguous().clone()
                    for k, v in strip_state_dict_prefix(state).items()
                }

        def export(name: str, ema: ModelEMA | None) -> None:
            if not inference or not use_ema or ema is None:
                prefix = name + "."
                live = {k.removeprefix(prefix): v for k, v in full.items() if k.startswith(prefix)}
                snapshot(name, live)
            if ema is not None and use_ema:
                snapshot(name if inference else name + "_ema", ema.module.state_dict())

        trainable = self.model_cfg.trainable
        if trainable.vae or self.save_frozen_weights:
            export("vae", self.vae_ema)
        if trainable.dit or self.save_frozen_weights:
            export("dit", self.dit_ema)
        if self.dino is not None and (trainable.dino or self.save_frozen_weights):
            export("dino", self.dino_ema)
        return exported

    def export_weights(self, accelerator: Accelerator) -> dict[str, dict[str, torch.Tensor]]:
        return self._export_weights(accelerator, inference=False, use_ema=True)

    def export_infer_weights(
        self, accelerator: Accelerator, use_ema: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return self._export_weights(accelerator, inference=True, use_ema=use_ema)

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
