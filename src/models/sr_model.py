"""Single-network super-resolution training paradigm (``SRModel``)."""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from models.base_model import BaseModel
from models.ema import ModelEMA, unwrap_model_for_ema
from utils.misc import load_weights, strip_state_dict_prefix

if TYPE_CHECKING:
    from accelerate import Accelerator

__all__ = ["SRModel"]

_UNSET: Any = object()


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


def _validate_target_mapping(cfg_node: Any, name: str) -> Any:
    if not isinstance(cfg_node, (DictConfig, Mapping)):
        raise ValueError(f"{name} must be a mapping containing '_target_', got {type(cfg_node).__name__}.")
    target = cfg_node.get("_target_", None)
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"{name} must define a non-empty string '_target_'.")
    return cfg_node


def _parse_optional_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        if not str(value).strip():
            raise ValueError(f"{name} cannot be an empty path.")
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{name} cannot be an empty string; pass None to disable.")
        return Path(stripped)
    raise TypeError(f"{name} must be a str, Path, or None, got {type(value).__name__}.")


def _parse_optional_key(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    raise TypeError(f"{name} must be a str or None, got {type(value).__name__}.")


def _parse_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}: {value!r}.")
    return value


def _parse_positive_real_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a positive finite real or None, got {value!r}.")
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be > 0 and finite, got {value!r}.")
    return val


class SRModel(BaseModel):
    """Single-network SR training paradigm with composable losses, FP32 EMA, and online GPU degradation."""

    def __init__(
        self,
        cfg: DictConfig | Mapping[str, Any],
        ema: DictConfig | Mapping[str, Any] | None = _UNSET,
        pretrain_path: str | Path | None = _UNSET,
        pretrain_strict: bool | None = _UNSET,
        pretrain_key: str | None = _UNSET,
    ) -> None:
        super().__init__(cfg)

        raw_ema = (
            OmegaConf.select(self.cfg, "model.ema", default=None) if ema is _UNSET else ema
        )
        if raw_ema is None:
            self.ema_enabled: bool = False
            self.ema_decay: float = 0.999
            self.ema_warmup_steps: int = 0
        elif isinstance(raw_ema, (DictConfig, Mapping)):
            self.ema_enabled = _parse_bool(raw_ema.get("enabled", True), "model.ema.enabled")
            raw_decay = raw_ema.get("decay", 0.999)
            if isinstance(raw_decay, bool) or not isinstance(raw_decay, numbers.Real):
                raise ValueError(f"model.ema.decay must be a finite real in (0, 1], got {raw_decay!r}.")
            decay_f = float(raw_decay)
            if not math.isfinite(decay_f) or not (0.0 < decay_f <= 1.0):
                raise ValueError(f"model.ema.decay must be in (0, 1], got {raw_decay!r}.")
            self.ema_decay = decay_f

            raw_warmup = raw_ema.get("warmup_steps", 0)
            if isinstance(raw_warmup, bool) or not isinstance(raw_warmup, int) or raw_warmup < 0:
                raise ValueError(
                    f"model.ema.warmup_steps must be a non-negative integer, got {raw_warmup!r}."
                )
            self.ema_warmup_steps = int(raw_warmup)
        else:
            raise TypeError(
                f"model.ema must be a mapping or None, got {type(raw_ema).__name__}."
            )

        raw_pretrain_path = (
            OmegaConf.select(self.cfg, "model.pretrain_path", default=None)
            if pretrain_path is _UNSET
            else pretrain_path
        )
        self.pretrain_path: Path | None = _parse_optional_path(
            raw_pretrain_path, "model.pretrain_path"
        )

        raw_pretrain_strict = (
            OmegaConf.select(self.cfg, "model.pretrain_strict", default=True)
            if pretrain_strict is _UNSET
            else pretrain_strict
        )
        self.pretrain_strict: bool = _parse_bool(raw_pretrain_strict, "model.pretrain_strict")

        raw_pretrain_key = (
            OmegaConf.select(self.cfg, "model.pretrain_key", default=None)
            if pretrain_key is _UNSET
            else pretrain_key
        )
        self.pretrain_key: str | None = _parse_optional_key(raw_pretrain_key, "model.pretrain_key")

        self.net_g: nn.Module | None = None
        self.ema: ModelEMA | None = None
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
        if not self._is_setup or self.accelerator is None or self.net_g is None or self.optimizer is None:
            raise RuntimeError("SRModel.setup(accelerator) must be called before using the model.")
        return self.accelerator

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
        """Execute strict 7-step model setup lifecycle on ``accelerator``."""
        if self._is_setup:
            raise RuntimeError("SRModel.setup() has already been called.")

        grad_clip_norm = _parse_positive_real_or_none(
            OmegaConf.select(self.cfg, "trainer.grad_clip_norm", default=None),
            "trainer.grad_clip_norm",
        )
        use_compile = _parse_bool(
            OmegaConf.select(self.cfg, "trainer.compile", default=False),
            "trainer.compile",
        )

        logger = logging.getLogger("srtrainer")

        # Step 1: Instantiate raw net_g, active losses, and optional degrader.
        arch_cfg = _validate_target_mapping(
            OmegaConf.select(self.cfg, "arch", default=None), "cfg.arch"
        )
        with accelerator.main_process_first():
            raw_net_g = hydra.utils.instantiate(arch_cfg)
        if not isinstance(raw_net_g, nn.Module):
            raise TypeError(f"Instantiated cfg.arch must be an nn.Module, got {type(raw_net_g).__name__}.")

        loss_cfg = OmegaConf.select(self.cfg, "loss", default=None)
        if not isinstance(loss_cfg, (DictConfig, Mapping)) or len(loss_cfg) == 0:
            raise ValueError("cfg.loss must be a non-empty mapping of named loss entries.")

        active_losses: dict[str, tuple[float, Callable[..., torch.Tensor]]] = {}
        for loss_name, entry in loss_cfg.items():
            if entry is None:
                continue
            if not isinstance(entry, (DictConfig, Mapping)):
                raise ValueError(
                    f"cfg.loss[{loss_name!r}] must be a mapping with 'weight' and 'criterion', "
                    f"got {type(entry).__name__}."
                )
            if "weight" not in entry:
                raise ValueError(f"cfg.loss[{loss_name!r}] is missing required key 'weight'.")
            raw_weight = entry["weight"]
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, numbers.Real):
                raise ValueError(
                    f"cfg.loss[{loss_name!r}].weight must be a finite real number, got {raw_weight!r}."
                )
            weight_val = float(raw_weight)
            if not math.isfinite(weight_val):
                raise ValueError(
                    f"cfg.loss[{loss_name!r}].weight must be finite, got {raw_weight!r}."
                )
            if weight_val <= 0.0:
                if weight_val < 0.0 and accelerator.is_main_process:
                    logger.warning(
                        "[Model] Skipping loss %r with non-positive weight=%.6g.",
                        str(loss_name),
                        weight_val,
                    )
                continue

            crit_cfg = _validate_target_mapping(
                entry.get("criterion", None), f"cfg.loss[{loss_name!r}].criterion"
            )
            with accelerator.main_process_first():
                criterion = hydra.utils.instantiate(crit_cfg)
            if not callable(criterion):
                raise TypeError(
                    f"Instantiated cfg.loss[{loss_name!r}].criterion must be callable, "
                    f"got {type(criterion).__name__}."
                )
            if isinstance(criterion, nn.Module):
                criterion = criterion.to(accelerator.device)
            active_losses[str(loss_name)] = (weight_val, criterion)

        if not active_losses:
            raise ValueError(
                "cfg.loss must contain at least one active loss entry with weight > 0."
            )

        deg_cfg = OmegaConf.select(self.cfg, "data.degradation", default=None)
        degrader: Any = None
        if deg_cfg is not None:
            _validate_target_mapping(deg_cfg, "cfg.data.degradation")
            degrader = hydra.utils.instantiate(deg_cfg)
            if not callable(degrader):
                raise TypeError(
                    f"Instantiated cfg.data.degradation must be callable, got {type(degrader).__name__}."
                )
            if isinstance(degrader, nn.Module):
                degrader = degrader.to(accelerator.device)
                degrader.eval()

        # Step 2: Optional pretrain loading onto raw net_g BEFORE EMA / compile / DDP.
        if self.pretrain_path is not None:
            missing_keys, unexpected_keys = load_weights(
                raw_net_g,
                self.pretrain_path,
                strict=self.pretrain_strict,
                key=self.pretrain_key,
            )
            if accelerator.is_main_process:
                logger.info(
                    "[Model] Loaded pretrained weights from %s (strict=%s, key=%r).",
                    self.pretrain_path,
                    self.pretrain_strict,
                    self.pretrain_key,
                )
                if not self.pretrain_strict and (missing_keys or unexpected_keys):
                    logger.warning(
                        "[Model] Pretrain strict=False load summary: missing=%d %s, unexpected=%d %s.",
                        len(missing_keys),
                        missing_keys,
                        len(unexpected_keys),
                        unexpected_keys,
                    )

        # Step 3: Clone FP32 EMA shadow BEFORE compile and BEFORE accelerator.prepare().
        ema_obj: ModelEMA | None = None
        if self.ema_enabled:
            ema_obj = ModelEMA(
                raw_net_g,
                decay=self.ema_decay,
                warmup_steps=self.ema_warmup_steps,
                device=accelerator.device,
            )

        # Step 4: Optional torch.compile on raw net_g.
        if use_compile:
            raw_net_g = torch.compile(raw_net_g)

        # Step 5: Instantiate partial optimizer and optional partial lr_scheduler.
        optim_root = OmegaConf.select(self.cfg, "optim", default=None)
        if not isinstance(optim_root, (DictConfig, Mapping)):
            raise ValueError("cfg.optim must be a mapping containing 'optimizer'.")
        opt_cfg = _validate_target_mapping(optim_root.get("optimizer", None), "cfg.optim.optimizer")
        trainable_params = [p for p in raw_net_g.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("Instantiated net_g has no trainable parameters (requires_grad=True).")

        opt_factory = hydra.utils.instantiate(opt_cfg, _convert_="all")
        optimizer = opt_factory(trainable_params) if callable(opt_factory) else opt_factory
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Instantiated cfg.optim.optimizer must produce a torch.optim.Optimizer, "
                f"got {type(optimizer).__name__}."
            )

        sched_cfg = optim_root.get("scheduler", None)
        lr_scheduler: Any = None
        if sched_cfg is not None:
            _validate_target_mapping(sched_cfg, "cfg.optim.scheduler")
            sched_factory = hydra.utils.instantiate(sched_cfg, _convert_="all")
            lr_scheduler = sched_factory(optimizer) if callable(sched_factory) else sched_factory
            for required_attr in ("step", "state_dict", "load_state_dict"):
                if not callable(getattr(lr_scheduler, required_attr, None)):
                    raise TypeError(
                        f"Instantiated cfg.optim.scheduler must expose callable {required_attr!r}."
                    )

        # Step 6: Prepare ONLY (net_g, optimizer) — NEVER pass lr_scheduler to prepare().
        prepared_net_g, prepared_optimizer = accelerator.prepare(raw_net_g, optimizer)
        prepared_net_g.train()

        # Step 7: Deterministic registration for Accelerate checkpointing (lr_scheduler -> ema).
        if lr_scheduler is not None:
            accelerator.register_for_checkpointing(lr_scheduler)
        if ema_obj is not None:
            accelerator.register_for_checkpointing(ema_obj)

        self.accelerator = accelerator
        self.grad_clip_norm = grad_clip_norm
        self.losses = active_losses
        self.degrader = degrader
        self.net_g = prepared_net_g
        self.ema = ema_obj
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
        """Run one micro-step forward/backward, updating optimizer/LR/EMA only on synced steps."""
        accelerator = self._require_setup()
        assert self.net_g is not None and self.optimizer is not None

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

        self.net_g.train()
        try:
            with accelerator.accumulate(self.net_g):
                with accelerator.autocast():
                    pred = self.net_g(lq)
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

                # Accelerate divides loss by gradient_accumulation_steps internally.
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
                    gn = accelerator.clip_grad_norm_(self.net_g.parameters(), self.grad_clip_norm)
                    if gn is not None:
                        grad_norm_val = (
                            float(gn.detach().item()) if isinstance(gn, torch.Tensor) else float(gn)
                        )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if not accelerator.optimizer_step_was_skipped:
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    if self.ema is not None:
                        self.ema.update(self.net_g)

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
        """Return an unwrapped inference network (EMA shadow when enabled and requested)."""
        self._require_setup()
        assert self.net_g is not None
        if use_ema and self.ema is not None:
            return self.ema.module
        return unwrap_model_for_ema(self.net_g)

    def export_weights(
        self,
        accelerator: Accelerator,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Export prefix-free, contiguous CPU weight snapshots for ``net_g`` (and ``net_g_ema``)."""
        self._require_setup()
        assert self.net_g is not None

        raw_net_sd = accelerator.get_state_dict(self.net_g, unwrap=True)
        clean_net_sd = {
            k: v.detach().cpu().contiguous().clone()
            for k, v in strip_state_dict_prefix(raw_net_sd).items()
        }
        exported: dict[str, dict[str, torch.Tensor]] = {"net_g": clean_net_sd}

        if self.ema is not None:
            clean_ema_sd = {
                k: v.detach().cpu().contiguous().clone()
                for k, v in strip_state_dict_prefix(self.ema.module.state_dict()).items()
            }
            exported["net_g_ema"] = clean_ema_sd

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
