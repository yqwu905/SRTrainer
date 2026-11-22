"""FP32 exponential moving average (EMA) shadow model with Accelerate checkpointing support."""

from __future__ import annotations

import copy
import math
import numbers
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from srtrainer.utils.misc import strip_state_dict_prefix

__all__ = ["ModelEMA", "unwrap_model_for_ema"]


def _validate_decay(decay: Any) -> float:
    if isinstance(decay, bool) or not isinstance(decay, numbers.Real):
        raise ValueError(f"EMA decay must be a finite real number in (0, 1], got {decay!r}.")
    decay_f = float(decay)
    if not math.isfinite(decay_f) or not (0.0 < decay_f <= 1.0):
        raise ValueError(f"EMA decay must be in (0, 1], got {decay!r}.")
    return decay_f


def _validate_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return int(value)


def unwrap_model_for_ema(model: nn.Module) -> nn.Module:
    """Safely unwrap nested DDP/DataParallel/FSDP and ``torch.compile`` wrappers without mutating ``model``.

    Always keeps ``keep_fp32_wrapper=True`` so Accelerate's mixed-precision forward wrapper on a
    live training model is never stripped in-place, and iterates to a fixed point so nested
    ``DDP(OptimizedModule(net))`` and ``OptimizedModule(DDP(net))`` both unwrap cleanly.
    """
    from accelerate.utils.other import extract_model_from_parallel

    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected nn.Module, got {type(model).__name__}.")

    cur: nn.Module = model
    prev: nn.Module | None = None
    while cur is not prev:
        prev = cur
        cur = extract_model_from_parallel(
            cur,
            keep_fp32_wrapper=True,
            keep_torch_compile=False,
        )
        orig = getattr(cur, "_orig_mod", None)
        if isinstance(orig, nn.Module):
            cur = orig
        mod = getattr(cur, "module", None)
        if (
            isinstance(mod, nn.Module)
            and list(cur._modules.keys()) == ["module"]
            and not list(cur.parameters(recurse=False))
            and not list(cur.buffers(recurse=False))
        ):
            cur = mod
    return cur


class ModelEMA:
    """FP32 exponential moving average shadow copy compatible with Accelerate checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}.")

        self.decay: float = _validate_decay(decay)
        self.warmup_steps: int = _validate_non_negative_int(warmup_steps, "warmup_steps")
        self.num_updates: int = 0

        target_device: torch.device | None
        if device is None:
            target_device = None
        elif isinstance(device, (torch.device, str)):
            target_device = torch.device(device)
        else:
            raise TypeError(
                f"device must be torch.device, str, or None, got {type(device).__name__}."
            )

        raw_model = unwrap_model_for_ema(model)
        shadow = copy.deepcopy(raw_model)
        # Cast floating parameters and buffers to FP32 while preserving non-floating buffer dtypes.
        shadow.float()
        shadow.eval()
        for param in shadow.parameters():
            param.requires_grad_(False)
        if target_device is not None:
            shadow.to(device=target_device)

        self.module: nn.Module = shadow

    def to(self, device: torch.device | str) -> ModelEMA:
        """Move the EMA shadow module to ``device`` while preserving FP32/integer dtypes."""
        if not isinstance(device, (torch.device, str)):
            raise TypeError(f"device must be torch.device or str, got {type(device).__name__}.")
        self.module.to(device=torch.device(device))
        return self

    def get_decay(self, num_updates: int | None = None) -> float:
        """Return effective EMA decay at ``num_updates`` (defaults to ``self.num_updates``)."""
        updates = (
            self.num_updates
            if num_updates is None
            else _validate_non_negative_int(num_updates, "num_updates")
        )
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1 + updates) / (self.warmup_steps + updates))

    def _module_device(self) -> torch.device:
        first_param = next(self.module.parameters(), None)
        if first_param is not None:
            return first_param.device
        first_buf = next(self.module.buffers(), None)
        if first_buf is not None:
            return first_buf.device
        return torch.device("cpu")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update FP32 EMA parameters and buffers from ``model``."""
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}.")

        src = unwrap_model_for_ema(model)
        ema_params = dict(self.module.named_parameters())
        src_params = dict(src.named_parameters())

        if ema_params.keys() != src_params.keys():
            missing = sorted(ema_params.keys() - src_params.keys())
            unexpected = sorted(src_params.keys() - ema_params.keys())
            raise ValueError(
                f"EMA parameter key mismatch: missing={missing}, unexpected={unexpected}."
            )
        for name, ema_p in ema_params.items():
            if ema_p.shape != src_params[name].shape:
                raise ValueError(
                    f"EMA parameter shape mismatch for {name!r}: "
                    f"expected {tuple(ema_p.shape)}, got {tuple(src_params[name].shape)}."
                )

        self.num_updates += 1
        decay = self.get_decay()

        for name, ema_p in ema_params.items():
            src_p = src_params[name].detach()
            if ema_p.is_floating_point():
                src_fp32 = src_p.to(device=ema_p.device, dtype=torch.float32)
                if decay < 1.0:
                    ema_p.lerp_(src_fp32, 1.0 - decay)
            else:
                ema_p.copy_(src_p.to(device=ema_p.device, dtype=ema_p.dtype))

        src_buffers = dict(src.named_buffers())
        for name, ema_b in self.module.named_buffers():
            if name not in src_buffers:
                continue
            src_b = src_buffers[name].detach()
            if ema_b.is_floating_point():
                ema_b.copy_(src_b.to(device=ema_b.device, dtype=torch.float32))
            else:
                ema_b.copy_(src_b.to(device=ema_b.device, dtype=ema_b.dtype))

    def state_dict(self) -> dict[str, Any]:
        """Return a CPU-detached, ``weights_only``-safe state dict for Accelerate checkpointing."""
        return {
            "decay": float(self.decay),
            "warmup_steps": int(self.warmup_steps),
            "num_updates": int(self.num_updates),
            "module": {
                k: v.detach().cpu().clone() for k, v in self.module.state_dict().items()
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore EMA weights and counters while preserving ``self.module`` device and FP32 dtype."""
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"state_dict must be a Mapping, got {type(state_dict).__name__}."
            )
        if "module" not in state_dict or not isinstance(state_dict["module"], Mapping):
            raise ValueError("ModelEMA state_dict must contain a 'module' state dict mapping.")

        new_num_updates = _validate_non_negative_int(
            state_dict.get("num_updates", 0), "num_updates"
        )
        new_decay = (
            _validate_decay(state_dict["decay"]) if "decay" in state_dict else self.decay
        )
        new_warmup_steps = (
            _validate_non_negative_int(state_dict["warmup_steps"], "warmup_steps")
            if "warmup_steps" in state_dict
            else self.warmup_steps
        )

        target_device = self._module_device()
        cur_sd = self.module.state_dict()
        cleaned = strip_state_dict_prefix(state_dict["module"])

        prepared: dict[str, torch.Tensor] = {}
        for key, value in cleaned.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"ModelEMA state_dict['module'][{key!r}] must be a torch.Tensor, "
                    f"got {type(value).__name__}."
                )
            if key in cur_sd:
                if cur_sd[key].is_floating_point():
                    prepared[key] = value.detach().to(device=target_device, dtype=torch.float32)
                else:
                    prepared[key] = value.detach().to(device=target_device, dtype=cur_sd[key].dtype)
            else:
                prepared[key] = value.detach().to(device=target_device)

        self.module.load_state_dict(prepared, strict=True)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

        self.num_updates = new_num_updates
        self.decay = new_decay
        self.warmup_steps = new_warmup_steps
