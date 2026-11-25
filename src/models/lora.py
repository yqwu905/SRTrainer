"""DiT LoRA configuration parser and injection hook stub."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig
from torch import nn

__all__ = [
    "DiTLoRAConfig",
    "initialize_dit_ffn_big_lora",
    "parse_dit_lora_config",
]

_ALLOWED_LORA_KEYS: frozenset[str] = frozenset({"enabled", "rank", "alpha"})


@dataclass(frozen=True)
class DiTLoRAConfig:
    """Validated DiT LoRA settings parsed from ``model.lora``."""

    enabled: bool = False
    rank: int = 16
    alpha: int | float = 16


def _parse_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}: {value!r}.")
    return value


def _parse_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _parse_positive_real(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value!r}.")
        return int(value)
    val_f = float(value)
    if not math.isfinite(val_f) or val_f <= 0.0:
        raise ValueError(f"{name} must be > 0 and finite, got {value!r}.")
    return val_f


def parse_dit_lora_config(raw: Any) -> DiTLoRAConfig:
    """Parse and validate ``model.lora`` (mapping or None) into a :class:`DiTLoRAConfig`."""
    if raw is None:
        return DiTLoRAConfig()
    if not isinstance(raw, (DictConfig, Mapping)):
        raise TypeError(f"model.lora must be a mapping or None, got {type(raw).__name__}.")

    unknown = set(map(str, raw.keys())) - _ALLOWED_LORA_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys {sorted(unknown)} in model.lora; "
            f"expected subset of {sorted(_ALLOWED_LORA_KEYS)}."
        )

    enabled = _parse_bool(raw.get("enabled", False), "model.lora.enabled")
    rank = _parse_positive_int(raw.get("rank", 16), "model.lora.rank")
    raw_alpha = raw.get("alpha", None)
    alpha: int | float = rank if raw_alpha is None else _parse_positive_real(raw_alpha, "model.lora.alpha")
    return DiTLoRAConfig(enabled=enabled, rank=rank, alpha=alpha)


def initialize_dit_ffn_big_lora(
    dit: nn.Module,
    lora_rank: int,
    lora_alpha: int | float,
) -> tuple[nn.Module, list[str]]:
    """Attach LoRA adapters to ``dit`` and return ``(dit, lora_target_modules)``.

    Stub hook for custom DiT LoRA initialization. Replace the body below with your
    concrete implementation (for example, PEFT ``LoraConfig`` + ``get_peft_model`` or
    custom in-place adapter injection).

    Contract expected by ``VOSRModel`` and ``tools/infer.py``:
    - Called AFTER loading base pretrained weights during training, and BEFORE loading
      full LoRA-trained checkpoints during inference.
    - Should freeze non-LoRA parameters (``requires_grad=False``) and leave LoRA
      parameters trainable (``requires_grad=True``).
    - Returns ``(dit_with_lora, lora_target_modules)``.
    """
    raise NotImplementedError(
        "initialize_dit_ffn_big_lora(dit, lora_rank, lora_alpha) is a stub in "
        "src/models/lora.py; implement your DiT LoRA strategy here."
    )
