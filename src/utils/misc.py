"""Weight I/O (safetensors/.pth), prefix stripping, and spatial padding/unpadding helpers."""

from __future__ import annotations

import numbers
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

_VALID_PAD_MODES = {"reflect", "replicate", "constant", "circular"}
_TORCH_WEIGHT_EXTENSIONS = {".pth", ".pt", ".bin", ".ckpt"}


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def resolve_2d_size(value: Any, name: str = "size") -> tuple[int, int]:
    """Normalize a square-or-rectangular spatial size into a positive ``(h, w)`` tuple.

    Accepted inputs are a positive integer (interpreted as square ``(value, value)``) or a
    2-element ``[h, w]`` sequence (``list``/``tuple``/OmegaConf ``ListConfig``). Bools, floats,
    strings, and sequences of any other length raise ``ValueError``.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be a positive integer or a 2-element [h, w] sequence, got {value!r}."
        )

    if isinstance(value, numbers.Integral):
        dim = int(value)
        if dim < 1:
            raise ValueError(
                f"{name} must be a positive integer or a 2-element [h, w] sequence, "
                f"got {value!r}."
            )
        return dim, dim

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise ValueError(f"{name} sequence must have length 2 ([h, w]), got {value!r}.")
        dims: list[int] = []
        for axis, raw in enumerate(value):
            if isinstance(raw, bool) or not isinstance(raw, numbers.Integral) or int(raw) < 1:
                raise ValueError(
                    f"{name}[{axis}] must be a positive integer, got {raw!r} "
                    f"(full value {value!r})."
                )
            dims.append(int(raw))
        return dims[0], dims[1]

    raise ValueError(
        f"{name} must be a positive integer or a 2-element [h, w] sequence, got {value!r}."
    )


def strip_state_dict_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: tuple[str, ...] = ("module.", "_orig_mod."),
) -> dict[str, torch.Tensor]:
    """Repeatedly strip wrapper prefixes (e.g., DDP ``module.``, compile ``_orig_mod.``) from keys."""
    active_prefixes = tuple(p for p in prefixes if p)
    cleaned: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        new_key = key
        changed = True
        while changed and active_prefixes:
            changed = False
            for prefix in active_prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = val
    return cleaned


def load_weights(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
    key: str | None = None,
) -> tuple[list[str], list[str]]:
    """Load model weights from ``.safetensors`` or ``.pth``/``.pt`` with prefix stripping.

    Returns ``(missing_keys, unexpected_keys)`` from ``model.load_state_dict``.
    """
    weight_path = Path(path)
    if not weight_path.is_file():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    suffix = weight_path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        if key is not None:
            raise KeyError(
                f"Key {key!r} is not supported for flat safetensors file {weight_path}; "
                "pass key=None."
            )
        state_dict: Mapping[str, torch.Tensor] = load_file(str(weight_path), device="cpu")
    elif suffix in _TORCH_WEIGHT_EXTENSIONS:
        raw_obj = torch.load(weight_path, map_location="cpu", weights_only=True)
        if not isinstance(raw_obj, Mapping):
            raise ValueError(
                f"Expected mapping checkpoint in {weight_path}, got {type(raw_obj).__name__}."
            )
        if key is not None:
            if key not in raw_obj:
                raise KeyError(
                    f"Key {key!r} not found in {weight_path}. "
                    f"Available keys: {sorted(map(str, raw_obj.keys()))}"
                )
            candidate = raw_obj[key]
            if not isinstance(candidate, Mapping):
                raise ValueError(
                    f"Expected mapping under key {key!r} in {weight_path}, "
                    f"got {type(candidate).__name__}."
                )
            state_dict = candidate
        else:
            if any(not isinstance(v, torch.Tensor) for v in raw_obj.values()):
                raise ValueError(
                    f"Checkpoint {weight_path} is nested "
                    f"(top-level keys: {sorted(map(str, raw_obj.keys()))}); "
                    "specify `key` (e.g. 'params_ema' or 'params')."
                )
            state_dict = raw_obj
    else:
        raise ValueError(
            f"Unsupported weight file format {suffix!r} for {weight_path}; "
            "expected '.safetensors', '.pth', or '.pt'."
        )

    cleaned = strip_state_dict_prefix(state_dict)
    incompatible = model.load_state_dict(cleaned, strict=strict)
    return (list(incompatible.missing_keys), list(incompatible.unexpected_keys))


def save_safetensors(
    state_dict: Mapping[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Atomically save a prefix-stripped, contiguous CPU clone ``state_dict`` to ``.safetensors``."""
    from safetensors.torch import save_file

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = strip_state_dict_prefix(state_dict)
    tensors: dict[str, torch.Tensor] = {}
    for k, v in cleaned.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for key {k!r}, got {type(v).__name__}.")
        tensors[k] = v.detach().cpu().contiguous().clone()

    clean_meta = {str(k): str(v) for k, v in metadata.items()} if metadata is not None else None

    fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        save_file(tensors, str(tmp_path), metadata=clean_meta)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


def _pad_bottom_right(
    tensor: torch.Tensor,
    pad_h: int,
    pad_w: int,
    mode: str,
) -> torch.Tensor:
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}.")
    if mode not in _VALID_PAD_MODES:
        raise ValueError(f"Unsupported pad mode={mode!r}; expected one of {sorted(_VALID_PAD_MODES)}.")
    if pad_h == 0 and pad_w == 0:
        return tensor

    orig_h, orig_w = tensor.shape[-2], tensor.shape[-1]
    effective_mode = (
        "replicate" if (mode == "reflect" and (pad_h >= orig_h or pad_w >= orig_w)) else mode
    )
    is_3d = tensor.ndim == 3
    x = tensor.unsqueeze(0) if is_3d else tensor
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode=effective_mode)
    return padded.squeeze(0) if is_3d else padded


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    mode: str = "reflect",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad bottom/right of a 3D/4D tensor so H and W are multiples of ``multiple``."""
    if not _is_positive_int(multiple):
        raise ValueError(f"multiple must be a positive integer, got {multiple!r}.")
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}.")

    orig_h, orig_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_h = (multiple - orig_h % multiple) % multiple
    pad_w = (multiple - orig_w % multiple) % multiple
    return _pad_bottom_right(tensor, pad_h, pad_w, mode), (orig_h, orig_w)


def pad_to_min_size(
    tensor: torch.Tensor,
    min_h: int,
    min_w: int,
    mode: str = "reflect",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad bottom/right of a 3D/4D tensor to at least ``(min_h, min_w)``."""
    if not _is_positive_int(min_h) or not _is_positive_int(min_w):
        raise ValueError(
            f"min_h and min_w must be positive integers, got min_h={min_h!r}, min_w={min_w!r}."
        )
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}.")

    orig_h, orig_w = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_h = max(0, min_h - orig_h)
    pad_w = max(0, min_w - orig_w)
    return _pad_bottom_right(tensor, pad_h, pad_w, mode), (orig_h, orig_w)


def unpad_tensor(
    tensor: torch.Tensor,
    orig_h: int,
    orig_w: int,
    scale: int = 1,
) -> torch.Tensor:
    """Crop bottom/right padding from a 3D/4D tensor back to ``(orig_h * scale, orig_w * scale)``."""
    if tensor.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}.")
    if not _is_positive_int(orig_h) or not _is_positive_int(orig_w) or not _is_positive_int(scale):
        raise ValueError(
            f"orig_h, orig_w, and scale must be positive integers, "
            f"got orig_h={orig_h!r}, orig_w={orig_w!r}, scale={scale!r}."
        )

    target_h = orig_h * scale
    target_w = orig_w * scale
    if tensor.shape[-2] < target_h or tensor.shape[-1] < target_w:
        raise ValueError(
            f"Tensor spatial size {(tensor.shape[-2], tensor.shape[-1])} is smaller than "
            f"target unpadded size {(target_h, target_w)}."
        )
    return tensor[..., :target_h, :target_w].contiguous()
