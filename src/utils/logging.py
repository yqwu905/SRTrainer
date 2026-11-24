"""Logging helpers: Rank-0 terminal/file logger, ETA/training line formatters, and tracker image logging."""

from __future__ import annotations

import logging
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from accelerate import Accelerator

_WARNED_UNSUPPORTED_TRACKERS: set[str] = set()


def setup_logger(
    log_file: Path | None,
    is_main_process: bool,
    name: str = "srtrainer",
) -> logging.Logger:
    """Configure an idempotent Rank-0 logger (stdout + optional append-mode log file).

    Explicitly sets ``logger.disabled = False`` to stay active even after Hydra's
    ``hydra/job_logging: disabled`` applies ``disable_existing_loggers: True``.
    """
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.disabled = False
    logger.propagate = False

    if not is_main_process:
        logger.setLevel(logging.CRITICAL + 1)
        logger.addHandler(logging.NullHandler())
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def format_eta(seconds: float) -> str:
    """Format remaining duration in seconds into a compact human-readable string."""
    if not math.isfinite(seconds):
        return "--"
    if seconds <= 0:
        return "0s"

    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_lr(value: float) -> str:
    # Compact scientific notation matching design.md §9.2 (e.g. 1.98e-04 -> 1.98e-4)
    return re.sub(r"e([+-])0([1-9])", r"e\1\2", f"{value:.2e}")


def format_train_log(
    step: int,
    total_iters: int,
    lr_dict: dict[str, float],
    scalars: dict[str, float],
    it_per_sec: float,
    mem_gb: float,
    eta_sec: float,
) -> str:
    """Format a single structured training summary line matching ``docs/design.md`` §9.2."""
    width = max(1, len(str(max(0, total_iters))))
    header = f"[Iter {step:0{width}d}/{total_iters}]"

    parts: list[str] = []
    if len(lr_dict) == 1:
        parts.append(f"lr: {_format_lr(next(iter(lr_dict.values())))}")
    elif len(lr_dict) > 1:
        lr_items = ", ".join(f"{k}={_format_lr(v)}" for k, v in lr_dict.items())
        parts.append(f"lr: {lr_items}")

    ordered_keys: list[str] = []
    if "loss/total" in scalars:
        ordered_keys.append("loss/total")
    for key in scalars:
        if key.startswith("loss/") and key != "loss/total":
            ordered_keys.append(key)
    for key in scalars:
        if not key.startswith("loss/") and key != "grad_norm":
            ordered_keys.append(key)
    if "grad_norm" in scalars:
        ordered_keys.append("grad_norm")

    for key in ordered_keys:
        val = scalars[key]
        if key == "grad_norm" or key.endswith("/grad_norm"):
            parts.append(f"{key}: {val:.2f}")
        else:
            parts.append(f"{key}: {val:.4f}")

    parts.append(f"{it_per_sec:.1f} it/s")
    parts.append(f"mem: {mem_gb:.1f} GB")
    parts.append(f"ETA: {format_eta(eta_sec)}")
    return f"{header} {' | '.join(parts)}"


def _prepare_image_chw(image: torch.Tensor) -> torch.Tensor:
    img = image
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(
            f"Expected image of shape [3, H, W] or [1, 3, H, W], got {tuple(image.shape)}."
        )
    return torch.nan_to_num(
        img.detach().to(device="cpu", dtype=torch.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)


def _chw_to_hwc_u8(img_chw: torch.Tensor) -> Any:
    return (
        img_chw.mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def log_image_dict_to_trackers(
    accelerator: Accelerator,
    images: Mapping[str, torch.Tensor],
    step: int,
) -> None:
    """Log a mapping of ``{tag: RGB image tensor}`` to active Accelerate trackers at ``step``."""
    if not getattr(accelerator, "is_main_process", True):
        return
    if not images:
        return

    trackers = getattr(accelerator, "trackers", None) or []
    if not trackers:
        return

    prepared_chw: dict[str, torch.Tensor] = {
        str(tag): _prepare_image_chw(image) for tag, image in images.items()
    }
    prepared_hwc_u8: dict[str, Any] | None = None

    def _get_all_hwc_u8() -> dict[str, Any]:
        nonlocal prepared_hwc_u8
        if prepared_hwc_u8 is None:
            prepared_hwc_u8 = {
                tag: _chw_to_hwc_u8(img_chw) for tag, img_chw in prepared_chw.items()
            }
        return prepared_hwc_u8

    for tracker in trackers:
        name = str(getattr(tracker, "name", type(tracker).__name__)).lower()
        if name == "tensorboard":
            writer = getattr(tracker, "writer", None) or tracker.tracker
            for tag, img_chw in prepared_chw.items():
                writer.add_image(tag, img_chw, global_step=step, dataformats="CHW")
            writer.flush()
        elif name == "wandb":
            import wandb

            run = tracker.tracker
            payload = {tag: wandb.Image(arr) for tag, arr in _get_all_hwc_u8().items()}
            run.log(payload, step=step)
        elif name == "swanlab":
            import swanlab

            payload = {tag: swanlab.Image(arr) for tag, arr in _get_all_hwc_u8().items()}
            tracker.tracker.log(payload, step=step)
        elif name == "mlflow":
            import mlflow
            from PIL import Image

            for tag, arr in _get_all_hwc_u8().items():
                mlflow.log_image(Image.fromarray(arr), key=tag, step=step)
        else:
            if name not in _WARNED_UNSUPPORTED_TRACKERS:
                _WARNED_UNSUPPORTED_TRACKERS.add(name)
                logging.getLogger("srtrainer").warning(
                    "Image logging is not implemented for tracker %r; skipping image tags %r.",
                    name,
                    list(prepared_chw.keys()),
                )


def log_images_to_trackers(
    accelerator: Accelerator,
    tag: str,
    image: torch.Tensor,
    step: int,
) -> None:
    """Log an RGB image tensor (``[3, H, W]`` or ``[1, 3, H, W]`` in ``[0, 1]``) to active Accelerate trackers."""
    log_image_dict_to_trackers(accelerator, {tag: image}, step=step)
