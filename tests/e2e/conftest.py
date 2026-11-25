"""Shared opaque-box E2E test harness for SRTrainer (Tiers 1-4).

All framework and ML library imports (``srtrainer``, ``torch``, ``numpy``,
``PIL``, ``hydra``, ``omegaconf``, ``accelerate``, ``safetensors``, ``yaml``)
are strictly deferred inside function/fixture bodies so
``python -m pytest --collect-only tests/e2e/`` succeeds cleanly before
implementation milestones M2-M5 exist.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import socket
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

_SAFE_HYDRA_STR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_./-]*$")
_HYDRA_KEY_ESC_RE = re.compile(r"([\\:,{}\[\]=()'\"\s])")
_HYDRA_KEYWORDS = frozenset(
    {
        "no",
        "yes",
        "true",
        "false",
        "null",
        "none",
        "nan",
        "inf",
        "-inf",
        "+inf",
        "infinity",
        "-infinity",
        "+infinity",
    }
)

# ---------------------------------------------------------------------------
# Repository paths & CPU-only VM resource budget constants
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_DIR: Path = REPO_ROOT / "configs"
TRAIN_SCRIPT: Path = REPO_ROOT / "tools" / "train.py"
INFER_SCRIPT: Path = REPO_ROOT / "tools" / "infer.py"

CPU_SCALE: int = 2
CPU_GT_SIZE: int = 32
CPU_BATCH_SIZE: int = 2
CPU_NUM_WORKERS: int = 0
CPU_MIXED_PRECISION: str = "no"
CPU_TOTAL_ITERS: int = 4
CPU_LPIPS_PRETRAINED: bool = False

# Ensure regular package ``tests`` (via ``tests/__init__.py``) resolves to this
# repository ahead of any third-party ``tests`` package in ``site-packages`` so
# Hydra ``_target_`` imports such as ``tests.e2e.custom_metrics`` stay isolated.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Subprocess & 2-process CPU distributed launchers
# ---------------------------------------------------------------------------
def find_free_port() -> int:
    """Allocate an ephemeral unused TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_subprocess_env(
    *,
    distributed_cpu: bool = False,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build deterministic CPU-safe subprocess environment with repo on PYTHONPATH."""
    env = os.environ.copy()
    py_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if distributed_cpu:
        env["ACCELERATE_USE_CPU"] = "true"
        env["MASTER_ADDR"] = "127.0.0.1"
        if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in env:
            env["GLOO_SOCKET_IFNAME"] = "lo0"
    if extra:
        env.update(extra)
    return env


def run_train_cli(
    overrides: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 180,
    check: bool = True,
    env_extra: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``tools/train.py`` as a real subprocess."""
    cmd = [sys.executable, str(TRAIN_SCRIPT), *overrides]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=build_subprocess_env(distributed_cpu=False, extra=env_extra),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"tools/train.py failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def run_infer_cli(
    overrides: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 180,
    check: bool = True,
    env_extra: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``tools/infer.py`` as a real subprocess."""
    cmd = [sys.executable, str(INFER_SCRIPT), *overrides]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=build_subprocess_env(distributed_cpu=False, extra=env_extra),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"tools/infer.py failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def run_torchrun_cpu(
    script_path: Path | str,
    args: Sequence[str],
    *,
    nproc_per_node: int = 2,
    cwd: Path | None = None,
    timeout: int = 180,
    check: bool = True,
    env_extra: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Launch genuine multi-process CPU distributed execution via torchrun.

    Uses ``python -m torch.distributed.run`` with ``ACCELERATE_USE_CPU=true``
    because ``accelerate launch --cpu --num_processes 2`` silently falls back to
    single-process execution on accelerate 1.16.0.
    """
    port = find_free_port()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc_per_node}",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        str(script_path),
        *args,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=build_subprocess_env(distributed_cpu=True, extra=env_extra),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"torchrun failed (rc={proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


# ---------------------------------------------------------------------------
# Hydra composition helper
# ---------------------------------------------------------------------------
def compose_hydra_cfg(
    config_name: str = "train",
    overrides: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> Any:
    """Compose a Hydra config cleanly with guaranteed GlobalHydra cleanup."""
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    cfg_dir = str((config_dir or CONFIGS_DIR).resolve())
    GlobalHydra.instance().clear()
    try:
        with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
            return hydra.compose(
                config_name=config_name,
                overrides=list(overrides or []),
            )
    finally:
        GlobalHydra.instance().clear()


# ---------------------------------------------------------------------------
# Synthetic image & manifest factories (deferred PIL + numpy)
# ---------------------------------------------------------------------------
def _infer_pil_format(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        norm = fmt.strip().upper()
        return "JPEG" if norm == "JPG" else norm
    ext = path.suffix.lower()
    mapping = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
    }
    return mapping.get(ext, "PNG")


def write_synthetic_image(
    path: Path | str,
    width: int,
    height: int,
    mode: str = "RGB",
    pattern: str = "gradient",
    color: tuple[int, ...] | None = None,
    seed: int = 0,
    fmt: str | None = None,
) -> Path:
    """Write a deterministic synthetic test image in RGB/L/RGBA format."""
    import numpy as np
    from PIL import Image

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pat = pattern.lower()

    xs = np.linspace(0.0, 1.0, max(width, 1), dtype=np.float32)
    ys = np.linspace(0.0, 1.0, max(height, 1), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    if pat in ("solid", "constant"):
        if mode == "L":
            val = int(color[0]) if color is not None else int((seed * 53 + 64) % 216 + 20)
            arr = np.full((height, width), val, dtype=np.uint8)
        elif mode == "RGBA":
            if color is not None:
                rgba = color if len(color) == 4 else (*color[:3], 255)
            else:
                rgba = (
                    (seed * 53 + 40) % 216 + 20,
                    (seed * 97 + 80) % 216 + 20,
                    (seed * 131 + 120) % 216 + 20,
                    200,
                )
            arr = np.empty((height, width, 4), dtype=np.uint8)
            for c in range(4):
                arr[..., c] = int(rgba[c])
        else:
            if color is not None:
                rgb = color[:3]
            else:
                rgb = (
                    (seed * 53 + 40) % 216 + 20,
                    (seed * 97 + 80) % 216 + 20,
                    (seed * 131 + 120) % 216 + 20,
                )
            arr = np.empty((height, width, 3), dtype=np.uint8)
            for c in range(3):
                arr[..., c] = int(rgb[c])
    elif pat == "checkerboard":
        block = 4 if min(height, width) >= 8 else 1
        cy = (np.arange(height) // block)[:, None]
        cx = (np.arange(width) // block)[None, :]
        parity = ((cy + cx + seed) % 2).astype(np.float32)
        if mode == "L":
            arr = (30.0 + 195.0 * parity).astype(np.uint8)
        else:
            r = (30.0 + 195.0 * parity).astype(np.uint8)
            g = (225.0 - 180.0 * parity).astype(np.uint8)
            b = np.broadcast_to(
                (60.0 + 140.0 * ((cy + seed) % 2)).astype(np.uint8),
                (height, width),
            )
            if mode == "RGBA":
                a = np.full((height, width), 240, dtype=np.uint8)
                arr = np.stack([r, g, b, a], axis=-1)
            else:
                arr = np.stack([r, g, b], axis=-1)
    elif pat in ("noise", "random"):
        rng = np.random.default_rng(seed)
        if mode == "L":
            arr = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
        elif mode == "RGBA":
            arr = rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)
        else:
            arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    else:
        # Asymmetric multi-channel gradient uniquely oriented under D4 transforms.
        shift = (seed % 7) * 9.0
        r = np.clip(20.0 + 200.0 * xx + 25.0 * yy + shift, 0.0, 255.0).astype(np.uint8)
        g = np.clip(15.0 + 210.0 * yy + 15.0 * (1.0 - xx) + shift * 0.5, 0.0, 255.0).astype(
            np.uint8
        )
        b = np.clip(30.0 + 140.0 * (0.7 * yy + 0.3 * xx) + shift, 0.0, 255.0).astype(np.uint8)
        if mode == "L":
            arr = r
        elif mode == "RGBA":
            a = np.full((height, width), 255, dtype=np.uint8)
            arr = np.stack([r, g, b, a], axis=-1)
        else:
            arr = np.stack([r, g, b], axis=-1)

    img = Image.fromarray(arr, mode=mode)
    pil_fmt = _infer_pil_format(out_path, fmt)
    if pil_fmt == "JPEG" and img.mode == "RGBA":
        img = img.convert("RGB")
    save_kwargs: dict[str, Any] = {}
    if pil_fmt == "JPEG":
        save_kwargs["quality"] = 95
    img.save(out_path, format=pil_fmt, **save_kwargs)
    return out_path


def make_aligned_paired_images(
    gt_path: Path | str,
    lq_path: Path | str,
    lq_h: int,
    lq_w: int,
    scale: int,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Create deterministic asymmetric paired PNGs where GT == nearest_repeat(LQ, scale)."""
    import numpy as np
    from PIL import Image

    gt_p = Path(gt_path)
    lq_p = Path(lq_path)
    gt_p.parent.mkdir(parents=True, exist_ok=True)
    lq_p.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, max(lq_w, 1), dtype=np.float32)
    ys = np.linspace(0.0, 1.0, max(lq_h, 1), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    base_r = 25.0 + 170.0 * xx + 30.0 * yy
    base_g = 25.0 + 170.0 * yy + 20.0 * (1.0 - xx)
    base_b = 30.0 + 150.0 * (0.65 * xx + 0.35 * yy)
    jitter = rng.integers(-15, 16, size=(lq_h, lq_w, 3))
    lq_arr = np.clip(np.stack([base_r, base_g, base_b], axis=-1) + jitter, 0, 255).astype(np.uint8)
    gt_arr = np.repeat(np.repeat(lq_arr, scale, axis=0), scale, axis=1)

    Image.fromarray(lq_arr, mode="RGB").save(lq_p, format=_infer_pil_format(lq_p, None))
    Image.fromarray(gt_arr, mode="RGB").save(gt_p, format=_infer_pil_format(gt_p, None))
    return gt_p, lq_p


# ---------------------------------------------------------------------------
# Opaque-box signature adapters & config builders
# ---------------------------------------------------------------------------
def get_toy_model_arch_dict(
    scale: int = CPU_SCALE,
    *,
    num_feat: int = 8,
    num_block: int = 1,
) -> dict[str, Any]:
    """Build a compact Hydra ``_target_`` dict for test-only ``DummySRNet``."""
    from tests.dummy_arch import DummySRNet

    params = inspect.signature(DummySRNet.__init__).parameters
    arch: dict[str, Any] = {"_target_": "tests.dummy_arch.DummySRNet"}
    for key, val in (
        ("num_in_ch", 3),
        ("num_out_ch", 3),
        ("num_feat", num_feat),
        ("num_block", num_block),
        ("scale", scale),
    ):
        if key in params:
            arch[key] = val
    return arch


def build_mixed_dataset(
    sources: Sequence[Mapping[str, Any]],
    *,
    gt_size: int = CPU_GT_SIZE,
    scale: int = CPU_SCALE,
    augment: Mapping[str, Any] | None = None,
    has_degradation: bool = True,
    kernel_cfg: Mapping[str, Any] | None = None,
) -> Any:
    """Opaque-box adapter instantiating ``MixedSRDataset`` across public signature variants."""
    from data import MixedSRDataset

    params = inspect.signature(MixedSRDataset.__init__).parameters
    kwargs: dict[str, Any] = {"sources": [dict(s) for s in sources]}
    if "gt_size" in params:
        kwargs["gt_size"] = gt_size
    if "scale" in params:
        kwargs["scale"] = scale
    if augment is not None:
        if "augment" in params:
            kwargs["augment"] = dict(augment)
        else:
            for aug_key in ("hflip", "vflip", "rot90"):
                if aug_key not in augment:
                    continue
                if aug_key in params:
                    kwargs[aug_key] = augment[aug_key]
                elif f"use_{aug_key}" in params:
                    kwargs[f"use_{aug_key}"] = augment[aug_key]

    if "has_degradation" in params:
        kwargs["has_degradation"] = has_degradation
        if "kernel_cfg" in params and kernel_cfg is not None:
            kwargs["kernel_cfg"] = dict(kernel_cfg)
    elif "use_degradation" in params:
        kwargs["use_degradation"] = has_degradation
        if "kernel_cfg" in params and kernel_cfg is not None:
            kwargs["kernel_cfg"] = dict(kernel_cfg)
    elif "degradation_cfg" in params:
        if not has_degradation:
            kwargs["degradation_cfg"] = None
        else:
            deg: dict[str, Any] = {"scale": scale}
            if kernel_cfg is not None:
                deg["kernel_cfg"] = dict(kernel_cfg)
            kwargs["degradation_cfg"] = deg
    elif "degradation" in params:
        if not has_degradation:
            kwargs["degradation"] = None
        else:
            deg = {"scale": scale}
            if kernel_cfg is not None:
                deg["kernel_cfg"] = dict(kernel_cfg)
            kwargs["degradation"] = deg
    elif "kernel_cfg" in params:
        kwargs["kernel_cfg"] = None if not has_degradation else dict(kernel_cfg or {})

    return MixedSRDataset(**kwargs)


def build_weighted_sampler(
    source_sizes: Sequence[int],
    ratios: Sequence[float],
    *,
    batch_size_per_gpu: int = 4,
    world_size: int = 1,
    rank: int = 0,
    base_seed: int = 42,
    start_micro_step: int = 0,
) -> Any:
    """Opaque-box adapter instantiating ``DistributedWeightedInfiniteSampler``."""
    from data import DistributedWeightedInfiniteSampler

    params = inspect.signature(DistributedWeightedInfiniteSampler.__init__).parameters
    kwargs: dict[str, Any] = {}

    if "source_sizes" in params:
        kwargs["source_sizes"] = list(source_sizes)
    elif "dataset_sizes" in params:
        kwargs["dataset_sizes"] = list(source_sizes)
    else:
        kwargs["sizes"] = list(source_sizes)

    if "ratios" in params:
        kwargs["ratios"] = list(ratios)
    else:
        kwargs["weights"] = list(ratios)

    if "batch_size_per_gpu" in params:
        kwargs["batch_size_per_gpu"] = batch_size_per_gpu
    elif "batch_size" in params:
        kwargs["batch_size"] = batch_size_per_gpu

    if "world_size" in params:
        kwargs["world_size"] = world_size
    elif "num_replicas" in params:
        kwargs["num_replicas"] = world_size

    if "rank" in params:
        kwargs["rank"] = rank

    if "base_seed" in params:
        kwargs["base_seed"] = base_seed
    elif "seed" in params:
        kwargs["seed"] = base_seed

    if "start_micro_step" in params:
        kwargs["start_micro_step"] = start_micro_step

    sampler = DistributedWeightedInfiniteSampler(**kwargs)
    if start_micro_step != 0 and "start_micro_step" not in params:
        sampler.set_start_micro_step(start_micro_step)
    return sampler


def call_validate_sources_config(
    sources: Sequence[Mapping[str, Any]],
    *,
    has_degradation: bool = True,
) -> Any:
    """Invoke data.validate_sources_config adapting to its public signature."""
    from data import validate_sources_config

    params = inspect.signature(validate_sources_config).parameters
    if "has_degradation" in params:
        return validate_sources_config(sources, has_degradation=has_degradation)
    positional = [
        p
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    ]
    if len(positional) >= 2:
        return validate_sources_config(sources, has_degradation)
    return validate_sources_config(sources)


def build_constant_bias_state_dict(
    net: Any,
    target_val: float,
) -> tuple[dict[str, Any], float]:
    """Build a valid state_dict whose forward pass on zeros yields a distinct constant mean.

    Works regardless of internal layer names or residual/interpolation skip
    connections by probing on an all-zero input tensor (where any linear/bicubic
    skip branch evaluates to 0) and measuring the exact clamped output mean.
    """
    import torch

    base_sd = {
        k: torch.zeros_like(v) if torch.is_floating_point(v) else v.clone()
        for k, v in net.state_dict().items()
    }
    bias_keys = [
        k for k, v in base_sd.items() if k.endswith(".bias") and torch.is_floating_point(v)
    ]
    probe_in = torch.zeros(1, 3, 16, 16, dtype=torch.float32)

    chosen_sd = base_sd
    observed_mean = 0.0
    for bkey in reversed(bias_keys):
        cand = {k: v.clone() for k, v in base_sd.items()}
        cand[bkey].fill_(float(target_val))
        net.load_state_dict(cand, strict=True)
        net.eval()
        with torch.no_grad():
            probe = net(probe_in).clamp(0.0, 1.0)
            mean_val = float(probe.mean().item())
        if abs(mean_val - float(target_val)) < 0.05:
            return cand, mean_val
        chosen_sd, observed_mean = cand, mean_val

    if not bias_keys:
        weight_keys = [
            k for k, v in base_sd.items() if k.endswith(".weight") and torch.is_floating_point(v)
        ]
        if weight_keys:
            chosen_sd = {k: v.clone() for k, v in base_sd.items()}
            chosen_sd[weight_keys[-1]].fill_(float(target_val))
            net.load_state_dict(chosen_sd, strict=True)
            net.eval()
            with torch.no_grad():
                observed_mean = float(net(probe_in).clamp(0.0, 1.0).mean().item())
    return chosen_sd, observed_mean


_build_constant_bias_state_dict = build_constant_bias_state_dict


def make_bias_tagged_exp_dir(
    exp_dir: Path | str,
    *,
    scale: int = CPU_SCALE,
    gt_size: int = CPU_GT_SIZE,
    include_best: bool = True,
    include_ema: bool = True,
    best_val_name: str = "set5",
    best_metric_name: str = "psnr",
) -> dict[str, Any]:
    """Create a synthetic ToyModel experiment directory with bias-tagged safetensors weights.

    Produces distinct deterministic outputs on solid-black inputs so Mode 1
    weight-selection priority (``best_*`` -> ``latest_ema`` -> ``latest`` ->
    ``weights_path`` override) is verified strictly via observable output pixels.
    """
    import hydra.utils
    from omegaconf import OmegaConf

    from utils.config import save_resolved_config
    from utils.misc import save_safetensors

    exp_path = Path(exp_dir)
    weights_dir = exp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    arch_dict = get_toy_model_arch_dict(scale=scale)
    cfg = compose_hydra_cfg("train", overrides=["experiment=vosr2_onestep_text"])
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.scale = scale
    cfg.exp_name = exp_path.name
    cfg.output_root = str(exp_path.parent)
    cfg.work_dir = str(exp_path)
    cfg.arch = OmegaConf.create(arch_dict)
    cfg.data.scale = scale
    cfg.data.gt_size = gt_size
    cfg.data.degradation = None
    cfg.metric.primary_dataset = best_val_name
    cfg.metric.primary_metric = best_metric_name
    cfg.trainer.mixed_precision = CPU_MIXED_PRECISION

    config_path = exp_path / "config_resolved.yaml"
    save_resolved_config(cfg, config_path)

    net = hydra.utils.instantiate(cfg.arch)
    targets = {
        "best": 0.80,
        "latest_ema": 0.50,
        "latest": 0.20,
        "override": 0.95,
    }
    expected_means: dict[str, float] = {}
    saved_paths: dict[str, Path] = {}

    sd_latest, mean_latest = build_constant_bias_state_dict(net, targets["latest"])
    saved_paths["latest"] = save_safetensors(sd_latest, weights_dir / "latest.safetensors")
    expected_means["latest"] = mean_latest

    if include_ema:
        sd_ema, mean_ema = build_constant_bias_state_dict(net, targets["latest_ema"])
        saved_paths["latest_ema"] = save_safetensors(sd_ema, weights_dir / "latest_ema.safetensors")
        expected_means["latest_ema"] = mean_ema

    if include_best:
        sd_best, mean_best = build_constant_bias_state_dict(net, targets["best"])
        best_file = weights_dir / f"best_{best_val_name}_{best_metric_name}.safetensors"
        saved_paths["best"] = save_safetensors(sd_best, best_file)
        expected_means["best"] = mean_best

    sd_override, mean_override = build_constant_bias_state_dict(net, targets["override"])
    saved_paths["override"] = save_safetensors(
        sd_override, exp_path / "override_weights.safetensors"
    )
    expected_means["override"] = mean_override

    return {
        "exp_dir": exp_path,
        "config_path": config_path,
        "weights_dir": weights_dir,
        "paths": saved_paths,
        "expected_means": expected_means,
    }


def _escape_hydra_key(key: Any) -> str:
    return _HYDRA_KEY_ESC_RE.sub(r"\\\1", str(key))


def fmt_hydra_value(val: Any) -> str:
    """Format a Python scalar/list/mapping cleanly for Hydra CLI overrides (no shell)."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, Mapping):
        inner = ",".join(f"{_escape_hydra_key(k)}:{fmt_hydra_value(v)}" for k, v in val.items())
        return "{" + inner + "}"
    if isinstance(val, (list, tuple)):
        inner = ",".join(fmt_hydra_value(item) for item in val)
        return f"[{inner}]"
    text = str(val)
    if not _SAFE_HYDRA_STR_RE.fullmatch(text) or text.lower() in _HYDRA_KEYWORDS:
        escaped = re.sub(r"(\\+)(?='|$)", r"\1\1", text).replace("'", r"\'")
        return f"'{escaped}'"
    return text


_fmt_hydra_value = fmt_hydra_value


def build_standalone_infer_arch_overrides(
    arch_dict: Mapping[str, Any],
    *,
    scale: int = CPU_SCALE,
    config_dir: Path | None = None,
) -> list[str]:
    """Build Hydra CLI overrides replacing ``arch`` cleanly for ``tools/infer.py`` Mode 2.

    Works across all valid ``configs/infer.yaml`` designs: defaults-group
    ``arch: toy_x4``, defaults-group ``arch: null``, plain key ``arch: null``,
    inline dict ``arch: {...}``, or absent ``arch``.
    """
    cfg_infer = compose_hydra_cfg("infer", config_dir=config_dir)
    overrides: list[str] = [f"++scale={scale}"]

    can_delete_arch_group = False
    if cfg_infer.get("arch") is not None:
        try:
            compose_hydra_cfg("infer", overrides=["~arch"], config_dir=config_dir)
            can_delete_arch_group = True
        except Exception:
            can_delete_arch_group = False

    if can_delete_arch_group:
        overrides.append("~arch")
        for key, val in arch_dict.items():
            overrides.append(f"+arch.{key}={fmt_hydra_value(val)}")
    else:
        existing_arch = cfg_infer.get("arch")
        if isinstance(existing_arch, Mapping):
            for old_key in existing_arch:
                if old_key not in arch_dict:
                    overrides.append(f"~arch.{old_key}")
        overrides.append(f"++arch={fmt_hydra_value(dict(arch_dict))}")
    return overrides


def make_minimal_train_cfg(
    work_dir: Path | str,
    train_sources: Sequence[Mapping[str, Any]],
    val_suites: Mapping[str, Any] | None = None,
    *,
    primary_dataset: str | None = None,
    primary_metric: str | None = None,
    scale: int = CPU_SCALE,
    gt_size: int = CPU_GT_SIZE,
    batch_size: int = CPU_BATCH_SIZE,
    total_iters: int = CPU_TOTAL_ITERS,
    log_interval: int = 2,
    val_interval: int = 2,
    save_interval: int = 2,
    max_keep_states: int = 3,
    grad_accum: int = 1,
    log_with: list[str] | str | None = None,
    use_ema: bool = True,
    has_degradation: bool = True,
    arch_cfg: Mapping[str, Any] | None = None,
    loss_cfg: Mapping[str, Any] | None = None,
    resume: str | None = None,
    append_timestamp: bool = False,
    val_save_visuals: bool = True,
    val_max_save_images: int = 2,
    val_use_ema: bool = True,
) -> Any:
    """Build a compact CPU-safe OmegaConf DictConfig for in-process ``Trainer``/``VOSRModel`` tests."""
    from omegaconf import OmegaConf
    from tests.dummy_arch import get_dummy_vosr_arch_dict

    work_path = Path(work_dir)
    cfg = compose_hydra_cfg("train", overrides=["experiment=vosr2_onestep_text"])
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

    cfg.scale = scale
    cfg.exp_name = work_path.name
    cfg.output_root = str(work_path.parent)
    cfg.work_dir = str(work_path)

    cfg.arch = OmegaConf.create(
        dict(arch_cfg) if arch_cfg is not None else get_dummy_vosr_arch_dict()
    )
    if loss_cfg is not None:
        cfg.loss = OmegaConf.create(dict(loss_cfg))
    else:
        cfg.loss = OmegaConf.create(
            {
                "l1": {
                    "weight": 1.0,
                    "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
                }
            }
        )

    cfg.model.scale = scale
    cfg.model.trainable = OmegaConf.create({"vae": False, "dit": True, "dino": False})
    cfg.model.pretrain = OmegaConf.create(
        {
            "vae": {"path": None, "strict": True, "key": None},
            "dit": {"path": None, "strict": True, "key": None},
            "dino": {"path": None, "strict": True, "key": None},
        }
    )
    cfg.model.ema.enabled = bool(use_ema)

    cfg.data.scale = scale
    cfg.data.gt_size = gt_size
    cfg.data.batch_size_per_gpu = batch_size
    cfg.data.num_workers = CPU_NUM_WORKERS
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.sources = OmegaConf.create([dict(s) for s in train_sources])
    if not has_degradation:
        cfg.data.degradation = None
    elif cfg.data.get("degradation") is not None:
        cfg.data.degradation.scale = scale

    cfg.metric.val_suites = OmegaConf.create(dict(val_suites or {}))
    cfg.metric.primary_dataset = primary_dataset
    cfg.metric.primary_metric = primary_metric

    cfg.trainer.total_iters = total_iters
    cfg.trainer.log_interval = log_interval
    cfg.trainer.val_interval = val_interval
    cfg.trainer.save_interval = save_interval
    cfg.trainer.max_keep_states = max_keep_states
    cfg.trainer.mixed_precision = CPU_MIXED_PRECISION
    cfg.trainer.gradient_accumulation_steps = grad_accum
    cfg.trainer.compile = False
    cfg.trainer.log_with = (
        []
        if log_with is None
        else ([log_with] if isinstance(log_with, str) else list(log_with))
    )
    cfg.trainer.resume = resume
    cfg.trainer.append_timestamp = append_timestamp
    cfg.trainer.val_save_visuals = val_save_visuals
    cfg.trainer.val_max_save_images = val_max_save_images
    cfg.trainer.val_use_ema = val_use_ema

    tile_size = gt_size // scale
    safe_overlap = min(4, max(0, (tile_size // 2) - 1))
    if cfg.trainer.get("val_tile") is not None:
        cfg.trainer.val_tile.overlap = safe_overlap

    return cfg


def make_minimal_train_overrides(
    output_root: Path | str,
    exp_name: str,
    train_sources: Sequence[Mapping[str, Any]],
    val_suites: Mapping[str, Any] | None = None,
    *,
    primary_dataset: str | None = None,
    primary_metric: str | None = None,
    scale: int = CPU_SCALE,
    gt_size: int = CPU_GT_SIZE,
    batch_size: int = CPU_BATCH_SIZE,
    total_iters: int = 2,
    log_interval: int = 1,
    val_interval: int = 2,
    save_interval: int = 2,
    max_keep_states: int = 3,
    grad_accum: int = 1,
    log_with: list[str] | str | None = None,
    use_ema: bool = True,
    has_degradation: bool = False,
    arch_num_feat: int = 8,
    arch_num_block: int = 1,
    experiment: str | None = None,
    extra_overrides: Sequence[str] | None = None,
) -> list[str]:
    """Build compact CPU-safe VOSR Hydra CLI overrides for ``tools/train.py`` runs."""
    from tests.dummy_arch import get_dummy_vosr_arch_dict

    del arch_num_feat, arch_num_block
    tile_size = gt_size // scale
    safe_overlap = min(4, max(0, (tile_size // 2) - 1))

    exp_name_cfg = experiment if experiment is not None else "vosr2_onestep_text"
    has_loss_override = any(
        ov.startswith(("loss=", "~loss", "+loss=")) for ov in (extra_overrides or ())
    )
    overrides: list[str] = [f"experiment={fmt_hydra_value(exp_name_cfg)}"]
    if not has_loss_override:
        overrides.append("loss=l1")

    normalized_log_with = (
        []
        if log_with is None
        else ([log_with] if isinstance(log_with, str) else list(log_with))
    )
    overrides.extend(
        [
            f"scale={scale}",
            f"output_root={fmt_hydra_value(str(output_root))}",
            f"exp_name={fmt_hydra_value(exp_name)}",
            "~arch/vae",
            "~arch/dit",
            "~arch/dino",
            f"+arch={fmt_hydra_value(get_dummy_vosr_arch_dict())}",
            f"data.gt_size={gt_size}",
            f"data.batch_size_per_gpu={batch_size}",
            f"data.num_workers={CPU_NUM_WORKERS}",
            "data.pin_memory=false",
            "data.persistent_workers=false",
            f"data.sources={fmt_hydra_value([dict(src) for src in train_sources])}",
            f"model.ema.enabled={'true' if use_ema else 'false'}",
            f"trainer.total_iters={total_iters}",
            f"trainer.log_interval={log_interval}",
            f"trainer.val_interval={val_interval}",
            f"trainer.save_interval={save_interval}",
            f"trainer.max_keep_states={max_keep_states}",
            f"trainer.gradient_accumulation_steps={grad_accum}",
            "trainer.mixed_precision='no'",
            "trainer.compile=false",
            "++trainer.append_timestamp=false",
            f"++trainer.val_tile.overlap={safe_overlap}",
            f"trainer.log_with={fmt_hydra_value(normalized_log_with)}",
        ]
    )

    if not has_degradation:
        overrides.append("data.degradation=null")

    overrides.append("metric=base_metric")
    if not val_suites:
        overrides.append("metric.primary_dataset=null")
        overrides.append("metric.primary_metric=null")
    else:
        serialized_suites: dict[str, Any] = {}
        for suite_name, suite_cfg in val_suites.items():
            ds = suite_cfg["dataset"]
            entry: dict[str, Any] = {
                "dataset": {
                    "root_dir": str(ds["root_dir"]),
                    "json_path": str(ds["json_path"]),
                },
                "metrics": {
                    m_name: dict(m_cfg or {})
                    for m_name, m_cfg in (suite_cfg.get("metrics") or {}).items()
                },
            }
            if "max_save_images" in suite_cfg:
                entry["max_save_images"] = suite_cfg["max_save_images"]
            serialized_suites[suite_name] = entry
        overrides.append(f"++metric.val_suites={fmt_hydra_value(serialized_suites)}")
        overrides.append(f"metric.primary_dataset={fmt_hydra_value(primary_dataset)}")
        overrides.append(f"metric.primary_metric={fmt_hydra_value(primary_metric)}")

    if extra_overrides:
        overrides.extend(extra_overrides)
    return overrides


# ---------------------------------------------------------------------------
# Core pytest fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture
def e2e_workdir(tmp_path: Path) -> dict[str, Path]:
    """Provide isolated temp subdirectories so no test writes into repo output dirs."""
    work = tmp_path / "work"
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    for directory in (work, data, outputs):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "root": tmp_path,
        "work": work,
        "data": data,
        "outputs": outputs,
    }


def reset_accelerate_singleton() -> None:
    """Reset Accelerate singleton and torch.distributed state in-process (deferred imports)."""
    try:
        from accelerate.state import AcceleratorState, GradientState

        AcceleratorState._reset_state(reset_partial_state=True)
        GradientState._reset_state()
    except ImportError:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def reset_accelerate_state() -> Iterator[None]:
    """Reset Accelerate singleton and torch.distributed state around every test."""
    reset_accelerate_singleton()
    try:
        yield
    finally:
        reset_accelerate_singleton()


@pytest.fixture
def accelerate_state_resetter() -> Callable[[], None]:
    """Expose explicit Accelerate singleton reset callable for multi-Accelerator tests."""
    return reset_accelerate_singleton


@pytest.fixture
def isolated_logger_name() -> Iterator[str]:
    """Yield a unique logger name and close/remove all handlers on teardown."""
    name = f"srtrainer.e2e.{uuid.uuid4().hex[:10]}"
    try:
        yield name
    finally:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


@pytest.fixture
def hydra_compose() -> Callable[..., Any]:
    return compose_hydra_cfg


@pytest.fixture
def synthetic_image_factory() -> Callable[..., Path]:
    return write_synthetic_image


@pytest.fixture
def aligned_paired_images_factory() -> Callable[..., tuple[Path, Path]]:
    return make_aligned_paired_images


# ---------------------------------------------------------------------------
# Synthetic train & validation manifest factory fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def paired_train_source_factory(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Create a pure-paired training source dict (strictly NO ``mode`` key)."""

    def _make(
        name: str = "paired_src",
        *,
        n_samples: int = 4,
        lq_h: int = 24,
        lq_w: int = 24,
        scale: int = CPU_SCALE,
        ratio: float = 1.0,
        subdir: str | None = None,
    ) -> dict[str, Any]:
        root = tmp_path / (subdir or f"train_{name}")
        entries: list[dict[str, str]] = []
        for idx in range(n_samples):
            gt_rel = f"hr/{idx:04d}.png"
            lq_rel = f"lr/{idx:04d}.png"
            make_aligned_paired_images(
                root / gt_rel,
                root / lq_rel,
                lq_h=lq_h,
                lq_w=lq_w,
                scale=scale,
                seed=idx + 10,
            )
            entries.append({"gt": gt_rel, "lq": lq_rel})
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return {
            "name": name,
            "root_dir": str(root),
            "json_path": str(manifest),
            "ratio": ratio,
        }

    return _make


@pytest.fixture
def gt_only_train_source_factory(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Create a pure gt_only training source dict: ``[{"gt": ...}]`` (NO ``lq``, NO ``mode``)."""

    def _make(
        name: str = "gt_only_src",
        *,
        n_samples: int = 4,
        gt_h: int = 48,
        gt_w: int = 48,
        ratio: float = 1.0,
        pattern: str = "checkerboard",
        subdir: str | None = None,
    ) -> dict[str, Any]:
        root = tmp_path / (subdir or f"train_{name}")
        entries: list[dict[str, str]] = []
        for idx in range(n_samples):
            gt_rel = f"hr/{idx:04d}.png"
            write_synthetic_image(
                root / gt_rel,
                width=gt_w,
                height=gt_h,
                pattern=pattern,
                seed=idx + 100,
            )
            entries.append({"gt": gt_rel})
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return {
            "name": name,
            "root_dir": str(root),
            "json_path": str(manifest),
            "ratio": ratio,
        }

    return _make


@pytest.fixture
def mixed_single_json_train_source_factory(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Create a single JSON manifest mixing paired + gt_only (absent/null/'' lq) & formats."""

    def _make(
        name: str = "mixed_single",
        *,
        lq_h: int = 24,
        lq_w: int = 24,
        scale: int = CPU_SCALE,
        ratio: float = 1.0,
        subdir: str | None = None,
    ) -> dict[str, Any]:
        root = tmp_path / (subdir or f"train_{name}")
        gt_h, gt_w = lq_h * scale, lq_w * scale

        # 0: paired PNG
        make_aligned_paired_images(
            root / "hr/0000.png", root / "lr/0000.png", lq_h=lq_h, lq_w=lq_w, scale=scale, seed=1
        )
        # 1: gt_only PNG (absent lq)
        write_synthetic_image(root / "hr/0001.png", width=gt_w, height=gt_h, seed=2)
        # 2: gt_only PNG (null lq)
        write_synthetic_image(root / "hr/0002.png", width=gt_w, height=gt_h, seed=3)
        # 3: gt_only PNG (empty string lq)
        write_synthetic_image(root / "hr/0003.png", width=gt_w, height=gt_h, seed=4)
        # 4: paired JPEG
        make_aligned_paired_images(
            root / "hr/0004.jpg", root / "lr/0004.jpg", lq_h=lq_h, lq_w=lq_w, scale=scale, seed=5
        )
        # 5: gt_only WebP
        write_synthetic_image(root / "hr/0005.webp", width=gt_w, height=gt_h, seed=6)

        entries: list[dict[str, Any]] = [
            {"gt": "hr/0000.png", "lq": "lr/0000.png"},
            {"gt": "hr/0001.png"},
            {"gt": "hr/0002.png", "lq": None},
            {"gt": "hr/0003.png", "lq": ""},
            {"gt": "hr/0004.jpg", "lq": "lr/0004.jpg"},
            {"gt": "hr/0005.webp"},
        ]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return {
            "source": {
                "name": name,
                "root_dir": str(root),
                "json_path": str(manifest),
                "ratio": ratio,
            },
            "expected_need_degradation": [False, True, True, True, False, True],
            "entries": entries,
        }

    return _make


@pytest.fixture
def val_suite_gt_label_factory(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Create a variable-resolution paired validation suite with ``label`` and nested metadata."""

    def _make(
        name: str = "set5",
        *,
        scale: int = CPU_SCALE,
        sizes: Sequence[tuple[int, int]] | None = None,
        subdir: str | None = None,
    ) -> dict[str, Any]:
        root = tmp_path / (subdir or f"val_{name}")
        lq_sizes = list(sizes) if sizes is not None else [(16, 20), (20, 16), (18, 18)]
        labels = ["HELLO 2026", "SRTRAINER", "OCR_TEST_03"]
        entries: list[dict[str, Any]] = []
        for idx, (lq_h, lq_w) in enumerate(lq_sizes):
            gt_rel = f"hr/{idx:04d}.png"
            lq_rel = f"lr/{idx:04d}.png"
            make_aligned_paired_images(
                root / gt_rel,
                root / lq_rel,
                lq_h=lq_h,
                lq_w=lq_w,
                scale=scale,
                seed=idx + 200,
            )
            entries.append(
                {
                    "lq": lq_rel,
                    "gt": gt_rel,
                    "label": labels[idx % len(labels)],
                    "nested": {"bbox": [idx, idx + 1, 10, 12], "tags": ["sr", f"img_{idx}"]},
                }
            )
        manifest = root / "val_manifest.json"
        manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return {
            "root_dir": str(root),
            "json_path": str(manifest),
            "entries": entries,
            "lq_sizes": lq_sizes,
        }

    return _make


@pytest.fixture
def val_suite_gt_less_meta_factory(tmp_path: Path) -> Callable[..., dict[str, Any]]:
    """Create a GT-less validation suite covering absent/null/'' gt + rich nested metadata."""

    def _make(
        name: str = "nogt_ocr",
        *,
        subdir: str | None = None,
    ) -> dict[str, Any]:
        root = tmp_path / (subdir or f"val_{name}")
        lq_sizes = [(16, 20), (20, 16), (18, 18)]
        entries: list[dict[str, Any]] = [
            {
                "lq": "lr/0000.png",
                "label": "HELLO 2026",
                "flag": False,
                "count": 0,
                "empty_list": [],
                "empty_dict": {},
                "nullable": None,
            },
            {
                "lq": "lr/0001.png",
                "gt": None,
                "label": "中文标签2026",
                "nested": {"domain": "doc", "scores": [1, 2.5]},
            },
            {
                "lq": "lr/0002.png",
                "gt": "",
                "label": "SRTRAINER_OCR",
                "nested": {"active": True, "meta_note": None},
            },
        ]
        for idx, (lq_h, lq_w) in enumerate(lq_sizes):
            write_synthetic_image(
                root / entries[idx]["lq"],
                width=lq_w,
                height=lq_h,
                pattern="gradient",
                seed=idx + 300,
            )
        manifest = root / "val_nogt.json"
        manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "root_dir": str(root),
            "json_path": str(manifest),
            "entries": entries,
            "lq_sizes": lq_sizes,
        }

    return _make


# ---------------------------------------------------------------------------
# Exposed adapter & runner fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def hydra_value_formatter() -> Callable[[Any], str]:
    return fmt_hydra_value


@pytest.fixture
def constant_bias_state_dict_factory() -> Callable[..., tuple[dict[str, Any], float]]:
    return build_constant_bias_state_dict


@pytest.fixture
def mixed_dataset_factory() -> Callable[..., Any]:
    return build_mixed_dataset


@pytest.fixture
def weighted_sampler_factory() -> Callable[..., Any]:
    return build_weighted_sampler


@pytest.fixture
def validate_sources_caller() -> Callable[..., Any]:
    return call_validate_sources_config


@pytest.fixture
def bias_tagged_exp_dir_factory() -> Callable[..., dict[str, Any]]:
    return make_bias_tagged_exp_dir


@pytest.fixture
def minimal_train_cfg_factory() -> Callable[..., Any]:
    return make_minimal_train_cfg


@pytest.fixture
def minimal_train_overrides_factory() -> Callable[..., list[str]]:
    return make_minimal_train_overrides


@pytest.fixture
def standalone_infer_arch_overrides_factory() -> Callable[..., list[str]]:
    return build_standalone_infer_arch_overrides


@pytest.fixture
def train_cli_runner() -> Callable[..., subprocess.CompletedProcess[str]]:
    return run_train_cli


@pytest.fixture
def infer_cli_runner() -> Callable[..., subprocess.CompletedProcess[str]]:
    return run_infer_cli


@pytest.fixture
def torchrun_cpu_runner() -> Callable[..., subprocess.CompletedProcess[str]]:
    return run_torchrun_cpu
