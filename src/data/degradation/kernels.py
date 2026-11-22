"""CPU blur and circular-sinc kernel generation for 2nd-order Real-ESRGAN degradation."""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from scipy import special

__all__ = [
    "VALID_KERNEL_TYPES",
    "circular_lowpass_kernel",
    "random_mixed_kernels",
    "KernelGenerator",
]

VALID_KERNEL_TYPES: tuple[str, ...] = (
    "iso",
    "aniso",
    "generalized_iso",
    "generalized_aniso",
    "plateau_iso",
    "plateau_aniso",
)

_DEFAULT_KERNEL_LIST: tuple[str, ...] = VALID_KERNEL_TYPES
_DEFAULT_KERNEL_PROB: tuple[float, ...] = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)


def _normalize_omegaconf(obj: Any, context: str) -> Any:
    if OmegaConf.is_config(obj):
        try:
            return OmegaConf.to_container(obj, resolve=True, throw_on_missing=True)
        except OmegaConfBaseException as exc:
            raise ValueError(f"Invalid OmegaConf value for {context}: {exc}") from exc
    return obj


def _validate_odd_kernel_size(size: Any, name: str = "kernel_size") -> int:
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError(f"{name} must be a positive odd integer, got {size!r}.")
    size_i = int(size)
    if size_i < 1 or size_i % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer, got {size_i}.")
    return size_i


def _to_1d_list(obj: Any, context: str) -> list[Any]:
    normalized = _normalize_omegaconf(obj, context)
    if isinstance(normalized, np.ndarray):
        if normalized.ndim != 1:
            raise ValueError(
                f"{context} must be a 1D sequence, got ndarray with shape {normalized.shape}."
            )
        return normalized.tolist()
    if isinstance(normalized, (str, bytes, Mapping)) or not isinstance(normalized, Sequence):
        raise ValueError(f"{context} must be a sequence, got {type(obj).__name__}.")
    return list(normalized)


def _validate_range(
    rng_pair: Sequence[float],
    name: str,
    positive_only: bool = True,
) -> tuple[float, float]:
    normalized = _to_1d_list(rng_pair, name)
    if len(normalized) != 2:
        raise ValueError(f"{name} must have length 2, got length {len(normalized)}.")

    vals: list[float] = []
    for idx, v in enumerate(normalized):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"{name}[{idx}] must be a real number, got {v!r}.")
        vf = float(v)
        if not math.isfinite(vf):
            raise ValueError(f"{name}[{idx}] must be finite, got {v!r}.")
        vals.append(vf)

    lo, hi = vals[0], vals[1]
    if lo > hi:
        raise ValueError(f"{name} requires low <= high, got ({lo}, {hi}).")
    if positive_only and lo <= 0.0:
        raise ValueError(f"{name} requires strictly positive bounds, got ({lo}, {hi}).")
    return lo, hi


def _validate_scalar_prob(prob: Any, name: str) -> float:
    if isinstance(prob, bool) or not isinstance(prob, numbers.Real):
        raise ValueError(f"{name} must be a float in [0.0, 1.0], got {prob!r}.")
    prob_f = float(prob)
    if not math.isfinite(prob_f) or prob_f < 0.0 or prob_f > 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0], got {prob!r}.")
    return prob_f


def _validate_kernel_types_and_probs(
    kernel_list: Sequence[str],
    kernel_prob: Sequence[float],
    list_name: str = "kernel_list",
    prob_name: str = "kernel_prob",
) -> tuple[tuple[str, ...], np.ndarray]:
    norm_list = _to_1d_list(kernel_list, list_name)
    norm_prob = _to_1d_list(kernel_prob, prob_name)

    if len(norm_list) == 0:
        raise ValueError(f"{list_name} must not be empty.")

    clean_types: list[str] = []
    for item in norm_list:
        if not isinstance(item, str) or item not in VALID_KERNEL_TYPES:
            raise ValueError(
                f"Unsupported kernel type {item!r} in {list_name}; "
                f"expected one of {VALID_KERNEL_TYPES}."
            )
        clean_types.append(item)

    if len(norm_prob) != len(clean_types):
        raise ValueError(
            f"Length of {prob_name} ({len(norm_prob)}) must match {list_name} ({len(clean_types)})."
        )

    clean_probs: list[float] = []
    for idx, p in enumerate(norm_prob):
        if isinstance(p, bool) or not isinstance(p, numbers.Real):
            raise ValueError(f"{prob_name}[{idx}] must be a real number, got {p!r}.")
        pf = float(p)
        if not math.isfinite(pf) or pf < 0.0:
            raise ValueError(f"{prob_name}[{idx}] must be finite and >= 0.0, got {p!r}.")
        clean_probs.append(pf)

    prob_arr = np.asarray(clean_probs, dtype=np.float64)
    prob_sum = float(prob_arr.sum())
    if prob_sum <= 0.0 or not math.isclose(prob_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(
            f"{prob_name} must sum to 1.0 (within 1e-4), got sum={prob_sum} for {clean_probs!r}."
        )
    prob_arr = prob_arr / prob_sum
    return tuple(clean_types), prob_arr


def _rand_uniform(
    rng: np.random.Generator | None,
    low: float,
    high: float,
    size: tuple[int, ...] | None = None,
) -> Any:
    if rng is None:
        return np.random.uniform(low, high, size=size)
    return rng.uniform(low, high, size=size)


def _rand_choice(
    rng: np.random.Generator | None,
    choices: Sequence[Any],
    p: np.ndarray | None = None,
) -> Any:
    if rng is None:
        return np.random.choice(choices, p=p)
    return rng.choice(choices, p=p)


def circular_lowpass_kernel(cutoff: float, kernel_size: int, pad_to: int = 0) -> np.ndarray:
    """Generate a normalized 2D circular low-pass (jinc/sinc) filter kernel.

    Uses ``scipy.special.j1`` with an explicit analytical limit at ``r == 0`` to avoid 0/0 warnings.
    """
    k_size = _validate_odd_kernel_size(kernel_size, "kernel_size")

    if isinstance(cutoff, bool) or not isinstance(cutoff, numbers.Real):
        raise ValueError(f"cutoff must be a finite positive float, got {cutoff!r}.")
    cutoff_f = float(cutoff)
    if not math.isfinite(cutoff_f) or cutoff_f <= 0.0:
        raise ValueError(f"cutoff must be finite and > 0.0, got {cutoff!r}.")

    if isinstance(pad_to, bool) or not isinstance(pad_to, (int, np.integer)):
        raise ValueError(f"pad_to must be 0 or an odd integer >= kernel_size, got {pad_to!r}.")
    pad_to_i = int(pad_to)
    if pad_to_i != 0:
        if pad_to_i < k_size or (pad_to_i - k_size) % 2 != 0:
            raise ValueError(
                f"pad_to must be 0 or an odd integer >= kernel_size ({k_size}), got {pad_to_i}."
            )

    half = k_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    r = np.hypot(xx, yy)

    kernel = np.empty_like(r, dtype=np.float64)
    nonzero = r > 0.0
    kernel[nonzero] = (
        cutoff_f * special.j1(cutoff_f * r[nonzero]) / (2.0 * np.pi * r[nonzero])
    )
    kernel[~nonzero] = (cutoff_f ** 2) / (4.0 * np.pi)

    k_sum = float(np.sum(kernel))
    if not math.isfinite(k_sum) or k_sum <= 0.0:
        raise ValueError(
            f"Generated circular_lowpass_kernel has invalid sum {k_sum} for cutoff={cutoff_f}, "
            f"kernel_size={k_size}."
        )
    kernel = kernel / k_sum

    if pad_to_i > k_size:
        pad_size = (pad_to_i - k_size) // 2
        kernel = np.pad(
            kernel,
            ((pad_size, pad_size), (pad_size, pad_size)),
            mode="constant",
            constant_values=0.0,
        )

    return kernel


def _sample_beta(rng: np.random.Generator | None, beta_range: tuple[float, float]) -> float:
    lo, hi = beta_range
    if lo < 1.0 < hi:
        if float(_rand_uniform(rng, 0.0, 1.0)) < 0.5:
            return float(_rand_uniform(rng, lo, 1.0))
        return float(_rand_uniform(rng, 1.0, hi))
    return float(_rand_uniform(rng, lo, hi))


def random_mixed_kernels(
    kernel_list: Sequence[str],
    kernel_prob: Sequence[float],
    kernel_size: int = 21,
    sigma_x_range: Sequence[float] = (0.6, 5.0),
    sigma_y_range: Sequence[float] = (0.6, 5.0),
    rotation_range: Sequence[float] = (-math.pi, math.pi),
    betag_range: Sequence[float] = (0.5, 4.0),
    betap_range: Sequence[float] = (1.0, 2.0),
    noise_range: Sequence[float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly sample and normalize one of the 6 Real-ESRGAN 2D blur kernels."""
    k_size = _validate_odd_kernel_size(kernel_size, "kernel_size")
    types, probs = _validate_kernel_types_and_probs(kernel_list, kernel_prob)
    sig_x_rng = _validate_range(sigma_x_range, "sigma_x_range", positive_only=True)
    sig_y_rng = _validate_range(sigma_y_range, "sigma_y_range", positive_only=True)
    rot_rng = _validate_range(rotation_range, "rotation_range", positive_only=False)
    bg_rng = _validate_range(betag_range, "betag_range", positive_only=True)
    bp_rng = _validate_range(betap_range, "betap_range", positive_only=True)
    noise_rng = (
        _validate_range(noise_range, "noise_range", positive_only=True)
        if noise_range is not None
        else None
    )

    kernel_type = str(_rand_choice(rng, types, p=probs))
    is_aniso = kernel_type.endswith("aniso")

    if is_aniso:
        sig_x = float(_rand_uniform(rng, sig_x_rng[0], sig_x_rng[1]))
        sig_y = float(_rand_uniform(rng, sig_y_rng[0], sig_y_rng[1]))
        theta = float(_rand_uniform(rng, rot_rng[0], rot_rng[1]))
    else:
        sig = float(_rand_uniform(rng, sig_x_rng[0], sig_x_rng[1]))
        sig_x = sig
        sig_y = sig
        theta = 0.0

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    inv_diag = np.diag([1.0 / (sig_x ** 2), 1.0 / (sig_y ** 2)])
    inv_sigma = rot @ inv_diag @ rot.T

    half = k_size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    grid = np.stack((xx, yy), axis=-1)

    q = np.einsum("ijk,kl,ijl->ij", grid, inv_sigma, grid)
    q = np.maximum(q, 0.0)

    if kernel_type in ("iso", "aniso"):
        kernel = np.exp(-0.5 * q)
    elif kernel_type in ("generalized_iso", "generalized_aniso"):
        beta = _sample_beta(rng, bg_rng)
        with np.errstate(over="ignore"):
            q_pow = np.power(q, beta)
        kernel = np.exp(-0.5 * q_pow)
    else:  # plateau_iso, plateau_aniso
        beta = _sample_beta(rng, bp_rng)
        with np.errstate(over="ignore"):
            q_pow = np.power(q, beta)
        kernel = np.reciprocal(q_pow + 1.0)

    if noise_rng is not None:
        noise = _rand_uniform(rng, noise_rng[0], noise_rng[1], size=kernel.shape)
        kernel = kernel * noise

    k_sum = float(np.sum(kernel))
    if not math.isfinite(k_sum) or k_sum <= 0.0:
        raise ValueError(f"Generated blur kernel has invalid sum {k_sum}.")
    return kernel / k_sum


class KernelGenerator:
    """CPU blur and circular-sinc kernel generator for 2nd-order Real-ESRGAN degradation.

    When ``rng`` is omitted in ``sample()``, randomness is drawn from ``np.random`` so PyTorch
    DataLoader multi-worker reseeding automatically yields distinct kernels per worker.
    """

    def __init__(self, kernel_cfg: Mapping[str, Any] | None = None) -> None:
        if kernel_cfg is None:
            cfg: dict[str, Any] = {}
        else:
            normalized = _normalize_omegaconf(kernel_cfg, "kernel_cfg")
            if not isinstance(normalized, Mapping):
                raise TypeError(
                    f"kernel_cfg must be a mapping or None, got {type(normalized).__name__}."
                )
            cfg = dict(normalized)

        self.kernel_size: int = _validate_odd_kernel_size(
            cfg.get("kernel_size", 21), "kernel_size"
        )

        raw_range = cfg.get("kernel_range")
        if raw_range is None:
            default_range = [k for k in range(7, self.kernel_size + 1, 2)]
            raw_range = default_range if default_range else [self.kernel_size]
        else:
            raw_range = _to_1d_list(raw_range, "kernel_range")
            if len(raw_range) == 0:
                raise ValueError("kernel_range must not be empty.")

        clean_range: list[int] = []
        for idx, k in enumerate(raw_range):
            ki = _validate_odd_kernel_size(k, f"kernel_range[{idx}]")
            if ki > self.kernel_size:
                raise ValueError(
                    f"kernel_range[{idx}]={ki} exceeds kernel_size={self.kernel_size}."
                )
            clean_range.append(ki)
        self.kernel_range: tuple[int, ...] = tuple(clean_range)

        self.kernel_list, self.kernel_prob = _validate_kernel_types_and_probs(
            cfg.get("kernel_list", _DEFAULT_KERNEL_LIST),
            cfg.get("kernel_prob", _DEFAULT_KERNEL_PROB),
            list_name="kernel_list",
            prob_name="kernel_prob",
        )
        self.blur_sigma = _validate_range(
            cfg.get("blur_sigma", (0.2, 3.0)), "blur_sigma", positive_only=True
        )
        self.betag_range = _validate_range(
            cfg.get("betag_range", (0.5, 4.0)), "betag_range", positive_only=True
        )
        self.betap_range = _validate_range(
            cfg.get("betap_range", (1.0, 2.0)), "betap_range", positive_only=True
        )
        self.sinc_prob = _validate_scalar_prob(cfg.get("sinc_prob", 0.1), "sinc_prob")

        self.kernel_list2, self.kernel_prob2 = _validate_kernel_types_and_probs(
            cfg.get("kernel_list2", _DEFAULT_KERNEL_LIST),
            cfg.get("kernel_prob2", _DEFAULT_KERNEL_PROB),
            list_name="kernel_list2",
            prob_name="kernel_prob2",
        )
        self.blur_sigma2 = _validate_range(
            cfg.get("blur_sigma2", (0.2, 1.5)), "blur_sigma2", positive_only=True
        )
        self.betag_range2 = _validate_range(
            cfg.get("betag_range2", (0.5, 4.0)), "betag_range2", positive_only=True
        )
        self.betap_range2 = _validate_range(
            cfg.get("betap_range2", (1.0, 2.0)), "betap_range2", positive_only=True
        )
        self.sinc_prob2 = _validate_scalar_prob(cfg.get("sinc_prob2", 0.1), "sinc_prob2")
        self.final_sinc_prob = _validate_scalar_prob(
            cfg.get("final_sinc_prob", 0.8), "final_sinc_prob"
        )

    def _sample_blur_kernel(
        self,
        kernel_list: tuple[str, ...],
        kernel_prob: np.ndarray,
        blur_sigma: tuple[float, float],
        betag_range: tuple[float, float],
        betap_range: tuple[float, float],
        sinc_prob: float,
        rng: np.random.Generator | None,
    ) -> torch.Tensor:
        k_size = int(_rand_choice(rng, self.kernel_range))
        if float(_rand_uniform(rng, 0.0, 1.0)) < sinc_prob:
            omega_low = math.pi / 3.0 if k_size < 13 else math.pi / 5.0
            omega_c = float(_rand_uniform(rng, omega_low, math.pi))
            kernel_np = circular_lowpass_kernel(omega_c, k_size, pad_to=self.kernel_size)
        else:
            kernel_np = random_mixed_kernels(
                kernel_list=kernel_list,
                kernel_prob=kernel_prob,
                kernel_size=k_size,
                sigma_x_range=blur_sigma,
                sigma_y_range=blur_sigma,
                rotation_range=(-math.pi, math.pi),
                betag_range=betag_range,
                betap_range=betap_range,
                noise_range=None,
                rng=rng,
            )
            if k_size < self.kernel_size:
                pad_size = (self.kernel_size - k_size) // 2
                kernel_np = np.pad(
                    kernel_np,
                    ((pad_size, pad_size), (pad_size, pad_size)),
                    mode="constant",
                    constant_values=0.0,
                )

        kernel_t = torch.tensor(kernel_np, dtype=torch.float32)
        kernel_t = kernel_t / kernel_t.sum()
        return kernel_t.contiguous()

    def sample(
        self,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``(kernel1, kernel2, sinc_kernel)`` FP32 tensors of shape ``[K, K]``."""
        kernel1 = self._sample_blur_kernel(
            kernel_list=self.kernel_list,
            kernel_prob=self.kernel_prob,
            blur_sigma=self.blur_sigma,
            betag_range=self.betag_range,
            betap_range=self.betap_range,
            sinc_prob=self.sinc_prob,
            rng=rng,
        )
        kernel2 = self._sample_blur_kernel(
            kernel_list=self.kernel_list2,
            kernel_prob=self.kernel_prob2,
            blur_sigma=self.blur_sigma2,
            betag_range=self.betag_range2,
            betap_range=self.betap_range2,
            sinc_prob=self.sinc_prob2,
            rng=rng,
        )

        if float(_rand_uniform(rng, 0.0, 1.0)) < self.final_sinc_prob:
            k_size = int(_rand_choice(rng, self.kernel_range))
            omega_c = float(_rand_uniform(rng, math.pi / 3.0, math.pi))
            sinc_np = circular_lowpass_kernel(omega_c, k_size, pad_to=self.kernel_size)
            sinc_t = torch.tensor(sinc_np, dtype=torch.float32)
            sinc_t = (sinc_t / sinc_t.sum()).contiguous()
        else:
            sinc_t = torch.zeros((self.kernel_size, self.kernel_size), dtype=torch.float32)
            sinc_t[self.kernel_size // 2, self.kernel_size // 2] = 1.0

        return kernel1, kernel2, sinc_t

    def __call__(
        self,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample(rng=rng)
