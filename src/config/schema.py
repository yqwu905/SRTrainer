"""Declarative Dataclass Structured Config schemas (single source of truth for SRTrainer).

Note: ``typing.Optional[T]`` is used instead of PEP 604 ``T | None`` because
OmegaConf 2.3.x Structured Config does not support ``types.UnionType``.
"""
# ruff: noqa: UP045

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from omegaconf import MISSING

__all__ = [
    "MixedPrecision",
    "NoiseInput",
    "BlendMode",
    "TrainLogImageMode",
    "SaveFormat",
    "VOSR_COMPONENTS",
    "VOSR_REQUIRED_COMPONENTS",
    "VALID_KERNEL_TYPES",
    "TileConfig",
    "PretrainEntryConfig",
    "VOSRTrainableConfig",
    "VOSRPretrainConfig",
    "LoRAConfig",
    "EMAConfig",
    "VOSRModelConfig",
    "ArchGroupConfig",
    "LossEntryConfig",
    "DataSourceConfig",
    "AugmentConfig",
    "KernelConfig",
    "RealESRGANDegraderConfig",
    "DataConfig",
    "OptimConfig",
    "ValDatasetConfig",
    "ValSuiteConfig",
    "MetricConfig",
    "TrainerConfig",
    "TrainConfig",
    "ComponentWeightsConfig",
    "InferSettingsConfig",
    "InferRootConfig",
]

VOSR_COMPONENTS: tuple[str, ...] = ("vae", "dit", "dino")
VOSR_REQUIRED_COMPONENTS: frozenset[str] = frozenset({"vae", "dit"})
VALID_KERNEL_TYPES: tuple[str, ...] = (
    "iso",
    "aniso",
    "generalized_iso",
    "generalized_aniso",
    "plateau_iso",
    "plateau_aniso",
)


class MixedPrecision(str, Enum):
    """Supported Accelerate mixed precision modes."""

    no = "no"
    fp16 = "fp16"
    bf16 = "bf16"


class NoiseInput(str, Enum):
    """Initial noise sampling mode for one-step VOSR diffusion."""

    gaussian = "gaussian"
    zero = "zero"


class BlendMode(str, Enum):
    """2D feathering window mode for tiled inference."""

    cosine = "cosine"
    linear = "linear"
    mean = "mean"


class TrainLogImageMode(str, Enum):
    """Layout mode for periodic training visuals logged to trackers."""

    comparison = "comparison"
    separate = "separate"


class SaveFormat(str, Enum):
    """Supported output image save formats for offline inference."""

    png = "png"
    jpg = "jpg"


@dataclass
class TileConfig:
    """Shared tiled inference configuration for validation and offline inference.

    ``size=None`` derives square tiles automatically from ``gt_size // scale`` (falling back
    to untiled inference when ``gt_size`` is unavailable). Explicit tile size is always a
    2-element ``[tile_h, tile_w]`` list of positive integers.
    """

    enabled: bool = True
    size: Optional[list[int]] = None
    overlap: int = 16
    batch_size: int = 4
    blend_mode: BlendMode = BlendMode.cosine
    pad_multiple: int = 8


@dataclass
class PretrainEntryConfig:
    """Optional pretrained checkpoint override for a single architecture component."""

    path: Optional[str] = None
    strict: bool = True
    key: Optional[str] = None


@dataclass
class VOSRTrainableConfig:
    """Per-component trainability switches for VOSRModel."""

    vae: bool = False
    dit: bool = True
    dino: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {"vae": self.vae, "dit": self.dit, "dino": self.dino}

    def get(self, component: str) -> bool:
        if component not in VOSR_COMPONENTS:
            raise KeyError(
                f"Unknown VOSR component {component!r}; expected one of {VOSR_COMPONENTS}."
            )
        return bool(getattr(self, component))


@dataclass
class VOSRPretrainConfig:
    """Per-component pretrain overrides for VOSRModel."""

    vae: PretrainEntryConfig = field(default_factory=PretrainEntryConfig)
    dit: PretrainEntryConfig = field(default_factory=PretrainEntryConfig)
    dino: PretrainEntryConfig = field(default_factory=PretrainEntryConfig)

    def get(self, component: str) -> PretrainEntryConfig:
        if component not in VOSR_COMPONENTS:
            raise KeyError(
                f"Unknown VOSR component {component!r}; expected one of {VOSR_COMPONENTS}."
            )
        return getattr(self, component)


@dataclass
class LoRAConfig:
    """LoRA fine-tuning settings applied exclusively to DiT when trainable."""

    enable: bool = False
    lora_rank: int = 32
    lora_alpha: float = 32.0
    train_pre_post: bool = True
    checkpoint: Optional[str] = None


@dataclass
class EMAConfig:
    """Exponential moving average settings applied to trainable components."""

    enabled: bool = True
    decay: float = 0.999
    warmup_steps: int = 0


@dataclass
class VOSRModelConfig:
    """Structured configuration for ``models.VOSRModel``."""

    _target_: str = "models.VOSRModel"
    scale: int = 1
    trainable: VOSRTrainableConfig = field(default_factory=VOSRTrainableConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    noise_input: NoiseInput = NoiseInput.gaussian
    posterior_mode: bool = True
    clamp_decode_in_train: bool = True
    eval_seed: Optional[int] = 42
    ema: EMAConfig = field(default_factory=EMAConfig)
    save_frozen_weights: bool = True
    pretrain: VOSRPretrainConfig = field(default_factory=VOSRPretrainConfig)


@dataclass
class ArchGroupConfig:
    """Composable architecture group containing independent ``vae``, ``dit``, and optional ``dino``."""

    vae: Optional[dict[str, Any]] = None
    dit: Optional[dict[str, Any]] = None
    dino: Optional[dict[str, Any]] = None

    def get(self, component: str) -> Optional[dict[str, Any]]:
        if component not in VOSR_COMPONENTS:
            raise KeyError(
                f"Unknown VOSR component {component!r}; expected one of {VOSR_COMPONENTS}."
            )
        return getattr(self, component)


@dataclass
class LossEntryConfig:
    """Single named loss entry combining scalar weight and Hydra ``_target_`` criterion."""

    weight: float = 1.0
    criterion: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSourceConfig:
    """Single training dataset source manifest definition (never contains legacy ``mode``)."""

    name: str = MISSING
    root_dir: str = MISSING
    json_path: str = MISSING
    ratio: float = 1.0
    glyphs_json: Optional[str] = None


@dataclass
class AugmentConfig:
    """Synchronized dihedral data augmentation switches."""

    hflip: bool = True
    vflip: bool = True
    rot90: bool = True


@dataclass
class KernelConfig:
    """CPU blur and circular-sinc kernel generation parameters."""

    kernel_size: int = 21
    kernel_range: Optional[list[int]] = None
    kernel_list: list[str] = field(default_factory=lambda: list(VALID_KERNEL_TYPES))
    kernel_prob: list[float] = field(
        default_factory=lambda: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    )
    blur_sigma: list[float] = field(default_factory=lambda: [0.2, 3.0])
    betag_range: list[float] = field(default_factory=lambda: [0.5, 4.0])
    betap_range: list[float] = field(default_factory=lambda: [1.0, 2.0])
    sinc_prob: float = 0.1
    kernel_list2: list[str] = field(default_factory=lambda: list(VALID_KERNEL_TYPES))
    kernel_prob2: list[float] = field(
        default_factory=lambda: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    )
    blur_sigma2: list[float] = field(default_factory=lambda: [0.2, 1.5])
    betag_range2: list[float] = field(default_factory=lambda: [0.5, 4.0])
    betap_range2: list[float] = field(default_factory=lambda: [1.0, 2.0])
    sinc_prob2: float = 0.1
    final_sinc_prob: float = 0.8


@dataclass
class RealESRGANDegraderConfig:
    """Structured config for ``data.degradation.RealESRGANDegrader``."""

    _target_: str = "data.degradation.RealESRGANDegrader"
    scale: int = 1
    use_usm: bool = False
    queue_size: int = 0
    kernel_cfg: KernelConfig = field(default_factory=KernelConfig)

    # 1st-order GPU degradation parameters
    resize_prob: list[float] = field(default_factory=lambda: [0.2, 0.7, 0.1])
    resize_range: list[float] = field(default_factory=lambda: [0.15, 1.5])
    gaussian_noise_prob: float = 0.5
    noise_range: list[float] = field(default_factory=lambda: [1.0, 30.0])
    poisson_scale_range: list[float] = field(default_factory=lambda: [0.05, 3.0])
    gray_noise_prob: float = 0.4
    jpeg_range: list[float] = field(default_factory=lambda: [30.0, 95.0])

    # 2nd-order GPU degradation parameters
    second_blur_prob: float = 0.8
    resize_prob2: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    resize_range2: list[float] = field(default_factory=lambda: [0.3, 1.2])
    gaussian_noise_prob2: float = 0.5
    noise_range2: list[float] = field(default_factory=lambda: [1.0, 25.0])
    poisson_scale_range2: list[float] = field(default_factory=lambda: [0.05, 2.5])
    gray_noise_prob2: float = 0.4
    jpeg_range2: list[float] = field(default_factory=lambda: [30.0, 95.0])

    # Optional USM hyperparameters
    usm_radius: int = 50
    usm_sigma: float = 0.0
    usm_weight: float = 0.5
    usm_threshold: int = 10


@dataclass
class DataConfig:
    """Training data pipeline configuration (validation suites belong strictly in ``MetricConfig``)."""

    gt_size: int = 512
    scale: int = 1
    batch_size_per_gpu: int = 1
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    text_crop_prob: float = 0.0
    sources: list[DataSourceConfig] = field(default_factory=list)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    degradation: Any = None


@dataclass
class OptimConfig:
    """Optimizer and optional LR scheduler partial Hydra target configurations."""

    optimizer: dict[str, Any] = field(default_factory=dict)
    scheduler: Optional[dict[str, Any]] = None


@dataclass
class ValDatasetConfig:
    """Validation dataset manifest definition inside a ``ValSuiteConfig``."""

    json_path: str = MISSING
    root_dir: Optional[str] = None


@dataclass
class ValSuiteConfig:
    """Single validation suite co-locating dataset and per-suite metric targets."""

    dataset: ValDatasetConfig = field(default_factory=ValDatasetConfig)
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_save_images: Optional[int] = None


@dataclass
class MetricConfig:
    """Validation suites and primary benchmark selection for best-checkpoint saving."""

    primary_dataset: Optional[str] = None
    primary_metric: Optional[str] = None
    val_suites: dict[str, ValSuiteConfig] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    """Iteration-driven Accelerate training loop, logging, checkpointing, and validation settings."""

    total_iters: int = 300000
    val_interval: int = 5000
    save_interval: int = 5000
    log_interval: int = 100
    max_keep_states: Optional[int] = 3

    mixed_precision: MixedPrecision = MixedPrecision.bf16
    gradient_accumulation_steps: int = 1
    grad_clip_norm: Optional[float] = 1.0
    compile: bool = False
    ddp_find_unused_parameters: bool = True

    log_with: list[str] = field(default_factory=lambda: ["tensorboard"])
    tracker_init_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)

    resume: Optional[str] = None
    append_timestamp: bool = False

    train_log_image_interval: int = 500
    train_max_log_images: int = 4
    train_log_image_mode: TrainLogImageMode = TrainLogImageMode.comparison

    val_save_visuals: bool = True
    val_max_save_images: int = 8
    val_use_ema: bool = True
    val_tile: TileConfig = field(default_factory=TileConfig)


@dataclass
class TrainConfig:
    """Root structured configuration for SRTrainer training."""

    exp_name: str = "debug_vosr2"
    scale: int = 1
    seed: int = 42
    output_root: str = "experiments"
    work_dir: str = "${output_root}/${exp_name}"

    arch: ArchGroupConfig = field(default_factory=ArchGroupConfig)
    loss: dict[str, LossEntryConfig] = field(default_factory=dict)
    model: VOSRModelConfig = field(default_factory=VOSRModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


@dataclass
class ComponentWeightsConfig:
    """Per-component paths or checkpoint dict keys for multi-arch VOSR inference."""

    vae: Optional[str] = None
    dit: Optional[str] = None
    dino: Optional[str] = None

    def get(self, component: str) -> Optional[str]:
        if component not in VOSR_COMPONENTS:
            raise KeyError(
                f"Unknown VOSR component {component!r}; expected one of {VOSR_COMPONENTS}."
            )
        return getattr(self, component)

    def as_dict(self) -> dict[str, Optional[str]]:
        return {"vae": self.vae, "dit": self.dit, "dino": self.dino}

    def any_set(self) -> bool:
        return any(getattr(self, c) is not None for c in VOSR_COMPONENTS)


@dataclass
class InferSettingsConfig:
    """Structured inference settings for ``tools/infer.py``."""

    exp_dir: Optional[str] = None
    weights_path: Optional[str] = None
    weights_key: Optional[str] = None
    component_weights: ComponentWeightsConfig = field(default_factory=ComponentWeightsConfig)
    component_weights_key: ComponentWeightsConfig = field(default_factory=ComponentWeightsConfig)
    use_ema: bool = True

    input: Optional[str] = None
    input_root: Optional[str] = None
    output_dir: str = "outputs/infer"
    recursive: bool = False
    suffix: str = ""
    save_format: SaveFormat = SaveFormat.png
    mixed_precision: MixedPrecision = MixedPrecision.bf16
    tile: TileConfig = field(default_factory=TileConfig)


@dataclass
class InferRootConfig:
    """Root structured configuration for ``tools/infer.py``."""

    scale: int = 1
    arch: Any = None
    model: Optional[VOSRModelConfig] = None
    infer: InferSettingsConfig = field(default_factory=InferSettingsConfig)
