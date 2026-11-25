"""Unit tests for Structured Config schemas, ConfigStore, unified YAML format, and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pytest
from hydra.core.config_store import ConfigStore
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from config import (
    AugmentConfig,
    BlendMode,
    ComponentWeightsConfig,
    ConfigValidationError,
    DataConfig,
    DataSourceConfig,
    EMAConfig,
    InferRootConfig,
    InferSettingsConfig,
    KernelConfig,
    MetricConfig,
    MixedPrecision,
    NoiseInput,
    RealESRGANDegraderConfig,
    SaveFormat,
    TileConfig,
    TrainConfig,
    TrainerConfig,
    TrainLogImageMode,
    VOSRModelConfig,
    load_augment_config,
    load_infer_config,
    load_kernel_config,
    load_tile_config,
    load_train_config,
)
from engine.inferencer import TiledInferencer
from engine.trainer import Trainer
from models import BaseModel, VOSRModel
from models.base_model import build_model
from tests.dummy_arch import get_dummy_vosr_arch_dict
from utils.config import load_resolved_config, save_resolved_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


def _compose(config_name: str, overrides: list[str] | None = None):
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        return hydra.compose(config_name=config_name, overrides=overrides or [])


def _minimal_train_dict(tmp_path: Path) -> dict[str, Any]:
    return {
        "exp_name": "unit_exp",
        "scale": 2,
        "seed": 42,
        "output_root": str(tmp_path),
        "work_dir": str(tmp_path / "unit_exp"),
        "arch": get_dummy_vosr_arch_dict(),
        "loss": {
            "l1": {
                "weight": 1.0,
                "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
            }
        },
        "model": {
            "_target_": "models.VOSRModel",
            "scale": 2,
            "trainable": {"vae": False, "dit": True, "dino": False},
            "ema": {"enabled": True, "decay": 0.99, "warmup_steps": 0},
        },
        "data": {
            "gt_size": 32,
            "scale": 2,
            "batch_size_per_gpu": 2,
            "num_workers": 0,
            "sources": [
                {
                    "name": "s0",
                    "root_dir": str(tmp_path),
                    "json_path": str(tmp_path / "train.json"),
                    "ratio": 1.0,
                }
            ],
            "degradation": None,
        },
        "optim": {
            "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1e-4},
            "scheduler": None,
        },
        "metric": {
            "primary_dataset": None,
            "primary_metric": None,
            "val_suites": {},
        },
        "trainer": {
            "total_iters": 4,
            "mixed_precision": "no",
            "log_with": [],
        },
    }


# ---------------------------------------------------------------------------
# Principle 1 & 2: Declarative Schema, ConfigStore & Single Source of Truth
# ---------------------------------------------------------------------------


def test_config_store_registrations_and_no_legacy_parse_helpers() -> None:
    cs = ConfigStore.instance()
    assert "base_train.yaml" in cs.repo
    assert "base_infer.yaml" in cs.repo
    assert "base_vosr.yaml" in cs.repo["model"]
    assert "base_trainer.yaml" in cs.repo["trainer"]
    assert "base_data.yaml" in cs.repo["data"]
    assert "base_realesrgan.yaml" in cs.repo["data"]["degradation"]
    assert "base_metric.yaml" in cs.repo["metric"]
    assert "base_infer_settings.yaml" in cs.repo["infer"]

    # Principle 1: no _parse_xxx structural helpers on VOSRModel or Trainer
    import engine.trainer as trainer_mod
    import models.vosr_model as vosr_mod

    for name in dir(vosr_mod):
        assert not name.startswith("_parse_"), f"Legacy helper left in vosr_model: {name}"
    for name in dir(trainer_mod):
        assert not name.startswith("_require_"), f"Legacy helper left in trainer: {name}"
    for name in dir(Trainer):
        assert not name.startswith("_parse_"), f"Legacy helper left on Trainer: {name}"


def test_single_source_of_truth_defaults_match_composed_yamls() -> None:
    train_cfg = _compose(
        "train",
        overrides=["~experiment", "arch/vae=qwen_vae2d", "arch/dit=vosr2_dit", "loss=l1"],
    )
    default_train = TrainConfig()
    default_model = VOSRModelConfig()
    default_trainer = TrainerConfig()
    default_data = DataConfig()
    default_deg = RealESRGANDegraderConfig()
    default_kernel = KernelConfig()

    assert train_cfg.exp_name == default_train.exp_name
    assert train_cfg.scale == default_train.scale
    assert train_cfg.seed == default_train.seed
    assert train_cfg.model.ema.decay == default_model.ema.decay
    assert train_cfg.model.noise_input == default_model.noise_input
    assert train_cfg.trainer.total_iters == default_trainer.total_iters
    assert train_cfg.trainer.mixed_precision == default_trainer.mixed_precision
    assert list(train_cfg.trainer.log_with) == list(default_trainer.log_with)
    assert train_cfg.trainer.val_tile.size == default_trainer.val_tile.size is None
    assert train_cfg.trainer.val_tile.overlap == default_trainer.val_tile.overlap
    assert train_cfg.data.gt_size == default_data.gt_size
    assert train_cfg.data.degradation.queue_size == default_deg.queue_size
    assert train_cfg.data.degradation.kernel_cfg.kernel_size == default_kernel.kernel_size
    assert train_cfg.data.degradation.kernel_cfg.kernel_range is None

    infer_cfg = _compose("infer")
    default_infer = InferSettingsConfig()
    assert infer_cfg.infer.tile.size == default_infer.tile.size is None
    assert infer_cfg.infer.tile.overlap == default_infer.tile.overlap
    assert infer_cfg.infer.save_format == default_infer.save_format

    # TiledInferencer.from_config(tile_cfg=None) inherits TileConfig defaults
    from torch import nn

    inf = TiledInferencer.from_config(model=nn.Identity(), scale=1, tile_cfg=None, gt_size=64)
    tile_defaults = TileConfig()
    assert inf.tile_enabled is tile_defaults.enabled
    assert inf.tile_overlap == tile_defaults.overlap
    assert inf.tile_batch_size == tile_defaults.batch_size
    assert inf.blend_mode == tile_defaults.blend_mode.value
    assert inf.pad_multiple == tile_defaults.pad_multiple


# ---------------------------------------------------------------------------
# Principle 3: Unified YAML Format (Reject Polymorphic & Unknown Inputs)
# ---------------------------------------------------------------------------


def test_unified_yaml_rejects_polymorphic_and_unknown_fields(tmp_path: Path) -> None:
    # 1. trainer.log_with must be list[str] (reject None and bare str)
    for bad_log_with in (None, "tensorboard"):
        raw = _minimal_train_dict(tmp_path)
        raw["trainer"]["log_with"] = bad_log_with
        with pytest.raises(ConfigValidationError):
            load_train_config(raw)

    with pytest.raises(ConfigCompositionException):
        _compose("train", overrides=["experiment=vosr2_onestep_text", "trainer.log_with=null"])

    with pytest.raises(ConfigCompositionException):
        _compose("train", overrides=["experiment=vosr2_onestep_text", "trainer.log_with=wandb"])

    # 2. tile.size must be null or [H, W] (reject "auto" and scalar int)
    for bad_tile_size in ("auto", 16, [16], [16, 16, 16]):
        with pytest.raises(ConfigValidationError):
            load_tile_config({"enabled": True, "size": bad_tile_size, "overlap": 4})

    with pytest.raises(ConfigCompositionException):
        _compose("infer", overrides=["infer.tile.size=auto"])

    with pytest.raises(ConfigCompositionException):
        _compose("infer", overrides=["infer.tile.size=16"])

    # 3. Unknown keys rejected by Structured Config
    raw_unknown_trainable = _minimal_train_dict(tmp_path)
    raw_unknown_trainable["model"]["trainable"]["extra_net"] = True
    with pytest.raises(ConfigValidationError, match="extra_net"):
        load_train_config(raw_unknown_trainable)

    raw_legacy_mode = _minimal_train_dict(tmp_path)
    raw_legacy_mode["data"]["sources"][0]["mode"] = "gt_only"
    with pytest.raises(ConfigValidationError, match="mode"):
        load_train_config(raw_legacy_mode)

    raw_legacy_text_boxes = _minimal_train_dict(tmp_path)
    raw_legacy_text_boxes["data"]["sources"][0]["text_boxes"] = "boxes.json"
    with pytest.raises(ConfigValidationError, match="text_boxes"):
        load_train_config(raw_legacy_text_boxes)

    # 4. Invalid Enum literals rejected
    raw_bad_mp = _minimal_train_dict(tmp_path)
    raw_bad_mp["trainer"]["mixed_precision"] = "fp8"
    with pytest.raises(ConfigValidationError):
        load_train_config(raw_bad_mp)

    raw_bad_noise = _minimal_train_dict(tmp_path)
    raw_bad_noise["model"]["noise_input"] = "uniform"
    with pytest.raises(ConfigValidationError):
        load_train_config(raw_bad_noise)

    raw_bad_mode = _minimal_train_dict(tmp_path)
    raw_bad_mode["trainer"]["train_log_image_mode"] = "grid"
    with pytest.raises(ConfigValidationError):
        load_train_config(raw_bad_mode)

    with pytest.raises(ConfigValidationError):
        load_infer_config({"infer": {"save_format": "bmp"}})


# ---------------------------------------------------------------------------
# Principle 4: Business Validation (Ranges & Cross-Field Dependencies)
# ---------------------------------------------------------------------------


def test_business_range_and_cross_field_validations(tmp_path: Path) -> None:
    # gt_size not divisible by scale
    raw = _minimal_train_dict(tmp_path)
    raw["data"]["gt_size"] = 33
    with pytest.raises(ConfigValidationError, match="divisible"):
        load_train_config(raw)

    # ema.decay out of (0, 1]
    raw = _minimal_train_dict(tmp_path)
    raw["model"]["ema"]["decay"] = 1.1
    with pytest.raises(ConfigValidationError, match="ema.decay"):
        load_train_config(raw)

    # All VOSR components frozen
    raw = _minimal_train_dict(tmp_path)
    raw["model"]["trainable"] = {"vae": False, "dit": False, "dino": False}
    with pytest.raises(ConfigValidationError, match="At least one active VOSR arch"):
        load_train_config(raw)

    # trainable.dino=True with arch.dino=None
    raw = _minimal_train_dict(tmp_path)
    raw["arch"] = get_dummy_vosr_arch_dict(with_dino=False)
    raw["model"]["trainable"] = {"vae": False, "dit": True, "dino": True}
    with pytest.raises(ConfigValidationError, match="trainable.dino"):
        load_train_config(raw)

    # Empty or all-zero loss weights
    raw = _minimal_train_dict(tmp_path)
    raw["loss"]["l1"]["weight"] = 0.0
    with pytest.raises(ConfigValidationError, match="weight > 0"):
        load_train_config(raw)

    # primary_dataset / primary_metric mismatch
    raw = _minimal_train_dict(tmp_path)
    raw["metric"]["primary_dataset"] = "set5"
    with pytest.raises(ConfigValidationError, match="must be set together"):
        load_train_config(raw)

    # tile overlap >= tile_size // 2
    with pytest.raises(ConfigValidationError, match="overlap"):
        load_tile_config({"enabled": True, "size": [16, 16], "overlap": 8})

    # KernelConfig even kernel_size & out-of-range kernel_range
    with pytest.raises(ConfigValidationError, match="kernel_size"):
        load_kernel_config({"kernel_size": 8})
    with pytest.raises(ConfigValidationError, match="kernel_range"):
        load_kernel_config({"kernel_size": 7, "kernel_range": [7, 9]})

    # AugmentConfig validates cleanly
    aug = load_augment_config({"hflip": True, "vflip": False, "rot90": True})
    assert isinstance(aug, AugmentConfig)
    assert aug == AugmentConfig(hflip=True, vflip=False, rot90=True)

    # Infer config: weights_path and component_weights mutually exclusive
    with pytest.raises(ConfigValidationError, match="Cannot set both"):
        load_infer_config(
            {
                "infer": {
                    "weights_path": "a.safetensors",
                    "component_weights": {"dit": "b.safetensors"},
                }
            }
        )


# ---------------------------------------------------------------------------
# Principle 5: Centralized Validation & Strongly-Typed Models
# ---------------------------------------------------------------------------


def test_centralized_validation_returns_typed_dataclasses_and_roundtrips(tmp_path: Path) -> None:
    raw = _minimal_train_dict(tmp_path)
    typed = load_train_config(raw)

    assert isinstance(typed, TrainConfig)
    assert isinstance(typed.model, VOSRModelConfig)
    assert isinstance(typed.model.ema, EMAConfig)
    assert isinstance(typed.data, DataConfig)
    assert isinstance(typed.data.sources[0], DataSourceConfig)
    assert isinstance(typed.metric, MetricConfig)
    assert isinstance(typed.trainer, TrainerConfig)
    assert isinstance(typed.trainer.val_tile, TileConfig)
    assert typed.trainer.mixed_precision is MixedPrecision.no
    assert typed.trainer.val_tile.blend_mode is BlendMode.cosine
    assert typed.trainer.train_log_image_mode is TrainLogImageMode.comparison
    assert typed.model.noise_input is NoiseInput.gaussian

    # Model directly consumes typed TrainConfig and rejects raw dict/DictConfig
    with pytest.raises(TypeError, match="TrainConfig"):
        VOSRModel(raw)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TrainConfig"):
        VOSRModel(OmegaConf.create(raw))  # type: ignore[arg-type]

    model = build_model(typed)
    assert isinstance(model, VOSRModel)
    assert isinstance(model, BaseModel)
    assert model.cfg is typed

    # Infer config typed object
    infer_typed = load_infer_config(_compose("infer"))
    assert isinstance(infer_typed, InferRootConfig)
    assert isinstance(infer_typed.infer, InferSettingsConfig)
    assert isinstance(infer_typed.infer.tile, TileConfig)
    assert isinstance(infer_typed.infer.component_weights, ComponentWeightsConfig)
    assert infer_typed.infer.save_format is SaveFormat.png

    # save_resolved_config -> load_resolved_config -> load_train_config roundtrip
    out_yaml = tmp_path / "config_resolved.yaml"
    save_resolved_config(typed, out_yaml)
    assert "${" not in out_yaml.read_text(encoding="utf-8")
    reloaded_dc = load_resolved_config(out_yaml)
    reloaded = load_train_config(reloaded_dc)
    assert reloaded.exp_name == typed.exp_name
    assert reloaded.trainer.mixed_precision is MixedPrecision.no
    assert reloaded.trainer.val_tile.size is None
