"""Tier 1 E2E tests — Area A: Hydra Configuration, Inheritance & Packaging.

Covers features F01-F12, F16, F20 and User Requirements #2, #3, #4, #5.
All non-stdlib imports (besides ``pytest``) are deferred inside test functions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tomllib


def _parse_dep_name(spec: str) -> str:
    """Extract normalized distribution name from a PEP 508 dependency specifier."""
    base = re.split(r"[<>=!~;\[\s@]", spec.strip(), maxsplit=1)[0]
    return base.lower().replace("_", "-")


def _contains_key_recursive(obj: Any, target_key: str) -> bool:
    if isinstance(obj, dict):
        if target_key in obj:
            return True
        return any(_contains_key_recursive(v, target_key) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_key_recursive(item, target_key) for item in obj)
    return False


def _yaml_contains_token(obj: Any, token: str) -> bool:
    """Return True if parsed YAML keys or string values contain ``token`` (ignoring comments)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and token in k:
                return True
            if _yaml_contains_token(v, token):
                return True
    elif isinstance(obj, list):
        return any(_yaml_contains_token(item, token) for item in obj)
    elif isinstance(obj, str):
        return token in obj
    return False


def test_t1_packaging_pyproject_and_gitignore(repo_root: Path) -> None:
    """F01, F20: pyproject.toml src layout, platform-neutral deps, dev extras, .gitignore."""
    pyproject_path = repo_root / "pyproject.toml"
    assert pyproject_path.is_file(), "pyproject.toml must exist at repository root"

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "srtrainer"

    deps = project.get("dependencies", [])
    dep_names = {_parse_dep_name(d) for d in deps}
    expected_deps = {
        "torch",
        "torchvision",
        "accelerate",
        "hydra-core",
        "omegaconf",
        "safetensors",
        "pillow",
        "lpips",
        "numpy",
    }
    missing = expected_deps - dep_names
    assert not missing, f"Missing required dependencies in pyproject.toml: {sorted(missing)}"

    # Dependencies must stay platform-neutral (no hardcoded CUDA wheel suffixes/URLs).
    for spec in deps:
        assert "+cu" not in spec.lower(), f"Platform-specific CUDA wheel pinned in pyproject: {spec}"

    dev_deps = {_parse_dep_name(d) for d in project.get("optional-dependencies", {}).get("dev", [])}
    assert {"pytest", "ruff"}.issubset(dev_deps)

    # Confirm flattened src/ layout and editable import resolution.
    assert (repo_root / "src" / "archs" / "__init__.py").is_file()
    import archs

    pkg_file = Path(archs.__file__).resolve()
    expected_pkg_root = (repo_root / "src").resolve()
    assert expected_pkg_root in pkg_file.parents

    # Confirm .gitignore ignores generated runtime artifacts.
    gitignore_path = repo_root / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore must exist at repository root"
    gitignore_lines = [
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any("experiments" in line for line in gitignore_lines)
    assert any("outputs" in line for line in gitignore_lines)
    assert any("__pycache__" in line for line in gitignore_lines)


def test_t1_subpackage_short_target_reexports() -> None:
    """F02, Req #5: Short-path __all__ re-exports expose VOSR2OneStep and built-in losses."""
    import hydra.utils

    import archs as archs_pkg
    import losses as losses_pkg

    assert hasattr(archs_pkg, "__all__")
    assert set(archs_pkg.__all__) == {"VOSR2OneStep"}, (
        f"Unexpected archs.__all__: {archs_pkg.__all__!r}"
    )
    assert hydra.utils.get_class("archs.VOSR2OneStep") is archs_pkg.VOSR2OneStep
    for forbidden_arch in ("ToyModel", "RRDBNet", "SwinIR", "TinySRNet"):
        assert not hasattr(archs_pkg, forbidden_arch), (
            f"Forbidden architecture {forbidden_arch!r} must not exist in archs"
        )

    assert hasattr(losses_pkg, "__all__")
    assert set(losses_pkg.__all__) == {"L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"}, (
        f"Unexpected losses.__all__: {losses_pkg.__all__!r}"
    )
    for sym in ("L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"):
        assert hydra.utils.get_class(f"losses.{sym}") is getattr(losses_pkg, sym)
    for forbidden_loss in ("CharbonnierLoss", "FFTLoss"):
        assert not hasattr(losses_pkg, forbidden_loss), (
            f"Forbidden loss {forbidden_loss!r} must not exist in losses"
        )


def test_t1_train_yaml_default_composition_and_hydra_guards(
    configs_dir: Path,
    hydra_compose: Callable[..., Any],
) -> None:
    """F03-F10: train.yaml defaults to vosr2 & scale=1, composes all groups, enforces Hydra guards."""
    import yaml

    cfg = hydra_compose("train")

    assert cfg.exp_name == "debug_vosr2"
    assert cfg.scale == 1
    assert cfg.seed == 42
    assert cfg.output_root == "experiments"
    assert cfg.work_dir == "experiments/debug_vosr2"

    for group in ("arch", "loss", "model", "data", "optim", "metric", "trainer"):
        assert group in cfg, f"Missing atomic group {group!r} in composed train.yaml"

    assert cfg.arch._target_ == "archs.VOSR2OneStep"
    assert cfg.arch.scale == 1
    assert cfg.data.scale == 1
    assert cfg.optim.scheduler.T_max == cfg.trainer.total_iters
    assert cfg.model._target_ == "models.SRModel"
    assert "ema" in cfg.model
    assert "pretrain_path" in cfg.model

    raw_train = yaml.safe_load((configs_dir / "train.yaml").read_text(encoding="utf-8"))
    assert raw_train["hydra"]["job"]["chdir"] is False
    assert raw_train["hydra"]["output_subdir"] is None

    defaults = raw_train["defaults"]
    assert isinstance(defaults, list)

    has_job_logging_disabled = any(
        (isinstance(item, dict) and item.get("override hydra/job_logging") == "disabled")
        or (isinstance(item, str) and "override hydra/job_logging" in item and "disabled" in item)
        for item in defaults
    )
    assert has_job_logging_disabled, (
        "configs/train.yaml must include `override hydra/job_logging: disabled` in defaults"
    )

    self_idx = next(
        (i for i, item in enumerate(defaults) if item == "_self_"),
        None,
    )
    exp_idx = next(
        (
            i
            for i, item in enumerate(defaults)
            if isinstance(item, dict)
            and any(k in ("experiment", "optional experiment") for k in item)
        ),
        None,
    )
    assert self_idx is not None and exp_idx is not None
    assert self_idx < exp_idx, (
        "`_self_` must precede `experiment: null` in configs/train.yaml defaults so "
        "# @package _global_ experiments can override exp_name/scale/seed"
    )


def test_t1_atomic_arch_loss_optim_instantiation(
    hydra_compose: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F04, F05, F08: Atomic vosr2 arch, l1/l1_lpips losses, and partial optims instantiate."""
    import hydra.utils
    import torch
    import torch.nn as nn
    from tests.helpers.fake_backends import patch_lpips_offline, patch_vosr2_backends

    from archs import VOSR2OneStep
    from losses import L1Loss, LPIPSLoss

    patch_vosr2_backends(monkeypatch, z_dim=4, dino_dim=16)
    patch_lpips_offline(monkeypatch)

    cfg = hydra_compose(
        "train",
        overrides=[
            "arch=vosr2",
            "arch.input_size=8",
            "arch.in_channels=8",
            "arch.out_channels=4",
            "arch.hidden_size=32",
            "arch.depth=1",
            "arch.num_heads=4",
            "arch.mlp_ratio=2.0",
            "arch.z_dims=16",
            "arch.encdim_ratio=2",
            "arch.dino_size=28",
            "arch.dino_layer=0",
            "loss=l1_lpips",
            "optim=adamw_cosine",
            "trainer.total_iters=100",
        ],
    )

    net = hydra.utils.instantiate(cfg.arch)
    assert isinstance(net, VOSR2OneStep)
    assert isinstance(net, nn.Module)
    assert net.scale == 1

    assert set(cfg.loss.keys()) == {"l1", "lpips"}
    for name, entry in cfg.loss.items():
        assert float(entry.weight) > 0.0, f"Expected positive weight for loss {name}"
        criterion = hydra.utils.instantiate(entry.criterion)
        assert isinstance(criterion, (L1Loss, LPIPSLoss))

    # Verify atomic loss=l1 config instantiates L1Loss cleanly.
    cfg_l1 = hydra_compose("train", overrides=["loss=l1"])
    assert set(cfg_l1.loss.keys()) == {"l1"}
    assert isinstance(hydra.utils.instantiate(cfg_l1.loss.l1.criterion), L1Loss)

    opt_builder = hydra.utils.instantiate(cfg.optim.optimizer)
    assert callable(opt_builder)
    opt = opt_builder(net.parameters())
    assert isinstance(opt, torch.optim.AdamW)

    sched_builder = hydra.utils.instantiate(cfg.optim.scheduler)
    assert callable(sched_builder)
    sched = sched_builder(opt)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 100

    # Verify adam_multistep atomic optim config.
    cfg_ms = hydra_compose("train", overrides=["optim=adam_multistep"])
    opt_ms = hydra.utils.instantiate(cfg_ms.optim.optimizer)(net.parameters())
    sched_ms = hydra.utils.instantiate(cfg_ms.optim.scheduler)(opt_ms)
    assert isinstance(opt_ms, torch.optim.Adam)
    assert isinstance(sched_ms, torch.optim.lr_scheduler.MultiStepLR)

    # Verify vosr2 scale interpolation cascades from global scale.
    cfg_vosr2_x2 = hydra_compose("train", overrides=["arch=vosr2", "scale=2"])
    assert cfg_vosr2_x2.arch._target_ == "archs.VOSR2OneStep"
    assert cfg_vosr2_x2.arch.scale == 2


def test_t1_multilevel_experiment_inheritance(
    hydra_compose: Callable[..., Any],
) -> None:
    """F12, F03: # @package _global_ experiment config overrides defaults cleanly."""
    exp_cfg = hydra_compose("train", overrides=["experiment=vosr2_onestep_text"])
    assert exp_cfg.exp_name == "vosr2_onestep_text"
    assert exp_cfg.work_dir == "experiments/vosr2_onestep_text"
    assert exp_cfg.scale == 1
    assert exp_cfg.arch._target_ == "archs.VOSR2OneStep"
    assert exp_cfg.arch.scale == 1
    assert exp_cfg.arch.depth == 36

    # Partial overrides preserve sibling keys from inherited groups.
    assert set(exp_cfg.loss.keys()) == {"l1", "lpips", "ocr_kl", "glyph"}
    assert exp_cfg.optim.optimizer.lr == pytest.approx(5.0e-6)
    assert list(exp_cfg.optim.optimizer.betas) == [0.9, 0.95]

    # Cascading interpolation & experiment trainer settings.
    assert exp_cfg.trainer.total_iters == 10000
    assert exp_cfg.optim.scheduler.T_max == 10000
    assert exp_cfg.trainer.val_interval == 1000
    assert exp_cfg.trainer.mixed_precision == "bf16"
    assert exp_cfg.trainer.val_tile.overlap == 64
    assert exp_cfg.trainer.val_tile.pad_multiple == 16


def test_t1_cli_override_precedence_and_interpolation_cascade(
    e2e_workdir: dict[str, Path],
    hydra_compose: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F12, F04, F07-F10: CLI overrides beat experiment layer and cascade through ${...}."""
    out_root = str(e2e_workdir["outputs"])
    cfg = hydra_compose(
        "train",
        overrides=[
            "experiment=vosr2_onestep_text",
            "scale=2",
            "optim.optimizer.lr=5e-5",
            "trainer.total_iters=1234",
            f"output_root={hydra_value_formatter(out_root)}",
            "exp_name=cli_exp",
        ],
    )

    assert cfg.scale == 2
    assert cfg.arch.scale == 2
    assert cfg.data.scale == 2
    assert cfg.data.degradation.scale == 2
    assert cfg.metric.val_suites.set5.metrics.psnr.crop_border == 2
    assert cfg.optim.optimizer.lr == pytest.approx(5.0e-5)
    assert cfg.trainer.total_iters == 1234
    assert cfg.optim.scheduler.T_max == 1234
    assert cfg.work_dir == f"{out_root}/cli_exp"


def test_t1_infer_yaml_composition_and_tile_defaults(
    configs_dir: Path,
    hydra_compose: Callable[..., Any],
) -> None:
    """F11: configs/infer.yaml and infer/default.yaml compose cleanly with Hydra guards."""
    import yaml

    cfg = hydra_compose("infer")
    assert cfg.infer.exp_dir is None
    assert cfg.infer.weights_path is None
    assert cfg.infer.use_ema is True
    assert cfg.infer.recursive is False
    assert cfg.infer.suffix == ""
    assert cfg.infer.save_format == "png"

    tile = cfg.infer.tile
    assert tile.enabled is True
    assert tile.size == "auto"
    assert tile.overlap == 16
    assert tile.batch_size == 4
    assert tile.blend_mode == "cosine"
    assert tile.pad_multiple == 8

    raw_infer = yaml.safe_load((configs_dir / "infer.yaml").read_text(encoding="utf-8"))
    assert raw_infer["hydra"]["job"]["chdir"] is False
    assert raw_infer["hydra"]["output_subdir"] is None
    defaults = raw_infer.get("defaults", [])
    has_job_logging_disabled = any(
        (isinstance(item, dict) and item.get("override hydra/job_logging") == "disabled")
        or (isinstance(item, str) and "override hydra/job_logging" in item and "disabled" in item)
        for item in defaults
    )
    assert has_job_logging_disabled


def test_t1_user_reqs_config_invariants(
    configs_dir: Path,
    hydra_compose: Callable[..., Any],
) -> None:
    """F04, F05, F07, F09, F11, F20: Enforce User Reqs #2, #3, #4, and #5 (ToyModel & L1/LPIPS)."""
    import yaml

    cfg = hydra_compose("train")

    # User Req #2: Zero `mode` field in data.sources or non-degradation dataset YAML files.
    for src in cfg.data.sources:
        assert "mode" not in src, f"Legacy 'mode' key forbidden in data.sources: {src}"
    for yaml_file in sorted((configs_dir / "data").rglob("*.yaml")):
        rel_parts = yaml_file.relative_to(configs_dir / "data").parts
        parsed = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if isinstance(parsed, dict) and isinstance(parsed.get("sources"), list):
            for src_entry in parsed["sources"]:
                assert not _contains_key_recursive(src_entry, "mode"), (
                    f"Legacy 'mode' key forbidden in sources of {yaml_file}"
                )
        if "degradation" not in rel_parts:
            assert not _contains_key_recursive(parsed, "mode"), (
                f"Legacy 'mode' key forbidden in dataset config {yaml_file}"
            )

    # User Req #3: Validation datasets live ONLY in metric.val_suites, NEVER in data.
    forbidden_val_keys = {"val", "val_datasets", "val_suites", "val_sources", "validation"}
    assert not (forbidden_val_keys & set(cfg.data.keys())), (
        f"Validation config keys forbidden under data: {set(cfg.data.keys())}"
    )
    primary_ds = cfg.metric.primary_dataset
    primary_m = cfg.metric.primary_metric
    assert primary_ds in cfg.metric.val_suites
    assert primary_m in cfg.metric.val_suites[primary_ds].metrics

    # Scope check: ONLY vosr2 arch & vosr2_onestep_text experiment (placeholder configs removed).
    arch_yaml_names = {p.name for p in (configs_dir / "arch").glob("*.yaml")}
    assert arch_yaml_names == {"vosr2.yaml"}

    exp_yaml_names = {p.name for p in (configs_dir / "experiment").glob("*.yaml")}
    assert exp_yaml_names == {"vosr2_onestep_text.yaml"}

    loss_yaml_names = {p.name for p in (configs_dir / "loss").glob("*.yaml")}
    assert {"l1.yaml", "l1_lpips.yaml"}.issubset(loss_yaml_names)

    # Zero references to SRPredictor or removed placeholder archs/losses in parsed YAML.
    forbidden_tokens = (
        "SRPredictor",
        "ToyModel",
        "RRDBNet",
        "SwinIR",
        "TinySRNet",
        "CharbonnierLoss",
        "FFTLoss",
    )
    for yaml_file in sorted(configs_dir.rglob("*.yaml")):
        parsed = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        for token in forbidden_tokens:
            assert not _yaml_contains_token(parsed, token), (
                f"Forbidden reference {token!r} found in {yaml_file}"
            )


def test_t1_config_utils_work_dir_setup_and_resolved_roundtrip(
    e2e_workdir: dict[str, Path],
    hydra_compose: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F16: resolve_work_dir, setup_experiment_dirs, save/load_resolved_config roundtrip."""
    from utils.config import (
        load_resolved_config,
        resolve_work_dir,
        save_resolved_config,
        setup_experiment_dirs,
    )

    cfg = hydra_compose(
        "train",
        overrides=[
            f"output_root={hydra_value_formatter(str(e2e_workdir['work']))}",
            "exp_name=roundtrip_exp",
            "++trainer.append_timestamp=false",
        ],
    )

    work_dir = resolve_work_dir(cfg)
    assert isinstance(work_dir, Path)
    assert work_dir == e2e_workdir["work"] / "roundtrip_exp"

    dirs = setup_experiment_dirs(work_dir, is_main_process=True)
    assert isinstance(dirs, dict)
    assert work_dir.is_dir()
    for subdir in ("checkpoints", "weights", "visuals", "trackers"):
        assert (work_dir / subdir).is_dir(), f"Expected subdirectory {subdir!r} to be created"

    resolved_path = work_dir / "config_resolved.yaml"
    save_resolved_config(cfg, resolved_path)
    assert resolved_path.is_file()

    raw_text = resolved_path.read_text(encoding="utf-8")
    assert "${" not in raw_text, "Saved config_resolved.yaml must have zero unresolved ${...}"

    loaded = load_resolved_config(resolved_path)
    assert loaded.exp_name == "roundtrip_exp"
    assert loaded.scale == 1
    assert loaded.work_dir == str(work_dir)
    assert loaded.optim.scheduler.T_max == cfg.trainer.total_iters


def test_t1_config_utils_normalize_mp_and_flatten_for_tracker(
    hydra_compose: Callable[..., Any],
) -> None:
    """F16: normalize_mixed_precision and flatten_config_for_tracker contracts."""
    from utils.config import flatten_config_for_tracker, normalize_mixed_precision

    assert normalize_mixed_precision("no") == "no"
    assert normalize_mixed_precision("fp16") == "fp16"
    assert normalize_mixed_precision("bf16") == "bf16"
    assert normalize_mixed_precision(False) == "no"
    assert normalize_mixed_precision(None) == "no"

    cfg = hydra_compose("train", overrides=["experiment=vosr2_onestep_text"])
    flat = flatten_config_for_tracker(cfg)
    assert isinstance(flat, dict) and len(flat) > 0

    assert flat.get("scale") == 1
    assert flat.get("trainer.total_iters") == 10000
    assert flat.get("optim.optimizer.lr") == pytest.approx(5.0e-6)

    for key, val in flat.items():
        assert isinstance(key, str), f"Flattened key must be str, got {type(key)}"
        assert isinstance(val, (int, float, str, bool)), (
            f"Flattened tracker value for {key!r} must be primitive, got {type(val)}: {val!r}"
        )
        if isinstance(val, str):
            assert "${" not in val, f"Unresolved interpolation leaked in {key!r}: {val!r}"
