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

    # Confirm src/srtrainer layout and editable import resolution.
    assert (repo_root / "src" / "srtrainer" / "__init__.py").is_file()
    import srtrainer

    pkg_file = Path(srtrainer.__file__).resolve()
    expected_pkg_root = (repo_root / "src" / "srtrainer").resolve()
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
    """F02, Req #5: Short-path __all__ re-exports expose ONLY ToyModel and L1Loss/LPIPSLoss."""
    import hydra.utils

    import srtrainer.archs as archs_pkg
    import srtrainer.losses as losses_pkg

    assert hasattr(archs_pkg, "__all__")
    assert set(archs_pkg.__all__) == {"ToyModel"}, (
        f"srtrainer.archs.__all__ must export ONLY ToyModel, got {archs_pkg.__all__!r}"
    )
    assert hydra.utils.get_class("srtrainer.archs.ToyModel") is archs_pkg.ToyModel
    for forbidden_arch in ("RRDBNet", "SwinIR", "TinySRNet"):
        assert not hasattr(archs_pkg, forbidden_arch), (
            f"Forbidden architecture {forbidden_arch!r} must not exist in srtrainer.archs"
        )

    assert hasattr(losses_pkg, "__all__")
    assert set(losses_pkg.__all__) == {"L1Loss", "LPIPSLoss"}, (
        f"srtrainer.losses.__all__ must export ONLY L1Loss & LPIPSLoss, got {losses_pkg.__all__!r}"
    )
    for sym in ("L1Loss", "LPIPSLoss"):
        assert hydra.utils.get_class(f"srtrainer.losses.{sym}") is getattr(losses_pkg, sym)
    for forbidden_loss in ("CharbonnierLoss", "FFTLoss"):
        assert not hasattr(losses_pkg, forbidden_loss), (
            f"Forbidden loss {forbidden_loss!r} must not exist in srtrainer.losses"
        )


def test_t1_train_yaml_default_composition_and_hydra_guards(
    configs_dir: Path,
    hydra_compose: Callable[..., Any],
) -> None:
    """F03-F10: train.yaml defaults to toy_x4, composes all groups, enforces Hydra guards."""
    import yaml

    cfg = hydra_compose("train")

    assert cfg.exp_name == "debug_sr_x4"
    assert cfg.scale == 4
    assert cfg.seed == 42
    assert cfg.output_root == "experiments"
    assert cfg.work_dir == "experiments/debug_sr_x4"

    for group in ("arch", "loss", "model", "data", "optim", "metric", "trainer"):
        assert group in cfg, f"Missing atomic group {group!r} in composed train.yaml"

    assert cfg.arch._target_ == "srtrainer.archs.ToyModel"
    assert cfg.arch.scale == 4
    assert cfg.data.scale == 4
    assert cfg.optim.scheduler.T_max == cfg.trainer.total_iters
    assert cfg.model._target_ == "srtrainer.models.SRModel"
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
) -> None:
    """F04, F05, F08: Atomic toy_x4 arch, l1/l1_lpips losses, and partial optims instantiate."""
    import hydra.utils
    import torch
    import torch.nn as nn

    from srtrainer.archs import ToyModel
    from srtrainer.losses import L1Loss, LPIPSLoss

    cfg = hydra_compose(
        "train",
        overrides=[
            "arch=toy_x4",
            "arch.num_feat=8",
            "arch.num_block=1",
            "loss=l1_lpips",
            "++loss.lpips.criterion.pretrained=false",
            "optim=adamw_cosine",
            "trainer.total_iters=100",
        ],
    )

    net = hydra.utils.instantiate(cfg.arch)
    assert isinstance(net, ToyModel)
    assert isinstance(net, nn.Module)

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

    # Verify toy_x4 scale interpolation cascades from global scale.
    cfg_toy_x2 = hydra_compose("train", overrides=["arch=toy_x4", "scale=2"])
    assert cfg_toy_x2.arch._target_ == "srtrainer.archs.ToyModel"
    assert cfg_toy_x2.arch.scale == 2


def test_t1_multilevel_experiment_inheritance(
    hydra_compose: Callable[..., Any],
) -> None:
    """F12, F03: Multi-level # @package _global_ inheritance (base_sr_x4 -> exp01_toy_mixed_x4)."""
    base_cfg = hydra_compose("train", overrides=["experiment=base_sr_x4"])
    assert base_cfg.scale == 4
    assert base_cfg.trainer.total_iters == 300000
    assert base_cfg.trainer.val_interval == 5000
    assert base_cfg.trainer.save_interval == 5000
    assert base_cfg.trainer.mixed_precision == "bf16"
    assert base_cfg.trainer.gradient_accumulation_steps == 1

    exp_cfg = hydra_compose("train", overrides=["experiment=exp01_toy_mixed_x4"])
    assert exp_cfg.exp_name == "exp01_toy_l1_lpips_x4"
    assert exp_cfg.work_dir == "experiments/exp01_toy_l1_lpips_x4"
    assert exp_cfg.arch._target_ == "srtrainer.archs.ToyModel"
    assert exp_cfg.arch.num_feat == 16

    # Partial overrides preserve sibling keys from inherited groups.
    assert exp_cfg.loss.lpips.weight == pytest.approx(0.05)
    assert exp_cfg.loss.l1.weight == pytest.approx(1.0)
    assert exp_cfg.optim.optimizer.lr == pytest.approx(1.0e-4)
    assert list(exp_cfg.optim.optimizer.betas) == [0.9, 0.99]

    # Cascading interpolation & inherited trainer settings from base_sr_x4.
    assert exp_cfg.trainer.total_iters == 200000
    assert exp_cfg.optim.scheduler.T_max == 200000
    assert exp_cfg.trainer.val_interval == 5000
    assert exp_cfg.trainer.mixed_precision == "bf16"


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
            "experiment=exp01_toy_mixed_x4",
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

    # User Req #5 (Scope Narrowing): ONLY toy_x4 arch & exp01_toy_mixed_x4 & L1/LPIPS losses.
    arch_yaml_names = {p.name for p in (configs_dir / "arch").glob("*.yaml")}
    assert "toy_x4.yaml" in arch_yaml_names
    assert "rrdbnet_x4.yaml" not in arch_yaml_names
    assert "swinir_x4.yaml" not in arch_yaml_names

    exp_yaml_names = {p.name for p in (configs_dir / "experiment").glob("*.yaml")}
    assert "exp01_toy_mixed_x4.yaml" in exp_yaml_names
    assert "exp01_rrdb_mixed_x4.yaml" not in exp_yaml_names

    loss_yaml_names = {p.name for p in (configs_dir / "loss").glob("*.yaml")}
    assert {"l1.yaml", "l1_lpips.yaml"}.issubset(loss_yaml_names)

    # User Req #4 & #5: Zero references to SRPredictor or removed archs/losses in parsed YAML.
    forbidden_tokens = (
        "SRPredictor",
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
    from srtrainer.utils.config import (
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
    assert loaded.scale == 4
    assert loaded.work_dir == str(work_dir)
    assert loaded.optim.scheduler.T_max == cfg.trainer.total_iters


def test_t1_config_utils_normalize_mp_and_flatten_for_tracker(
    hydra_compose: Callable[..., Any],
) -> None:
    """F16: normalize_mixed_precision and flatten_config_for_tracker contracts."""
    from srtrainer.utils.config import flatten_config_for_tracker, normalize_mixed_precision

    assert normalize_mixed_precision("no") == "no"
    assert normalize_mixed_precision("fp16") == "fp16"
    assert normalize_mixed_precision("bf16") == "bf16"
    assert normalize_mixed_precision(False) == "no"
    assert normalize_mixed_precision(None) == "no"

    cfg = hydra_compose("train", overrides=["experiment=exp01_toy_mixed_x4"])
    flat = flatten_config_for_tracker(cfg)
    assert isinstance(flat, dict) and len(flat) > 0

    assert flat.get("scale") == 4
    assert flat.get("trainer.total_iters") == 200000
    assert flat.get("optim.optimizer.lr") == pytest.approx(1.0e-4)

    for key, val in flat.items():
        assert isinstance(key, str), f"Flattened key must be str, got {type(key)}"
        assert isinstance(val, (int, float, str, bool)), (
            f"Flattened tracker value for {key!r} must be primitive, got {type(val)}: {val!r}"
        )
        if isinstance(val, str):
            assert "${" not in val, f"Unresolved interpolation leaked in {key!r}: {val!r}"
