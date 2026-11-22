"""Tier 2 E2E tests — Area A: Hydra Configuration, Inheritance & Packaging Boundaries.

Covers boundary, edge, and negative contracts for F01-F12, F16, F20, F35 and
User Requirements #2-#5. All non-stdlib imports (besides ``pytest``) are
strictly deferred inside test functions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _collect_targets_recursive(obj: Any, out: list[Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "_target_":
                out.append(v)
            _collect_targets_recursive(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_targets_recursive(item, out)


def _iter_yaml_strings(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                found.append(k)
            found.extend(_iter_yaml_strings(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_iter_yaml_strings(item))
    elif isinstance(obj, str):
        found.append(obj)
    return found


def _collect_dict_keys_recursive(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                keys.add(k)
            keys.update(_collect_dict_keys_recursive(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_dict_keys_recursive(item))
    return keys


def test_t2_a01_normalize_mixed_precision_coercions_and_rejections() -> None:
    """F16: normalize_mixed_precision handles YAML bool/null/case coercions and rejects bad inputs."""
    from utils.config import normalize_mixed_precision

    for no_val in (False, None, "", "none", "None", "NO", "No", "no"):
        assert normalize_mixed_precision(no_val) == "no", f"Expected 'no' for {no_val!r}"

    assert normalize_mixed_precision("fp16") == "fp16"
    assert normalize_mixed_precision("FP16") == "fp16"
    assert normalize_mixed_precision("bf16") == "bf16"
    assert normalize_mixed_precision("BF16") == "bf16"

    for bad_val in ("fp8", "int4", True, 123, 0, []):
        with pytest.raises(ValueError, match="."):
            normalize_mixed_precision(bad_val)


def test_t2_a02_setup_experiment_dirs_non_main_noop_and_idempotency(
    e2e_workdir: dict[str, Path],
) -> None:
    """F16: setup_experiment_dirs is a filesystem no-op on non-main ranks and idempotent on main."""
    from utils.config import setup_experiment_dirs

    non_main_dir = e2e_workdir["work"] / "rank1_noop"
    dirs_nm = setup_experiment_dirs(non_main_dir, is_main_process=False)

    assert not non_main_dir.exists(), "Non-main process must not create directories on disk"
    assert isinstance(dirs_nm, dict)
    assert {"checkpoints", "weights", "visuals", "trackers"}.issubset(dirs_nm.keys())
    for key, path in dirs_nm.items():
        assert isinstance(path, Path), f"Expected Path for {key!r}, got {type(path)}"
        assert not path.exists(), f"Non-main process unexpectedly created {path}"

    main_dir = e2e_workdir["work"] / "rank0_idempotent"
    dirs1 = setup_experiment_dirs(main_dir, is_main_process=True)
    assert main_dir.is_dir()
    for sub in ("checkpoints", "weights", "visuals", "trackers"):
        assert dirs1[sub].is_dir(), f"Expected subdirectory {sub!r} to exist"

    sentinel = dirs1["weights"] / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    dirs2 = setup_experiment_dirs(main_dir, is_main_process=True)
    assert dirs1 == dirs2
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_t2_a03_resolve_work_dir_timestamp_toggle(
    e2e_workdir: dict[str, Path],
    hydra_compose: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F16: resolve_work_dir returns exact path when append_timestamp=False and suffix when True."""
    from utils.config import resolve_work_dir

    work_root = e2e_workdir["work"]
    cfg_bare = hydra_compose(
        "train",
        overrides=[
            f"output_root={hydra_value_formatter(str(work_root))}",
            "exp_name=ts_exp",
            "++trainer.append_timestamp=false",
            "trainer.resume=null",
        ],
    )
    wd_bare = resolve_work_dir(cfg_bare)
    assert wd_bare == work_root / "ts_exp"

    cfg_ts = hydra_compose(
        "train",
        overrides=[
            f"output_root={hydra_value_formatter(str(work_root))}",
            "exp_name=ts_exp",
            "++trainer.append_timestamp=true",
            "trainer.resume=null",
        ],
    )
    wd_ts = resolve_work_dir(cfg_ts)
    assert isinstance(wd_ts, Path)
    assert wd_ts != wd_bare
    assert wd_ts.parent == wd_bare.parent
    assert re.fullmatch(r"ts_exp_\d{8}_\d{6}", wd_ts.name), f"Unexpected timestamp dir: {wd_ts.name}"


def test_t2_a04_load_resolved_config_missing_and_edge_roundtrip(
    e2e_workdir: dict[str, Path],
    hydra_compose: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F16: load_resolved_config fails fast on missing files and preserves edge YAML types."""
    from utils.config import load_resolved_config, save_resolved_config

    with pytest.raises(FileNotFoundError):
        load_resolved_config(e2e_workdir["work"] / "missing_resolved.yaml")

    cfg = hydra_compose(
        "train",
        overrides=[
            f"output_root={hydra_value_formatter(str(e2e_workdir['work']))}",
            "exp_name=rt_edge",
            "trainer.mixed_precision='no'",
            "trainer.resume=null",
            "optim.optimizer.lr=1e-4",
            "optim.optimizer.betas=[0.9,0.99]",
        ],
    )
    out_path = e2e_workdir["work"] / "rt_edge" / "config_resolved.yaml"
    save_resolved_config(cfg, out_path)
    assert out_path.is_file()
    assert "${" not in out_path.read_text(encoding="utf-8")

    loaded = load_resolved_config(out_path)
    assert loaded.trainer.mixed_precision == "no"
    assert type(loaded.trainer.mixed_precision) is str
    assert loaded.trainer.resume is None
    assert list(loaded.optim.optimizer.betas) == [0.9, 0.99]
    assert isinstance(loaded.optim.optimizer.lr, float)
    assert loaded.optim.optimizer.lr == pytest.approx(1.0e-4)


def test_t2_a05_flatten_config_for_tracker_edge_values() -> None:
    """F16: flatten_config_for_tracker sanitizes None, empty containers, lists, and interpolations."""
    from omegaconf import OmegaConf

    from utils.config import flatten_config_for_tracker

    cfg = OmegaConf.create(
        {
            "none_val": None,
            "empty_dict": {},
            "empty_list": [],
            "data": {
                "sources": [
                    {
                        "name": "df2k",
                        "root_dir": "/d1",
                        "json_path": "/d1.json",
                        "ratio": 0.7,
                    },
                    {
                        "name": "real",
                        "root_dir": "/d2",
                        "json_path": "/d2.json",
                        "ratio": 0.3,
                    },
                ]
            },
            "flags": {"ema": True, "compile": False},
            "nums": {"iters": 1000, "lr": 1.0e-4},
            "base": 4,
            "interp_int": "${base}",
            "interp_str": "x${base}_run",
        }
    )

    flat = flatten_config_for_tracker(cfg)
    assert isinstance(flat, dict) and len(flat) > 0
    for k, v in flat.items():
        assert isinstance(k, str), f"Non-str key in flattened config: {k!r}"
        assert isinstance(v, (int, float, str, bool)), (
            f"Unsupported tracker hparam type for {k!r}: {type(v)}"
        )
        if isinstance(v, str):
            assert "${" not in v, f"Unresolved interpolation remained in {k!r}: {v!r}"

    assert flat["flags.ema"] is True
    assert flat["flags.compile"] is False
    assert flat["nums.iters"] == 1000
    assert flat["nums.lr"] == pytest.approx(1.0e-4)
    assert flat["interp_int"] == 4
    assert flat["interp_str"] == "x4_run"
    assert flat["data.sources.0.name"] == "df2k"
    assert flat["data.sources.1.ratio"] == pytest.approx(0.3)
    for edge_key in ("none_val", "empty_dict", "empty_list"):
        assert edge_key in flat
        assert isinstance(flat[edge_key], str)


def test_t2_a06_hydra_composition_invalid_overrides_fail_fast(
    hydra_compose: Callable[..., Any],
) -> None:
    """F03, F04, F05, F12, Req #5: Invalid Hydra overrides and removed configs fail fast."""
    from hydra.errors import ConfigCompositionException

    invalid_override_cases = [
        ["experiment=nonexistent_exp_xyz"],
        ["arch=rrdbnet_x4"],
        ["arch=swinir_x4"],
        ["loss=charbonnier"],
        ["nonexistent_group=foo"],
    ]
    for overrides in invalid_override_cases:
        with pytest.raises(ConfigCompositionException):
            hydra_compose("train", overrides=overrides)


def test_t2_a07_hydra_cli_edge_overrides_and_partial_loss_override(
    hydra_compose: Callable[..., Any],
) -> None:
    """F05, F10, F12, F16: Edge CLI overrides for trackers, mixed_precision, and partial loss weight."""
    from utils.config import normalize_mixed_precision

    cfg_null_log = hydra_compose("train", overrides=["trainer.log_with=null"])
    assert cfg_null_log.trainer.log_with is None

    cfg_multi_log = hydra_compose("train", overrides=["trainer.log_with=[tensorboard,wandb]"])
    assert list(cfg_multi_log.trainer.log_with) == ["tensorboard", "wandb"]

    cfg_mp_no = hydra_compose("train", overrides=["trainer.mixed_precision=no"])
    assert normalize_mixed_precision(cfg_mp_no.trainer.mixed_precision) == "no"

    cfg_zero_lpips = hydra_compose(
        "train",
        overrides=["experiment=exp01_toy_mixed_x4", "loss.lpips.weight=0.0"],
    )
    assert cfg_zero_lpips.loss.lpips.weight == pytest.approx(0.0)
    assert cfg_zero_lpips.loss.l1.weight == pytest.approx(1.0)
    assert cfg_zero_lpips.loss.l1.criterion._target_ == "losses.L1Loss"
    assert cfg_zero_lpips.loss.lpips.criterion._target_ == "losses.LPIPSLoss"


def test_t2_a08_realesrgan_2nd_standalone_schema_and_bounds(configs_dir: Path) -> None:
    """F07: realesrgan_2nd.yaml standalone schema, normalized probabilities, and numeric bounds."""
    import yaml

    deg_path = configs_dir / "data" / "degradation" / "realesrgan_2nd.yaml"
    raw = yaml.safe_load(deg_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert raw["_target_"] == "data.degradation.RealESRGANDegrader"
    assert raw["scale"] == "${scale}"
    assert isinstance(raw["use_usm"], bool)
    assert isinstance(raw["queue_size"], int) and raw["queue_size"] >= 0

    kcfg = raw["kernel_cfg"]
    assert isinstance(kcfg, dict)
    ksize = kcfg["kernel_size"]
    assert isinstance(ksize, int) and ksize >= 3 and ksize % 2 == 1

    krange = kcfg["kernel_range"]
    assert isinstance(krange, list) and len(krange) >= 1
    assert all(isinstance(k, int) and 1 <= k <= ksize and k % 2 == 1 for k in krange)

    for list_key, prob_key in (("kernel_list", "kernel_prob"), ("kernel_list2", "kernel_prob2")):
        klist = kcfg[list_key]
        kprob = kcfg[prob_key]
        assert len(klist) == 6 and len(kprob) == 6
        assert all(float(p) >= 0.0 for p in kprob)
        assert sum(float(p) for p in kprob) == pytest.approx(1.0, abs=1e-5)

    for range_key in (
        "blur_sigma",
        "blur_sigma2",
        "betag_range",
        "betag_range2",
        "betap_range",
        "betap_range2",
    ):
        bounds = kcfg[range_key]
        assert len(bounds) == 2 and float(bounds[0]) <= float(bounds[1]), (
            f"Invalid kernel_cfg range {range_key!r}: {bounds!r}"
        )

    for range_key in (
        "resize_range",
        "resize_range2",
        "noise_range",
        "noise_range2",
        "poisson_scale_range",
        "poisson_scale_range2",
        "jpeg_range",
        "jpeg_range2",
    ):
        bounds = raw[range_key]
        assert len(bounds) == 2 and float(bounds[0]) <= float(bounds[1]), (
            f"Invalid degradation range {range_key!r}: {bounds!r}"
        )

    for jpeg_key in ("jpeg_range", "jpeg_range2"):
        lo, hi = float(raw[jpeg_key][0]), float(raw[jpeg_key][1])
        assert 1.0 <= lo <= hi <= 100.0, f"JPEG range out of [1, 100]: {jpeg_key}={raw[jpeg_key]}"

    for rprob_key in ("resize_prob", "resize_prob2"):
        rprob = raw[rprob_key]
        assert len(rprob) == 3 and all(float(p) >= 0.0 for p in rprob)
        assert sum(float(p) for p in rprob) == pytest.approx(1.0, abs=1e-5)

    for prob_container, prob_keys in (
        (kcfg, ("sinc_prob", "sinc_prob2", "final_sinc_prob")),
        (
            raw,
            (
                "second_blur_prob",
                "gaussian_noise_prob",
                "gaussian_noise_prob2",
                "gray_noise_prob",
                "gray_noise_prob2",
            ),
        ),
    ):
        for pkey in prob_keys:
            val = float(prob_container[pkey])
            assert 0.0 <= val <= 1.0, f"Probability {pkey!r} out of [0, 1]: {val}"


def test_t2_a09_static_audit_all_configs_yaml_contracts(configs_dir: Path) -> None:
    """F01-F12, F20, F35, Reqs #2-#5: Whole-tree static audit of configs/**/*.yaml contracts."""
    import yaml

    yaml_files = sorted(configs_dir.rglob("*.yaml"))
    assert len(yaml_files) >= 15, f"Expected >= 15 YAML config files, found {len(yaml_files)}"

    forbidden_tokens = (
        "SRPredictor",
        "RRDBNet",
        "SwinIR",
        "TinySRNet",
        "CharbonnierLoss",
        "FFTLoss",
    )
    forbidden_val_keys = {"val", "val_datasets", "val_suites", "val_sources", "validation"}

    for yaml_file in yaml_files:
        parsed = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict), f"Config {yaml_file} must parse to a YAML mapping"

        targets: list[Any] = []
        _collect_targets_recursive(parsed, targets)
        for target in targets:
            assert isinstance(target, str) and target.startswith(
                ("archs.", "data.", "engine.", "losses.", "metrics.", "models.", "utils.", "torch.")
            ), f"Invalid _target_ {target!r} in {yaml_file}"

        for token_str in _iter_yaml_strings(parsed):
            assert "srpredictor" not in token_str.lower(), (
                f"Forbidden SRPredictor reference in {yaml_file}: {token_str!r}"
            )
            for forbidden in forbidden_tokens:
                assert forbidden not in token_str, (
                    f"Forbidden token {forbidden!r} in {yaml_file}: {token_str!r}"
                )

    data_dir = configs_dir / "data"
    for data_yaml in sorted(data_dir.rglob("*.yaml")):
        parsed_data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        all_keys = _collect_dict_keys_recursive(parsed_data)
        bad_val_keys = all_keys & forbidden_val_keys
        assert not bad_val_keys, (
            f"Validation keys {sorted(bad_val_keys)} forbidden under configs/data/: {data_yaml}"
        )

        sources = parsed_data.get("sources") if isinstance(parsed_data, dict) else None
        if isinstance(sources, list):
            for entry in sources:
                assert not (isinstance(entry, dict) and "mode" in entry), (
                    f"Legacy 'mode' key forbidden in sources entry of {data_yaml}"
                )
        if "degradation" not in data_yaml.relative_to(data_dir).parts:
            assert "mode" not in all_keys, f"Legacy 'mode' key forbidden in {data_yaml}"

    arch_files = {p.name for p in (configs_dir / "arch").glob("*.yaml")}
    assert arch_files == {"toy_x4.yaml"}, f"Unexpected arch configs: {sorted(arch_files)}"

    loss_files = {p.name for p in (configs_dir / "loss").glob("*.yaml")}
    assert loss_files == {"l1.yaml", "l1_lpips.yaml"}, (
        f"Unexpected loss configs: {sorted(loss_files)}"
    )
