"""Tier 4 E2E tests — Real-World Multi-Feature Application Scenarios.

Covers 6 complete end-user workflows exercising ``tools/train.py``,
``tools/infer.py``, and 2-process CPU ``torchrun`` end-to-end using exclusively
``ToyModel`` and ``L1Loss``/``LPIPSLoss(pretrained=False)``.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.conftest import vosr_infer_overrides

_CANONICAL_ITER_LOG_RE = re.compile(
    r"\[Iter\s+\d+/\d+\].*lr[^:]*:\s*[0-9.eE+-]+.*loss/total:\s*[0-9.eE+-]+.*it/s.*mem.*ETA"
)


def _extract_iter_lines(log_text: str, step: int, total_iters: int) -> list[str]:
    pat = re.compile(rf"\[Iter\s+0*{step}/0*{total_iters}\]")
    return [line for line in log_text.splitlines() if pat.search(line)]


def _extract_lr_from_line(line: str) -> float:
    m = re.search(r"lr[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", line)
    assert m is not None, f"Could not parse lr from log line: {line!r}"
    return float(m.group(1))


def _py_ast_contains_symbol(source: str, symbol: str) -> bool:
    """Return True if Python AST references ``symbol`` outside comments and docstrings."""
    tree = ast.parse(source)
    docstring_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_ids.add(id(node.body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                return True
        elif isinstance(node, ast.Name):
            if node.id == symbol:
                return True
        elif isinstance(node, ast.Attribute):
            if node.attr == symbol:
                return True
        elif isinstance(node, ast.alias):
            if symbol in node.name.split(".") or node.asname == symbol:
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and symbol in node.module.split("."):
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstring_ids and symbol in node.value:
                return True
    return False


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


def _collect_dict_keys(obj: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                keys.add(k)
            keys.update(_collect_dict_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            keys.update(_collect_dict_keys(item))
    return keys


def test_t4_scenario1_mixed_blind_paired_train_to_tiled_infer(
    e2e_workdir: dict[str, Path],
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    minimal_train_overrides_factory: Callable[..., list[str]],
    train_cli_runner: Callable[..., Any],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """Scenario 1: Mixed blind+paired exp01_toy_mixed_x4 training -> tools/infer.py x4."""
    import hydra.utils
    import torch
    from PIL import Image

    from engine.inferencer import TiledInferencer
    from utils.config import load_resolved_config
    from utils.image import load_image
    from utils.misc import load_weights

    gt_src = gt_only_train_source_factory(name="s1_gt", n_samples=4, gt_h=80, gt_w=80, ratio=0.6)
    pair_src = paired_train_source_factory(
        name="s1_pair", n_samples=4, lq_h=20, lq_w=20, scale=4, ratio=0.4
    )
    assert "mode" not in gt_src and "mode" not in pair_src

    val_info = val_suite_gt_label_factory(name="set5", scale=4, sizes=[(16, 16), (16, 20)])
    val_suites = {
        "set5": {
            "dataset": {"root_dir": val_info["root_dir"], "json_path": val_info["json_path"]},
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 4,
                    "test_y_channel": True,
                },
                "ssim": {
                    "_target_": "metrics.SSIMMetric",
                    "crop_border": 4,
                    "test_y_channel": True,
                },
            },
        }
    }

    infer_img = synthetic_image_factory(
        e2e_workdir["data"] / "s1_infer" / "test_24x28.png",
        width=28,
        height=24,
        pattern="gradient",
        seed=41,
    )

    exp_name = "s1_mixed_x4"
    exp_dir = e2e_workdir["work"] / exp_name
    train_ovs = minimal_train_overrides_factory(
        e2e_workdir["work"],
        exp_name,
        [gt_src, pair_src],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=4,
        gt_size=64,
        batch_size=2,
        total_iters=2,
        log_interval=1,
        val_interval=2,
        save_interval=2,
        max_keep_states=2,
        log_with=None,
        use_ema=True,
        has_degradation=True,
        arch_num_feat=8,
        arch_num_block=1,
        experiment="vosr2_onestep_text",
        extra_overrides=[
            "loss=l1_lpips",
            "loss.lpips.weight=0.05",
            "loss.lpips.criterion._target_=tests.helpers.fake_backends.DummyLPIPSLoss",
            "optim.optimizer.lr=1.0e-4",
            "data.degradation.kernel_cfg.kernel_size=7",
            "data.degradation.kernel_cfg.kernel_range=[7]",
        ],
    )
    train_proc = train_cli_runner(train_ovs, check=True)
    assert train_proc.returncode == 0

    resolved_path = exp_dir / "config_resolved.yaml"
    assert resolved_path.is_file()
    assert "${" not in resolved_path.read_text(encoding="utf-8")
    resolved_cfg = load_resolved_config(resolved_path)
    assert resolved_cfg.scale == 4
    assert resolved_cfg.data.gt_size == 64
    assert resolved_cfg.loss.lpips.weight == pytest.approx(0.05)
    assert resolved_cfg.optim.optimizer.lr == pytest.approx(1.0e-4)
    assert resolved_cfg.arch.vae._target_ == "tests.dummy_arch.DummyVAE"
    assert resolved_cfg.arch.dit._target_ == "tests.dummy_arch.DummyDiT"
    assert resolved_cfg.arch.dino._target_ == "tests.dummy_arch.DummyDino"

    log_text = (exp_dir / "train.log").read_text(encoding="utf-8")
    assert _CANONICAL_ITER_LOG_RE.search(log_text), (
        f"train.log did not match canonical summary format:\n{log_text}"
    )
    for token in ("loss/l1", "loss/lpips", "loss/total"):
        assert token in log_text, f"Missing {token!r} in train.log:\n{log_text}"

    assert (exp_dir / "checkpoints" / "state_iter_000002").is_dir()
    for wname in (
        "iter_000002_vae.safetensors",
        "iter_000002_dit.safetensors",
        "iter_000002_dit_ema.safetensors",
        "iter_000002_dino.safetensors",
        "latest_vae.safetensors",
        "latest_dit.safetensors",
        "latest_dit_ema.safetensors",
        "latest_dino.safetensors",
        "best_set5_psnr_vae.safetensors",
        "best_set5_psnr_dit.safetensors",
        "best_set5_psnr_dino.safetensors",
    ):
        assert (exp_dir / "weights" / wname).is_file(), f"Missing weight artifact {wname}"

    vis_pngs = sorted((exp_dir / "visuals").rglob("*.png"))
    assert len(vis_pngs) >= 1
    expected_3panel_widths = {3 * (lq_w * 4) for _, lq_w in val_info["lq_sizes"]}
    for png in vis_pngs:
        with Image.open(png) as im:
            assert im.size[0] in expected_3panel_widths

    out_dir = e2e_workdir["outputs"] / "s1_infer_out"
    fmt = hydra_value_formatter
    infer_proc = infer_cli_runner(
        [
            *vosr_infer_overrides(
                4, exp_dir / "weights", vae="best_set5_psnr_vae.safetensors",
                dit="best_set5_psnr_dit.safetensors", dino="best_set5_psnr_dino.safetensors",
            ),
            f"infer.input={fmt(str(infer_img.parent))}",
            f"infer.output_dir={fmt(str(out_dir))}",
            "infer.mixed_precision='no'",
            "infer.tile.enabled=true",
            "infer.tile.size=[16,16]",
            "infer.tile.overlap=4",
            "infer.tile.blend_mode=cosine",
        ],
        check=True,
    )
    assert infer_proc.returncode == 0

    out_png = out_dir / "test_24x28.png"
    assert out_png.is_file()
    sr_cli = load_image(out_png)
    assert sr_cli.shape == (3, 96, 112)
    assert sr_cli.dtype == torch.float32
    assert torch.isfinite(sr_cli).all()
    assert float(sr_cli.min()) >= 0.0 and float(sr_cli.max()) <= 1.0

    from models import VOSRPipeline

    ref_vae = hydra.utils.instantiate(resolved_cfg.arch.vae)
    ref_dit = hydra.utils.instantiate(resolved_cfg.arch.dit)
    ref_dino = hydra.utils.instantiate(resolved_cfg.arch.dino)
    load_weights(ref_vae, exp_dir / "weights" / "best_set5_psnr_vae.safetensors", strict=True)
    load_weights(ref_dit, exp_dir / "weights" / "best_set5_psnr_dit.safetensors", strict=True)
    load_weights(ref_dino, exp_dir / "weights" / "best_set5_psnr_dino.safetensors", strict=True)
    net_ref = VOSRPipeline(vae=ref_vae, dit=ref_dit, dino=ref_dino, scale=4, eval_seed=42)
    inf_ref = TiledInferencer(
        model=net_ref,
        scale=4,
        tile_enabled=True,
        tile_size=None,
        tile_overlap=4,
        tile_batch_size=4,
        blend_mode="cosine",
        pad_multiple=8,
        gt_size=64,
        mixed_precision="no",
        device="cpu",
    )
    sr_direct = inf_ref.infer_tensor(load_image(infer_img).unsqueeze(0)).squeeze(0)
    assert (sr_cli - sr_direct).abs().max().item() <= (1.0 / 255.0) + 1e-4


def test_t4_scenario2_preempted_run_resume_auto_consistency(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    minimal_train_overrides_factory: Callable[..., list[str]],
    train_cli_runner: Callable[..., Any],
) -> None:
    """Scenario 2: Preempted run + resume=auto matches uninterrupted 4-step baseline + idempotent relaunch."""
    import torch
    from safetensors.torch import load_file

    # Exact-size 16x16 -> 32x32 paired sources across 2 sources (zero crop randomness).
    s0 = paired_train_source_factory(name="s2_src0", n_samples=4, lq_h=16, lq_w=16, scale=2, ratio=0.6)
    s1 = paired_train_source_factory(name="s2_src1", n_samples=4, lq_h=16, lq_w=16, scale=2, ratio=0.4)
    val_info = val_suite_gt_label_factory(name="set5", scale=2, sizes=[(16, 16), (16, 20)])
    val_suites = {
        "set5": {
            "dataset": {"root_dir": val_info["root_dir"], "json_path": val_info["json_path"]},
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                }
            },
        }
    }

    det_extras = [
        "seed=42",
        "optim.scheduler.T_max=4",
        "data.augment.hflip=false",
        "data.augment.vflip=false",
        "data.augment.rot90=false",
    ]

    def _run(exp_name: str, total_iters: int, *, resume_auto: bool = False) -> Path:
        extras = list(det_extras)
        if resume_auto:
            extras.append("trainer.resume=auto")
        ovs = minimal_train_overrides_factory(
            e2e_workdir["work"],
            exp_name,
            [s0, s1],
            val_suites=val_suites,
            primary_dataset="set5",
            primary_metric="psnr",
            scale=2,
            gt_size=32,
            batch_size=2,
            total_iters=total_iters,
            grad_accum=2,
            log_interval=1,
            val_interval=2,
            save_interval=2,
            max_keep_states=2,
            has_degradation=False,
            use_ema=True,
            log_with=None,
            extra_overrides=extras,
        )
        train_cli_runner(ovs, check=True)
        return e2e_workdir["work"] / exp_name

    # 1. Run A: Uninterrupted 4-step baseline
    dir_a = _run("s2_baseline", total_iters=4, resume_auto=False)

    # 2. Run B Part 1: Preempted at step 2 (with identical T_max=4!)
    dir_b = _run("s2_preempted", total_iters=2, resume_auto=False)
    assert (dir_b / "checkpoints" / "state_iter_000002").is_dir()

    # 3. Run B Part 2: Resume auto steps 2 -> 4
    _run("s2_preempted", total_iters=4, resume_auto=True)

    ckpts_b = {p.name for p in (dir_b / "checkpoints").iterdir() if p.is_dir()}
    assert len(ckpts_b) <= 2
    assert ckpts_b == {"state_iter_000002", "state_iter_000004"}

    log_a = (dir_a / "train.log").read_text(encoding="utf-8")
    log_b = (dir_b / "train.log").read_text(encoding="utf-8")
    lr_a = _extract_lr_from_line(_extract_iter_lines(log_a, 4, 4)[-1])
    lr_b = _extract_lr_from_line(_extract_iter_lines(log_b, 4, 4)[-1])
    assert lr_b == pytest.approx(lr_a, rel=1e-4)

    for wname in (
        "latest_vae.safetensors",
        "latest_dit.safetensors",
        "latest_dit_ema.safetensors",
        "latest_dino.safetensors",
        "best_set5_psnr_dit.safetensors",
    ):
        sd_a = load_file(str(dir_a / "weights" / wname))
        sd_b = load_file(str(dir_b / "weights" / wname))
        assert set(sd_b.keys()) == set(sd_a.keys())
        for k in sd_a:
            assert torch.allclose(sd_b[k], sd_a[k], atol=1e-5), (
                f"Weight mismatch in {wname} key {k!r} between baseline and resumed run"
            )

    # Non-vacuity: steps 3-4 genuinely updated DiT weights beyond step 2
    sd_b2 = load_file(str(dir_b / "weights" / "iter_000002_dit.safetensors"))
    sd_b4 = load_file(str(dir_b / "weights" / "iter_000004_dit.safetensors"))
    assert any(not torch.equal(sd_b2[k], sd_b4[k]) for k in sd_b2)

    snap_latest = load_file(str(dir_b / "weights" / "latest_dit.safetensors"))
    snap_ema = load_file(str(dir_b / "weights" / "latest_dit_ema.safetensors"))

    # 4. Run B Part 3: Idempotent re-launch at total_iters=4
    _run("s2_preempted", total_iters=4, resume_auto=True)
    log_b_after = (dir_b / "train.log").read_text(encoding="utf-8")
    assert not re.search(r"\[Iter\s+0*5/", log_b_after)

    after_latest = load_file(str(dir_b / "weights" / "latest_dit.safetensors"))
    after_ema = load_file(str(dir_b / "weights" / "latest_dit_ema.safetensors"))
    assert all(torch.equal(after_latest[k], snap_latest[k]) for k in snap_latest)
    assert all(torch.equal(after_ema[k], snap_ema[k]) for k in snap_ema)


def test_t4_scenario3_text_sr_custom_meta_ocr_and_gtless_suites(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    val_suite_gt_less_meta_factory: Callable[..., dict[str, Any]],
    minimal_train_overrides_factory: Callable[..., list[str]],
    train_cli_runner: Callable[..., Any],
) -> None:
    """Scenario 3: 3 heterogeneous val_suites (GT PSNR/SSIM, GT OCR-A, GT-less lower-is-better)."""
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from safetensors.torch import load_file

    from utils.config import load_resolved_config

    train_src = paired_train_source_factory(name="s3_train", n_samples=4, lq_h=16, lq_w=16, scale=2)
    bench_gt = val_suite_gt_label_factory(name="bench_gt", scale=2, sizes=[(16, 16), (16, 20)])
    text_ocr = val_suite_gt_label_factory(name="text_ocr", scale=2, sizes=[(16, 20), (20, 16)])
    real_nogt = val_suite_gt_less_meta_factory(name="real_nogt")
    state_file = e2e_workdir["work"] / "s3_err_state.json"

    val_suites = {
        "bench_gt": {
            "dataset": {"root_dir": bench_gt["root_dir"], "json_path": bench_gt["json_path"]},
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                },
                "ssim": {
                    "_target_": "metrics.SSIMMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                },
            },
        },
        "text_ocr": {
            "dataset": {"root_dir": text_ocr["root_dir"], "json_path": text_ocr["json_path"]},
            "metrics": {
                "ocr_a": {
                    "_target_": "tests.e2e.custom_metrics.MockOCRAMetric",
                    "label_key": "label",
                    "strict": True,
                }
            },
        },
        "real_nogt": {
            "dataset": {"root_dir": real_nogt["root_dir"], "json_path": real_nogt["json_path"]},
            "metrics": {
                "err_score": {
                    "_target_": "tests.e2e.custom_metrics.MockLowerIsBetterMetric",
                    "sequence": [0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.6, 0.6, 0.6],
                    "state_file": str(state_file),
                }
            },
        },
    }

    exp_name = "s3_hetero_val"
    work_dir = e2e_workdir["work"] / exp_name
    ovs = minimal_train_overrides_factory(
        e2e_workdir["work"],
        exp_name,
        [train_src],
        val_suites=val_suites,
        primary_dataset="real_nogt",
        primary_metric="err_score",
        scale=2,
        gt_size=32,
        batch_size=2,
        total_iters=3,
        log_interval=1,
        val_interval=1,
        save_interval=1,
        max_keep_states=3,
        has_degradation=False,
        use_ema=True,
        log_with=None,
        extra_overrides=[
            "optim.optimizer.lr=1.0e-2",
            "model.ema.decay=0.5",
            "trainer.val_save_visuals=true",
            "trainer.val_max_save_images=2",
        ],
    )
    train_cli_runner(ovs, check=True)

    # 1. Req #3 check on resolved config
    resolved = load_resolved_config(work_dir / "config_resolved.yaml")
    data_keys = _collect_dict_keys(OmegaConf.to_container(resolved.data, resolve=True))
    assert not (data_keys & {"val", "val_datasets", "val_suites", "val_sources", "validation"})
    assert set(resolved.metric.val_suites.keys()) == {"bench_gt", "text_ocr", "real_nogt"}

    # 2. All 3 suites and 4 metrics reported in train.log without NaN
    log_text = (work_dir / "train.log").read_text(encoding="utf-8")
    log_lower = log_text.lower()
    for token in ("bench_gt", "text_ocr", "real_nogt", "psnr", "ssim", "ocr_a", "err_score"):
        assert token in log_lower, f"Missing {token!r} in train.log:\n{log_text}"
    for line in log_text.splitlines():
        if any(s in line.lower() for s in ("bench_gt", "text_ocr", "real_nogt")):
            assert not re.search(r"\bnan\b", line.lower()), f"Unexpected NaN in val line: {line!r}"

    # 3. Visual panel widths: 3-panel for GT suites, 2-panel for GT-less suite
    for suite_name, info, panels in (
        ("bench_gt", bench_gt, 3),
        ("text_ocr", text_ocr, 3),
        ("real_nogt", real_nogt, 2),
    ):
        pngs = sorted((work_dir / "visuals").rglob(f"{suite_name}/*.png"))
        assert len(pngs) >= 1, f"Expected visuals for suite {suite_name!r}"
        expected_widths = {panels * (lq_w * 2) for _, lq_w in info["lq_sizes"]}
        for png in pngs:
            with Image.open(png) as im:
                assert im.size[0] in expected_widths, (
                    f"Suite {suite_name!r} expected {panels}-panel width in {expected_widths}, got {im.size}"
                )

    # 4. higher_is_better=False best-selection proof: step 2 (0.2) beats step 1 (0.8) & step 3 (0.6)
    best_path = work_dir / "weights" / "best_real_nogt_err_score_dit.safetensors"
    assert best_path.is_file()
    sd_best = load_file(str(best_path))
    sd_iter2 = load_file(str(work_dir / "weights" / "iter_000002_dit_ema.safetensors"))
    sd_iter3 = load_file(str(work_dir / "weights" / "iter_000003_dit_ema.safetensors"))

    assert any(not torch.allclose(sd_iter2[k], sd_iter3[k], atol=1e-6) for k in sd_iter2)
    assert all(torch.allclose(sd_best[k], sd_iter2[k], atol=1e-6) for k in sd_best)
    assert any(not torch.allclose(sd_best[k], sd_iter3[k], atol=1e-6) for k in sd_best)


def test_t4_scenario4_vosr_nested_checkpoint_and_jpeg_output(
    tmp_path: Path, synthetic_image_factory: Callable[..., Path], infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    import torch
    from PIL import Image
    from tests.dummy_arch import DummyVAE, get_dummy_vosr_arch_dict

    # A nested BasicSR checkpoint gives two distinguishable decoded pixel values.
    vae = DummyVAE()
    weights = {key: torch.zeros_like(t) for key, t in vae.state_dict().items()}
    ema = {**weights, "dec_conv.bias": torch.full_like(weights["dec_conv.bias"], 0.5)}
    path = tmp_path / "vae.pth"
    torch.save({"params": weights, "params_ema": ema}, path)
    inputs = tmp_path / "images"
    synthetic_image_factory(inputs / "odd.png", width=23, height=17)
    out = tmp_path / "out"
    infer_cli_runner([
        "model=vosr", "scale=2",
        f"++arch={hydra_value_formatter(get_dummy_vosr_arch_dict(with_dino=False))}",
        f"model.pretrain.vae.path={path}", "model.pretrain.vae.key=params_ema",
        f"infer.input={inputs}", f"infer.output_dir={out}", "infer.save_format=jpg",
        "infer.tile.size=[16,16]", "infer.tile.overlap=4", "infer.mixed_precision=no",
    ], check=True)
    with Image.open(out / "odd.jpg") as image:
        assert image.format == "JPEG" and image.size == (46, 34)
        assert image.getextrema() == ((191, 191),) * 3


def test_t4_scenario5_distributed_2proc_train_val_and_offline_infer(
    repo_root: Path,
    e2e_workdir: dict[str, Path],
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    minimal_train_overrides_factory: Callable[..., list[str]],
    torchrun_cpu_runner: Callable[..., Any],
    infer_cli_runner: Callable[..., Any],
) -> None:
    """Scenario 5: 2-proc CPU torchrun training + odd 5-image val + single-device offline inference."""
    from utils.image import load_image

    # Step 1: 2-process CPU distributed training with mixed sources & odd 5-image stateless PSNR val
    gt_src = gt_only_train_source_factory(name="s5_gt", n_samples=4, gt_h=48, gt_w=48, ratio=0.5)
    pair_src = paired_train_source_factory(
        name="s5_pair", n_samples=4, lq_h=24, lq_w=24, scale=2, ratio=0.5
    )
    odd_val = val_suite_gt_label_factory(
        name="odd5", scale=2, sizes=[(16, 16), (16, 20), (20, 16), (18, 22), (22, 18)]
    )
    val_suites = {
        "odd5": {
            "dataset": {"root_dir": odd_val["root_dir"], "json_path": odd_val["json_path"]},
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                }
            },
        }
    }

    exp_name = "s5_dist_train"
    work_dir = e2e_workdir["work"] / exp_name
    train_ovs = minimal_train_overrides_factory(
        e2e_workdir["work"],
        exp_name,
        [gt_src, pair_src],
        val_suites=val_suites,
        primary_dataset="odd5",
        primary_metric="psnr",
        scale=2,
        gt_size=32,
        batch_size=2,
        total_iters=4,
        log_interval=1,
        val_interval=2,
        save_interval=2,
        max_keep_states=2,
        has_degradation=True,
        use_ema=True,
        log_with=None,
        extra_overrides=[
            "optim.optimizer.lr=2.0e-4",
            "optim.scheduler.T_max=10",
            "optim.scheduler.eta_min=1.0e-6",
            "data.degradation.kernel_cfg.kernel_size=7",
            "data.degradation.kernel_cfg.kernel_range=[7]",
        ],
    )
    torchrun_cpu_runner(repo_root / "tools" / "train.py", train_ovs, nproc_per_node=2, check=True)

    assert (work_dir / "config_resolved.yaml").is_file()
    assert (work_dir / "train.log").is_file()
    assert (work_dir / "checkpoints" / "state_iter_000004").is_dir()
    for wname in (
        "iter_000004_vae.safetensors",
        "iter_000004_dit.safetensors",
        "iter_000004_dit_ema.safetensors",
        "iter_000004_dino.safetensors",
        "latest_vae.safetensors",
        "latest_dit.safetensors",
        "latest_dit_ema.safetensors",
        "latest_dino.safetensors",
        "best_odd5_psnr_vae.safetensors",
        "best_odd5_psnr_dit.safetensors",
        "best_odd5_psnr_dino.safetensors",
    ):
        assert (work_dir / "weights" / wname).is_file(), f"Missing weight artifact {wname}"

    log_text = (work_dir / "train.log").read_text(encoding="utf-8")
    iter4_lines = _extract_iter_lines(log_text, step=4, total_iters=4)
    assert len(iter4_lines) == 1, f"Expected single Rank-0 Iter 4 line:\n{log_text}"
    logged_lr = _extract_lr_from_line(iter4_lines[0])
    expected_lr_4 = 1.0e-6 + 0.5 * (2.0e-4 - 1.0e-6) * (1.0 + math.cos(math.pi * 4 / 10))
    buggy_lr_8 = 1.0e-6 + 0.5 * (2.0e-4 - 1.0e-6) * (1.0 + math.cos(math.pi * 8 / 10))
    assert logged_lr == pytest.approx(expected_lr_4, rel=2e-2)
    assert logged_lr > 2.5 * buggy_lr_8

    vis_dir = work_dir / "visuals" / "iter_000004" / "odd5"
    if vis_dir.is_dir():
        vis_files = list(vis_dir.glob("*.png"))
        assert len(vis_files) <= 5

    # Infer on the five images with the components exported by distributed training.
    infer_root = e2e_workdir["data"] / "s5_infer_root"
    infer_sizes = [(20, 24), (24, 20), (18, 26), (22, 22), (26, 18)]
    for idx, (lq_h, lq_w) in enumerate(infer_sizes):
        synthetic_image_factory(infer_root / f"m{idx}.png", width=lq_w, height=lq_h, seed=500 + idx)

    out_dir = e2e_workdir["outputs"] / "s5_infer_out"
    infer_proc = infer_cli_runner([
        *vosr_infer_overrides(
            2, work_dir / "weights", vae="best_odd5_psnr_vae.safetensors",
            dit="best_odd5_psnr_dit.safetensors", dino="best_odd5_psnr_dino.safetensors",
        ),
        f"infer.input={infer_root}", f"infer.output_dir={out_dir}",
        "infer.mixed_precision=no", "infer.tile.size=[16,16]", "infer.tile.overlap=4",
    ], check=True)

    out_pngs = list(out_dir.rglob("*.png"))
    assert sorted(p.name for p in out_pngs) == [f"m{i}.png" for i in range(5)]
    by_name = {p.name: p for p in out_pngs}
    for idx, (lq_h, lq_w) in enumerate(infer_sizes):
        t = load_image(by_name[f"m{idx}.png"])
        assert t.shape == (3, lq_h * 2, lq_w * 2)
        assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0

    combined = f"{infer_proc.stdout}\n{infer_proc.stderr}"
    assert "img/s" in combined and re.search(r"(?<!\d)5(?!\d)", combined), (
        f"Expected inference summary with '5' and 'img/s':\n{combined}"
    )


def test_t4_scenario6_two_stage_l1_to_l1_lpips_finetune_trackers_and_infer(
    repo_root: Path,
    configs_dir: Path,
    e2e_workdir: dict[str, Path],
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    minimal_train_overrides_factory: Callable[..., list[str]],
    train_cli_runner: Callable[..., Any],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """Scenario 6: Stage 1 L1 -> Stage 2 pretrain+l1_lpips+USM+Pool+TB -> edge infer -> Reqs #1-#5."""
    import torch
    import yaml
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    import archs as archs_pkg
    import losses as losses_pkg
    from utils.config import load_resolved_config
    from utils.image import load_image

    fmt = hydra_value_formatter

    # Stage 1: L1Loss pretraining at gt_size=64, scale=2
    s1_pair = paired_train_source_factory(
        name="s6_stage1_pair", n_samples=4, lq_h=32, lq_w=32, scale=2, ratio=1.0
    )
    val_info = val_suite_gt_label_factory(name="set5", scale=2, sizes=[(32, 32)])
    val_suites = {
        "set5": {
            "dataset": {"root_dir": val_info["root_dir"], "json_path": val_info["json_path"]},
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                }
            },
        }
    }

    stage1_dir = e2e_workdir["work"] / "s6_stage1_l1"
    ovs_s1 = minimal_train_overrides_factory(
        e2e_workdir["work"],
        "s6_stage1_l1",
        [s1_pair],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=2,
        gt_size=64,
        batch_size=2,
        total_iters=2,
        log_interval=1,
        val_interval=2,
        save_interval=2,
        has_degradation=False,
        use_ema=True,
        log_with=None,
    )
    train_cli_runner(ovs_s1, check=True)
    stage1_dit_ema = stage1_dir / "weights" / "latest_dit_ema.safetensors"
    assert stage1_dit_ema.is_file()
    stage1_ema_sd = load_file(str(stage1_dit_ema))

    # Stage 2: Fine-tuning from model.pretrain.dit + l1_lpips + USMSharp + TrainingPool(4) + TensorBoard
    s2_gt = gt_only_train_source_factory(name="s6_s2_gt", n_samples=4, gt_h=80, gt_w=80, ratio=0.5)
    s2_pair = paired_train_source_factory(
        name="s6_s2_pair", n_samples=4, lq_h=40, lq_w=40, scale=2, ratio=0.5
    )
    stage2_dir = e2e_workdir["work"] / "s6_stage2_finetune"
    ovs_s2 = minimal_train_overrides_factory(
        e2e_workdir["work"],
        "s6_stage2_finetune",
        [s2_gt, s2_pair],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=2,
        gt_size=64,
        batch_size=2,
        total_iters=4,
        log_interval=1,
        val_interval=2,
        save_interval=2,
        has_degradation=True,
        use_ema=True,
        log_with=["tensorboard"],
        extra_overrides=[
            "loss=l1_lpips",
            "loss.lpips.criterion._target_=tests.helpers.fake_backends.DummyLPIPSLoss",
            "data.degradation.use_usm=true",
            "data.degradation.queue_size=4",
            "data.degradation.kernel_cfg.kernel_size=7",
            "data.degradation.kernel_cfg.kernel_range=[7]",
            f"model.pretrain.dit.path={fmt(str(stage1_dit_ema))}",
            "model.pretrain.dit.strict=true",
            "model.pretrain.dit.key=null",
            "trainer.resume=null",
        ],
    )
    train_cli_runner(ovs_s2, check=True)

    # 1. Fresh global_step=0 start through step 4 with l1 + lpips + total logged cleanly
    s2_log = (stage2_dir / "train.log").read_text(encoding="utf-8")
    assert _extract_iter_lines(s2_log, 1, 4) and _extract_iter_lines(s2_log, 4, 4)
    for token in ("loss/l1", "loss/lpips", "loss/total"):
        assert token in s2_log, f"Missing {token!r} in stage2 train.log:\n{s2_log}"
    assert not re.search(r"\bnan\b", s2_log.lower())

    # 2. Pretrained weights cloned into BOTH live DiT and DiT EMA prior to step 0
    from config import load_train_config
    from models.base_model import build_model

    s2_resolved = load_resolved_config(stage2_dir / "config_resolved.yaml")
    accelerate_state_resetter()
    probe_model = build_model(load_train_config(s2_resolved))
    probe_model.setup(Accelerator(cpu=True, mixed_precision="no"))
    probe_main_sd = probe_model.get_infer_network(use_ema=False).dit.state_dict()
    probe_ema_sd = probe_model.get_infer_network(use_ema=True).dit.state_dict()
    for k, ref_tensor in stage1_ema_sd.items():
        assert torch.allclose(probe_main_sd[k].detach().cpu(), ref_tensor, atol=1e-6)
        assert torch.allclose(probe_ema_sd[k].detach().cpu(), ref_tensor, atol=1e-6)

    # 3. Non-empty TensorBoard events written
    tfevents = sorted((stage2_dir / "trackers").rglob("events.out.tfevents.*"))
    assert len(tfevents) >= 1 and all(p.stat().st_size > 0 for p in tfevents)

    # Stage 3: tools/infer.py on heterogeneous large / strip / tiny edge images
    edge_dir = e2e_workdir["data"] / "s6_edge_imgs"
    synthetic_image_factory(edge_dir / "large_48x52.png", width=52, height=48, seed=601)
    synthetic_image_factory(edge_dir / "strip_9x37.png", width=37, height=9, seed=602)
    synthetic_image_factory(edge_dir / "tiny_3x5.png", width=5, height=3, seed=603)

    out_edge = e2e_workdir["outputs"] / "s6_edge_out"
    infer_cli_runner(
        [
            *vosr_infer_overrides(
                2, stage2_dir / "weights", vae="best_set5_psnr_vae.safetensors",
                dit="best_set5_psnr_dit.safetensors", dino="best_set5_psnr_dino.safetensors",
            ),
            f"infer.input={fmt(str(edge_dir))}",
            f"infer.output_dir={fmt(str(out_edge))}",
            "infer.mixed_precision='no'",
            "infer.tile.enabled=true",
            "infer.tile.size=[32,32]",
            "infer.tile.overlap=4",
        ],
        check=True,
    )

    expected_shapes = {
        "large_48x52.png": (3, 96, 104),
        "strip_9x37.png": (3, 18, 74),
        "tiny_3x5.png": (3, 6, 10),
    }
    for fname, exp_shape in expected_shapes.items():
        fpath = out_edge / fname
        assert fpath.is_file(), f"Missing inferred edge image {fpath}"
        tensor = load_image(fpath)
        assert tensor.shape == exp_shape
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0

    # Stage 4: Full User Requirements #1-#5 End-to-End Compliance Audit
    # Req #1: Stage 2 gt_only manifest entries contain 'gt' and NO 'lq'
    s2_gt_entries = json.loads(Path(s2_gt["json_path"]).read_text(encoding="utf-8"))
    assert isinstance(s2_gt_entries, list) and len(s2_gt_entries) > 0
    for entry in s2_gt_entries:
        assert "gt" in entry and "lq" not in entry

    # Req #2 & #3: Zero 'mode' in data.sources and zero val keys under data
    s2_data_dict = OmegaConf.to_container(s2_resolved.data, resolve=True)
    assert isinstance(s2_data_dict, dict)
    for src_entry in s2_data_dict["sources"]:
        assert "mode" not in src_entry
    assert not (_collect_dict_keys(s2_data_dict) & {"val", "val_datasets", "val_suites", "val_sources"})
    assert "set5" in s2_resolved.metric.val_suites

    for data_yaml in sorted((configs_dir / "data").rglob("*.yaml")):
        parsed = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        assert not (_collect_dict_keys(parsed) & {"val", "val_datasets", "val_suites", "val_sources"})
        if isinstance(parsed, dict) and isinstance(parsed.get("sources"), list):
            assert all("mode" not in s for s in parsed["sources"] if isinstance(s, dict))

    # Req #4 & #5: Zero SRPredictor/VOSR2OneStep/SRModel/ToyModel across src/tools/configs
    forbidden_tokens = (
        "SRPredictor",
        "VOSR2OneStep",
        "models.SRModel",
        "ToyModel",
        "RRDBNet",
        "SwinIR",
        "TinySRNet",
        "CharbonnierLoss",
        "FFTLoss",
    )
    for folder_name in ("src", "tools", "configs"):
        folder = repo_root / folder_name
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() == ".py":
                src_text = path.read_text(encoding="utf-8", errors="replace")
                assert not _py_ast_contains_symbol(src_text, "SRPredictor"), (
                    f"Forbidden SRPredictor symbol found in {path}"
                )
            elif path.suffix.lower() in {".yaml", ".yml"}:
                parsed_cfg = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
                for token_str in _iter_yaml_strings(parsed_cfg):
                    assert "srpredictor" not in token_str.lower(), (
                        f"Forbidden SRPredictor found in {path}"
                    )
                    for forbidden in forbidden_tokens:
                        assert forbidden not in token_str, (
                            f"Forbidden token {forbidden!r} found in {path}: {token_str!r}"
                        )

    assert set(archs_pkg.__all__) == {"AutoencoderKLQwenImage2D", "VOSR2DiT", "DINOv2Encoder"}
    assert set(losses_pkg.__all__) == {"L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"}
