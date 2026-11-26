"""Tier 3 E2E tests — Cross-Feature Combinations (Seams D<->E<->F + Distributed).

Covers 6 cross-feature pairwise seam integrations across Trainer validation,
multi-suite GT/GT-less metrics & visuals, 2-process CPU distributed exact
validation gather parity, distributed timestamp broadcast + LR exactness,
explicit checkpoint inference on odd images, and TensorBoard resume/pruning.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

import json
import math
import re
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.e2e.conftest import vosr_infer_overrides


def _extract_iter_lines(log_text: str, step: int, total_iters: int) -> list[str]:
    pat = re.compile(rf"\[Iter\s+0*{step}/0*{total_iters}\]")
    return [line for line in log_text.splitlines() if pat.search(line)]


def _extract_lr_from_line(line: str) -> float:
    m = re.search(r"lr[^:]*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", line)
    assert m is not None, f"Could not parse lr from log line: {line!r}"
    return float(m.group(1))


def test_t3_trainer_validation_and_tools_infer_tiled_output_parity(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    synthetic_image_factory: Callable[..., Path],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """Seam E x F (F36, F37, F45, F46, F48): Trainer checkpoint -> tools/infer.py tiled parity."""
    import hydra.utils
    import torch
    from safetensors.torch import load_file

    from engine import Trainer
    from engine.inferencer import TiledInferencer
    from utils.config import load_resolved_config
    from utils.image import load_image
    from utils.misc import load_weights

    src = paired_train_source_factory(name="t3_parity_src", n_samples=4, lq_h=16, lq_w=16, scale=2)
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

    work_dir = e2e_workdir["work"] / "t3_tiled_parity"
    cfg = minimal_train_cfg_factory(
        work_dir,
        [src],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=2,
        gt_size=32,
        total_iters=2,
        val_interval=2,
        save_interval=2,
        log_interval=1,
        use_ema=True,
        val_use_ema=True,
        has_degradation=False,
        log_with=None,
    )
    cfg.trainer.val_tile.enabled = True
    cfg.trainer.val_tile.size = None
    cfg.trainer.val_tile.overlap = 4
    cfg.trainer.val_tile.blend_mode = "cosine"
    Trainer(cfg).fit()

    best_vae = work_dir / "weights" / "best_set5_psnr_vae.safetensors"
    best_dit = work_dir / "weights" / "best_set5_psnr_dit.safetensors"
    best_dino = work_dir / "weights" / "best_set5_psnr_dino.safetensors"
    latest_dit_ema = work_dir / "weights" / "latest_dit_ema.safetensors"
    assert best_vae.is_file() and best_dit.is_file() and best_dino.is_file() and latest_dit_ema.is_file()
    sd_best = load_file(str(best_dit))
    sd_ema = load_file(str(latest_dit_ema))
    assert set(sd_best.keys()) == set(sd_ema.keys())
    assert all(torch.equal(sd_best[k], sd_ema[k]) for k in sd_best)

    probe_img = synthetic_image_factory(
        e2e_workdir["data"] / "probe_26x30.png",
        width=30,
        height=26,
        pattern="gradient",
        seed=37,
    )
    out_dir = e2e_workdir["outputs"] / "t3_tiled_parity_out"
    fmt = hydra_value_formatter
    infer_cli_runner(
        [
            *vosr_infer_overrides(
                2, work_dir / "weights", vae=best_vae.name, dit=best_dit.name, dino=best_dino.name,
            ),
            f"infer.input={fmt(str(probe_img.parent))}",
            f"infer.output_dir={fmt(str(out_dir))}",
            "infer.mixed_precision='no'",
            "infer.tile.enabled=true",
            "infer.tile.size=[16,16]",
            "infer.tile.overlap=4",
            "infer.tile.blend_mode=cosine",
        ],
        check=True,
    )

    from models import VOSRPipeline

    resolved = load_resolved_config(work_dir / "config_resolved.yaml")
    ref_vae = hydra.utils.instantiate(resolved.arch.vae)
    ref_dit = hydra.utils.instantiate(resolved.arch.dit)
    ref_dino = hydra.utils.instantiate(resolved.arch.dino)
    load_weights(ref_vae, best_vae, strict=True)
    load_weights(ref_dit, best_dit, strict=True)
    load_weights(ref_dino, best_dino, strict=True)
    net = VOSRPipeline(vae=ref_vae, dit=ref_dit, dino=ref_dino, scale=2, eval_seed=42)
    inferencer = TiledInferencer(
        model=net,
        scale=2,
        tile_enabled=True,
        tile_size=None,
        tile_overlap=4,
        tile_batch_size=4,
        blend_mode="cosine",
        pad_multiple=8,
        gt_size=32,
        mixed_precision="no",
        device="cpu",
    )
    sr_direct = inferencer.infer_tensor(load_image(probe_img).unsqueeze(0)).squeeze(0)

    sr_cli = load_image(out_dir / probe_img.name)
    assert sr_cli.shape == sr_direct.shape == (3, 52, 60)
    assert (sr_cli - sr_direct).abs().max().item() <= (1.0 / 255.0) + 1e-4


def test_t3_multi_suite_gt_and_gtless_custom_meta_metrics_and_visuals(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    val_suite_gt_less_meta_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seam D x E (F09, F17, F24, F30-F34, F46): Multi-suite GT + GT-less meta metrics & visuals."""
    from PIL import Image
    from tests.helpers.fake_backends import patch_lpips_offline

    from engine import Trainer

    patch_lpips_offline(monkeypatch)

    train_src = paired_train_source_factory(
        name="ms_train", n_samples=4, lq_h=16, lq_w=16, scale=2
    )
    # GT sizes 64x64 and 64x72 satisfy >= 64x64 spatial floor for AlexNet LPIPSMetric.
    gt_info = val_suite_gt_label_factory(name="gt_suite", scale=2, sizes=[(32, 32), (32, 36)])
    nogt_info = val_suite_gt_less_meta_factory(name="meta_nogt_suite")

    val_suites = {
        "gt_suite": {
            "dataset": {"root_dir": gt_info["root_dir"], "json_path": gt_info["json_path"]},
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
                "lpips": {
                    "_target_": "metrics.LPIPSMetric",
                    "net": "alex",
                },
            },
        },
        "meta_nogt_suite": {
            "dataset": {"root_dir": nogt_info["root_dir"], "json_path": nogt_info["json_path"]},
            "metrics": {
                "ocr_a": {
                    "_target_": "tests.e2e.custom_metrics.MockOCRAMetric",
                    "label_key": "label",
                    "strict": True,
                }
            },
        },
    }

    work_dir = e2e_workdir["work"] / "t3_multi_suite"
    cfg = minimal_train_cfg_factory(
        work_dir,
        [train_src],
        val_suites=val_suites,
        primary_dataset="gt_suite",
        primary_metric="psnr",
        scale=2,
        gt_size=32,
        total_iters=2,
        val_interval=2,
        save_interval=2,
        log_interval=1,
        log_with=None,
        has_degradation=False,
        val_save_visuals=True,
        val_max_save_images=2,
    )
    Trainer(cfg).fit()

    for comp in ("vae", "dit", "dino"):
        assert (work_dir / "weights" / f"best_gt_suite_psnr_{comp}.safetensors").is_file()

    log_text = (work_dir / "train.log").read_text(encoding="utf-8")
    log_lower = log_text.lower()
    for token in ("gt_suite", "meta_nogt_suite", "psnr", "ssim", "lpips", "ocr_a"):
        assert token in log_lower, f"Expected validation token {token!r} in train.log:\n{log_text}"
    for line in log_text.splitlines():
        if any(s in line.lower() for s in ("gt_suite", "meta_nogt_suite")):
            assert not re.search(r"\bnan\b", line.lower()), f"Found NaN on val log line: {line!r}"

    gt_pngs = sorted((work_dir / "visuals").rglob("gt_suite/*.png"))
    assert 1 <= len(gt_pngs) <= 2
    expected_gt_widths = {3 * (lq_w * 2) for _, lq_w in gt_info["lq_sizes"]}
    expected_gt_heights = {lq_h * 2 for lq_h, _ in gt_info["lq_sizes"]}
    for png in gt_pngs:
        with Image.open(png) as im:
            w, h = im.size
        assert w in expected_gt_widths and h in expected_gt_heights, (
            f"Expected 3-panel [LQ|SR|GT] dims for {png}, got {(w, h)}"
        )

    nogt_pngs = sorted((work_dir / "visuals").rglob("meta_nogt_suite/*.png"))
    assert 1 <= len(nogt_pngs) <= 2
    expected_nogt_widths = {2 * (lq_w * 2) for _, lq_w in nogt_info["lq_sizes"]}
    expected_nogt_heights = {lq_h * 2 for lq_h, _ in nogt_info["lq_sizes"]}
    for png in nogt_pngs:
        with Image.open(png) as im:
            w, h = im.size
        assert w in expected_nogt_widths and h in expected_nogt_heights, (
            f"Expected 2-panel [LQ|SR] dims for {png}, got {(w, h)}"
        )


def test_t3_2proc_distributed_validation_odd_suite_exact_gather_parity(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    aligned_paired_images_factory: Callable[..., tuple[Path, Path]],
    minimal_train_cfg_factory: Callable[..., Any],
    torchrun_cpu_runner: Callable[..., Any],
) -> None:
    """Seam D x E x Distributed (F24, F31, F46, F48): 2-proc odd val gather_for_metrics exact parity."""
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from tests.dummy_arch import DummyDino, DummyDiT, DummyVAE

    from engine.inferencer import TiledInferencer
    from metrics import PSNRMetric
    from models import VOSRPipeline
    from utils.image import load_image
    from utils.misc import load_weights, save_safetensors

    torch.manual_seed(909)
    frozen_vae = DummyVAE(z_dim=4, spatial_compression_ratio=2)
    frozen_dit = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8)
    frozen_dino = DummyDino(embed_dim=8, image_size=16, patch_size=4)
    frozen_vae_st = save_safetensors(
        frozen_vae.state_dict(), e2e_workdir["work"] / "frozen_vae.safetensors"
    )
    frozen_dit_st = save_safetensors(
        frozen_dit.state_dict(), e2e_workdir["work"] / "frozen_dit.safetensors"
    )
    frozen_dino_st = save_safetensors(
        frozen_dino.state_dict(), e2e_workdir["work"] / "frozen_dino.safetensors"
    )
    frozen_dit_sd = load_file(str(frozen_dit_st))

    # Build non-vacuous odd 5-image validation suite (5 % 2 == 1) where sample 0 and sample 4
    # have distinctly lower PSNR so any duplicated padding sample shifts the mean by > 0.3 dB.
    val_root = e2e_workdir["data"] / "val_odd5"
    entries: list[dict[str, str]] = []

    synthetic_image_factory(
        val_root / "lr/0000.png", width=16, height=16, pattern="solid", color=(255, 255, 255)
    )
    synthetic_image_factory(
        val_root / "hr/0000.png", width=32, height=32, pattern="solid", color=(0, 0, 0)
    )
    entries.append({"lq": "lr/0000.png", "gt": "hr/0000.png"})

    for idx, (lq_h, lq_w) in enumerate(((16, 20), (20, 16), (18, 22)), start=1):
        gt_rel = f"hr/{idx:04d}.png"
        lq_rel = f"lr/{idx:04d}.png"
        aligned_paired_images_factory(
            val_root / gt_rel, val_root / lq_rel, lq_h=lq_h, lq_w=lq_w, scale=2, seed=910 + idx
        )
        entries.append({"lq": lq_rel, "gt": gt_rel})

    synthetic_image_factory(
        val_root / "lr/0004.png", width=18, height=22, pattern="solid", color=(128, 128, 128)
    )
    synthetic_image_factory(
        val_root / "hr/0004.png", width=36, height=44, pattern="solid", color=(240, 240, 240)
    )
    entries.append({"lq": "lr/0004.png", "gt": "hr/0004.png"})

    val_json = val_root / "manifest.json"
    val_json.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    # Compute exact analytical oracle in-process with identical TiledInferencer & PSNRMetric.
    oracle_vae = DummyVAE(z_dim=4, spatial_compression_ratio=2)
    oracle_dit = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8)
    oracle_dino = DummyDino(embed_dim=8, image_size=16, patch_size=4)
    load_weights(oracle_vae, frozen_vae_st, strict=True)
    load_weights(oracle_dit, frozen_dit_st, strict=True)
    load_weights(oracle_dino, frozen_dino_st, strict=True)
    oracle_net = VOSRPipeline(
        vae=oracle_vae, dit=oracle_dit, dino=oracle_dino, scale=2, eval_seed=42
    )
    oracle_inf = TiledInferencer(
        model=oracle_net,
        scale=2,
        tile_enabled=True,
        tile_size=None,
        tile_overlap=4,
        tile_batch_size=4,
        blend_mode="cosine",
        pad_multiple=8,
        gt_size=32,
        mixed_precision="no",
        device="cpu",
    )
    psnr_metric = PSNRMetric(crop_border=2, test_y_channel=True)
    per_img_psnr: list[float] = []
    for entry in entries:
        lq_t = load_image(val_root / entry["lq"]).unsqueeze(0)
        gt_t = load_image(val_root / entry["gt"]).unsqueeze(0)
        sr_t = oracle_inf.infer_tensor(lq_t)
        per_img_psnr.append(psnr_metric.compute(sr_t, gt_t))

    oracle_5 = sum(per_img_psnr) / 5.0
    oracle_dup0 = (sum(per_img_psnr) + per_img_psnr[0]) / 6.0
    oracle_dup4 = (sum(per_img_psnr) + per_img_psnr[4]) / 6.0
    assert abs(oracle_5 - oracle_dup0) > 0.3, (
        f"Oracle not sufficiently distinct from dup0: {oracle_5} vs {oracle_dup0}"
    )
    assert abs(oracle_5 - oracle_dup4) > 0.1, (
        f"Oracle not sufficiently distinct from dup4: {oracle_5} vs {oracle_dup4}"
    )

    train_src = paired_train_source_factory(name="odd_train", n_samples=4, lq_h=16, lq_w=16, scale=2)
    val_suites = {
        "odd5": {
            "dataset": {"root_dir": str(val_root), "json_path": str(val_json)},
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                }
            },
        }
    }

    def _write_run_cfg(name: str) -> tuple[Path, Path]:
        run_work = e2e_workdir["work"] / name
        cfg = minimal_train_cfg_factory(
            run_work,
            [train_src],
            val_suites=val_suites,
            primary_dataset="odd5",
            primary_metric="psnr",
            scale=2,
            gt_size=32,
            total_iters=1,
            val_interval=1,
            save_interval=1,
            log_interval=1,
            use_ema=False,
            val_use_ema=False,
            has_degradation=False,
            val_save_visuals=True,
            val_max_save_images=10,
            log_with=None,
        )
        cfg.model.pretrain.vae.path = str(frozen_vae_st)
        cfg.model.pretrain.dit.path = str(frozen_dit_st)
        cfg.model.pretrain.dino.path = str(frozen_dino_st)
        cfg.optim.optimizer.lr = 0.0
        cfg.optim.optimizer.weight_decay = 0.0
        cfg.optim.scheduler.eta_min = 0.0
        cfg.trainer.val_tile.enabled = True
        cfg.trainer.val_tile.size = None
        cfg.trainer.val_tile.overlap = 4
        cfg.trainer.val_tile.blend_mode = "cosine"
        cfg_path = e2e_workdir["root"] / f"{name}_cfg.yaml"
        OmegaConf.save(cfg, cfg_path, resolve=True)
        return cfg_path, run_work

    cfg_1p_path, work_1p = _write_run_cfg("odd5_1proc")
    cfg_2p_path, work_2p = _write_run_cfg("odd5_2proc")

    # Minimal opaque-box driver script with safe PyTorch 2.15.0+cpu gloo teardown.
    driver_script = e2e_workdir["root"] / "run_val_parity.py"
    driver_script.write_text(
        textwrap.dedent(
            """\
            import json
            import os
            import re
            import sys
            from pathlib import Path

            import torch.distributed as dist
            from omegaconf import OmegaConf

            from engine import Trainer


            def _extract_psnr_from_state(trainer) -> float | None:
                train_state = getattr(trainer, "train_state", None)
                best_metrics = getattr(train_state, "best_metrics", None)
                if not isinstance(best_metrics, dict) or not best_metrics:
                    return None
                if "odd5" in best_metrics and isinstance(best_metrics["odd5"], dict):
                    if "psnr" in best_metrics["odd5"]:
                        return float(best_metrics["odd5"]["psnr"])
                for key, val in best_metrics.items():
                    key_l = str(key).lower()
                    if "odd5" in key_l and "psnr" in key_l:
                        return float(val)
                if len(best_metrics) == 1:
                    only_val = next(iter(best_metrics.values()))
                    if isinstance(only_val, (int, float)):
                        return float(only_val)
                return None


            def _extract_psnr_from_log(log_path: Path) -> float:
                text = log_path.read_text(encoding="utf-8")
                matches = []
                for line in text.splitlines():
                    if "odd5" in line.lower():
                        for m in re.finditer(
                            r"psnr\\s*[:=]\\s*([0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
                            line,
                            flags=re.IGNORECASE,
                        ):
                            matches.append(float(m.group(1)))
                if not matches:
                    raise RuntimeError(f"Could not parse odd5 psnr from {log_path}:\\n{text}")
                return matches[-1]


            def main() -> None:
                cfg_path = Path(sys.argv[1])
                out_json = Path(sys.argv[2])
                cfg = OmegaConf.load(cfg_path)
                work_dir = Path(str(cfg.work_dir))
                try:
                    trainer = Trainer(cfg)
                    trainer.fit()
                    if int(os.environ.get("RANK", "0")) == 0:
                        val = _extract_psnr_from_state(trainer)
                        exact = val is not None
                        if val is None:
                            val = _extract_psnr_from_log(work_dir / "train.log")
                        visuals = sorted(p.name for p in (work_dir / "visuals").rglob("*.png"))
                        out_json.write_text(
                            json.dumps({"psnr": float(val), "exact": exact, "visuals": visuals}),
                            encoding="utf-8",
                        )
                finally:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                        dist.destroy_process_group()


            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )

    res_1p_path = e2e_workdir["root"] / "res_1p.json"
    res_2p_path = e2e_workdir["root"] / "res_2p.json"
    torchrun_cpu_runner(driver_script, [str(cfg_1p_path), str(res_1p_path)], nproc_per_node=1, check=True)
    torchrun_cpu_runner(driver_script, [str(cfg_2p_path), str(res_2p_path)], nproc_per_node=2, check=True)

    # Confirm lr=0, wd=0 preserved exact frozen weights across both 1-proc and 2-proc runs.
    for run_work in (work_1p, work_2p):
        run_sd = load_file(str(run_work / "weights" / "latest_dit.safetensors"))
        assert set(run_sd.keys()) == set(frozen_dit_sd.keys())
        assert all(torch.equal(run_sd[k], frozen_dit_sd[k]) for k in frozen_dit_sd)

    res_1p = json.loads(res_1p_path.read_text(encoding="utf-8"))
    res_2p = json.loads(res_2p_path.read_text(encoding="utf-8"))

    if res_1p["exact"] and res_2p["exact"]:
        assert res_2p["psnr"] == pytest.approx(res_1p["psnr"], rel=1e-5)
        assert res_2p["psnr"] == pytest.approx(oracle_5, rel=1e-5)
    else:
        assert res_2p["psnr"] == pytest.approx(res_1p["psnr"], abs=1e-2)
        assert res_2p["psnr"] == pytest.approx(oracle_5, abs=1e-2)

    assert abs(res_2p["psnr"] - oracle_dup0) > 0.25, (
        f"2-proc PSNR {res_2p['psnr']} appears polluted by duplicated sample 0 ({oracle_dup0})"
    )
    assert len(res_1p["visuals"]) == 5
    assert 3 <= len(res_2p["visuals"]) <= 5, (
        f"Expected 3..5 unique visuals in 2-proc odd5 validation (zero padded duplicates), "
        f"got {res_2p['visuals']}"
    )
    assert len(res_2p["visuals"]) == len(set(res_2p["visuals"])), (
        f"Duplicate visual filenames in 2-proc validation: {res_2p['visuals']}"
    )
    assert set(res_2p["visuals"]).issubset(set(res_1p["visuals"])), (
        f"Unexpected 2-proc visual filenames {res_2p['visuals']} vs 1-proc {res_1p['visuals']}"
    )


def test_t3_2proc_distributed_timestamp_broadcast_and_lr_schedule_exactness(
    repo_root: Path,
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_overrides_factory: Callable[..., list[str]],
    torchrun_cpu_runner: Callable[..., Any],
) -> None:
    """Seam A x E x Distributed (F08, F10, F16, F43, F46-F48): Timestamp broadcast & 4-step LR."""
    pair_src = paired_train_source_factory(name="ts_dist_src", n_samples=4, lq_h=16, lq_w=16, scale=2)
    ts_root = e2e_workdir["work"] / "ts_dist_root"
    ts_root.mkdir(parents=True, exist_ok=True)

    overrides = minimal_train_overrides_factory(
        ts_root,
        "dist_ts_exp",
        [pair_src],
        val_suites={},
        scale=2,
        gt_size=32,
        batch_size=2,
        total_iters=4,
        log_interval=1,
        val_interval=0,
        save_interval=4,
        has_degradation=False,
        use_ema=False,
        log_with=None,
        extra_overrides=[
            "optim.optimizer.lr=2.0e-4",
            "optim.scheduler.T_max=10",
            "optim.scheduler.eta_min=1.0e-6",
        ],
    )
    overrides = [o for o in overrides if o != "++trainer.append_timestamp=false"]
    overrides.append("++trainer.append_timestamp=true")

    torchrun_cpu_runner(repo_root / "tools" / "train.py", overrides, nproc_per_node=2, check=True)

    dirs = [p for p in ts_root.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"Expected 1 broadcast timestamp dir, found {[d.name for d in dirs]}"
    ts_dir = dirs[0]
    assert re.fullmatch(r"dist_ts_exp_\d{8}_\d{6}", ts_dir.name)
    assert (ts_dir / "config_resolved.yaml").is_file()
    assert (ts_dir / "train.log").is_file()

    log_text = (ts_dir / "train.log").read_text(encoding="utf-8")
    iter4_lines = _extract_iter_lines(log_text, step=4, total_iters=4)
    assert len(iter4_lines) == 1, f"Expected exactly 1 Rank-0 Iter 4 line:\n{log_text}"

    logged_lr = _extract_lr_from_line(iter4_lines[0])
    expected_lr_4 = 1.0e-6 + 0.5 * (2.0e-4 - 1.0e-6) * (1.0 + math.cos(math.pi * 4 / 10))
    buggy_lr_8 = 1.0e-6 + 0.5 * (2.0e-4 - 1.0e-6) * (1.0 + math.cos(math.pi * 8 / 10))
    assert logged_lr == pytest.approx(expected_lr_4, rel=2e-2)
    assert logged_lr > 2.5 * buggy_lr_8


def test_t3_vosr_checkpoint_on_non_square_odd_tiled_image(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    synthetic_image_factory: Callable[..., Path],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """Seam B x F (F13, F36-F38, F40): Exported VOSR components on prime-odd 29x41 tiled image."""
    import torch

    from engine import Trainer
    from utils.image import load_image

    src = paired_train_source_factory(name="odd_tile_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    exp_dir = e2e_workdir["work"] / "toy_odd_exp"
    cfg = minimal_train_cfg_factory(
        exp_dir,
        [src],
        val_suites={},
        scale=2,
        gt_size=32,
        total_iters=2,
        save_interval=2,
        val_interval=0,
        log_interval=1,
        has_degradation=False,
        use_ema=True,
        log_with=None,
    )
    Trainer(cfg).fit()

    odd_img = synthetic_image_factory(
        e2e_workdir["data"] / "odd_29x41.png",
        width=41,
        height=29,
        pattern="gradient",
        seed=311,
    )
    out_dir = e2e_workdir["outputs"] / "toy_odd_out"
    fmt = hydra_value_formatter
    proc = infer_cli_runner(
        [
            *vosr_infer_overrides(
                2, exp_dir / "weights", vae="latest_vae.safetensors",
                dit="latest_dit_ema.safetensors", dino="latest_dino.safetensors",
            ),
            f"infer.input={fmt(str(odd_img.parent))}",
            f"infer.output_dir={fmt(str(out_dir))}",
            "infer.mixed_precision='no'",
            "infer.tile.enabled=true",
            "infer.tile.size=[16,16]",
            "infer.tile.overlap=4",
            "infer.tile.blend_mode=linear",
        ],
        check=True,
    )
    assert proc.returncode == 0

    out_png = out_dir / odd_img.name
    assert out_png.is_file()
    out = load_image(out_png)
    assert out.shape == (3, 58, 82)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_t3_tensorboard_tracker_with_resume_auto_rolling_pruning_and_best(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    minimal_train_overrides_factory: Callable[..., list[str]],
    train_cli_runner: Callable[..., Any],
) -> None:
    """Seam D x E x Trackers (F10, F16, F44-F46, F48): TensorBoard + resume=auto + rolling pruning."""
    from safetensors.torch import load_file

    pair_src = paired_train_source_factory(name="tb_src", n_samples=4, lq_h=16, lq_w=16, scale=2)
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

    exp_name = "t3_tb_resume"
    work_dir = e2e_workdir["work"] / exp_name

    # Part 1: Steps 0 -> 2
    ovs_p1 = minimal_train_overrides_factory(
        e2e_workdir["work"],
        exp_name,
        [pair_src],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=2,
        gt_size=32,
        batch_size=2,
        total_iters=2,
        log_interval=1,
        val_interval=1,
        save_interval=1,
        max_keep_states=2,
        log_with=["tensorboard"],
        use_ema=True,
        has_degradation=False,
    )
    train_cli_runner(ovs_p1, check=True)

    ckpts_p1 = {p.name for p in (work_dir / "checkpoints").iterdir() if p.is_dir()}
    assert ckpts_p1 == {"state_iter_000001", "state_iter_000002"}
    for comp in ("vae", "dit", "dino"):
        assert (work_dir / "weights" / f"best_set5_psnr_{comp}.safetensors").is_file()
    log_p1 = (work_dir / "train.log").read_text(encoding="utf-8")
    assert _extract_iter_lines(log_p1, 1, 2) and _extract_iter_lines(log_p1, 2, 2)
    tfevents_p1 = sorted((work_dir / "trackers").rglob("events.out.tfevents.*"))
    assert len(tfevents_p1) >= 1 and all(p.stat().st_size > 0 for p in tfevents_p1)

    # Part 2: Resume auto Steps 2 -> 4
    ovs_p2 = minimal_train_overrides_factory(
        e2e_workdir["work"],
        exp_name,
        [pair_src],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=2,
        gt_size=32,
        batch_size=2,
        total_iters=4,
        log_interval=1,
        val_interval=1,
        save_interval=1,
        max_keep_states=2,
        log_with=["tensorboard"],
        use_ema=True,
        has_degradation=False,
        extra_overrides=["trainer.resume=auto"],
    )
    train_cli_runner(ovs_p2, check=True)

    ckpts_p2 = {p.name for p in (work_dir / "checkpoints").iterdir() if p.is_dir()}
    assert ckpts_p2 == {"state_iter_000003", "state_iter_000004"}

    log_p2 = (work_dir / "train.log").read_text(encoding="utf-8")
    assert _extract_iter_lines(log_p2, 1, 2) and _extract_iter_lines(log_p2, 2, 2)
    assert _extract_iter_lines(log_p2, 3, 4) and _extract_iter_lines(log_p2, 4, 4)

    for wname in (
        "best_set5_psnr_vae.safetensors",
        "best_set5_psnr_dit.safetensors",
        "best_set5_psnr_dino.safetensors",
        "latest_vae.safetensors",
        "latest_dit.safetensors",
        "latest_dit_ema.safetensors",
        "latest_dino.safetensors",
        "iter_000004_vae.safetensors",
        "iter_000004_dit.safetensors",
        "iter_000004_dit_ema.safetensors",
        "iter_000004_dino.safetensors",
    ):
        wpath = work_dir / "weights" / wname
        assert wpath.is_file(), f"Missing weight file {wpath}"
        assert len(load_file(str(wpath))) > 0

    tfevents_p2 = sorted((work_dir / "trackers").rglob("events.out.tfevents.*"))
    assert len(tfevents_p2) >= 1 and all(p.stat().st_size > 0 for p in tfevents_p2)
