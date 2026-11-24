"""Tier 2 E2E tests — Area E: Training Engine, EMA, Checkpointing & Resume Boundaries.

Covers boundary, edge, and negative contracts for F41-F48 using exclusively
``ToyModel`` and ``L1Loss`` / ``LPIPSLoss(pretrained=False)``.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def test_t2_e01_vosr_model_empty_or_nonpositive_loss_fails_fast(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """F43: VOSRModel fails fast with ValueError when cfg.loss is empty or has no positive weights."""
    from accelerate import Accelerator

    from models import VOSRModel

    src = paired_train_source_factory(name="loss_fail_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    invalid_loss_cfgs: list[dict[str, Any]] = [
        {},
        {"l1": {"weight": 0.0, "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"}}},
        {"l1": {"weight": -0.5, "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"}}},
        {
            "l1": {"weight": 0.0, "criterion": {"_target_": "losses.L1Loss"}},
            "lpips": {
                "weight": -0.5,
                "criterion": {
                    "_target_": "losses.LPIPSLoss",
                    "net": "alex",
                },
            },
        },
    ]

    for idx, loss_cfg in enumerate(invalid_loss_cfgs):
        accelerate_state_resetter()
        cfg = minimal_train_cfg_factory(
            e2e_workdir["work"] / f"bad_loss_{idx}",
            [src],
            val_suites={},
            has_degradation=False,
            use_ema=False,
            loss_cfg=loss_cfg,
        )
        acc = Accelerator(cpu=True, mixed_precision="no")
        with pytest.raises(ValueError, match="."):
            model = VOSRModel(cfg)
            model.setup(acc)


def test_t2_e02_vosr_model_pretrain_edge_cases(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """F43: VOSRModel model.pretrain.dit enforces missing file/key/strict errors and partial load into EMA."""
    import torch
    from accelerate import Accelerator
    from tests.dummy_arch import DummyDiT

    from models import VOSRModel
    from utils.misc import save_safetensors

    torch.manual_seed(42)
    ref_dit = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8)
    ref_sd = {
        k: (v.detach().cpu() + 0.25 if torch.is_floating_point(v) else v.detach().cpu().clone())
        for k, v in ref_dit.state_dict().items()
    }

    pth_no_key = e2e_workdir["work"] / "pretrain_no_ema_key.pth"
    torch.save({"params": ref_sd}, pth_no_key)

    keys = list(ref_sd.keys())
    assert len(keys) >= 2
    mismatch_sd = {k: ref_sd[k] for k in keys[:-1]}
    shared_keys = list(mismatch_sd.keys())
    mismatch_sd["unexpected_dummy.weight"] = torch.ones(4, dtype=torch.float32)
    st_mismatch = save_safetensors(mismatch_sd, e2e_workdir["work"] / "pretrain_mismatch.safetensors")

    src = paired_train_source_factory(name="pt_edge_src", n_samples=2, lq_h=16, lq_w=16, scale=2)

    # (a) Missing pretrain file -> FileNotFoundError
    accelerate_state_resetter()
    cfg_a = minimal_train_cfg_factory(
        e2e_workdir["work"] / "pt_missing", [src], val_suites={}, has_degradation=False, use_ema=True
    )
    cfg_a.model.pretrain.dit.path = str(e2e_workdir["work"] / "no_such_weights.safetensors")
    cfg_a.model.pretrain.dit.strict = True
    cfg_a.model.pretrain.dit.key = None
    with pytest.raises(FileNotFoundError):
        VOSRModel(cfg_a).setup(Accelerator(cpu=True, mixed_precision="no"))

    # (b) Missing pretrain key in .pth -> KeyError / ValueError
    accelerate_state_resetter()
    cfg_b = minimal_train_cfg_factory(
        e2e_workdir["work"] / "pt_bad_key", [src], val_suites={}, has_degradation=False, use_ema=True
    )
    cfg_b.model.pretrain.dit.path = str(pth_no_key)
    cfg_b.model.pretrain.dit.strict = True
    cfg_b.model.pretrain.dit.key = "params_ema"
    with pytest.raises((KeyError, ValueError)):
        VOSRModel(cfg_b).setup(Accelerator(cpu=True, mixed_precision="no"))

    # (c) Key mismatch with strict=True -> RuntimeError
    accelerate_state_resetter()
    cfg_c = minimal_train_cfg_factory(
        e2e_workdir["work"] / "pt_strict_fail",
        [src],
        val_suites={},
        has_degradation=False,
        use_ema=True,
    )
    cfg_c.model.pretrain.dit.path = str(st_mismatch)
    cfg_c.model.pretrain.dit.strict = True
    cfg_c.model.pretrain.dit.key = None
    with pytest.raises(RuntimeError):
        VOSRModel(cfg_c).setup(Accelerator(cpu=True, mixed_precision="no"))

    # (d) Partial mismatch with strict=False succeeds & clones into DiT EMA
    accelerate_state_resetter()
    cfg_d = minimal_train_cfg_factory(
        e2e_workdir["work"] / "pt_non_strict",
        [src],
        val_suites={},
        has_degradation=False,
        use_ema=True,
    )
    cfg_d.model.pretrain.dit.path = str(st_mismatch)
    cfg_d.model.pretrain.dit.strict = False
    cfg_d.model.pretrain.dit.key = None
    model_d = VOSRModel(cfg_d)
    model_d.setup(Accelerator(cpu=True, mixed_precision="no"))

    main_sd = model_d.get_infer_network(use_ema=False).dit.state_dict()
    ema_sd = model_d.get_infer_network(use_ema=True).dit.state_dict()
    for k in shared_keys:
        assert torch.allclose(main_sd[k].detach().cpu(), ref_sd[k], atol=1e-6)
        assert torch.allclose(ema_sd[k].detach().cpu(), ref_sd[k], atol=1e-6)


def test_t2_e03_vosr_model_optimizer_step_was_skipped_gating(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """F42, F43: optimizer_step_was_skipped=True freezes lr_scheduler and EMA; False advances both."""
    import torch
    from accelerate import Accelerator

    from models import VOSRModel

    class _OverflowSimAccelerator(Accelerator):
        """Genuine Accelerator subclass simulating FP16 GradScaler overflow on CPU."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.simulate_step_skipped: bool = False

        @property
        def optimizer_step_was_skipped(self) -> bool:
            if self.simulate_step_skipped:
                return True
            return super().optimizer_step_was_skipped

    src = paired_train_source_factory(name="skip_gate_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    cfg = minimal_train_cfg_factory(
        e2e_workdir["work"] / "skip_gate",
        [src],
        val_suites={},
        scale=2,
        gt_size=32,
        total_iters=4,
        grad_accum=1,
        use_ema=True,
        has_degradation=False,
    )
    cfg.optim.optimizer.lr = 1.0e-2
    cfg.optim.scheduler.T_max = 4
    cfg.optim.scheduler.eta_min = 1.0e-6
    cfg.model.ema.decay = 0.5
    cfg.model.ema.warmup_steps = 0

    acc = _OverflowSimAccelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=1)
    model = VOSRModel(cfg)
    model.setup(acc)

    lr_init = next(iter(model.get_lr_dict().values()))
    assert lr_init == pytest.approx(1.0e-2, rel=1e-6)

    main_init = {
        k: v.detach().cpu().clone()
        for k, v in model.get_infer_network(use_ema=False).state_dict().items()
    }
    ema_init = {
        k: v.detach().cpu().clone()
        for k, v in model.get_infer_network(use_ema=True).state_dict().items()
    }

    torch.manual_seed(3)
    raw_batch = {
        "lq": torch.rand(2, 3, 16, 16, dtype=torch.float32),
        "gt": torch.rand(2, 3, 32, 32, dtype=torch.float32),
        "need_degradation": torch.tensor([False, False], dtype=torch.bool),
    }
    batch = model.preprocess_batch(raw_batch)

    # Phase 1: Skipped step -> lr_scheduler and EMA stay frozen even though main net moved.
    acc.simulate_step_skipped = True
    _, stepped1 = model.train_step(batch, global_step=0)
    assert stepped1 is True

    main_after_1 = model.get_infer_network(use_ema=False).state_dict()
    assert any(
        not torch.equal(main_after_1[k].detach().cpu(), main_init[k])
        for k in main_init
        if torch.is_floating_point(main_init[k])
    ), "Expected main parameters to move on CPU backward so EMA gating assertion is non-vacuous"

    assert next(iter(model.get_lr_dict().values())) == pytest.approx(lr_init, rel=1e-7)
    ema_after_1 = model.get_infer_network(use_ema=True).state_dict()
    for k, init_tensor in ema_init.items():
        assert torch.equal(ema_after_1[k].detach().cpu(), init_tensor), (
            f"EMA tensor {k!r} mutated despite optimizer_step_was_skipped=True"
        )

    # Phase 2: Positive control -> lr advances by exact step 1 (not step 2) and EMA updates.
    acc.simulate_step_skipped = False
    _, stepped2 = model.train_step(batch, global_step=1)
    assert stepped2 is True

    expected_lr_1 = 1.0e-6 + 0.5 * (1.0e-2 - 1.0e-6) * (1.0 + math.cos(math.pi * 1 / 4))
    assert next(iter(model.get_lr_dict().values())) == pytest.approx(expected_lr_1, rel=1e-4)

    ema_after_2 = model.get_infer_network(use_ema=True).state_dict()
    ema_diff = sum(
        float((ema_after_2[k].detach().cpu() - ema_init[k]).abs().sum().item())
        for k in ema_init
        if torch.is_floating_point(ema_init[k])
    )
    assert ema_diff > 0.0, "Expected EMA weights to update when optimizer_step_was_skipped=False"


def test_t2_e04_checkpointer_numerical_pruning_and_keep_all(
    e2e_workdir: dict[str, Path],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """F45: Checkpointer prunes numerically across 6-to-7 digit boundaries and honors max_keep=0."""
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from tests.dummy_arch import DummySRNet

    from engine import Checkpointer

    def _make_acc(work_dir: Path) -> Any:
        accelerate_state_resetter()
        proj_cfg = ProjectConfiguration(
            project_dir=str(work_dir),
            automatic_checkpoint_naming=False,
        )
        acc = Accelerator(cpu=True, mixed_precision="no", project_config=proj_cfg)
        net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=4, num_block=1, scale=2)
        acc.prepare(net)
        return acc

    # Part A1: 6-to-7 digit sequence with max_keep_states=2
    work_a1 = e2e_workdir["work"] / "prune_6_to_7"
    acc_a1 = _make_acc(work_a1)
    ckpt_a1 = Checkpointer(work_a1, max_keep_states=2)
    for step in (5000, 10000, 15000, 1000000):
        ckpt_a1.save_state(acc_a1, step=step)
    surviving_a1 = {p.name for p in (work_a1 / "checkpoints").iterdir() if p.is_dir()}
    assert surviving_a1 == {"state_iter_015000", "state_iter_1000000"}
    assert ckpt_a1.find_latest_state() == work_a1 / "checkpoints" / "state_iter_1000000"

    # Part A2: Lexicographical-order trap ('state_iter_1000000' < 'state_iter_200000')
    work_a2 = e2e_workdir["work"] / "prune_lex_trap"
    acc_a2 = _make_acc(work_a2)
    ckpt_a2 = Checkpointer(work_a2, max_keep_states=2)
    for step in (200000, 300000, 1000000):
        ckpt_a2.save_state(acc_a2, step=step)
    surviving_a2 = {p.name for p in (work_a2 / "checkpoints").iterdir() if p.is_dir()}
    assert surviving_a2 == {"state_iter_300000", "state_iter_1000000"}
    assert ckpt_a2.find_latest_state() == work_a2 / "checkpoints" / "state_iter_1000000"

    # Part B: max_keep_states=0 retains every state
    work_b = e2e_workdir["work"] / "prune_keep_all"
    acc_b = _make_acc(work_b)
    ckpt_b = Checkpointer(work_b, max_keep_states=0)
    for step in (1, 2, 3, 4):
        ckpt_b.save_state(acc_b, step=step)
    surviving_b = {p.name for p in (work_b / "checkpoints").iterdir() if p.is_dir()}
    assert surviving_b == {
        "state_iter_000001",
        "state_iter_000002",
        "state_iter_000003",
        "state_iter_000004",
    }
    assert ckpt_b.find_latest_state() == work_b / "checkpoints" / "state_iter_000004"


def test_t2_e05_checkpointer_and_trainer_resume_auto_empty_vs_missing_path(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """F45, F46: resume='auto' on empty dir starts cleanly at step 0; explicit missing path fails."""
    from engine import Checkpointer, Trainer

    work_empty = e2e_workdir["work"] / "empty_ckpt"
    ckpt = Checkpointer(work_empty, max_keep_states=2)
    assert ckpt.find_latest_state() is None
    assert ckpt.resolve_resume_path("auto") is None
    assert ckpt.resolve_resume_path(None) is None
    with pytest.raises(FileNotFoundError):
        ckpt.resolve_resume_path(str(work_empty / "checkpoints" / "state_iter_000010"))

    src = paired_train_source_factory(name="resume_empty_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    work_auto = e2e_workdir["work"] / "trainer_auto_empty"
    cfg_auto = minimal_train_cfg_factory(
        work_auto,
        [src],
        val_suites={},
        has_degradation=False,
        total_iters=2,
        save_interval=2,
        val_interval=0,
        log_interval=1,
        resume="auto",
    )
    Trainer(cfg_auto).fit()
    assert (work_auto / "checkpoints" / "state_iter_000002").is_dir()
    assert (work_auto / "weights" / "iter_000002_dit.safetensors").is_file()
    assert (work_auto / "weights" / "latest_dit.safetensors").is_file()

    accelerate_state_resetter()
    cfg_bad = minimal_train_cfg_factory(
        e2e_workdir["work"] / "trainer_bad_resume",
        [src],
        val_suites={},
        has_degradation=False,
        total_iters=2,
        save_interval=2,
        val_interval=0,
        resume="/no/such/state_iter_000010",
    )
    with pytest.raises(FileNotFoundError):
        trainer_bad = Trainer(cfg_bad)
        trainer_bad.fit()


def test_t2_e06_trainer_resume_already_completed_run_is_noop(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """F44, F45, F46: Resuming when global_step >= total_iters executes 0 steps and keeps weights."""
    import torch
    from safetensors.torch import load_file

    from engine import Trainer

    src = paired_train_source_factory(name="done_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    work_dir = e2e_workdir["work"] / "completed_run"

    cfg_init = minimal_train_cfg_factory(
        work_dir,
        [src],
        val_suites={},
        has_degradation=False,
        total_iters=4,
        save_interval=4,
        val_interval=0,
        log_interval=2,
        use_ema=True,
    )
    Trainer(cfg_init).fit()
    assert (work_dir / "checkpoints" / "state_iter_000004").is_dir()

    sd_main_4 = load_file(str(work_dir / "weights" / "latest_dit.safetensors"))
    sd_ema_4 = load_file(str(work_dir / "weights" / "latest_dit_ema.safetensors"))

    for target_iters in (4, 2):
        accelerate_state_resetter()
        cfg_resume = minimal_train_cfg_factory(
            work_dir,
            [src],
            val_suites={},
            has_degradation=False,
            total_iters=target_iters,
            save_interval=1,
            val_interval=0,
            log_interval=1,
            use_ema=True,
            resume="auto",
        )
        Trainer(cfg_resume).fit()

        ckpt_dirs = {p.name for p in (work_dir / "checkpoints").iterdir() if p.is_dir()}
        assert ckpt_dirs == {"state_iter_000004"}
        assert not (work_dir / "weights" / "iter_000005_dit.safetensors").exists()

        sd_main_now = load_file(str(work_dir / "weights" / "latest_dit.safetensors"))
        sd_ema_now = load_file(str(work_dir / "weights" / "latest_dit_ema.safetensors"))
        assert set(sd_main_now.keys()) == set(sd_main_4.keys())
        assert set(sd_ema_now.keys()) == set(sd_ema_4.keys())
        assert all(torch.equal(sd_main_now[k], sd_main_4[k]) for k in sd_main_4)
        assert all(torch.equal(sd_ema_now[k], sd_ema_4[k]) for k in sd_ema_4)


def test_t2_e07_trainer_lower_is_better_and_nan_metric_best_guard(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """F30, F34, F44-F46: Best tracking honors higher_is_better=False and guards against NaN."""
    import torch
    from safetensors.torch import load_file

    from engine import Trainer

    val_root = e2e_workdir["data"] / "val_1img"
    synthetic_image_factory(val_root / "lr/0000.png", width=16, height=16, seed=77)
    val_json = val_root / "val.json"
    val_json.write_text(json.dumps([{"lq": "lr/0000.png", "label": "ONE"}]), encoding="utf-8")

    src = paired_train_source_factory(name="best_guard_src", n_samples=2, lq_h=16, lq_w=16, scale=2)

    # Part A: higher_is_better=False with sequence [0.5, 0.2, 0.4] -> step 2 wins!
    work_low = e2e_workdir["work"] / "lower_best"
    seq_state = e2e_workdir["work"] / "lower_seq_state.json"
    val_suites_low = {
        "low_suite": {
            "dataset": {"root_dir": str(val_root), "json_path": str(val_json)},
            "metrics": {
                "err": {
                    "_target_": "tests.e2e.custom_metrics.MockLowerIsBetterMetric",
                    "sequence": [0.5, 0.2, 0.4],
                    "state_file": str(seq_state),
                }
            },
        }
    }
    cfg_low = minimal_train_cfg_factory(
        work_low,
        [src],
        val_suites=val_suites_low,
        primary_dataset="low_suite",
        primary_metric="err",
        total_iters=3,
        val_interval=1,
        save_interval=1,
        log_interval=1,
        use_ema=False,
        val_use_ema=False,
        has_degradation=False,
        val_save_visuals=False,
    )
    cfg_low.optim.optimizer.lr = 1.0e-2
    Trainer(cfg_low).fit()

    sd_1 = load_file(str(work_low / "weights" / "iter_000001_dit.safetensors"))
    sd_2 = load_file(str(work_low / "weights" / "iter_000002_dit.safetensors"))
    sd_3 = load_file(str(work_low / "weights" / "iter_000003_dit.safetensors"))
    sd_best = load_file(str(work_low / "weights" / "best_low_suite_err_dit.safetensors"))
    assert any(not torch.equal(sd_1[k], sd_2[k]) for k in sd_1)
    assert any(not torch.equal(sd_2[k], sd_3[k]) for k in sd_2)
    assert all(torch.equal(sd_best[k], sd_2[k]) for k in sd_best), (
        "Expected best_low_suite_err_dit.safetensors to hold step-2 weights (lowest error 0.2)"
    )

    # Part B1: Fresh run with MockNaNMetric never writes best_*.safetensors
    accelerate_state_resetter()
    work_nan_fresh = e2e_workdir["work"] / "nan_fresh"
    val_suites_nan = {
        "nan_suite": {
            "dataset": {"root_dir": str(val_root), "json_path": str(val_json)},
            "metrics": {"nan_m": {"_target_": "tests.e2e.custom_metrics.MockNaNMetric"}},
        }
    }
    cfg_nan_fresh = minimal_train_cfg_factory(
        work_nan_fresh,
        [src],
        val_suites=val_suites_nan,
        primary_dataset="nan_suite",
        primary_metric="nan_m",
        total_iters=1,
        val_interval=1,
        save_interval=1,
        log_interval=1,
        use_ema=False,
        val_use_ema=False,
        has_degradation=False,
        val_save_visuals=False,
    )
    Trainer(cfg_nan_fresh).fit()
    assert list((work_nan_fresh / "weights").glob("best_*.safetensors")) == []

    # Part B2: Resumed run with MockNaNMetric never overwrites existing finite best weights/state
    accelerate_state_resetter()
    work_nan_keep = e2e_workdir["work"] / "nan_keep"
    val_suites_p1 = {
        "s1": {
            "dataset": {"root_dir": str(val_root), "json_path": str(val_json)},
            "metrics": {
                "score": {
                    "_target_": "tests.e2e.custom_metrics.MockOCRAMetric",
                    "label_key": "label",
                    "strict": True,
                }
            },
        }
    }
    cfg_p1 = minimal_train_cfg_factory(
        work_nan_keep,
        [src],
        val_suites=val_suites_p1,
        primary_dataset="s1",
        primary_metric="score",
        total_iters=1,
        val_interval=1,
        save_interval=1,
        log_interval=1,
        use_ema=False,
        val_use_ema=False,
        has_degradation=False,
        val_save_visuals=False,
    )
    cfg_p1.optim.optimizer.lr = 1.0e-2
    Trainer(cfg_p1).fit()

    best_path = work_nan_keep / "weights" / "best_s1_score_dit.safetensors"
    assert best_path.is_file()
    sd_best_step1 = load_file(str(best_path))

    accelerate_state_resetter()
    val_suites_p2 = {
        "s1": {
            "dataset": {"root_dir": str(val_root), "json_path": str(val_json)},
            "metrics": {"score": {"_target_": "tests.e2e.custom_metrics.MockNaNMetric"}},
        }
    }
    cfg_p2 = minimal_train_cfg_factory(
        work_nan_keep,
        [src],
        val_suites=val_suites_p2,
        primary_dataset="s1",
        primary_metric="score",
        total_iters=2,
        val_interval=1,
        save_interval=1,
        log_interval=1,
        use_ema=False,
        val_use_ema=False,
        has_degradation=False,
        val_save_visuals=False,
        resume="auto",
    )
    cfg_p2.optim.optimizer.lr = 1.0e-2
    trainer2 = Trainer(cfg_p2)
    trainer2.fit()

    sd_step2 = load_file(str(work_nan_keep / "weights" / "iter_000002_dit.safetensors"))
    sd_best_after = load_file(str(best_path))
    assert any(not torch.equal(sd_step2[k], sd_best_step1[k]) for k in sd_step2)
    assert all(torch.equal(sd_best_after[k], sd_best_step1[k]) for k in sd_best_after)

    train_state = getattr(trainer2, "train_state", None)
    if train_state is not None and isinstance(getattr(train_state, "best_metrics", None), dict):
        for mkey, mval in train_state.best_metrics.items():
            assert math.isfinite(float(mval)), f"best_metrics[{mkey!r}] corrupted to {mval!r}"


def test_t2_e08_trainer_val_suites_config_boundaries_and_val_interval_zero(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """F09, F46: Trainer rejects bad primary metric refs, allows empty val_suites, honors val_interval=0."""
    from engine import Trainer

    src = paired_train_source_factory(name="val_cfg_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    val_info = val_suite_gt_label_factory(name="set5", scale=2, sizes=[(16, 16)])
    valid_suites = {
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

    bad_primary_cases = [
        ("missing_suite", "psnr", valid_suites),
        ("set5", "missing_metric", valid_suites),
        ("set5", "psnr", {}),
    ]
    for idx, (p_ds, p_metric, suites) in enumerate(bad_primary_cases):
        accelerate_state_resetter()
        cfg_bad = minimal_train_cfg_factory(
            e2e_workdir["work"] / f"bad_primary_{idx}",
            [src],
            val_suites=suites,
            primary_dataset=p_ds,
            primary_metric=p_metric,
            total_iters=1,
            val_interval=1,
            save_interval=1,
            has_degradation=False,
        )
        with pytest.raises(ValueError, match="."):
            t_bad = Trainer(cfg_bad)
            t_bad.fit()

    # Empty val_suites={} with primary_dataset=None, primary_metric=None
    accelerate_state_resetter()
    work_noval = e2e_workdir["work"] / "empty_val_suites"
    cfg_noval = minimal_train_cfg_factory(
        work_noval,
        [src],
        val_suites={},
        primary_dataset=None,
        primary_metric=None,
        total_iters=2,
        val_interval=1,
        save_interval=2,
        log_interval=1,
        has_degradation=False,
    )
    Trainer(cfg_noval).fit()
    assert (work_noval / "weights" / "latest_dit.safetensors").is_file()
    assert (work_noval / "checkpoints" / "state_iter_000002").is_dir()
    assert list((work_noval / "weights").glob("best_*.safetensors")) == []
    assert list((work_noval / "visuals").rglob("*.png")) == []

    # val_interval=0 disables mid-training validation without ZeroDivisionError
    accelerate_state_resetter()
    work_v0 = e2e_workdir["work"] / "val_interval_zero"
    val0_root = e2e_workdir["data"] / "val0_suite"
    synthetic_image_factory(val0_root / "lr/0000.png", width=16, height=16, seed=88)
    val0_json = val0_root / "val0.json"
    val0_json.write_text(json.dumps([{"lq": "lr/0000.png"}]), encoding="utf-8")
    val0_state = e2e_workdir["work"] / "val0_state.json"

    suite_v0 = {
        "s0": {
            "dataset": {"root_dir": str(val0_root), "json_path": str(val0_json)},
            "metrics": {
                "m0": {
                    "_target_": "tests.e2e.custom_metrics.MockLowerIsBetterMetric",
                    "sequence": [0.1, 0.2, 0.3],
                    "state_file": str(val0_state),
                }
            },
        }
    }
    cfg_v0 = minimal_train_cfg_factory(
        work_v0,
        [src],
        val_suites=suite_v0,
        primary_dataset="s0",
        primary_metric="m0",
        total_iters=3,
        val_interval=0,
        save_interval=3,
        log_interval=1,
        has_degradation=False,
        val_save_visuals=True,
    )
    Trainer(cfg_v0).fit()
    assert (work_v0 / "weights" / "iter_000003_dit.safetensors").is_file()
    assert not (work_v0 / "visuals" / "iter_000001").exists()
    assert not (work_v0 / "visuals" / "iter_000002").exists()
    if val0_state.exists():
        invocations = int(json.loads(val0_state.read_text(encoding="utf-8"))["idx"])
        assert invocations <= 1, (
            f"val_interval=0 must not run mid-training validations; saw {invocations} invocations"
        )


def test_t2_e09_trainer_mid_validation_preserves_grad_accum_cadence(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_less_meta_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """F43, F46: Mid-training validation resets gradient_state so grad_accum=2 stays [2, 2, 2]."""
    from engine import Trainer

    src = paired_train_source_factory(name="accum_val_src", n_samples=4, lq_h=16, lq_w=16, scale=2)
    val3 = val_suite_gt_less_meta_factory(name="accum_val3")
    trace_file = e2e_workdir["work"] / "grad_accum_trace.log"

    recording_l1_loss_cfg = {
        "l1": {
            "weight": 1.0,
            "criterion": {
                "_target_": "tests.e2e.custom_metrics.RecordingL1Loss",
                "trace_file": str(trace_file),
                "reduction": "mean",
                "trace_tag": "train_micro",
            },
        }
    }
    val_suites = {
        "accum_val3": {
            "dataset": {"root_dir": val3["root_dir"], "json_path": val3["json_path"]},
            "metrics": {
                "ocr_a": {
                    "_target_": "tests.e2e.custom_metrics.MockOCRAMetric",
                    "label_key": "label",
                    "strict": True,
                    "trace_file": str(trace_file),
                    "trace_tag": "val",
                }
            },
        }
    }

    work_dir = e2e_workdir["work"] / "grad_accum_val"
    cfg = minimal_train_cfg_factory(
        work_dir,
        [src],
        val_suites=val_suites,
        primary_dataset="accum_val3",
        primary_metric="ocr_a",
        total_iters=3,
        grad_accum=2,
        val_interval=1,
        save_interval=3,
        log_interval=1,
        has_degradation=False,
        use_ema=False,
        val_save_visuals=False,
        loss_cfg=recording_l1_loss_cfg,
    )
    Trainer(cfg).fit()

    assert (work_dir / "checkpoints" / "state_iter_000003").is_dir()
    events = [
        line.strip()
        for line in trace_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    step_micro_counts: list[int] = []
    val_burst_counts: list[int] = []
    idx = 0
    while idx < len(events):
        tag = events[idx]
        run_len = 0
        while idx < len(events) and events[idx] == tag:
            run_len += 1
            idx += 1
        if tag == "train_micro":
            step_micro_counts.append(run_len)
        elif tag == "val":
            val_burst_counts.append(run_len)

    assert events and events[0] == "train_micro", (
        f"Expected training micro-steps to precede validation, got leading event {events[:3]}"
    )
    assert step_micro_counts == [2, 2, 2], (
        f"Expected [2, 2, 2] micro-steps per optimizer step across mid-training validations, "
        f"got {step_micro_counts} (raw events: {events})"
    )
    assert events.count("train_micro") == 6
    assert len(val_burst_counts) == 3, (
        f"Expected 3 validation bursts (after steps 1, 2, 3), got {val_burst_counts}"
    )
    assert val_burst_counts[:2] == [3, 3], (
        f"Expected 3 validated images in each mid-training validation round, got {val_burst_counts}"
    )
    assert val_burst_counts[2] in (3, 6), (
        f"Expected 3 validated images (or 6 if final validation also runs) after final step, "
        f"got {val_burst_counts}"
    )
