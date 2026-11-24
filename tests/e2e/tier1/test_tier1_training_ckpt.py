"""Tier 1 E2E tests — Area E: Training Engine, EMA, Dual-Track Checkpointing & CLI.

Covers features F02, F41-F48 within the CPU-only 2-vCPU VM resource budget.
All non-stdlib imports (besides ``pytest``) are deferred inside test functions.
"""

from __future__ import annotations

import abc
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def _get_ema_module(ema: Any) -> Any:
    """Extract the shadow nn.Module from ModelEMA without coupling to an uncontracted name."""
    import torch.nn as nn

    for attr in ("module", "ema_model", "ema", "model", "shadow"):
        candidate = getattr(ema, attr, None)
        if isinstance(candidate, nn.Module):
            return candidate
    if isinstance(ema, nn.Module):
        return ema
    raise AssertionError(
        f"Could not locate shadow nn.Module on ModelEMA instance of type {type(ema).__name__}"
    )


def test_t1_base_model_abc_and_vosr_model_setup_contracts(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """F41, F43: BaseModel ABC enforcement and VOSRModel setup/export/lr contracts."""
    import hydra.utils
    import torch
    import torch.nn as nn
    from accelerate import Accelerator

    from models import BaseModel, VOSRModel

    assert issubclass(BaseModel, abc.ABC)
    src = paired_train_source_factory(name="vosr_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    cfg = minimal_train_cfg_factory(
        e2e_workdir["work"] / "vosr_setup",
        [src],
        val_suites={},
        has_degradation=False,
        use_ema=True,
    )

    with pytest.raises(TypeError):
        BaseModel(cfg)  # type: ignore[abstract]

    class _MinimalConcreteModel(BaseModel):
        def setup(self, accelerator: Any) -> None:
            del accelerator

        def train_step(self, batch: dict[str, Any], global_step: int) -> tuple[dict[str, float], bool]:
            del batch, global_step
            return {}, True

        def get_infer_network(self, use_ema: bool = True) -> nn.Module:
            del use_ema
            return nn.Identity()

        def export_weights(self, accelerator: Any) -> dict[str, dict[str, Any]]:
            del accelerator
            return {}

    dummy = _MinimalConcreteModel(cfg)
    sentinel_batch = {"lq": torch.zeros(1, 3, 4, 4)}
    assert dummy.preprocess_batch(sentinel_batch) is sentinel_batch

    model = hydra.utils.instantiate(cfg.model, cfg=cfg, _recursive_=False)
    assert isinstance(model, VOSRModel)
    assert isinstance(model, BaseModel)

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model.setup(accelerator)

    infer_ema = model.get_infer_network(use_ema=True)
    infer_main = model.get_infer_network(use_ema=False)
    assert isinstance(infer_ema, nn.Module)
    assert isinstance(infer_main, nn.Module)
    assert infer_ema is not infer_main

    lr_dict = model.get_lr_dict()
    assert isinstance(lr_dict, dict) and len(lr_dict) >= 1
    assert all(isinstance(k, str) and isinstance(v, float) and v > 0.0 for k, v in lr_dict.items())

    exported = model.export_weights(accelerator)
    assert {"vae", "dit", "dit_ema", "dino"} == set(exported.keys())
    for tag in ("vae", "dit", "dit_ema", "dino"):
        sd = exported[tag]
        assert len(sd) > 0
        for k, tensor in sd.items():
            assert not k.startswith("module.") and not k.startswith("_orig_mod.")
            assert tensor.device.type == "cpu"
            assert tensor.is_contiguous()


def test_t1_model_ema_fp32_update_warmup_and_state_roundtrip() -> None:
    """F42: ModelEMA FP32 shadow weights, warmup acceleration, DDP unwrap, state_dict roundtrip."""
    import torch
    import torch.nn as nn
    from tests.dummy_arch import DummySRNet

    from models import ModelEMA

    torch.manual_seed(4)
    net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    init_params = {k: v.detach().clone() for k, v in net.named_parameters()}

    ema_no_warm = ModelEMA(net, decay=0.999, warmup_steps=0)
    ema_warm = ModelEMA(net, decay=0.999, warmup_steps=10)

    ema_no_warm_mod = _get_ema_module(ema_no_warm)
    ema_warm_mod = _get_ema_module(ema_warm)

    assert ema_no_warm_mod.training is False
    for p in ema_no_warm_mod.parameters():
        assert p.dtype == torch.float32
        assert p.requires_grad is False

    # Shift live weights by +1.0 and compare single-step EMA drift with/without warmup.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)

    ema_no_warm.update(net)
    ema_warm.update(net)

    drift_no_warm = sum(
        (p.detach() - init_params[name]).abs().mean().item()
        for name, p in ema_no_warm_mod.named_parameters()
    )
    drift_warm = sum(
        (p.detach() - init_params[name]).abs().mean().item()
        for name, p in ema_warm_mod.named_parameters()
    )
    assert drift_no_warm > 0.0
    assert drift_warm > drift_no_warm, (
        f"Expected warmup EMA drift ({drift_warm}) > fixed-decay drift ({drift_no_warm})"
    )

    # Unwrap compatibility (simulated DDP wrapper exposing .module).
    class _DDPLikeWrapper(nn.Module):
        def __init__(self, inner: nn.Module) -> None:
            super().__init__()
            self.module = inner

    wrapped = _DDPLikeWrapper(net)
    ema_no_warm.update(wrapped)

    # state_dict / load_state_dict roundtrip preserves values and device placement.
    saved_state = ema_no_warm.state_dict()
    snapshot = {k: v.detach().clone() for k, v in ema_no_warm_mod.state_dict().items()}
    with torch.no_grad():
        for p in ema_no_warm_mod.parameters():
            p.zero_()
    ema_no_warm.load_state_dict(saved_state)
    restored_sd = _get_ema_module(ema_no_warm).state_dict()
    for k, expected in snapshot.items():
        restored = restored_sd[k]
        assert restored.device == expected.device
        assert torch.equal(restored, expected)


def test_t1_vosr_model_named_loss_dict_raw_values_and_skip_nonpositive(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F43: Named L1/LPIPS loss dict logs raw sub-losses, weighted total, skips weight <= 0."""
    import torch
    from accelerate import Accelerator
    from tests.helpers.fake_backends import patch_lpips_offline

    from models import VOSRModel

    patch_lpips_offline(monkeypatch)
    # gt_size=64 so AlexNet LPIPS runs cleanly at 64x64.
    src = paired_train_source_factory(name="loss_src", n_samples=2, lq_h=32, lq_w=32, scale=2)
    loss_cfg = {
        "l1": {
            "weight": 1.0,
            "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
        },
        "lpips": {
            "weight": 0.5,
            "criterion": {
                "_target_": "losses.LPIPSLoss",
                "net": "alex",
            },
        },
        "l1_disabled": {
            "weight": 0.0,
            "criterion": {"_target_": "losses.L1Loss", "reduction": "mean"},
        },
    }
    cfg = minimal_train_cfg_factory(
        e2e_workdir["work"] / "named_losses",
        [src],
        val_suites={},
        scale=2,
        gt_size=64,
        has_degradation=False,
        use_ema=False,
        loss_cfg=loss_cfg,
    )

    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model = VOSRModel(cfg)
    model.setup(accelerator)

    torch.manual_seed(12)
    batch = model.preprocess_batch(
        {
            "lq": torch.rand(2, 3, 32, 32, dtype=torch.float32),
            "gt": torch.rand(2, 3, 64, 64, dtype=torch.float32),
            "need_degradation": torch.tensor([False, False]),
        }
    )
    scalars, stepped = model.train_step(batch, global_step=0)
    assert stepped is True
    assert "loss/l1" in scalars
    assert "loss/lpips" in scalars
    assert "loss/total" in scalars
    assert "loss/l1_disabled" not in scalars, "Zero-weight loss must be skipped and not logged"
    assert scalars["loss/l1"] > 0.0 and scalars["loss/lpips"] > 0.0

    expected_total = 1.0 * scalars["loss/l1"] + 0.5 * scalars["loss/lpips"]
    assert scalars["loss/total"] == pytest.approx(expected_total, rel=1e-4)


def test_t1_vosr_model_pretrain_safetensors_and_basicsr_pth_cloned_to_ema(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """F43: Per-arch pretrain (.safetensors and .pth params_ema) loads into DiT and clones to EMA."""
    import torch
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState
    from tests.dummy_arch import DummyDiT

    from models import VOSRModel
    from utils.misc import save_safetensors

    torch.manual_seed(21)
    ref_dit = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8)
    ref_sd = {
        k: (
            (v.detach().cpu().clone() + 0.123)
            if torch.is_floating_point(v)
            else v.detach().cpu().clone()
        )
        for k, v in ref_dit.state_dict().items()
    }
    zero_sd = {k: torch.zeros_like(v) for k, v in ref_sd.items()}

    st_path = e2e_workdir["work"] / "pretrain_dit_ref.safetensors"
    save_safetensors(ref_sd, st_path)

    pth_path = e2e_workdir["work"] / "pretrain_dit_basicsr.pth"
    torch.save({"params_ema": ref_sd, "params": zero_sd}, pth_path)

    src = paired_train_source_factory(name="pretrain_src", n_samples=2, lq_h=16, lq_w=16, scale=2)

    # Case (a): .safetensors pretrain loaded into both main DiT and EMA before step 0.
    cfg_st = minimal_train_cfg_factory(
        e2e_workdir["work"] / "run_pretrain_st",
        [src],
        val_suites={},
        has_degradation=False,
        use_ema=True,
    )
    cfg_st.model.pretrain.dit.path = str(st_path)
    cfg_st.model.pretrain.dit.strict = True
    cfg_st.model.pretrain.dit.key = None

    acc1 = Accelerator(cpu=True, mixed_precision="no")
    model_st = VOSRModel(cfg_st)
    model_st.setup(acc1)
    for use_ema in (False, True):
        dit_sd = model_st.get_infer_network(use_ema=use_ema).dit.state_dict()
        for k, expected in ref_sd.items():
            assert torch.allclose(dit_sd[k].cpu(), expected, atol=1e-6), (
                f"Safetensors pretrain mismatch (use_ema={use_ema}) on key {k}"
            )

    # Reset Accelerate singleton cleanly between in-process Accelerator instances.
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()

    # Case (b): BasicSR .pth with key="params_ema".
    cfg_pth = minimal_train_cfg_factory(
        e2e_workdir["work"] / "run_pretrain_pth",
        [src],
        val_suites={},
        has_degradation=False,
        use_ema=True,
    )
    cfg_pth.model.pretrain.dit.path = str(pth_path)
    cfg_pth.model.pretrain.dit.strict = True
    cfg_pth.model.pretrain.dit.key = "params_ema"

    acc2 = Accelerator(cpu=True, mixed_precision="no")
    model_pth = VOSRModel(cfg_pth)
    model_pth.setup(acc2)
    for use_ema in (False, True):
        dit_sd = model_pth.get_infer_network(use_ema=use_ema).dit.state_dict()
        for k, expected in ref_sd.items():
            assert torch.allclose(dit_sd[k].cpu(), expected, atol=1e-6), (
                f"BasicSR .pth key='params_ema' mismatch (use_ema={use_ema}) on key {k}"
            )


def test_t1_vosr_model_grad_accum_and_lr_scheduler_step_cadence(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """F43: grad_accum=2 cadence, lr_scheduler NOT prepared, Step 7 checkpointing of LR & EMA."""
    import torch
    from accelerate import Accelerator

    from models import VOSRModel

    base_lr = 2.0e-4
    eta_min = 1.0e-6
    t_max = 4

    src = paired_train_source_factory(name="accum_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    cfg = minimal_train_cfg_factory(
        e2e_workdir["work"] / "grad_accum_lr",
        [src],
        val_suites={},
        has_degradation=False,
        use_ema=True,
        total_iters=t_max,
        grad_accum=2,
    )
    cfg.optim.optimizer.lr = base_lr
    cfg.optim.scheduler.T_max = t_max
    cfg.optim.scheduler.eta_min = eta_min

    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
    model = VOSRModel(cfg)
    model.setup(accelerator)

    # Guardrail #5: lr_scheduler must NEVER be passed to accelerator.prepare().
    assert len(getattr(accelerator, "_schedulers", [])) == 0, (
        "lr_scheduler must NOT be passed to accelerator.prepare()"
    )

    def _current_lr() -> float:
        return next(iter(model.get_lr_dict().values()))

    assert _current_lr() == pytest.approx(base_lr, rel=1e-6)

    stepped_flags: list[bool] = []
    observed_lrs: list[float] = []
    global_step = 0
    for micro in range(4):
        torch.manual_seed(micro)
        batch = model.preprocess_batch(
            {
                "lq": torch.rand(2, 3, 16, 16, dtype=torch.float32),
                "gt": torch.rand(2, 3, 32, 32, dtype=torch.float32),
                "need_degradation": torch.tensor([False, False]),
            }
        )
        _, stepped = model.train_step(batch, global_step=global_step)
        stepped_flags.append(stepped)
        observed_lrs.append(_current_lr())
        if stepped:
            global_step += 1

    assert stepped_flags == [False, True, False, True]
    assert observed_lrs[0] == pytest.approx(base_lr, rel=1e-6)
    assert observed_lrs[2] == pytest.approx(observed_lrs[1], rel=1e-6)

    expected_lr_step2 = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * 2 / t_max))
    assert observed_lrs[3] == pytest.approx(expected_lr_step2, rel=1e-4)

    # Step 6 & 7 verification: save_state emits NO scheduler.bin, while register_for_checkpointing
    # genuinely persists & restores BOTH lr_scheduler state and ModelEMA shadow weights.
    state_dir = e2e_workdir["work"] / "srmodel_step7_state"
    accelerator.save_state(str(state_dir))
    assert state_dir.is_dir()
    assert not (state_dir / "scheduler.bin").exists(), (
        "Unexpected scheduler.bin: lr_scheduler must not be passed to accelerator.prepare()"
    )

    saved_lr = _current_lr()
    ema_net = model.get_infer_network(use_ema=True)
    saved_ema_sd = {k: v.detach().cpu().clone() for k, v in ema_net.state_dict().items()}

    # Advance 1 optimizer step (2 micro-steps) to step 3 and corrupt EMA weights in memory.
    for micro in range(4, 6):
        torch.manual_seed(micro)
        batch = model.preprocess_batch(
            {
                "lq": torch.rand(2, 3, 16, 16, dtype=torch.float32),
                "gt": torch.rand(2, 3, 32, 32, dtype=torch.float32),
                "need_degradation": torch.tensor([False, False]),
            }
        )
        _, stepped = model.train_step(batch, global_step=global_step)
        if stepped:
            global_step += 1

    expected_lr_step3 = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * 3 / t_max))
    assert _current_lr() == pytest.approx(expected_lr_step3, rel=1e-4)
    assert _current_lr() != pytest.approx(saved_lr, rel=1e-3)

    with torch.no_grad():
        for p in ema_net.parameters():
            p.zero_()

    # Restore saved state at step 2 -> LR and EMA shadow weights must restore verbatim.
    accelerator.load_state(str(state_dir))
    assert _current_lr() == pytest.approx(saved_lr, rel=1e-6)
    restored_ema_sd = model.get_infer_network(use_ema=True).state_dict()
    for k, expected in saved_ema_sd.items():
        assert torch.allclose(restored_ema_sd[k].cpu(), expected, atol=1e-6), (
            f"EMA shadow weight {k!r} was not restored via register_for_checkpointing"
        )

    # Stepping 1 more optimizer step (2 micro-steps) must reach step 3 LR (not step 4 eta_min),
    # proving lr_scheduler's own step counter was checkpointed and restored.
    post_restore_stepped: list[bool] = []
    for micro in range(6, 8):
        torch.manual_seed(micro)
        batch = model.preprocess_batch(
            {
                "lq": torch.rand(2, 3, 16, 16, dtype=torch.float32),
                "gt": torch.rand(2, 3, 32, 32, dtype=torch.float32),
                "need_degradation": torch.tensor([False, False]),
            }
        )
        _, stepped = model.train_step(batch, global_step=2)
        post_restore_stepped.append(stepped)
    assert post_restore_stepped == [False, True]
    assert _current_lr() == pytest.approx(expected_lr_step3, rel=1e-4)


def test_t1_train_state_and_checkpointer_dual_track_and_resolve(
    e2e_workdir: dict[str, Path],
) -> None:
    """F02, F44, F45: Engine re-exports, TrainState roundtrip, Checkpointer Track 1/2/best/resume."""
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from tests.dummy_arch import DummySRNet

    import engine as engine_pkg
    from engine import Checkpointer, TrainState

    for sym in ("Trainer", "TrainState", "Checkpointer", "TiledInferencer"):
        assert hasattr(engine_pkg, sym), f"Missing engine.{sym} re-export"

    state = TrainState(global_step=7, best_metrics={"set5/psnr": 31.5})
    state_copy = TrainState()
    state_copy.load_state_dict(state.state_dict())
    assert state_copy.global_step == 7
    assert state_copy.best_metrics == {"set5/psnr": 31.5}

    work_dir = e2e_workdir["work"] / "ckpt_exp"
    work_dir.mkdir(parents=True, exist_ok=True)
    proj_cfg = ProjectConfiguration(
        project_dir=str(work_dir),
        automatic_checkpoint_naming=False,
    )
    accelerator = Accelerator(cpu=True, mixed_precision="no", project_config=proj_cfg)

    net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    net = accelerator.prepare(net)
    accelerator.register_for_checkpointing(state)

    ckpt = Checkpointer(work_dir, max_keep_states=3)
    state_dir = ckpt.save_state(accelerator, step=2)
    expected_state_dir = work_dir / "checkpoints" / "state_iter_000002"
    if state_dir is not None:
        assert Path(state_dir) == expected_state_dir
    assert expected_state_dir.is_dir()

    raw_sd = {k: v.detach().cpu().contiguous().clone() for k, v in net.state_dict().items()}
    ckpt.save_weights(accelerator, {"net_g": raw_sd, "net_g_ema": raw_sd}, step=2)
    weights_dir = work_dir / "weights"
    for fname in (
        "iter_000002.safetensors",
        "iter_000002_ema.safetensors",
        "latest.safetensors",
        "latest_ema.safetensors",
    ):
        assert (weights_dir / fname).is_file(), f"Missing Track 2 weight file: {fname}"

    best_path = ckpt.save_best(accelerator, raw_sd, "set5", "psnr")
    expected_best_path = weights_dir / "best_set5_psnr.safetensors"
    if best_path is not None:
        assert Path(best_path) == expected_best_path
    assert expected_best_path.is_file()

    assert ckpt.find_latest_state() == expected_state_dir
    assert ckpt.resolve_resume_path(None) is None
    assert ckpt.resolve_resume_path("auto") == expected_state_dir
    assert ckpt.resolve_resume_path(str(expected_state_dir)) == expected_state_dir


def test_t1_trainer_fit_full_dir_layout_train_log_visuals_and_best(
    e2e_workdir: dict[str, Path],
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """F46, F48: Trainer.fit() produces full dir layout, train.log, Track 1/2 ckpts, visuals, best."""
    from PIL import Image

    from engine import Trainer

    gt_src = gt_only_train_source_factory(name="mix_gt", n_samples=4, gt_h=48, gt_w=48, ratio=0.5)
    pair_src = paired_train_source_factory(
        name="mix_pair", n_samples=4, lq_h=24, lq_w=24, scale=2, ratio=0.5
    )
    val_info = val_suite_gt_label_factory(name="set5", scale=2, sizes=[(16, 16), (16, 20)])

    val_suites = {
        "set5": {
            "dataset": {
                "root_dir": val_info["root_dir"],
                "json_path": val_info["json_path"],
            },
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
        }
    }

    work_dir = e2e_workdir["work"] / "trainer_fit_smoke"
    cfg = minimal_train_cfg_factory(
        work_dir,
        [gt_src, pair_src],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=2,
        gt_size=[32, 40],
        batch_size=2,
        total_iters=4,
        log_interval=2,
        val_interval=2,
        save_interval=2,
        log_with=None,
        use_ema=True,
        has_degradation=True,
        val_save_visuals=True,
        val_max_save_images=2,
        val_use_ema=True,
    )

    Trainer(cfg).fit()

    resolved_cfg_path = work_dir / "config_resolved.yaml"
    assert resolved_cfg_path.is_file()
    assert "${" not in resolved_cfg_path.read_text(encoding="utf-8")

    train_log_path = work_dir / "train.log"
    assert train_log_path.is_file()
    log_text = train_log_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\[Iter\s+\d+/\d+\].*lr[^:]*:.*\|.*loss/total:.*\|.*it/s.*\|.*mem:.*GB.*\|.*ETA:"
    )
    assert pattern.search(log_text), f"train.log did not match expected format:\n{log_text}"

    # Track 1 & Track 2 checkpoints + best weights (saved per independent arch).
    assert (work_dir / "checkpoints" / "state_iter_000002").is_dir()
    assert (work_dir / "checkpoints" / "state_iter_000004").is_dir()
    for fname in (
        "iter_000002_vae.safetensors",
        "iter_000002_dit.safetensors",
        "iter_000002_dit_ema.safetensors",
        "iter_000002_dino.safetensors",
        "iter_000004_vae.safetensors",
        "iter_000004_dit.safetensors",
        "iter_000004_dit_ema.safetensors",
        "iter_000004_dino.safetensors",
        "latest_vae.safetensors",
        "latest_dit.safetensors",
        "latest_dit_ema.safetensors",
        "latest_dino.safetensors",
        "best_set5_psnr_vae.safetensors",
        "best_set5_psnr_dit.safetensors",
        "best_set5_psnr_dino.safetensors",
    ):
        assert (work_dir / "weights" / fname).is_file(), f"Missing {fname}"

    # Visual comparison images: 3-panel [LQ_bicubic | SR | GT] -> width == 3 * gt_w.
    visual_pngs = sorted((work_dir / "visuals").rglob("*.png"))
    assert len(visual_pngs) >= 1
    with Image.open(visual_pngs[0]) as vis_img:
        vis_w, vis_h = vis_img.size
    valid_widths = {3 * (lq_w * 2) for _, lq_w in val_info["lq_sizes"]}
    valid_heights = {lq_h * 2 for lq_h, _ in val_info["lq_sizes"]}
    assert vis_w in valid_widths and vis_h in valid_heights


def test_t1_trainer_tensorboard_tracker_flat_hparams_and_images(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """F46: TensorBoard tracker initializes with flattened hparams and writes event files."""
    from engine import Trainer

    pair_src = paired_train_source_factory(
        name="tb_pair", n_samples=2, lq_h=16, lq_w=16, scale=2, ratio=1.0
    )
    val_info = val_suite_gt_label_factory(name="set5", scale=2, sizes=[(16, 16)])
    val_suites = {
        "set5": {
            "dataset": {
                "root_dir": val_info["root_dir"],
                "json_path": val_info["json_path"],
            },
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                },
            },
        }
    }

    work_dir = e2e_workdir["work"] / "tb_tracker_exp"
    cfg = minimal_train_cfg_factory(
        work_dir,
        [pair_src],
        val_suites=val_suites,
        primary_dataset="set5",
        primary_metric="psnr",
        scale=2,
        gt_size=32,
        batch_size=2,
        total_iters=2,
        log_interval=1,
        val_interval=2,
        save_interval=2,
        log_with=["tensorboard"],
        use_ema=True,
        has_degradation=False,
        val_save_visuals=True,
        val_max_save_images=1,
    )

    Trainer(cfg).fit()

    tfevents = list((work_dir / "trackers").rglob("events.out.tfevents.*"))
    assert len(tfevents) >= 1, f"Expected TensorBoard events under {work_dir / 'trackers'}"


def test_t1_tools_train_cli_subprocess_cwd_independent(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    val_suite_gt_label_factory: Callable[..., dict[str, Any]],
    minimal_train_overrides_factory: Callable[..., list[str]],
    train_cli_runner: Callable[..., Any],
) -> None:
    """F47, F48: tools/train.py CLI runs CWD-independently with zero stray .hydra/ directory."""
    pair_src = paired_train_source_factory(
        name="cli_pair", n_samples=2, lq_h=16, lq_w=16, scale=2, ratio=1.0
    )
    val_info = val_suite_gt_label_factory(name="set5", scale=2, sizes=[(16, 16)])
    val_suites = {
        "set5": {
            "dataset": {
                "root_dir": val_info["root_dir"],
                "json_path": val_info["json_path"],
            },
            "metrics": {
                "psnr": {
                    "_target_": "metrics.PSNRMetric",
                    "crop_border": 2,
                    "test_y_channel": True,
                },
            },
        }
    }

    external_cwd = e2e_workdir["root"] / "outside_cwd"
    external_cwd.mkdir(parents=True, exist_ok=True)

    output_root = e2e_workdir["work"]
    exp_name = "cli_smoke_exp"
    overrides = minimal_train_overrides_factory(
        output_root,
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
        val_interval=2,
        save_interval=2,
        log_with=None,
        use_ema=True,
        has_degradation=False,
    )

    proc = train_cli_runner(overrides, cwd=external_cwd, check=True)
    assert proc.returncode == 0

    work_dir = output_root / exp_name
    assert not (external_cwd / ".hydra").exists(), "Hydra must not emit stray .hydra/ in CWD"
    assert not (work_dir / ".hydra").exists(), "Hydra output_subdir must be null"
    assert (work_dir / "config_resolved.yaml").is_file()
    assert (work_dir / "train.log").is_file()
    for fname in ("latest_vae.safetensors", "latest_dit.safetensors", "latest_dino.safetensors"):
        assert (work_dir / "weights" / fname).is_file()
    assert (work_dir / "checkpoints" / "state_iter_000002").is_dir()
