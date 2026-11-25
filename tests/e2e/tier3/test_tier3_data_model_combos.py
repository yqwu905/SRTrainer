"""Tier 3 E2E tests — Cross-Feature Pairwise Combinations (Seams A<->B<->C<->E).

Covers 6 cross-feature pairwise seam integrations across Hydra inheritance,
``ToyModel`` + ``L1Loss``/``LPIPSLoss(pretrained=False)``, ``MixedSRDataset`` +
``DistributedWeightedInfiniteSampler`` + ``RealESRGANDegrader``/``USMSharp``/``TrainingPool``,
``SRModel`` + ``ModelEMA`` + ``Checkpointer`` + ``TrainState``.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


def test_t3_hydra_experiment_inheritance_into_vosr_model_lpips_and_cosine_t_max(
    hydra_compose: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seam A x B x E (F03-F06, F08, F12-F15, F43): experiment -> VOSRModel + LPIPS + Cosine LR."""
    import torch
    from accelerate import Accelerator
    from tests.dummy_arch import get_dummy_vosr_arch_dict
    from tests.helpers.fake_backends import patch_lpips_offline

    from models import VOSRModel, VOSRPipeline

    patch_lpips_offline(monkeypatch)
    cfg = hydra_compose(
        "train",
        overrides=[
            "experiment=vosr2_onestep_text",
            "~arch/vae",
            "~arch/dit",
            "~arch/dino",
            f"+arch={hydra_value_formatter(get_dummy_vosr_arch_dict())}",
            "scale=2",
            "loss=l1_lpips",
            "loss.lpips.weight=0.05",
            "optim.optimizer.lr=1.0e-4",
            "data.gt_size=64",
            "trainer.total_iters=6",
            "trainer.mixed_precision='no'",
        ],
    )

    assert cfg.exp_name == "vosr2_onestep_text"
    assert cfg.arch.vae._target_ == "tests.dummy_arch.DummyVAE"
    assert cfg.arch.dit._target_ == "tests.dummy_arch.DummyDiT"
    assert cfg.arch.dino._target_ == "tests.dummy_arch.DummyDino"
    assert cfg.model.scale == 2
    assert cfg.data.scale == 2
    assert cfg.data.degradation.scale == 2
    assert cfg.optim.optimizer.lr == pytest.approx(1.0e-4)
    assert cfg.optim.scheduler._target_ == "torch.optim.lr_scheduler.CosineAnnealingLR"
    assert cfg.optim.scheduler.T_max == 6
    assert cfg.loss.l1.weight == pytest.approx(1.0)
    assert cfg.loss.lpips.weight == pytest.approx(0.05)

    from config import load_train_config
    from models.base_model import build_model

    model = build_model(load_train_config(cfg, require_data=False, require_trainer_full=False))
    assert isinstance(model, VOSRModel)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    model.setup(accelerator)

    assert isinstance(model.get_infer_network(use_ema=False), VOSRPipeline)
    assert len(getattr(accelerator, "_schedulers", [])) == 0

    lr_0 = next(iter(model.get_lr_dict().values()))
    assert lr_0 == pytest.approx(1.0e-4, rel=1e-6)

    torch.manual_seed(301)
    ksize = int(cfg.data.degradation.kernel_cfg.kernel_size)
    raw_batch = {
        "gt": torch.rand(2, 3, 64, 64, dtype=torch.float32),
        "lq": torch.rand(2, 3, 32, 32, dtype=torch.float32),
        "need_degradation": torch.tensor([False, False], dtype=torch.bool),
        "kernel1": torch.zeros(2, ksize, ksize, dtype=torch.float32),
        "kernel2": torch.zeros(2, ksize, ksize, dtype=torch.float32),
        "sinc_kernel": torch.zeros(2, ksize, ksize, dtype=torch.float32),
    }

    batch1 = model.preprocess_batch({k: v.clone() for k, v in raw_batch.items()})
    scalars1, stepped1 = model.train_step(batch1, global_step=0)
    assert stepped1 is True
    assert {"loss/l1", "loss/lpips", "loss/total"}.issubset(scalars1.keys())
    assert scalars1["loss/l1"] > 0.0 and scalars1["loss/lpips"] > 0.0
    assert math.isfinite(scalars1["loss/total"])
    expected_total = 1.0 * scalars1["loss/l1"] + 0.05 * scalars1["loss/lpips"]
    assert scalars1["loss/total"] == pytest.approx(expected_total, rel=1e-4)

    lr_1 = next(iter(model.get_lr_dict().values()))
    batch2 = model.preprocess_batch({k: v.clone() for k, v in raw_batch.items()})
    _, stepped2 = model.train_step(batch2, global_step=1)
    assert stepped2 is True
    lr_2 = next(iter(model.get_lr_dict().values()))

    eta_min = float(cfg.optim.scheduler.eta_min)
    expected_lr_1 = eta_min + 0.5 * (1.0e-4 - eta_min) * (1.0 + math.cos(math.pi * 1 / 6))
    expected_lr_2 = eta_min + 0.5 * (1.0e-4 - eta_min) * (1.0 + math.cos(math.pi * 2 / 6))
    assert lr_1 == pytest.approx(expected_lr_1, rel=1e-4)
    assert lr_2 == pytest.approx(expected_lr_2, rel=1e-4)


def test_t3_mixed_dataset_weighted_sampler_usm_pool_degrader_and_vosr_model(
    e2e_workdir: dict[str, Path],
    gt_only_train_source_factory: Callable[..., dict[str, Any]],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_single_json_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
    weighted_sampler_factory: Callable[..., Any],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """Seam C x E (F21-F23, F25-F29, F43): Mixed sources + Sampler + USM + Pool + VOSRModel."""
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate

    from data.degradation import RealESRGANDegrader

    # gt_size=64 ensures USMSharp default radius=50 satisfies reflect pad (50 < 64).
    gt_src = gt_only_train_source_factory(name="c_gt", n_samples=4, gt_h=64, gt_w=64, ratio=0.4)
    pair_src = paired_train_source_factory(
        name="c_pair", n_samples=4, lq_h=32, lq_w=32, scale=2, ratio=0.3
    )
    mixed_info = mixed_single_json_train_source_factory(
        name="c_mix", lq_h=32, lq_w=32, scale=2, ratio=0.3
    )
    sources = [gt_src, pair_src, mixed_info["source"]]
    assert all("mode" not in s for s in sources)

    kcfg = {"kernel_size": 7, "kernel_range": [7]}
    ds = mixed_dataset_factory(
        sources, gt_size=64, scale=2, has_degradation=True, kernel_cfg=kcfg
    )
    sampler = weighted_sampler_factory(
        [4, 4, 6], [0.4, 0.3, 0.3], batch_size_per_gpu=2, world_size=1, rank=0, base_seed=42
    )
    loader = DataLoader(ds, batch_size=2, sampler=sampler, num_workers=0)

    cfg = minimal_train_cfg_factory(
        e2e_workdir["work"] / "t3_mix_pool",
        sources,
        val_suites={},
        scale=2,
        gt_size=64,
        batch_size=2,
        has_degradation=True,
        use_ema=False,
    )
    cfg.data.degradation.use_usm = True
    cfg.data.degradation.queue_size = 4
    cfg.data.degradation.kernel_cfg.kernel_size = 7
    cfg.data.degradation.kernel_cfg.kernel_range = [7]
    cfg.optim.optimizer.lr = 1.0e-3

    from config import load_train_config
    from models.base_model import build_model

    model = build_model(load_train_config(cfg))
    model.setup(Accelerator(cpu=True, mixed_precision="no"))
    init_sd = {
        k: v.detach().cpu().clone()
        for k, v in model.get_infer_network(use_ema=False).state_dict().items()
    }

    # 1. Deterministic pre-pool masked invariance check (1 gt_only + 1 paired)
    probe_batch = default_collate([ds[(0, 0)], ds[(1, 0)]])
    mask = probe_batch["need_degradation"].bool()
    assert mask.tolist() == [True, False]
    orig_lq = probe_batch["lq"].clone()
    orig_gt = probe_batch["gt"].clone()

    out_probe = RealESRGANDegrader(scale=2, use_usm=True, queue_size=0)(probe_batch)
    assert torch.equal(out_probe["lq"][~mask], orig_lq[~mask])
    assert torch.equal(out_probe["gt"][~mask], orig_gt[~mask])
    assert torch.isfinite(out_probe["lq"][mask]).all()
    assert float(out_probe["lq"][mask].min()) >= 0.0 and float(out_probe["lq"][mask].max()) <= 1.0
    assert out_probe["lq"][mask].abs().sum().item() > 0.0
    assert torch.isfinite(out_probe["gt"][mask]).all()
    assert float(out_probe["gt"][mask].min()) >= 0.0 and float(out_probe["gt"][mask].max()) <= 1.0

    # 2. End-to-end 4 micro-steps through DataLoader -> preprocess_batch (USM+Pool(4)) -> train_step
    loader_iter = iter(loader)
    for step in range(4):
        raw = next(loader_iter)
        assert raw["gt"].shape == (2, 3, 64, 64)
        assert raw["lq"].shape == (2, 3, 32, 32)
        proc = model.preprocess_batch(raw)
        assert proc["lq"].shape == (2, 3, 32, 32)
        assert proc["gt"].shape == (2, 3, 64, 64)
        assert proc["lq"].dtype == torch.float32 and proc["gt"].dtype == torch.float32
        assert torch.isfinite(proc["lq"]).all() and torch.isfinite(proc["gt"]).all()
        assert float(proc["lq"].min()) >= 0.0 and float(proc["lq"].max()) <= 1.0
        assert float(proc["gt"].min()) >= 0.0 and float(proc["gt"].max()) <= 1.0
        assert not proc["lq"].requires_grad and not proc["gt"].requires_grad

        scalars, stepped = model.train_step(proc, global_step=step)
        assert stepped is True
        assert math.isfinite(scalars["loss/total"]) and scalars["loss/total"] > 0.0

    after_sd = model.get_infer_network(use_ema=False).state_dict()
    for k, v in after_sd.items():
        assert torch.isfinite(v).all(), f"Non-finite parameter {k!r} after mixed pool training"
    max_delta = max(
        float((v.detach().cpu() - init_sd[k]).abs().max().item())
        for k, v in after_sd.items()
        if torch.is_floating_point(v)
    )
    assert max_delta > 0.0


def test_t3_pure_paired_degradation_null_vosr_scale4_and_multistep_lr(
    hydra_compose: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
) -> None:
    """Seam A x B x C x E (F04, F05, F07, F08, F13, F14, F21-F23, F43): degradation=null + MultiStepLR."""
    import torch
    from accelerate import Accelerator
    from tests.dummy_arch import get_dummy_vosr_arch_dict
    from torch.utils.data import DataLoader

    from models import VOSRPipeline

    pair_src = paired_train_source_factory(
        name="pure_x4", n_samples=4, lq_h=16, lq_w=16, scale=4, ratio=1.0
    )
    cfg = hydra_compose(
        "train",
        overrides=[
            "~experiment",
            "scale=4",
            "~arch/vae",
            "~arch/dit",
            f"+arch={hydra_value_formatter(get_dummy_vosr_arch_dict())}",
            "loss=l1",
            "optim=adam_multistep",
            "optim.optimizer.lr=2.0e-4",
            "optim.scheduler.milestones=[2,4]",
            "optim.scheduler.gamma=0.5",
            "data.gt_size=64",
            "data.batch_size_per_gpu=2",
            "data.num_workers=0",
            "data.pin_memory=false",
            "data.persistent_workers=false",
            "data.degradation=null",
            f"data.sources={hydra_value_formatter([pair_src])}",
            "trainer.total_iters=5",
            "trainer.mixed_precision='no'",
            "model.ema.enabled=false",
        ],
    )

    assert cfg.data.degradation is None
    assert cfg.model.scale == 4
    assert cfg.optim.optimizer._target_ == "torch.optim.Adam"
    assert cfg.optim.scheduler._target_ == "torch.optim.lr_scheduler.MultiStepLR"

    ds = mixed_dataset_factory([pair_src], gt_size=64, scale=4, has_degradation=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    from config import load_train_config
    from models.base_model import build_model

    model = build_model(load_train_config(cfg, require_trainer_full=False))
    model.setup(Accelerator(cpu=True, mixed_precision="no"))
    assert isinstance(model.get_infer_network(use_ema=False), VOSRPipeline)
    assert next(iter(model.get_lr_dict().values())) == pytest.approx(2.0e-4, rel=1e-6)

    init_sd = {
        k: v.detach().cpu().clone()
        for k, v in model.get_infer_network(use_ema=False).state_dict().items()
    }

    loader_iter = iter(loader)
    observed_lrs: list[float] = []
    for step in range(5):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        assert "kernel1" not in batch
        assert "kernel2" not in batch
        assert "sinc_kernel" not in batch
        assert batch["lq"].shape == (2, 3, 16, 16)
        assert batch["gt"].shape == (2, 3, 64, 64)
        assert batch["need_degradation"].tolist() == [False, False]

        orig_lq = batch["lq"].clone()
        orig_gt = batch["gt"].clone()
        proc = model.preprocess_batch(batch)
        assert torch.equal(proc["lq"].cpu(), orig_lq)
        assert torch.equal(proc["gt"].cpu(), orig_gt)

        scalars, stepped = model.train_step(proc, global_step=step)
        assert stepped is True
        assert math.isfinite(scalars["loss/l1"])
        assert scalars["loss/total"] == pytest.approx(scalars["loss/l1"], rel=1e-5)
        for p in model.get_infer_network(use_ema=False).parameters():
            assert torch.isfinite(p).all()

        observed_lrs.append(next(iter(model.get_lr_dict().values())))

    assert observed_lrs == pytest.approx([2.0e-4, 1.0e-4, 1.0e-4, 5.0e-5, 5.0e-5], rel=1e-5)

    after_sd = model.get_infer_network(use_ema=False).state_dict()
    max_delta = max(
        float((v.detach().cpu() - init_sd[k]).abs().max().item())
        for k, v in after_sd.items()
        if torch.is_floating_point(v)
    )
    assert max_delta > 0.0


def test_t3_pretrain_basicsr_pth_with_ema_warmup_grad_clip_and_accum(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
) -> None:
    """Seam B x E (F13, F19, F42, F43): BasicSR .pth pretrain + EMA warmup + grad_clip + grad_accum."""
    import torch
    from accelerate import Accelerator
    from tests.dummy_arch import DummyDiT

    torch.manual_seed(34)
    ref_dit = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8)
    sd_ema = {
        k: (v.detach().cpu() + 0.05 if torch.is_floating_point(v) else v.detach().cpu().clone())
        for k, v in ref_dit.state_dict().items()
    }
    sd_params = {
        k: (torch.zeros_like(v.detach().cpu()) if torch.is_floating_point(v) else v.detach().cpu().clone())
        for k, v in ref_dit.state_dict().items()
    }
    pth_path = e2e_workdir["work"] / "basicsr_dit_pretrain.pth"
    torch.save({"params_ema": sd_ema, "params": sd_params}, pth_path)

    src = paired_train_source_factory(name="t3_pth_src", n_samples=2, lq_h=16, lq_w=16, scale=2)
    cfg = minimal_train_cfg_factory(
        e2e_workdir["work"] / "t3_pth_ema",
        [src],
        val_suites={},
        scale=2,
        gt_size=32,
        batch_size=2,
        grad_accum=2,
        has_degradation=False,
        use_ema=True,
    )
    cfg.model.pretrain.dit.path = str(pth_path)
    cfg.model.pretrain.dit.strict = True
    cfg.model.pretrain.dit.key = "params_ema"
    cfg.model.ema.decay = 0.999
    cfg.model.ema.warmup_steps = 4
    cfg.trainer.grad_clip_norm = 0.5
    cfg.optim.optimizer.lr = 1.0e-2

    from config import load_train_config
    from models.base_model import build_model

    accelerator = Accelerator(cpu=True, mixed_precision="no", gradient_accumulation_steps=2)
    model = build_model(load_train_config(cfg))
    model.setup(accelerator)

    def _snapshot(use_ema: bool) -> dict[str, Any]:
        return {
            k: v.detach().cpu().clone()
            for k, v in model.get_infer_network(use_ema=use_ema).dit.state_dict().items()
        }

    def _l1_drift(sd_a: dict[str, Any], sd_b: dict[str, Any]) -> float:
        return sum(
            float((sd_a[k] - sd_b[k]).abs().sum().item())
            for k in sd_a
            if torch.is_floating_point(sd_a[k])
        )

    main_0 = _snapshot(use_ema=False)
    ema_0 = _snapshot(use_ema=True)
    for k, ref_v in sd_ema.items():
        assert torch.allclose(main_0[k], ref_v, atol=1e-6)
        assert torch.allclose(ema_0[k], ref_v, atol=1e-6)

    def _make_batch(seed: int) -> dict[str, Any]:
        torch.manual_seed(seed)
        return model.preprocess_batch(
            {
                "lq": torch.rand(2, 3, 16, 16, dtype=torch.float32),
                "gt": torch.rand(2, 3, 32, 32, dtype=torch.float32),
                "need_degradation": torch.tensor([False, False], dtype=torch.bool),
            }
        )

    # Micro-step 1 (non-sync)
    _, stepped1 = model.train_step(_make_batch(101), global_step=0)
    assert stepped1 is False
    assert all(torch.equal(_snapshot(False)[k], main_0[k]) for k in main_0)
    assert all(torch.equal(_snapshot(True)[k], ema_0[k]) for k in ema_0)

    # Micro-step 2 (sync step 1)
    scalars2, stepped2 = model.train_step(_make_batch(102), global_step=0)
    assert stepped2 is True
    if "grad_norm" in scalars2:
        assert math.isfinite(scalars2["grad_norm"]) and scalars2["grad_norm"] >= 0.0

    main_1 = _snapshot(use_ema=False)
    ema_1 = _snapshot(use_ema=True)
    main_d1 = _l1_drift(main_1, main_0)
    ema_d1 = _l1_drift(ema_1, ema_0)
    assert main_d1 > 0.0 and ema_d1 > 0.0
    assert ema_d1 <= main_d1 + 1e-6
    assert ema_d1 > 0.002 * main_d1, (
        f"Expected EMA warmup_steps=4 to move faster than 1-0.999=0.001: ema_d1={ema_d1}, main_d1={main_d1}"
    )

    # Micro-step 3 (non-sync)
    _, stepped3 = model.train_step(_make_batch(103), global_step=1)
    assert stepped3 is False
    assert all(torch.equal(_snapshot(True)[k], ema_1[k]) for k in ema_1)

    # Micro-step 4 (sync step 2)
    _, stepped4 = model.train_step(_make_batch(104), global_step=1)
    assert stepped4 is True
    main_2 = _snapshot(use_ema=False)
    ema_2 = _snapshot(use_ema=True)
    ema_step2_drift = _l1_drift(ema_2, ema_1)
    main_step2_drift = _l1_drift(main_2, ema_1)
    assert 0.0 < ema_step2_drift < main_step2_drift
    assert all(torch.isfinite(v).all() for v in main_2.values())
    assert all(torch.isfinite(v).all() for v in ema_2.values())


def test_t3_sampler_o1_resume_coupled_with_checkpointer_and_train_state(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    mixed_dataset_factory: Callable[..., Any],
    weighted_sampler_factory: Callable[..., Any],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """Seam C x E (F25, F42-F45): Sampler O(1) seek + Checkpointer + TrainState exact parity."""
    import torch
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from torch.utils.data._utils.collate import default_collate

    from engine import Checkpointer, TrainState

    s0 = paired_train_source_factory(name="res_s0", n_samples=4, lq_h=16, lq_w=16, scale=2, ratio=0.6)
    s1 = paired_train_source_factory(name="res_s1", n_samples=4, lq_h=16, lq_w=16, scale=2, ratio=0.4)
    work_dir = e2e_workdir["work"] / "t3_o1_resume"

    def _build_stack(*, init_seed: int, sampler_seed: int = 77) -> tuple[Any, ...]:
        accelerate_state_resetter()
        torch.manual_seed(init_seed)
        cfg = minimal_train_cfg_factory(
            work_dir,
            [s0, s1],
            val_suites={},
            scale=2,
            gt_size=32,
            batch_size=2,
            total_iters=4,
            grad_accum=2,
            has_degradation=False,
            use_ema=True,
        )
        cfg.data.augment = {"hflip": False, "vflip": False, "rot90": False}
        cfg.optim.scheduler.T_max = 4

        proj_cfg = ProjectConfiguration(
            project_dir=str(work_dir),
            automatic_checkpoint_naming=False,
        )
        acc = Accelerator(
            cpu=True,
            mixed_precision="no",
            gradient_accumulation_steps=2,
            project_config=proj_cfg,
        )
        from config import load_train_config
        from models.base_model import build_model

        model = build_model(load_train_config(cfg))
        model.setup(acc)
        train_state = TrainState()
        acc.register_for_checkpointing(train_state)

        ds = mixed_dataset_factory(
            [s0, s1],
            gt_size=32,
            scale=2,
            augment={"hflip": False, "vflip": False, "rot90": False},
            has_degradation=False,
        )
        sampler = weighted_sampler_factory(
            [4, 4], [0.6, 0.4], batch_size_per_gpu=2, world_size=1, rank=0, base_seed=sampler_seed
        )
        ckpt = Checkpointer(work_dir, max_keep_states=3)
        return acc, model, train_state, ckpt, ds, sampler

    # Run A: continuous 8 micro-steps (4 optimizer steps), saving state at step=2
    acc_a, model_a, state_a, ckpt_a, ds_a, sampler_a = _build_stack(init_seed=101)
    samp_iter_a = iter(sampler_a)
    cont_pairs_tail: list[tuple[int, int]] = []
    for micro in range(8):
        pairs = [next(samp_iter_a) for _ in range(2)]
        if micro >= 4:
            cont_pairs_tail.extend(pairs)
        batch = default_collate([ds_a[p] for p in pairs])
        proc = model_a.preprocess_batch(batch)
        _, stepped = model_a.train_step(proc, global_step=state_a.global_step)
        if stepped:
            state_a.global_step += 1
        if micro == 3:
            assert state_a.global_step == 2
            state_a.best_metrics["set5/psnr"] = 29.25
            ckpt_a.save_state(acc_a, step=2)

    assert state_a.global_step == 4
    cont_lr = next(iter(model_a.get_lr_dict().values()))
    cont_main_sd = {
        k: v.detach().cpu().clone()
        for k, v in model_a.get_infer_network(use_ema=False).state_dict().items()
    }
    cont_ema_sd = {
        k: v.detach().cpu().clone()
        for k, v in model_a.get_infer_network(use_ema=True).state_dict().items()
    }

    # Run B: fresh objects (different init_seed=999), load state_iter_000002, O(1) seek sampler
    acc_b, model_b, state_b, ckpt_b, ds_b, sampler_b = _build_stack(init_seed=999)
    resume_dir = ckpt_b.resolve_resume_path("auto")
    assert resume_dir == work_dir / "checkpoints" / "state_iter_000002"
    acc_b.load_state(str(resume_dir))
    assert state_b.global_step == 2
    assert state_b.best_metrics == {"set5/psnr": 29.25}

    sampler_b.set_start_micro_step(state_b.global_step * 2)
    samp_iter_b = iter(sampler_b)
    resumed_pairs_tail: list[tuple[int, int]] = []
    for _ in range(4):
        pairs = [next(samp_iter_b) for _ in range(2)]
        resumed_pairs_tail.extend(pairs)
        batch = default_collate([ds_b[p] for p in pairs])
        proc = model_b.preprocess_batch(batch)
        _, stepped = model_b.train_step(proc, global_step=state_b.global_step)
        if stepped:
            state_b.global_step += 1

    assert resumed_pairs_tail == cont_pairs_tail
    assert state_b.global_step == 4
    assert next(iter(model_b.get_lr_dict().values())) == pytest.approx(cont_lr, rel=1e-6)

    resumed_main_sd = model_b.get_infer_network(use_ema=False).state_dict()
    resumed_ema_sd = model_b.get_infer_network(use_ema=True).state_dict()
    for k in cont_main_sd:
        assert torch.allclose(resumed_main_sd[k].detach().cpu(), cont_main_sd[k], atol=1e-5)
    for k in cont_ema_sd:
        assert torch.allclose(resumed_ema_sd[k].detach().cpu(), cont_ema_sd[k], atol=1e-5)


def test_t3_vosr_scales_export_weights_safetensors_reload_parity(
    e2e_workdir: dict[str, Path],
    paired_train_source_factory: Callable[..., dict[str, Any]],
    minimal_train_cfg_factory: Callable[..., Any],
    accelerate_state_resetter: Callable[[], None],
) -> None:
    """Seam B x E (F13, F19, F42, F43, F45): VOSR x2/x4 per-arch export -> safetensors -> reload parity."""
    import torch
    from accelerate import Accelerator
    from safetensors import safe_open
    from tests.dummy_arch import DummyDino, DummyDiT, DummyVAE

    from engine import Checkpointer
    from models import VOSRPipeline
    from utils.misc import load_weights

    for scale in (2, 4):
        accelerate_state_resetter()
        torch.manual_seed(360 + scale)
        work_dir = e2e_workdir["work"] / f"t3_export_x{scale}"
        src = paired_train_source_factory(
            name=f"export_x{scale}", n_samples=2, lq_h=16, lq_w=16, scale=scale
        )
        cfg = minimal_train_cfg_factory(
            work_dir,
            [src],
            val_suites={},
            scale=scale,
            gt_size=16 * scale,
            batch_size=2,
            has_degradation=False,
            use_ema=True,
        )
        cfg.optim.optimizer.lr = 5.0e-3
        cfg.model.ema.decay = 0.9

        from config import load_train_config
        from models.base_model import build_model

        accelerator = Accelerator(cpu=True, mixed_precision="no")
        model = build_model(load_train_config(cfg))
        model.setup(accelerator)

        for step in range(2):
            batch = model.preprocess_batch(
                {
                    "lq": torch.rand(2, 3, 16, 16, dtype=torch.float32),
                    "gt": torch.rand(2, 3, 16 * scale, 16 * scale, dtype=torch.float32),
                    "need_degradation": torch.tensor([False, False], dtype=torch.bool),
                }
            )
            _, stepped = model.train_step(batch, global_step=step)
            assert stepped is True

        x = torch.rand(1, 3, 13, 17, dtype=torch.float32)
        pipe_g_live = model.get_infer_network(use_ema=False).eval()
        pipe_ema_live = model.get_infer_network(use_ema=True).eval()
        with torch.no_grad():
            y_live_g = pipe_g_live(x)
            y_live_ema = pipe_ema_live(x)
        assert (y_live_g - y_live_ema).abs().max().item() > 1e-6

        exported = model.export_weights(accelerator)
        assert {"vae", "dit", "dit_ema", "dino"} == set(exported.keys())
        for tag in ("vae", "dit", "dit_ema", "dino"):
            assert len(exported[tag]) > 0
            for key in exported[tag]:
                assert not key.startswith("module.") and not key.startswith("_orig_mod.")

        ckpt = Checkpointer(work_dir)
        ckpt.save_weights(accelerator, exported, step=2)

        weights_dir = work_dir / "weights"
        expected_files = (
            "iter_000002_vae.safetensors",
            "iter_000002_dit.safetensors",
            "iter_000002_dit_ema.safetensors",
            "iter_000002_dino.safetensors",
            "latest_vae.safetensors",
            "latest_dit.safetensors",
            "latest_dit_ema.safetensors",
            "latest_dino.safetensors",
        )
        for fname in expected_files:
            fpath = weights_dir / fname
            assert fpath.is_file(), f"Missing exported safetensors file {fpath}"
            with safe_open(str(fpath), framework="pt", device="cpu") as fh:
                saved_keys = list(fh.keys())
                assert len(saved_keys) > 0
                for sk in saved_keys:
                    assert not sk.startswith("module.") and not sk.startswith("_orig_mod.")

        reloaded_vae = DummyVAE(z_dim=4, spatial_compression_ratio=2)
        reloaded_dit = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8)
        reloaded_dit_ema = DummyDiT(in_channels=8, out_channels=4, patch_size=2, hidden_ch=8, z_dims=8)
        reloaded_dino = DummyDino(embed_dim=8, image_size=16, patch_size=4)
        for net_zero in (reloaded_vae, reloaded_dit, reloaded_dit_ema, reloaded_dino):
            with torch.no_grad():
                for param in net_zero.parameters():
                    param.zero_()

        assert load_weights(reloaded_vae, weights_dir / "latest_vae.safetensors", strict=True) == ([], [])
        assert load_weights(reloaded_dit, weights_dir / "latest_dit.safetensors", strict=True) == ([], [])
        assert load_weights(reloaded_dit_ema, weights_dir / "latest_dit_ema.safetensors", strict=True) == ([], [])
        assert load_weights(reloaded_dino, weights_dir / "latest_dino.safetensors", strict=True) == ([], [])

        reloaded_pipe_g = VOSRPipeline(
            vae=reloaded_vae, dit=reloaded_dit, dino=reloaded_dino, scale=scale, eval_seed=42
        ).eval()
        reloaded_pipe_ema = VOSRPipeline(
            vae=reloaded_vae, dit=reloaded_dit_ema, dino=reloaded_dino, scale=scale, eval_seed=42
        ).eval()
        with torch.no_grad():
            y_reload_g = reloaded_pipe_g(x)
            y_reload_ema = reloaded_pipe_ema(x)

        assert y_reload_g.shape == (1, 3, 13 * scale, 17 * scale)
        assert torch.allclose(y_reload_g, y_live_g, atol=1e-6)
        assert torch.allclose(y_reload_ema, y_live_ema, atol=1e-6)
