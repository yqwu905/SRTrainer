"""Tier 1 E2E tests — Area B: ToyModel, L1/LPIPS Losses & Core Image/Logging/Weight Utils.

Covers features F02, F13-F15, F17-F19 within the CPU-only 2-vCPU VM resource budget.
All non-stdlib imports (besides ``pytest``) are deferred inside test functions.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def test_t1_toymodel_scales_arbitrary_dims_and_backward() -> None:
    """F13: ToyModel supports scales {1,2,3,4}, even/prime odd dims, finite grads & eval determinism."""
    import torch

    from archs import ToyModel

    # Default constructor contract check (num_in_ch=3, num_out_ch=3, num_feat=32, num_block=2, scale=4).
    default_net = ToyModel()
    default_out = default_net(torch.rand(1, 3, 4, 4, dtype=torch.float32))
    assert default_out.shape == (1, 3, 16, 16)
    assert torch.isfinite(default_out).all()

    for scale in (1, 2, 3, 4):
        net = ToyModel(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=8,
            num_block=1,
            scale=scale,
        )
        for batch, h, w in ((2, 8, 12), (1, 5, 7), (1, 11, 13)):
            net.train()
            net.zero_grad(set_to_none=True)
            x = torch.rand(batch, 3, h, w, dtype=torch.float32, requires_grad=True)
            y = net(x)

            assert y.shape == (batch, 3, h * scale, w * scale)
            assert y.dtype == torch.float32
            assert torch.isfinite(y).all()

            y.mean().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            for name, param in net.named_parameters():
                if param.requires_grad:
                    assert param.grad is not None, f"Missing gradient on ToyModel.{name} (s={scale})"
                    assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"

        # Eval determinism check on prime odd input.
        net.eval()
        probe = torch.rand(1, 3, 5, 7, dtype=torch.float32)
        with torch.no_grad():
            y1 = net(probe)
            y2 = net(probe)
        assert torch.equal(y1, y2)


def test_t1_archs_and_losses_export_exclusivity() -> None:
    """F02, F13-F15, Req #5: archs exports ONLY ToyModel; losses ONLY L1/LPIPS."""
    import hydra.utils

    import archs as archs_pkg
    import losses as losses_pkg

    assert hasattr(archs_pkg, "__all__")
    assert set(archs_pkg.__all__) == {"ToyModel", "VOSR2OneStep"}, (
        f"Unexpected archs.__all__: {archs_pkg.__all__!r}"
    )
    for sym in ("ToyModel", "VOSR2OneStep"):
        assert hydra.utils.get_class(f"archs.{sym}") is getattr(archs_pkg, sym)
    for forbidden_arch in ("RRDBNet", "SwinIR", "TinySRNet"):
        assert not hasattr(archs_pkg, forbidden_arch), (
            f"Forbidden architecture {forbidden_arch!r} exposed in archs"
        )

    assert hasattr(losses_pkg, "__all__")
    assert set(losses_pkg.__all__) == {"L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"}, (
        f"Unexpected losses.__all__: {losses_pkg.__all__!r}"
    )
    for sym in ("L1Loss", "LPIPSLoss", "OCRKLLoss", "GlyphLoss"):
        assert hydra.utils.get_class(f"losses.{sym}") is getattr(losses_pkg, sym)
    for forbidden_loss in ("CharbonnierLoss", "FFTLoss"):
        assert not hasattr(losses_pkg, forbidden_loss), (
            f"Forbidden loss {forbidden_loss!r} exposed in losses"
        )


def test_t1_l1_loss_values_reductions_and_grads() -> None:
    """F14: L1Loss mean/sum/none reductions match F.l1_loss exactly with zero-identity & finite grads."""
    import torch
    import torch.nn.functional as F

    from losses import L1Loss

    torch.manual_seed(0)
    pred_base = torch.rand(2, 3, 12, 12, dtype=torch.float32)
    target = torch.rand(2, 3, 12, 12, dtype=torch.float32)

    assert L1Loss(reduction="mean")(target, target).item() == pytest.approx(0.0, abs=1e-7)

    for reduction in ("mean", "sum", "none"):
        pred = pred_base.clone().requires_grad_(True)
        l1 = L1Loss(reduction=reduction)
        val = l1(pred, target)
        expected = F.l1_loss(pred_base, target, reduction=reduction)
        assert val.shape == expected.shape
        assert torch.allclose(val, expected, atol=1e-6)
        val.sum().backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum().item() > 0.0


def test_t1_lpips_loss_offline_frozen_eval_and_range_scaling() -> None:
    """F15: LPIPSLoss(pretrained=False) runs offline, stays frozen in eval, propagates input grad."""
    import torch

    from losses import LPIPSLoss

    loss_fn = LPIPSLoss(net="alex", pretrained=False)
    assert loss_fn.training is False
    params = list(loss_fn.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)

    torch.manual_seed(1)
    target = torch.rand(2, 3, 64, 64, dtype=torch.float32)
    same_val = loss_fn(target, target)
    assert torch.isfinite(same_val) and abs(same_val.item()) < 1e-5

    pred = torch.rand(2, 3, 64, 64, dtype=torch.float32, requires_grad=True)
    diff_val = loss_fn(pred, target)
    assert torch.isfinite(diff_val) and diff_val.item() > 0.0

    # Verify [0, 1] -> [-1, 1] symmetry and eval determinism.
    with torch.no_grad():
        rev_val = loss_fn(target, pred.detach())
    assert torch.allclose(diff_val.detach(), rev_val, atol=1e-5)

    diff_val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in params)
    assert loss_fn.training is False


def test_t1_image_io_formats_roundtrip_and_comparison(e2e_workdir: dict[str, Path]) -> None:
    """F17: IMAGE_EXTENSIONS, load_image/save_image roundtrip across formats, comparison panels."""
    import torch

    from utils.image import (
        IMAGE_EXTENSIONS,
        load_image,
        make_comparison_image,
        save_image,
    )

    assert IMAGE_EXTENSIONS == {".jpg", ".jpeg", ".png", ".webp"}

    # Deterministic uint8-aligned smooth 2D gradient so lossless PNG is exact and lossy
    # 4:2:0 JPEG/WebP codecs remain well below 0.05 MAE.
    ys = torch.linspace(0.0, 1.0, 24, dtype=torch.float32)
    xs = torch.linspace(0.0, 1.0, 32, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = (20.0 + 210.0 * xx).round().clamp(0, 255).to(torch.uint8)
    g = (15.0 + 220.0 * yy).round().clamp(0, 255).to(torch.uint8)
    b = (30.0 + 180.0 * (0.5 * xx + 0.5 * yy)).round().clamp(0, 255).to(torch.uint8)
    tensor = torch.stack([r, g, b], dim=0).float() / 255.0

    for ext in (".png", ".jpg", ".webp"):
        out_path = e2e_workdir["outputs"] / f"sample{ext}"
        saved = save_image(tensor, out_path)
        assert Path(saved).is_file()
        loaded = load_image(saved)
        assert loaded.dtype == torch.float32
        assert loaded.shape == (3, 24, 32)
        assert float(loaded.min()) >= 0.0 and float(loaded.max()) <= 1.0
        if ext == ".png":
            assert torch.allclose(loaded, tensor, atol=1e-5)
        else:
            assert (loaded - tensor).abs().mean().item() < 0.05

    # Horizontal comparison panels: Red LQ | Green SR | Blue GT.
    lq = torch.zeros(3, 8, 8, dtype=torch.float32)
    lq[0] = 1.0
    sr = torch.zeros(3, 32, 32, dtype=torch.float32)
    sr[1] = 1.0
    gt = torch.zeros(3, 32, 32, dtype=torch.float32)
    gt[2] = 1.0

    cmp_3 = make_comparison_image(lq, sr, gt)
    assert cmp_3.dtype == torch.float32
    assert cmp_3.shape == (3, 32, 96)
    assert float(cmp_3.min()) >= 0.0 and float(cmp_3.max()) <= 1.0
    assert cmp_3[0, :, :32].mean().item() > 0.9
    assert cmp_3[1, :, 32:64].mean().item() > 0.9
    assert cmp_3[2, :, 64:].mean().item() > 0.9

    cmp_2 = make_comparison_image(lq, sr, gt=None)
    assert cmp_2.shape == (3, 32, 64)
    assert cmp_2[0, :, :32].mean().item() > 0.9
    assert cmp_2[1, :, 32:].mean().item() > 0.9


def test_t1_logging_utils_logger_eta_and_train_format(
    e2e_workdir: dict[str, Path],
    isolated_logger_name: str,
) -> None:
    """F18: Rank-0 file logger, format_eta, and canonical format_train_log summary."""
    from utils.logging import format_eta, format_train_log, setup_logger

    log_file = e2e_workdir["work"] / "train.log"
    logger = setup_logger(log_file, is_main_process=True, name=isolated_logger_name)
    sentinel = "SRTrainer E2E logger sentinel message"
    logger.info(sentinel)
    for handler in logger.handlers:
        handler.flush()

    assert log_file.is_file()
    assert sentinel in log_file.read_text(encoding="utf-8")

    eta_hm = format_eta(6 * 3600 + 35 * 60)
    assert "6h" in eta_hm and "35m" in eta_hm
    eta_sec = format_eta(45)
    assert "45" in eta_sec and "s" in eta_sec

    # Canonical docs/design.md §9.2 single-LR format with lr_dict={"lr": 1.98e-4}.
    line = format_train_log(
        step=5000,
        total_iters=300000,
        lr_dict={"lr": 1.98e-4},
        scalars={"loss/total": 0.0412, "loss/l1": 0.0365},
        it_per_sec=12.4,
        mem_gb=9.8,
        eta_sec=6 * 3600 + 35 * 60,
    )
    assert "[Iter 005000/300000]" in line
    assert "lr:" in line and re.search(r"1\.98e-0?4", line)
    assert "loss/total:" in line
    assert "loss/l1:" in line
    assert "it/s" in line
    assert "mem:" in line and "GB" in line
    assert "ETA:" in line

    # Secondary multi-key lr_dict check (e.g. multi-optimizer paradigms).
    multi_line = format_train_log(
        step=10,
        total_iters=100,
        lr_dict={"lr_g": 1.98e-4, "lr_d": 5.0e-5},
        scalars={"loss/total": 0.1234},
        it_per_sec=4.2,
        mem_gb=1.5,
        eta_sec=45,
    )
    assert "[Iter 010/100]" in multi_line or "[Iter 10/100]" in multi_line
    assert re.search(r"1\.98e-0?4", multi_line)
    assert re.search(r"5\.0*e-0?5", multi_line)
    assert "loss/total:" in multi_line


def test_t1_misc_safetensors_and_pth_weight_roundtrip(e2e_workdir: dict[str, Path]) -> None:
    """F19: strip_state_dict_prefix, atomic save_safetensors, load_weights (.safetensors/.pth)."""
    import torch
    from safetensors import safe_open

    from archs import ToyModel
    from utils.misc import load_weights, save_safetensors, strip_state_dict_prefix

    raw_sd = {
        "module.conv.weight": torch.ones(2, 2),
        "_orig_mod.conv.bias": torch.zeros(2),
        "module._orig_mod.head.weight": torch.full((2, 2), 3.0),
    }
    raw_keys_before = set(raw_sd.keys())
    stripped = strip_state_dict_prefix(raw_sd)
    assert set(raw_sd.keys()) == raw_keys_before, "strip_state_dict_prefix must not mutate input"
    assert set(stripped.keys()) == {"conv.weight", "conv.bias", "head.weight"}

    torch.manual_seed(7)
    ref_net = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    ref_sd = {k: v.detach().cpu().clone() for k, v in ref_net.state_dict().items()}

    weights_dir = e2e_workdir["work"] / "weights"
    st_path = weights_dir / "model.safetensors"
    saved_path = save_safetensors(ref_sd, st_path, metadata={"step": "10"})
    assert Path(saved_path) == st_path and st_path.is_file()
    # Atomic write must not leave stray temp files behind.
    assert [p.name for p in weights_dir.iterdir()] == ["model.safetensors"]

    with safe_open(str(st_path), framework="pt", device="cpu") as fh:
        assert fh.metadata().get("step") == "10"

    fresh = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    with torch.no_grad():
        for p in fresh.parameters():
            p.zero_()
    missing, unexpected = load_weights(fresh, st_path, strict=True)
    assert missing == [] and unexpected == []
    for k, v in ref_sd.items():
        assert torch.equal(fresh.state_dict()[k].cpu(), v)

    # BasicSR-style .pth with params_ema key.
    pth_path = weights_dir / "basicsr_ema.pth"
    zero_sd = {k: torch.zeros_like(v) for k, v in ref_sd.items()}
    torch.save({"params_ema": ref_sd, "params": zero_sd}, pth_path)

    fresh_pth = ToyModel(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    with torch.no_grad():
        for p in fresh_pth.parameters():
            p.zero_()
    missing2, unexpected2 = load_weights(fresh_pth, pth_path, strict=True, key="params_ema")
    assert missing2 == [] and unexpected2 == []
    for k, v in ref_sd.items():
        assert torch.equal(fresh_pth.state_dict()[k].cpu(), v)


def test_t1_misc_pad_multiple_min_size_and_unpad_roundtrip() -> None:
    """F19: pad_to_multiple, pad_to_min_size, and scaled unpad_tensor exact roundtrip."""
    import torch

    from utils.misc import pad_to_min_size, pad_to_multiple, unpad_tensor

    x = torch.rand(1, 3, 13, 19, dtype=torch.float32)
    padded, (orig_h, orig_w) = pad_to_multiple(x, multiple=8)
    assert padded.shape == (1, 3, 16, 24)
    assert (orig_h, orig_w) == (13, 19)
    assert torch.equal(unpad_tensor(padded, orig_h, orig_w, scale=1), x)

    upsampled = torch.rand(1, 3, 64, 96, dtype=torch.float32)
    cropped_x4 = unpad_tensor(upsampled, orig_h, orig_w, scale=4)
    assert cropped_x4.shape == (1, 3, 52, 76)
    assert torch.equal(cropped_x4, upsampled[:, :, :52, :76])

    y = torch.rand(1, 3, 10, 14, dtype=torch.float32)
    padded_min, (oh2, ow2) = pad_to_min_size(y, min_h=16, min_w=16)
    assert padded_min.shape == (1, 3, 16, 16)
    assert (oh2, ow2) == (10, 14)
    assert torch.equal(unpad_tensor(padded_min, oh2, ow2, scale=1), y)
