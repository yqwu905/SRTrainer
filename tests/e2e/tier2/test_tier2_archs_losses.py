"""Tier 2 E2E tests — Area B: Placeholder Architecture (ToyModel), Losses & Core Utilities.

Covers boundary, edge, and negative contracts for F13-F15 and F17-F19 using
exclusively ``ToyModel``, ``L1Loss``, and ``LPIPSLoss(pretrained=False)``.
All non-stdlib imports (besides ``pytest``) are strictly deferred inside test functions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest


def test_t2_b01_dummysrnet_edge_scales_tiny_dims_channels_and_invalid_args() -> None:
    """F13: DummySRNet supports scale=1/3/4, tiny prime dims, 1-ch, 0-block, and rejects bad args."""
    import torch
    from tests.dummy_arch import DummySRNet

    for scale in (1, 3, 4):
        net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=scale)
        for h, w in ((1, 1), (3, 5), (7, 9)):
            net.zero_grad(set_to_none=True)
            x = torch.rand(1, 3, h, w, dtype=torch.float32, requires_grad=True)
            y = net(x)
            assert y.shape == (1, 3, h * scale, w * scale)
            assert y.dtype == torch.float32
            assert torch.isfinite(y).all()
            y.mean().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            grads = [p.grad for p in net.parameters() if p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    gray_net = DummySRNet(num_in_ch=1, num_out_ch=1, num_feat=8, num_block=1, scale=2)
    x_gray = torch.rand(1, 1, 3, 5, dtype=torch.float32, requires_grad=True)
    y_gray = gray_net(x_gray)
    assert y_gray.shape == (1, 1, 6, 10)
    assert torch.isfinite(y_gray).all()
    y_gray.mean().backward()
    assert x_gray.grad is not None and torch.isfinite(x_gray.grad).all()

    zero_block_net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=0, scale=2)
    y_zb = zero_block_net(torch.rand(1, 3, 3, 5, dtype=torch.float32))
    assert y_zb.shape == (1, 3, 6, 10)
    assert torch.isfinite(y_zb).all()

    invalid_kwargs_list = [
        {"scale": 0},
        {"scale": -2},
        {"num_in_ch": 0},
        {"num_out_ch": 0},
        {"num_feat": 0},
        {"num_block": -1},
    ]
    for bad_kwargs in invalid_kwargs_list:
        with pytest.raises(ValueError, match="."):
            DummySRNet(**bad_kwargs)


def test_t2_b02_l1_loss_zero_diff_large_residuals_and_invalid_reduction() -> None:
    """F14: L1Loss handles exact zero residuals, large residuals, and invalid reduction."""
    import torch

    from losses import L1Loss

    torch.manual_seed(14)
    target = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    pred = target.clone().requires_grad_(True)
    val = L1Loss(reduction="mean")(pred, target)
    assert val.item() == pytest.approx(0.0, abs=1e-7)
    val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.equal(pred.grad, torch.zeros_like(pred))

    pred_big = torch.full((2, 3, 8, 8), 1e4, dtype=torch.float32, requires_grad=True)
    target_big = torch.full((2, 3, 8, 8), -1e4, dtype=torch.float32)
    val_big = L1Loss(reduction="mean")(pred_big, target_big)
    assert torch.isfinite(val_big)
    assert val_big.item() == pytest.approx(2e4, rel=1e-5)
    val_big.backward()
    assert pred_big.grad is not None and torch.isfinite(pred_big.grad).all()

    with pytest.raises(ValueError, match="."):
        L1Loss(reduction="invalid")


def test_t2_b03_lpips_loss_train_toggle_frozen_vgg_offline_and_invalid_net(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F15: LPIPSLoss stays frozen/eval across train(True), supports VGG, rejects bad net."""
    import torch
    from tests.helpers.fake_backends import patch_lpips_offline

    from losses import LPIPSLoss

    patch_lpips_offline(monkeypatch)
    loss_fn = LPIPSLoss(net="alex")
    loss_fn.train(True)
    assert loss_fn.training is False, "LPIPSLoss.train(True) must preserve eval mode"

    params = list(loss_fn.parameters())
    assert len(params) > 0 and all(not p.requires_grad for p in params)

    torch.manual_seed(15)
    pred_base = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    target = torch.rand(1, 3, 64, 64, dtype=torch.float32)
    pred1 = pred_base.clone().requires_grad_(True)
    pred2 = pred_base.clone().requires_grad_(True)

    out1 = loss_fn(pred1, target)
    out2 = loss_fn(pred2, target)
    assert torch.equal(out1.detach(), out2.detach())

    out1.backward()
    assert pred1.grad is not None and torch.isfinite(pred1.grad).all()
    assert all(p.grad is None for p in params)

    vgg_loss = LPIPSLoss(net="vgg")
    v = vgg_loss(
        torch.rand(1, 3, 64, 64, dtype=torch.float32),
        torch.rand(1, 3, 64, 64, dtype=torch.float32),
    )
    assert v.ndim == 0 and torch.isfinite(v) and v.item() >= 0.0

    with pytest.raises(ValueError, match="."):
        LPIPSLoss(net="invalid_net")


def test_t2_b04_image_utils_grayscale_rgba_clamp_4d_nested_and_missing(
    e2e_workdir: dict[str, Path],
    synthetic_image_factory: Callable[..., Path],
) -> None:
    """F17: load_image/save_image/make_comparison_image handle L, RGBA, 4D clamp, nested dirs."""
    import torch

    from utils.image import load_image, make_comparison_image, save_image

    p_gray = synthetic_image_factory(
        e2e_workdir["data"] / "gray.png",
        width=12,
        height=10,
        mode="L",
        pattern="solid",
        color=(128,),
    )
    t_gray = load_image(p_gray)
    assert t_gray.shape == (3, 10, 12)
    assert t_gray.dtype == torch.float32
    assert torch.equal(t_gray[0], t_gray[1]) and torch.equal(t_gray[1], t_gray[2])
    assert t_gray.mean().item() == pytest.approx(128.0 / 255.0, abs=1e-3)

    p_rgba = synthetic_image_factory(
        e2e_workdir["data"] / "rgba.png",
        width=12,
        height=10,
        mode="RGBA",
        pattern="solid",
        color=(255, 0, 128, 100),
    )
    t_rgba = load_image(p_rgba)
    assert t_rgba.shape == (3, 10, 12)
    assert t_rgba.dtype == torch.float32
    assert t_rgba[0].mean().item() == pytest.approx(1.0, abs=1e-3)
    assert t_rgba[1].mean().item() == pytest.approx(0.0, abs=1e-3)
    assert t_rgba[2].mean().item() == pytest.approx(128.0 / 255.0, abs=1e-3)

    t_4d = torch.linspace(-0.5, 1.8, 3 * 8 * 10, dtype=torch.float32).reshape(1, 3, 8, 10)
    nested_path = e2e_workdir["outputs"] / "a" / "b" / "clamped.png"
    saved = save_image(t_4d, nested_path)
    assert Path(saved).is_file()
    reloaded = load_image(saved)
    assert reloaded.shape == (3, 8, 10)
    assert reloaded.min().item() == pytest.approx(0.0, abs=1e-5)
    assert reloaded.max().item() == pytest.approx(1.0, abs=1e-5)

    lq_4d = torch.rand(1, 3, 4, 6, dtype=torch.float32)
    sr_4d = torch.rand(1, 3, 16, 24, dtype=torch.float32)
    gt_4d = torch.rand(1, 3, 16, 24, dtype=torch.float32)
    comp3 = make_comparison_image(lq_4d, sr_4d, gt_4d)
    assert comp3.shape == (3, 16, 72)
    assert float(comp3.min()) >= 0.0 and float(comp3.max()) <= 1.0

    comp2 = make_comparison_image(lq_4d, sr_4d, gt=None)
    assert comp2.shape == (3, 16, 48)
    assert float(comp2.min()) >= 0.0 and float(comp2.max()) <= 1.0

    with pytest.raises(FileNotFoundError):
        load_image(e2e_workdir["data"] / "does_not_exist.png")


def test_t2_b05_logging_utils_non_main_dedup_eta_edges_and_noop_tracker(
    e2e_workdir: dict[str, Path],
    isolated_logger_name: str,
) -> None:
    """F18: setup_logger suppresses non-main writes, deduplicates handlers; format_eta edge safety."""
    import torch
    from accelerate import Accelerator

    from utils.logging import format_eta, log_images_to_trackers, setup_logger

    nm_name = f"{isolated_logger_name}.nm"
    nm_log = e2e_workdir["work"] / "nm_dir" / "train.log"
    try:
        lg_nm = setup_logger(nm_log, is_main_process=False, name=nm_name)
        lg_nm.info("ignored non-main log message")
        assert not nm_log.exists()
        assert not nm_log.parent.exists()
    finally:
        nm_logger = logging.getLogger(nm_name)
        for handler in list(nm_logger.handlers):
            nm_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    main_log = e2e_workdir["work"] / "dedup.log"
    lg1 = setup_logger(main_log, is_main_process=True, name=isolated_logger_name)
    n1 = len(lg1.handlers)
    lg2 = setup_logger(main_log, is_main_process=True, name=isolated_logger_name)
    assert lg1 is lg2
    assert len(lg2.handlers) == n1

    sentinel = "DEDUP_SENTINEL_E2E_B05"
    lg2.info(sentinel)
    for handler in lg2.handlers:
        handler.flush()
    assert main_log.read_text(encoding="utf-8").count(sentinel) == 1

    assert format_eta(0) == "0s"
    assert format_eta(-1.0) == "0s"
    for bad_sec in (float("inf"), float("-inf"), float("nan")):
        s = format_eta(bad_sec)
        assert isinstance(s, str) and len(s) > 0
        assert "nan" not in s.lower() and "inf" not in s.lower()

    s_long = format_eta(50 * 3600 + 125)
    assert isinstance(s_long, str) and ("d" in s_long or "h" in s_long)

    acc = Accelerator(cpu=True, log_with=None)
    log_images_to_trackers(acc, "val/noop", torch.rand(3, 8, 8, dtype=torch.float32), step=1)


def test_t2_b06_misc_safetensors_aliasing_noncontig_and_load_weights_errors(
    e2e_workdir: dict[str, Path],
) -> None:
    """F19: save_safetensors handles non-contiguous/aliased/grad tensors; load_weights error paths."""
    import torch
    from safetensors.torch import load_file
    from tests.dummy_arch import DummySRNet

    from utils.misc import load_weights, save_safetensors

    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    transposed = base.t()
    assert not transposed.is_contiguous()
    aliased = base
    grad_t = torch.full((2, 2), 2.5, dtype=torch.float32, requires_grad=True)

    sd = {
        "module._orig_mod.a": transposed,
        "module.b": aliased,
        "_orig_mod.c": grad_t,
    }
    st_path = save_safetensors(sd, e2e_workdir["work"] / "edge.safetensors")
    loaded = load_file(str(st_path))
    assert set(loaded.keys()) == {"a", "b", "c"}
    assert torch.equal(loaded["a"], transposed.contiguous())
    assert torch.equal(loaded["b"], base)
    assert torch.equal(loaded["c"], grad_t.detach())

    net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    with pytest.raises(FileNotFoundError):
        load_weights(net, e2e_workdir["work"] / "nonexistent_weights.safetensors", strict=True)

    bad_st = save_safetensors(
        {"wrong_key": torch.zeros(2, 2, dtype=torch.float32)},
        e2e_workdir["work"] / "mismatch.safetensors",
    )
    with pytest.raises(RuntimeError):
        load_weights(net, bad_st, strict=True)

    missing, unexpected = load_weights(net, bad_st, strict=False)
    assert len(missing) > 0
    assert "wrong_key" in unexpected

    pth_path = e2e_workdir["work"] / "params_only.pth"
    torch.save({"params": net.state_dict()}, pth_path)
    with pytest.raises((KeyError, ValueError)):
        load_weights(net, pth_path, strict=True, key="no_such_key")


def test_t2_b07_misc_pad_reflect_fallback_and_aligned_noop() -> None:
    """F19: pad_to_multiple & pad_to_min_size fall back reflect->replicate and no-op when aligned."""
    import torch

    from utils.misc import pad_to_min_size, pad_to_multiple, unpad_tensor

    torch.manual_seed(19)
    x = torch.rand(1, 3, 3, 3, dtype=torch.float32)
    px, (oh, ow) = pad_to_multiple(x, multiple=8, mode="reflect")
    assert px.shape == (1, 3, 8, 8)
    assert (oh, ow) == (3, 3)
    assert torch.isfinite(px).all()
    assert torch.equal(unpad_tensor(px, oh, ow, scale=1), x)

    y = torch.rand(1, 3, 2, 3, dtype=torch.float32)
    py, (oh2, ow2) = pad_to_min_size(y, min_h=16, min_w=16, mode="reflect")
    assert py.shape == (1, 3, 16, 16)
    assert (oh2, ow2) == (2, 3)
    assert torch.isfinite(py).all()
    assert torch.equal(unpad_tensor(py, oh2, ow2, scale=1), y)

    z = torch.rand(1, 3, 16, 24, dtype=torch.float32)
    pz, (oh3, ow3) = pad_to_multiple(z, multiple=8, mode="reflect")
    assert pz.shape == z.shape and (oh3, ow3) == (16, 24) and torch.equal(pz, z)

    pz_min, (oh4, ow4) = pad_to_min_size(z, min_h=16, min_w=16, mode="reflect")
    assert pz_min.shape == z.shape and (oh4, ow4) == (16, 24) and torch.equal(pz_min, z)
