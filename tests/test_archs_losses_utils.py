"""Unit tests for ToyModel, L1Loss/LPIPSLoss, and shared utils (including Iter 1 defect regressions)."""

from __future__ import annotations

import enum
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import hydra
import lpips
import numpy as np
import pytest
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf
from PIL import Image
from safetensors import safe_open
from torch.nn import functional as F

from losses import L1Loss, LPIPSLoss
from tests.dummy_arch import DummySRNet
from utils import (
    IMAGE_EXTENSIONS,
    flatten_config_for_tracker,
    format_eta,
    format_train_log,
    load_image,
    load_resolved_config,
    load_weights,
    log_images_to_trackers,
    make_comparison_image,
    normalize_mixed_precision,
    pad_to_min_size,
    pad_to_multiple,
    resolve_work_dir,
    save_image,
    save_resolved_config,
    save_safetensors,
    setup_experiment_dirs,
    setup_logger,
    strip_state_dict_prefix,
    unpad_tensor,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = str((REPO_ROOT / "configs").resolve())


@pytest.fixture
def reset_accelerate_state():
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


# ---------------------------------------------------------------------------
# Test-only DummySRNet sanity check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("in_hw", [(1, 1), (7, 11)])
def test_dummysrnet_scales_arbitrary_sizes_grad_and_bf16(
    scale: int,
    in_hw: tuple[int, int],
) -> None:
    net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=16, num_block=2, scale=scale)
    x = torch.randn(2, 3, *in_hw, requires_grad=True)
    y = net(x)
    assert y.shape == (2, 3, in_hw[0] * scale, in_hw[1] * scale)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        y_bf16 = net(torch.randn(1, 3, *in_hw))
    assert y_bf16.shape == (1, 3, in_hw[0] * scale, in_hw[1] * scale)
    assert torch.isfinite(y_bf16).all()


def test_dummysrnet_zero_blocks_channels_and_strict_validation() -> None:
    net_zero = DummySRNet(num_in_ch=1, num_out_ch=3, num_feat=8, num_block=0, scale=2)
    assert net_zero(torch.randn(1, 1, 5, 7)).shape == (1, 3, 10, 14)

    # Strict validation: reject bool, float (e.g. 4.0), and <= 0 for positive-int args
    for bad_pos in (0, -1, -4, False, True, 4.0, 2.5, "4", None):
        with pytest.raises(ValueError):
            DummySRNet(scale=bad_pos)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DummySRNet(num_in_ch=bad_pos)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DummySRNet(num_out_ch=bad_pos)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            DummySRNet(num_feat=bad_pos)  # type: ignore[arg-type]

    # Strict validation: reject < 0, bool, float for num_block
    for bad_blocks in (-1, -5, False, True, 2.0, 1.5, "2", None):
        with pytest.raises(ValueError):
            DummySRNet(num_block=bad_blocks)  # type: ignore[arg-type]

    # Forward input validation
    net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    with pytest.raises(TypeError):
        net(np.zeros((1, 3, 4, 4), dtype=np.float32))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        net(torch.randn(3, 4, 4))  # 3D
    with pytest.raises(ValueError):
        net(torch.randn(1, 4, 4, 4))  # wrong channels
    with pytest.raises(ValueError):
        net(torch.randn(1, 3, 0, 4))  # empty spatial dim


# ---------------------------------------------------------------------------
# Losses: L1Loss & LPIPSLoss (ONLY built-in losses)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_l1_loss_numeric_grad_and_validation(reduction: str) -> None:
    pred = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randn(2, 3, 8, 8)

    criterion = L1Loss(reduction=reduction)
    val = criterion(pred, target)
    expected = F.l1_loss(pred, target, reduction=reduction)
    assert torch.allclose(val, expected)

    if reduction == "mean":
        val.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    with pytest.raises(ValueError):
        L1Loss(reduction="invalid")
    with pytest.raises(ValueError):
        criterion(torch.randn(2, 3, 8, 8), torch.randn(1, 3, 8, 8))


@pytest.mark.parametrize("net", ["alex", "vgg", "squeeze"])
def test_lpips_loss_offline_nonnegative_frozen_eval_lock_and_grad(
    net: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    patch_lpips_offline(monkeypatch)
    torch.manual_seed(123)
    criterion = LPIPSLoss(net=net, version="0.1")

    # All internal parameters frozen
    params = list(criterion.parameters())
    assert len(params) > 0
    assert all(not p.requires_grad for p in params)

    # Permanently locked in eval mode even after .train(True)
    criterion.train(True)
    assert criterion.training is False
    assert criterion.lpips.training is False
    assert all(not m.training for m in criterion.modules())

    # Verify exact [0, 1] -> [-1, 1] rescaling passed to underlying lpips.LPIPS
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _hook(_mod, inputs):
        captured.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

    handle = criterion.lpips.register_forward_pre_hook(_hook)
    try:
        zeros = torch.zeros(1, 3, 64, 64)
        ones = torch.ones(1, 3, 64, 64)
        criterion(zeros, ones)
    finally:
        handle.remove()
    assert len(captured) == 1
    assert torch.allclose(captured[0][0], torch.full((1, 3, 64, 64), -1.0))
    assert torch.allclose(captured[0][1], torch.full((1, 3, 64, 64), 1.0))

    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 64, 64)

    # Identical inputs -> ~0, distinct inputs -> strictly > 0, deterministic across calls
    d_same = criterion(x, x)
    d_diff_1 = criterion(x, y)
    d_diff_2 = criterion(x, y)
    assert d_same.ndim == 0 and d_same.item() == pytest.approx(0.0, abs=1e-6)
    assert d_diff_1.item() > 0.0
    assert d_diff_1.item() == d_diff_2.item()

    # Gradient flow through pred only, BF16 autocast safe
    pred = x.clone().requires_grad_(True)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss_val = criterion(pred, y)
    loss_val.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in criterion.parameters())


def test_lpips_loss_validation_and_bundled_pth_weights_only_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.helpers.fake_backends import patch_lpips_offline

    patch_lpips_offline(monkeypatch)
    with pytest.raises(ValueError):
        LPIPSLoss(net="resnet")
    with pytest.raises(ValueError):
        LPIPSLoss(version="9.9")

    criterion = LPIPSLoss(net="alex")
    with pytest.raises(ValueError):
        criterion(torch.rand(3, 64, 64), torch.rand(3, 64, 64))
    with pytest.raises(ValueError):
        criterion(torch.rand(1, 1, 64, 64), torch.rand(1, 1, 64, 64))
    with pytest.raises(ValueError):
        criterion(torch.rand(1, 3, 64, 64), torch.rand(2, 3, 64, 64))

    # Verify bundled lpips v0.1 .pth files load cleanly with PyTorch default weights_only=True
    weights_dir = Path(lpips.__file__).resolve().parent / "weights" / "v0.1"
    for net_name in ("alex", "vgg", "squeeze"):
        pth_path = weights_dir / f"{net_name}.pth"
        assert pth_path.is_file()
        state = torch.load(pth_path, map_location="cpu", weights_only=True)
        assert isinstance(state, dict) and len(state) > 0


# ---------------------------------------------------------------------------
# Utils: config.py
# ---------------------------------------------------------------------------


def test_normalize_mixed_precision() -> None:
    assert normalize_mixed_precision(None) == "no"
    assert normalize_mixed_precision(False) == "no"
    for val in ("no", "NO", "none", "False", "fp32", "off", ""):
        assert normalize_mixed_precision(val) == "no"
    assert normalize_mixed_precision("fp16") == "fp16"
    assert normalize_mixed_precision("BF16") == "bf16"

    for bad in (True, "int8", 16):
        with pytest.raises(ValueError):
            normalize_mixed_precision(bad)


def test_resolve_work_dir_single_process(reset_accelerate_state) -> None:
    from accelerate.state import PartialState

    # append_timestamp=False -> unchanged
    cfg_no_ts = OmegaConf.create(
        {"work_dir": "experiments/demo", "trainer": {"append_timestamp": False, "resume": None}}
    )
    assert resolve_work_dir(cfg_no_ts) == Path("experiments/demo")
    assert PartialState._shared_state == {}

    # append_timestamp=True + trailing slash -> timestamped & leaves PartialState clean
    cfg_ts = OmegaConf.create(
        {"work_dir": "experiments/demo/", "trainer": {"append_timestamp": True, "resume": None}}
    )
    resolved = resolve_work_dir(cfg_ts)
    assert re.match(r"^experiments/demo_\d{8}_\d{6}$", resolved.as_posix())
    assert PartialState._shared_state == {}

    # Immediately creating Accelerator(cpu=True) must succeed cleanly
    acc = Accelerator(mixed_precision="no", cpu=True)
    assert acc.device.type == "cpu"

    # Resuming skips timestamp even when append_timestamp=True
    cfg_resume = OmegaConf.create(
        {"work_dir": "experiments/demo", "trainer": {"append_timestamp": True, "resume": "auto"}}
    )
    assert resolve_work_dir(cfg_resume) == Path("experiments/demo")

    with pytest.raises(ValueError):
        resolve_work_dir(OmegaConf.create({"work_dir": ""}))


@pytest.mark.parametrize("mp_mode", ["no", "bf16"])
def test_resolve_work_dir_distributed_2proc_cpu_torchrun(mp_mode: str, tmp_path: Path) -> None:
    out_dir = tmp_path / f"dist_{mp_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = f"""
import os, time
from pathlib import Path
import torch.distributed as dist
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedType
from utils.config import resolve_work_dir

if int(os.environ.get("RANK", "0")) == 1:
    time.sleep(1.2)
cfg = OmegaConf.create({{"work_dir": "experiments/dist_test/", "trainer": {{"append_timestamp": True, "resume": None}}}})
wd = resolve_work_dir(cfg)
acc = Accelerator(mixed_precision={mp_mode!r})
assert acc.num_processes == 2, f"Expected 2 processes, got {{acc.num_processes}}"
assert acc.distributed_type == DistributedType.MULTI_CPU, f"Got {{acc.distributed_type}}"
assert acc.mixed_precision == {mp_mode!r}, f"Got {{acc.mixed_precision}}"
(Path({str(out_dir)!r}) / f"rank_{{acc.process_index}}.txt").write_text(
    wd.as_posix(), encoding="utf-8"
)
acc.wait_for_everyone()
acc.end_training()
if dist.is_initialized():
    dist.destroy_process_group()
"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        master_port = int(sock.getsockname()[1])

    env = os.environ.copy()
    env["ACCELERATE_USE_CPU"] = "true"
    env["OMP_NUM_THREADS"] = "1"
    if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in env:
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    extra_pp = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{extra_pp}:{existing_pp}" if existing_pp else extra_pp
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={master_port}",
            "--no_python",
            sys.executable,
            "-c",
            script,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    wd0 = (out_dir / "rank_0.txt").read_text(encoding="utf-8").strip()
    wd1 = (out_dir / "rank_1.txt").read_text(encoding="utf-8").strip()
    assert wd0 == wd1
    assert re.match(r"^experiments/dist_test_\d{8}_\d{6}$", wd0)


def test_setup_experiment_dirs_and_resolved_config_io(tmp_path: Path) -> None:
    # Non-main process must not create any directories
    non_main_root = tmp_path / "non_main_exp"
    dirs_non_main = setup_experiment_dirs(non_main_root, is_main_process=False)
    assert set(dirs_non_main.keys()) == {"work_dir", "trackers", "checkpoints", "weights", "visuals"}
    assert not non_main_root.exists()

    # Main process creates all 5 canonical directories
    main_root = tmp_path / "main_exp"
    dirs_main = setup_experiment_dirs(main_root, is_main_process=True)
    for d in dirs_main.values():
        assert d.is_dir()

    # Resolved config save / load round-trip on real Hydra vosr2_onestep_text config
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        hydra_cfg = hydra.compose(config_name="train", overrides=["experiment=vosr2_onestep_text"])
    cfg_path = main_root / "config_resolved.yaml"
    save_resolved_config(hydra_cfg, cfg_path)
    raw_yaml = cfg_path.read_text(encoding="utf-8")
    assert "${" not in raw_yaml

    loaded = load_resolved_config(cfg_path)
    assert loaded.exp_name == "vosr2_onestep_text"
    assert loaded.work_dir == "experiments/vosr2_onestep_text"
    assert loaded.arch.vae._target_ == "archs.AutoencoderKLQwenImage2D"
    assert loaded.arch.dit._target_ == "archs.VOSR2DiT"
    assert loaded.arch.dino._target_ == "archs.DINOv2Encoder"
    assert loaded.arch.dit.depth == 36
    assert loaded.model._target_ == "models.VOSRModel"
    assert loaded.optim.scheduler.T_max == 10000
    assert list(loaded.optim.optimizer.betas) == [0.9, 0.95]

    with pytest.raises(FileNotFoundError):
        load_resolved_config(tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError):
        load_resolved_config(main_root)  # directory, not file


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("corrupt.yaml", "a: [1, 2\nb: :::invalid\n"),
        ("tabs.yaml", "a:\n\t- 1\n"),
        ("scalar_str.yaml", "hello_world\n"),
        ("scalar_int.yaml", "42\n"),
        ("scalar_float.yaml", "3.14\n"),
        ("scalar_bool.yaml", "true\n"),
        ("list_root.yaml", "- item1\n- item2\n"),
        ("empty.yaml", ""),
        ("comment_only.yaml", "# only a comment\n"),
        ("null_root.yaml", "null\n"),
        ("tilde_null.yaml", "~\n"),
    ],
)
def test_load_resolved_config_rejects_corrupt_and_non_mapping_yaml(
    tmp_path: Path,
    name: str,
    payload: str,
) -> None:
    bad_path = tmp_path / name
    bad_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_resolved_config(bad_path)


def test_load_resolved_config_rejects_invalid_utf8(tmp_path: Path) -> None:
    bad_bytes = tmp_path / "bad_utf8.yaml"
    bad_bytes.write_bytes(b"\xff\xfeinvalid")
    with pytest.raises(ValueError):
        load_resolved_config(bad_bytes)


def test_flatten_config_for_tracker_and_real_tensorboard_init(
    tmp_path: Path,
    reset_accelerate_state,
) -> None:
    with hydra.initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["experiment=vosr2_onestep_text"])

    flat = flatten_config_for_tracker(cfg)
    assert len(flat) > 10
    for k, v in flat.items():
        assert isinstance(k, str)
        assert type(v) in (bool, int, float, str), f"Unsafe type {type(v)} for {k}={v!r}"
    assert flat["exp_name"] == "vosr2_onestep_text"
    assert flat["arch.vae._target_"] == "archs.AutoencoderKLQwenImage2D"
    assert flat["arch.dit._target_"] == "archs.VOSR2DiT"
    assert flat["arch.dino._target_"] == "archs.DINOv2Encoder"
    assert flat["arch.dit.depth"] == 36
    assert flat["model._target_"] == "models.VOSRModel"
    assert flat["model.pretrain.dit.path"] == "null"
    assert flat["optim.optimizer.betas"] == "[0.9, 0.95]"
    assert yaml.safe_dump(flat)

    # Regression for Challenger 2 Defect 2: multi-element np.ndarray & IntEnum/StrEnum/subclasses
    class ModeInt(enum.IntEnum):
        FAST = 1
        SLOW = 2

    class ModeStr(enum.StrEnum):
        COSINE = "cosine"

    class CustomInt(int):
        pass

    tricky = {
        "empty": {},
        "flag": True,
        "none_val": None,
        "np_scalar_int": np.int64(7),
        "np_scalar_float": np.float32(1.5),
        "np_scalar_bool": np.bool_(False),
        "np_0d": np.array(2.5),
        "np_size1": np.array([9]),
        "np_1d": np.array([1, 2, 3]),
        "np_2d": np.array([[1, 2], [3, 4]]),
        "np_empty": np.array([]),
        "enum_int": ModeInt.FAST,
        "enum_str": ModeStr.COSINE,
        "enum_list": [ModeInt.FAST, ModeStr.COSINE],
        "custom_int": CustomInt(42),
    }
    extra = flatten_config_for_tracker(tricky)
    for k, v in extra.items():
        assert isinstance(k, str)
        assert type(v) in (bool, int, float, str), f"Unsafe type {type(v)} for {k}={v!r}"
    assert extra["empty"] == "{}"
    assert extra["flag"] is True
    assert extra["none_val"] == "null"
    assert extra["np_scalar_int"] == 7
    assert extra["np_scalar_float"] == pytest.approx(1.5)
    assert extra["np_scalar_bool"] is False
    assert extra["np_0d"] == pytest.approx(2.5)
    assert extra["np_size1"] == 9
    assert extra["np_1d"] == "[1, 2, 3]"
    assert extra["np_2d"] == "[[1, 2], [3, 4]]"
    assert extra["np_empty"] == "[]"
    assert extra["enum_int"] == "FAST"
    assert extra["enum_str"] == "COSINE"
    assert extra["enum_list"] == '["FAST", "COSINE"]'
    assert extra["custom_int"] == 42 and type(extra["custom_int"]) is int
    dumped = yaml.safe_dump(extra)
    assert isinstance(dumped, str) and "FAST" in dumped

    # Real Accelerate TensorBoard init_trackers must accept flattened config cleanly
    proj_cfg = ProjectConfiguration(
        project_dir=str(tmp_path),
        logging_dir=str(tmp_path / "trackers"),
        automatic_checkpoint_naming=False,
    )
    acc = Accelerator(
        mixed_precision="no",
        cpu=True,
        log_with="tensorboard",
        project_config=proj_cfg,
    )
    acc.init_trackers("tb_test", config=flat)
    acc.end_training()
    assert (tmp_path / "trackers" / "tb_test").is_dir()


# ---------------------------------------------------------------------------
# Utils: image.py
# ---------------------------------------------------------------------------


def test_image_io_roundtrip_modes_and_errors(tmp_path: Path) -> None:
    assert IMAGE_EXTENSIONS == {".jpg", ".jpeg", ".png", ".webp"}

    # Lossless PNG exact 0..255/255.0 round-trip
    levels = torch.arange(256, dtype=torch.float32) / 255.0
    tensor = levels.view(1, 16, 16).repeat(3, 1, 1)
    png_path = save_image(tensor, tmp_path / "nested" / "exact.png")
    assert png_path.is_file()
    loaded = load_image(png_path)
    assert loaded.shape == (3, 16, 16)
    assert loaded.dtype == torch.float32
    assert loaded.is_contiguous()
    assert torch.allclose(loaded, tensor, atol=1e-6)

    # 4D input, out-of-range clamp, JPEG / WEBP formats, explicit save_format override
    tensor_4d = (
        torch.tensor([-0.5, 0.5, 1.5], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 8, 8)
    )
    jpg_path = save_image(tensor_4d, tmp_path / "sample.png", save_format="jpg", jpg_quality=95)
    assert jpg_path.suffix == ".jpg" and jpg_path.is_file()
    assert load_image(jpg_path).shape == (3, 8, 8)

    webp_path = save_image(tensor, tmp_path / "sample.webp")
    assert webp_path.is_file() and load_image(webp_path).shape == (3, 16, 16)

    # Non-RGB modes (RGBA, L, P) convert cleanly to RGB [3, H, W]
    for mode, data in (
        ("RGBA", np.full((8, 10, 4), 128, dtype=np.uint8)),
        ("L", np.full((8, 10), 200, dtype=np.uint8)),
    ):
        p = tmp_path / f"img_{mode}.png"
        Image.fromarray(data, mode=mode).save(p)
        t = load_image(p)
        assert t.shape == (3, 8, 10) and t.dtype == torch.float32 and t.is_contiguous()

    pal_path = tmp_path / "img_P.png"
    Image.fromarray(np.full((8, 10, 3), 64, dtype=np.uint8)).convert("P").save(pal_path)
    assert load_image(pal_path).shape == (3, 8, 10)

    # Error checks
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "does_not_exist.png")
    with pytest.raises(ValueError):
        save_image(tensor, tmp_path / "bad.bmp")
    with pytest.raises(ValueError):
        save_image(torch.rand(1, 8, 8), tmp_path / "bad_ch.png")
    with pytest.raises(ValueError):
        save_image(tensor, tmp_path / "bad_q.jpg", jpg_quality=0)


def test_make_comparison_image() -> None:
    # Sharp step edge to trigger bicubic overshoot and verify strict [0, 1] clamping
    lq = torch.zeros(3, 8, 8)
    lq[:, :, 4:] = 1.0
    sr = torch.rand(1, 3, 32, 32)
    gt = torch.rand(3, 32, 32)

    cmp_3 = make_comparison_image(lq, sr, gt)
    assert cmp_3.shape == (3, 32, 96)
    assert cmp_3.dtype == torch.float32
    assert cmp_3.min().item() >= 0.0 and cmp_3.max().item() <= 1.0

    cmp_2 = make_comparison_image(lq.unsqueeze(0), sr, gt=None)
    assert cmp_2.shape == (3, 32, 64)

    with pytest.raises(ValueError):
        make_comparison_image(lq, sr, gt=torch.rand(3, 16, 16))


# ---------------------------------------------------------------------------
# Utils: logging.py
# ---------------------------------------------------------------------------


def test_setup_logger_idempotent_append_and_disabled_reset(tmp_path: Path) -> None:
    log_path = tmp_path / "exp" / "train.log"
    logger_name = "srtrainer.test_logger"

    # Simulate Hydra hydra/job_logging: disabled setting logger.disabled = True
    pre_logger = logging.getLogger(logger_name)
    pre_logger.disabled = True

    logger = setup_logger(log_path, is_main_process=True, name=logger_name)
    assert logger.disabled is False
    logger.info("first line")

    # Re-init must be idempotent (no duplicate handlers) and append (not overwrite)
    logger = setup_logger(log_path, is_main_process=True, name=logger_name)
    assert len(logger.handlers) == 2
    logger.info("second line")

    content = log_path.read_text(encoding="utf-8")
    assert content.count("first line") == 1
    assert content.count("second line") == 1

    # Non-main process stays silent and never creates log file
    non_main_log = tmp_path / "non_main" / "train.log"
    silent = setup_logger(non_main_log, is_main_process=False, name="srtrainer.test_silent")
    silent.info("should not appear")
    assert not non_main_log.exists()


def test_format_eta_and_format_train_log_exact_design_parity() -> None:
    assert format_eta(0) == "0s"
    assert format_eta(-5) == "0s"
    assert format_eta(45) == "45s"
    assert format_eta(750) == "12m 30s"
    assert format_eta(23700) == "6h 35m"
    assert format_eta(94500) == "1d 2h 15m"
    assert format_eta(float("inf")) == "--"
    assert format_eta(float("-inf")) == "--"
    assert format_eta(float("nan")) == "--"

    scalars = {
        "loss/l1": 0.0365,
        "loss/lpips": 0.0940,
        "loss/total": 0.0412,
        "grad_norm": 0.42,
    }
    expected = (
        "[Iter 005000/300000] lr: 1.98e-4 | loss/total: 0.0412 | loss/l1: 0.0365 | "
        "loss/lpips: 0.0940 | grad_norm: 0.42 | 12.4 it/s | mem: 9.8 GB | ETA: 6h 35m"
    )

    # Single-key lr_dict with key "lr" AND with non-"lr" key "net_g" (Reviewer 1 Defect 1 regression)
    for lr_dict in ({"lr": 1.98e-4}, {"net_g": 1.98e-4}):
        line = format_train_log(
            step=5000,
            total_iters=300000,
            lr_dict=lr_dict,
            scalars=scalars,
            it_per_sec=12.4,
            mem_gb=9.8,
            eta_sec=23700,
        )
        assert line == expected

    # Multi-optimizer lr_dict always starts with "lr:" and preserves optimizer keys
    multi_lr_line = format_train_log(
        step=5000,
        total_iters=300000,
        lr_dict={"net_g": 1.0e-4, "net_d": 2.0e-4},
        scalars={"loss/total": 0.0412},
        it_per_sec=12.4,
        mem_gb=9.8,
        eta_sec=23700,
    )
    assert multi_lr_line == (
        "[Iter 005000/300000] lr: net_g=1.00e-4, net_d=2.00e-4 | "
        "loss/total: 0.0412 | 12.4 it/s | mem: 9.8 GB | ETA: 6h 35m"
    )

    # Empty lr_dict / scalars edge case
    minimal = format_train_log(
        step=1,
        total_iters=10,
        lr_dict={},
        scalars={},
        it_per_sec=2.0,
        mem_gb=0.0,
        eta_sec=4.5,
    )
    assert minimal == "[Iter 01/10] 2.0 it/s | mem: 0.0 GB | ETA: 4s"


def test_log_images_to_trackers_real_tb_and_mocks(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    reset_accelerate_state,
) -> None:
    img_3d = torch.rand(3, 16, 16)
    img_4d = torch.rand(1, 3, 16, 16)

    # Empty trackers / non-main process no-ops
    log_images_to_trackers(SimpleNamespace(is_main_process=True, trackers=[]), "t", img_3d, 1)
    log_images_to_trackers(
        SimpleNamespace(is_main_process=False, trackers=[MagicMock()]), "t", img_3d, 1
    )

    # Real TensorBoard tracker
    proj_cfg = ProjectConfiguration(
        project_dir=str(tmp_path),
        logging_dir=str(tmp_path / "trackers"),
        automatic_checkpoint_naming=False,
    )
    acc = Accelerator(
        mixed_precision="no",
        cpu=True,
        log_with="tensorboard",
        project_config=proj_cfg,
    )
    acc.init_trackers("img_test")
    log_images_to_trackers(acc, "val/set5_sample0", img_3d, step=10)
    log_images_to_trackers(acc, "val/set5_sample1", img_4d, step=20)
    acc.end_training()
    event_files = list((tmp_path / "trackers" / "img_test").glob("events.out.tfevents.*"))
    assert len(event_files) >= 1 and event_files[0].stat().st_size > 0

    # Mocked WandB tracker (via fake sys.modules["wandb"])
    fake_run = MagicMock()
    fake_wandb = SimpleNamespace(Image=lambda arr: ("wandb_img", arr.shape, arr.dtype))
    fake_tracker = SimpleNamespace(name="wandb", tracker=fake_run)
    orig_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        log_images_to_trackers(
            SimpleNamespace(is_main_process=True, trackers=[fake_tracker]),
            "val/demo",
            img_3d,
            step=5,
        )
    finally:
        if orig_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = orig_wandb
    fake_run.log.assert_called_once()
    logged_payload, logged_kwargs = fake_run.log.call_args
    assert logged_kwargs["step"] == 5
    assert logged_payload[0]["val/demo"] == ("wandb_img", (16, 16, 3), np.uint8)

    # Unsupported tracker warns once across repeated calls and never raises
    unsupported = SimpleNamespace(name="custom_mock_tracker_xyz", tracker=object())
    sr_logger = logging.getLogger("srtrainer")
    orig_disabled = sr_logger.disabled
    orig_propagate = sr_logger.propagate
    sr_logger.disabled = False
    # pytest 9+ attaches caplog.handler directly to non-propagating loggers; enabling
    # propagate when caplog.handler is already attached would capture each record twice.
    if caplog.handler not in sr_logger.handlers:
        sr_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="srtrainer"):
            log_images_to_trackers(
                SimpleNamespace(is_main_process=True, trackers=[unsupported]), "val/u", img_3d, 1
            )
            log_images_to_trackers(
                SimpleNamespace(is_main_process=True, trackers=[unsupported]), "val/u", img_3d, 2
            )
    finally:
        sr_logger.disabled = orig_disabled
        sr_logger.propagate = orig_propagate
    warnings_for_custom = [r for r in caplog.records if "custom_mock_tracker_xyz" in r.message]
    assert len(warnings_for_custom) == 1


# ---------------------------------------------------------------------------
# Utils: misc.py
# ---------------------------------------------------------------------------


def test_strip_prefix_save_safetensors_and_load_weights(tmp_path: Path) -> None:
    t = torch.randn(4, 4)
    raw = {
        "module.weight": t,
        "_orig_mod.bias": t,
        "module._orig_mod.nested": t.t(),  # non-contiguous + aliased storage!
        "_orig_mod.module.other": t,
        "plain": t,
    }
    stripped = strip_state_dict_prefix(raw)
    assert set(stripped.keys()) == {"weight", "bias", "nested", "other", "plain"}
    assert "module.weight" in raw  # original not mutated

    # save_safetensors handles shared storage, non-contiguous views, metadata, and atomic write
    st_path = save_safetensors(
        raw, tmp_path / "weights" / "shared.safetensors", metadata={"step": "42"}
    )
    assert st_path.is_file()
    assert not list((tmp_path / "weights").glob("*.tmp"))
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        assert f.metadata() == {"step": "42"}
        assert set(f.keys()) == {"weight", "bias", "nested", "other", "plain"}

    # Round-trip load_weights with .safetensors (strict=True & strict=False) using DummySRNet
    model_src = DummySRNet(num_feat=8, num_block=1, scale=2)
    model_dst = DummySRNet(num_feat=8, num_block=1, scale=2)
    prefixed_sd = {f"module._orig_mod.{k}": v for k, v in model_src.state_dict().items()}
    w_path = save_safetensors(prefixed_sd, tmp_path / "toy.safetensors")

    missing, unexpected = load_weights(model_dst, w_path, strict=True)
    assert missing == [] and unexpected == []
    for k, v in model_src.state_dict().items():
        assert torch.equal(model_dst.state_dict()[k], v)

    # strict=False returns (missing, unexpected)
    partial_sd = dict(model_src.state_dict())
    removed_key = next(iter(partial_sd))
    del partial_sd[removed_key]
    partial_sd["extra_dummy"] = torch.zeros(1)
    partial_path = save_safetensors(partial_sd, tmp_path / "partial.safetensors")
    missing_f, unexpected_f = load_weights(model_dst, partial_path, strict=False)
    assert missing_f == [removed_key]
    assert unexpected_f == ["extra_dummy"]

    with pytest.raises(RuntimeError):
        load_weights(model_dst, partial_path, strict=True)
    with pytest.raises(KeyError):
        load_weights(model_dst, w_path, key="params")

    # .pth flat & nested checkpoints
    pth_flat = tmp_path / "flat.pth"
    torch.save(prefixed_sd, pth_flat)
    model_dst2 = DummySRNet(num_feat=8, num_block=1, scale=2)
    assert load_weights(model_dst2, pth_flat, strict=True) == ([], [])

    pth_nested = tmp_path / "nested.pth"
    torch.save({"params_ema": prefixed_sd, "params": prefixed_sd}, pth_nested)
    model_dst3 = DummySRNet(num_feat=8, num_block=1, scale=2)
    assert load_weights(model_dst3, pth_nested, strict=True, key="params_ema") == ([], [])

    with pytest.raises(KeyError, match="params_ema"):
        load_weights(model_dst3, pth_nested, key="nonexistent")
    with pytest.raises(ValueError, match="nested"):
        load_weights(model_dst3, pth_nested, key=None)
    with pytest.raises(FileNotFoundError):
        load_weights(model_dst3, tmp_path / "missing.safetensors")
    bad_ext = tmp_path / "model.onnx"
    bad_ext.write_bytes(b"dummy")
    with pytest.raises(ValueError):
        load_weights(model_dst3, bad_ext)


def test_pad_and_unpad_helpers_including_reflect_fallback() -> None:
    # 4D pad_to_multiple + unpad_tensor with scale=1 and scale=4
    x4 = torch.randn(1, 3, 13, 19)
    p4, (oh, ow) = pad_to_multiple(x4, multiple=8, mode="reflect")
    assert (oh, ow) == (13, 19)
    assert p4.shape == (1, 3, 16, 24)
    assert torch.equal(unpad_tensor(p4, oh, ow, scale=1), x4)

    p4_up = F.interpolate(p4, scale_factor=4, mode="nearest")
    unpadded_up = unpad_tensor(p4_up, oh, ow, scale=4)
    assert unpadded_up.shape == (1, 3, 52, 76)

    # 3D fast-path (pad == 0)
    x3_aligned = torch.randn(3, 16, 24)
    p3_same, orig = pad_to_multiple(x3_aligned, multiple=8)
    assert orig == (16, 24) and p3_same.shape == (3, 16, 24)

    # Tiny tensors trigger reflect -> replicate fallback without RuntimeError
    for tiny_shape in ((3, 1, 1), (1, 3, 3, 5), (3, 2, 10)):
        tiny = torch.randn(*tiny_shape)
        pm, (th, tw) = pad_to_multiple(tiny, multiple=8, mode="reflect")
        assert pm.shape[-2] % 8 == 0 and pm.shape[-1] % 8 == 0
        assert torch.equal(unpad_tensor(pm, th, tw), tiny)

        pmin, (mh, mw) = pad_to_min_size(tiny, min_h=16, min_w=16, mode="reflect")
        assert pmin.shape[-2] >= 16 and pmin.shape[-1] >= 16
        assert torch.equal(unpad_tensor(pmin, mh, mw), tiny)

    # Validation errors
    with pytest.raises(ValueError):
        pad_to_multiple(x4, multiple=0)
    with pytest.raises(ValueError):
        pad_to_min_size(x4, min_h=0, min_w=8)
    with pytest.raises(ValueError):
        unpad_tensor(x4, orig_h=20, orig_w=20)
