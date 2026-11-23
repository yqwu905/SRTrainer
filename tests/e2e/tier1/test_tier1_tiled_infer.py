"""Tier 1 E2E tests — Area F: Shared Tiled Inference & Multi-Process Distributed CLI.

Covers features F35-F40 and User Requirement #4 (NO SRPredictor).
Imports ``from engine.inferencer import TiledInferencer`` inside test
functions so Area F executes cleanly on M4 before M5 populates ``engine/__init__.py``.
All non-stdlib imports (besides ``pytest``) are deferred inside test functions.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


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


def _yaml_obj_contains_token(obj: Any, token: str) -> bool:
    """Return True if parsed YAML structure contains ``token`` in any key or string value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and token in k:
                return True
            if _yaml_obj_contains_token(v, token):
                return True
    elif isinstance(obj, list):
        return any(_yaml_obj_contains_token(item, token) for item in obj)
    elif isinstance(obj, str):
        return token in obj
    return False


def test_t1_no_srpredictor_across_repo_and_exports(repo_root: Path) -> None:
    """F35, Req #4: Zero occurrences of SRPredictor across src/, tools/, configs/ and exports."""
    import yaml

    for folder_name in ("src", "tools", "configs"):
        folder = repo_root / folder_name
        assert folder.is_dir(), f"Expected directory {folder} to exist"
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            suffix = path.suffix.lower()
            if suffix == ".py":
                source = path.read_text(encoding="utf-8", errors="replace")
                assert not _py_ast_contains_symbol(source, "SRPredictor"), (
                    f"Forbidden SRPredictor symbol found in {path}"
                )
            elif suffix in {".yaml", ".yml"}:
                parsed = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
                assert not _yaml_obj_contains_token(parsed, "SRPredictor"), (
                    f"Forbidden SRPredictor reference found in {path}"
                )

    import engine
    import engine.inferencer as inferencer_mod

    assert not hasattr(engine, "SRPredictor")
    assert not hasattr(inferencer_mod, "SRPredictor")


def test_t1_tiled_inferencer_auto_int_and_hw_tile_sizes_from_config() -> None:
    """F36: TiledInferencer.from_config supports size='auto', int, and [H, W] cleanly."""
    import torch
    import torch.nn as nn
    from omegaconf import OmegaConf

    from engine.inferencer import TiledInferencer

    model = nn.Upsample(scale_factor=2, mode="nearest")
    torch.manual_seed(0)
    lq = torch.rand(1, 3, 40, 48, dtype=torch.float32)
    ref = model(lq)

    configs = [
        {
            "enabled": True,
            "size": "auto",
            "overlap": 4,
            "batch_size": 4,
            "blend_mode": "cosine",
            "pad_multiple": 8,
        },
        {
            "enabled": True,
            "size": 16,
            "overlap": 4,
            "batch_size": 4,
            "blend_mode": "cosine",
            "pad_multiple": 8,
        },
        {
            "enabled": True,
            "size": [16, 24],
            "overlap": 4,
            "batch_size": 4,
            "blend_mode": "cosine",
            "pad_multiple": 8,
        },
    ]

    from_cfg_params = inspect.signature(TiledInferencer.from_config).parameters
    cfg_kwarg = next(
        (name for name in ("tile_cfg", "cfg", "tile_config", "config") if name in from_cfg_params),
        "tile_cfg",
    )
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in from_cfg_params.values())

    for tile_cfg in configs:
        call_kwargs: dict[str, Any] = {
            "model": model,
            "scale": 2,
            cfg_kwarg: OmegaConf.create(tile_cfg),
            "gt_size": 32,
            "mixed_precision": "no",
            "device": "cpu",
        }
        if not has_var_kw:
            call_kwargs = {k: v for k, v in call_kwargs.items() if k in from_cfg_params}
        inferencer = TiledInferencer.from_config(**call_kwargs)
        out = inferencer.infer_tensor(lq)
        assert out.shape == (1, 3, 80, 96)
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
        assert (out - ref).abs().max().item() < 1e-5


def test_t1_tiled_inferencer_feathering_seamlessness_and_batch_parity() -> None:
    """F36: Cosine/linear/mean 2D feathering produces seamless output invariant to tile_batch_size."""
    import torch
    import torch.nn as nn

    from engine.inferencer import TiledInferencer

    ys = torch.linspace(0.05, 0.95, 40, dtype=torch.float32)
    xs = torch.linspace(0.05, 0.95, 48, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    lq = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0).unsqueeze(0)

    model = nn.Upsample(scale_factor=2, mode="nearest")
    ref = model(lq)

    for blend_mode in ("cosine", "linear", "mean"):
        inf_b1 = TiledInferencer(
            model=model,
            scale=2,
            tile_enabled=True,
            tile_size=16,
            tile_overlap=4,
            tile_batch_size=1,
            blend_mode=blend_mode,
            pad_multiple=8,
            mixed_precision="no",
            device="cpu",
        )
        inf_b4 = TiledInferencer(
            model=model,
            scale=2,
            tile_enabled=True,
            tile_size=16,
            tile_overlap=4,
            tile_batch_size=4,
            blend_mode=blend_mode,
            pad_multiple=8,
            mixed_precision="no",
            device="cpu",
        )
        out_b1 = inf_b1.infer_tensor(lq)
        out_b4 = inf_b4.infer_tensor(lq)
        assert out_b1.shape == ref.shape
        assert (out_b1 - ref).abs().max().item() < 1e-5, f"Seam error under blend_mode={blend_mode}"
        assert torch.allclose(out_b1, out_b4, atol=1e-6)


def test_t1_tiled_inferencer_preserves_training_flag_and_clamps_range() -> None:
    """F36: TiledInferencer runs in eval(), restores model.training, and clamps output to [0, 1]."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from engine.inferencer import TiledInferencer

    class _OutOfRangeProbeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_training: list[bool] = []

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.seen_training.append(self.training)
            up = F.interpolate(x, scale_factor=2, mode="nearest")
            return 2.3 * up - 0.5  # Maps [0, 1] -> [-0.5, 1.8]

    model = _OutOfRangeProbeModel()
    lq = torch.linspace(0.0, 1.0, 20 * 24, dtype=torch.float32).view(1, 1, 20, 24).repeat(1, 3, 1, 1)

    inferencer = TiledInferencer(
        model=model,
        scale=2,
        tile_enabled=True,
        tile_size=16,
        tile_overlap=4,
        tile_batch_size=2,
        blend_mode="cosine",
        device="cpu",
    )

    model.train()
    assert model.training is True
    out_from_train = inferencer.infer_tensor(lq)
    assert model.training is True, "TiledInferencer must restore model.training=True"
    assert model.seen_training and all(flag is False for flag in model.seen_training)
    assert out_from_train.dtype == torch.float32
    assert float(out_from_train.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(out_from_train.max()) == pytest.approx(1.0, abs=1e-6)

    model.eval()
    assert model.training is False
    out_from_eval = inferencer.infer_tensor(lq)
    assert model.training is False, "TiledInferencer must keep model.training=False"
    assert torch.allclose(out_from_train, out_from_eval, atol=1e-6)


def test_t1_tiled_inferencer_untiled_pad_multiple_exact_crop() -> None:
    """F36: Untiled mode pads odd dimensions to pad_multiple and crops back to exact H*s, W*s."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from engine.inferencer import TiledInferencer

    class _MultipleOf8StrictModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.shape[-2] % 8 == 0 and x.shape[-1] % 8 == 0, (
                f"Expected input padded to multiple of 8, got {tuple(x.shape)}"
            )
            return F.interpolate(x, scale_factor=2, mode="nearest")

    torch.manual_seed(6)
    lq = torch.rand(1, 3, 19, 27, dtype=torch.float32)
    inferencer = TiledInferencer(
        model=_MultipleOf8StrictModel(),
        scale=2,
        tile_enabled=False,
        pad_multiple=8,
        mixed_precision="no",
        device="cpu",
    )
    out = inferencer.infer_tensor(lq)
    assert out.shape == (1, 3, 38, 54)
    assert out.dtype == torch.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    expected = F.interpolate(lq, scale_factor=2, mode="nearest")
    assert torch.allclose(out, expected, atol=1e-5)


def test_t1_tools_infer_mode1_exp_dir_restore_and_weight_priority(
    e2e_workdir: dict[str, Path],
    bias_tagged_exp_dir_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F37: tools/infer.py Mode 1 auto-restores exp_dir and honors best -> latest_ema -> latest -> override."""
    from utils.image import load_image

    exp_info = bias_tagged_exp_dir_factory(
        e2e_workdir["work"] / "exp_mode1",
        scale=2,
        gt_size=32,
        include_best=True,
        include_ema=True,
    )
    exp_dir: Path = exp_info["exp_dir"]
    exp_means: dict[str, float] = exp_info["expected_means"]

    # Solid black 20x24 image (> gt_size//scale == 16 so auto tiling activates;
    # zero input ensures any skip connection contributes 0 and output matches expected_means).
    input_img = synthetic_image_factory(
        e2e_workdir["data"] / "probe.png",
        width=24,
        height=20,
        pattern="solid",
        color=(0, 0, 0),
    )

    def _run_and_measure(tag: str, extra: list[str] | None = None) -> float:
        out_dir = e2e_workdir["outputs"] / tag
        overrides = [
            f"infer.exp_dir={hydra_value_formatter(str(exp_dir))}",
            f"infer.input={hydra_value_formatter(str(input_img))}",
            f"infer.output_dir={hydra_value_formatter(str(out_dir))}",
            "infer.mixed_precision='no'",
            "infer.tile.enabled=true",
            "infer.tile.size=auto",
            "infer.tile.overlap=4",
            *(extra or []),
        ]
        proc = infer_cli_runner(overrides, check=True)
        assert proc.returncode == 0
        out_png = out_dir / "probe.png"
        assert out_png.is_file()
        tensor = load_image(out_png)
        assert tensor.shape == (3, 40, 48)
        return float(tensor.mean().item())

    # 1. best_* present -> highest priority.
    m_best = _run_and_measure("prio_best")
    assert m_best == pytest.approx(exp_means["best"], abs=0.02)

    # 2. Remove best_* -> falls back to latest_ema (use_ema=true).
    exp_info["paths"]["best"].unlink()
    m_ema = _run_and_measure("prio_ema", ["infer.use_ema=true"])
    assert m_ema == pytest.approx(exp_means["latest_ema"], abs=0.02)

    # 3. use_ema=false -> selects latest.safetensors.
    m_latest = _run_and_measure("prio_latest", ["infer.use_ema=false"])
    assert m_latest == pytest.approx(exp_means["latest"], abs=0.02)

    # 4. Explicit weights_path overrides auto-selection.
    m_override = _run_and_measure(
        "prio_override",
        [f"infer.weights_path={hydra_value_formatter(str(exp_info['paths']['override']))}"],
    )
    assert m_override == pytest.approx(exp_means["override"], abs=0.02)

    # Confirm all 4 observed pixel means are decisively distinct.
    distinct = [m_best, m_ema, m_latest, m_override]
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            assert abs(distinct[i] - distinct[j]) > 0.1


def test_t1_tools_infer_mode2_standalone_inputs_recursive_json_and_jpg(
    e2e_workdir: dict[str, Path],
    synthetic_image_factory: Callable[..., Path],
    constant_bias_state_dict_factory: Callable[..., tuple[dict[str, Any], float]],
    standalone_infer_arch_overrides_factory: Callable[..., list[str]],
    infer_cli_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F37, F38: tools/infer.py Mode 2 standalone ToyModel .pth (params_ema) on single/recursive/JSON."""
    import torch
    from PIL import Image
    from tests.dummy_arch import DummySRNet

    from utils.image import load_image

    # Build BasicSR-style .pth with distinct non-zero params_ema vs zeroed params.
    net = DummySRNet(num_in_ch=3, num_out_ch=3, num_feat=8, num_block=1, scale=2)
    sd_ema, expected_mean = constant_bias_state_dict_factory(net, 0.65)
    assert expected_mean > 0.4, f"Expected non-trivial bias-tagged output mean, got {expected_mean}"
    sd_zero = {
        k: torch.zeros_like(v) if torch.is_floating_point(v) else v.clone()
        for k, v in net.state_dict().items()
    }
    pth_path = e2e_workdir["work"] / "toy_basicsr.pth"
    torch.save({"params_ema": sd_ema, "params": sd_zero}, pth_path)

    arch_overrides = standalone_infer_arch_overrides_factory(
        {
            "_target_": "tests.dummy_arch.DummySRNet",
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 8,
            "num_block": 1,
            "scale": 2,
        },
        scale=2,
    )
    common = [
        *arch_overrides,
        "infer.exp_dir=null",
        f"infer.weights_path={hydra_value_formatter(str(pth_path))}",
        "++infer.weights_key=params_ema",
        "infer.mixed_precision='no'",
        "infer.tile.enabled=true",
        "infer.tile.size=16",
        "infer.tile.overlap=4",
    ]

    # (a) Single image with suffix="_x2" and save_format="jpg".
    single_in = synthetic_image_factory(
        e2e_workdir["data"] / "single.png", width=20, height=16, pattern="solid", color=(0, 0, 0)
    )
    out_single_dir = e2e_workdir["outputs"] / "mode2_single"
    infer_cli_runner(
        [
            *common,
            f"infer.input={hydra_value_formatter(str(single_in))}",
            f"infer.output_dir={hydra_value_formatter(str(out_single_dir))}",
            "infer.suffix=_x2",
            "infer.save_format=jpg",
        ],
        check=True,
    )
    jpg_out = out_single_dir / "single_x2.jpg"
    assert jpg_out.is_file()
    with Image.open(jpg_out) as im:
        assert im.format == "JPEG"
        assert im.size == (40, 32)
    # Confirms params_ema was loaded rather than zeroed params.
    assert load_image(jpg_out).mean().item() == pytest.approx(expected_mean, abs=0.05)

    # (b) Recursive directory preserving nested subdirs with identical basenames.
    rec_root = e2e_workdir["data"] / "rec_dir"
    synthetic_image_factory(rec_root / "sub1" / "a.png", width=16, height=16, seed=1)
    synthetic_image_factory(rec_root / "sub2" / "a.png", width=20, height=18, seed=2)
    out_rec_dir = e2e_workdir["outputs"] / "mode2_recursive"
    infer_cli_runner(
        [
            *common,
            f"infer.input={hydra_value_formatter(str(rec_root))}",
            f"infer.output_dir={hydra_value_formatter(str(out_rec_dir))}",
            "infer.recursive=true",
            "infer.save_format=png",
        ],
        check=True,
    )
    out_sub1 = out_rec_dir / "sub1" / "a.png"
    out_sub2 = out_rec_dir / "sub2" / "a.png"
    assert out_sub1.is_file() and out_sub2.is_file()
    assert load_image(out_sub1).shape == (3, 32, 32)
    assert load_image(out_sub2).shape == (3, 36, 40)

    # (c) JSON manifest mixing `lq` and `gt`-only fallback with input_root.
    json_root = e2e_workdir["data"] / "json_infer"
    synthetic_image_factory(json_root / "lr" / "img_lq.png", width=18, height=16, seed=3)
    synthetic_image_factory(json_root / "hr" / "img_gt_fallback.png", width=22, height=18, seed=4)
    manifest = json_root / "infer_list.json"
    manifest.write_text(
        json.dumps(
            [
                {"lq": "lr/img_lq.png", "gt": "hr/unused.png"},
                {"gt": "hr/img_gt_fallback.png"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    out_json_dir = e2e_workdir["outputs"] / "mode2_json"
    infer_cli_runner(
        [
            *common,
            f"infer.input={hydra_value_formatter(str(manifest))}",
            f"infer.input_root={hydra_value_formatter(str(json_root))}",
            f"infer.output_dir={hydra_value_formatter(str(out_json_dir))}",
            "infer.save_format=png",
        ],
        check=True,
    )
    json_outputs = sorted(out_json_dir.rglob("*.png"))
    assert len(json_outputs) == 2
    by_name = {p.name: load_image(p).shape for p in json_outputs}
    assert by_name["img_lq.png"] == (3, 32, 36)
    assert by_name["img_gt_fallback.png"] == (3, 36, 44)


def test_t1_tools_infer_2proc_cpu_distributed_odd_image_count(
    repo_root: Path,
    e2e_workdir: dict[str, Path],
    bias_tagged_exp_dir_factory: Callable[..., dict[str, Any]],
    synthetic_image_factory: Callable[..., Path],
    torchrun_cpu_runner: Callable[..., Any],
    hydra_value_formatter: Callable[[Any], str],
) -> None:
    """F39, F40: 2-process CPU torchrun tools/infer.py on 5 images writes 5 unique PNGs & summary."""
    from utils.image import load_image

    exp_info = bias_tagged_exp_dir_factory(
        e2e_workdir["work"] / "exp_dist_infer",
        scale=2,
        gt_size=32,
        include_best=True,
        include_ema=True,
    )

    in_dir = e2e_workdir["data"] / "odd5"
    for idx in range(5):
        synthetic_image_factory(in_dir / f"img_{idx}.png", width=24, height=20, seed=idx + 10)

    out_dir = e2e_workdir["outputs"] / "dist_odd5"
    proc = torchrun_cpu_runner(
        repo_root / "tools" / "infer.py",
        [
            f"infer.exp_dir={hydra_value_formatter(str(exp_info['exp_dir']))}",
            f"infer.input={hydra_value_formatter(str(in_dir))}",
            f"infer.output_dir={hydra_value_formatter(str(out_dir))}",
            "infer.mixed_precision='no'",
            "infer.tile.enabled=true",
            "infer.tile.size=auto",
            "infer.tile.overlap=4",
        ],
        nproc_per_node=2,
        check=True,
    )
    assert proc.returncode == 0

    out_pngs = sorted(out_dir.glob("*.png"))
    assert [p.name for p in out_pngs] == [f"img_{i}.png" for i in range(5)]
    for png in out_pngs:
        assert load_image(png).shape == (3, 40, 48)

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    assert "5" in combined_output and "img/s" in combined_output
