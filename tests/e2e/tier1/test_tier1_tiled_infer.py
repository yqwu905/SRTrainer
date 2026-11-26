"""Tier 1 E2E tests — Area F: Shared Tiled Inference & Single-Device CLI.

Covers features F35-F40 and User Requirement #4 (NO SRPredictor).
Imports ``from engine.inferencer import TiledInferencer`` inside test
functions so Area F executes cleanly on M4 before M5 populates ``engine/__init__.py``.
All non-stdlib imports (besides ``pytest``) are deferred inside test functions.
"""

from __future__ import annotations

import ast
import inspect
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
            "size": None,
            "overlap": 4,
            "batch_size": 4,
            "blend_mode": "cosine",
            "pad_multiple": 8,
        },
        {
            "enabled": True,
            "size": [16, 16],
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


def test_t1_directory_cli_writes_jpeg(
    tmp_path: Path, synthetic_image_factory: Callable[..., Path], infer_cli_runner: Callable[..., Any],
) -> None:
    from PIL import Image

    inputs = tmp_path / "images"
    synthetic_image_factory(inputs / "sample.PNG", width=23, height=19)
    synthetic_image_factory(inputs / "other.jpg", width=17, height=21)
    synthetic_image_factory(inputs / "subdir" / "ignored.png", width=8, height=8)
    output = tmp_path / "out"
    proc = infer_cli_runner([
        "++model._target_=tests.dummy_arch.DummySRNet",
        "++model.scale=2", "scale=2", "infer.device=cpu", "infer.save_format=jpg",
        "infer.suffix=_sr", "infer.tile.size=[16,16]", "infer.tile.overlap=4",
        f"infer.input={inputs}", f"infer.output_dir={output}",
    ], check=True)
    assert "2 images" in proc.stdout
    assert sorted(p.name for p in output.glob("*.jpg")) == ["other_sr.jpg", "sample_sr.jpg"]
    for name, size in (("other_sr.jpg", (34, 42)), ("sample_sr.jpg", (46, 38))):
        with Image.open(output / name) as image:
            assert image.format == "JPEG" and image.size == size
