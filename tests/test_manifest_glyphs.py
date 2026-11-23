"""Regression tests for manifest external glyph alignment, key indexing, and tests package isolation."""

from __future__ import annotations

import copy
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from data.manifest import (
    _load_external_glyphs_map,
    _normalize_glyph_key,
    _normalize_text_lines,
    load_train_manifest,
    validate_sources_config,
)
from data.mixed_dataset import MixedSRDataset, train_collate_fn
from losses import GlyphLoss
from tests.e2e.conftest import REPO_ROOT, build_subprocess_env
from tests.helpers.fake_backends import patch_textverifier_loader


def _glyph(ch: str, bbox: list[int] | None = None) -> dict[str, Any]:
    return {"char": ch, "bbox": bbox if bbox is not None else [4, 4, 24, 24]}


def _line(
    ch: str = "中",
    *,
    bbox: list[int] | None = None,
    alignment_valid: Any = inspect._empty,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "transcript": ch,
        "bbox": bbox if bbox is not None else [2, 2, 28, 28],
        "glyphs": [_glyph(ch, bbox=[4, 4, 24, 24])],
    }
    if alignment_valid is not inspect._empty:
        item["alignment_valid"] = alignment_valid
    if extra:
        item.update(extra)
    return item


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_rgb_png(path: Path, size: tuple[int, int] = (32, 32), seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    arr = rng.integers(30, 225, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _make_stub_glyph_loss(monkeypatch: pytest.MonkeyPatch) -> GlyphLoss:
    torch.manual_seed(123)
    encoder = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1, bias=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    prototypes = torch.randn(3, 2, 4)
    patch_textverifier_loader(
        monkeypatch,
        encoder=encoder,
        prototypes=prototypes,
        char_to_index={"中": 0, "文": 1, "春": 2},
    )
    return GlyphLoss(
        batch_size=4,
        input_range="0_1",
        sync_ddp_counts=False,
    )


# ---------------------------------------------------------------------------
# 1. Image-level & line-level alignment_valid truth table and strict typing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("img_aligned", "line_aligned", "expected"),
    [
        (None, None, True),
        (None, True, True),
        (None, False, False),
        (True, None, True),
        (True, True, True),
        (True, False, False),
        (False, None, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_external_wrapper_alignment_valid_truth_table(
    tmp_path: Path,
    img_aligned: bool | None,
    line_aligned: bool | None,
    expected: bool,
) -> None:
    wrapper: dict[str, Any] = {
        "lines": [
            _line(
                "中",
                alignment_valid=inspect._empty if line_aligned is None else line_aligned,
            )
        ]
    }
    if img_aligned is not None:
        wrapper["alignment_valid"] = img_aligned

    glyphs_path = _write_json(tmp_path / "glyphs.json", {"hr/page.png": wrapper})
    manifest_path = _write_json(tmp_path / "train.json", [{"gt": "hr/page.png"}])

    samples, _ = load_train_manifest(manifest_path, tmp_path, glyphs_json=glyphs_path)
    assert len(samples) == 1
    assert samples[0].text_lines is not None
    assert len(samples[0].text_lines) == 1
    assert samples[0].text_lines[0]["alignment_valid"] is expected
    assert samples[0].text_lines[0]["valid"] is True


def test_normalize_text_lines_keyword_only_and_non_mutating() -> None:
    sig = inspect.signature(_normalize_text_lines)
    param = sig.parameters["image_alignment_valid"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is True

    raw_lines = [
        _line("中", alignment_valid=True, extra={"difficult": False, "ignore": False}),
        _line("文", extra={"ignore": True}),
        _line("春", extra={"transcript": "###"}),
    ]
    snapshot = copy.deepcopy(raw_lines)

    with pytest.raises(TypeError):
        _normalize_text_lines(raw_lines, "ctx", False)  # type: ignore[misc]

    parsed = _normalize_text_lines(raw_lines, "ctx", image_alignment_valid=False)
    assert raw_lines == snapshot, "_normalize_text_lines must not mutate caller input"
    assert [line["alignment_valid"] for line in parsed] == [False, False, False]
    assert [line["valid"] for line in parsed] == [True, False, False]


@pytest.mark.parametrize("bad_val", [None, 0, 1, "true", "false", [], {}])
def test_alignment_valid_strict_bool_validation_image_and_line_and_no_short_circuit(
    tmp_path: Path,
    bad_val: Any,
) -> None:
    # 1) Invalid image-level alignment_valid in external wrapper
    g_img = _write_json(
        tmp_path / "bad_img.json",
        {"page.png": {"alignment_valid": bad_val, "lines": [_line("中")]}},
    )
    with pytest.raises(ValueError, match="alignment_valid"):
        _load_external_glyphs_map(g_img)

    # 2) Invalid direct image_alignment_valid argument
    with pytest.raises(ValueError, match="alignment_valid"):
        _normalize_text_lines([_line("中")], "ctx", image_alignment_valid=bad_val)

    # 3) Invalid line-level alignment_valid with default/True parent
    g_line = _write_json(
        tmp_path / "bad_line.json",
        {"page.png": {"alignment_valid": True, "lines": [_line("中", alignment_valid=bad_val)]}},
    )
    with pytest.raises(ValueError, match="alignment_valid"):
        _load_external_glyphs_map(g_line)

    # 4) Parent False MUST NOT short-circuit invalid child alignment_valid!
    g_parent_false = _write_json(
        tmp_path / "parent_false_bad_child.json",
        {"page.png": {"alignment_valid": False, "lines": [_line("中", alignment_valid=bad_val)]}},
    )
    with pytest.raises(ValueError, match="alignment_valid"):
        _load_external_glyphs_map(g_parent_false)

    with pytest.raises(ValueError, match="alignment_valid"):
        _normalize_text_lines(
            [_line("中", alignment_valid=bad_val)],
            "ctx",
            image_alignment_valid=False,
        )


def test_legacy_list_bare_stem_inline_priority_and_text_boxes_alias(tmp_path: Path) -> None:
    # Legacy external list format + bare stem (including non-image dot suffix preservation)
    glyphs_path = _write_json(
        tmp_path / "glyphs.json",
        {
            "page01": [_line("中")],
            "page.v2": [_line("文")],
        },
    )
    manifest_path = _write_json(
        tmp_path / "train.json",
        [
            # Inline text_lines beats inline glyphs and external
            {
                "gt": "hr/page01.png",
                "text_lines": [_line("春")],
                "glyphs": [_glyph("文")],
            },
            # Inline text_lines=None falls through to inline glyphs (beating external)
            {
                "gt": "hr/page01.png",
                "text_lines": None,
                "glyphs": [_glyph("文")],
            },
            # Both inline None -> falls through to external bare stem
            {"gt": "hr/page01.png", "text_lines": None, "glyphs": None},
            # Bare stem with non-image suffix ".v2" matches "hr/page.v2.png"
            {"gt": "hr/page.v2.png"},
            # External configured but no match -> ()
            {"gt": "hr/unmatched.png"},
        ],
    )

    samples, _ = load_train_manifest(manifest_path, tmp_path, glyphs_json=glyphs_path)
    assert [g["char"] for g in samples[0].text_lines[0]["glyphs"]] == ["春"]
    assert [g["char"] for g in samples[1].text_lines[0]["glyphs"]] == ["文"]
    assert [g["char"] for g in samples[2].text_lines[0]["glyphs"]] == ["中"]
    assert samples[2].text_lines[0]["alignment_valid"] is True
    assert [g["char"] for g in samples[3].text_lines[0]["glyphs"]] == ["文"]
    assert samples[4].text_lines == ()

    # No external config & no inline -> text_lines is None
    plain_manifest = _write_json(tmp_path / "plain.json", [{"gt": "hr/page01.png"}])
    plain_samples, _ = load_train_manifest(plain_manifest, tmp_path, glyphs_json=None)
    assert plain_samples[0].text_lines is None

    # validate_sources_config supports text_boxes alias and isolates indices per source
    g1 = _write_json(tmp_path / "g1.json", {"dir_a/same.png": [_line("中")]})
    g2 = _write_json(tmp_path / "g2.json", {"dir_b/same.png": [_line("文")]})
    m1 = _write_json(tmp_path / "m1.json", [{"gt": "x/same.png"}])
    m2 = _write_json(tmp_path / "m2.json", [{"gt": "y/same.png"}])
    all_samples, _ = validate_sources_config(
        [
            {
                "name": "src1",
                "root_dir": str(tmp_path),
                "json_path": str(m1),
                "glyphs_json": str(g1),
                "ratio": 0.5,
            },
            {
                "name": "src2",
                "root_dir": str(tmp_path),
                "json_path": str(m2),
                "text_boxes": str(g2),
                "ratio": 0.5,
            },
        ],
        has_degradation=True,
        is_main_process=False,
    )
    assert [g["char"] for g in all_samples[0][0].text_lines[0]["glyphs"]] == ["中"]
    assert [g["char"] for g in all_samples[1][0].text_lines[0]["glyphs"]] == ["文"]


# ---------------------------------------------------------------------------
# 2. Exact key priority, ambiguity errors, normalization, and non-fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reverse_order", [False, True])
def test_exact_page_vs_sub_page_both_orders_and_ambiguous_fallback(
    tmp_path: Path,
    reverse_order: bool,
) -> None:
    items = [
        ("page.png", [_line("中")]),
        ("sub/page.png", [_line("文")]),
    ]
    if reverse_order:
        items.reverse()
    glyphs_path = _write_json(tmp_path / "glyphs.json", dict(items))
    manifest_exact = _write_json(
        tmp_path / "exact.json",
        [{"gt": "page.png"}, {"gt": "sub/page.png"}],
    )

    samples, _ = load_train_manifest(
        manifest_exact, tmp_path, source_name="order_test", glyphs_json=glyphs_path
    )
    assert [g["char"] for g in samples[0].text_lines[0]["glyphs"]] == ["中"]
    assert [g["char"] for g in samples[1].text_lines[0]["glyphs"]] == ["文"]

    # Fallback via ambiguous basename raises ValueError with order-independent sorted candidates
    manifest_ambig = _write_json(tmp_path / "ambig.json", [{"gt": "other/page.png"}])
    with pytest.raises(ValueError) as exc_info:
        load_train_manifest(
            manifest_ambig, tmp_path, source_name="order_test", glyphs_json=glyphs_path
        )
    msg = str(exc_info.value)
    assert str(glyphs_path) in msg
    assert str(manifest_ambig) in msg
    assert "other/page.png" in msg
    assert "basename" in msg
    assert "['page.png', 'sub/page.png']" in msg


def test_same_basename_different_dirs_coexist_exact_but_error_on_ambiguous_fallback(
    tmp_path: Path,
) -> None:
    glyphs_path = _write_json(
        tmp_path / "glyphs.json",
        {
            "dir_b/page.png": [_line("文")],
            "dir_a/page.png": [_line("中")],
        },
    )
    # Exact paths coexist without error
    m_exact = _write_json(
        tmp_path / "m_exact.json",
        [{"gt": "dir_a/page.png"}, {"gt": "dir_b/page.png"}],
    )
    samples, _ = load_train_manifest(m_exact, tmp_path, glyphs_json=glyphs_path)
    assert [g["char"] for g in samples[0].text_lines[0]["glyphs"]] == ["中"]
    assert [g["char"] for g in samples[1].text_lines[0]["glyphs"]] == ["文"]

    # Ambiguous fallback via different dir raises ValueError with sorted candidates
    m_fallback = _write_json(tmp_path / "m_fallback.json", [{"gt": "dir_c/page.png"}])
    with pytest.raises(ValueError) as exc_info:
        load_train_manifest(m_fallback, tmp_path, source_name="dup_src", glyphs_json=glyphs_path)
    msg = str(exc_info.value)
    assert str(glyphs_path) in msg
    assert "dir_c/page.png" in msg
    assert "basename" in msg
    assert "['dir_a/page.png', 'dir_b/page.png']" in msg


def test_same_stem_different_extensions_and_basename_stem_cross(tmp_path: Path) -> None:
    # Same stem, different extensions
    glyphs_ext = _write_json(
        tmp_path / "glyphs_ext.json",
        {
            "a/page.png": [_line("中")],
            "b/page.jpg": [_line("文")],
        },
    )
    m_ext = _write_json(
        tmp_path / "m_ext.json",
        [{"gt": "x/page.png"}, {"gt": "y/page.jpg"}],
    )
    samples, _ = load_train_manifest(m_ext, tmp_path, glyphs_json=glyphs_ext)
    assert [g["char"] for g in samples[0].text_lines[0]["glyphs"]] == ["中"]
    assert [g["char"] for g in samples[1].text_lines[0]["glyphs"]] == ["文"]

    m_webp = _write_json(tmp_path / "m_webp.json", [{"gt": "z/page.webp"}])
    with pytest.raises(ValueError) as exc_info:
        load_train_manifest(m_webp, tmp_path, glyphs_json=glyphs_ext)
    msg = str(exc_info.value)
    assert "stem" in msg
    assert "['a/page.png', 'b/page.jpg']" in msg

    # Stem and basename cross: bare stem "page" + "sub/page.png"
    glyphs_cross = _write_json(
        tmp_path / "glyphs_cross.json",
        {
            "page": [_line("中")],
            "sub/page.png": [_line("文")],
        },
    )
    m_cross = _write_json(
        tmp_path / "m_cross.json",
        [{"gt": "page"}, {"gt": "other/page.png"}],
    )
    cross_samples, _ = load_train_manifest(m_cross, tmp_path, glyphs_json=glyphs_cross)
    assert [g["char"] for g in cross_samples[0].text_lines[0]["glyphs"]] == ["中"]
    assert [g["char"] for g in cross_samples[1].text_lines[0]["glyphs"]] == ["文"]

    m_cross_stem = _write_json(tmp_path / "m_cross_stem.json", [{"gt": "other/page.jpg"}])
    with pytest.raises(ValueError) as exc_cross:
        load_train_manifest(m_cross_stem, tmp_path, glyphs_json=glyphs_cross)
    assert "stem" in str(exc_cross.value)
    assert "['page', 'sub/page.png']" in str(exc_cross.value)

    # Ambiguous basename MUST error immediately without continuing to stem
    glyphs_no_fallthrough = _write_json(
        tmp_path / "glyphs_no_ft.json",
        {
            "a/page.png": [_line("中")],
            "b/page.png": [_line("文")],
            "c/page.jpg": [_line("春")],
        },
    )
    m_ft = _write_json(tmp_path / "m_ft.json", [{"gt": "d/page.png"}])
    with pytest.raises(ValueError) as exc_ft:
        load_train_manifest(m_ft, tmp_path, glyphs_json=glyphs_no_fallthrough)
    ft_msg = str(exc_ft.value)
    assert "basename" in ft_msg
    assert "['a/page.png', 'b/page.png']" in ft_msg
    assert "c/page.jpg" not in ft_msg


def test_pure_string_path_normalization_and_duplicate_full_key_error(tmp_path: Path) -> None:
    assert _normalize_glyph_key(r".\sub\\dir\.\my page.PNG") == "sub/dir/my page.PNG"
    assert _normalize_glyph_key("a/../b//./Page 1.png") == "a/../b/Page 1.png"
    assert _normalize_glyph_key("///abs//./path.png") == "/abs/path.png"

    # Case, spaces, and ".." preserved without filesystem access
    glyphs_norm = _write_json(
        tmp_path / "glyphs_norm.json",
        {
            r".\Sub Dir\\My Page.png": [_line("中")],
            "sub dir/my page.png": [_line("文")],
            "a/../b/page.png": [_line("春")],
            "b/page.png": [_line("中")],
        },
    )
    m_norm = _write_json(
        tmp_path / "m_norm.json",
        [
            {"gt": "Sub Dir/./My Page.png"},
            {"gt": r"sub dir\my page.png"},
            {"gt": "a/../b//./page.png"},
            {"gt": "./b/page.png"},
        ],
    )
    samples, _ = load_train_manifest(m_norm, tmp_path, glyphs_json=glyphs_norm)
    assert [g["char"] for g in samples[0].text_lines[0]["glyphs"]] == ["中"]
    assert [g["char"] for g in samples[1].text_lines[0]["glyphs"]] == ["文"]
    assert [g["char"] for g in samples[2].text_lines[0]["glyphs"]] == ["春"]
    assert [g["char"] for g in samples[3].text_lines[0]["glyphs"]] == ["中"]

    # Duplicate full key after normalization raises ValueError
    glyphs_dup = _write_json(
        tmp_path / "glyphs_dup.json",
        {
            r"sub\page.png": [_line("中")],
            "./sub//./page.png": [_line("文")],
        },
    )
    with pytest.raises(ValueError, match="Duplicate normalized glyph key"):
        _load_external_glyphs_map(glyphs_dup)


def test_empty_exact_and_inline_empty_do_not_fallback(tmp_path: Path) -> None:
    glyphs_path = _write_json(
        tmp_path / "glyphs.json",
        {
            "sub/page.png": {"alignment_valid": True, "lines": []},
            "page": [_line("中")],
            "solo": [_line("中")],
        },
    )
    manifest_path = _write_json(
        tmp_path / "train.json",
        [
            # Exact hit with empty () must NOT fall back to bare stem "page"
            {"gt": "sub/page.png"},
            # Unique basename hit with empty () must NOT fall back to bare stem "page"
            {"gt": "other/page.png"},
            # Inline empty text_lines=[] must NOT fall back to inline glyphs or external
            {"gt": "other/page.jpg", "text_lines": [], "glyphs": [_glyph("文")]},
            # Inline empty glyphs=[] must NOT fall back to external
            {"gt": "other/page.jpg", "text_lines": None, "glyphs": []},
            # Control: unique stem fallback works when neither exact nor basename exists
            {"gt": "other/solo.jpg"},
        ],
    )
    samples, _ = load_train_manifest(manifest_path, tmp_path, glyphs_json=glyphs_path)
    assert samples[0].text_lines == ()
    assert samples[1].text_lines == ()
    assert samples[2].text_lines == ()
    assert samples[3].text_lines == ({"valid": True, "alignment_valid": True, "glyphs": []},)
    assert [g["char"] for g in samples[4].text_lines[0]["glyphs"]] == ["中"]


# ---------------------------------------------------------------------------
# 3. Full chain: manifest -> MixedSRDataset -> train_collate_fn -> GlyphLoss
# ---------------------------------------------------------------------------


def test_full_chain_alignment_invalid_zero_grad_and_mixed_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_rgb_png(tmp_path / "hr/bad.png", (32, 32), seed=1)
    _write_rgb_png(tmp_path / "lr/bad.png", (8, 8), seed=2)
    _write_rgb_png(tmp_path / "hr/good.png", (32, 32), seed=3)
    _write_rgb_png(tmp_path / "lr/good.png", (8, 8), seed=4)

    glyphs_path = _write_json(
        tmp_path / "glyphs.json",
        {
            "hr/bad.png": {
                "alignment_valid": False,
                "lines": [_line("中", bbox=[2, 2, 28, 28])],
            },
            "hr/good.png": {
                "alignment_valid": True,
                "lines": [_line("文", bbox=[2, 2, 28, 28])],
            },
        },
    )
    manifest_path = _write_json(
        tmp_path / "train.json",
        [
            {"gt": "hr/bad.png", "lq": "lr/bad.png"},
            {"gt": "hr/good.png", "lq": "lr/good.png"},
        ],
    )

    ds = MixedSRDataset(
        sources=[
            {
                "name": "chain_src",
                "root_dir": str(tmp_path),
                "json_path": str(manifest_path),
                "glyphs_json": str(glyphs_path),
                "ratio": 1.0,
            }
        ],
        gt_size=32,
        scale=4,
        augment={"hflip": False, "vflip": False, "rot90": False},
        has_degradation=False,
        is_main_process=False,
        text_crop_prob=1.0,
    )
    assert ds.has_text_annotations is True

    bad_item = ds[0]
    good_item = ds[1]
    assert bad_item["text_lines"] == []
    assert len(good_item["text_lines"]) == 1
    assert [g["char"] for g in good_item["text_lines"][0]["glyphs"]] == ["文"]

    loss_fn = _make_stub_glyph_loss(monkeypatch)

    # Case A: Invalid-only batch -> scored=0, zero loss, backward succeeds with strictly zero grad
    bad_batch = train_collate_fn([bad_item])
    pred_bad = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss_bad = loss_fn(pred_bad, bad_batch["gt"], batch=bad_batch)
    assert loss_fn.last_counts == (0, 0)
    assert loss_bad.item() == 0.0
    loss_bad.backward()
    assert pred_bad.grad is not None
    assert torch.count_nonzero(pred_bad.grad) == 0

    # Case B: Valid-only control -> scored=1, finite non-zero grad
    good_batch = train_collate_fn([good_item])
    pred_good = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss_good = loss_fn(pred_good, good_batch["gt"], batch=good_batch)
    assert loss_fn.last_counts == (1, 0)
    assert torch.isfinite(loss_good)
    loss_good.backward()
    assert pred_good.grad is not None and torch.isfinite(pred_good.grad).all()
    assert pred_good.grad.abs().sum().item() > 0.0

    # Case C: Mixed batch [bad, good] -> invalid image strictly zero grad, valid image non-zero grad
    mixed_batch = train_collate_fn([bad_item, good_item])
    pred_mixed = torch.rand(2, 3, 32, 32, requires_grad=True)
    loss_mixed = loss_fn(pred_mixed, mixed_batch["gt"], batch=mixed_batch)
    assert loss_fn.last_counts == (1, 0)
    assert torch.isfinite(loss_mixed)
    loss_mixed.backward()
    assert pred_mixed.grad is not None and torch.isfinite(pred_mixed.grad).all()
    assert torch.count_nonzero(pred_mixed.grad[0]) == 0
    assert pred_mixed.grad[1].abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# 4. Test infra: third-party site-packages/tests shadowing + external cwd
# ---------------------------------------------------------------------------


def test_repo_tests_package_beats_third_party_site_packages_tests_from_external_cwd(
    tmp_path: Path,
) -> None:
    fake_site = tmp_path / "fake_site_packages"
    fake_tests_init = fake_site / "tests" / "__init__.py"
    fake_tests_init.parent.mkdir(parents=True, exist_ok=True)
    fake_tests_init.write_text("THIRD_PARTY_SHADOW = True\n", encoding="utf-8")

    external_cwd = tmp_path / "external_cwd"
    external_cwd.mkdir(parents=True, exist_ok=True)

    # Confirm repo tests/__init__.py exists as a regular package marker
    assert (REPO_ROOT / "tests" / "__init__.py").is_file()

    # Subprocess with fake_site_packages on PYTHONPATH (after REPO_ROOT via build_subprocess_env)
    # and external_cwd as CWD must resolve tests.* to this repository, NOT fake_site_packages.
    env = build_subprocess_env()
    env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], str(fake_site)])

    probe_code = """
import json
from pathlib import Path
import hydra.utils
import tests
import tests.e2e.custom_metrics as cm
import tests.test_train_smoke as ts

cls_cm = hydra.utils.get_class("tests.e2e.custom_metrics.MockOCRAMetric")
cls_ts = hydra.utils.get_class("tests.test_train_smoke.MetaScoreMetric")
print(json.dumps({
    "tests_file": str(Path(tests.__file__).resolve()),
    "cm_file": str(Path(cm.__file__).resolve()),
    "ts_file": str(Path(ts.__file__).resolve()),
    "has_shadow_attr": hasattr(tests, "THIRD_PARTY_SHADOW"),
    "cm_cls_ok": cls_cm is cm.MockOCRAMetric,
    "ts_cls_ok": cls_ts is ts.MetaScoreMetric,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=str(external_cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    info = json.loads(proc.stdout.strip())
    assert info["has_shadow_attr"] is False
    assert info["tests_file"] == str((REPO_ROOT / "tests" / "__init__.py").resolve())
    assert info["cm_file"] == str((REPO_ROOT / "tests" / "e2e" / "custom_metrics.py").resolve())
    assert info["ts_file"] == str((REPO_ROOT / "tests" / "test_train_smoke.py").resolve())
    assert info["cm_cls_ok"] is True
    assert info["ts_cls_ok"] is True

    # Negative control: prove that WITHOUT tests/__init__.py (PEP 420 namespace package),
    # fake_site_packages/tests/__init__.py DOES shadow a repo placed earlier on PYTHONPATH!
    fake_ns_repo = tmp_path / "ns_repo"
    (fake_ns_repo / "tests" / "e2e").mkdir(parents=True, exist_ok=True)
    (fake_ns_repo / "tests" / "e2e" / "custom_metrics.py").write_text("X = 1\n", encoding="utf-8")
    ns_env = os.environ.copy()
    ns_env["PYTHONPATH"] = os.pathsep.join([str(fake_ns_repo), str(fake_site)])
    ns_proc = subprocess.run(
        [sys.executable, "-c", "import tests; assert getattr(tests, 'THIRD_PARTY_SHADOW', False)"],
        cwd=str(external_cwd),
        env=ns_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ns_proc.returncode == 0, (
        "Negative control failed: expected fake site-packages/tests/__init__.py "
        "to shadow namespace tests/ directory."
    )
