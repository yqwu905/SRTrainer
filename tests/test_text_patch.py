"""Paired text-patch configuration, geometry, and training integration."""

from __future__ import annotations

import json
import pickle
import random
from itertools import combinations
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from config import (
    ConfigValidationError,
    DataSourceConfig,
    TextPatchConfig,
    load_text_patch_config,
    load_train_config,
)
from data import text_patch as patch_module
from data.degradation.realesrgan import RealESRGANDegrader
from data.mixed_dataset import MixedSRDataset, train_collate_fn
from data.text_patch import (
    TextPatchOverlay,
    TextPatchSample,
    load_text_patch_manifest,
    polygons_overlap,
)
from tests.test_structured_configs import _minimal_train_dict
from utils.image import load_image
from utils.text_geometry import canonical_quad, rectify_text_roi


def _json(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def _patch(tmp_path, scale=1, *, label="Ab", rgba=False, hw=(8, 16), **overrides):
    # Asymmetric two-axis gradient makes flips, misalignment and corner ordering observable.
    h, w = hw
    y, x = np.mgrid[:h, :w]
    rgb = np.stack((40 + x * 170 // w, 35 + y * 180 // h, np.full_like(x, 200)), axis=-1)
    alpha = np.choose(x * 3 // w, [0, 128, 255])
    gt, lq = rgb, 255 - rgb
    if rgba:
        gt, lq = (np.dstack((p, alpha)) for p in (gt, lq))
    for name, pixels, factor in (("pg.png", gt, scale), ("pl.png", lq, 1)):
        Image.fromarray(pixels.repeat(factor, axis=0).repeat(factor, axis=1).astype("uint8")).save(
            tmp_path / name
        )
    manifest = _json(tmp_path / "patch.json", [{"gt": "pg.png", "lq": "pl.png", "label": label}])
    return TextPatchConfig(json_path=manifest, root_dir=str(tmp_path), **overrides)


def _sources(tmp_path, scale, samples, *, hw=(64, 64), **extra):
    h, w = hw
    for name, shape, color in (("gt.png", (w, h), 120), ("lq.png", (w // scale, h // scale), 33)):
        Image.new("RGB", shape, (color,) * 3).save(tmp_path / name)
    return [
        dict(
            name="src",
            root_dir=str(tmp_path),
            json_path=_json(tmp_path / "src.json", samples),
            ratio=1.0,
            **extra,
        )
    ]


def _coverage(quad, height, width):
    # Independent inverse basis solve at destination pixel centers.
    q = np.asarray(quad)
    ys, xs = np.mgrid[:height, :width] + 0.5
    delta = np.stack([xs, ys]).reshape(2, -1) - q[0, :, None]
    uv = np.linalg.solve(np.array([q[1] - q[0], q[3] - q[0]]).T, delta)
    return torch.from_numpy(((uv >= 0) & (uv < 1)).all(axis=0).reshape(height, width))


def test_config_composition(tmp_path):
    defaults = DataSourceConfig().text_patch
    assert defaults == TextPatchConfig(None, None, [1, 3], [1.0, 1.0], [0.0, 0.0], 1.0, 50)
    assert not defaults.is_configured()
    assert load_text_patch_config(None) == defaults

    raw = _minimal_train_dict(tmp_path)
    data = raw.pop("data")
    (src,) = data["sources"]
    data["sources"].extend([
        {**src, "name": "a", "text_patch": {"json_path": "a.json", "root_dir": "a"}},
        {**src, "name": "b", "text_patch": {
            "json_path": "b.json", "root_dir": "b", "num_patches": [0, 4], "scale_range": [0.5, 1.5],
        }},
    ])
    (tmp_path / "data").mkdir()
    OmegaConf.save({"defaults": ["base_data", "_self_"], **data}, tmp_path / "data" / "patch.yaml")
    OmegaConf.save({"defaults": ["data/patch", "_self_"], **raw}, tmp_path / "train.yaml")
    with hydra.initialize_config_dir(config_dir=str(tmp_path), version_base=None):
        with pytest.raises(ConfigCompositionException, match=r"data\.text_patch"):
            hydra.compose(config_name="train", overrides=["data.text_patch.json_path=p.json"])
        composed = hydra.compose(
            config_name="train", overrides=["data.sources.2.text_patch.rotate_range=[-20,30]"]
        )
    plain, a, b = (src.text_patch for src in load_train_config(composed).data.sources)
    assert plain == defaults
    assert a == TextPatchConfig("a.json", "a")
    assert b == TextPatchConfig("b.json", "b", [0, 4], [0.5, 1.5], [-20, 30])
    assert load_text_patch_config({"prob": 0, "num_patches": [0, 0], "max_attempts": 1}).prob == 0


def test_load_train_config_rejects_global_text_patch(tmp_path):
    raw = _minimal_train_dict(tmp_path)
    raw["data"]["text_patch"] = {"json_path": "patch.json", "root_dir": "patch"}
    with pytest.raises(ConfigValidationError, match=r"data\.text_patch"):
        load_train_config(raw)


@pytest.mark.parametrize("patch", [None, {"unknown": 0}, {"prob": 1.1}, {"root_dir": "p"}])
def test_invalid_patch_identifies_source(tmp_path, patch):
    raw = _minimal_train_dict(tmp_path)
    sources = raw["data"]["sources"]
    sources.append({**sources[0], "text_patch": patch})
    with pytest.raises(ConfigValidationError, match=r"data\.sources\[1\]\.text_patch"):
        load_train_config(raw)


def test_manifest_validates_patch_before_reading_source_files(tmp_path):
    from data.manifest import validate_sources_config

    sources = _minimal_train_dict(tmp_path)["data"]["sources"]
    sources.append({**sources[0], "text_patch": {"max_attempts": 0}})
    with pytest.raises(ConfigValidationError, match=r"data\.sources\[1\]\.text_patch\.max_attempts"):
        validate_sources_config(sources, has_degradation=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("json_path", "patch.json"),
        ("root_dir", "."),
        ("json_path", " "),
        ("root_dir", ""),
        ("typo", 1),
        ("num_patches", []),
        ("num_patches", [1.5, 2]),
        ("num_patches", [True, 2]),
        ("num_patches", [-1, 2]),
        ("num_patches", [3, 2]),
        ("scale_range", [0, 1]),
        ("scale_range", [float("inf"), float("inf")]),
        ("scale_range", [2, 1]),
        ("rotate_range", [1]),
        ("rotate_range", [2, 1]),
        ("rotate_range", [float("nan"), 1]),
        ("rotate_range", [False, 1]),
        ("max_attempts", 0),
        ("max_attempts", True),
        ("prob", float("nan")),
        ("prob", -0.1),
        ("prob", 1.1),
        ("prob", True),
    ],
)
def test_invalid_config(key, value):
    with pytest.raises(ConfigValidationError, match=key):
        load_text_patch_config({key: value})


@pytest.mark.parametrize("key", ["label", "text", "transcript", "transcription"])
def test_manifest_aliases(tmp_path, key):
    manifest = _json(tmp_path / "m.json", [{"lq": "l/p.png", "gt": "g/p.png", key: " 你好 "}])
    assert load_text_patch_manifest(manifest, tmp_path) == [
        TextPatchSample(tmp_path / "l/p.png", tmp_path / "g/p.png", " 你好 ")
    ]


@pytest.mark.parametrize(
    "entry",
    [
        {},
        [],
        [1],
        [{"gt": "g", "label": "a"}],
        [{"gt": "", "lq": "l", "text": "a"}],
        [{"gt": "g", "lq": 3, "text": "a"}],
        [{"gt": "/g", "lq": "l", "text": "a"}],
        [{"gt": "g", "lq": "l"}],
        [{"gt": "g", "lq": "l", "label": None}],
        [{"gt": "g", "lq": "l", "label": " ", "text": "ignored"}],
    ],
)
def test_invalid_manifest(tmp_path, entry):
    manifest = _json(tmp_path / "m.json", entry)
    with pytest.raises(ValueError):
        load_text_patch_manifest(manifest, tmp_path)


def test_patch_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_patch_manifest(tmp_path / "missing", tmp_path)
    cfg = _patch(tmp_path, scale=1)
    overlay = TextPatchOverlay(cfg, scale=4)
    with pytest.raises(ValueError, match="scale 4"):
        overlay.apply(torch.zeros(3, 64, 64), torch.zeros(3, 16, 16), [])
    (tmp_path / "pl.png").unlink()
    with pytest.raises(FileNotFoundError):
        overlay.apply(torch.zeros(3, 64, 64), torch.zeros(3, 16, 16), [])


def test_paired_sources_use_their_own_patch_probability(tmp_path):
    cfg = _patch(tmp_path, num_patches=[1, 1])
    sources = _sources(tmp_path, 1, [{"gt": "gt.png", "lq": "lq.png"}])
    plain = sources[0]
    sources.extend([
        {**plain, "name": "always", "text_patch": cfg},
        {**plain, "name": "never", "text_patch": {**vars(cfg), "prob": 0.0}},
    ])
    ds = MixedSRDataset(sources, 64, has_degradation=False)
    ordinary, patched, skipped = [ds[i] for i in range(len(ds))]
    for item in (ordinary, patched, skipped):
        assert {"text_patch_mask", "text_patch_bg_gt"}.isdisjoint(item)
    assert ordinary["text_lines"] == skipped["text_lines"] == []
    assert [line["transcript"] for line in patched["text_lines"]] == ["Ab"]
    for key in ("gt", "lq"):
        assert torch.equal(ordinary[key], skipped[key])
        assert not torch.equal(patched[key], ordinary[key])


@pytest.mark.parametrize("scale", [1, 4])
@pytest.mark.parametrize("rgba", [False, True])
def test_aligned_crop_and_dihedral_preserve_patch_pixels(tmp_path, monkeypatch, scale, rgba):
    cfg = _patch(tmp_path, scale, rgba=rgba, num_patches=[1, 1])
    sources = _sources(tmp_path, scale, [{"gt": "gt.png", "lq": "lq.png"}], hw=(80, 100), text_patch=cfg)
    ds = MixedSRDataset(
        sources,
        64,
        scale=scale,
        has_degradation=False,
        augment={"hflip": True, "vflip": True, "rot90": True},
    )
    monkeypatch.setattr(random, "random", lambda: 0.0)  # Force flips on the source.
    monkeypatch.setattr(random, "randint", lambda lo, hi: hi)  # Crop off origin and rotate.
    item = ds[0]
    assert ds.has_text_annotations and not ds.has_source_text_annotations
    assert {"text_patch_mask", "text_patch_bg_gt"}.isdisjoint(item)
    (line,) = item["text_lines"]
    assert line["transcript"] == "Ab" and line["alignment_valid"] and line["valid"]
    q = np.array(line["quad"])
    assert not (q % scale).any()
    for key, name, factor, bg in (("gt", "pg.png", 1, 120), ("lq", "pl.png", scale, 33)):
        x, y = (q[0] / factor).astype(int)
        patch = load_image(tmp_path / name)
        h, w = patch.shape[-2:]
        observed = item[key][:, y : y + h, x : x + w]
        if rgba:
            with Image.open(tmp_path / name) as img:
                alpha = torch.from_numpy(np.array(img.getchannel("A"))).float() / 255.0
            patch = patch * alpha + (1 - alpha) * (bg / 255.0)
        assert torch.equal(observed, patch)


@pytest.mark.parametrize("scale", [1, 4])
@pytest.mark.parametrize("zoom", [0.75, 1.33])
@pytest.mark.parametrize("rgba", [False, True])
def test_warp_rectifies_upright_at_both_resolutions(tmp_path, scale, zoom, rgba):
    cfg = _patch(
        tmp_path,
        scale,
        hw=(20, 36),
        rgba=rgba,
        num_patches=[1, 1],
        scale_range=[zoom, zoom],
        rotate_range=[127, 127],
    )
    gt, lq = torch.zeros(3, 240, 240), torch.zeros(3, 240 // scale, 240 // scale)
    random.seed(471)
    mask = lq.new_zeros(1, *lq.shape[-2:])
    (line,) = TextPatchOverlay(cfg, scale).apply(gt, lq, [], mask)
    assert 0 <= mask.min() <= mask.max() <= 1
    for canvas, name, factor in ((gt, "pg.png", 1), (lq, "pl.png", scale)):
        quad = np.array(line["quad"]) / factor
        inside = _coverage(quad, *canvas.shape[-2:])
        assert not canvas[:, ~inside].any()
        ref = load_image(tmp_path / name)
        if rgba:
            with Image.open(tmp_path / name) as img:
                alpha = torch.from_numpy(np.array(img.getchannel("A"))).float() / 255
            ref *= alpha
        else:
            assert canvas[:, inside].min() >= ref.min() - 1e-6
        readback = rectify_text_roi(canvas[None], quad, tuple(ref.shape[-2:]))[0]
        # Edges and alpha transitions have undergone a downsample and two bilinear filters.
        assert (readback[:, 2:-2, 2:-2] - ref[:, 2:-2, 2:-2]).abs().mean() < 0.025


def test_rotated_non_overlap_and_pixel_coverage(tmp_path):
    random.seed(41)
    cfg = _patch(
        tmp_path,
        scale=4,
        hw=(4, 7),
        num_patches=[7, 7],
        scale_range=[0.55, 1.6],
        rotate_range=[-75, 75],
    )
    gt, lq = torch.zeros(3, 128, 160), torch.zeros(3, 32, 40)
    lines = TextPatchOverlay(cfg, 4).apply(gt, lq, [])
    assert len(lines) == 7
    quads = [np.array(line["quad"]) for line in lines]
    for a, b in combinations(quads, 2):
        assert not polygons_overlap(a, b)
    for canvas, factor in ((gt, 1), (lq, 4)):
        masks = torch.stack([_coverage(q / factor, *canvas.shape[-2:]) for q in quads])
        assert masks.sum(dim=0).max() == 1
        assert torch.equal(canvas.any(dim=0), masks.any(dim=0))


def test_sat_distinguishes_contact_and_intersecting_bounding_boxes():
    a = np.array([[0.0, 2.0], [2.0, 0.0], [4.0, 2.0], [2.0, 4.0]])
    assert not polygons_overlap(a, a + [2, 2])
    assert not polygons_overlap(a, a + [2.1, 2.1])
    assert polygons_overlap(a, a + [1.999, 2])
    assert polygons_overlap(a, a)


@pytest.mark.parametrize(
    ("size", "prob", "count", "expected"),
    [
        (16, 1.0, [5, 5], 1),
        (8, 1.0, [2, 2], 0),
        (16, 0.0, [1, 1], 0),
        (16, 1.0, [0, 0], 0),
    ],
)
def test_exhausted_or_disabled_placement(tmp_path, size, prob, count, expected):
    cfg = _patch(tmp_path, hw=(16, 16), label=" ### ", prob=prob, num_patches=count, max_attempts=2)
    gt = torch.zeros(3, size, size)
    lines = TextPatchOverlay(cfg, 1).apply(gt, gt.clone(), [])
    assert len(lines) == expected
    assert bool(gt.any()) == bool(expected)
    assert all(not line["valid"] for line in lines)


@pytest.mark.parametrize("mode", ["LA", "P"])
def test_png_transparency_modes(tmp_path, mode):
    cfg = _patch(tmp_path, hw=(8, 8), num_patches=[1, 1])
    if mode == "LA":
        image = Image.new("LA", (8, 8), (213, 0))
        image.putpixel((3, 4), (213, 255))
    else:
        image = Image.new("P", (8, 8), 0)
        image.putpalette([0, 0, 0, 213, 213, 213] + [0] * 762)
        image.putpixel((3, 4), 1)
        image.info["transparency"] = 0
    image.save(tmp_path / "pl.png")
    lq = torch.zeros(3, 8, 8)
    TextPatchOverlay(cfg, 1).apply(lq.clone(), lq, [])
    assert lq.count_nonzero() == 3
    torch.testing.assert_close(lq[:, 4, 3], torch.full((3,), 213 / 255))


@pytest.mark.parametrize("external", [False, True])
def test_occluded_annotations_and_text_losses(tmp_path, monkeypatch, external):
    from losses import GlyphLoss, OCRKLLoss
    from tests.helpers.fake_backends import patch_ppocr_loader, patch_textverifier_loader

    cfg = _patch(tmp_path, hw=(16, 24), num_patches=[1, 1])
    lines = [
        {
            "bbox": [0, 0, 45, 16],
            "transcript": "AB",
            "glyphs": [
                {"char": "A", "bbox": [2, 2, 10, 10]},
                {"char": "B", "bbox": [30, 2, 40, 10]},
            ],
        },
        {"bbox": [8, 20, 30, 32], "transcript": "safe"},
        {"bbox": [0, 0, 4, 4]},
    ]
    record = {"gt": "gt.png", "lq": "lq.png"}
    options = {}
    if external:
        options["glyphs_json"] = _json(tmp_path / "lines.json", {"gt.png": lines})
    else:
        record["text_lines"] = lines
    sources = _sources(tmp_path, 1, [record], text_patch=cfg, **options)
    monkeypatch.setattr(random, "randint", lambda lo, hi: lo)
    item = MixedSRDataset(sources, 64, has_degradation=False)[0]
    remaining_glyph, safe_line, added = item["text_lines"]
    assert remaining_glyph == {
        "glyphs": [{"char": "B", "quad": canonical_quad([30, 2, 40, 10]).tolist()}]
    }
    assert safe_line["transcript"] == "safe" and added["transcript"] == "Ab"
    batch = train_collate_fn([item])
    target = batch["gt"]
    pred = torch.rand_like(target).requires_grad_()
    patch_ppocr_loader(monkeypatch)
    patch_textverifier_loader(monkeypatch, embed_dim=16)
    glyph, ocr = GlyphLoss(), OCRKLLoss(amp_dtype=None, use_checkpoint=False)
    for criterion in (ocr, glyph):
        criterion(pred, target, batch=batch).backward()
        assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
        pred.grad = None
    assert glyph.last_counts == (1, 0)
    assert ocr.last_counts[0] == 3


def test_mixed_batch_workers_and_actual_degradation(tmp_path):
    scale = 4
    cfg = _patch(tmp_path, scale, rgba=True, num_patches=[1, 1])
    sources = _sources(tmp_path, scale, [{"gt": "gt.png", "lq": "lq.png"}, {"gt": "gt.png"}], text_patch=cfg)
    ds = MixedSRDataset(
        sources, 64, scale=scale, logger=lambda *args: None, is_main_process=False
    )
    restored = pickle.loads(pickle.dumps(ds))
    (batch,) = DataLoader(restored, batch_size=2, num_workers=2, collate_fn=train_collate_fn)
    assert batch["need_degradation"].tolist() == [False, True]
    pm = batch["text_patch_mask"]
    assert pm.shape == (2, 1, 16, 16) and pm.dtype == torch.float32
    assert not pm[0].any()
    quad = np.array(batch["text_lines"][1][0]["quad"])
    x, y = (quad[0] / scale).astype(int)
    with Image.open(tmp_path / "pl.png") as img:
        alpha = torch.from_numpy(np.array(img.getchannel("A"))).float() / 255
    h, w = alpha.shape
    expected_alpha = torch.zeros_like(pm[1, 0])
    expected_alpha[y : y + h, x : x + w] = alpha
    assert torch.equal(pm[1, 0], expected_alpha)
    color = load_image(tmp_path / "pl.png")
    expected_premul = torch.zeros_like(batch["lq"][1])
    expected_premul[:, y : y + h, x : x + w] = color * alpha
    assert torch.equal(batch["lq"][1], expected_premul)

    assert torch.equal(batch["text_patch_bg_gt"][0], batch["gt"][0])
    _check_clean_background_degradation(batch, load_image(tmp_path / "gt.png"), use_usm=False)

    degrader = RealESRGANDegrader(scale=scale)
    paired = {k: v[:1] for k, v in batch.items()}
    result = degrader(paired)
    assert "text_patch_bg_gt" not in result
    assert torch.equal(result["lq"], paired["lq"])
    for invalid in (1, None, pm[:, 0], pm[:, :, :, :-1]):
        with pytest.raises(ValueError, match="text_patch_mask"):
            degrader({**batch, "text_patch_mask": invalid})
    for invalid in (None, "invalid", paired["gt"][0], paired["gt"][:, :1], paired["gt"][..., :-1]):
        with pytest.raises(ValueError, match="text_patch_bg_gt"):
            degrader({**paired, "text_patch_bg_gt": invalid})


def _check_clean_background_degradation(batch, clean_gt, use_usm):
    mask = batch["need_degradation"]
    clean = clean_gt.expand_as(batch["gt"])
    assert torch.equal(batch["text_patch_bg_gt"][mask], clean[mask])
    patched = batch["text_patch_mask"][mask].flatten(1).any(dim=1)
    assert torch.equal((batch["gt"][mask] != clean[mask]).flatten(1).any(dim=1), patched)
    degrader = RealESRGANDegrader(scale=4, use_usm=use_usm, usm_radius=5)

    def run(b):
        random.seed(23)
        torch.manual_seed(23)
        # Fix the final order to resize + filter + JPEG. Capture JPEG output to
        # compute the reference composite before the degrader's 8-bit rounding.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(random, "random", lambda: 0.25)
            return degrader(b)

    clean_batch = {k: v for k, v in batch.items() if not k.startswith("text_patch_")}
    jpeg_outputs = []
    with degrader.jpeger.register_forward_hook(lambda mod, args, out: jpeg_outputs.append(out)):
        background = run({**clean_batch, "gt": clean})
    raw_background = jpeg_outputs[-1]
    assert torch.equal((raw_background * 255).round() / 255, background["lq"][mask])
    coverage = batch["text_patch_mask"][mask]
    assert (coverage == 0).any() and ((coverage > 0) & (coverage < 1)).any()
    expected_lq = (raw_background * (1 - coverage) + batch["lq"][mask]).mul(255).round() / 255
    actual = run(batch)
    assert "text_patch_bg_gt" not in actual
    assert torch.equal(actual["lq"][mask], expected_lq)
    assert torch.equal(actual["lq"][~mask], batch["lq"][~mask])
    assert torch.equal(actual["gt"][~mask], batch["gt"][~mask])
    expected_gt = batch["gt"][mask]
    if use_usm:
        expected_gt = degrader.usm_sharpener(
            expected_gt, weight=degrader.usm_weight, threshold=degrader.usm_threshold
        )
    assert torch.equal(actual["gt"][mask], expected_gt)
    assert torch.equal(actual["lq"][mask][~patched], background["lq"][mask][~patched])
    return actual


def test_three_sources_in_one_worker_batch(tmp_path, monkeypatch, caplog):
    from losses import GlyphLoss, OCRKLLoss
    from tests.helpers.fake_backends import patch_ppocr_loader, patch_textverifier_loader

    scale = 4
    base_line = {
        "bbox": [2, 2, 18, 14], "transcript": "A",
        "glyphs": [{"bbox": [2, 2, 18, 14], "char": "A"}],
    }
    records = [{"gt": "gt.png", "lq": "lq.png", "text_lines": [base_line]}, {"gt": "gt.png"}]
    sources = _sources(tmp_path, scale, records)
    patch_colors = {"SRC_A": (220, 111, 7), "SRC_B": (13, 41, 159)}
    for label, count, zoom, angle, prob, alpha in (
        ("SRC_A", 1, 1, 0, 1, 128), ("SRC_B", 2, 1.25, 30, 0.95, 255)
    ):
        root = tmp_path / label
        root.mkdir()
        cfg = _patch(
            root, scale, label=label, hw=(4, 6), num_patches=[count, count],
            scale_range=[zoom, zoom], rotate_range=[angle, angle], prob=prob, max_attempts=80,
        )
        Image.new("RGBA", (6, 4), (*patch_colors[label], alpha)).save(root / "pl.png")
        # Exercise both nested Path values in a mapping and a TextPatchConfig instance.
        patch = {**vars(cfg), "json_path": Path(cfg.json_path), "root_dir": root} if count == 1 else cfg
        sources.append({**sources[0], "name": label, "text_patch": patch})

    with caplog.at_level("INFO", logger="srtrainer"):
        ds = MixedSRDataset(sources, 64, scale=scale)
    for label in patch_colors:
        assert f"[Data] {label} text_patch: 1 pairs from" in caplog.text
    assert ds.has_text_patch and ds.has_source_text_annotations and ds.has_text_annotations
    assert ds[(0, 1)]["text_lines"] == []
    loader = DataLoader(
        ds, batch_sampler=[[(0, 0), (0, 1), (1, 1), (2, 1)]], num_workers=2,
        collate_fn=train_collate_fn, generator=torch.Generator().manual_seed(7),
    )
    (batch,) = loader
    assert batch["source_id"].tolist() == [0, 0, 1, 2]
    assert batch["need_degradation"].tolist() == [False, True, True, True]
    assert not batch["text_patch_mask"][:2].any()
    clean_gt = load_image(tmp_path / "gt.png")
    assert torch.equal(batch["gt"][:2], clean_gt.expand_as(batch["gt"][:2]))
    assert torch.equal(batch["lq"][0], load_image(tmp_path / "lq.png"))
    assert not batch["lq"][1].any()
    plain_lines, plain_gt_lines, a_lines, b_lines = batch["text_lines"]
    assert plain_lines == ds[(0, 0)]["text_lines"]
    assert len(plain_lines) == 1 and "transcript" not in plain_lines[0] and not plain_gt_lines
    assert [line["transcript"] for line in a_lines] == ["SRC_A"]
    assert [line["transcript"] for line in b_lines] == ["SRC_B", "SRC_B"]
    assert not polygons_overlap(b_lines[0]["quad"], b_lines[1]["quad"])

    for idx, label, lines, alpha in ((2, "SRC_A", a_lines, 128), (3, "SRC_B", b_lines, 255)):
        coverage = torch.stack([
            _coverage(np.array(line["quad"]) / scale, 16, 16) for line in lines
        ]).any(dim=0)
        expected_alpha = coverage.float() * (alpha / 255)
        color = torch.tensor(patch_colors[label]).view(3, 1, 1) / 255
        torch.testing.assert_close(batch["text_patch_mask"][idx, 0], expected_alpha)
        torch.testing.assert_close(batch["lq"][idx], color * expected_alpha)
    result = _check_clean_background_degradation(batch, clean_gt, use_usm=False)

    patch_ppocr_loader(monkeypatch)
    patch_textverifier_loader(monkeypatch, embed_dim=16)
    for loss in (OCRKLLoss(amp_dtype=None, use_checkpoint=False), GlyphLoss()):
        pred = torch.rand_like(result["gt"]).requires_grad_()
        loss(pred, result["gt"], batch=result).backward()
        assert loss.last_counts[0] > 0
        assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


@pytest.mark.parametrize("use_usm", [False, True])
def test_all_gt_only_degrade_clean_background(tmp_path, use_usm):
    cfg = _patch(tmp_path, scale=4, rgba=True, num_patches=[1, 1])
    sources = _sources(tmp_path, 4, [{"gt": "gt.png"}], text_patch=cfg)
    ds = MixedSRDataset(sources, 64, scale=4)
    batch = train_collate_fn([ds[0], ds[0]])
    assert batch["need_degradation"].tolist() == [True, True]
    _check_clean_background_degradation(batch, load_image(tmp_path / "gt.png"), use_usm)


def test_trainer_passes_patch_config(tmp_path):
    from engine.trainer import Trainer

    raw = _minimal_train_dict(tmp_path)
    raw["data"]["sources"] = _sources(tmp_path, 2, [{"gt": "gt.png", "lq": "lq.png"}])
    raw["data"]["sources"][0]["text_patch"] = {"num_patches": [-1, 2]}
    with pytest.raises(ConfigValidationError, match="num_patches"):
        load_train_config(raw)
    cfg = _patch(tmp_path, scale=2, num_patches=[1, 1])
    raw["data"]["sources"][0]["text_patch"] = cfg
    raw["trainer"]["total_iters"] = 1
    trainer = Trainer(raw)
    assert (
        trainer.train_dataset.has_text_patch and trainer.train_loader.collate_fn is train_collate_fn
    )
    assert next(iter(trainer.train_loader))["text_lines"][0][0]["transcript"] == "Ab"
    assert trainer.fit().global_step == 1


def test_ocr_glyph_hull_does_not_cross_occlusion(tmp_path, monkeypatch):
    cfg = _patch(tmp_path, hw=(12, 12), num_patches=[1, 1])
    quad = canonical_quad([0, 4, 32, 16]).tolist()
    glyphs = [
        {"char": "A", "quad": canonical_quad([1, 5, 6, 15]).tolist()},
        {"char": "B", "quad": canonical_quad([24, 5, 30, 15]).tolist()},
    ]
    # Position the patch between the two letters.
    monkeypatch.setattr(random, "randint", lambda lo, hi: min(hi, 10))
    bg = torch.zeros(3, 32, 32)
    left, right, added = TextPatchOverlay(cfg, 1).apply(
        bg, bg.clone(), [{"quad": quad, "glyphs": glyphs}]
    )
    assert left["glyphs"] == glyphs[:1] and right["glyphs"] == glyphs[1:]
    assert all(not polygons_overlap(g["quad"], added["quad"]) for g in (left, right))


def test_downscale_antialias_and_positive_rotation(tmp_path):
    cfg = _patch(
        tmp_path, hw=(32, 64), num_patches=[1, 1], scale_range=[0.25, 0.25], rotate_range=[90, 90]
    )
    # A one-in-four stripe pattern aliases to black under simple bilinear subsampling.
    stripes = np.zeros((32, 64, 3), dtype=np.uint8)
    stripes[:, ::4] = 255
    Image.fromarray(stripes).save(tmp_path / "pg.png")
    canvas = torch.zeros(3, 16, 8)
    (line,) = TextPatchOverlay(cfg, 1).apply(canvas, canvas.clone(), [])
    assert line["quad"] == [[0, 16], [0, 0], [8, 0], [8, 16]]
    torch.testing.assert_close(canvas[:, 2:-2, 2:-2], torch.full((3, 12, 4), 0.25))


def test_default_dataset_preserves_keys_tensors_annotations_and_rng(tmp_path):
    records = [
        {
            "gt": "gt.png",
            "lq": "lq.png",
            "text_lines": [{"quad": [2, 2, 8, 8], "transcript": "source"}],
        },
        {"gt": "gt.png"},
    ]
    sources = _sources(tmp_path, 1, records)
    sources.append({**sources[0], "name": "second"})
    baseline = MixedSRDataset(sources, 64)
    configured = MixedSRDataset([{**s, "text_patch": TextPatchConfig()} for s in sources], 64)

    def get(ds, idx):
        random.seed(25)
        np.random.seed(25)
        torch.manual_seed(25)
        item = ds[idx]
        return item, random.getstate(), np.random.get_state(), torch.get_rng_state()

    for idx in range(len(baseline)):
        expected, py, np_state, torch_state = get(baseline, idx)
        actual, actual_py, actual_np, actual_torch = get(configured, idx)
        assert actual_py == py
        np.testing.assert_equal(actual_np, np_state)
        assert torch.equal(actual_torch, torch_state)
        assert actual.keys() == expected.keys()
        assert {"text_patch_mask", "text_patch_bg_gt"}.isdisjoint(actual)
        assert actual["text_lines"] == expected["text_lines"]
        if idx == 0:
            assert actual["text_lines"] == [{"quad": canonical_quad([2, 2, 8, 8]).tolist()}]
        assert all(torch.equal(expected[k], v) for k, v in actual.items() if k != "text_lines")


@pytest.mark.parametrize("attempts", [1, 2])
def test_tiny_rotated_patch_requires_coverage_at_both_resolutions(tmp_path, monkeypatch, attempts):
    cfg = _patch(
        tmp_path,
        scale=4,
        hw=(1, 1),
        rotate_range=[45, 45],
        num_patches=[1, 1],
        max_attempts=attempts,
    )
    # The first 4x4 diamond is centered between four LQ pixel centers. The retry
    # moves to an LQ pixel center, so only the second candidate can be committed.
    centers = iter([4, 4, 10, 10])
    monkeypatch.setattr(random, "uniform", lambda lo, hi: lo if lo == hi else next(centers))
    gt, lq, mask = torch.zeros(3, 16, 16), torch.zeros(3, 4, 4), torch.zeros(1, 4, 4)
    lines = TextPatchOverlay(cfg, 4).apply(gt, lq, [], mask)
    assert not gt[:, :7, :7].any()
    assert len(lines) == attempts - 1
    assert bool(lq.any()) == bool(gt.any()) == bool(mask.any()) == bool(lines)
    if lines:
        assert torch.equal(gt.any(0), _coverage(lines[0]["quad"], 16, 16))
        assert torch.equal(lq.any(0), _coverage(np.array(lines[0]["quad"]) / 4, 4, 4))


@pytest.mark.parametrize(("edge", "expected"), [(4.5, 1), (5, 1), (6, 2)])
def test_opposite_touching_edges_respect_pixel_occupancy(tmp_path, monkeypatch, edge, expected):
    cfg = _patch(
        tmp_path, scale=2, hw=(2, 2), num_patches=[2, 2], max_attempts=1, rotate_range=[180, 180]
    )
    left = canonical_quad([[edge, 4], [edge - 4, 4], [edge - 4, 0], [edge, 0]])
    right = canonical_quad([edge, 0, edge + 4, 4])
    assert not polygons_overlap(left, right)
    candidates = iter([left, right])
    monkeypatch.setattr(patch_module, "_place_quad", lambda *args: next(candidates))
    gt, lq = torch.zeros(3, 8, 12), torch.zeros(3, 4, 6)
    lines = TextPatchOverlay(cfg, 2).apply(gt, lq, [])
    assert len(lines) == expected
    for canvas, factor in ((gt, 1), (lq, 2)):
        coverage = torch.stack(
            [_coverage(np.array(line["quad"]) / factor, *canvas.shape[-2:]) for line in lines]
        )
        assert coverage.sum(0).max() == 1
        assert torch.equal(canvas.any(0), coverage.any(0))


def test_rotated_quad_and_render_bounds(monkeypatch):
    # Drive the placement to within one floating-point ULP of each canvas extremity.
    ends = iter([False, True])
    monkeypatch.setattr(
        random,
        "uniform",
        lambda lo, hi: np.nextafter(hi, np.inf) if next(ends) else np.nextafter(lo, -np.inf),
    )
    q = patch_module._place_quad((4, 6), (16, 20), 1, 1, 45)
    assert q[:, 0].min() == 0 and q[:, 1].max() == 16
    patch = torch.ones(4, 4, 4)
    q = canonical_quad([-1e-15, -1e-15, 4 + 1e-15, 4 + 1e-15])
    rendered = patch_module._render_patch((4, 4), patch, q, 1, False)
    canvas = torch.zeros(3, 4, 4)
    rendered.commit(canvas, torch.zeros(4, 4, dtype=torch.bool))
    assert canvas.eq(1).all()
    assert patch_module._render_patch((4, 4), patch, q + 10, 1, False) is None


@pytest.mark.parametrize("transparent", ["gt", "lq", "both"])
def test_transparent_candidate_preserves_canvases_mask_and_lines(tmp_path, transparent):
    cfg = _patch(tmp_path, hw=(8, 8), num_patches=[1, 1], max_attempts=2)
    for key, name in (("gt", "pg.png"), ("lq", "pl.png")):
        if transparent in (key, "both"):
            Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(tmp_path / name)
    lines = [{"quad": canonical_quad([0, 0, 8, 8]).tolist(), "transcript": "original"}]
    gt, lq = torch.rand(3, 8, 8), torch.rand(3, 8, 8)
    gt_before, lq_before, mask = gt.clone(), lq.clone(), torch.zeros(1, 8, 8)
    assert TextPatchOverlay(cfg, 1).apply(gt, lq, lines, mask) == lines
    assert torch.equal(gt, gt_before) and torch.equal(lq, lq_before) and not mask.any()


def test_transparent_margins_preserve_source_lines_glyphs_and_hull(tmp_path):
    cfg = _patch(tmp_path, hw=(10, 10), num_patches=[1, 1])
    img = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    img.putpixel((7, 7), (255, 255, 255, 128))
    img.save(tmp_path / "pg.png")

    def quad(box):
        return canonical_quad(box).tolist()

    # The opaque pixel lies in the slanted line's bounding box, outside its polygon.
    slanted = {"quad": [[0, 5], [9, 9], [9, 10], [0, 6]], "transcript": "visible"}
    a = {"char": "A", "quad": quad([1, 1, 3, 4])}
    b = {"char": "B", "quad": quad([5, 1, 8, 4])}
    c = {"char": "C", "quad": quad([7, 7, 9, 9])}
    partly_hidden = {"quad": quad([0, 0, 10, 10]), "transcript": "ABC", "glyphs": [a, b, c]}
    canvas = torch.zeros(3, 10, 10)
    kept, glyph_line, added = TextPatchOverlay(cfg, 1).apply(
        canvas, canvas.clone(), [slanted, partly_hidden]
    )
    assert kept == slanted and glyph_line == {"glyphs": [a, b]}
    assert added["transcript"] == "Ab"


def test_failed_geometry_only_reads_headers_once(tmp_path, monkeypatch):
    cfg = _patch(tmp_path, hw=(16, 16), num_patches=[1, 1], max_attempts=4)
    opened = []
    original = Image.open

    def count_open(path, *args, **kwargs):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", count_open)
    monkeypatch.setattr(patch_module, "_read_patch", lambda path: pytest.fail(f"decoded {path}"))
    overlay = TextPatchOverlay(cfg, 1)
    assert not opened
    canvas = torch.zeros(3, 8, 8)
    assert overlay.apply(canvas, canvas.clone(), []) == []
    assert sorted(p.name for p in opened) == ["pg.png", "pl.png"]
    restored = pickle.loads(pickle.dumps(overlay))
    assert restored.apply(canvas, canvas.clone(), []) == []
    assert len(opened) == 2


def test_sat_rejected_pair_is_never_decoded(tmp_path, monkeypatch):
    cfg = _patch(tmp_path, hw=(8, 8), num_patches=[2, 2], max_attempts=3)
    Image.new("RGB", (8, 8)).save(tmp_path / "second.png")
    _json(
        tmp_path / "patch.json",
        [
            {"gt": "pg.png", "lq": "pl.png", "label": "first"},
            {"gt": "second.png", "lq": "second.png", "label": "second"},
        ],
    )
    overlay = TextPatchOverlay(cfg, 1)
    picks = iter([overlay.samples[0], *([overlay.samples[1]] * 3)])
    monkeypatch.setattr(random, "choice", lambda seq: next(picks))
    original = patch_module._read_patch
    decoded = []

    def read(path):
        decoded.append(path.name)
        return original(path)

    monkeypatch.setattr(patch_module, "_read_patch", read)
    canvas = torch.zeros(3, 8, 8)
    (first,) = overlay.apply(canvas, canvas.clone(), [])
    assert first["transcript"] == "first" and decoded == ["pg.png", "pl.png"]


@pytest.mark.parametrize("as_mapping", [False, True])
def test_load_train_config_rejects_patch_with_degradation_queue(tmp_path, as_mapping):
    from config import RealESRGANDegraderConfig

    raw = _minimal_train_dict(tmp_path)
    sources = raw["data"]["sources"]
    sources.append({**sources[0], "text_patch": {"json_path": "patch.json", "root_dir": "patch"}})
    deg = RealESRGANDegraderConfig(queue_size=8)
    raw["data"]["degradation"] = vars(deg) if as_mapping else deg
    with pytest.raises(ConfigValidationError, match=r"data\.sources\[1\]\.text_patch.*queue_size"):
        load_train_config(raw)


@pytest.mark.parametrize(
    ("bad_path", "error"),
    [
        ("missing_json", FileNotFoundError),
        ("directory_json", FileNotFoundError),
        ("missing_root", FileNotFoundError),
        ("file_root", ValueError),
    ],
)
def test_trainer_and_manifest_fail_fast_on_patch_paths(tmp_path, monkeypatch, bad_path, error):
    from engine import trainer as trainer_module

    manifest = _json(tmp_path / "patch.json", [{"gt": "g.png", "lq": "l.png", "label": "A"}])
    json_path = {
        "missing_json": str(tmp_path / "missing.json"),
        "directory_json": str(tmp_path),
    }.get(bad_path, manifest)
    root = {"missing_root": str(tmp_path / "missing"), "file_root": manifest}.get(
        bad_path, str(tmp_path)
    )
    raw = _minimal_train_dict(tmp_path)
    sources = raw["data"]["sources"]
    sources.append({**sources[0], "text_patch": {"json_path": json_path, "root_dir": root}})
    cfg = load_train_config(raw)
    assert cfg.data.sources[1].text_patch.is_configured()
    monkeypatch.setattr(
        trainer_module, "resolve_work_dir", lambda *_: pytest.fail("resolved work_dir")
    )
    monkeypatch.setattr(
        trainer_module, "Accelerator", lambda **_: pytest.fail("created accelerator")
    )
    field = "json_path" if bad_path.endswith("json") else "root_dir"
    with pytest.raises(error, match=field):
        load_text_patch_manifest(json_path, root)
    with pytest.raises(error, match=rf"data\.sources\[1\]\.text_patch\.{field}"):
        trainer_module.Trainer(cfg)
    assert not (tmp_path / "unit_exp").exists()


def test_shared_edge_patch_drops_contaminated_source_line(tmp_path, monkeypatch):
    cfg = _patch(tmp_path, hw=(4, 4), num_patches=[1, 1], max_attempts=1, rotate_range=[180, 180])
    placed = canonical_quad([[4.5, 4.0], [0.5, 4.0], [0.5, 0.0], [4.5, 0.0]])
    source = {"quad": canonical_quad([4.5, 0.0, 8.5, 4.0]).tolist(), "transcript": "adjacent"}
    assert not polygons_overlap(placed, source["quad"])
    monkeypatch.setattr(patch_module, "_place_quad", lambda *args: placed)
    gt, lq = torch.zeros(3, 8, 12), torch.zeros(3, 8, 12)
    lines = TextPatchOverlay(cfg, 1).apply(gt, lq, [source])
    assert gt[:, :4, 4].any()
    assert [line["transcript"] for line in lines] == ["Ab"]


def test_degrader_clamps_background_before_semitransparent_patch_blend(monkeypatch):
    from data.degradation import realesrgan as deg_module

    degrader = RealESRGANDegrader(
        scale=1,
        resize_prob=(0.0, 0.0, 1.0),
        resize_range=(1.0, 1.0),
        gaussian_noise_prob=1.0,
        noise_range=(0.0, 0.0),
        jpeg_range=(100.0, 100.0),
        second_blur_prob=0.0,
        resize_prob2=(0.0, 0.0, 1.0),
        resize_range2=(1.0, 1.0),
        gaussian_noise_prob2=1.0,
        noise_range2=(0.0, 0.0),
        jpeg_range2=(100.0, 100.0),
    )
    # Take the else branch so filter2d(out, sk) runs immediately before patch blending and
    # produces negative sinc/bicubic undershoot (-0.10547).
    monkeypatch.setattr(random, "random", lambda: 0.9)
    monkeypatch.setattr(deg_module, "filter2d", lambda x, k: torch.full_like(x, -0.10547))

    k = torch.zeros(1, 3, 3)
    k[:, 1, 1] = 1.0
    batch = {
        "gt": torch.zeros(1, 3, 8, 8),
        "text_patch_bg_gt": torch.zeros(1, 3, 8, 8),
        "lq": torch.full((1, 3, 8, 8), 0.5),
        "text_patch_mask": torch.full((1, 1, 8, 8), 0.5),
        "need_degradation": torch.tensor([True]),
        "kernel1": k,
        "kernel2": k,
        "sinc_kernel": k,
    }
    out = degrader(batch)
    expected = torch.full((1, 3, 8, 8), round(0.5 * 255.0) / 255.0)
    torch.testing.assert_close(out["lq"], expected, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ("+data.text_patch={json_path:old.json,root_dir:old}", "text_patch"),
        ("+data.sources.1.text_patch.scale_ragne=[0.9,1.1]", r"data\.sources\[1\]\.text_patch\.scale_ragne"),
    ],
)
def test_hydra_plus_overrides_cannot_bypass_structured_validation(tmp_path, override, match):
    from pathlib import Path

    repo_configs = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(config_dir=str(repo_configs), version_base=None):
        composed = hydra.compose(
            config_name="train",
            overrides=[
                "experiment=vosr2_onestep_text",
                "data.sources=[{name:s0,root_dir:/tmp,json_path:/tmp/a.json,ratio:0.5},"
                "{name:s1,root_dir:/tmp,json_path:/tmp/b.json,ratio:0.5}]",
                override,
            ],
        )
    with pytest.raises(ConfigValidationError, match=match):
        load_train_config(composed)


def test_validate_sources_and_trainer_fail_fast_on_non_json_patch_manifest(tmp_path, monkeypatch):
    from data.manifest import validate_sources_config
    from engine import trainer as trainer_module

    sources = _sources(tmp_path, 1, [{"gt": "gt.png", "lq": "lq.png"}])
    png_path = str(tmp_path / "gt.png")
    mixed_sources = [
        sources[0],
        {
            **sources[0],
            "name": "bad_patch_src",
            "text_patch": {"json_path": png_path, "root_dir": str(tmp_path)},
        },
    ]
    with pytest.raises(ValueError, match=r"data\.sources\[1\]\.text_patch\.json_path"):
        validate_sources_config(mixed_sources, has_degradation=False, is_main_process=False)

    raw = _minimal_train_dict(tmp_path)
    raw["scale"] = 1
    raw["model"]["scale"] = 1
    raw["data"]["scale"] = 1
    raw["data"]["sources"] = mixed_sources
    cfg = load_train_config(raw)
    monkeypatch.setattr(
        trainer_module, "resolve_work_dir", lambda *_: pytest.fail("resolved work_dir")
    )
    monkeypatch.setattr(
        trainer_module, "Accelerator", lambda **_: pytest.fail("created accelerator")
    )
    with pytest.raises(ValueError, match=r"data\.sources\[1\]\.text_patch\.json_path"):
        trainer_module.Trainer(cfg)
    assert not (tmp_path / "unit_exp").exists()


def test_dataclass_source_with_nested_path_objects(tmp_path):
    from config import DataSourceConfig
    from data.manifest import validate_sources_config

    patch_cfg = _patch(tmp_path, scale=1, label="PathOK", num_patches=[1, 1])
    _sources(tmp_path, 1, [{"gt": "gt.png", "lq": "lq.png"}])
    src_obj = DataSourceConfig(
        name="dc_src",
        root_dir=tmp_path,  # type: ignore[arg-type]
        json_path=tmp_path / "src.json",  # type: ignore[arg-type]
        ratio=1.0,
        text_patch=TextPatchConfig(
            json_path=tmp_path / "patch.json",  # type: ignore[arg-type]
            root_dir=tmp_path,  # type: ignore[arg-type]
            num_patches=patch_cfg.num_patches,
        ),
    )
    samples, stats = validate_sources_config([src_obj], has_degradation=False, is_main_process=False)
    assert len(samples) == 1 and stats[0].name == "dc_src"
    ds = MixedSRDataset([src_obj], 64, scale=1, has_degradation=False, is_main_process=False)
    assert ds[0]["text_lines"][0]["transcript"] == "PathOK"


@pytest.mark.parametrize("bad_q", ["8", True, -1])
def test_load_train_config_rejects_invalid_degradation_queue_size_type(tmp_path, bad_q):
    raw = _minimal_train_dict(tmp_path)
    raw["data"]["degradation"] = {
        "_target_": "data.degradation.RealESRGANDegrader",
        "queue_size": bad_q,
    }
    with pytest.raises(ConfigValidationError, match=r"data\.degradation\.queue_size"):
        load_train_config(raw)
