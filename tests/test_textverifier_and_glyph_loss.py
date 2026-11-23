"""Unit, contract, asset-validation, DDP, and gated real-asset tests for TextVerifier & GlyphLoss."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from archs import VOSR2OneStep
from archs.textverifier import (
    TEXTVERIFIER_BACKBONE_NORM_EPS,
    TEXTVERIFIER_EMBED_DIM,
    TEXTVERIFIER_IMAGE_SIZE,
    TEXTVERIFIER_PROJECTOR_NORM_EPS,
    TEXTVERIFIER_VISION_NUM_KEYS,
    Attention,
    Block,
    ImageProjector,
    TextVerifierVision,
)
from archs.vosr2.qwen_vae2d import AutoencoderKLQwenImage2D
from losses import GlyphLoss, LPIPSLoss, OCRKLLoss
from losses.textverifier import (
    DEFAULT_TEXTVERIFIER_REPO,
    MANIFEST_SHA256_FILENAME,
    PROTOTYPE_METADATA_FILENAME,
    PROTOTYPES_FILENAME,
    STAGE3_CHECKPOINT_FILENAME,
    TEXTVERIFIER_ASSET_FILES,
    _load_prototype_bank,
    _load_vision_encoder,
    load_textverifier_assets,
    resolve_textverifier_files,
)
from metrics import LPIPSMetric
from tests.helpers.fake_backends import (
    FakeTextVerifierEncoder,
    make_fake_textverifier_assets,
    patch_textverifier_loader,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(bundle_dir: Path, filenames: tuple[str, ...] | None = None) -> Path:
    names = filenames or (
        STAGE3_CHECKPOINT_FILENAME,
        PROTOTYPES_FILENAME,
        PROTOTYPE_METADATA_FILENAME,
    )
    lines = [f"{_sha256_bytes((bundle_dir / name).read_bytes())}  {name}" for name in names]
    manifest_path = bundle_dir / MANIFEST_SHA256_FILENAME
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def _make_valid_metadata(num_chars: int = 3, k: int = 2, threshold: float = 0.0215) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "architecture": "TextVerifier Stage-3, ViT-L/16 image encoder, IDS-aware dual encoder",
        "image_size": TEXTVERIFIER_IMAGE_SIZE,
        "embedding_dim": TEXTVERIFIER_EMBED_DIM,
        "prototype_shape": [num_chars, k, TEXTVERIFIER_EMBED_DIM],
        "num_characters": num_chars,
        "K": k,
        "threshold": threshold,
        "prototype_norm_min": 0.9999998,
        "prototype_norm_max": 1.0000001,
    }


def _make_valid_prototype_pack(
    chars: list[str] | None = None,
    k: int = 2,
    threshold: float = 0.0215,
    seed: int = 7,
) -> dict[str, Any]:
    chars = chars or ["中", "文", "春"]
    gen = torch.Generator().manual_seed(seed)
    bank = F.normalize(torch.randn((len(chars), k, TEXTVERIFIER_EMBED_DIM), generator=gen), dim=-1)
    return {
        "prototypes": bank,
        "chars": chars,
        "char_to_index": {ch: i for i, ch in enumerate(chars)},
        "K": k,
        "thr": threshold,
    }


def _create_synthetic_bundle_files(
    bundle_dir: Path,
    *,
    stage3_payload: Any = b"stub-stage3-bytes",
    proto_pack: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(stage3_payload, bytes):
        (bundle_dir / STAGE3_CHECKPOINT_FILENAME).write_bytes(stage3_payload)
    else:
        torch.save(stage3_payload, bundle_dir / STAGE3_CHECKPOINT_FILENAME)

    pack = proto_pack if proto_pack is not None else _make_valid_prototype_pack()
    torch.save(pack, bundle_dir / PROTOTYPES_FILENAME)

    meta = metadata if metadata is not None else _make_valid_metadata(
        num_chars=len(pack["chars"]),
        k=int(pack["K"]),
        threshold=float(pack["thr"]),
    )
    (bundle_dir / PROTOTYPE_METADATA_FILENAME).write_text(json.dumps(meta), encoding="utf-8")
    _write_manifest(bundle_dir)
    return bundle_dir


# ---------------------------------------------------------------------------
# 1. TextVerifierVision Architecture & 298-Key Contract Tests
# ---------------------------------------------------------------------------


def test_textverifier_vision_meta_298_keys_shapes_and_epsilons() -> None:
    with torch.device("meta"):
        model = TextVerifierVision()

    sd = model.state_dict()
    assert len(sd) == TEXTVERIFIER_VISION_NUM_KEYS == 298

    backbone_keys = [k for k in sd if k.startswith("image_encoder.backbone.")]
    projector_keys = [k for k in sd if k.startswith("image_projector.")]
    assert len(backbone_keys) == 294
    assert len(projector_keys) == 4
    assert len(backbone_keys) + len(projector_keys) == len(sd)

    # Exact shapes for stem, embeddings, all 24 blocks, fc_norm, and projector
    assert sd["image_encoder.backbone.cls_token"].shape == (1, 1, 1024)
    assert sd["image_encoder.backbone.pos_embed"].shape == (1, 65, 1024)
    assert sd["image_encoder.backbone.patch_embed.proj.weight"].shape == (1024, 3, 16, 16)
    assert sd["image_encoder.backbone.patch_embed.proj.bias"].shape == (1024,)
    assert sd["image_encoder.backbone.fc_norm.weight"].shape == (1024,)
    assert sd["image_encoder.backbone.fc_norm.bias"].shape == (1024,)

    for idx in range(24):
        prefix = f"image_encoder.backbone.blocks.{idx}."
        assert sd[f"{prefix}norm1.weight"].shape == (1024,)
        assert sd[f"{prefix}norm1.bias"].shape == (1024,)
        assert sd[f"{prefix}attn.qkv.weight"].shape == (3072, 1024)
        assert sd[f"{prefix}attn.qkv.bias"].shape == (3072,)
        assert sd[f"{prefix}attn.proj.weight"].shape == (1024, 1024)
        assert sd[f"{prefix}attn.proj.bias"].shape == (1024,)
        assert sd[f"{prefix}norm2.weight"].shape == (1024,)
        assert sd[f"{prefix}norm2.bias"].shape == (1024,)
        assert sd[f"{prefix}mlp.fc1.weight"].shape == (4096, 1024)
        assert sd[f"{prefix}mlp.fc1.bias"].shape == (4096,)
        assert sd[f"{prefix}mlp.fc2.weight"].shape == (1024, 4096)
        assert sd[f"{prefix}mlp.fc2.bias"].shape == (1024,)

    assert set(projector_keys) == {
        "image_projector.fc1.weight",
        "image_projector.ln.weight",
        "image_projector.ln.bias",
        "image_projector.fc2.weight",
    }
    assert sd["image_projector.fc1.weight"].shape == (1024, 1024)
    assert sd["image_projector.ln.weight"].shape == (1024,)
    assert sd["image_projector.ln.bias"].shape == (1024,)
    assert sd["image_projector.fc2.weight"].shape == (1024, 1024)

    # Prohibited extra keys must not exist
    assert "image_encoder.backbone.norm.weight" not in sd
    assert "image_projector.fc1.bias" not in sd
    assert "image_projector.fc2.bias" not in sd

    # Distinct LayerNorm epsilons: backbone=1e-6 vs projector=1e-5
    backbone = model.image_encoder.backbone
    assert len(backbone.blocks) == 24
    assert backbone.fc_norm.eps == TEXTVERIFIER_BACKBONE_NORM_EPS == 1e-6
    for blk in backbone.blocks:
        assert blk.norm1.eps == 1e-6
        assert blk.norm2.eps == 1e-6
        assert blk.attn.num_heads == 16
        assert blk.attn.head_dim == 64
    assert model.image_projector.ln.eps == TEXTVERIFIER_PROJECTOR_NORM_EPS == 1e-5


def test_textverifier_attention_block_and_projector_exact_math() -> None:
    torch.manual_seed(11)
    attn = Attention(dim=32, num_heads=4).eval()
    x = torch.randn(2, 9, 32, requires_grad=True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = attn(x)

    # Explicit manual QKV attention reference (verifies single head_dim**-0.5 scaling)
    b, n, c = x.shape
    qkv = attn.qkv(x).reshape(b, n, 3, 4, 8).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    scores = (q @ k.transpose(-2, -1)) * (8**-0.5)
    ref = attn.proj((scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c))
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)

    blk = Block(dim=32, num_heads=4, mlp_ratio=4.0, norm_eps=1e-6).eval()
    y = blk(x)
    assert y.shape == x.shape and torch.isfinite(y).all()

    proj = ImageProjector(dim=32, norm_eps=1e-5).eval()
    assert proj.fc1.bias is None and proj.fc2.bias is None
    p = proj(y.mean(dim=1))
    assert p.shape == (2, 32) and torch.isfinite(p).all()


def test_textverifier_backbone_rejects_bad_input_shape() -> None:
    with torch.device("meta"):
        model = TextVerifierVision()
    with pytest.raises(ValueError, match="128, 128"):
        model(torch.empty(1, 3, 64, 64, device="meta"))
    with pytest.raises(ValueError, match="128, 128"):
        model(torch.empty(1, 1, 128, 128, device="meta"))


# ---------------------------------------------------------------------------
# 2. Strict Asset Resolution & Integrity / Schema Validation Tests
# ---------------------------------------------------------------------------


def test_resolve_textverifier_files_strict_local_never_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    def _forbid_download(**_kwargs: Any) -> str:
        raise AssertionError("snapshot_download must not be called when model_dir is provided")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _forbid_download)

    # Missing directory
    with pytest.raises(FileNotFoundError):
        resolve_textverifier_files(model_dir=tmp_path / "missing_dir", repo_id=DEFAULT_TEXTVERIFIER_REPO)

    # File instead of directory
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="directory"):
        resolve_textverifier_files(model_dir=file_path, repo_id=DEFAULT_TEXTVERIFIER_REPO)

    # Legacy 2-file directory missing metadata & manifest
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / STAGE3_CHECKPOINT_FILENAME).write_bytes(b"a")
    (partial / PROTOTYPES_FILENAME).write_bytes(b"b")
    with pytest.raises(FileNotFoundError, match="prototype_metadata.json"):
        resolve_textverifier_files(model_dir=partial, repo_id=DEFAULT_TEXTVERIFIER_REPO)

    # Complete 4-file directory succeeds
    full = _create_synthetic_bundle_files(tmp_path / "full_bundle")
    files = resolve_textverifier_files(model_dir=full, repo_id=DEFAULT_TEXTVERIFIER_REPO)
    assert files.root_dir == full
    assert files.stage3_path == full / STAGE3_CHECKPOINT_FILENAME
    assert files.source.startswith("local:")


def test_resolve_textverifier_files_remote_single_snapshot_no_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    bundle = _create_synthetic_bundle_files(tmp_path / "snap_abc123")
    calls: list[dict[str, Any]] = []

    def _fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(bundle)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)

    files = resolve_textverifier_files(
        model_dir=None,
        repo_id="ATH-MaaS/TextVerifier-CJK-CLIP",
        cache_dir=tmp_path / "cache",
        local_files_only=True,
    )
    assert len(calls) == 1
    assert "revision" not in calls[0]
    assert calls[0]["repo_id"] == "ATH-MaaS/TextVerifier-CJK-CLIP"
    assert set(calls[0]["allow_patterns"]) == set(TEXTVERIFIER_ASSET_FILES)
    assert calls[0]["local_files_only"] is True
    assert calls[0]["cache_dir"] == str(tmp_path / "cache")
    assert files.root_dir == bundle
    assert files.source == "hf:ATH-MaaS/TextVerifier-CJK-CLIP@snap_abc123"


def test_manifest_and_metadata_validation_failures(tmp_path: Path) -> None:
    # 1. Checksum mismatch
    b1 = _create_synthetic_bundle_files(tmp_path / "bad_sha")
    (b1 / STAGE3_CHECKPOINT_FILENAME).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"\[checksum\] SHA-256 mismatch"):
        load_textverifier_assets(model_dir=b1)

    # 2. Path traversal in manifest
    b2 = _create_synthetic_bundle_files(tmp_path / "bad_path")
    (b2 / MANIFEST_SHA256_FILENAME).write_text(
        f"{'a' * 64}  ../textverifier_stage3.pth\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"\[checksum\] Unsafe path"):
        load_textverifier_assets(model_dir=b2)

    # 3. Duplicate manifest entry
    b3 = _create_synthetic_bundle_files(tmp_path / "dup_entry")
    h = _sha256_bytes((b3 / STAGE3_CHECKPOINT_FILENAME).read_bytes())
    (b3 / MANIFEST_SHA256_FILENAME).write_text(
        f"{h}  {STAGE3_CHECKPOINT_FILENAME}\n{h}  {STAGE3_CHECKPOINT_FILENAME}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"\[checksum\] Duplicate"):
        load_textverifier_assets(model_dir=b3)

    # 4. Invalid metadata image_size
    bad_meta = _make_valid_metadata()
    bad_meta["image_size"] = 224
    b4 = _create_synthetic_bundle_files(tmp_path / "bad_meta", metadata=bad_meta)
    with pytest.raises(ValueError, match=r"\[schema\] Unsupported image_size"):
        load_textverifier_assets(model_dir=b4)

    # 5. Bool K in metadata
    bad_k_meta = _make_valid_metadata()
    bad_k_meta["K"] = True
    b5 = _create_synthetic_bundle_files(tmp_path / "bad_k", metadata=bad_k_meta)
    with pytest.raises(ValueError, match=r"\[schema\] Invalid K"):
        load_textverifier_assets(model_dir=b5)


def test_load_vision_encoder_strict_keys_shapes_finite_and_text_tower_whitelist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dummy_ckpt = tmp_path / STAGE3_CHECKPOINT_FILENAME
    dummy_ckpt.write_bytes(b"placeholder")

    with torch.device("meta"):
        template = TextVerifierVision()
    expected_shapes = {k: tuple(v.shape) for k, v in template.state_dict().items()}

    def _make_full_state() -> dict[str, torch.Tensor]:
        # Tiny memory footprint: broadcast single-element views when testing validation errors,
        # or small real tensors via expand() (cloned on verified load).
        state: dict[str, torch.Tensor] = {}
        for k, shape in expected_shapes.items():
            state[k] = torch.zeros(1, dtype=torch.float16).expand(shape)
        # Official non-image keys explicitly ignored
        state["text_encoder.embeddings.word_embeddings.weight"] = torch.zeros(2, 2)
        state["text_projector.fc1.weight"] = torch.zeros(2, 2)
        state["logit_scale"] = torch.tensor(1.0)
        return state

    # Unexpected unknown key rejected
    bad_extra = _make_full_state()
    bad_extra["mystery_head.weight"] = torch.zeros(1)
    monkeypatch.setattr("losses.textverifier.torch.load", lambda *_a, **_kw: {"model_state_dict": bad_extra})
    with pytest.raises(ValueError, match=r"\[unexpected key\]"):
        _load_vision_encoder(dummy_ckpt)

    # Missing image key rejected
    bad_missing = _make_full_state()
    del bad_missing["image_projector.fc2.weight"]
    monkeypatch.setattr("losses.textverifier.torch.load", lambda *_a, **_kw: {"model_state_dict": bad_missing})
    with pytest.raises(ValueError, match=r"\[missing/unexpected key\]"):
        _load_vision_encoder(dummy_ckpt)

    # Wrong shape rejected
    bad_shape = _make_full_state()
    bad_shape["image_projector.fc2.weight"] = torch.zeros(16, 16)
    monkeypatch.setattr("losses.textverifier.torch.load", lambda *_a, **_kw: {"model_state_dict": bad_shape})
    with pytest.raises(ValueError, match=r"\[shape\]"):
        _load_vision_encoder(dummy_ckpt)

    # Non-finite weight rejected
    bad_nan = _make_full_state()
    nan_w = torch.zeros(1024, 1024, dtype=torch.float32)
    nan_w[0, 0] = float("nan")
    bad_nan["image_projector.fc1.weight"] = nan_w
    monkeypatch.setattr("losses.textverifier.torch.load", lambda *_a, **_kw: {"model_state_dict": bad_nan})
    with pytest.raises(ValueError, match=r"\[non-finite\]"):
        _load_vision_encoder(dummy_ckpt)


def test_load_prototype_bank_validation_failures_and_happy_path(tmp_path: Path) -> None:
    meta = _make_valid_metadata(num_chars=3, k=2, threshold=0.0215)

    # Happy path
    good_pack = _make_valid_prototype_pack(chars=["中", "文", "春"], k=2, threshold=0.0215)
    good_path = tmp_path / "good.pt"
    torch.save(good_pack, good_path)
    bank, mapping, thr = _load_prototype_bank(good_path, meta)
    assert bank.shape == (3, 2, TEXTVERIFIER_EMBED_DIM)
    assert mapping == {"中": 0, "文": 1, "春": 2}
    assert thr == pytest.approx(0.0215)

    # Zero-norm prototype vector rejected
    zero_pack = _make_valid_prototype_pack(chars=["中", "文", "春"], k=2, threshold=0.0215)
    zero_pack["prototypes"] = zero_pack["prototypes"].clone()
    zero_pack["prototypes"][1, 0].zero_()
    zero_path = tmp_path / "zero.pt"
    torch.save(zero_pack, zero_path)
    with pytest.raises(ValueError, match=r"\[non-finite\].*zero"):
        _load_prototype_bank(zero_path, meta)

    # NaN in prototype bank rejected
    nan_pack = _make_valid_prototype_pack(chars=["中", "文", "春"], k=2, threshold=0.0215)
    nan_pack["prototypes"] = nan_pack["prototypes"].clone()
    nan_pack["prototypes"][0, 0, 0] = float("nan")
    nan_path = tmp_path / "nan.pt"
    torch.save(nan_pack, nan_path)
    with pytest.raises(ValueError, match=r"\[non-finite\]"):
        _load_prototype_bank(nan_path, meta)

    # Duplicate chars rejected
    dup_pack = _make_valid_prototype_pack(chars=["中", "文", "春"], k=2, threshold=0.0215)
    dup_pack["chars"] = ["中", "中", "春"]
    dup_path = tmp_path / "dup.pt"
    torch.save(dup_pack, dup_path)
    with pytest.raises(ValueError, match=r"\[mapping\] Duplicate"):
        _load_prototype_bank(dup_path, meta)

    # Non-row-aligned char_to_index rejected
    bad_map_pack = _make_valid_prototype_pack(chars=["中", "文", "春"], k=2, threshold=0.0215)
    bad_map_pack["char_to_index"] = {"中": 1, "文": 0, "春": 2}
    bad_map_path = tmp_path / "bad_map.pt"
    torch.save(bad_map_pack, bad_map_path)
    with pytest.raises(ValueError, match=r"\[mapping\] Non-row-aligned"):
        _load_prototype_bank(bad_map_path, meta)

    # Bool K rejected
    bool_k_pack = _make_valid_prototype_pack(chars=["中", "文", "春"], k=2, threshold=0.0215)
    bool_k_pack["K"] = True
    bool_k_path = tmp_path / "bool_k.pt"
    torch.save(bool_k_pack, bool_k_path)
    with pytest.raises(ValueError, match=r"\[schema\] Invalid K"):
        _load_prototype_bank(bool_k_path, meta)


# ---------------------------------------------------------------------------
# 3. Removed Stub / Revision / DI API Tightening Tests
# ---------------------------------------------------------------------------


def test_removed_stub_and_revision_kwargs_raise_type_error() -> None:
    for bad_kwargs in (
        {"revision": "abc"},
        {"pretrained": False},
        {"encoder": nn.Identity()},
        {"prototypes": torch.ones(1, 1, 4)},
        {"char_to_index": {"中": 0}},
    ):
        with pytest.raises(TypeError):
            GlyphLoss(**bad_kwargs)  # type: ignore[arg-type]

    for bad_ocr_kwargs in (
        {"pretrained": False},
        {"num_classes": 10},
        {"recognizer": nn.Identity()},
        {"input_height": 16},
    ):
        with pytest.raises(TypeError):
            OCRKLLoss(**bad_ocr_kwargs)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        LPIPSLoss(net="alex", pretrained=False)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        LPIPSMetric(net="alex", pretrained=False)  # type: ignore[call-arg]

    for bad_vosr_kwargs in (
        {"vae_pretrained": False},
        {"dino_pretrained": False},
        {"vae_base_dim": 8},
        {"vae_dim_mult": (1, 2)},
        {"vae_num_res_blocks": 1},
    ):
        with pytest.raises(TypeError):
            VOSR2OneStep(**bad_vosr_kwargs)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        AutoencoderKLQwenImage2D.from_pretrained("CSWRY/VOSR", revision="main")  # type: ignore[call-arg]

    # C1: Non-16 z_dim requires explicit latents_mean and latents_std
    with pytest.raises(ValueError, match="latents_mean"):
        AutoencoderKLQwenImage2D(base_dim=8, z_dim=4, dim_mult=(1, 2), num_res_blocks=1)


# ---------------------------------------------------------------------------
# 4. GlyphLoss Chunk Invariance & Range Semantics
# ---------------------------------------------------------------------------


def test_glyph_loss_chunk_invariance_and_input_range(monkeypatch: pytest.MonkeyPatch) -> None:
    assets = make_fake_textverifier_assets(embed_dim=TEXTVERIFIER_EMBED_DIM, num_prototypes=4, seed=9)
    patch_textverifier_loader(monkeypatch, assets=assets)

    loss_bs1 = GlyphLoss(batch_size=1, input_range="0_1", sync_ddp_counts=False)
    loss_bs4 = GlyphLoss(batch_size=4, input_range="0_1", sync_ddp_counts=False)
    loss_m11 = GlyphLoss(batch_size=2, input_range="-1_1", sync_ddp_counts=False)

    torch.manual_seed(12)
    pred_01_a = torch.rand(2, 3, 32, 32, requires_grad=True)
    pred_01_b = pred_01_a.detach().clone().requires_grad_(True)
    pred_m11 = (pred_01_a.detach() * 2.0 - 1.0).requires_grad_(True)

    lines = [
        [{"glyphs": [{"char": "中", "bbox": [2, 2, 14, 14]}, {"char": "文", "bbox": [16, 2, 28, 14]}]}],
        [{"glyphs": [{"char": "春", "bbox": [4, 16, 20, 30]}]}],
    ]

    v1 = loss_bs1(pred_01_a, None, text_lines=lines)
    v4 = loss_bs4(pred_01_b, None, text_lines=lines)
    vm = loss_m11(pred_m11, None, text_lines=lines)

    torch.testing.assert_close(v1, v4, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(v1, vm, atol=1e-6, rtol=1e-5)

    v1.backward()
    v4.backward()
    torch.testing.assert_close(pred_01_a.grad, pred_01_b.grad, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Two-Process CPU Gloo DDP Glyph-Count Weighting Parity
# ---------------------------------------------------------------------------


def _ddp_glyph_worker(
    rank: int,
    world_size: int,
    port: int,
    case: str,
    out_file: str,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        # Install test-only fake loader inside worker process (no src test flags!)
        import losses.glyph_loss as gl_mod

        fake_assets = make_fake_textverifier_assets(
            encoder=FakeTextVerifierEncoder(embed_dim=16, seed=3),
            embed_dim=16,
            num_prototypes=3,
            seed=3,
        )
        orig_loader = gl_mod.load_textverifier_assets
        gl_mod.load_textverifier_assets = lambda **_kw: fake_assets
        try:
            loss_fn = GlyphLoss(batch_size=2, input_range="0_1", sync_ddp_counts=True)
        finally:
            gl_mod.load_textverifier_assets = orig_loader

        torch.manual_seed(101)
        net = nn.Conv2d(3, 3, kernel_size=1)
        ddp_net = nn.parallel.DistributedDataParallel(net)

        torch.manual_seed(202 + rank)
        inp = torch.rand(1, 3, 24, 24)

        if case == "unequal":
            lines = (
                [[{"glyphs": [{"char": "中", "bbox": [2, 2, 10, 10]}, {"char": "文", "bbox": [12, 2, 20, 10]}, {"char": "春", "bbox": [2, 12, 10, 20]}]}]]
                if rank == 0
                else [[{"glyphs": [{"char": "夏", "bbox": [4, 4, 16, 16]}]}]]
            )
        elif case == "one_zero":
            lines = (
                [[{"glyphs": [{"char": "中", "bbox": [2, 2, 10, 10]}, {"char": "文", "bbox": [12, 2, 20, 10]}]}]]
                if rank == 0
                else [[]]
            )
        else:  # both_zero
            lines = [[]]

        pred = ddp_net(inp)
        loss = loss_fn(pred, None, text_lines=lines)
        loss.backward()

        if rank == 0:
            torch.save(
                {
                    "weight_grad": net.weight.grad.detach().clone(),
                    "bias_grad": net.bias.grad.detach().clone(),
                },
                out_file,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["unequal", "one_zero", "both_zero"])
def test_glyph_loss_ddp_two_process_gloo_matches_global_glyph_mean(
    tmp_path: Path,
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])

    out_file = tmp_path / f"ddp_{case}.pt"
    mp.spawn(
        _ddp_glyph_worker,
        args=(2, port, case, str(out_file)),
        nprocs=2,
        join=True,
    )
    ddp_grads = torch.load(out_file, map_location="cpu", weights_only=True)

    # Single-process reference over the exact same combined 2-image global batch
    fake_assets = make_fake_textverifier_assets(
        encoder=FakeTextVerifierEncoder(embed_dim=16, seed=3),
        embed_dim=16,
        num_prototypes=3,
        seed=3,
    )
    patch_textverifier_loader(monkeypatch, assets=fake_assets)
    ref_loss_fn = GlyphLoss(batch_size=2, input_range="0_1", sync_ddp_counts=False)

    torch.manual_seed(101)
    ref_net = nn.Conv2d(3, 3, kernel_size=1)
    torch.manual_seed(202)
    inp0 = torch.rand(1, 3, 24, 24)
    torch.manual_seed(203)
    inp1 = torch.rand(1, 3, 24, 24)
    global_inp = torch.cat([inp0, inp1], dim=0)

    if case == "unequal":
        global_lines = [
            [{"glyphs": [{"char": "中", "bbox": [2, 2, 10, 10]}, {"char": "文", "bbox": [12, 2, 20, 10]}, {"char": "春", "bbox": [2, 12, 10, 20]}]}],
            [{"glyphs": [{"char": "夏", "bbox": [4, 4, 16, 16]}]}],
        ]
    elif case == "one_zero":
        global_lines = [
            [{"glyphs": [{"char": "中", "bbox": [2, 2, 10, 10]}, {"char": "文", "bbox": [12, 2, 20, 10]}]}],
            [],
        ]
    else:
        global_lines = [[], []]

    ref_loss = ref_loss_fn(ref_net(global_inp), None, text_lines=global_lines)
    ref_loss.backward()

    torch.testing.assert_close(ddp_grads["weight_grad"], ref_net.weight.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(ddp_grads["bias_grad"], ref_net.bias.grad, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 6. Gated Real TextVerifier Stage-3 Integration Test
# ---------------------------------------------------------------------------


@pytest.mark.requires_textverifier_weights
def test_real_textverifier_stage3_bundle_load_forward_and_backward() -> None:
    """Real asset integration gate; requires TEXTVERIFIER_MODEL_DIR pointing to 4-file bundle."""
    model_dir = os.environ.get("TEXTVERIFIER_MODEL_DIR")
    if not model_dir:
        pytest.fail(
            "TEXTVERIFIER_MODEL_DIR must be set to a local 4-file TextVerifier Stage-3 bundle "
            "when running requires_textverifier_weights tests."
        )

    loss_fn = GlyphLoss(model_dir=model_dir, batch_size=2, sync_ddp_counts=False)
    assert len(loss_fn.char_to_index) == 8189
    assert tuple(loss_fn.prototypes.shape) == (8189, 10, 1024)
    assert len(loss_fn.encoder.state_dict()) == TEXTVERIFIER_VISION_NUM_KEYS

    pred = torch.rand(1, 3, 64, 64, requires_grad=True)
    loss = loss_fn(
        pred,
        None,
        text_lines=[[{"glyphs": [{"char": "中", "bbox": [4, 4, 36, 36]}]}]],
    )
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0
    assert all(p.grad is None for p in loss_fn.parameters())
