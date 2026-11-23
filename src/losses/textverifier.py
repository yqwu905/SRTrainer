"""Strict asset resolution, integrity verification, and loading for TextVerifier Stage-3."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from archs.textverifier import (
    TEXTVERIFIER_EMBED_DIM,
    TEXTVERIFIER_IMAGE_SIZE,
    TextVerifierVision,
)

__all__ = [
    "DEFAULT_TEXTVERIFIER_REPO",
    "STAGE3_CHECKPOINT_FILENAME",
    "PROTOTYPES_FILENAME",
    "PROTOTYPE_METADATA_FILENAME",
    "MANIFEST_SHA256_FILENAME",
    "TEXTVERIFIER_ASSET_FILES",
    "TextVerifierFiles",
    "TextVerifierAssets",
    "resolve_textverifier_files",
    "load_textverifier_assets",
]

DEFAULT_TEXTVERIFIER_REPO = "ATH-MaaS/TextVerifier-CJK-CLIP"
STAGE3_CHECKPOINT_FILENAME = "textverifier_stage3.pth"
PROTOTYPES_FILENAME = "textverifier_k10_q90.pt"
PROTOTYPE_METADATA_FILENAME = "prototype_metadata.json"
MANIFEST_SHA256_FILENAME = "manifest.sha256"

_CHECKSUMMED_FILES: tuple[str, ...] = (
    STAGE3_CHECKPOINT_FILENAME,
    PROTOTYPES_FILENAME,
    PROTOTYPE_METADATA_FILENAME,
)
TEXTVERIFIER_ASSET_FILES: tuple[str, ...] = (
    *_CHECKSUMMED_FILES,
    MANIFEST_SHA256_FILENAME,
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_IGNORED_CHECKPOINT_PREFIXES: tuple[str, ...] = ("text_encoder.", "text_projector.")
_IGNORED_CHECKPOINT_KEYS: frozenset[str] = frozenset({"logit_scale"})
_IMAGE_CHECKPOINT_PREFIXES: tuple[str, ...] = ("image_encoder.", "image_projector.")


@dataclass(frozen=True)
class TextVerifierFiles:
    """Resolved paths for all four files of a TextVerifier Stage-3 release bundle."""

    root_dir: Path
    stage3_path: Path
    prototypes_path: Path
    metadata_path: Path
    manifest_path: Path
    source: str


@dataclass(frozen=True)
class TextVerifierAssets:
    """Verified frozen TextVerifier Stage-3 vision encoder and normalized prototype bank."""

    encoder: TextVerifierVision
    prototypes: torch.Tensor
    char_to_index: dict[str, int]
    metadata: dict[str, Any]
    files: TextVerifierFiles
    sha256_digests: dict[str, str]
    threshold: float


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bundle_dir(root: Path, source: str) -> TextVerifierFiles:
    if not root.exists():
        raise FileNotFoundError(f"TextVerifier asset directory not found ({source}): {root}")
    if not root.is_dir():
        raise ValueError(f"TextVerifier asset path must be a directory ({source}), got file: {root}")

    resolved = {name: root / name for name in TEXTVERIFIER_ASSET_FILES}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"TextVerifier asset bundle incomplete under {root} ({source}); "
            f"missing required files: {missing}."
        )
    return TextVerifierFiles(
        root_dir=root,
        stage3_path=resolved[STAGE3_CHECKPOINT_FILENAME],
        prototypes_path=resolved[PROTOTYPES_FILENAME],
        metadata_path=resolved[PROTOTYPE_METADATA_FILENAME],
        manifest_path=resolved[MANIFEST_SHA256_FILENAME],
        source=source,
    )


def resolve_textverifier_files(
    *,
    model_dir: str | Path | None,
    repo_id: str = DEFAULT_TEXTVERIFIER_REPO,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> TextVerifierFiles:
    """Resolve all four TextVerifier release files from a strict local bundle or one HF snapshot.

    When ``model_dir`` is provided, only that local directory is consulted (never downloads).
    When ``model_dir`` is ``None``, fetches all four files atomically from a single
    ``huggingface_hub.snapshot_download`` call without any pinned revision.
    """
    if not isinstance(local_files_only, bool):
        raise ValueError(
            f"local_files_only must be a bool, got {type(local_files_only).__name__}."
        )

    if model_dir is not None:
        if isinstance(model_dir, str) and not model_dir.strip():
            raise ValueError("model_dir must be a non-empty path or None.")
        return _verify_bundle_dir(Path(model_dir), source=f"local:{model_dir}")

    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError(f"repo_id must be a non-empty string when model_dir is None, got {repo_id!r}.")

    from huggingface_hub import snapshot_download

    dl_kwargs: dict[str, Any] = {
        "repo_id": repo_id.strip(),
        "allow_patterns": list(TEXTVERIFIER_ASSET_FILES),
        "local_files_only": local_files_only,
    }
    if cache_dir is not None:
        dl_kwargs["cache_dir"] = str(cache_dir)

    snapshot_root = Path(snapshot_download(**dl_kwargs))
    return _verify_bundle_dir(
        snapshot_root,
        source=f"hf:{repo_id.strip()}@{snapshot_root.name}",
    )


def _parse_and_verify_manifest(files: TextVerifierFiles) -> dict[str, str]:
    manifest_path = files.manifest_path
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"[checksum] Failed to read {manifest_path}: {exc}") from exc

    expected_set = set(_CHECKSUMMED_FILES)
    declared: dict[str, str] = {}

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"[checksum] Malformed manifest line {line_no} in {manifest_path}: {raw_line!r}."
            )
        digest, filename = parts
        if not _SHA256_HEX_RE.fullmatch(digest):
            raise ValueError(
                f"[checksum] Invalid SHA-256 hex digest at line {line_no} in {manifest_path}: {digest!r}."
            )
        if "/" in filename or "\\" in filename or ".." in filename or Path(filename).name != filename:
            raise ValueError(
                f"[checksum] Unsafe path in manifest line {line_no} of {manifest_path}: {filename!r}."
            )
        if filename not in expected_set:
            raise ValueError(
                f"[checksum] Unexpected file entry {filename!r} in {manifest_path}; "
                f"expected only {sorted(expected_set)}."
            )
        if filename in declared:
            raise ValueError(
                f"[checksum] Duplicate manifest entry for {filename!r} in {manifest_path}."
            )
        declared[filename] = digest.lower()

    missing_entries = [name for name in _CHECKSUMMED_FILES if name not in declared]
    if missing_entries:
        raise ValueError(
            f"[checksum] Missing required entries {missing_entries} in {manifest_path}."
        )

    path_map = {
        STAGE3_CHECKPOINT_FILENAME: files.stage3_path,
        PROTOTYPES_FILENAME: files.prototypes_path,
        PROTOTYPE_METADATA_FILENAME: files.metadata_path,
    }
    actual_digests: dict[str, str] = {}
    for name in _CHECKSUMMED_FILES:
        file_path = path_map[name]
        actual = _sha256_file(file_path)
        expected = declared[name]
        if actual != expected:
            raise ValueError(
                f"[checksum] SHA-256 mismatch for {file_path}: "
                f"expected {expected}, computed {actual}."
            )
        actual_digests[name] = actual

    return actual_digests


def _load_and_validate_metadata(metadata_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"[schema] Failed to parse {metadata_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"[schema] Expected JSON object in {metadata_path}, got {type(raw).__name__}."
        )

    image_size = raw.get("image_size")
    if not _is_strict_int(image_size) or image_size != TEXTVERIFIER_IMAGE_SIZE:
        raise ValueError(
            f"[schema] Unsupported image_size={image_size!r} in {metadata_path}; "
            f"expected {TEXTVERIFIER_IMAGE_SIZE}."
        )

    embedding_dim = raw.get("embedding_dim")
    if not _is_strict_int(embedding_dim) or embedding_dim != TEXTVERIFIER_EMBED_DIM:
        raise ValueError(
            f"[schema] Unsupported embedding_dim={embedding_dim!r} in {metadata_path}; "
            f"expected {TEXTVERIFIER_EMBED_DIM}."
        )

    num_chars = raw.get("num_characters")
    if not _is_strict_int(num_chars) or num_chars < 1:
        raise ValueError(
            f"[schema] Invalid num_characters={num_chars!r} in {metadata_path}; expected positive int."
        )

    k_val = raw.get("K")
    if not _is_strict_int(k_val) or k_val < 1:
        raise ValueError(
            f"[schema] Invalid K={k_val!r} in {metadata_path}; expected positive int."
        )

    proto_shape = raw.get("prototype_shape")
    expected_shape = [num_chars, k_val, TEXTVERIFIER_EMBED_DIM]
    if (
        not isinstance(proto_shape, list)
        or len(proto_shape) != 3
        or any(not _is_strict_int(dim) for dim in proto_shape)
        or proto_shape != expected_shape
    ):
        raise ValueError(
            f"[schema] Inconsistent prototype_shape={proto_shape!r} in {metadata_path}; "
            f"expected {expected_shape}."
        )

    threshold = raw.get("threshold")
    if not _is_finite_real(threshold):
        raise ValueError(
            f"[schema] Invalid threshold={threshold!r} in {metadata_path}; expected finite real."
        )

    norm_min = raw.get("prototype_norm_min")
    norm_max = raw.get("prototype_norm_max")
    if norm_min is not None:
        if not _is_finite_real(norm_min) or float(norm_min) <= 0.0:
            raise ValueError(
                f"[schema] Invalid prototype_norm_min={norm_min!r} in {metadata_path}."
            )
    if norm_max is not None:
        if not _is_finite_real(norm_max) or float(norm_max) <= 0.0:
            raise ValueError(
                f"[schema] Invalid prototype_norm_max={norm_max!r} in {metadata_path}."
            )
    if norm_min is not None and norm_max is not None and float(norm_min) > float(norm_max):
        raise ValueError(
            f"[schema] prototype_norm_min ({norm_min}) > prototype_norm_max ({norm_max}) "
            f"in {metadata_path}."
        )

    return raw


def _load_vision_encoder(stage3_path: Path) -> TextVerifierVision:
    try:
        raw_ckpt = torch.load(stage3_path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise ValueError(f"[checkpoint] Failed to load {stage3_path}: {exc}") from exc

    if isinstance(raw_ckpt, Mapping) and "model_state_dict" in raw_ckpt:
        state = raw_ckpt["model_state_dict"]
    else:
        state = raw_ckpt

    if not isinstance(state, Mapping):
        raise ValueError(
            f"[checkpoint] Expected state_dict mapping in {stage3_path}, got {type(state).__name__}."
        )

    raw_image_state: dict[str, torch.Tensor] = {}
    unexpected_non_image: list[str] = []
    for key, val in state.items():
        if not isinstance(key, str):
            raise ValueError(
                f"[checkpoint] Non-string key {key!r} in {stage3_path}."
            )
        if key.startswith(_IMAGE_CHECKPOINT_PREFIXES):
            if not isinstance(val, torch.Tensor):
                raise ValueError(
                    f"[checkpoint] Expected tensor for key {key!r} in {stage3_path}, "
                    f"got {type(val).__name__}."
                )
            raw_image_state[key] = val
        elif key in _IGNORED_CHECKPOINT_KEYS or key.startswith(_IGNORED_CHECKPOINT_PREFIXES):
            continue
        else:
            unexpected_non_image.append(key)

    if unexpected_non_image:
        raise ValueError(
            f"[unexpected key] Unexpected keys in {stage3_path}: {sorted(unexpected_non_image)[:10]}."
        )

    with torch.device("meta"):
        encoder = TextVerifierVision()

    expected_sd = encoder.state_dict()
    expected_keys = set(expected_sd.keys())
    found_keys = set(raw_image_state.keys())

    missing_keys = sorted(expected_keys - found_keys)
    extra_image_keys = sorted(found_keys - expected_keys)
    if missing_keys or extra_image_keys:
        raise ValueError(
            f"[missing/unexpected key] Image weights in {stage3_path} do not match "
            f"TextVerifierVision: missing={missing_keys[:8]} (total {len(missing_keys)}), "
            f"unexpected={extra_image_keys[:8]} (total {len(extra_image_keys)})."
        )

    verified_state: dict[str, torch.Tensor] = {}
    for key, expected_tensor in expected_sd.items():
        src_tensor = raw_image_state[key]
        if tuple(src_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"[shape] Shape mismatch for {key!r} in {stage3_path}: "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(src_tensor.shape)}."
            )
        if not src_tensor.is_floating_point():
            raise ValueError(
                f"[schema] Expected floating-point weight for {key!r} in {stage3_path}, "
                f"got dtype={src_tensor.dtype}."
            )
        # Clone into owned CPU FP32 tensor so mmap reference to the full checkpoint can be freed.
        tensor_fp32 = src_tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(tensor_fp32).all():
            raise ValueError(f"[non-finite] Non-finite values in {key!r} from {stage3_path}.")
        verified_state[key] = tensor_fp32

    del raw_image_state, state, raw_ckpt

    encoder.load_state_dict(verified_state, strict=True, assign=True)
    for name, param in encoder.named_parameters():
        if param.device.type == "meta":
            raise RuntimeError(f"[checkpoint] Parameter {name!r} remained on meta device.")
    for name, buf in encoder.named_buffers():
        if buf.device.type == "meta":
            raise RuntimeError(f"[checkpoint] Buffer {name!r} remained on meta device.")

    return encoder.eval().requires_grad_(False)


def _load_prototype_bank(
    prototypes_path: Path,
    metadata: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, int], float]:
    try:
        pack = torch.load(prototypes_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"[schema] Failed to load {prototypes_path}: {exc}") from exc

    if not isinstance(pack, Mapping):
        raise ValueError(
            f"[schema] Expected mapping in {prototypes_path}, got {type(pack).__name__}."
        )

    required_keys = ("prototypes", "chars", "char_to_index", "K", "thr")
    missing_pack_keys = [k for k in required_keys if k not in pack]
    if missing_pack_keys:
        raise ValueError(
            f"[schema] Missing required keys {missing_pack_keys} in {prototypes_path}."
        )

    k_val = pack["K"]
    if not _is_strict_int(k_val) or k_val < 1:
        raise ValueError(
            f"[schema] Invalid K={k_val!r} in {prototypes_path}; expected positive int."
        )
    if k_val != int(metadata["K"]):
        raise ValueError(
            f"[schema] Prototype K mismatch in {prototypes_path}: "
            f"pack K={k_val}, metadata K={metadata['K']}."
        )

    thr_raw = pack["thr"]
    if not _is_finite_real(thr_raw):
        raise ValueError(
            f"[schema] Invalid thr={thr_raw!r} in {prototypes_path}; expected finite real."
        )
    thr_val = float(thr_raw)
    meta_thr = float(metadata["threshold"])
    if not math.isclose(thr_val, meta_thr, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"[schema] Threshold mismatch in {prototypes_path}: "
            f"pack thr={thr_val}, metadata threshold={meta_thr}."
        )

    bank = pack["prototypes"]
    if not isinstance(bank, torch.Tensor) or not bank.is_floating_point():
        raise ValueError(
            f"[schema] Expected floating-point tensor 'prototypes' in {prototypes_path}."
        )
    expected_shape = tuple(int(d) for d in metadata["prototype_shape"])
    if bank.ndim != 3 or tuple(bank.shape) != expected_shape:
        raise ValueError(
            f"[shape] Prototype bank shape mismatch in {prototypes_path}: "
            f"expected {expected_shape}, got {tuple(bank.shape)}."
        )

    num_chars = expected_shape[0]
    chars_raw = pack["chars"]
    if isinstance(chars_raw, (str, bytes)) or not isinstance(chars_raw, Sequence):
        raise ValueError(
            f"[mapping] Expected sequence 'chars' in {prototypes_path}, got {type(chars_raw).__name__}."
        )
    chars_list = list(chars_raw)
    if len(chars_list) != num_chars:
        raise ValueError(
            f"[mapping] Length of chars ({len(chars_list)}) != num_characters ({num_chars}) "
            f"in {prototypes_path}."
        )
    seen_chars: set[str] = set()
    for idx, ch in enumerate(chars_list):
        if not isinstance(ch, str) or len(ch) == 0:
            raise ValueError(
                f"[mapping] Invalid character at chars[{idx}]={ch!r} in {prototypes_path}."
            )
        if ch in seen_chars:
            raise ValueError(
                f"[mapping] Duplicate character {ch!r} in chars of {prototypes_path}."
            )
        seen_chars.add(ch)

    raw_mapping = pack["char_to_index"]
    if not isinstance(raw_mapping, Mapping):
        raise ValueError(
            f"[mapping] Expected mapping 'char_to_index' in {prototypes_path}, "
            f"got {type(raw_mapping).__name__}."
        )
    if set(raw_mapping.keys()) != seen_chars:
        raise ValueError(
            f"[mapping] char_to_index keys do not match chars in {prototypes_path}."
        )
    char_to_index: dict[str, int] = {}
    for idx, ch in enumerate(chars_list):
        mapped_idx = raw_mapping[ch]
        if not _is_strict_int(mapped_idx) or mapped_idx != idx:
            raise ValueError(
                f"[mapping] Non-row-aligned char_to_index[{ch!r}]={mapped_idx!r} "
                f"(expected {idx}) in {prototypes_path}."
            )
        char_to_index[ch] = int(mapped_idx)

    bank_fp32 = bank.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(bank_fp32).all():
        raise ValueError(f"[non-finite] Non-finite values found in prototype bank {prototypes_path}.")

    norms = bank_fp32.norm(dim=-1)
    min_norm = float(norms.min().item())
    max_norm = float(norms.max().item())
    if min_norm <= 1e-6:
        raise ValueError(
            f"[non-finite] Prototype bank {prototypes_path} contains zero/near-zero norm vector "
            f"(min_norm={min_norm:.3e})."
        )

    meta_norm_min = metadata.get("prototype_norm_min")
    meta_norm_max = metadata.get("prototype_norm_max")
    if meta_norm_min is not None and meta_norm_max is not None:
        if 0.99 <= float(meta_norm_min) <= 1.01 and 0.99 <= float(meta_norm_max) <= 1.01:
            if abs(min_norm - 1.0) > 1e-3 or abs(max_norm - 1.0) > 1e-3:
                raise ValueError(
                    f"[schema] Prototype norms in {prototypes_path} deviate from metadata unit-norm "
                    f"contract: computed [{min_norm:.6f}, {max_norm:.6f}]."
                )

    normalized_bank = F.normalize(bank_fp32, dim=-1)
    return normalized_bank, char_to_index, thr_val


def load_textverifier_assets(
    *,
    model_dir: str | Path | None = None,
    repo_id: str = DEFAULT_TEXTVERIFIER_REPO,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> TextVerifierAssets:
    """Resolve, checksum-verify, validate, and load real TextVerifier Stage-3 assets."""
    files = resolve_textverifier_files(
        model_dir=model_dir,
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    digests = _parse_and_verify_manifest(files)
    metadata = _load_and_validate_metadata(files.metadata_path)
    encoder = _load_vision_encoder(files.stage3_path)
    prototypes, char_to_index, threshold = _load_prototype_bank(files.prototypes_path, metadata)

    return TextVerifierAssets(
        encoder=encoder,
        prototypes=prototypes,
        char_to_index=char_to_index,
        metadata=metadata,
        files=files,
        sha256_digests=digests,
        threshold=threshold,
    )
