"""Differentiable TextVerifier Stage-3 per-character glyph loss (``GlyphLoss``).

Adapted from ``yqwu905/VOSR`` (``models/glyph_loss.py`` and ``models/text_losses.py``)
for SRTrainer's RGB ``[0, 1]`` tensor convention and batch metadata plumbing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from archs.textverifier import TEXTVERIFIER_IMAGE_SIZE
from losses.textverifier import DEFAULT_TEXTVERIFIER_REPO, load_textverifier_assets
from utils.text_geometry import canonical_quad, rectify_text_roi

__all__ = [
    "DEFAULT_TEXTVERIFIER_REPO",
    "canonical_quad",
    "rectify_text_roi",
    "GlyphLoss",
    "TextVerifierGlyphLoss",
]

_VALID_INPUT_RANGES: frozenset[str] = frozenset({"0_1", "-1_1"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


class GlyphLoss(nn.Module):
    """Per-character TextVerifier Stage-3 glyph loss (``1 - max cosine similarity``).

    Adapted from ``yqwu905/VOSR`` (``models/glyph_loss.py::TextVerifierGlyphLoss``).
    Sets ``needs_batch = True`` so ``VOSRModel.train_step`` automatically forwards ``batch``
    carrying crop-relative ``text_lines`` (or ``glyphs``) annotations.

    Loads verified real TextVerifier Stage-3 weights and prototype banks exclusively via
    :func:`losses.textverifier.load_textverifier_assets`. The vision tower, image projector,
    and prototype bank are frozen and permanently locked in ``eval()`` mode; gradients
    propagate via activation-checkpointed encoder forward and differentiable
    ``rectify_text_roi`` crops back into ``pred``.
    """

    needs_batch: bool = True

    def __init__(
        self,
        model_dir: str | Path | None = None,
        repo_id: str = DEFAULT_TEXTVERIFIER_REPO,
        batch_size: int = 8,
        input_range: str = "0_1",
        sync_ddp_counts: bool = True,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if not _is_positive_int(batch_size):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
        if input_range not in _VALID_INPUT_RANGES:
            raise ValueError(
                f"Unsupported input_range={input_range!r}; expected one of {sorted(_VALID_INPUT_RANGES)}."
            )
        if not isinstance(sync_ddp_counts, bool):
            raise ValueError(
                f"sync_ddp_counts must be a bool, got {type(sync_ddp_counts).__name__}."
            )
        if not isinstance(local_files_only, bool):
            raise ValueError(
                f"local_files_only must be a bool, got {type(local_files_only).__name__}."
            )

        assets = load_textverifier_assets(
            model_dir=model_dir,
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.batch_size = int(batch_size)
        self.input_range = input_range
        self.sync_ddp_counts = sync_ddp_counts
        self.encoder: nn.Module = assets.encoder.eval().requires_grad_(False)
        self.register_buffer(
            "prototypes",
            F.normalize(assets.prototypes.detach().float(), dim=-1),
            persistent=False,
        )
        self.char_to_index: dict[str, int] = dict(assets.char_to_index)
        self.asset_metadata: dict[str, Any] = dict(assets.metadata)
        self.asset_digests: dict[str, str] = dict(assets.sha256_digests)
        self.asset_source: str = assets.files.source
        self.last_counts: tuple[int, int] = (0, 0)  # (local_scored, local_unsupported)
        self.eval()

    def train(self, mode: bool = True) -> GlyphLoss:
        """Keep the frozen TextVerifier vision encoder permanently locked in eval mode."""
        return super().train(False)

    def _to_verifier_range(self, pred: torch.Tensor) -> torch.Tensor:
        x = pred.float()
        if self.input_range == "0_1":
            x = x * 2.0 - 1.0
        return x.clamp(-1.0, 1.0)

    def _score_chunk(
        self,
        images: torch.Tensor,
        entries: list[tuple[int, np.ndarray, int]],
    ) -> torch.Tensor:
        crops: list[torch.Tensor] = []
        rows: list[int] = []
        target_hw = (TEXTVERIFIER_IMAGE_SIZE, TEXTVERIFIER_IMAGE_SIZE)
        for index, quad, row in entries:
            top, right, bottom, left = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            size = (max(1, round(float(max(left, right)))), max(1, round(float(max(top, bottom)))))
            crop = rectify_text_roi(images[index : index + 1], quad, size)
            crops.append(
                F.interpolate(crop, target_hw, mode="bilinear", align_corners=False, antialias=True)
            )
            rows.append(row)

        pixels = torch.cat(crops, dim=0)
        if pixels.requires_grad and torch.is_grad_enabled():
            features = checkpoint(
                self.encoder,
                pixels,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            features = self.encoder(pixels)

        features = F.normalize(features.float(), dim=-1)
        proto_rows = self.prototypes[rows].to(device=features.device, dtype=torch.float32)
        similarities = torch.einsum("nd,nkd->nk", features, proto_rows)
        return 1.0 - similarities.max(dim=1).values

    def _collect_entries(
        self,
        text_lines: Sequence[Sequence[Mapping[str, Any]]],
    ) -> tuple[list[tuple[int, np.ndarray, int]], int]:
        entries: list[tuple[int, np.ndarray, int]] = []
        unsupported = 0
        for img_idx, per_image in enumerate(text_lines):
            for item in per_image:
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"Expected mapping in text_lines[{img_idx}], got {type(item).__name__}."
                    )
                # Accept either VOSR line dicts ({"glyphs": [...]}) or flat glyph dicts ({"char": ..., ...}).
                if "glyphs" in item:
                    glyphs_iter = item.get("glyphs") or ()
                elif "char" in item:
                    glyphs_iter = (item,)
                else:
                    glyphs_iter = ()

                for glyph in glyphs_iter:
                    if not isinstance(glyph, Mapping) or "char" not in glyph:
                        raise ValueError(
                            f"Each glyph entry must be a mapping with 'char' and 'quad'/'bbox', got {glyph!r}."
                        )
                    ch = glyph["char"]
                    row = self.char_to_index.get(ch)
                    if row is None:
                        unsupported += 1
                        continue
                    geom = glyph.get("quad", glyph.get("bbox"))
                    if geom is None:
                        raise ValueError(
                            f"Glyph {ch!r} is missing 'quad' or 'bbox' geometry."
                        )
                    quad = canonical_quad(geom, name=f"glyph {ch!r}")
                    entries.append((img_idx, quad, row))
        return entries, unsupported

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        text_lines: Sequence[Sequence[Mapping[str, Any]]] | None = None,
        batch: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        del target  # GlyphLoss compares predicted glyphs against frozen TextVerifier prototypes.

        if pred.ndim != 4 or pred.shape[1] != 3:
            raise ValueError(
                f"GlyphLoss expects 4D RGB pred tensor (B, 3, H, W), got shape {tuple(pred.shape)}."
            )

        resolved_lines = text_lines
        if resolved_lines is None and batch is not None:
            resolved_lines = batch.get("text_lines", batch.get("glyphs"))
        if resolved_lines is None:
            raise ValueError(
                "GlyphLoss requires per-image glyph annotations via `text_lines` or "
                "`batch['text_lines']` / `batch['glyphs']`. Add `text_lines` or `glyphs` to "
                "your training manifest JSON (or configure `glyphs_json` on data.sources)."
            )
        if len(resolved_lines) != pred.shape[0]:
            raise ValueError(
                f"Length of text_lines ({len(resolved_lines)}) must match batch size ({pred.shape[0]})."
            )

        entries, unsupported = self._collect_entries(resolved_lines)
        scored = len(entries)
        self.last_counts = (scored, unsupported)

        if not entries:
            # Connected zero keeps autograd / DDP backward graphs valid on glyph-free micro-batches.
            loss = pred.flatten()[:0].float().sum()
        else:
            with torch.amp.autocast(device_type=pred.device.type, enabled=False):
                images = self._to_verifier_range(pred)
                scores = [
                    self._score_chunk(images, entries[i : i + self.batch_size])
                    for i in range(0, len(entries), self.batch_size)
                ]
                loss = torch.cat(scores, dim=0).mean()

        if (
            self.sync_ddp_counts
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            world = dist.get_world_size()
            counts = torch.tensor([scored, unsupported], device=pred.device, dtype=torch.long)
            dist.all_reduce(counts)
            global_scored = counts[0].clamp_min(1)
            loss = loss * (scored * world / global_scored.to(dtype=loss.dtype))

        return loss


TextVerifierGlyphLoss = GlyphLoss
