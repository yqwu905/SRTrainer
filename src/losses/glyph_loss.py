"""Differentiable TextVerifier Stage-3 per-character glyph loss (``GlyphLoss``).

Adapted from ``yqwu905/VOSR`` (``models/glyph_loss.py`` and ``models/text_losses.py``)
for SRTrainer's RGB ``[0, 1]`` tensor convention and batch metadata plumbing.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "DEFAULT_TEXTVERIFIER_REPO",
    "DEFAULT_TEXTVERIFIER_REVISION",
    "canonical_quad",
    "rectify_text_roi",
    "GlyphLoss",
    "TextVerifierGlyphLoss",
]

DEFAULT_TEXTVERIFIER_REPO = "ATH-MaaS/TextVerifier-CJK-CLIP"
DEFAULT_TEXTVERIFIER_REVISION = "adc233a117cbb039842b245b5acf84e0143a5f11"
_VALID_INPUT_RANGES: frozenset[str] = frozenset({"0_1", "-1_1"})
_OFFLINE_TEST_CHARS: tuple[str, ...] = (
    "中",
    "文",
    "春",
    "夏",
    "秋",
    "冬",
    "山",
    "河",
    "日",
    "月",
    "A",
    "B",
    "C",
    "0",
    "1",
    "2",
)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    return (
        abs(
            float(
                np.sum(
                    pts[:, 0] * np.roll(pts[:, 1], -1) - pts[:, 1] * np.roll(pts[:, 0], -1)
                )
            )
        )
        * 0.5
    )


def canonical_quad(value: Any, name: str = "glyph") -> np.ndarray:
    """Validate and canonicalize an xyxy AABB, 8-float quad, or 4x2 cyclic TL/TR/BR/BL quad."""
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}: expected finite xyxy AABB or four cyclic (x, y) corners."
        ) from exc

    if points.shape == (4,):
        left, top, right, bottom = points
        if not np.isfinite(points).all() or right <= left or bottom <= top:
            raise ValueError(f"{name}: AABB must be finite with x1 > x0 and y1 > y0.")
        points = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float64,
        )
    elif points.shape == (8,):
        points = points.reshape(4, 2)

    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{name}: expected finite xyxy AABB or four cyclic (x, y) corners.")

    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 1e-8) or np.all(cross < -1e-8)) or _polygon_area(points) < 1e-6:
        raise ValueError(f"{name}: quad must be convex, non-degenerate, and cyclic (not crossed).")
    if np.all(cross < 0):
        points = points[[0, 3, 2, 1]]
    return points


def rectify_text_roi(
    images: torch.Tensor,
    quad: Any,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Differentiable perspective crop from cyclic 4-corner geometry.

    Solves the constant 8x8 homography on CPU in float64 (device-agnostic), then samples via
    ``F.grid_sample`` so gradients propagate cleanly back to ``images``.
    """
    if images.ndim != 4:
        raise ValueError(f"rectify_text_roi expects 4D NCHW images, got shape {tuple(images.shape)}.")
    pts = canonical_quad(quad, "glyph ROI")
    out_h, out_w = output_size
    if not _is_positive_int(out_h) or not _is_positive_int(out_w):
        raise ValueError(
            f"output_size must contain positive integers (H, W), got {output_size!r}."
        )

    matrix: list[list[float]] = []
    rhs: list[float] = []
    for (u, v), (x, y) in zip(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), pts, strict=True):
        matrix.extend(
            (
                [u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x],
                [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y],
            )
        )
        rhs.extend((float(x), float(y)))

    try:
        h = torch.linalg.solve(
            torch.tensor(matrix, dtype=torch.float64),
            torch.tensor(rhs, dtype=torch.float64),
        )
    except RuntimeError as exc:
        raise ValueError("Glyph quad does not define a stable perspective transform.") from exc

    transform = torch.cat((h, torch.ones(1, dtype=torch.float64))).reshape(3, 3)
    if not torch.isfinite(transform).all():
        raise ValueError("Glyph quad produced a non-finite perspective transform.")

    transform = transform.to(device=images.device, dtype=torch.float32)
    ys = (torch.arange(out_h, device=images.device, dtype=torch.float32) + 0.5) / out_h
    xs = (torch.arange(out_w, device=images.device, dtype=torch.float32) + 0.5) / out_w
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    source = torch.stack((uu, vv, torch.ones_like(uu)), dim=-1) @ transform.T
    denom = source[..., 2:]
    if (denom.abs() < 1e-7).any():
        raise ValueError("Glyph quad crosses the perspective horizon.")
    source_xy = source[..., :2] / denom
    grid = source_xy / source_xy.new_tensor([images.shape[-1], images.shape[-2]]) * 2.0 - 1.0
    return F.grid_sample(
        images.float(),
        grid[None].expand(images.shape[0], -1, -1, -1),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )


def _resolve_textverifier_files(
    model_dir: str | Path | None,
    repo_id: str,
    revision: str | None,
) -> tuple[Path, Path]:
    stage3_name = "textverifier_stage3.pth"
    proto_name = "textverifier_k10_q90.pt"

    if model_dir is not None:
        root = Path(model_dir)
        s3 = root / stage3_name
        pr = root / proto_name
        if s3.is_file() and pr.is_file():
            return s3, pr
        if root.exists() and not root.is_dir():
            raise ValueError(f"model_dir must be a directory path, got file: {root}")
        # If an explicit non-repo local directory is missing weights, download into it if repo_id given.
        if not repo_id:
            raise FileNotFoundError(
                f"TextVerifier weights ({stage3_name}, {proto_name}) not found under {root}."
            )

    from huggingface_hub import hf_hub_download

    dl_kwargs: dict[str, Any] = {}
    if revision is not None:
        dl_kwargs["revision"] = revision
    if model_dir is not None:
        dl_kwargs["local_dir"] = str(model_dir)

    s3_path = Path(hf_hub_download(repo_id=repo_id, filename=stage3_name, **dl_kwargs))
    pr_path = Path(hf_hub_download(repo_id=repo_id, filename=proto_name, **dl_kwargs))
    return s3_path, pr_path


def _load_pretrained_textverifier(
    model_dir: str | Path | None,
    repo_id: str,
    revision: str | None,
) -> tuple[nn.Module, torch.Tensor, dict[str, int]]:
    import timm

    stage3_path, proto_path = _resolve_textverifier_files(model_dir, repo_id, revision)
    state = torch.load(stage3_path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError(f"Unexpected TextVerifier checkpoint format in {stage3_path}.")

    backbone = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        img_size=128,
        patch_size=16,
        num_classes=0,
        global_pool="avg",
    )
    dim = int(backbone.embed_dim)
    projector = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(dim, dim, bias=False)),
                ("ln", nn.LayerNorm(dim)),
                ("act", nn.GELU()),
                ("fc2", nn.Linear(dim, dim, bias=False)),
            ]
        )
    )
    for module, prefix in ((backbone, "image_encoder.backbone."), (projector, "image_projector.")):
        sub_state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        module.load_state_dict(sub_state, strict=True)

    pack = torch.load(proto_path, map_location="cpu", weights_only=True)
    bank = pack["prototypes"]
    mapping = dict(pack["char_to_index"])
    k_val = int(pack["K"])
    chars = pack["chars"]
    if (
        bank.shape != (len(mapping), k_val, dim)
        or k_val < 1
        or len(chars) != len(mapping)
        or any(mapping.get(ch) != idx for idx, ch in enumerate(chars))
    ):
        raise ValueError(
            "TextVerifier prototype bank dimensions or character indices are inconsistent."
        )
    return nn.Sequential(backbone, projector), bank, mapping


def _build_offline_textverifier(
    embed_dim: int = 32,
    num_prototypes: int = 4,
) -> tuple[nn.Module, torch.Tensor, dict[str, int]]:
    """Lightweight deterministic offline encoder + prototype bank for network-free unit tests."""
    gen = torch.Generator().manual_seed(0)
    encoder = nn.Sequential(
        nn.Conv2d(3, embed_dim, kernel_size=4, stride=4, bias=False),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    mapping = {ch: idx for idx, ch in enumerate(_OFFLINE_TEST_CHARS)}
    prototypes = torch.randn((len(mapping), num_prototypes, embed_dim), generator=gen)
    return encoder, prototypes, mapping


class GlyphLoss(nn.Module):
    """Per-character TextVerifier Stage-3 glyph loss (``1 - max cosine similarity``).

    Adapted from ``yqwu905/VOSR`` (``models/glyph_loss.py::TextVerifierGlyphLoss``).
    Sets ``needs_batch = True`` so ``SRModel.train_step`` automatically forwards ``batch``
    carrying crop-relative ``text_lines`` (or ``glyphs``) annotations.

    The vision tower, image projector, and prototype bank are frozen and permanently locked in
    ``eval()`` mode; gradients propagate via activation-checkpointed encoder forward and
    differentiable ``rectify_text_roi`` crops back into ``pred``.
    """

    needs_batch: bool = True

    def __init__(
        self,
        model_dir: str | Path | None = None,
        repo_id: str = DEFAULT_TEXTVERIFIER_REPO,
        revision: str | None = DEFAULT_TEXTVERIFIER_REVISION,
        batch_size: int = 8,
        pretrained: bool = True,
        input_range: str = "0_1",
        sync_ddp_counts: bool = True,
        encoder: nn.Module | None = None,
        prototypes: torch.Tensor | None = None,
        char_to_index: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        if not _is_positive_int(batch_size):
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
        if input_range not in _VALID_INPUT_RANGES:
            raise ValueError(
                f"Unsupported input_range={input_range!r}; expected one of {sorted(_VALID_INPUT_RANGES)}."
            )
        if not isinstance(pretrained, bool):
            raise ValueError(f"pretrained must be a bool, got {type(pretrained).__name__}.")
        if not isinstance(sync_ddp_counts, bool):
            raise ValueError(
                f"sync_ddp_counts must be a bool, got {type(sync_ddp_counts).__name__}."
            )

        injected = (encoder is not None, prototypes is not None, char_to_index is not None)
        if any(injected):
            if not all(injected):
                raise ValueError(
                    "encoder, prototypes, and char_to_index must all be provided together."
                )
            assert encoder is not None and prototypes is not None and char_to_index is not None
            enc_mod = encoder
            proto_tensor = prototypes
            char_map = dict(char_to_index)
        elif pretrained:
            enc_mod, proto_tensor, char_map = _load_pretrained_textverifier(
                model_dir=model_dir,
                repo_id=repo_id,
                revision=revision,
            )
        else:
            enc_mod, proto_tensor, char_map = _build_offline_textverifier()

        if proto_tensor.ndim != 3 or proto_tensor.shape[0] != len(char_map) or proto_tensor.shape[1] < 1:
            raise ValueError(
                f"Expected prototypes shape (len(char_to_index)={len(char_map)}, K>=1, D), "
                f"got {tuple(proto_tensor.shape)}."
            )

        self.batch_size = int(batch_size)
        self.pretrained = pretrained
        self.input_range = input_range
        self.sync_ddp_counts = sync_ddp_counts
        self.encoder: nn.Module = enc_mod.eval().requires_grad_(False)
        self.register_buffer(
            "prototypes",
            F.normalize(proto_tensor.detach().float(), dim=-1),
            persistent=False,
        )
        self.char_to_index: dict[str, int] = char_map
        self.last_counts: tuple[int, int] = (0, 0)  # (scored, unsupported)
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
        for index, quad, row in entries:
            top, right, bottom, left = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            size = (max(1, round(float(max(left, right)))), max(1, round(float(max(top, bottom)))))
            crop = rectify_text_roi(images[index : index + 1], quad, size)
            crops.append(
                F.interpolate(crop, (128, 128), mode="bilinear", align_corners=False, antialias=True)
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
