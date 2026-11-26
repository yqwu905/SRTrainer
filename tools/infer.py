"""Single-device offline inference on a directory of PNG and JPG images."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from config import InferRootConfig, load_infer_config, register_configs
from engine.inferencer import TiledInferencer
from models import build_infer_model
from utils.image import load_image, save_image
from utils.npu import setup_npu

setup_npu()
register_configs()

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@dataclass(frozen=True)
class InferItem:
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class InferSummary:
    total_images: int
    elapsed_sec: float
    images_per_sec: float
    output_dir: Path


def collect_input_items(
    input_dir: str | Path | None, output_dir: Path, *, suffix: str, save_format: str,
) -> list[InferItem]:
    if not input_dir or not Path(input_dir).is_dir():
        raise ValueError(f"infer.input must be an existing image directory, got {input_dir!r}.")
    items = [
        InferItem(path, output_dir / f"{path.stem}{suffix}.{save_format}")
        for path in sorted(Path(input_dir).iterdir())
        if path.suffix.lower() in {".png", ".jpg"} and path.is_file()
    ]
    if not items:
        raise ValueError(f"No .png or .jpg files found in {input_dir}.")
    return items


def run_infer(cfg: InferRootConfig | DictConfig | Mapping[str, Any]) -> InferSummary:
    infer_root = load_infer_config(cfg)
    inf = infer_root.infer
    output_dir = Path(inf.output_dir)
    items = collect_input_items(
        inf.input, output_dir, suffix=inf.suffix, save_format=inf.save_format.value,
    )
    device = torch.device(inf.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_infer_model(infer_root, device=device)
    inferencer = TiledInferencer.from_config(
        model=model, scale=infer_root.scale, tile_cfg=inf.tile,
        mixed_precision=inf.mixed_precision, device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for item in items:
        image = load_image(item.input_path).unsqueeze(0)
        save_image(inferencer.infer_tensor(image), item.output_path)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return InferSummary(len(items), elapsed, len(items) / elapsed, output_dir)


@hydra.main(version_base=None, config_path=_CONFIG_DIR, config_name="infer")
def main(cfg: DictConfig) -> None:
    summary = run_infer(cfg)
    print(
        f"{summary.total_images} images in {summary.elapsed_sec:.2f}s "
        f"({summary.images_per_sec:.2f} img/s) saved to {summary.output_dir}"
    )


if __name__ == "__main__":
    main()
