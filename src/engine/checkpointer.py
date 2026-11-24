"""Dual-track checkpoint manager: rolling Accelerate full states and atomic safetensors weights."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from utils.misc import save_safetensors

if TYPE_CHECKING:
    from accelerate import Accelerator

__all__ = ["Checkpointer"]

_STATE_DIR_RE = re.compile(r"^state_iter_(\d+)$")


def _validate_step(step: Any) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}.")
    return int(step)


def _validate_name_component(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a non-empty string, got {type(value).__name__}.")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must be a non-empty string.")
    bad_chars = {"/", "\\", "\x00"}
    if os.sep:
        bad_chars.add(os.sep)
    if os.altsep:
        bad_chars.add(os.altsep)
    if cleaned in {".", ".."} or ".." in cleaned or any(ch in cleaned for ch in bad_chars):
        raise ValueError(f"{label} contains unsafe path characters: {value!r}.")
    return cleaned


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=dst.parent,
        prefix=f".{dst.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp_path)
        os.replace(tmp_path, dst)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class Checkpointer:
    """Dual-track checkpointer managing resumable Accelerate states and pure safetensors weights.

    - Track 1 (``checkpoints/state_iter_XXXXXX/``): full collective ``accelerator.save_state()``
      snapshots with integer-sorted Rank-0 rolling pruning to ``max_keep_states``.
    - Track 2 (``weights/*.safetensors``): atomic Rank-0 inference weight exports (``iter_*``,
      ``iter_*_ema``, ``latest``, ``latest_ema``, and ``best_<val_name>_<metric_name>``).
    """

    def __init__(
        self,
        work_dir: str | Path,
        max_keep_states: int | None = 3,
    ) -> None:
        if isinstance(work_dir, Path):
            if not str(work_dir).strip():
                raise ValueError("work_dir must be a non-empty path.")
            root = work_dir
        elif isinstance(work_dir, str):
            stripped = work_dir.strip()
            if not stripped:
                raise ValueError("work_dir must be a non-empty path.")
            root = Path(stripped)
        else:
            raise ValueError(
                f"work_dir must be a str or Path, got {type(work_dir).__name__}."
            )

        if max_keep_states is None:
            keep: int | None = None
        elif isinstance(max_keep_states, bool) or not isinstance(max_keep_states, int):
            raise ValueError(
                f"max_keep_states must be an integer or None, got {max_keep_states!r}."
            )
        elif max_keep_states <= 0:
            keep = None
        else:
            keep = int(max_keep_states)

        self.work_dir: Path = root
        self.checkpoints_dir: Path = self.work_dir / "checkpoints"
        self.weights_dir: Path = self.work_dir / "weights"
        self.max_keep_states: int | None = keep

    def _list_state_dirs(self) -> list[tuple[int, Path]]:
        if not self.checkpoints_dir.is_dir():
            return []
        entries: list[tuple[int, Path]] = []
        for entry in self.checkpoints_dir.iterdir():
            if not entry.is_dir():
                continue
            match = _STATE_DIR_RE.match(entry.name)
            if match is None:
                continue
            entries.append((int(match.group(1)), entry))
        entries.sort(key=lambda item: (item[0], item[1].name))
        return entries

    def save_state(self, accelerator: Accelerator, step: int) -> Path:
        """Save Track 1 full resumable Accelerate state across all ranks and prune old states."""
        step_i = _validate_step(step)
        state_dir = self.checkpoints_dir / f"state_iter_{step_i:06d}"

        if accelerator.is_main_process:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()

        # Every rank must call save_state so per-rank RNG states and collectives succeed.
        accelerator.save_state(str(state_dir))
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and self.max_keep_states is not None:
            entries = self._list_state_dirs()
            if len(entries) > self.max_keep_states:
                to_remove = entries[: len(entries) - self.max_keep_states]
                for _, old_path in to_remove:
                    shutil.rmtree(old_path)

        accelerator.wait_for_everyone()
        return state_dir

    def save_weights(
        self,
        accelerator: Accelerator,
        exported_weights: Mapping[str, Mapping[str, torch.Tensor]],
        step: int,
    ) -> dict[str, Path]:
        """Save Track 2 atomic safetensors weights (``iter_*`` and ``latest*``) on Rank 0."""
        step_i = _validate_step(step)
        if not isinstance(exported_weights, Mapping):
            raise TypeError(
                f"exported_weights must be a Mapping, got {type(exported_weights).__name__}."
            )

        saved: dict[str, Path] = {}
        if accelerator.is_main_process:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            for key, state_dict in exported_weights.items():
                if state_dict is None:
                    continue
                if not isinstance(state_dict, Mapping) or len(state_dict) == 0:
                    raise ValueError(
                        f"Exported state_dict for {key!r} must be a non-empty tensor mapping."
                    )

                if key == "net_g":
                    iter_path = self.weights_dir / f"iter_{step_i:06d}.safetensors"
                    latest_path = self.weights_dir / "latest.safetensors"
                    latest_key = "latest"
                elif key == "net_g_ema":
                    iter_path = self.weights_dir / f"iter_{step_i:06d}_ema.safetensors"
                    latest_path = self.weights_dir / "latest_ema.safetensors"
                    latest_key = "latest_ema"
                else:
                    safe_key = _validate_name_component(key, "exported_weights key")
                    iter_path = self.weights_dir / f"iter_{step_i:06d}_{safe_key}.safetensors"
                    latest_path = self.weights_dir / f"latest_{safe_key}.safetensors"
                    latest_key = f"latest_{safe_key}"

                save_safetensors(
                    state_dict,
                    iter_path,
                    metadata={"step": str(step_i), "name": str(key)},
                )
                _atomic_copy_file(iter_path, latest_path)
                saved[str(key)] = iter_path
                saved[latest_key] = latest_path

        accelerator.wait_for_everyone()
        return saved

    def save_best(
        self,
        accelerator: Accelerator,
        state_dict: Mapping[str, torch.Tensor],
        val_name: str,
        metric_name: str,
        *,
        step: int | None = None,
        metric_value: float | None = None,
    ) -> Path:
        """Save Track 2 atomic best-metric safetensors weights on Rank 0."""
        safe_val = _validate_name_component(val_name, "val_name")
        safe_metric = _validate_name_component(metric_name, "metric_name")
        best_path = self.weights_dir / f"best_{safe_val}_{safe_metric}.safetensors"

        if accelerator.is_main_process:
            if not isinstance(state_dict, Mapping) or len(state_dict) == 0:
                raise ValueError("Best checkpoint state_dict must be a non-empty tensor mapping.")
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            metadata: dict[str, str] = {
                "val_name": safe_val,
                "metric_name": safe_metric,
            }
            if step is not None:
                metadata["step"] = str(_validate_step(step))
            if metric_value is not None:
                metadata["metric_value"] = str(metric_value)
            save_safetensors(state_dict, best_path, metadata=metadata)

        accelerator.wait_for_everyone()
        return best_path

    def save_best_weights(
        self,
        accelerator: Accelerator,
        exported_weights: Mapping[str, Mapping[str, torch.Tensor]],
        val_name: str,
        metric_name: str,
        *,
        step: int | None = None,
        metric_value: float | None = None,
    ) -> dict[str, Path]:
        """Save Track 2 atomic best-metric safetensors weights for single- or multi-arch models."""
        safe_val = _validate_name_component(val_name, "val_name")
        safe_metric = _validate_name_component(metric_name, "metric_name")
        if not isinstance(exported_weights, Mapping) or len(exported_weights) == 0:
            raise ValueError(
                "exported_weights must be a non-empty mapping of component name -> state_dict."
            )

        non_null_items = [
            (str(k), sd) for k, sd in exported_weights.items() if sd is not None
        ]
        if not non_null_items:
            raise ValueError("exported_weights contains no non-null state dicts to save.")

        saved: dict[str, Path] = {}
        if accelerator.is_main_process:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            is_single_net_g = len(non_null_items) == 1 and non_null_items[0][0] == "net_g"
            for key, state_dict in non_null_items:
                if not isinstance(state_dict, Mapping) or len(state_dict) == 0:
                    raise ValueError(
                        f"Best checkpoint state_dict for {key!r} must be a non-empty tensor mapping."
                    )
                if is_single_net_g:
                    best_path = self.weights_dir / f"best_{safe_val}_{safe_metric}.safetensors"
                else:
                    safe_key = _validate_name_component(key, "exported_weights key")
                    best_path = (
                        self.weights_dir / f"best_{safe_val}_{safe_metric}_{safe_key}.safetensors"
                    )
                metadata: dict[str, str] = {
                    "val_name": safe_val,
                    "metric_name": safe_metric,
                    "name": key,
                }
                if step is not None:
                    metadata["step"] = str(_validate_step(step))
                if metric_value is not None:
                    metadata["metric_value"] = str(metric_value)
                save_safetensors(state_dict, best_path, metadata=metadata)
                saved[key] = best_path

        accelerator.wait_for_everyone()
        return saved

    def find_latest_state(self) -> Path | None:
        """Return the highest-iteration ``checkpoints/state_iter_*`` directory, or ``None``."""
        entries = self._list_state_dirs()
        if not entries:
            return None
        return entries[-1][1]

    def resolve_resume_path(self, resume: str | Path | None) -> Path | None:
        """Resolve ``trainer.resume`` (``None``, ``'auto'``, or explicit state directory path)."""
        if resume is None or resume is False:
            return None
        if isinstance(resume, bool):
            raise ValueError(
                "Ambiguous trainer.resume=True; specify 'auto', a checkpoint directory path, or null."
            )
        if not isinstance(resume, (str, Path)):
            raise ValueError(
                f"trainer.resume must be str, Path, or None, got {type(resume).__name__}: {resume!r}."
            )

        raw = str(resume).strip()
        if raw.lower() in {"", "none", "null", "false"}:
            return None
        if raw.lower() == "auto":
            return self.find_latest_state()

        explicit = Path(raw)
        if not explicit.exists():
            raise FileNotFoundError(f"Resume checkpoint directory not found: {explicit}")
        if not explicit.is_dir():
            raise ValueError(
                f"Resume path must be a checkpoint state directory, got file: {explicit} "
                "(use model.pretrain.<arch>.path for weight files)."
            )
        return explicit
