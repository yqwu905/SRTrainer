"""Pure-iteration Accelerate training engine with distributed exact validation and dual-track saving."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, GradientState, PartialState
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from config.schema import KernelConfig, TrainConfig
from config.validation import load_kernel_config, load_train_config, to_resolved_dict_config
from data import (
    DistributedWeightedInfiniteSampler,
    MixedSRDataset,
    ValImageDataset,
    val_collate_fn,
)
from data.mixed_dataset import train_collate_fn
from engine.checkpointer import Checkpointer
from engine.inferencer import TiledInferencer
from engine.state import TrainState
from metrics.base import BaseMetric
from models.base_model import BaseModel, build_model
from models.ema import unwrap_model_for_ema
from utils.config import (
    flatten_config_for_tracker,
    resolve_work_dir,
    save_resolved_config,
    setup_experiment_dirs,
)
from utils.image import load_image, make_comparison_image, save_image
from utils.logging import (
    format_train_log,
    log_image_dict_to_trackers,
    log_images_to_trackers,
    setup_logger,
)

__all__ = ["Trainer"]


@dataclass
class _ValSuite:
    name: str
    dataset: ValImageDataset
    loader: Any
    metrics: dict[str, BaseMetric]
    max_save_images: int
    vis_stems: list[str]


def _filter_tracker_init_kwargs(
    raw: dict[str, dict[str, Any]],
    active_log_with: list[str],
) -> dict[str, dict[str, Any]]:
    if not raw or not active_log_with:
        return {}
    keep_all = any(item.lower() == "all" for item in active_log_with)
    active_set = {item.lower() for item in active_log_with}
    return {
        key: dict(kwargs)
        for key, kwargs in raw.items()
        if keep_all or key.lower() in active_set
    }


def _extract_kernel_cfg(degradation: Any) -> KernelConfig | None:
    if degradation is None:
        return None
    if is_dataclass(degradation) and not isinstance(degradation, type):
        kcfg = getattr(degradation, "kernel_cfg", None)
    elif isinstance(degradation, Mapping):
        kcfg = degradation.get("kernel_cfg", None)
    else:
        return None
    return load_kernel_config(kcfg)


def _reenable_srtrainer_loggers() -> None:
    """Re-enable ``srtrainer`` and ``srtrainer.*`` loggers disabled by Hydra job_logging."""
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if (name == "srtrainer" or name.startswith("srtrainer.")) and isinstance(
            obj, logging.Logger
        ):
            obj.disabled = False


def _build_vis_stems(dataset: ValImageDataset) -> list[str]:
    raw_stems = [
        (Path(sample.lq_path).stem or f"img_{idx:04d}")
        for idx, sample in enumerate(dataset.samples)
    ]
    counts = Counter(raw_stems)
    return [
        stem if counts[stem] == 1 else f"{stem}_{idx:04d}"
        for idx, stem in enumerate(raw_stems)
    ]


class Trainer:
    """Iteration-driven Accelerate training engine with distributed exact validation."""

    def __init__(self, cfg: TrainConfig | DictConfig | Mapping[str, Any]) -> None:
        # Capture existing singleton/distributed state BEFORE anything touches PartialState.
        self._had_existing_state: bool = bool(PartialState._shared_state)
        self._had_existing_dist: bool = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self._training_ended: bool = False
        self.accelerator: Accelerator | None = None

        # Step 1: Centralized Structured Config loading & fail-fast business validation
        # (no directories or Accelerate state created yet).
        config: TrainConfig = load_train_config(cfg)
        self.config: TrainConfig = config

        t_cfg = config.trainer
        d_cfg = config.data
        m_cfg = config.metric

        self.total_iters: int = t_cfg.total_iters
        self.log_interval: int = t_cfg.log_interval
        self.val_interval: int = t_cfg.val_interval
        self.save_interval: int = t_cfg.save_interval
        self.max_keep_states: int | None = (
            None if (t_cfg.max_keep_states is None or t_cfg.max_keep_states <= 0) else t_cfg.max_keep_states
        )
        self.grad_accum_steps: int = t_cfg.gradient_accumulation_steps
        self.mixed_precision: str = t_cfg.mixed_precision.value
        self.train_log_image_interval: int = t_cfg.train_log_image_interval
        self.train_max_log_images: int = t_cfg.train_max_log_images
        self.train_log_image_mode: str = t_cfg.train_log_image_mode.value
        self.val_save_visuals: bool = t_cfg.val_save_visuals
        self.val_max_save_images: int = t_cfg.val_max_save_images
        self.val_use_ema: bool = t_cfg.val_use_ema

        self.seed: int = config.seed
        self.scale: int = config.scale
        self.gt_size: int = d_cfg.gt_size
        self.primary_dataset: str | None = m_cfg.primary_dataset
        self.primary_metric: str | None = m_cfg.primary_metric

        active_log_with: list[str] | None = list(t_cfg.log_with) if t_cfg.log_with else None
        sanitized_tracker_init_kwargs = _filter_tracker_init_kwargs(
            t_cfg.tracker_init_kwargs,
            t_cfg.log_with,
        )

        # Steps 2-10: Clean up Accelerate/distributed singletons if construction fails mid-way.
        try:
            # Step 2: Multi-rank safe work_dir resolution BEFORE Accelerator(...).
            base_work_dir = Path(config.work_dir)
            self.work_dir: Path = resolve_work_dir(config)
            self.config.work_dir = str(self.work_dir)
            self.cfg: DictConfig = to_resolved_dict_config(self.config)

            # Step 3: Initialize Accelerator + ProjectConfiguration.
            project_config = ProjectConfiguration(
                project_dir=str(self.work_dir),
                logging_dir=str(self.work_dir / "trackers"),
                automatic_checkpoint_naming=False,
            )
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=t_cfg.ddp_find_unused_parameters,
            )
            self.accelerator = Accelerator(
                mixed_precision=self.mixed_precision,
                gradient_accumulation_steps=self.grad_accum_steps,
                log_with=active_log_with,
                project_config=project_config,
                kwargs_handlers=[ddp_kwargs],
            )

            # Step 4: Rank-0 experiment directory creation & config_resolved.yaml persistence.
            self.accelerator.wait_for_everyone()
            if (
                self.accelerator.is_main_process
                and base_work_dir.resolve() != self.work_dir.resolve()
                and base_work_dir.is_dir()
            ):
                # Remove empty placeholder dir created by Hydra's hydra.run.dir=${work_dir}
                # before append_timestamp resolved the actual broadcast timestamped work_dir.
                try:
                    base_work_dir.rmdir()
                except OSError:
                    pass
            self.dirs: dict[str, Path] = setup_experiment_dirs(
                self.work_dir,
                is_main_process=self.accelerator.is_main_process,
            )
            if self.accelerator.is_main_process:
                save_resolved_config(self.cfg, self.work_dir / "config_resolved.yaml")
            self.accelerator.wait_for_everyone()

            # Step 5: Rank-0 structured logger + Accelerate trackers.
            self.logger: logging.Logger = setup_logger(
                self.work_dir / "train.log",
                is_main_process=self.accelerator.is_main_process,
                name="srtrainer",
            )
            _reenable_srtrainer_loggers()

            if self.accelerator.log_with:
                flat_cfg = flatten_config_for_tracker(self.cfg)
                self.accelerator.init_trackers(
                    project_name=self.config.exp_name,
                    config=flat_cfg,
                    init_kwargs=sanitized_tracker_init_kwargs,
                )

            # Step 6: Phase-A identical cross-rank seeding before model & metric initialization.
            set_seed(self.seed, device_specific=False)

            # Step 7: Instantiate & setup model with typed TrainConfig, TrainState, Checkpointer.
            model_obj = build_model(self.config)
            model_obj.setup(self.accelerator)
            self.model: BaseModel = model_obj

            self.train_state: TrainState = TrainState()
            self.accelerator.register_for_checkpointing(self.train_state)
            self.checkpointer: Checkpointer = Checkpointer(
                work_dir=self.work_dir,
                max_keep_states=self.max_keep_states,
            )

            # Step 8: Training data pipeline (unprepared train_loader + dedicated Generator).
            has_degradation = d_cfg.degradation is not None
            kernel_cfg = _extract_kernel_cfg(d_cfg.degradation) if has_degradation else None
            self.train_dataset = MixedSRDataset(
                sources=d_cfg.sources,
                gt_size=self.gt_size,
                scale=self.scale,
                augment=d_cfg.augment,
                has_degradation=has_degradation,
                kernel_cfg=kernel_cfg,
                is_main_process=self.accelerator.is_main_process,
                logger=self.logger,
                text_crop_prob=d_cfg.text_crop_prob,
            )
            self.train_sampler = DistributedWeightedInfiniteSampler(
                source_sizes=self.train_dataset.source_sizes,
                ratios=self.train_dataset.source_ratios,
                batch_size_per_gpu=d_cfg.batch_size_per_gpu,
                rank=self.accelerator.process_index,
                world_size=self.accelerator.num_processes,
                base_seed=self.seed,
                start_micro_step=0,
            )
            pin_memory = d_cfg.pin_memory and self.accelerator.device.type == "cuda"
            persistent_workers = d_cfg.persistent_workers and d_cfg.num_workers > 0
            train_dl_gen = torch.Generator()
            train_dl_gen.manual_seed(self.seed + self.accelerator.process_index)

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=d_cfg.batch_size_per_gpu,
                sampler=self.train_sampler,
                num_workers=d_cfg.num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=train_collate_fn if self.train_dataset.has_text_annotations else None,
                generator=train_dl_gen,
            )

            # Step 9: Validation suites (val_loader prepared ONCE in __init__, metrics cached on device).
            self.val_suites: dict[str, _ValSuite] = {}
            for s_name, suite_cfg in m_cfg.val_suites.items():
                val_ds = ValImageDataset(
                    root_dir=suite_cfg.dataset.root_dir,
                    json_path=suite_cfg.dataset.json_path,
                )
                vis_stems = _build_vis_stems(val_ds)
                val_dl_gen = torch.Generator()
                val_dl_gen.manual_seed(self.seed)
                raw_val_loader = DataLoader(
                    val_ds,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=val_collate_fn,
                    pin_memory=pin_memory,
                    generator=val_dl_gen,
                )
                prepared_val_loader = self.accelerator.prepare(raw_val_loader)

                instantiated_metrics: dict[str, BaseMetric] = {}
                with self.accelerator.main_process_first():
                    for m_name, metric_target_cfg in suite_cfg.metrics.items():
                        metric = hydra.utils.instantiate(metric_target_cfg)
                        if not isinstance(metric, BaseMetric):
                            raise TypeError(
                                f"Instantiated metric {s_name}.{m_name} must be a BaseMetric, "
                                f"got {type(metric).__name__}."
                            )
                        metric.to(self.accelerator.device)
                        instantiated_metrics[m_name] = metric

                suite_max_vis = (
                    self.val_max_save_images
                    if suite_cfg.max_save_images is None
                    else suite_cfg.max_save_images
                )
                self.val_suites[s_name] = _ValSuite(
                    name=s_name,
                    dataset=val_ds,
                    loader=prepared_val_loader,
                    metrics=instantiated_metrics,
                    max_save_images=suite_max_vis,
                    vis_stems=vis_stems,
                )

            # Step 10: Phase-B per-rank seeding & optional resume restoration.
            set_seed(self.seed, device_specific=True)
            resume_cfg = t_cfg.resume
            resume_path = self.checkpointer.resolve_resume_path(resume_cfg)
            if resume_path is None:
                if resume_cfg is not None and resume_cfg.strip().lower() == "auto":
                    self.logger.warning(
                        "[Resume] trainer.resume='auto' requested, but no checkpoint state was "
                        "found under %s; starting fresh from step 0.",
                        self.checkpointer.checkpoints_dir,
                    )
            else:
                self.accelerator.load_state(str(resume_path))
                if self.accelerator.step % self.grad_accum_steps != 0:
                    self.accelerator.step = self.train_state.global_step * self.grad_accum_steps
                start_micro_step = self.train_state.global_step * self.grad_accum_steps
                self.train_sampler.set_start_micro_step(start_micro_step)
                self.logger.info(
                    "[Resume] Restored state from %s at global_step=%d (start_micro_step=%d).",
                    resume_path,
                    self.train_state.global_step,
                    start_micro_step,
                )

            self.accelerator.wait_for_everyone()
        except BaseException:
            self._cleanup_on_init_failure()
            raise

    def _cleanup_on_init_failure(self) -> None:
        if self.accelerator is not None and not self._training_ended:
            self._training_ended = True
            try:
                self.accelerator.end_training()
            except Exception:
                pass
        if (
            not self._had_existing_dist
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()
        if not self._had_existing_state:
            AcceleratorState._reset_state(reset_partial_state=True)
            GradientState._reset_state()

    def _flush_logger(self) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        for handler in list(logger.handlers):
            try:
                handler.flush()
            except Exception:
                pass

    def _get_mem_gb(self) -> float:
        assert self.accelerator is not None
        if self.accelerator.device.type == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated(self.accelerator.device)) / (1024.0**3)
        return 0.0

    def _save_checkpoint(self, step: int) -> None:
        assert self.accelerator is not None
        self.checkpointer.save_state(self.accelerator, step)
        exported = self.model.export_weights(self.accelerator)
        self.checkpointer.save_weights(self.accelerator, exported, step)

    def _should_log_train_images(self, step: int) -> bool:
        return (
            self.accelerator is not None
            and self.accelerator.is_main_process
            and bool(self.accelerator.trackers)
            and self.train_log_image_interval > 0
            and self.train_max_log_images > 0
            and (step % self.train_log_image_interval == 0 or step == self.total_iters)
        )

    def _log_train_images(self, step: int) -> None:
        assert self.accelerator is not None
        visuals = self.model.pop_train_visuals()
        if not visuals:
            return
        if not self.accelerator.is_main_process or not self.accelerator.trackers:
            return

        lq = visuals.get("lq")
        gt = visuals.get("gt")
        pred = visuals.get("pred")
        if (
            not isinstance(lq, torch.Tensor)
            or not isinstance(gt, torch.Tensor)
            or not isinstance(pred, torch.Tensor)
            or lq.ndim != 4
            or gt.ndim != 4
            or pred.ndim != 4
        ):
            return

        num_images = min(
            self.train_max_log_images,
            int(lq.shape[0]),
            int(gt.shape[0]),
            int(pred.shape[0]),
        )
        if num_images <= 0:
            return

        lq_clean = torch.nan_to_num(
            lq[:num_images].detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        pred_clean = torch.nan_to_num(
            pred[:num_images].detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        gt_clean = torch.nan_to_num(
            gt[:num_images].detach().to(device="cpu", dtype=torch.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        image_payload: dict[str, torch.Tensor] = {}
        if self.train_log_image_mode == "comparison":
            for idx in range(num_images):
                image_payload[f"train/lq_pred_gt/sample_{idx:02d}"] = make_comparison_image(
                    lq=lq_clean[idx],
                    sr=pred_clean[idx],
                    gt=gt_clean[idx],
                )
        else:
            for idx in range(num_images):
                image_payload[f"train/sample_{idx:02d}/lq"] = lq_clean[idx]
                image_payload[f"train/sample_{idx:02d}/pred"] = pred_clean[idx]
                image_payload[f"train/sample_{idx:02d}/gt"] = gt_clean[idx]

        log_image_dict_to_trackers(self.accelerator, image_payload, step=step)

    def validate(self, step: int) -> dict[str, dict[str, float]]:
        """Run distributed exact validation across all ``metric.val_suites`` using ``TiledInferencer``."""
        assert self.accelerator is not None
        if not self.val_suites:
            return {}

        infer_net = unwrap_model_for_ema(self.model.get_infer_network(use_ema=self.val_use_ema))
        was_training = infer_net.training
        infer_net.eval()

        inferencer = TiledInferencer.from_config(
            model=infer_net,
            scale=self.scale,
            tile_cfg=self.config.trainer.val_tile,
            gt_size=self.gt_size,
            mixed_precision=self.mixed_precision,
            device=self.accelerator.device,
        )

        results: dict[str, dict[str, float]] = {}
        tracker_metrics: dict[str, float] = {}
        metric_dtype = (
            torch.float32 if self.accelerator.device.type in {"mps", "npu"} else torch.float64
        )

        try:
            for suite_name, suite in self.val_suites.items():
                metric_numerators: dict[str, float] = {m_name: 0.0 for m_name in suite.metrics}
                metric_denominators: dict[str, float] = {m_name: 0.0 for m_name in suite.metrics}
                local_vis_cache: dict[int, torch.Tensor] = {}

                for metric in suite.metrics.values():
                    if hasattr(metric, "clear_sample_cache"):
                        metric.clear_sample_cache()

                try:
                    for batch in suite.loader:
                        lq = batch["lq"]
                        gt = batch.get("gt")
                        meta = batch["meta"]

                        pred = inferencer.infer_tensor(lq)
                        if gt is not None:
                            gt = gt.to(device=pred.device, dtype=torch.float32)

                        for m_name, metric in suite.metrics.items():
                            raw_stats = metric.compute_stats(pred, target=gt, meta=meta)
                            if (
                                not isinstance(raw_stats, (tuple, list))
                                or len(raw_stats) != 2
                                or isinstance(raw_stats[0], bool)
                                or isinstance(raw_stats[1], bool)
                            ):
                                raise ValueError(
                                    f"Metric {suite_name}.{m_name}.compute_stats() must return a "
                                    f"(numerator, denominator) numeric 2-tuple, got {raw_stats!r}."
                                )
                            num_val = float(raw_stats[0])
                            den_val = float(raw_stats[1])
                            if not math.isfinite(den_val) or den_val < 0.0:
                                raise ValueError(
                                    f"Metric {suite_name}.{m_name}.compute_stats() returned "
                                    f"invalid denominator {den_val!r}; expected finite >= 0."
                                )
                            stats_tensor = torch.tensor(
                                [[num_val, den_val]],
                                dtype=metric_dtype,
                                device=self.accelerator.device,
                            )
                            gathered = self.accelerator.gather_for_metrics(stats_tensor)
                            gathered_cpu = gathered.detach().cpu()
                            if gathered_cpu.ndim != 2 or gathered_cpu.shape[1] != 2:
                                raise RuntimeError(
                                    f"Expected gathered stats shape [N, 2] for {suite_name}.{m_name}, "
                                    f"got {tuple(gathered_cpu.shape)}."
                                )
                            for row_num, row_den in gathered_cpu.tolist():
                                metric_numerators[m_name] += float(row_num)
                                metric_denominators[m_name] += float(row_den)

                        gs = self.accelerator.gradient_state
                        is_padded_tail = (
                            self.accelerator.num_processes > 1
                            and gs.end_of_dataloader
                            and gs.remainder > 0
                            and self.accelerator.process_index >= gs.remainder
                        )
                        sample_idx = int(meta.get("index", 0))
                        if (
                            self.val_save_visuals
                            and suite.max_save_images > 0
                            and (not is_padded_tail)
                            and 0 <= sample_idx < suite.max_save_images
                        ):
                            cmp_img = make_comparison_image(lq=lq, sr=pred, gt=gt)
                            vis_stem = suite.vis_stems[sample_idx]
                            save_path = (
                                self.dirs["visuals"]
                                / f"iter_{step:06d}"
                                / suite_name
                                / f"{vis_stem}_cmp.png"
                            )
                            save_image(cmp_img, save_path)
                            if self.accelerator.is_main_process and self.accelerator.trackers:
                                local_vis_cache[sample_idx] = cmp_img.detach().cpu()
                finally:
                    for metric in suite.metrics.values():
                        if hasattr(metric, "clear_sample_cache"):
                            metric.clear_sample_cache()
                    if hasattr(suite.loader, "end"):
                        suite.loader.end()
                    if hasattr(suite.loader, "reset"):
                        suite.loader.reset()
                    gs = self.accelerator.gradient_state
                    if gs.in_dataloader and gs.active_dataloader is suite.loader:
                        gs._remove_dataloader(suite.loader)

                self.accelerator.wait_for_everyone()

                # Deterministic Rank-0 tracker image logging after all ranks finish saving visuals.
                if (
                    self.accelerator.is_main_process
                    and self.accelerator.trackers
                    and self.val_save_visuals
                    and suite.max_save_images > 0
                ):
                    vis_limit = min(len(suite.dataset), suite.max_save_images)
                    for idx in range(vis_limit):
                        vis_stem = suite.vis_stems[idx]
                        img_to_log = local_vis_cache.get(idx)
                        if img_to_log is None:
                            save_path = (
                                self.dirs["visuals"]
                                / f"iter_{step:06d}"
                                / suite_name
                                / f"{vis_stem}_cmp.png"
                            )
                            if save_path.is_file():
                                img_to_log = load_image(save_path)
                        if img_to_log is not None:
                            log_images_to_trackers(
                                self.accelerator,
                                tag=f"val/{suite_name}/{vis_stem}",
                                image=img_to_log,
                                step=step,
                            )

                suite_results: dict[str, float] = {}
                for m_name in suite.metrics:
                    total_num = metric_numerators[m_name]
                    total_den = metric_denominators[m_name]
                    mean_val = float(total_num / total_den) if total_den > 0.0 else float("nan")
                    suite_results[m_name] = mean_val
                    tracker_metrics[f"val/{suite_name}/{m_name}"] = mean_val
                results[suite_name] = suite_results

                if suite_results:
                    summary = " | ".join(f"{k}: {v:.4f}" for k, v in suite_results.items())
                    self.logger.info(
                        "[Val Iter %06d] %s | %s (N=%d)",
                        step,
                        suite_name,
                        summary,
                        len(suite.dataset),
                    )
                else:
                    self.logger.info(
                        "[Val Iter %06d] %s | (visuals only, N=%d)",
                        step,
                        suite_name,
                        len(suite.dataset),
                    )

            if tracker_metrics:
                self.accelerator.log(tracker_metrics, step=step)

            if self.primary_dataset is not None and self.primary_metric is not None:
                current_val = results[self.primary_dataset][self.primary_metric]
                higher_is_better = bool(
                    self.val_suites[self.primary_dataset].metrics[self.primary_metric].higher_is_better
                )
                best_key = f"{self.primary_dataset}/{self.primary_metric}"
                prev_best = self.train_state.best_metrics.get(best_key)
                is_new_best = self.train_state.update_best(
                    best_key,
                    current_val,
                    higher_is_better=higher_is_better,
                )
                if is_new_best:
                    best_weights = self.model.export_infer_weights(
                        self.accelerator,
                        use_ema=self.val_use_ema,
                    )
                    saved_best = self.checkpointer.save_best_weights(
                        self.accelerator,
                        best_weights,
                        self.primary_dataset,
                        self.primary_metric,
                        step=step,
                        metric_value=current_val,
                    )
                    prev_str = "None" if prev_best is None else f"{prev_best:.4f}"
                    saved_names = ", ".join(p.name for p in saved_best.values())
                    self.logger.info(
                        "[Best] Updated %s: %s -> %.4f (saved %s)",
                        best_key,
                        prev_str,
                        current_val,
                        saved_names or f"best_{self.primary_dataset}_{self.primary_metric}.safetensors",
                    )
        finally:
            if was_training:
                infer_net.train(True)

        self.accelerator.wait_for_everyone()
        return results

    def fit(self) -> TrainState:
        """Execute the pure-iteration training loop until ``train_state.global_step == total_iters``."""
        assert self.accelerator is not None
        try:
            start_step = self.train_state.global_step
            if start_step >= self.total_iters:
                self.logger.info(
                    "[Train] Already at global_step=%d >= total_iters=%d; skipping training loop.",
                    start_step,
                    self.total_iters,
                )
                self.accelerator.wait_for_everyone()
                return self.train_state

            self.logger.info(
                "[Train] Starting training from step %d to %d (grad_accum=%d, processes=%d, mp=%s, work_dir=%s)",
                start_step,
                self.total_iters,
                self.grad_accum_steps,
                self.accelerator.num_processes,
                self.mixed_precision,
                self.work_dir,
            )

            train_iter = iter(self.train_loader)
            scalar_buf: dict[str, list[float]] = {}
            window_steps = 0
            window_start = time.perf_counter()
            steps_run = 0
            last_val_step: int | None = None
            last_save_step: int | None = None

            while self.train_state.global_step < self.total_iters:
                batch = next(train_iter)
                batch = self.model.preprocess_batch(batch)
                next_step = self.train_state.global_step + 1
                capture_train_visuals = self._should_log_train_images(next_step)
                self.model.set_capture_train_visuals(
                    capture_train_visuals,
                    max_images=self.train_max_log_images if capture_train_visuals else 0,
                )
                step_scalars, stepped = self.model.train_step(batch, self.train_state.global_step)

                if self.log_interval > 0:
                    for key, val in step_scalars.items():
                        scalar_buf.setdefault(key, []).append(float(val))

                if not stepped:
                    continue

                self.train_state.global_step += 1
                step = self.train_state.global_step
                steps_run += 1
                window_steps += 1

                # 1) Periodic training scalar logging.
                if self.log_interval > 0 and (
                    step % self.log_interval == 0 or step == self.total_iters
                ):
                    elapsed = max(time.perf_counter() - window_start, 1e-9)
                    it_per_sec = window_steps / elapsed
                    remaining = max(0, self.total_iters - step)
                    eta_sec = remaining / it_per_sec if it_per_sec > 0.0 else float("inf")
                    mem_gb = self._get_mem_gb()

                    avg_scalars = {
                        k: float(sum(vals) / len(vals))
                        for k, vals in scalar_buf.items()
                        if vals
                    }
                    lr_dict = self.model.get_lr_dict()

                    log_line = format_train_log(
                        step=step,
                        total_iters=self.total_iters,
                        lr_dict=lr_dict,
                        scalars=avg_scalars,
                        it_per_sec=it_per_sec,
                        mem_gb=mem_gb,
                        eta_sec=eta_sec,
                    )
                    self.logger.info(log_line)

                    tracker_payload: dict[str, float] = dict(avg_scalars)
                    for lr_k, lr_v in lr_dict.items():
                        t_key = (
                            lr_k if (lr_k == "lr" or lr_k.startswith("lr/")) else f"lr/{lr_k}"
                        )
                        tracker_payload[t_key] = float(lr_v)
                    self.accelerator.log(tracker_payload, step=step)

                    scalar_buf.clear()
                    window_steps = 0
                    window_start = time.perf_counter()

                # 2) Periodic training visuals, validation, and checkpointing (paused from it/s).
                pause_start = time.perf_counter()
                did_pause = False

                if self._should_log_train_images(step):
                    self._log_train_images(step)
                    did_pause = True
                else:
                    self.model.pop_train_visuals()

                # 3) Periodic / final validation BEFORE checkpoint save so best_metrics & RNG stay synced.
                should_val = bool(self.val_suites) and (
                    (self.val_interval > 0 and step % self.val_interval == 0)
                    or step == self.total_iters
                )
                if should_val:
                    self.validate(step)
                    last_val_step = step
                    did_pause = True

                # 4) Periodic / final dual-track checkpoint save.
                should_save = (
                    (self.save_interval > 0 and step % self.save_interval == 0)
                    or step == self.total_iters
                )
                if should_save:
                    self._save_checkpoint(step)
                    last_save_step = step
                    did_pause = True

                if did_pause:
                    window_start += time.perf_counter() - pause_start

            # End-of-loop defensive fallback (no-op when step == total_iters already ran val & save).
            if steps_run > 0:
                final_step = self.train_state.global_step
                if self.val_suites and last_val_step != final_step:
                    self.validate(final_step)
                    last_val_step = final_step
                if last_save_step != final_step:
                    self._save_checkpoint(final_step)
                    last_save_step = final_step

            self.accelerator.wait_for_everyone()
            self.logger.info(
                "[Train] Finished training at step %d/%d.",
                self.train_state.global_step,
                self.total_iters,
            )
            return self.train_state
        finally:
            self.model.set_capture_train_visuals(False, max_images=0)
            self._flush_logger()
            if not self._training_ended:
                self._training_ended = True
                self.accelerator.end_training()
            if (
                not self._had_existing_dist
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.destroy_process_group()
