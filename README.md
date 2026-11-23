# SRTrainer

面向图像超分与底层视觉复原的模块化训练与推理框架，基于 **PyTorch + Hugging Face Accelerate + Hydra** 构建。

详细架构设计与全流程说明请参见 [docs/design.md](docs/design.md)。

## 核心特性

- **分层训练引擎 + Accelerate**：`Trainer` 统一管理 `Accelerator`、纯 Iteration 驱动循环、多验证集分布式精确评估（`gather_for_metrics`）、双轨 Checkpoint 与多 Tracker 日志；`BaseModel` / `SRModel` 封装网络、FP32 `ModelEMA`、任意可组合 Loss 字典与单步更新（`lr_scheduler` 不经 `prepare()`，避免多卡倍速步进）。
- **纯 Hydra `_target_` 分级配置与多级继承**：按 `arch / loss / model / data / optim / metric / trainer / infer` 拆分原子配置组，在 `configs/experiment/` 中通过 `# @package _global_` 多级继承并快速覆写局部参数；包级 `__init__.py` 提供短路径重导出。
- **内置网络与损失函数**：`src/archs/` 提供单步生成复原/超分网络 `archs.VOSR2OneStep`（含冻结 Qwen-Image 2D VAE 与 DINOv2-ViT-L/14，默认 `scale=1` 无上采样复原，也支持任意正整数倍率，权重与官方 VOSR2 checkpoint 键完全对齐）；`src/losses/` 内置 `L1Loss`、`LPIPSLoss`、基于冻结 PP-OCRv5 的 `OCRKLLoss` 与基于冻结 TextVerifier Stage-3 的 `GlyphLoss`，同时支持通过 Hydra `_target_` 直接接入任意自定义网络与损失函数。
- **统一 JSON 清单 + 样本级混合与 GPU 掩码 Real-ESRGAN 退化**：
  - 仅读取独立图像文件（`.jpg/.png/.jpeg/.webp`），`gt_only` 与 `paired` 统一使用字典列表 JSON，自动根据样本是否含有效 `lq` 判定模式（配置中无需 `mode`）；支持在清单内嵌 `text_lines` / `glyphs` 或在数据源上挂载 VOSR `glyphs_json` 以启用 `GlyphLoss`；
  - `MixedSRDataset` + `DistributedWeightedInfiniteSampler` 严格按 `ratio` 跨数据源采样并支持 O(1) 断点续训；
  - `RealESRGANDegrader` 在 GPU FP32 下仅对 `gt_only` 子集执行二阶模糊/缩放/噪声/DiffJPEG/Sinc 退化，与真实 `paired` 样本在同 Batch 平滑混合。
- **验证集内聚于 `metric` 配置组**：每个验证套件在 `configs/metric/` 中独立绑定数据集与所需指标，并将 JSON 附加字段（如 `label` / `text_lines` / `annotation_size`）完整透传到 `BaseMetric.compute(pred, target, meta)` 与 `BaseMetric.compute_stats(pred, target, meta)`；内置 BasicSR/MATLAB BT.601 对齐的 `PSNRMetric`、`SSIMMetric`、`LPIPSMetric`，以及基于 PaddleOCR 3.x 的文本识别指标 `OCRAccuracyMetric`（OCR-A）、`CERMetric`（CER）与 `OneMinusNEDMetric`（1-NED）。
- **双轨 Checkpoint**：滚动保存完整续训状态（`checkpoints/state_iter_XXXXXX/`，支持 `resume=auto`）+ 原子导出纯推理权重（`weights/*.safetensors`，含 `iter_*`、`_ema`、`latest`、`best_*`）并持久化 `config_resolved.yaml`。
- **共享 `TiledInferencer` + `tools/infer.py` 分布式推理**：支持实验目录自动还原与独立 `infer.yaml` 两种模式；支持按训练输入尺寸自动推导 `tile_size`（`gt_size // scale`）与可配置 `overlap` 羽化无缝融合；支持单卡与 Accelerate 多卡分布式离线推理。

## 安装

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境（固定 `torch==2.9.0`）：

```bash
# 同步项目及开发依赖（含 pytest / ruff）
uv sync

# 可选：安装额外实验看板依赖（wandb / mlflow / swanlab）
uv sync --extra trackers

# 可选：安装文本 OCR 评测依赖（PaddleOCR 3.x + OpenCV，用于 OCR-A / CER / 1-NED）
uv sync --extra ocr
```

## JSON 数据清单格式

所有数据清单统一采用相对路径字典列表，配合配置中的 `root_dir` 解析为绝对路径：

- **训练 `gt_only`（无 `lq` 字段，自动走 GPU 在线退化）**：
  ```json
  [
    {"gt": "DIV2K_train_HR/0001.png"},
    {"gt": "DIV2K_train_HR/0002.png"}
  ]
  ```
- **训练 `paired`（含 `lq` 字段，自动跳过退化；可选 `text_lines` / `glyphs` 用于 `GlyphLoss`）**：
  ```json
  [
    {
      "gt": "hr/0001.png",
      "lq": "lr_x4/0001.png",
      "text_lines": [
        {
          "transcript": "中文",
          "bbox": [12, 16, 75, 48],
          "glyphs": [
            {"char": "中", "bbox": [13, 16, 41, 47]},
            {"char": "文", "bbox": [46, 17, 75, 48]}
          ]
        }
      ]
    },
    {"gt": "hr/0002.png", "lq": "lr_x4/0002.png"}
  ]
  ```
- **验证集清单（在 `configs/metric/` 中配置，`gt` 可选，其余字段自动透传到 `meta`；OCR 指标支持单行整图 `label` 与多文本框 `text_lines` 两种标注模式）**：
  ```json
  [
    {"lq": "lr/0001.png", "gt": "hr/0001.png", "label": "HELLO 2026"},
    {
      "lq": "lr/0002.png",
      "annotation_size": [256, 128],
      "text_lines": [
        {"transcript": "SRTRAINER", "bbox": [16, 20, 220, 64]},
        {"transcription": "文本", "points": [[16, 72], [120, 72], [120, 112], [16, 112]]}
      ]
    }
  ]
  ```
  > **OCR 指标说明（`configs/metric/text_ocr.yaml`）**：
  > - 提供 `metrics.OCRAccuracyMetric`（OCR-A，越大越好）、`metrics.CERMetric`（CER，越小越好，允许 `> 1` 不截断）与 `metrics.OneMinusNEDMetric`（1-NED，越大越好），默认使用 `PP-OCRv5_server_rec`、`ocr_batch_size=1`、`ocr_device="cpu"`（出于跨平台可移植性默认固定 CPU，区别于 VOSR `evaluate.py` 的自动选设备；支持通过 `ocr_device` 与 `ocr_model_dir` / `ocr_cls_model_dir` 显式覆盖）。
  > - 同进程、同配置的三项指标共享单个 PaddleOCR 识别模型并复用同一样本的一次识别结果（有界仅缓存最后一个样本，检查图像/标注/配置变化，不持有 GPU 张量或计算图）。
  > - 通过 `BaseMetric.compute_stats()` 返回可加和的 `(numerator, denominator)`，在多卡 `gather_for_metrics` 后按全局有效框/字符数加权归约，避免先求每图均值再平均带来的偏差。
  > - **多进程权重准备建议**：由于 OCR 识别后端在首次验证时延迟初始化，多进程（DDP）训练前请先在单进程预下载权重并为各 rank 显式配置同一只读本地 `ocr_model_dir`（及可选 `ocr_cls_model_dir`），避免多 rank 首次并发下载同一缓存目录；单进程预热后的验证不等同于冷启动并发下载验收。

## 快速开始

### 1. 训练与多级配置覆盖

```bash
# 单卡或通过 accelerate launch 启动多卡训练（基于实验继承配置并临时覆写参数）
uv run accelerate launch tools/train.py \
  experiment=vosr2_onestep_text \
  optim.optimizer.lr=5e-6 \
  trainer.total_iters=10000 \
  trainer.log_with='[tensorboard]'

# 自动从最近一次状态目录精确续训
uv run accelerate launch tools/train.py \
  experiment=vosr2_onestep_text \
  trainer.resume=auto
```

### 2. 推理（`tools/infer.py`）

```bash
# 实验目录自动还原模式（自动加载 config_resolved.yaml、最优/EMA 权重与训练输入尺寸 tile_size）
uv run python tools/infer.py \
  infer.exp_dir=experiments/vosr2_onestep_text \
  infer.input=/data/test_imgs \
  infer.output_dir=outputs/vosr2_test \
  infer.tile.size=auto \
  infer.tile.overlap=64

# 多卡分布式离线推理 JSON 清单
uv run accelerate launch --num_processes 4 tools/infer.py \
  infer.exp_dir=experiments/vosr2_onestep_text \
  infer.input=/data/manifests/large_test.json \
  infer.input_root=/data/large_test \
  infer.output_dir=outputs/vosr2_large_test \
  infer.tile.size=auto \
  infer.tile.overlap=64 \
  infer.tile.batch_size=1

# 独立配置模式（不依赖训练实验目录）
uv run python tools/infer.py \
  arch=vosr2 \
  scale=1 \
  infer.weights_path=/path/to/weights.safetensors \
  infer.input=/data/test.png \
  infer.tile.size=512 \
  infer.tile.overlap=64
```

### 3. TextVerifier Stage-3 资产准备与迁移说明（`GlyphLoss`）

`GlyphLoss` 仅支持真实的 TextVerifier Stage-3 视觉编码器（已纯 PyTorch 内置于 `src/archs/textverifier.py`，无需 `timm`）与校验通过的 4 文件资产包：

- `textverifier_stage3.pth`
- `textverifier_k10_q90.pt`
- `prototype_metadata.json`
- `manifest.sha256`

默认在 `model_dir: null` 时通过单次 `huggingface_hub.snapshot_download` 从 `ATH-MaaS/TextVerifier-CJK-CLIP` 拉取完整 4 文件并校验 SHA-256 与 schema。若需离线或精确复现训练，请先将 4 个文件下载至本地目录并配置 `model_dir`（或使用已缓存目录并设置 `local_files_only: true`）：

```bash
huggingface-cli download ATH-MaaS/TextVerifier-CJK-CLIP \
  textverifier_stage3.pth \
  textverifier_k10_q90.pt \
  prototype_metadata.json \
  manifest.sha256 \
  --local-dir /path/to/textverifier_assets
```

> **配置迁移说明**：已移除 `GlyphLoss` 的 `revision`、`pretrained` 与测试注入参数，以及 `VOSR2OneStep` / `OCRKLLoss` / `LPIPSLoss` / `LPIPSMetric` 的 `pretrained` 离线伪模式参数；若旧配置中包含上述字段，请直接删除并指向真实本地权重或默认仓库。

### 4. 扩展自定义组件

- **自定义网络 / Loss**：编写普通 PyTorch `nn.Module`，在 `configs/arch/*.yaml` 或 `configs/loss/*.yaml` 中填写 `_target_` 即可。
- **自定义指标（含附加标注输入，如 OCR-A）**：继承 `metrics.BaseMetric`，实现 `compute(self, pred, target=None, meta=None) -> float`（若需跨样本加权归约可覆写 `compute_stats(...) -> tuple[float, float]`），并在 `configs/metric/*.yaml` 对应验证套件的 `metrics` 中通过 `_target_` 挂载。

## 测试与代码检查

```bash
uv run ruff check src tools tests
OMP_NUM_THREADS=1 uv run pytest
```
