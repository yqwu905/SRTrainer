# SRTrainer

面向图像超分与底层视觉复原的模块化训练与推理框架，基于 **PyTorch + Hugging Face Accelerate + Hydra** 构建。

详细架构设计与全流程说明请参见 [docs/design.md](docs/design.md)。

## 核心特性

- **分层训练引擎 + Accelerate**：`Trainer` 统一管理 `Accelerator`、纯 Iteration 驱动循环、多验证集分布式精确评估（`gather_for_metrics`）、双轨 Checkpoint 与多 Tracker 日志；`BaseModel` / `SRModel` 封装网络、FP32 `ModelEMA`、任意可组合 Loss 字典与单步更新（`lr_scheduler` 不经 `prepare()`，避免多卡倍速步进）。
- **纯 Hydra `_target_` 分级配置与多级继承**：按 `arch / loss / model / data / optim / metric / trainer / infer` 拆分原子配置组，在 `configs/experiment/` 中通过 `# @package _global_` 多级继承并快速覆写局部参数；包级 `__init__.py` 提供短路径重导出。
- **内置网络与损失函数**：`src/srtrainer/archs/` 提供轻量占位网络 `srtrainer.archs.ToyModel` 与单步生成超分网络 `srtrainer.archs.VOSR2OneStep`（含冻结 Qwen-Image 2D VAE 与 DINOv2-ViT-L/14，权重与官方 VOSR2 checkpoint 键完全对齐）；`src/srtrainer/losses/` 内置 `L1Loss`、`LPIPSLoss`、基于冻结 PP-OCRv5 的 `OCRKLLoss` 与基于冻结 TextVerifier Stage-3 的 `GlyphLoss`，同时支持通过 Hydra `_target_` 直接接入任意自定义网络与损失函数。
- **统一 JSON 清单 + 样本级混合与 GPU 掩码 Real-ESRGAN 退化**：
  - 仅读取独立图像文件（`.jpg/.png/.jpeg/.webp`），`gt_only` 与 `paired` 统一使用字典列表 JSON，自动根据样本是否含有效 `lq` 判定模式（配置中无需 `mode`）；支持在清单内嵌 `text_lines` / `glyphs` 或在数据源上挂载 VOSR `glyphs_json` 以启用 `GlyphLoss`；
  - `MixedSRDataset` + `DistributedWeightedInfiniteSampler` 严格按 `ratio` 跨数据源采样并支持 O(1) 断点续训；
  - `RealESRGANDegrader` 在 GPU FP32 下仅对 `gt_only` 子集执行二阶模糊/缩放/噪声/DiffJPEG/Sinc 退化，与真实 `paired` 样本在同 Batch 平滑混合。
- **验证集内聚于 `metric` 配置组**：每个验证套件在 `configs/metric/` 中独立绑定数据集与所需指标，并将 JSON 附加字段（如 OCR-A 的 `label`）完整透传到 `BaseMetric.compute(pred, target, meta)`；内置 BasicSR/MATLAB BT.601 对齐的 `PSNRMetric`、`SSIMMetric` 与 `LPIPSMetric`。
- **双轨 Checkpoint**：滚动保存完整续训状态（`checkpoints/state_iter_XXXXXX/`，支持 `resume=auto`）+ 原子导出纯推理权重（`weights/*.safetensors`，含 `iter_*`、`_ema`、`latest`、`best_*`）并持久化 `config_resolved.yaml`。
- **共享 `TiledInferencer` + `tools/infer.py` 分布式推理**：支持实验目录自动还原与独立 `infer.yaml` 两种模式；支持按训练输入尺寸自动推导 `tile_size`（`gt_size // scale`）与可配置 `overlap` 羽化无缝融合；支持单卡与 Accelerate 多卡分布式离线推理。

## 安装

```bash
pip install -e ".[dev]"
# 可选：安装额外实验看板依赖（wandb / mlflow / swanlab）
pip install -e ".[trackers]"
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
- **验证集清单（在 `configs/metric/` 中配置，`gt` 可选，其余字段自动透传到 `meta`）**：
  ```json
  [
    {"lq": "lr/0001.png", "gt": "hr/0001.png", "label": "HELLO 2026"},
    {"lq": "lr/0002.png", "label": "SRTRAINER"}
  ]
  ```

## 快速开始

### 1. 训练与多级配置覆盖

```bash
# 单卡或通过 accelerate launch 启动多卡训练（基于实验继承配置并临时覆写参数）
accelerate launch tools/train.py \
  experiment=exp01_toy_mixed_x4 \
  optim.optimizer.lr=5e-5 \
  trainer.total_iters=50000 \
  trainer.log_with='[tensorboard]'

# 自动从最近一次状态目录精确续训
accelerate launch tools/train.py \
  experiment=exp01_toy_mixed_x4 \
  trainer.resume=auto
```

### 2. 推理（`tools/infer.py`）

```bash
# 实验目录自动还原模式（自动加载 config_resolved.yaml、最优/EMA 权重与训练输入尺寸 tile_size）
python tools/infer.py \
  infer.exp_dir=experiments/exp01_toy_l1_lpips_x4 \
  infer.input=/data/test_imgs \
  infer.output_dir=outputs/exp01_test \
  infer.tile.size=auto \
  infer.tile.overlap=16

# 多卡分布式离线推理 JSON 清单
accelerate launch --num_processes 4 tools/infer.py \
  infer.exp_dir=experiments/exp01_toy_l1_lpips_x4 \
  infer.input=/data/manifests/large_test.json \
  infer.input_root=/data/large_test \
  infer.output_dir=outputs/exp01_large_test \
  infer.tile.size=auto \
  infer.tile.overlap=16 \
  infer.tile.batch_size=8

# 独立配置模式（不依赖训练实验目录）
python tools/infer.py \
  arch=toy_x4 \
  scale=4 \
  infer.weights_path=/path/to/weights.safetensors \
  infer.input=/data/test.png \
  infer.tile.size=64 \
  infer.tile.overlap=16
```

### 3. 扩展自定义组件

- **自定义网络 / Loss**：编写普通 PyTorch `nn.Module`，在 `configs/arch/*.yaml` 或 `configs/loss/*.yaml` 中填写 `_target_` 即可。
- **自定义指标（含附加标注输入，如 OCR-A）**：继承 `srtrainer.metrics.BaseMetric`，实现 `compute(self, pred, target=None, meta=None) -> float`，并在 `configs/metric/*.yaml` 对应验证套件的 `metrics` 中通过 `_target_` 挂载。

## 测试与代码检查

```bash
ruff check src tools tests
OMP_NUM_THREADS=1 python -m pytest
```
