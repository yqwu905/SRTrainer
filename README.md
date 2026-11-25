# SRTrainer

面向图像超分与底层视觉复原的模块化训练与推理框架，基于 **PyTorch + Hugging Face Accelerate + Hydra** 构建。

详细架构设计与全流程说明请参见 [docs/design.md](docs/design.md)。

## 核心特性

- **分层训练引擎 + Accelerate**：`Trainer` 统一管理 `Accelerator`、纯 Iteration 驱动循环、多验证集分布式精确评估（`gather_for_metrics`）、双轨 Checkpoint 与多 Tracker 日志；`BaseModel` / `VOSRModel`（+ `VOSRPipeline`）封装可替换多架构组装、逐组件 `trainable` 开关、按可训练组件维护的 FP32 `ModelEMA`、任意可组合 Loss 字典与单步更新（`lr_scheduler` 不经 `prepare()`，避免多卡倍速步进）。
- **纯 Hydra `_target_` 分级配置与多级继承**：按 `arch/{vae,dit,dino} / loss / model / data / optim / metric / trainer / infer` 拆分原子配置组，在 `configs/experiment/` 中通过 `# @package _global_` 多级继承并快速覆写局部参数；包级 `__init__.py` 提供短路径重导出。
- **独立解耦架构与损失函数**：`src/archs/` 将原单体 VOSR2 拆解为三个独立可替换 `nn.Module` 架构与独立配置文件：
  - `archs.AutoencoderKLQwenImage2D`（`configs/arch/vae/qwen_vae2d.yaml`）：Qwen-Image 2D VAE（Encoder + Decoder）；
  - `archs.VOSR2DiT`（`configs/arch/dit/vosr2_dit.yaml`）：VOSR2 单步隐空间 DiT；
  - `archs.DINOv2Encoder`（`configs/arch/dino/dinov2_vitl14.yaml`）：可选 DINOv2 视觉特征提取器（可通过 `~arch/dino` 或 `arch.dino=null` 关闭）；
  `VOSRModel`（`configs/model/vosr.yaml`）负责组装任意满足接口契约的 `vae` / `dit` / 可选 `dino`，通过 `model.trainable.{vae,dit,dino}` 自由配置每个架构是否参与训练，并在 Checkpoint 中按架构分别保存与恢复权重。`src/losses/` 内置 `L1Loss`、`LPIPSLoss`、基于冻结 PP-OCRv5 的 `OCRKLLoss` 与基于冻结 TextVerifier Stage-3 的 `GlyphLoss`。
- **统一 JSON 清单 + 样本级混合与 GPU 掩码 Real-ESRGAN 退化**：
  - 仅读取独立图像文件（`.jpg/.png/.jpeg/.webp`），`gt_only` 与 `paired` 统一使用字典列表 JSON，自动根据样本是否含有效 `lq` 判定模式（配置中无需 `mode`）；支持在清单内嵌 `text_lines` / `glyphs` 或在数据源上挂载 VOSR `glyphs_json` 以启用 `GlyphLoss`；
  - `MixedSRDataset` + `DistributedWeightedInfiniteSampler` 严格按 `ratio` 跨数据源采样并支持 O(1) 断点续训；
  - `RealESRGANDegrader` 在 GPU FP32 下仅对 `gt_only` 子集执行二阶模糊/缩放/噪声/DiffJPEG/Sinc 退化，与真实 `paired` 样本在同 Batch 平滑混合。
- **验证集内聚于 `metric` 配置组**：每个验证套件在 `configs/metric/` 中独立绑定数据集与所需指标，并将 JSON 附加字段（如 `label` / `text_lines` / `annotation_size`）完整透传到 `BaseMetric.compute(pred, target, meta)` 与 `BaseMetric.compute_stats(pred, target, meta)`；内置 BasicSR/MATLAB BT.601 对齐的 `PSNRMetric`、`SSIMMetric`、`LPIPSMetric`，以及基于 RapidOCR 的文本识别指标 `OCRAccuracyMetric`（OCR-A）、`CERMetric`（CER）与 `OneMinusNEDMetric`（1-NED）。
- **双轨 Checkpoint（按架构独立保存）**：滚动保存完整续训状态（`checkpoints/state_iter_XXXXXX/`，支持 `resume=auto`）+ 按架构原子导出纯推理权重（`weights/iter_XXXXXX_{vae,dit,dino}.safetensors`、可训练组件 `*_ema.safetensors`、`latest_{arch}[_ema].safetensors`、`best_<val>_<metric>_{arch}.safetensors`）并持久化 `config_resolved.yaml`。
- **矩形裁剪**：`data.gt_size` 可填整数或 `[高, 宽]`。例如 `gt_size: [256, 512]` 配合 `scale: 4`，训练时的 LQ 裁剪和推理时的自动分块均为 64×128，此时可将分块重叠设为 16 像素。矩形裁剪启用旋转增强时只使用 0° 和 180°，保持批内尺寸一致。
- **共享 `TiledInferencer` + `tools/infer.py` 分布式推理**：支持实验目录自动还原与独立 `infer.yaml` 两种模式；支持按训练输入尺寸自动推导 `tile_size`（`gt_size // scale`）与可配置 `overlap` 羽化无缝融合；支持单卡与 Accelerate 多卡分布式离线推理。

## 安装

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖与虚拟环境（固定 `torch==2.9.0`）：

```bash
# 同步项目及开发依赖（含 pytest / ruff）
uv sync

# 可选：安装额外实验看板依赖（wandb / mlflow / swanlab）
uv sync --extra trackers

# 可选：安装文本 OCR 评测依赖（RapidOCR + OpenCV，复用项目 PyTorch 后端）
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
  > - 提供 `metrics.OCRAccuracyMetric`（OCR-A，越大越好）、`metrics.CERMetric`（CER，越小越好，允许 `> 1`）与 `metrics.OneMinusNEDMetric`（1-NED，越大越好）。RapidOCR 通过 PyTorch 后端运行 `PP-OCRv5_server_rec`，默认 `ocr_batch_size=1`、`ocr_device="cpu"`。包括 Ascend NPU 在内的训练设备可独立配置。显式设置 `ocr_device=gpu[:ID]`（或 `cuda[:ID]`）、`npu[:ID]` 或 `mps` 时，将直接复用当前环境的 PyTorch / `torch_npu` / MPS 设备，无需安装 ONNX Runtime。
  > - 同进程、同配置的三项指标共享一个识别模型并复用同一样本的一次识别结果。缓存仅保留最后一个样本的文本与缓存键。
  > - 通过 `BaseMetric.compute_stats()` 返回可加和的 `(numerator, denominator)`，在多卡 `gather_for_metrics` 后按全局有效框/字符数加权归约，避免先求每图均值再平均带来的偏差。
  > - 默认权重与字典在首次验证时下载到 `${XDG_CACHE_HOME:-$HOME/.cache}/srtrainer/rapidocr`。离线环境可通过 `ocr_model_dir` 和可选的 `ocr_cls_model_dir` 指定 RapidOCR PyTorch `.pth` 权重文件或包含该文件的目录（文件名需与 RapidOCR 内置架构名一致，如 `ch_PP-OCRv5_rec_server.pth`），并在同目录提供 `dict.txt` 或 `ppocrv5_dict.txt` 字典文件。配置 `ocr_cls_model: ch_ppocr_mobile_v2.0_cls` 可启用 0/180 度方向分类。

## 快速开始

### 1. 训练与多级配置覆盖

```bash
# 单卡或通过 accelerate launch 启动多卡训练（基于实验继承配置并临时覆写参数）
uv run accelerate launch tools/train.py \
  experiment=vosr2_onestep_text \
  optim.optimizer.lr=5e-6 \
  trainer.total_iters=10000 \
  trainer.log_with='[tensorboard]'

# 按组件控制训练/冻结，或替换任意子架构 / 关闭可选 DINO
uv run accelerate launch tools/train.py \
  experiment=vosr2_onestep_text \
  model.trainable.vae=true \
  model.trainable.dit=true \
  model.trainable.dino=false

# 自动从最近一次状态目录精确续训
uv run accelerate launch tools/train.py \
  experiment=vosr2_onestep_text \
  trainer.resume=auto
```

### 2. 推理（`tools/infer.py`）

```bash
# 实验目录自动还原模式（自动加载 config_resolved.yaml、按组件还原最优/EMA 权重与训练输入尺寸 tile_size）
uv run python tools/infer.py \
  infer.exp_dir=experiments/vosr2_onestep_text \
  infer.input=/data/test_imgs \
  infer.output_dir=outputs/vosr2_test \
  infer.tile.size=null \
  infer.tile.overlap=64

# 多卡分布式离线推理 JSON 清单
uv run accelerate launch --num_processes 4 tools/infer.py \
  infer.exp_dir=experiments/vosr2_onestep_text \
  infer.input=/data/manifests/large_test.json \
  infer.input_root=/data/large_test \
  infer.output_dir=outputs/vosr2_large_test \
  infer.tile.size=null \
  infer.tile.overlap=64 \
  infer.tile.batch_size=1

# 独立配置模式（按组件组合 arch/vae、arch/dit、arch/dino 与各组件权重）
uv run python tools/infer.py \
  arch/vae=qwen_vae2d \
  arch/dit=vosr2_dit \
  arch/dino=dinov2_vitl14 \
  scale=1 \
  infer.component_weights.vae=/path/to/vae.safetensors \
  infer.component_weights.dit=/path/to/dit.safetensors \
  infer.component_weights.dino=/path/to/dino.safetensors \
  infer.input=/data/test.png \
  'infer.tile.size=[512,512]' \
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

> **配置迁移说明**：原单体 `archs.VOSR2OneStep`（`configs/arch/vosr2.yaml`）与 `models.SRModel`（`configs/model/sr.yaml`）已拆解重构为独立架构 `archs.AutoencoderKLQwenImage2D`、`archs.VOSR2DiT`、`archs.DINOv2Encoder`（`configs/arch/{vae,dit,dino}/*.yaml`）与 `models.VOSRModel`（`configs/model/vosr.yaml`），预训练权重入口迁移至 `model.pretrain.{vae,dit,dino}.{path,strict,key}`；同时已移除 `GlyphLoss` 的 `revision`、`pretrained` 与测试注入参数，以及 `OCRKLLoss` / `LPIPSLoss` / `LPIPSMetric` 的 `pretrained` 离线伪模式参数。

### 4. 扩展自定义组件

- **自定义子架构 / Loss**：编写普通 PyTorch `nn.Module`，在 `configs/arch/{vae,dit,dino}/*.yaml` 或 `configs/loss/*.yaml` 中填写 `_target_` 即可在 `VOSRModel` 中直接替换对应组件。
- **自定义指标（含附加标注输入，如 OCR-A）**：继承 `metrics.BaseMetric`，实现 `compute(self, pred, target=None, meta=None) -> float`（若需跨样本加权归约可覆写 `compute_stats(...) -> tuple[float, float]`），并在 `configs/metric/*.yaml` 对应验证套件的 `metrics` 中通过 `_target_` 挂载。

## 测试与代码检查

```bash
uv run ruff check src tools tests
OMP_NUM_THREADS=1 uv run pytest
```
