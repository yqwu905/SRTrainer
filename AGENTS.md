# AGENTS.md

本文件面向在本仓库中协作的开发者与 AI Agent，约定项目工程规范，特别是配置系统的核心设计原则。

## 项目概览

SRTrainer 是面向图像超分与底层视觉复原的模块化训练与推理框架，基于 **PyTorch + Hugging Face Accelerate + Hydra + OmegaConf Structured Config** 构建。

常用命令：

```bash
# 代码静态检查
uv run ruff check src tools tests

# 运行全部测试（在 Apple Silicon / 无 CUDA 环境建议加 ACCELERATE_USE_CPU=true）
ACCELERATE_USE_CPU=true OMP_NUM_THREADS=1 uv run pytest
```

---

## 配置系统设计原则（Hydra + Dataclass + OmegaConf Structured Config）

本项目的全部训练与推理配置必须严格遵循 **Hydra + Dataclass + OmegaConf Structured Config** 架构，并遵守以下五条核心原则：

### 1. 配置结构全部声明式定义
- 所有配置分组的字段名、类型、嵌套层级均通过 Python `@dataclass`（位于 `src/config/schema.py`）声明式定义，并通过 `src/config/store.py`（及 `src/hydra_plugins/`）注册到 Hydra `ConfigStore`。
- **禁止**在模型、引擎或工具脚本中编写 `_parse_xxx()` 函数手工描述字段结构、解析字典键、做类型转换或拦截未知键；结构合法性与未知键检查统一交由 OmegaConf Structured Config 完成。

### 2. 每个字段只定义一次（Single Source of Truth）
- 每个配置字段的类型、默认值与嵌套关系只在 Dataclass 中定义一次。
- `configs/` 下的原子 YAML 文件通过 Hydra `defaults` 继承对应的 Structured Config 基线（如 `base_train`、`base_vosr`、`base_trainer`、`base_data`、`base_metric`、`base_infer`、`base_infer_settings`、`base_realesrgan`），YAML 中只保留 `${...}` 插值绑定、实验差异或具体数据源/架构参数，避免重复抄写默认值。
- **禁止**在业务代码中通过 `OmegaConf.select(cfg, "...", default=...)` 或 `cfg.get("...", default)` 重复硬编码默认值。

### 3. 统一 YAML 配置格式
- 消除同一配置项的多种多态输入形式（减少字符串 / 字典 / 布尔 / `None` 混用），保持 YAML 与 CLI 表达唯一、确定、可静态类型检查：
  - `model.trainable`：始终为 `VOSRTrainableConfig`（`{vae, dit, dino}` 布尔字段）。
  - `model.ema`：始终为 `EMAConfig`，通过 `ema.enabled: false` 关闭，不使用 `ema: null`。
  - `model.pretrain.{vae,dit,dino}`：始终为 `PretrainEntryConfig`（`{path, strict, key}`），不接受裸路径字符串或 `null` 替代结构体。
  - `trainer.log_with`：始终为 `list[str]`（空列表 `[]` 表示关闭 tracker），不接受裸字符串、`null` 或 `"none"`/`"null"` 魔术字符串。
  - `trainer.mixed_precision` / `infer.mixed_precision`：使用 `MixedPrecision` 枚举（`"no" | "fp16" | "bf16"`；YAML 中 `"no"` 必须加引号），不接受 `null`、`""`、`"off"`、`"fp32"` 等别名。
  - `trainer.resume`：`Optional[str]`，仅接受 `null`、`"auto"` 或有效 checkpoint 目录路径，不接受 `false`、`""`、`"none"` 等别名。
  - `trainer.*_interval`：统一为非负整数 `int`（`0` 表示禁用），不混用 `null`。
  - `trainer.val_tile` / `infer.tile`：始终为 `TileConfig`；`tile.size` 统一为 `Optional[list[int]]`（`null` 表示自动按 `gt_size // scale` 推导，`[H, W]` 表示显式指定分块尺寸），不再混用 `"auto"` 字符串与标量整数。
  - `data.sources[*]`：始终为 `DataSourceConfig`，统一使用 `glyphs_json`（不使用旧别名 `text_boxes`，严禁 `mode`）。
  - `data.augment`：始终为 `AugmentConfig`（`{hflip, vflip, rot90}`），不传 `null`。
  - `infer.weights_path` / `infer.weights_key`（单架构）与 `infer.component_weights` / `infer.component_weights_key`（多架构 VOSR）拆分独立字段，避免同一字段在字符串与字典之间多态切换。
- 离散取值选项统一使用 `(str, Enum)`（如 `MixedPrecision`、`NoiseInput`、`BlendMode`、`TrainLogImageMode`、`SaveFormat`）。

### 4. 只保留必要的业务校验
- 字段缺失、字段类型错误、非法枚举值、未知字段等结构性校验全部交给 OmegaConf Structured Config 自动完成。
- 手写校验逻辑**仅保留**静态类型无法表达的领域业务规则：
  - 参数数值范围（如 `scale >= 1`、`total_iters >= 1`、`0 < ema.decay <= 1`、`0 <= text_crop_prob <= 1`、`0 <= tile.overlap < min(tile_h, tile_w) // 2`、奇数核尺寸、概率分布和为 1 等）；
  - 跨字段依赖关系（如 `gt_size % scale == 0`、`data.scale == scale`、`primary_dataset` 与 `primary_metric` 成对出现且存在于 `val_suites`、`trainable.<comp>` 与 `pretrain.<comp>.path` 要求对应 `arch.<comp>` 非空、至少有一个活跃组件可训练、`resume` 显式路径存在且为目录等）。

### 5. 配置校验集中在训练与推理入口
- 所有配置的 Structured Config 合并、类型化实例化与跨字段业务校验均集中在入口层（`src/config/validation.py` 中的 `load_train_config` / `load_infer_config`，由 `tools/train.py`、`Trainer.__init__` 与 `tools/infer.py` 调用）。
- 校验失败必须 fail-fast（在创建实验目录、初始化 `Accelerator` 或下载权重之前抛出 `ConfigValidationError` / `ValueError` / `FileNotFoundError`）。
- 模型内部（`BaseModel`、`VOSRModel`）以及下游组件（`MixedSRDataset`、`TiledInferencer`、`KernelGenerator`）直接接收类型明确的 Dataclass 配置对象（`TrainConfig`、`VOSRModelConfig`、`TileConfig`、`AugmentConfig`、`KernelConfig` 等），不再在内部重复解析或校验配置结构。
