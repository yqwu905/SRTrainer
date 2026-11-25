# OCR 文本指标适配设计与验收约定

## 基线和范围

- 工程基线：`ae81eda810d98a0aeff66ed28bac7fcddb3d016f`，当前扁平 `src/metrics` 布局。
- 参考：`yqwu905/VOSR` 的 `evaluate.py`，固定提交 `9fbcf8fe22b399a6fbb41e011748322363ae9ee0`，文件 blob `ad6e85aa59a4bb4e54c70653ef850768ec54e547`。
- 为现有训练验证套件增加 OCR-A、CER、1-NED，不增加第二套评估命令，不改变训练损失，不扩展 LQ/GT 对照报表或尺寸分桶。
- 本设计由协调者负责；Coder 负责实现及单元/集成测试；验证者独立检查差异、运行测试并明确给出验收结论。只有验收通过后才提交 PR，不自动合并。

## 与参考一致的数学语义

识别文本和标注均先 NFKC 规范化，再删除全部 Unicode 空白；默认区分大小写，`ignore_case=true` 时使用 `lower()`。编辑距离是按 Unicode 字符计算的单位代价 Levenshtein 距离。

设所有有效文本框的识别值为 p、标注为 g、编辑距离为 d：

- OCR-A = 完全匹配文本框数 / 有效文本框数，越大越好。
- CER = 所有 d 之和 / 所有 g 的字符数之和，越小越好；允许大于 1，禁止截断。
- 1-NED = 所有 `1 - d / max(len(p), len(g))` 之和 / 有效文本框数，越大越好。
- 不能先求每图指标再平均。例：第一图一个 `("", "A")`，第二图三个各长 10 字且识别正确的框，全局结果必须为 OCR-A=0.75、CER=1/31、1-NED=0.75。
- 与参考相同，仅原始标签 `strip()` 后恰好为 `###` 或 `*` 的标签视为哨兵并跳过；全角 `＊` / `＃＃＃` 不因此跳过。规范化后为空的标签也跳过；默认跳过 difficult，显式 `include_difficult` 才包括；空识别预测不跳过，而是按删除错误计分。

## 接口与分布式归约

保留 `BaseMetric.compute(pred, target=None, meta=None) -> float`、`.to()` 和 `higher_is_better`，新增非抽象 `compute_stats` 返回可加和的 `(numerator, denominator)`；默认实现返回 `(compute(...), 1)`，已有自定义指标无需修改。

Trainer 对每个样本每项指标构造 `[1, 2]` 统计量张量，再调用 `accelerator.gather_for_metrics`。首维必须是样本维，尾部补齐才能同时移除一整个样本的分子、分母；禁止摊平成 `[2]`，也禁止先按 rank 累加后 all_reduce。汇总分子和分母后求比值，保留现有 MPS/NPU float32、其他设备 float64 的 gather dtype。

被忽略/零有效框样本返回 `(0, 0)`；整套无有效框时结果为 NaN，不能输出虚假的 0/1，不能更新 best checkpoint。现有 `TrainState.update_best` 已跳过 NaN。PSNR 的正无穷必须继续有效。

公开 Hydra 路径：`metrics.OCRAccuracyMetric`、`metrics.CERMetric`、`metrics.OneMinusNEDMetric`。配置键建议为 `ocr_a`、`cer`、`one_minus_ned`，文档展示名仍为 OCR-A、CER、1-NED。若命名有调整，需同步测试、示例和本设计。

## 标注合同

沿用验证集 JSON manifest 和原样透传的 `meta`，不要求修改训练数据格式：

1. 单行整图模式：`{"lq": "line.png", "label": "文本"}`；`label_key` 可配置。
2. 框模式：`text_lines` 为列表，每项读取 `transcript` / `transcription` / `text`，以及 `points` / `quad` / `bbox` / `box`。四边形使用循环顺序，bbox 使用 xyxy；points 多边形可含超过四点。
3. `text_lines` 存在时优先，空列表代表没有有效框，不能偷偷回退整图 label。缺少两种标注或格式损坏应明确报错，而非伪装为被忽略样本。
4. 坐标默认在预测图像素空间。可用 `meta.annotation_size=[width,height]` 明确提供原标注画布尺寸，按预测宽高分别缩放；LQ/GT 坐标均通过该字段声明，禁止从模型倍率、文件名或 target 静默猜测。
5. `valid=false`、`ignore=true`、`alignment_valid=false` 的行跳过；图级 `alignment_valid=false` 跳过全部。严格验证布尔值；include_difficult 不得复活其他无效标注。
6. 可不提供 GT；target 不应成为 OCR 指标的必要输入。标签必须为字符串，几何必须具有合法有限坐标/正尺寸；空字符串或规范化后为空的标签按前述规则跳过。

## 裁剪和识别

- RGB `[1,3,H,W]` 浮点预测先 detach，限制到 [0,1]，乘 255 后 `round()`，转换为 CPU uint8 BGR，与工程 `save_image` 量化一致；不修改输入 tensor 或 meta，不保留训练计算图。
- 对四边形调整逆时针顺序为顺时针并保留起点；超过四点使用 OpenCV 最小面积矩形；采用与参考一致的 `INTER_CUBIC`、`BORDER_REPLICATE` 透视裁剪。高宽比至少 1.5 时逆时针旋转 90 度。
- 重用已有几何验证能力，但不能直接用现有 PyTorch 双线性 ROI 替代参考 OpenCV 三次插值而声称像素一致。退化/交叉/非有限框明确拒绝。
- 后端使用 RapidOCR 的批量识别接口和 PyTorch 引擎（`EngineType.TORCH`），默认模型 `PP-OCRv5_server_rec`，默认 batch_size=1。可选 `ch_ppocr_mobile_v2.0_cls` 做 0/180 度判断。识别直接接收标注裁剪，并检查输出条数与输入框数一致。
- `ocr_device` 默认 CPU，独立于训练设备，支持 `cpu`、`gpu[:ID]`/`cuda[:ID]`、`npu[:ID]`（需 `torch_npu`）和 `mps`。`.to(torch_device)` 保留 OCR 设备设置。`ocr_model_dir` 和 `ocr_cls_model_dir` 支持本地 RapidOCR PyTorch `.pth` 文件或目录（识别模型同目录可提供 `dict.txt` / `ppocrv5_dict.txt`）。
- RapidOCR 和 OpenCV 作为可选 OCR 依赖，延迟导入和初始化。缺包时提示 `uv sync --extra ocr`。
- 同进程、同配置的三项指标共享模型，并复用同一样本的一次识别结果；缓存只能有界保留最后一个样本，必须检查图像/标注/配置变化，不允许下一次验证错误复用上次预测。不依赖 inference-mode tensor 不具备的 `_version`；不能保留 GPU tensor/计算图。测试用 monkeypatch 或替代测试后端，生产接口不添加 fake/pretrained=false 模式。
- 首次验证时自动下载模型权重与字典。同一缓存目录中的进程通过文件锁串行预下载，离线运行可配置本地 PyTorch `.pth` 权重。

## 交付范围与所有权

- 协调者：本设计、基线确认、任务分派、验证证据汇总、最终 commit/push/PR。
- Coder：`src/metrics/`、`src/engine/trainer.py`、必要时 `src/data/` 的最小改动、`configs/metric/text_ocr.yaml`、`pyproject.toml`、`README.md`、`docs/design.md`、相关 tests。不要改协调者设计文件，不要修改 main 工作目录，不要触碰他人的 worktree。
- 验证者：只读评审实现，在输出目录保存独立报告/探针；不得替 Coder 修代码。缺陷回报后由 Coder 修复，再验收同一最终差异。
- 开发/评审期间不要自行 commit、push、创建 PR。不要更改凭据或安装/升级全局 lh。

## 独立验收矩阵

1. 公式：完全匹配、替换/插入/删除、空预测、Unicode/NFKC/空白/大小写、CER>1；对固定参考实现做差分或独立手算比对。
2. 标注：整图 label、多个 text_lines、GT-less、所有 ignore 组合、全空套件、配置错误、合法四边形/多边形、退化框、显式尺寸缩放、无输入变更。
3. 图像：RGB/BGR 通道、uint8 转换、透视裁剪、阈值 1.5 旋转、可选 180 度分类、少/多返回条目显式报错。
4. 缓存：三指标一次识别；预测变化/原地更新、标注变化、重复 validate 不串缓存；不会持有无界 GPU 图像或梯度图。
5. Trainer：混用现有 PSNR/SSIM/自定义标量指标和 OCR；跨图片不同文本数/长度的加权数学；CER 驱动 best checkpoint 方向正确；全空不保存 best。
6. 分布式：真实 2-process CPU、奇数样本数量，包含零框样本；与串行全局手算一致，证明 padding 未计入。仅 mock gather 不足以通过该项。
7. 兼容：运行现有 metrics/config/trainer smoke 回归和改动文件的 Ruff 检查，确认普通训练环境可在未安装 OCR 可选依赖时导入指标。
8. 后端：使用真实 `PP-OCRv5_server_rec` 权重运行 CPU 冒烟，覆盖整图和文本框。

## 初版 PaddleOCR 实现验收记录

**2026-11-23 独立验收结论：PASS，限本次 OCR 指标交付及下述实测范围。**

指定 Coder 完成开发，指定验证者在实现冻结后独立复核并签字。冻结时间为 `2026-11-23T12:10:11Z`，最终独立复核时间为 `2026-11-23T12:20:22.874727Z`。协调者在取得 PASS 后追加本节，仅增加验收说明，未改变已验收的十个实现、配置、测试和使用文档文件。

| 检查 | 结果与边界 |
| --- | --- |
| 新增 OCR 自动化测试 | 12 passed，0 failed；有/无可选 OCR 依赖的两种环境均通过 |
| 独立公式与裁剪对照 | 414 案例分别调用三项公开指标；7 组裁剪与固定 VOSR 参考逐像素一致 |
| 真实识别后端 | 本地 PP-OCRv5_server_rec CPU 真实权重；整图和四边形均得到 OCR-A=1、CER=0、1-NED=1 |
| 真实双进程端到端 | 公开三指标 + 真实 PaddleOCR + Trainer.validate + Accelerate，5 张图含零框及不同框数，两个 rank 与手算一致 |
| 双进程手算结果 | OCR-A=4/6、CER=10/45、1-NED=22/27；重复验证一致，全空时 NaN 不更新 best |
| 兼容性 | 旧标量指标、PSNR 正无穷、SSIM、GT-less、CER 最优方向、奇数尾部去重均通过独立探针 |
| Hydra / Ruff | 指定 experiment 与 metric=text_ocr 可组合；改动 Python 文件 Ruff 通过 |
| 构建读回 | 协调者成功构建 wheel，读回 metrics/ocr_metrics.py 与冻结源码字节完全相同 |

回归不是全绿：OCR + 原 metrics/Hydra 为 **53 passed / 6 failed**，原 Trainer 为 **1 passed / 9 failed**，邻接数据/切片推理为 **73 passed / 4 failed**。失败项已逐项与基线对照；验证者另外从固定基线的干净 git archive 独立复现对应配置/邻接失败，未发现新增失败。问题包括旧测试未提供必填 experiment、旧配置预期、裸 torchrun 不在 PATH，以及本机 standalone rendezvous 地址解析导致超时。本次未扩大范围修改这些基线问题；新的真实双进程验收使用显式回环地址并通过。

复现入口：

```sh
python -m pytest tests/test_ocr_metrics.py -q
python tools/train.py experiment=vosr2_onestep_text metric=text_ocr --cfg job
ruff check src/metrics/base.py src/metrics/__init__.py src/metrics/ocr_metrics.py src/engine/trainer.py tests/test_metrics.py tests/test_ocr_metrics.py
uv build --wheel
```

真实模型实测环境为 macOS arm64、Python 3.12、torch 2.9.0、Paddle 3.3.1、PaddleOCR 3.7.0。真实 OCR 双进程探针使用 identity SR 测试网络与记录型 checkpointer，因此不等价于完整生产 SR 训练或真实 checkpoint 落盘/恢复验收。CUDA/MPS/NPU 实机、真实方向分类权重、多 rank 冷启动同时下载、完整生产数据集准确率和长期性能/内存压测未验收。

验收追踪：核心模块 SHA256 为 `2913510c7a64cf648ba9ab36375f9c27b5a99389752b5f670ed79ed24d3bc449`；冻结凭证 SHA256 为 `4803b60d331aa0dc37c563e88c7906fc2311429c67d0bb7de53412f1d32df827`。验证者的最终报告、独立探针、日志与全部文件指纹保存在协调工作区的 `outputs/ocr-metrics-coordination/reviewer/`。最终提交及 PR 编号以 Git 历史和 PR 页面为准，不在本文件中写入自引用提交哈希。
