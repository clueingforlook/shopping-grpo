# 实验结果

这里保存已完成实验的报告、逐任务指标、训练曲线和机器可读摘要。各次结果的数值与历史来源记录保持不变，目录和文件名已整理为无个人前缀的名称。

## Eval v3：严格 Gold

Eval v3 直接读取 ShopSimulator 结构化终局，不调用 LLM Judge 或 Rubric。以下均以同一组 200 个任务为分母：

| 模型与报告 | Gold 成功 | 完成购买 | 基础设施无效 | 不可验证 |
|---|---:|---:|---:|---:|
| [Base](eval-v3/base/eval-v3-summary.md) | 3（1.5%） | 3（1.5%） | 9 | 140 |
| [SFT](eval-v3/sft/eval-v3-summary.md) | 114（57.0%） | 185（92.5%） | 3 | 9 |
| [GRPO step 250 LoRA](eval-v3/grpo-step250-lora/eval-v3-summary.md) | 122（61.0%） | 184（92.0%） | 2 | 7 |
| [GRPO step 500 LoRA](eval-v3/grpo-step500-lora/eval-v3-summary.md) | 123（61.5%） | 188（94.0%） | 1 | 6 |

“基础设施无效”和“不可验证”是独立报告字段，可能重叠，不能相加当作新的任务总数。基础设施异常单列，不解释为模型错误；此表保留既有运行结果，没有补跑后替换分母。每个模型目录同时提供 `eval-v3-summary.json`、`eval-v3-task-results.jsonl` 和来源 manifest，可复核品类、属性、规格、价格及难度分组。

SFT→step 500 的 Gold 提升为 4.5 个百分点。按全部 200 个 `task_id` 对齐已有逐任务记录：

| 比较 | 非 Gold → Gold | Gold → 非 Gold | 两者均 Gold | 两者均非 Gold |
|---|---:|---:|---:|---:|
| Base → SFT | 111 | 0 | 3 | 86 |
| SFT → step 250 | 9 | 1 | 113 | 77 |
| SFT → step 500 | 12 | 3 | 111 | 74 |
| step 250 → step 500 | 5 | 4 | 118 | 73 |

这是固定分母配对统计，没有另外筛选有效样本；与 Eval v2 的“干净配对”统计不同。step 250→500 净增加 1 个 Gold，现有记录不足以据此认定稳定的泛化优势。Final-200 已参与评测与 Reward 设计讨论，属于开发评测集。

## GRPO 训练指标

- [500 步训练曲线](grpo-training/training-curves.svg)、[训练 CSV](grpo-training/training-metrics.csv)、[验证 CSV](grpo-training/validation-metrics.csv)、[摘要](grpo-training/training-curve-summary.json)。
- [250 步曲线快照](grpo-training/training-curves-step250.svg)及同目录带 `-step250` 后缀的 CSV / JSON。

训练原计划 300 步，后来续训至 500 步，每 50 步验证一次。固定 50 题验证集的 Gold 从 step 0 的 48% 到 step 250/500 的 52%；step 400/450 为 54%。因此 step 500 是已保存正式评测的 checkpoint，不能称为验证集最优 checkpoint。训练小批次 Gold、验证 Gold 和 Final-200 Gold 应分别阅读。

本目录包含从日志提取的指标与曲线，**不包含完整原始训练日志**。

## Eval v2 与早期分析

Eval v2 使用冻结 Rubric 和 LLM Judge，并保留环境结果、归一化购买结果和过程归因：

- [评测说明](docs/05-eval-v2评测说明.md)、[最终结果](docs/06-eval-v2最终结果.md)、[典型轨迹](docs/07-eval-v2典型轨迹.md)。
- [Base / SFT 配对报告](eval-v2/base-vs-sft/comparison-summary.md)及对应逐任务 JSONL。
- [V4 Pro / Prompt v2 / Gold v6 校准](eval-v2/judge-calibration-v4-pro-prompt-v2/judge-calibration-manifest-gold-v6.json)。

Eval v2 的 SFT 购买正确率为 118/200（59.0%），归一化结果包含 117 条 Gold 和 1 条有效替代；Eval v3 的环境严格 Gold 为 114/200（57.0%）。这两个指标不能混用。24 案例 Judge 校准中，五维分数一致率为 84.17%，证据核验单项为 62.5%；过程分数适合辅助分析，不能当作独立验证的真实标签。

`docs/01`—`04` 保留早期评测说明、暂定结果、案例及问题记录，其中已注明的暂定 Rubric 数字仍按历史状态保留。

## 产物与来源

仓库保留摘要、逐任务评分与 provenance，不包含原始轨迹、采集事件、中间 Judge 请求、review queue、完整 checkpoint 或模型权重。曾经未正确加载 LoRA 的 step 250 / step 300 结果已排除；本页仅列显式加载 LoRA 后保存的结果。

GRPO step 250/500 LoRA 的既有发布入口为 [GitHub Release](https://github.com/clueingforlook/shopping-grpo-longhorizon/releases/tag/wlx-grpo-lora-v1)。外部标签以及 JSON / JSONL / CSV / SVG 中的历史名称、来源路径和哈希保持原值；历史来源路径并不表示对应产物已包含在本地。完整复现所需材料见 [实验概览](../docs/experiments.md)。
