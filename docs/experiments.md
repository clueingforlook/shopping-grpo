# 实验概览

项目参考 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon)，继续使用 ShopSimulator，逐步增加自研交互控制、SFT 数据构建、步骤级强化学习和多版本评测。本页汇总已有文档与本地冻结结果，详细设计和历史记录继续保留。

## 实验主线

1. **数据阶段**：用 Teacher 在训练任务池采样，校准任务难度，清洗并冻结严格 Gold SFT 数据。
2. **SFT 阶段**：以 Qwen3.5-2B 为基础模型执行 LoRA SFT，学习完成购物任务所需的工具动作序列。
3. **评测与失败分析**：对 Base / SFT 运行 Final-200；Eval v2 分析需求满足、购买结果和过程能力。
4. **RL 阶段**：针对规格选择、错误购买和持续无进展，设计 Reward v4 与 Step-GRPO，从 SFT merged 模型继续训练。
5. **结构化比较**：Eval v3 使用同一 Final-200，比较 Base、SFT 和正确加载 LoRA 的 GRPO step 250 / 500。

基础 `train_grpo.py` 及其配置继续保留。本文的 RL 结果对应 `train_rl.py` / `rl.sh` 的 Step-GRPO 路径，不能仅凭入口名相似就视为同一方案。

## SFT 数据与实验材料

早期冻结文档记录 Teacher 为 `deepseek-v4-flash`、Thinking 开启；私有 `reasoning_content` 仅保留在 Raw，最终 SFT 移除。环境为 Environment v2.1 / Reward v3 / Observation v2 / Tool Schema v2；最终序列使用固定 revision 的 Qwen3.5-2B tokenizer 和官方 Chat Template，长度上限 24,576 Token。

| 阶段记录 | 数量 |
|---|---:|
| 难度校准 | 200 任务 × 3 次有效尝试 = 600 |
| 完整原始历史 | 3,380 条轨迹 |
| 第一版严格 Gold SFT | 428 条，Train / Validation = 385 / 43 |
| 当前公开 `data/sft/` | 428 条，Train / Validation = 379 / 49 |

早期冻结记录与当前公开快照的切分不同，本地缺原始归档，不能据总量相同断言二者的样本和切分关系。训练脚本的默认参数也不等于缺失的历史 SFT 运行 manifest；本页不据此补造当时的训练配置。详见 [数据说明](../data/README.md)和 [SFT 流水线](sft-data-pipeline.md)。

## Step-GRPO 设置

| 项目 | 已记录方案 |
|---|---|
| 硬件 | 单张 NVIDIA RTX PRO 6000 96GB |
| 模型 | Qwen3.5-2B + SFT merged checkpoint，LoRA r=16 / alpha=32 |
| 算法 | Reward v4 + Step-GRPO；无 Value Model 或学习式 Reward Model |
| 训练 / 验证任务 | 1,000 / 50，easy / medium / hard 为 20% / 60% / 20% |
| 批次 | 2 任务 × 4 轨迹 = 8 轨迹/更新 |
| 采样 | temperature 0.7、top_p 0.9 |
| 优化 | 学习率 `1e-6`、warmup 3%、PPO clip 0.2、token-mean |
| 长度与步骤 | 总长度 24,576 Token，最大环境 step 35，单轮生成上限 512 Token |
| 关闭项 | KL reward、KL loss、entropy 计算、Step-LATA |
| 训练长度 | 默认 300 步；已保存指标覆盖延长后的 500 步 |
| 保存 / 验证 | 每 50 步；训练开始前验证 |

ORM 对完整 Gold、部分购买、未购买和错误类目分别评分。PRM 只定位到持续无进展搜索/翻页、错误规格选择或明确错误购买的动作，直接修正该 step 的 token 优势；环境 observation token 不参与更新。无效轨迹 Mask，不参与组内均值。详细公式保持在 [Reward 设计](rl-reward.md)。

固定验证集 Gold：step 0 为 48%；250/500 为 52%；400/450 为 54%。仓库保存了 step 250/500 的正式 Eval v3 结果，没有相应的 step 400/450 正式结果，不能称 step 500 为验证集最优。

## Eval v3 的主要结论

固定评测条件记录为 temperature=0、top_p=1、seed=42、thinking 关闭、上下文 24,576、最大环境 step=35、最大 assistant turn=45。四个模型使用同一组 200 个任务；Base / SFT 的 Eval v3 来自已有轨迹的离线评分，GRPO 重新生成轨迹。

| 模型 | Gold | ASIN 匹配 | 完成购买 |
|---|---:|---:|---:|
| Base | 3/200（1.5%） | 3/200（1.5%） | 3/200（1.5%） |
| SFT | 114/200（57.0%） | 142/200（71.0%） | 185/200（92.5%） |
| GRPO step 250 LoRA | 122/200（61.0%） | 149/200（74.5%） | 184/200（92.0%） |
| GRPO step 500 LoRA | 123/200（61.5%） | 149/200（74.5%） | 188/200（94.0%） |

SFT 的主要收益是把搜索和候选操作继续执行为购买。Step-GRPO step 500 的严格 Gold 比 SFT 高 4.5 个百分点；ASIN 匹配率仍高于 Gold，说明找到目标商品后，规格等终局条件仍是重要失败来源。

当前结果包含各模型分别为 9 / 3 / 2 / 1 的基础设施无效任务，以固定 200 为分母保留；不是补跑故障后的新结果。详细分项、同任务改善/退化统计和源 JSON 链接见 [结果索引](../results/README.md)。

## Eval v2 的作用与边界

Eval v2 保留自然语言 Rubric、LLM Judge、归一化购买结果和过程分析。Base / SFT 购买正确为 3/200 与 118/200；SFT 的 118 包含归一化后的 117 Gold 和 1 有效替代，不能与 Eval v3 的环境严格 Gold 114 混算。

最终分析中，SFT 首要失败以规格错误 44 条、循环无进展 11 条、超预算 10 条为主，成为 Reward v4 的设计依据。最新 V4 Pro / Prompt v2 / Gold v6 的 24 案例校准，五维分数一致率为 84.17%，但证据核验单项只有 62.5%；Gold 和 Prompt 也参与过开发，因此这些过程评分只作为分析证据。

## 复核与复现边界

- **可在本地复核**：公开训练/评测数据与哈希、Eval v2/v3 摘要、Eval v3 逐任务分项、GRPO 训练/验证 CSV 与曲线。
- **环境材料已内置**：ShopSimulator 源码与商品压缩档；安装依赖、解压和建索引后可启动环境。
- **完整重跑还需补齐**：历史 SFT 原始归档及精确切分、难度特征和拟合模型、SFT adapter / merged 权重、原始评测轨迹及运行 manifest、完整训练日志或 checkpoint。当前 `outputs/` 材料没有随仓库快照提供。

Final-200 已参与评测和 Reward 设计讨论，属于固定开发评测集。现有结果没有重复随机种子实验或独立测试集证据；尤其 step 250→500 只净增加 1 个 Gold，不应解释为已确认的稳定泛化收益。

结果文件中的历史名称、旧服务器路径、协议版本和 SHA-256 继续保留，用于来源核对；目录重命名不会生成缺失的产物，也不会改变原实验数值。
