# 文档导航

项目主线是：ShopSimulator 提供购物环境，Harness 管理模型与环境的交互，SFT 学习成功轨迹，Step-GRPO 继续优化购物决策，评测模块记录结果和失败原因。

## 推荐阅读顺序

| 顺序 | 文档 | 内容与状态 |
|---|---|---|
| 1 | [实验概览](experiments.md) | 当前做过的实验、主要配置、结果与复现边界 |
| 2 | [Harness 设计](harness/design.md) / [使用说明](harness/usage.md) | 模型、工具、环境适配器、Runner 与阶段接口 |
| 3 | [SFT 数据流水线](sft-data-pipeline.md) | 任务规划、难度校准、采样、清洗与第一版冻结记录 |
| 4 | [RL Reward](rl-reward.md) | Reward v4 的 ORM、步骤 PRM 和信用分配 |
| 5 | [RL 配置](rl-config.md) | Step-GRPO 参数、预检查、冒烟、训练及续训 |
| 6 | [Eval v3](eval-v3.md) | 当前结构化终局评测的指标和运行入口 |
| 7 | [Eval 设计](eval.md) | 包含 Rubric、LLM Judge 与归因的评测设计 |

配套索引：[数据](../data/README.md)、[脚本](../scripts/README.md)、[配置](../configs/README.md)、[实验结果](../results/README.md)。

命名对照与本次验证记录见 [迁移说明](migration.md)。

## 模块边界

- `src/shopping_grpo/harness/`：一次任务的交互契约、动作检查、上下文管理、轨迹记录，以及 SFT 和评测编排。
- `src/shopping_grpo/training/`：训练数据读取、LoRA SFT、veRL Agent Loop、Reward v4 和 Step-GRPO 优势计算。
- `src/shopping_grpo/evaluation/`：评测指标与报告生成。
- `environments/ShopSimulator/`：环境状态、商品数据、搜索和原生终局判定。

SFT 与评测可复用消息级 Runner；GRPO 保留 veRL 自己的 token 级循环，主要共享 Harness 的规则和契约。

## 阅读历史记录时

详细文档保留各阶段形成时的设计和事实，应结合以下状态阅读：

- [Harness 说明大纲](harness-说明.md) 仍是讲解重写大纲，含待展开章节，不代表这些章节已经完成。
- SFT 流水线末尾“只完成数据阶段”描述的是第一版数据冻结时点；项目后续已有 SFT、GRPO 和评测结果。
- SFT 早期冻结记录为 385/43，当前公开 `data/sft/` 为 379/49。两者总数均为 428，来源对应关系见 [数据说明](../data/README.md)。
- RL 配置默认上限 300 步，保存的训练指标实际覆盖延长后的 500 步；step 400/450 的验证 Gold 高于 step 500。
- Eval v1 的暂定 Rubric、Eval v2 的 Judge 分析、Eval v3 的环境 Gold 属于不同版本，结果不合并统计。旧说明和典型轨迹均保留在 [结果文档](../results/docs/)。

当前文件与代码入口采用无个人前缀的名称。冻结结果里的模型名、协议版本、旧服务器路径、哈希及外部 Release 标签保留原值，用于追溯当时的运行；其中历史路径不是当前仓库里的可用文件路径。
