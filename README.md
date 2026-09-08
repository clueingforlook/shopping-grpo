# Shopping GRPO

基于 ShopSimulator 的购物智能体后训练与评测项目，参考 [YYHDBL/shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 开发。项目围绕 **数据采集 → LoRA SFT → Step-GRPO → 轨迹评测** 展开，保留基础训练方案和各阶段实验记录。

当前主线包括自研 Harness、SFT 数据流水线、Reward v4 的步骤级奖励与优势计算，以及 Base、SFT、GRPO 的固定任务评测。结构整理只调整命名、模块位置和阅读入口，训练任务、奖励定义及既有实验结果保持原有内容。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 了解模块职责和完整流程 | [文档导航](docs/README.md) |
| 查看做过的实验、配置与结论 | [实验概览](docs/experiments.md) |
| 查看真实结果和训练曲线 | [结果索引](results/README.md) |
| 确认数据规模、划分和来源 | [数据说明](data/README.md) |
| 查找采集、训练和评测命令 | [脚本索引](scripts/README.md) |
| 区分基础 GRPO 与 Step-GRPO 配置 | [配置说明](configs/README.md) |
| 查看改名对照、历史兼容与验证结果 | [迁移说明](docs/migration.md) |

## 主要功能

- **环境交互**：使用结构化工具调用搜索商品、核验详情、选择规格和购买；Harness 统一检查动作、控制上下文、隔离隐藏答案并记录轨迹。
- **SFT 数据构建**：任务规划、难度校准、并发采样、技术重试、严格 Gold 清洗、按任务切分与哈希冻结。
- **LoRA SFT**：基于 Transformers + PEFT，只对 Assistant Token 计算训练损失。
- **Step-GRPO**：基于 veRL，将终局 ORM 与定位到具体动作的 PRM 结合为 token 级优势；不使用 Value Model 或学习式 Reward Model。
- **评测与分析**：Eval v2 保留 Rubric 和 LLM Judge 分析；Eval v3 使用环境结构化终局，报告严格 Gold、购买率与失败分项。

SFT 和评测使用消息级 Harness 运行流程；GRPO 保留 veRL 的 token 级循环，通过桥接共享工具、规则和数据契约。

## 项目结构

```text
configs/                      基础 GRPO、Step-GRPO 与工具配置
data/                         冻结的训练和评测数据
docs/                         设计、流程和实验说明
environments/ShopSimulator/    内置购物环境和商品压缩档
examples/                     小型示例
patches/                      veRL 运行时补丁
results/                      已保存的实验指标、报告和曲线
scripts/                      采集、训练、评测与维护入口
src/shopping_grpo/
  harness/                    统一交互、SFT 流水线和轨迹评测
  environment/                环境客户端与动作契约
  training/                   SFT 和 GRPO 训练实现
  evaluation/                 评测与统计模块
tests/
  harness/                    Harness 与数据流水线测试
```

## 快速检查

基础包要求 Python 3.10 及以上。先安装 [uv](https://docs.astral.sh/uv/)，再在仓库根目录执行：

```bash
uv sync --extra dev
uv run shopping-grpo smoke
uv run pytest
```

`shopping-grpo smoke` 是不需要模型、GPU 或在线 ShopSimulator 的 CPU 冒烟入口。`pytest` 用于运行测试集；涉及可选依赖或服务的检查需要先准备对应环境。

## 训练和在线评测

Bash 启动脚本和完整训练环境面向 Linux / WSL。SFT、GRPO 及本地模型推理需要 CUDA GPU；已记录的 Step-GRPO 方案使用单张 NVIDIA RTX PRO 6000 96GB。`setup.sh` 默认分别使用 Python 3.12 和 3.10 创建训练与 ShopSimulator 环境。

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
```

ShopSimulator 默认地址为 `http://127.0.0.1:5700`，另开终端可运行 `uv run python scripts/smoke_shop_env.py` 检查接口。

使用仓库公开 SFT 快照进行训练的示例：

```bash
uv run python scripts/train_lora_sft.py \
  --model Qwen/Qwen3.5-2B \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output outputs/models/sft-lora
```

Step-GRPO 主入口为 `scripts/rl.sh`，对应配置是 `configs/step_grpo.yaml`。准备好 SFT merged 模型或可合并的 Base / SFT adapter 后，按 [RL 配置说明](docs/rl-config.md) 运行 `preflight`、`smoke`、`train` 或 `resume`。`scripts/train_grpo.py` 与 `configs/grpo.yaml` 保留基础 GRPO 路径。

## 已保存的主要结果

Eval v3 在同一组 200 个任务上读取 ShopSimulator 的严格 Gold 结果，固定以 200 为分母：

| 模型 | Gold 成功 |
|---|---:|
| Base | 3/200（1.5%） |
| SFT | 114/200（57.0%） |
| GRPO step 250 LoRA | 122/200（61.0%） |
| GRPO step 500 LoRA | 123/200（61.5%） |

SFT 到 step 500 提升 4.5 个百分点。上述是已有实验记录，仍包含各次运行报告的基础设施无效任务；完整数量和口径见 [结果索引](results/README.md)。Eval v2 的 SFT 购买正确率为 59.0%，采用另一套结果归一化与 Judge 分析口径，应与 Eval v3 分开阅读。

当前仓库包含冻结任务、结果摘要、逐任务评测结果和训练指标；原始采样轨迹、完整运行日志、模型权重等不在本地快照中。公开 SFT 数据划分与早期实验冻结记录也有差异，详见 [数据说明](data/README.md)。因此可直接检查现有指标，重跑历史实验还需补齐其来源数据和模型产物。

该 Final-200 已参与评测和奖励设计讨论，属于固定开发评测集。训练生成的模型、检查点和轨迹写入被忽略的 `outputs/` 目录。
