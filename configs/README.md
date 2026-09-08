# 配置说明

仓库保留基础 GRPO 和自研 Step-GRPO 两条训练路径。选择入口时使用与之配套的配置和任务数据。

| 配置 | 使用位置 | 职责 |
|---|---|---|
| `step_grpo.yaml` | `scripts/rl.sh` / `scripts/train_rl.py` | 主线 Reward v4 + token 级 Step-GRPO |
| `step_agent_loop.yaml` | Step-GRPO | 注册步骤级训练使用的购物 Agent Loop |
| `grpo.yaml` | `scripts/train_grpo.py` | 保留的基础 GRPO 方案 |
| `agent_loop.yaml` | 基础 GRPO | 基础购物 Agent Loop |
| `tools.json` | 训练运行路径 | veRL 工具配置；Step-GRPO 启动器也使用此文件 |
| `harness_tools.json` | Harness | Harness 的结构化工具定义 |

环境契约文件位于 `data/environment.json` 和 `data/step-environment.json`。`shopsimulator-reward-v3` 是环境返回数据的版本标识；Step-GRPO 的训练分数由 Reward v4 重新计算。

## Step-GRPO 主配置

| 项目 | 配置值 |
|---|---|
| 模型 | Qwen3.5-2B 的 SFT merged 模型，继续训练 LoRA |
| LoRA | rank 16、alpha 32、all-linear |
| 数据 | `data/step-grpo/`，Train 1,000 / Validation 50 |
| 批次 | 每次 2 个任务，每任务 4 条轨迹 |
| 采样 | temperature 0.7、top_p 0.9 |
| 学习率 | `1e-6`，warmup 3% |
| 长度 | prompt 4,096、response 20,480、总长度 24,576 Token |
| 更新 | PPO clip 0.2、token-mean |
| 未启用 | Value Model、学习式 Reward Model、KL reward/loss、entropy、Step-LATA |
| 保存与验证 | 每 50 步；训练前先验证 |
| 默认训练上限 | 300 步；历史运行续训至 500 步 |

长度、显存、步骤奖励、动态采样及续训指纹共同影响实验可比性。完整说明见 [RL 配置](../docs/rl-config.md)和 [Reward 设计](../docs/rl-reward.md)。

使用启动脚本提供模型、数据和输出路径；YAML 中的环境变量由对应启动器准备。配置默认值与历史实验已执行的步数分开记录，不用修改默认上限来改写旧实验事实。
