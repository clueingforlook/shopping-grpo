# WLX RL 配置方案

> 目标：在单张 NVIDIA RTX PRO 6000 96GB 上，从 WLX SFT 模型继续训练，不使用 Value Model 或学习式 Reward Model。Reward 定义见 [wlx-rl-reward.md](./wlx-rl-reward.md)。

## 1. 主方案

```text
模型：Qwen3.5-2B + SFT merged checkpoint
算法：自定义 step 级 GRPO（wlx_step_grpo）
参数更新：LoRA r=16, alpha=32
Reward：WLX Reward v4（ORM + step PRM）
Value Model：无
Reward Model：无
```

ShopSimulator 环境保持不变，负责执行任务并记录最终购买结果；训练时根据这些结果计算 WLX Reward v4，不把隐藏答案暴露给模型。

`data/wlx-environment.json` 中保留的 `shopsimulator-reward-v3` 只表示环境返回数据的结构版本，用来校验字段是否完整；真正进入训练的分数始终由 WLX Reward v4 重新计算。

## 2. 任务选择

直接复用环境已有任务和 WLX 已有的 `easy / medium / hard` 难度标签，不生成新任务。当前已冻结为：

```text
data/wlx-grpo/wlx-train.parquet
data/wlx-grpo/wlx-validation.parquet
data/wlx-grpo/wlx-metadata.json
```

```text
固定训练池：1,000 题
  easy   200（20%）
  medium 600（60%）
  hard   200（20%）

固定验证集：50 题
  easy    10
  medium  30
  hard    10
```

要求：

- 训练池、验证集和 `data/evaluation/tasks.jsonl` 按 `task_id` 严格互斥；
- 每个难度内部均匀采样，一轮内尽量不重复；
- 难度标签只用于分层采样，不进入 Reward；
- 每个任务一次生成 4 条轨迹，4 条必须使用相同任务和不同采样随机性。

Medium 占比最高，因为它最容易同时产生成功和失败轨迹，组内优势最有效；Easy 和 Hard 用于保持基础能力与困难任务覆盖。

## 3. Step 级实现

当前实现不直接使用轨迹末尾标量训练，而是记录准确 step span，再生成 token-level advantage。

### 3.1 Rollout 记录

每次 assistant 生成前后记录 `response_mask` 长度，得到该 step 的模型 token 区间：

```text
step_span = [start_token, end_token)
```

同时保存：

```text
step_spans       每个 step 的 token 区间
step_tool        工具名和必要参数
step_prm         对应 PRM 修正
step_rule        触发的规则名，便于审计
```

现有 veRL `response_mask` 已经把模型 token 标为 1、环境 observation 标为 0，继续沿用，不重新分词猜边界。

### 3.2 自定义优势计算

同一任务的有效轨迹先用 ORM 计算组内优势：

```text
A_orm_i = clip((O_i - mean_group) / max(std_group, 0.1), -2, 2)
```

再对每个 step 计算：

```text
A_i,t = A_orm_i + p_i,t
```

- 错误规格 step：`A_i,t = min(A_i,t, -0.25)`；
- 明确错误购买 step：`A_i,t = min(A_i,t, -0.50)`；
- 连续无进展搜索/翻页：只减 `0.25`，不强制变成负数；
- observation token 的 advantage 固定为 0；
- 基础设施无效轨迹整条 Mask，不参与组均值和更新；
- 同组不足 2 条有效轨迹时丢弃该组并重新采样。

PRM 不能先求和到轨迹总分；否则又会退化成轨迹级信用分配。

### 3.3 veRL 接口

项目模块已注册 `wlx_step_grpo` advantage estimator，并在 Ray worker 启动时安装窄范围运行时接口：

1. 将 `step_spans / step_prm / step_rule` 从 `extra_fields` 传入自定义 estimator；
2. 让 estimator 直接返回完整的 token-level `advantages`；
3. 保持 PPO-clip 形式的 actor loss，但不引入 Value Model；
4. 在 worker 启动时显式导入注册模块，避免 Ray 子进程找不到 estimator。

## 4. 动态采样

现有动态采样只保留“轨迹总 Reward 不同”的组，会错误丢掉 ORM 相同但 PRM 不同的有效组。更新后的组保留条件为：

```text
有效轨迹至少 2 条
并且满足以下任一条件：
1. 组内 ORM 不完全相同；
2. 至少一条轨迹存在非零 step PRM。
```

仍保留：

```text
max_num_gen_batches = 3
max_consecutive_skipped_updates = 10
```

如果 ORM 全相同且所有 PRM 都是 0，该组没有学习信号，直接重新采样。

## 5. 主训练参数

| 配置 | 值 |
| --- | ---: |
| `train_batch_size` | 2 个任务 |
| `rollout.n` | 4 条轨迹/任务 |
| 每次更新轨迹数 | 8 |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `ppo_mini_batch_size` | 2 |
| `ppo_micro_batch_size_per_gpu` | 1 |
| `learning_rate` | `1e-6` |
| warmup | 3% |
| PPO clip | `0.2` |
| entropy | 0 |
| KL reward / KL loss | 关闭 / 关闭 |
| LoRA | `r=16, alpha=32, all-linear` |
| 最大环境 step | 35 |
| 单轮最大生成 | 512 token |
| 最大 prompt / response | 4,096 / 20,480 |
| 最大总长度 | 24,576 |
| loss 聚合 | `token-mean` |
| 最大训练步数 | 300 |
| 保存 / 验证频率 | 50 / 50 |
| `val_before_train` | `true` |

两次 OOM 的共同原因是 `calculate_entropy=true`：它在 Qwen3.5 上保留完整词表的 entropy 计算图，明显抬高 actor backward 峰值。原项目产出 GRPO checkpoint 时该项为 `false`，后来才为诊断指标打开。因此 WLX 恢复原项目的 24K 长度，同时关闭该项。

其余显存参数保持：

```text
vLLM gpu_memory_utilization = 0.45
max_num_seqs = 8
agent workers = 8
actor/ref micro batch = 1
gradient checkpointing = true
optimizer_offload = true
param_offload = true
```

先以稳定运行优先，不立即关闭 offload。确认峰值显存有明显余量后，再单独调整 offload，不与 Reward 或学习率同时改动。

## 6. Step-LATA 配置

主配置默认关闭：

```text
wlx_step_lata.enable = false
```

原因：对 SFT Eval 的 200 条轨迹按正式 Qwen3.5 Chat Template 统计，共 1,659 个 assistant step：

```text
P50 = 16 token
P75 = 35 token
P95 = 40 token
Max = 106 token
```

这些 step 基本只有工具调用，没有显式 reasoning；长度差主要来自搜索词和规格字符串，而不是推理深度。此时按长度加权可能错误地削弱长规格值或长搜索词。

Reward 文档保留其公式供以后实验使用；本次实现不启用。如果以后模型开始生成明显的 step 内推理，再实现并启用：

```text
L_ref = 16
w_i,t = clip(sqrt(L_ref / L_i,t), 0.5, 2.0)
A_i,t,k = clip(A_i,t * w_i,t, -2, 2)
```

## 7. 训练监控

| 类别 | 必须记录 |
| --- | --- |
| Reward | ORM 均值/范围、Gold/部分购买/未购买/错误类目比例、各 PRM 触发数 |
| 组内信号 | Reward 标准差、有效组/重采样比例、`std<0.1` 比例 |
| Advantage | 均值/标准差/范围、正/负/零 token 比例、clip 比例 |
| 优化 | policy loss、learning rate、grad norm、PPO ratio、clip fraction、`actor/ppo_kl` |
| Agent | Gold、规格错误率、超预算率、未购买率、平均 step、repeat_loop 率 |
| 系统 | 无效轨迹率、上下文超限率、峰值显存、每步耗时、rollout 吞吐 |

KL 惩罚仍关闭，但记录当前策略与 rollout 旧策略之间的 `actor/ppo_kl`，用于发现单次更新过猛；不为此额外加载 Reference Model。`entropy_coeff=0`，并关闭 `calculate_entropy`：该统计不参与优化，却会在 Qwen3.5 上产生很大的完整词表反向显存峰值。所有其余指标落盘到运行目录，SwanLab 只作可选的在线展示。

验证与选 checkpoint 以严格 Gold 为主，日志中的对应指标为 `wlx_gold`；`wlx_orm` 只作辅助观察，不按训练 Reward 或最后一步直接选模型。

## 8. 保存、验证与断点续训

```text
每 50 步：保存完整 checkpoint
每 50 步：在固定 validation 集上验证
100/200/300 步：作为里程碑 checkpoint 保留
训练结束：按 validation Gold 选 checkpoint，导出后只做一次正式 Eval
```

不在训练中每 100 步使用 `data/evaluation/tasks.jsonl`，否则会把保留测试集变成调参集。

checkpoint 必须包含 LoRA Actor、Optimizer/LR Scheduler、当前 global step 和 Dataloader 状态。默认不自动删除，选完最终模型后再清理。

断点续训规则：

- 同一 `run-name` 继续使用原输出目录；
- `resume` 自动找最新的完整 `global_step_*`；
- 也支持 `--resume-from outputs/.../global_step_100`；
- 恢复前校验模型、任务划分、Reward 配置和主要训练参数的指纹，不一致则拒绝续训；
- 非空目录中没有合法 checkpoint 时拒绝启动，避免覆盖旧结果。

意外中断后从上一个完整 checkpoint 继续，因此最多重做 49 个更新步。

## 9. 实现形式与启动方式

对用户只提供一个入口 `scripts/wlx_rl.sh`，Reward、advantage 和日志则保持为带 `wlx_` 前缀的独立模块：

```bash
# 只检查，不训练
bash scripts/wlx_rl.sh preflight

# 5 步冒烟测试
bash scripts/wlx_rl.sh smoke --run-name wlx-rl-v4-smoke-24k-noentropy

# 正式训练
bash scripts/wlx_rl.sh train --run-name wlx-rl-v4-main-24k-noentropy

# 从该运行最新的完整 checkpoint 继续
bash scripts/wlx_rl.sh resume --run-name wlx-rl-v4-main-24k-noentropy

# 从当前 checkpoint 延长到 500 步
bash scripts/wlx_rl.sh resume \
  --run-name wlx-rl-v4-main-24k-noentropy \
  --total-steps 500
```

本轮不限制完整 checkpoint 数量，因此会保留 step 300/350/400/450/500；训练结束后先导出每个约 45 MB 的 LoRA，再清理不再需要的完整 checkpoint。需要节省更多空间时才使用 `--max-checkpoints 2`。

`preflight` 不合并模型、不启动训练；冻结任务文件缺失时会先按固定 seed 生成。首次执行 `smoke` 或 `train` 时，如果 merged checkpoint 不存在，会自动将 `outputs/models/wlx-sft-own-data-v1-lora` 合并到 `outputs/models/wlx-sft-own-data-v1-merged`。

正式训练前先确认 ShopSimulator 服务已经在 `http://127.0.0.1:5700` 运行；如地址不同，追加 `--env-url URL`。终端只显示每步摘要和错误；完整原始输出保存在运行目录的 `wlx-train.log`。

服务器上用 `screen` 保持会话：

```bash
screen -S wlx-rl
cd /root/autodl-tmp/my-work/shopping-grpo-longhorizon
bash scripts/wlx_rl.sh smoke --run-name wlx-rl-v4-smoke-24k-noentropy
```

smoke 完成 5 步并保存 checkpoint 后，再用上面的 `train --run-name wlx-rl-v4-main-24k-noentropy` 开始正式训练。旧的 `wlx-rl-v4-main` 和 `wlx-rl-v4-smoke-16k` 都在首次反向传播时 OOM，且没有 checkpoint，不能使用 `resume`。

按 `Ctrl+A`，再按 `D` 退出 screen，训练会继续运行。

```bash
screen -ls             # 查看会话
screen -r wlx-rl       # 重新进入
screen -d -r wlx-rl    # 会话被其他终端占用时强制接回
```

`screen` 只防 SSH 断开；如果服务器重启或进程崩溃，使用上面的 `resume` 命令。

## 10. 实施顺序

1. 实现并单测 step span、PRM 定位、自定义 advantage 和 resume；
2. 离线回放现有轨迹，确认惩罚落在正确 step；
3. 运行 preflight 和 5 步 smoke；
4. 确认 Mask、指标、checkpoint 恢复和显存正常后，再启动正式训练。
