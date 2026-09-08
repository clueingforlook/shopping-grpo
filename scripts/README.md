# 脚本索引

所有命令从仓库根目录执行。Python 脚本通过 `uv run python scripts/<脚本名> --help` 查看参数；Bash 启动脚本面向 Linux / WSL。模型训练与本地推理需先准备相应的 CUDA 环境。

## 主线入口

| 阶段 | 入口 | 用途 |
|---|---|---|
| 环境准备 | `setup.sh` | 安装训练依赖、准备隔离的 ShopSimulator 环境、解压商品与构建索引 |
| 环境启动 | `start_environment.sh`、`smoke_shop_env.py` | 启动服务与检查接口 |
| SFT 数据 | `sft_data_pipeline.py` | 预计算、校准计划、拟合难度、正式计划、采样与构建 |
| SFT 辅助 | `sft_task_planner.py`、`sft_prepare_final_input.py`、`sft_organize_data.py` | 任务规划、最终输入准备与数据归档 |
| LoRA SFT | `train_lora_sft.py` | Transformers + PEFT 训练 |
| 模型合并 | `merge_lora_adapter.py` | 将指定 LoRA 合并到其对应 Base |
| Step-GRPO | `rl.sh` → `train_rl.py` | `preflight`、`smoke`、`train`、`resume`，使用 Reward v4 和步骤级优势 |
| RL 数据 | `prepare_rl_data.py` | 从冻结的任务和难度材料生成 20/60/20 任务池 |
| 模型服务 | `serve_evaluation_model.sh` | 启动评测模型，可显式加载 LoRA |
| Eval v3 | `run_eval_v3.py` | `run` 在线评测；`score` 对已有轨迹离线评分 |
| 训练曲线 | `plot_grpo_metrics.py` | 从完整日志生成 SVG、CSV 和摘要 |

准备好模型产物和 ShopSimulator 后，Step-GRPO 的常用命令为：

```bash
bash scripts/rl.sh preflight
bash scripts/rl.sh smoke --run-name rl-v4-smoke-24k-noentropy
bash scripts/rl.sh train --run-name rl-v4-main-24k-noentropy
bash scripts/rl.sh resume --run-name rl-v4-main-24k-noentropy --total-steps 500
```

`preflight` 仍需能找到 merged 模型或用于合并的 Base / SFT adapter，不是纯 CPU 冒烟命令。模型路径、服务地址和续训限制见 [RL 配置](../docs/rl-config.md)。纯 CPU 冒烟入口为 `uv run shopping-grpo smoke`。

## Eval v2 与轨迹分析

| 任务 | 脚本 |
|---|---|
| 运行模型、离线评测与报告 | `run_model_evaluation.py`、`run_offline_evaluation.py`、`evaluate_trajectories.py` |
| Eval v2 流程 | `run_eval_v2.sh`、`generate_eval_rubrics.py`、`run_eval_judge.py` |
| 难度与成对比较 | `label_evaluation_difficulty.py`、`compare_evaluation_runs.py` |
| Judge 校准与人工 Gold 管理 | `calibrate_eval_judge.py`、`prepare_eval_judge_gold.py`、`build_eval_v2_gold.py`、`apply_eval_judge_gold_review.py`、`revise_eval_judge_gold.py`、`confirm_eval_judge_gold.py`、`finalize_eval_judge_gold.py` |

Eval v2 需要对应的 Rubric、Judge 配置和源轨迹；Eval v3 使用环境结构化终局。历史摘要不替代这些源材料，具体流程见 [Eval 设计](../docs/eval.md)和 [Eval v3](../docs/eval-v3.md)。

## 保留的基础路径与维护工具

- `collect_sft_data.py`：基础 SFT 采集入口。
- `train_grpo.py`：基础 GRPO 训练，读取 `configs/grpo.yaml`；与主线 `train_rl.py` 区分。
- `serve_model.sh`、`evaluate_shop_benchmark.py`、`export_grpo.sh`：基础模型服务、评测和导出路径。
- `check_grpo_runtime.py`、`apply_verl_dynamic_sampling_patch.py`：veRL 运行时检查与补丁。
- `prepare_github_release.py`：发布材料整理。

详细实验设置和历史产物要求见 [实验概览](../docs/experiments.md)。脚本名称去除了个人前缀，已有结果中的版本标识与运行来源仍按历史记录保留。
