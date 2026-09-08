# 命名与目录整理说明

本次整理移除文件名、Python 模块、类、函数、常量及启动环境变量中的个人前缀，并把 Harness 纳入可安装的 `shopping_grpo` 包。原有基础 GRPO 与后续 Step-GRPO 两条路径分别保留。

## 主要路径对照

| 原路径 | 当前路径 |
|---|---|
| `wlx-docs/` | `docs/` |
| `wlx-results/` | `results/` |
| `wlx-harness-core/wlx_harness_core/` | `src/shopping_grpo/harness/` |
| `wlx-harness-core/wlx-tests/` | `tests/harness/` |
| `wlx-harness-core/wlx-tools.json` | `configs/harness_tools.json` |
| `wlx-harness-core/wlx-harness-design.md` | `docs/harness/design.md` |
| `wlx-harness-core/wlx-harness-usage.md` | `docs/harness/usage.md` |
| `configs/wlx_grpo.yaml` | `configs/step_grpo.yaml` |
| `configs/wlx_agent_loop.yaml` | `configs/step_agent_loop.yaml` |
| `data/wlx-grpo/` | `data/step-grpo/` |
| `data/wlx-environment.json` | `data/step-environment.json` |
| `scripts/wlx_rl.sh` / `scripts/wlx_train_rl.py` | `scripts/rl.sh` / `scripts/train_rl.py` |
| `evaluation/wlx_eval_v3.py` | `evaluation/eval_v3.py` |
| `training/grpo/wlx_reward_v4.py` | `training/grpo/reward_v4.py` |
| `training/grpo/wlx_step_grpo.py` | `training/grpo/step_grpo.py` |

其余脚本、测试、文档和结果文件移除文件名中的 `wlx_` 或 `wlx-`。共迁移 167 个文件路径；原 324 个文件全部保留。目录说明分别见 [脚本索引](../scripts/README.md)、[配置说明](../configs/README.md) 和 [结果索引](../results/README.md)。

Python 导入示例：

```python
from shopping_grpo.harness import EpisodeRunner, HarnessConfig, SftDataPipeline
from shopping_grpo.training.grpo.reward_v4 import compute_orm, compute_prm
```

`WLX_GRPO_*` 环境变量改为 `GRPO_*`，`WLX_SHOPPING_*` 改为 `SHOPPING_*`；教师与评测相关变量同样去掉 `WLX_` 前缀。启动器与 YAML 已同步。Step-GRPO 的 advantage estimator 名称为 `step_grpo`，可选配置节为 `step_lata`。

## 历史记录的兼容边界

- 冻结数据、JSON/JSONL 结果、CSV 指标和原始 SVG 曲线保持原字节。历史模型名、来源路径、版本标识、哈希和图内标签用于追溯，仍保留当时的文字。
- `schema_version`、Prompt 版本、难度版本及训练记录字段属于持久化协议。例如 `wlx-reward-v4`、`wlx_reward/*` 和 `wlx_step_*` 字段继续可读，Python 符号本身已改为正常名称。
- 现有外部 Release URL 和标签保持原地址；本地整理不会修改已发布的远端附件。
- 新运行的目录和文件名使用无前缀名称。旧 `outputs/` 不在当前快照中，没有自动搬迁；若另有旧产物，应先保留原副本，再按新入口显式指定输入路径。历史归档中的来源路径不能直接当作当前仓库路径。
- 训练续跑会核对源代码与配置的指纹。重构后的代码不能直接承诺通过旧运行目录的 `resume` 检查；需要延续原实验时使用原代码快照及对应产物，不应删除或绕过指纹保护。

基础 `configs/grpo.yaml` 与 `data/grpo/` 保持原用途。Step-GRPO 配置、任务和环境契约使用独立名称，避免覆盖两套不同的冻结内容。

## 验证记录

在当前 Windows / Python 3.13 环境完成以下检查，未启动 GPU 训练、模型服务或新的正式评测：

| 检查 | 结果 |
|---|---|
| 原文件完整性 | 324 个原文件均有对应文件 |
| 冻结内容校验 | 72 个数据/结果等文件 SHA-256 与备份完全一致 |
| 提示词校验 | SFT、Judge、Rubric 的实际提示词 SHA-256 与原版一致 |
| Python 静态检查 | 205 个原 Python 文件可解析；本地 import 和动态模块目标均可定位 |
| 原通过测试 + 新增迁移测试 | 213 + 8 = **221 项通过** |
| CPU 冒烟 | 5 项通过 |
| 主要 Python 入口 `--help` | 11 个入口通过 |
| 离线 wheel | 44 个 Harness 模块完整打包；隔离安装后导入与工具注册表验证通过 |

完整测试在整理前为 **213 passed / 15 failed / 1 skipped / 11 errors**；整理后为 **221 passed / 15 failed / 1 skipped / 11 errors**。原有失败涉及 observation/rollout 断言、缺少 `hydra`/`transformers` 等依赖、旧 reward 测试接口不匹配，以及当前 Windows 无 `fcntl`、未安装 `torch`/`verl`。本次没有通过修改奖励、训练逻辑或削弱断言来消除这些既有问题。

Harness 的纯 Python 接口已可从安装包独立导入；正式 SFT 使用 wheel 且不在仓库目录运行时，仍需显式传入评测隔离任务的 `held_out_tasks_path`。

本地原始备份和审计报告保存在仓库外、同级的 `.refactor-audit/`：`original-project.zip` 是整理前快照，`original-files.json` 保存完整路径对照与原始哈希，测试、wheel 和提示词校验报告也在该目录。该目录是本地审计材料，不是训练产物。
