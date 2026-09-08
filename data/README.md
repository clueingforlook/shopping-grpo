# 数据说明

该目录保存仓库公开的数据快照与两套 GRPO 任务划分。生成轨迹和模型训练产物写入 `outputs/`，冻结数据的规模与 SHA-256 记录在相邻的 `metadata.json` 中。

## 当前目录

| 用途 | 文件 | 数量 | 说明 |
|---|---|---:|---|
| 公开 SFT 快照 | `sft/train.jsonl`、`sft/validation.jsonl` | 379 / 49 | 共 428 条，当前文件行数与元数据哈希一致 |
| 基础 GRPO | `grpo/train.parquet`、`grpo/validation.parquet` | 1,000 / 50 | 同目录保留 JSONL，对应基础训练入口 |
| Step-GRPO | `step-grpo/train.parquet`、`step-grpo/validation.parquet` | 1,000 / 50 | 对应 `scripts/rl.sh` 与 `configs/step_grpo.yaml` |
| 固定评测 | `evaluation/tasks.jsonl` | 200 | Base、SFT、GRPO 共用 Final-200 |
| 环境契约 | `environment.json`、`step-environment.json` | — | 各运行路径使用的环境版本与接口记录 |

两套 GRPO 数据文件有不同的冻结哈希，不能仅因数量相同就互换。Step-GRPO 的固定难度配比为：

| 划分 | easy | medium | hard | 合计 |
|---|---:|---:|---:|---:|
| Train | 200 | 600 | 200 | 1,000 |
| Validation | 10 | 30 | 10 | 50 |

Step-GRPO 元数据记录 seed=42，构建时排除 428 个 SFT 和 200 个评测任务，共 628 个 `task_id`。难度标签只用于采样与分组统计，不进入 Reward。

## SFT 历史冻结与当前快照

[SFT 数据流水线](../docs/sft-data-pipeline.md)记录的第一版实验包含：200 个难度校准任务、600 次有效尝试、3,380 条原始历史轨迹，最终严格 Gold SFT 共 428 条，Train / Validation 为 **385 / 43**。

当前公开 `data/sft/` 的实测划分则为 **379 / 49**。两者总量都是 428，但本地不包含早期 `outputs/` 原始归档，不能仅凭总数相同确认样本和切分的对应关系。本次整理保留这两组事实，不重新划分数据，也不把公开快照表述为早期模型实验的完整复现输入。

自研采集与构建入口是 `scripts/sft_data_pipeline.py`；`scripts/collect_sft_data.py` 保留基础采集路径。新数据应通过采样、泄漏检查、清洗和冻结流程生成新的版本，再用于后续训练。

## 评测集与来源

Final-200 的 SHA-256 为：

```text
2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208
```

元数据记录环境 `shopsimulator-environment-v2.1`、环境奖励记录版本 `shopsimulator-reward-v3`，与训练任务重叠数为 0。该评测集已参与模型比较及 Reward 设计讨论，属于固定开发评测集；任务不重叠不等同于完全未参与设计的独立测试集。

`step-grpo/metadata.json` 中保存的难度版本、Prompt 版本和源文件哈希是历史冻结标识，继续保留原值。任务文件迁移位置没有改变数据字节。

## 本地材料范围

ShopSimulator 商品压缩档位于 `environments/ShopSimulator/shop_env/data/`，由 `scripts/setup.sh` 解压并建立搜索索引。历史 SFT 原始轨迹、难度校准特征和拟合模型、完整采集 provenance 不包含在当前快照中。

已有 `step-grpo/` 冻结文件可直接读取；重新运行 `scripts/prepare_rl_data.py` 还需要其指定的公开任务、难度特征与冻结难度模型。数据和模型复现边界见 [实验概览](../docs/experiments.md)。
