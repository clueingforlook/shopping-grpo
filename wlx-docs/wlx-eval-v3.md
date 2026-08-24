# WLX Eval v3：确定性终局评测

## 1. 目的

`wlx-eval-v3` 用 ShopSimulator 的结构化终局记录评价一个指定模型，不调用 LLM Judge，也不使用旧 Rubric。一次可以只运行 Base、SFT 或 GRPO 中的一种。

核心问题只有两个：

1. 模型是否买到环境定义的 Gold 商品及正确规格；
2. 没有命中 Gold 时，错在 ASIN、品类、属性、规格还是价格。

Eval v3 是评测协议版本，与 ShopSimulator 的 `Reward v3` 不是一回事。

## 2. 固定评测集

- 任务文件：`data/evaluation/tasks.jsonl`
- 元数据：`data/evaluation/metadata.json`
- 任务数：固定 200 个 `task_id`
- 任务文件 SHA256：`2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208`
- 环境：`shopsimulator-environment-v2.1`
- 数据记录显示与训练集重叠数为 0

这就是 SFT 后 Final-200 使用的同一批任务，不重新抽样、不替换任务。

这 200 个任务已经参与过评测和 Reward 设计讨论，因此属于固定开发评测集。它适合比较 Base、SFT、GRPO；若以后需要严格的最终论文测试，还应另留一批完全未参与设计的任务。

## 3. Gold 判定不变

Gold 完全沿用环境原生结果：

> 最终购买的 ASIN 正确，并且任务要求的规格选项也全部正确，才记为 Gold 成功。

实现时直接读取环境终局的 `gold_purchase` / `purchase_success`，不自行重写 Gold 规则。

非 Gold 商品即使很相似，也不能算 Gold；它只在下列分项指标中获得部分满足结果。

## 4. 评测指标

### 4.1 主指标

| 指标 | 含义 |
|---|---|
| `gold_rate` | Gold 成功任务数 ÷ 200；最终选 checkpoint 的首要指标 |
| `purchase_rate` | 完成购买的任务比例 |
| `no_purchase_rate` | 没有完成购买的任务比例 |
| `asin_match_rate` | 最终购买 ASIN 与 Gold ASIN 相同的任务比例 |

### 4.2 失败拆解

| 指标 | 环境结构化依据 | 含义 |
|---|---|---|
| `category_pass_rate` | `hard_gates.category` | 购买商品品类是否满足任务 |
| `attribute_ratio` | `brand`、`model`、`core_functions` 的通过数/要求数 | 品牌、型号和核心属性满足比例 |
| `option_ratio` | `key_options` 的通过数/要求数 | 颜色、尺寸、容量、套装等所选规格满足比例 |
| `price_pass_rate` | `hard_gates.budget` 与最终规格实际价格 | 是否满足任务预算 |
| `all_constraints_rate` | 品类、全部属性、全部规格和价格同时通过 | 非 Gold 购买是否完整满足结构化任务要求 |
| `unverifiable_rate` | 环境的 `unverifiable` 状态 | 环境记录不足，无法可靠计算分项的比例 |

属性和规格必须分开报告：属性是商品本身的品牌、型号和功能；规格是购买时实际选择的颜色、尺码、容量等选项。

### 4.3 没有购买时

没有购买的任务统一记为：

- `Gold = 0`
- `ASIN Match = 0`
- 品类、属性、规格、价格和全部约束的任务级得分均为 0
- 同时计入 `no_purchase_rate`

这样模型不能通过“不购买”来回避错误，也不会提高任何结果满足率。

## 5. 运行规则

每个 Eval v3 Run 只对应一个模型角色：

1. `base`；
2. `sft`；
3. `grpo`，通常使用固定 validation Gold 选出的 checkpoint。

不要求一次把三种模型全部跑完。只评 GRPO 时，直接建立一个 GRPO Run；以后需要比较时，再用相同任务和运行设置对齐不同 Run。

正式 Run 沿用 Final-200 的运行口径：

- `temperature = 0.0`
- `top_p = 1.0`
- `seed = 42`
- `enable_thinking = false`
- 上下文窗口 24576
- `max_steps = 35`
- `max_assistant_turns = 45`
- 同一 System Prompt、工具 Schema、Observation 配置和环境版本

旧 Base/SFT Final-200 的完整轨迹已经保存，可以分别离线生成 Eval v3 指标；GRPO 按相同口径新生成 200 条轨迹。只有需要做跨模型比较时，才要求各 Run 的运行合同一致。

## 6. 统计口径

- 主结果固定以 200 为分母，并同时报告分子，例如 `108/200 = 54%`。
- 基础设施故障单独报告，不解释为模型错误；正式结论前应在新的 Run 中补跑对应任务。
- `unverifiable` 不能当作通过；单独报告数量，并在固定分母指标中按未通过处理。
- 除总体结果外，继续按 easy、medium、hard 分难度报告。
- 逐任务按相同 `task_id` 比较 Base→SFT、SFT→GRPO 的改善、退化和不变数量。

## 7. 输出文件

每种模型写入自己的独立目录，不覆盖旧 Eval，也不要求同时存在：

```text
outputs/evaluation/eval-v3/
├── wlx-base/             # 可选
├── wlx-sft/              # 可选
└── wlx-grpo-step-XXX/    # 可单独运行
```

每个模型至少保存：

- `wlx-eval-v3-manifest.json`：模型、任务集、环境和运行配置指纹；
- `wlx-eval-v3-task-results.jsonl`：逐任务 Gold 和各结构化分项；
- `wlx-eval-v3-summary.json`：总体及分难度统计；
- `wlx-eval-v3-summary.md`：便于阅读的简表。

跨模型比较是后续可选步骤，不属于单模型 Eval v3 的必需产物。

## 8. 实现与启动

入口脚本：`scripts/wlx_run_eval_v3.py`。所有新增脚本、文件和输出名均使用 `wlx_` 或 `wlx-` 前缀。

### 8.1 只评 GRPO

先从 checkpoint 提取独立 LoRA。当前 step 500 已提取到
`outputs/models/wlx-grpo-step500-lora`。使用冻结的 SFT merged 模型作为底座，
并显式加载 GRPO LoRA：

```bash
bash scripts/wlx_serve_evaluation_model.sh \
  --base-model outputs/models/wlx-sft-own-data-v1-merged \
  --lora-adapter outputs/models/wlx-grpo-step500-lora \
  --served-model wlx-grpo-step500
```

环境与模型服务启动后，一条命令运行 Final-200 并生成 Eval v3：

```bash
.venv/bin/python scripts/wlx_run_eval_v3.py run \
  --model-role grpo \
  --model-name wlx-grpo-step500-lora \
  --served-model wlx-grpo-step500 \
  --model-artifact outputs/models/wlx-grpo-step500-lora \
  --tokenizer outputs/models/wlx-sft-own-data-v1-merged \
  --output-dir outputs/evaluation/eval-v3/wlx-grpo-step500-lora
```

命令支持断点续跑：中断后原样重跑，只补尚未完成的任务。正式运行默认校验 Final-200 的任务数量和 SHA256。

完成后查看简明报告：

```bash
less outputs/evaluation/eval-v3/wlx-grpo-step500-lora/wlx-eval-v3-summary.md
```

逐任务结果保存在 `wlx-eval-v3-task-results.jsonl`，总体和分难度统计同时保存在 `wlx-eval-v3-summary.json`。

### 8.2 对已有轨迹离线评分

离线评分不启动模型或环境，例如只生成旧 SFT 的 Eval v3：

```bash
.venv/bin/python scripts/wlx_run_eval_v3.py score \
  --model-role sft \
  --model-name wlx-sft-final200-v2 \
  --trajectories outputs/evaluation/wlx-sft-final200-v2/wlx-trajectories.jsonl \
  --run-manifest outputs/evaluation/wlx-sft-final200-v2/wlx-run-manifest.json \
  --output-dir outputs/evaluation/eval-v3/wlx-sft
```

### 8.3 绘制 GRPO 训练曲线

不需要额外绘图库，脚本直接从完整训练日志生成 SVG 和 CSV：

```bash
.venv/bin/python scripts/wlx_plot_grpo_metrics.py \
  --log outputs/models/wlx-rl-v4-main-24k-noentropy/wlx-train.log
```

默认输出到 `wlx-training-plots/`：

- `wlx-training-curves.svg`：ORM、Gold、KL、梯度、有效组和响应长度曲线；
- `wlx-training-metrics.csv`：每个训练 step 的原始指标；
- `wlx-validation-metrics.csv`：固定 validation 的 Gold、ORM 和轮数；
- `wlx-training-curve-summary.json`：最后一个训练点和验证点。

曲线中浅蓝色是训练小批次原值，深蓝色是 10 步滑动平均，红点是固定 validation。判断模型效果以红色 validation 点和 Final-200 为主，不能把单步训练 Gold 当作正式 Eval。

## 9. 与旧 Eval v2 的关系

- Eval v2：使用冻结 Rubric 和 LLM Judge，主要评价自然语言需求与过程质量。
- Eval v3：只使用环境结构化终局，不使用 Rubric、LLM Judge 或人工语义判断。

Eval v3 的主要结论看 `Gold`；品类、属性、规格、价格只用于解释 Gold 成功或失败的原因。旧 Eval v2 结果保留用于追溯，但不与 Eval v3 的指标混算。
