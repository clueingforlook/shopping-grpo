# Eval Run：wlx-sft-final200-v2

> 中文名称用于阅读；括号内英文是程序字段名。

## 核心计数

- 任务：200
- 可直接评价：184
- 需要复核：16
- 非模型无效：0
- 格式正确：200/200（100.0%）
- 购买正确：118/200（59.0%）
- 可直接评价任务中的 购买正确：114/184（62.0%）
- 复核队列：52/200（26.0%）

## 结果分布

- 目标商品购买成功（`gold_purchase`）：117
- 未购买（`no_purchase`）：15
- 部分符合（`partial_purchase`）：48
- 无法验证（`unverifiable`）：11
- 有效替代商品（`valid_alternative_purchase`）：1
- 错误购买（`wrong_purchase`）：8

## 环境结果分布

- 目标商品购买成功（`gold_purchase`）：114
- 非法动作达到上限（`invalid_action_limit`）：1
- 部分符合或替代商品（`partial_alternative_purchase`）：57
- 上下文预算错误（`policy:ContextBudgetError`）：3
- 重复循环（`repeat_loop`）：11
- Reward 无法验证（`reward_unverifiable`）：5
- 错误购买（`wrong_purchase`）：9

## Rubric 需求状态

共检查 1428 条用户要求：

- 满足（`satisfied`）：1112
- 未知（`unknown`）：270
- 违反（`violated`）：46

## 第一失败分布

- 动作或上下文预算耗尽（`context_or_action_budget_exhausted`）：3
- 核验不足（`insufficient_verification`）：6
- 非法动作（`invalid_action`）：1
- 无模型失败（`none`）：119
- 超预算购买（`over_budget_purchase`）：10
- 过早购买（`premature_purchase`）：5
- 重复循环无进展（`repeat_loop_no_progress`）：11
- 规格/选项错误（`wrong_option_selection`）：44
- `wrong_product_selection`：1

## 过程质量分数

- 候选利用（`candidate_utilization`）：0分 19，1分 35，2分 146
- 购买决策（`decision_quality`）：0分 55，1分 31，2分 114
- 证据核验（`evidence_verification`）：0分 28，1分 63，2分 109
- 搜索策略（`search_strategy`）：0分 0，1分 38，2分 162
- 终止效率（`termination_efficiency`）：0分 53，1分 20，2分 127

> 过程分数由通过 Gold Set 校准的 Judge 生成；有歧义的样本保留在复核队列。
