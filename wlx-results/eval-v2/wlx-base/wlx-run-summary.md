# WLX Eval Run：wlx-base-final200-v2

> 中文名称用于阅读；括号内英文是程序字段名。

## 核心计数

- 任务：200
- 可直接评价：197
- 需要复核：0
- 非模型无效：3
- 格式正确：196/200（98.0%）
- WLX 购买正确：3/200（1.5%）
- 可直接评价任务中的 WLX 购买正确：3/197（1.5%）
- 复核队列：8/200（4.0%）

## WLX 结果分布

- 目标商品购买成功（`gold_purchase`）：3
- 未购买（`no_purchase`）：197

## 环境结果分布

- 目标商品购买成功（`gold_purchase`）：3
- 教师输出截断（`internal:TeacherOutputTruncatedError`）：3
- 非法动作达到上限（`invalid_action_limit`）：126
- 助手轮数达到上限（`max_assistant_turns`）：1
- 工具注册错误（`model:ToolRegistryError`）：4
- 上下文预算错误（`policy:ContextBudgetError`）：6
- 重复循环（`repeat_loop`）：57

## Rubric 需求状态

共检查 1428 条用户要求：

- 满足（`satisfied`）：23
- 未知（`unknown`）：1405

## 第一失败分布

- 动作或上下文预算耗尽（`context_or_action_budget_exhausted`）：7
- 非法动作（`invalid_action`）：130
- 无模型失败（`none`）：3
- 重复循环无进展（`repeat_loop_no_progress`）：57
- 轨迹被标记为基础设施无效（`trajectory_marked_infrastructure_invalid`）：3

## 过程质量分数

- 候选利用（`candidate_utilization`）：0分 54，1分 132，2分 14
- 购买决策（`decision_quality`）：0分 194，1分 3，2分 3
- 证据核验（`evidence_verification`）：0分 86，1分 108，2分 6
- 搜索策略（`search_strategy`）：0分 0，1分 146，2分 54
- 终止效率（`termination_efficiency`）：0分 196，1分 2，2分 2

> 过程分数由通过 Gold Set 校准的 Judge 生成；有歧义的样本保留在复核队列。
