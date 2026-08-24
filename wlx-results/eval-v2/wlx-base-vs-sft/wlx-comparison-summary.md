# base → sft 自动汇总

> 中文名称用于阅读；括号内英文是程序字段名。

共同任务：200

## 购买成功转移

- 两者都失败（`both_failure`）：69
- 两者都成功（`both_success`）：3
- 只有候选模型成功（`candidate_only_success`）：109

## 格式转移

- 基线错、候选对（`fail->ok`）：4
- 两者都对（`ok->ok`）：196

## Rubric 需求状态变化

- 持续满足（`satisfied->satisfied`）：23
- 未知变为满足（`unknown->satisfied`）：1089
- 仍然未知（`unknown->unknown`）：270
- 未知变为违反（`unknown->violated`）：46

## 过程分数变化

- 候选利用（`candidate_utilization`）：改善 147，不变 43，退步 10，平均变化 0.835
- 购买决策（`decision_quality`）：改善 142，不变 58，退步 0，平均变化 1.250
- 证据核验（`evidence_verification`）：改善 137，不变 57，退步 6，平均变化 0.805
- 搜索策略（`search_strategy`）：改善 109，不变 90，退步 1，平均变化 0.540
- 终止效率（`termination_efficiency`）：改善 145，不变 55，退步 0，平均变化 1.340
