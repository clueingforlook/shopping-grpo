"""WLX Eval v2 的单 Run 汇总、成对比较与 Markdown 报告。"""

from __future__ import annotations

from collections import Counter
import math
from statistics import mean, median
from typing import Any, Mapping, Sequence

from wlx_harness_core.wlx_eval_contracts import WLX_EVAL_REPORT_VERSION


BEHAVIOR_DISTRIBUTION_FIELDS = (
    "attempted_tool_calls",
    "executed_tool_calls",
    "guard_rejection_count",
    "search_count",
    "unique_search_query_count",
    "unique_product_open_count",
    "detail_view_count",
    "option_selection_count",
    "purchase_call_count",
    "exact_consecutive_repeat_count",
    "input_token_sum",
    "max_turn_input_tokens",
    "truncated_observation_count",
)


# JSON/JSONL 保留稳定的英文键；人读 Markdown 始终同时显示中文含义。
REPORT_LABELS = {
    "search_strategy": "搜索策略",
    "candidate_utilization": "候选利用",
    "evidence_verification": "证据核验",
    "decision_quality": "购买决策",
    "termination_efficiency": "终止效率",
    "gold_purchase": "目标商品购买成功",
    "valid_alternative_purchase": "有效替代商品",
    "partial_purchase": "部分符合",
    "partial_alternative_purchase": "部分符合或替代商品",
    "wrong_purchase": "错误购买",
    "no_purchase": "未购买",
    "unverifiable": "无法验证",
    "reward_unverifiable": "Reward 无法验证",
    "invalid_action_limit": "非法动作达到上限",
    "max_assistant_turns": "助手轮数达到上限",
    "repeat_loop": "重复循环",
    "internal:TeacherOutputTruncatedError": "教师输出截断",
    "model:ToolRegistryError": "工具注册错误",
    "policy:ContextBudgetError": "上下文预算错误",
    "context_or_action_budget_exhausted": "动作或上下文预算耗尽",
    "insufficient_verification": "核验不足",
    "invalid_action": "非法动作",
    "none": "无模型失败",
    "over_budget_purchase": "超预算购买",
    "premature_purchase": "过早购买",
    "repeat_loop_no_progress": "重复循环无进展",
    "trajectory_marked_infrastructure_invalid": "轨迹被标记为基础设施无效",
    "wrong_option_selection": "规格/选项错误",
    "satisfied": "满足",
    "violated": "违反",
    "unknown": "未知",
    "conflicting": "冲突",
    "unknown->satisfied": "未知变为满足",
    "unknown->violated": "未知变为违反",
    "unknown->unknown": "仍然未知",
    "satisfied->satisfied": "持续满足",
    "satisfied->violated": "满足变为违反",
    "satisfied->unknown": "满足变为未知",
    "violated->satisfied": "违反变为满足",
    "violated->violated": "持续违反",
    "violated->unknown": "违反变为未知",
    "both_failure": "两者都失败",
    "both_success": "两者都成功",
    "candidate_only_success": "只有候选模型成功",
    "baseline_only_success": "只有基线模型成功",
    "fail->ok": "基线错、候选对",
    "ok->ok": "两者都对",
    "ok->fail": "基线对、候选错",
    "fail->fail": "两者都错",
}


def summarize_evaluations(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    """汇总一个模型 Run；所有比例同时保留分子和分母。"""

    records = _unique_rows(rows)
    total = len(records)
    eligibility = Counter(
        _nested_text(row, "eligibility", "status") or "missing" for row in records
    )
    eligible = sum(
        _nested_text(row, "eligibility", "status") == "eligible" for row in records
    )
    eligible_records = [
        row
        for row in records
        if _nested_text(row, "eligibility", "status") == "eligible"
    ]
    responsibility = Counter(
        _nested_text(row, "eligibility", "responsibility") or "missing"
        for row in records
    )
    format_ok = sum(row.get("format_correct") is True for row in records)
    format_reasons = Counter()
    for row in records:
        error = row.get("first_format_error")
        if isinstance(error, Mapping) and error.get("reason"):
            format_reasons[str(error["reason"])] += 1

    normalized_outcomes = Counter(
        _nested_text(row, "outcome", "normalized_outcome") or "missing"
        for row in records
    )
    environment_outcomes = Counter(
        _nested_text(row, "outcome", "environment", "reward_type") or "missing"
        for row in records
    )
    purchase_correct = sum(row.get("purchase_correct") is True for row in records)
    eligible_purchase_correct = sum(
        row.get("purchase_correct") is True for row in eligible_records
    )
    outcome_agreement = Counter(
        _agreement_label(_nested(row, "outcome", "success_agreement"))
        for row in records
    )
    reviews = sum(_nested(row, "review", "required") is True for row in records)
    failures = Counter(
        _nested_text(row, "failure_attribution", "primary_failure") or "none"
        for row in records
    )
    failure_responsibility = Counter(
        _nested_text(row, "failure_attribution", "responsibility") or "missing"
        for row in records
    )

    tool_attempts: Counter[str] = Counter()
    requirement_checks: Counter[str] = Counter()
    requirement_panel_statuses: Counter[str] = Counter()
    process_panel_statuses: Counter[str] = Counter()
    process_scores: dict[str, Counter[int]] = {}
    for row in records:
        counts = _nested(row, "deterministic_behavior", "tool_counts_attempted")
        if isinstance(counts, Mapping):
            for name, value in counts.items():
                tool_attempts[str(name)] += _integer(value)
        items = _nested(row, "requirements", "items")
        requirement_panel_statuses[
            _nested_text(row, "requirements", "status") or "missing"
        ] += 1
        process_panel_statuses[
            _nested_text(row, "process_quality", "status") or "missing"
        ] += 1
        dimensions = _nested(row, "process_quality", "dimensions")
        if isinstance(dimensions, Mapping):
            for name, dimension in dimensions.items():
                if isinstance(dimension, Mapping) and dimension.get("score") in {0, 1, 2}:
                    process_scores.setdefault(str(name), Counter())[
                        int(dimension["score"])
                    ] += 1
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            for item in items:
                if isinstance(item, Mapping):
                    requirement_checks[str(item.get("status") or "missing")] += 1

    behavior_distributions = {
        field: _distribution(
            [
                _numeric(_nested(row, "deterministic_behavior", field))
                for row in records
            ]
        )
        for field in BEHAVIOR_DISTRIBUTION_FIELDS
    }
    summary = {
        "schema_version": WLX_EVAL_REPORT_VERSION,
        "run_name": run_name,
        "tasks": {
            "selected": total,
            "eligible": eligible,
            "review_required": eligibility["review_required"],
            "non_model_invalid": eligibility["non_model_invalid"],
        },
        "eligibility": {
            "counts": dict(sorted(eligibility.items())),
            "responsibility_counts": dict(sorted(responsibility.items())),
        },
        "format": {
            "correct": _ratio(format_ok, total),
            "first_error_counts": dict(sorted(format_reasons.items())),
        },
        "outcome": {
            "purchase_correct": _ratio(purchase_correct, total),
            "purchase_correct_eligible": _ratio(
                eligible_purchase_correct, len(eligible_records)
            ),
            "normalized_counts": dict(sorted(normalized_outcomes.items())),
            "environment_counts": dict(sorted(environment_outcomes.items())),
            "success_agreement_counts": dict(sorted(outcome_agreement.items())),
        },
        "requirements": {
            "panel_status_counts": dict(sorted(requirement_panel_statuses.items())),
            "check_counts": dict(sorted(requirement_checks.items())),
        },
        "deterministic_behavior": {
            "tool_attempt_counts": dict(sorted(tool_attempts.items())),
            "distributions": behavior_distributions,
        },
        "process_quality": {
            "panel_status_counts": dict(sorted(process_panel_statuses.items())),
            "dimension_score_counts": {
                name: {str(score): count for score, count in sorted(counts.items())}
                for name, counts in sorted(process_scores.items())
            },
        },
        "failure_attribution": {
            "primary_failure_counts": dict(sorted(failures.items())),
            "responsibility_counts": dict(sorted(failure_responsibility.items())),
        },
        "review": {"required": _ratio(reviews, total)},
    }
    return summary


def compare_evaluation_runs(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """按 task_id 对齐两个 Run，返回汇总与每题转移记录。"""

    baseline = {int(row["task_id"]): row for row in _unique_rows(baseline_rows)}
    candidate = {int(row["task_id"]): row for row in _unique_rows(candidate_rows)}
    common = sorted(set(baseline).intersection(candidate))
    only_baseline = sorted(set(baseline).difference(candidate))
    only_candidate = sorted(set(candidate).difference(baseline))
    paired: list[dict[str, Any]] = []
    transition_counts: Counter[str] = Counter()
    all_transition_counts: Counter[str] = Counter()
    format_transitions: Counter[str] = Counter()
    outcome_transitions: Counter[str] = Counter()
    failure_transitions: Counter[str] = Counter()
    requirement_transitions: Counter[str] = Counter()
    process_score_deltas: dict[str, list[int]] = {}

    for task_id in common:
        before = baseline[task_id]
        after = candidate[task_id]
        before_success = before.get("purchase_correct") is True
        after_success = after.get("purchase_correct") is True
        transition = _success_transition(before_success, after_success)
        comparable = (
            _nested_text(before, "eligibility", "status") == "eligible"
            and _nested_text(after, "eligibility", "status") == "eligible"
        )
        before_format = before.get("format_correct") is True
        after_format = after.get("format_correct") is True
        format_transition = f"{_bool_label(before_format)}->{_bool_label(after_format)}"
        before_outcome = _nested_text(before, "outcome", "normalized_outcome") or "missing"
        after_outcome = _nested_text(after, "outcome", "normalized_outcome") or "missing"
        outcome_transition = f"{before_outcome}->{after_outcome}"
        before_failure = (
            _nested_text(before, "failure_attribution", "primary_failure") or "none"
        )
        after_failure = (
            _nested_text(after, "failure_attribution", "primary_failure") or "none"
        )
        failure_transition = f"{before_failure}->{after_failure}"
        all_transition_counts[transition] += 1
        if comparable:
            transition_counts[transition] += 1
        format_transitions[format_transition] += 1
        outcome_transitions[outcome_transition] += 1
        failure_transitions[failure_transition] += 1
        before_requirements = {
            str(item.get("requirement_id")): str(item.get("status") or "missing")
            for item in (_nested(before, "requirements", "items") or [])
            if isinstance(item, Mapping)
        }
        after_requirements = {
            str(item.get("requirement_id")): str(item.get("status") or "missing")
            for item in (_nested(after, "requirements", "items") or [])
            if isinstance(item, Mapping)
        }
        for requirement_id in set(before_requirements).intersection(after_requirements):
            requirement_transitions[
                f"{before_requirements[requirement_id]}->{after_requirements[requirement_id]}"
            ] += 1
        before_dimensions = _nested(before, "process_quality", "dimensions")
        after_dimensions = _nested(after, "process_quality", "dimensions")
        if isinstance(before_dimensions, Mapping) and isinstance(after_dimensions, Mapping):
            for name in set(before_dimensions).intersection(after_dimensions):
                left = before_dimensions[name]
                right = after_dimensions[name]
                if (
                    isinstance(left, Mapping)
                    and isinstance(right, Mapping)
                    and left.get("score") in {0, 1, 2}
                    and right.get("score") in {0, 1, 2}
                ):
                    process_score_deltas.setdefault(str(name), []).append(
                        int(right["score"]) - int(left["score"])
                    )
        paired.append(
            {
                "task_id": task_id,
                "success_transition": transition,
                "format_transition": format_transition,
                "outcome_transition": outcome_transition,
                "failure_transition": failure_transition,
                "comparable": comparable,
                "baseline_review_required": _nested(before, "review", "required") is True,
                "candidate_review_required": _nested(after, "review", "required") is True,
            }
        )

    summary = {
        "schema_version": WLX_EVAL_REPORT_VERSION,
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "paired_tasks": len(common),
        "comparable_paired_tasks": sum(row["comparable"] for row in paired),
        "only_in_baseline": only_baseline,
        "only_in_candidate": only_candidate,
        "success_transitions": dict(sorted(transition_counts.items())),
        "success_transitions_all_pairs": dict(sorted(all_transition_counts.items())),
        "format_transitions": dict(sorted(format_transitions.items())),
        "outcome_transitions": dict(sorted(outcome_transitions.items())),
        "failure_transitions": dict(sorted(failure_transitions.items())),
        "requirement_status_transitions": dict(sorted(requirement_transitions.items())),
        "process_score_deltas": {
            name: {
                "tasks": len(values),
                "improved": sum(value > 0 for value in values),
                "unchanged": sum(value == 0 for value in values),
                "regressed": sum(value < 0 for value in values),
                "mean_delta": mean(values) if values else None,
            }
            for name, values in sorted(process_score_deltas.items())
        },
        "baseline_summary": summarize_evaluations(
            list(baseline.values()), run_name=baseline_name
        ),
        "candidate_summary": summarize_evaluations(
            list(candidate.values()), run_name=candidate_name
        ),
    }
    return summary, paired


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """将单 Run JSON 汇总渲染为便于阅读的短报告。"""

    tasks = summary.get("tasks") or {}
    fmt = summary.get("format") or {}
    outcome = summary.get("outcome") or {}
    review = summary.get("review") or {}
    lines = [
        f"# WLX Eval Run：{summary.get('run_name') or 'unnamed'}",
        "",
        "> 中文名称用于阅读；括号内英文是程序字段名。",
        "",
        "## 核心计数",
        "",
        f"- 任务：{tasks.get('selected', 0)}",
        f"- 可直接评价：{tasks.get('eligible', 0)}",
        f"- 需要复核：{tasks.get('review_required', 0)}",
        f"- 非模型无效：{tasks.get('non_model_invalid', 0)}",
        f"- 格式正确：{_ratio_text(fmt.get('correct'))}",
        f"- WLX 购买正确：{_ratio_text(outcome.get('purchase_correct'))}",
        (
            "- 可直接评价任务中的 WLX 购买正确："
            f"{_ratio_text(outcome.get('purchase_correct_eligible'))}"
        ),
        f"- 复核队列：{_ratio_text(review.get('required'))}",
        "",
        "## WLX 结果分布",
        "",
    ]
    for name, count in sorted((outcome.get("normalized_counts") or {}).items()):
        lines.append(f"- {_display_label(name)}：{count}")
    lines.extend(["", "## 环境结果分布", ""])
    for name, count in sorted((outcome.get("environment_counts") or {}).items()):
        lines.append(f"- {_display_label(name)}：{count}")
    lines.extend(["", "## Rubric 需求状态", ""])
    requirements = summary.get("requirements") or {}
    requirement_counts = requirements.get("check_counts") or {}
    if requirement_counts:
        total_requirements = sum(_integer(value) for value in requirement_counts.values())
        lines.append(f"共检查 {total_requirements} 条用户要求：")
        lines.append("")
        for name, count in sorted(requirement_counts.items()):
            lines.append(f"- {_display_label(name)}：{count}")
    else:
        lines.append("- 没有可用的冻结 Rubric 结果")
    lines.extend(["", "## 第一失败分布", ""])
    attribution = summary.get("failure_attribution") or {}
    for name, count in sorted(
        (attribution.get("primary_failure_counts") or {}).items()
    ):
        lines.append(f"- {_display_label(name)}：{count}")
    lines.extend(["", "## 过程质量分数", ""])
    process = summary.get("process_quality") or {}
    scores = process.get("dimension_score_counts") or {}
    if scores:
        for name, counts in sorted(scores.items()):
            lines.append(
                f"- {_display_label(name)}：0分 {counts.get('0', 0)}，"
                f"1分 {counts.get('1', 0)}，2分 {counts.get('2', 0)}"
            )
    else:
        lines.append("- 尚未运行经校准的 Judge")
    lines.extend(
        [
            "",
            "> 过程分数由通过 Gold Set 校准的 Judge 生成；有歧义的样本保留在复核队列。",
            "",
        ]
    )
    return "\n".join(lines)


def render_comparison_markdown(summary: Mapping[str, Any]) -> str:
    """将成对比较汇总渲染成 Markdown。"""

    lines = [
        f"# {summary.get('baseline_name')} → {summary.get('candidate_name')} 自动汇总",
        "",
        "> 中文名称用于阅读；括号内英文是程序字段名。",
        "",
        f"共同任务：{summary.get('paired_tasks', 0)}",
        "",
        "## 购买成功转移",
        "",
    ]
    for name, count in sorted((summary.get("success_transitions") or {}).items()):
        lines.append(f"- {_display_label(name)}：{count}")
    lines.extend(["", "## 格式转移", ""])
    for name, count in sorted((summary.get("format_transitions") or {}).items()):
        lines.append(f"- {_display_label(name)}：{count}")
    lines.extend(["", "## Rubric 需求状态变化", ""])
    requirement_transitions = summary.get("requirement_status_transitions") or {}
    if requirement_transitions:
        for name, count in sorted(requirement_transitions.items()):
            lines.append(f"- {_display_label(name)}：{count}")
    else:
        lines.append("- 没有可比较的 Rubric 要求")
    lines.extend(["", "## 过程分数变化", ""])
    for name, value in sorted((summary.get("process_score_deltas") or {}).items()):
        lines.append(
            f"- {_display_label(name)}：改善 {value['improved']}，不变 {value['unchanged']}，"
            f"退步 {value['regressed']}，平均变化 {value['mean_delta']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def _display_label(name: object) -> str:
    """为 Markdown 报告返回“中文（英文键）”形式的稳定标签。"""

    key = str(name)
    chinese = REPORT_LABELS.get(key)
    return f"{chinese}（`{key}`）" if chinese else f"`{key}`"


def _unique_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("evaluation rows must be objects")
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError("evaluation row has invalid task_id")
        if task_id in seen:
            raise ValueError(f"duplicate task_id in evaluation rows: {task_id}")
        seen.add(task_id)
        result.append(row)
    return result


def _nested(value: Mapping[str, Any], *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_text(value: Mapping[str, Any], *keys: str) -> str | None:
    result = _nested(value, *keys)
    text = str(result).strip() if result is not None else ""
    return text or None


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": numerator / denominator if denominator else None,
    }


def _ratio_text(value: object) -> str:
    if not isinstance(value, Mapping):
        return "0/0"
    numerator = _integer(value.get("numerator"))
    denominator = _integer(value.get("denominator"))
    rate = value.get("rate")
    suffix = f"（{float(rate):.1%}）" if isinstance(rate, (int, float)) else ""
    return f"{numerator}/{denominator}{suffix}"


def _distribution(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {"count": 0, "min": None, "median": None, "mean": None, "p90": None, "max": None}
    ordered = sorted(clean)
    p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _agreement_label(value: object) -> str:
    if value is True:
        return "agree"
    if value is False:
        return "disagree"
    return "not_comparable"


def _success_transition(before: bool, after: bool) -> str:
    if before and after:
        return "both_success"
    if before and not after:
        return "baseline_only_success"
    if not before and after:
        return "candidate_only_success"
    return "both_failure"


def _bool_label(value: bool) -> str:
    return "ok" if value else "fail"


__all__ = [
    "BEHAVIOR_DISTRIBUTION_FIELDS",
    "compare_evaluation_runs",
    "render_comparison_markdown",
    "render_summary_markdown",
    "summarize_evaluations",
]
