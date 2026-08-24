"""WLX Eval 的轨迹完整性、评测资格与责任来源初判。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from wlx_harness_core.wlx_eval_contracts import (
    ELIGIBLE,
    MODEL_RESPONSIBILITY,
    NON_MODEL_INVALID,
    NON_MODEL_RESPONSIBILITY,
    REVIEW_REQUIRED,
    UNDETERMINED_RESPONSIBILITY,
    WLX_EVAL_ELIGIBILITY_VERSION,
)


def evaluate_eligibility(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """只判断轨迹能否公平评价；模型自己的失败仍保持 eligible。"""

    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    issues: list[dict[str, Any]] = []
    error = trajectory.get("error")
    error = error if isinstance(error, Mapping) else {}
    category = _optional_text(error.get("category"))
    error_type = _optional_text(error.get("error_type"))
    termination = _termination_reason(trajectory)

    if trajectory.get("release_error"):
        issues.append(
            _issue(
                "environment_release_failed",
                source="harness",
                detail=trajectory.get("release_error"),
            )
        )
        return _result(NON_MODEL_INVALID, NON_MODEL_RESPONSIBILITY, issues)

    if termination == "reward_unverifiable":
        issues.append(
            _issue(
                "environment_reward_unverifiable",
                source="environment",
            )
        )
        return _result(REVIEW_REQUIRED, NON_MODEL_RESPONSIBILITY, issues)

    # 上下文预算耗尽保留完整模型行为，是可评价的执行失败，
    # 而不是坏样本。
    if _is_context_budget_failure(category, error_type, termination):
        issues.append(
            _issue(
                "context_or_action_budget_exhausted",
                source="model_execution",
                detail={"category": category, "error_type": error_type},
            )
        )
        return _result(ELIGIBLE, MODEL_RESPONSIBILITY, issues)

    # 模型生成了不存在的工具、非法参数或主动停止，
    # 都属于模型能力问题。
    if category == "model" or str(trajectory.get("termination_category") or "") in {
        "invalid_action",
        "limit_reached",
        "model_stopped",
    }:
        if category == "model":
            issues.append(
                _issue(
                    "model_action_error",
                    source="model",
                    detail={"error_type": error_type},
                )
            )
        return _result(ELIGIBLE, MODEL_RESPONSIBILITY, issues)

    if trajectory.get("infrastructure_invalid") is True:
        issues.append(
            _issue(
                "trajectory_marked_infrastructure_invalid",
                source=category or "harness",
                detail={"error_type": error_type, "termination_reason": termination},
            )
        )
        return _result(NON_MODEL_INVALID, NON_MODEL_RESPONSIBILITY, issues)

    if category in {"environment", "infrastructure", "release", "internal"}:
        issues.append(
            _issue(
                f"runtime_error:{category}",
                source=category,
                detail={"error_type": error_type},
            )
        )
        return _result(NON_MODEL_INVALID, NON_MODEL_RESPONSIBILITY, issues)

    if category in {"protocol", "policy"}:
        issues.append(
            _issue(
                f"runtime_error:{category}",
                source="harness",
                detail={"error_type": error_type},
            )
        )
        return _result(NON_MODEL_INVALID, NON_MODEL_RESPONSIBILITY, issues)

    if str(trajectory.get("status") or "") in {
        "error",
        "environment_release_failed",
    }:
        issues.append(
            _issue(
                "unclassified_runtime_error",
                source="unknown",
                detail={"termination_reason": termination},
            )
        )
        return _result(NON_MODEL_INVALID, UNDETERMINED_RESPONSIBILITY, issues)

    return _result(ELIGIBLE, UNDETERMINED_RESPONSIBILITY, issues)


def apply_outcome_review(
    eligibility: Mapping[str, Any],
    *,
    review_reasons: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """将结果分歧提升到复核，但不覆盖更严重的非模型无效。"""

    result = deepcopy(dict(eligibility))
    reasons = [str(item) for item in review_reasons if str(item).strip()]
    if not reasons or result.get("status") == NON_MODEL_INVALID:
        return result
    result["status"] = REVIEW_REQUIRED
    if result.get("responsibility") == MODEL_RESPONSIBILITY:
        result["responsibility"] = UNDETERMINED_RESPONSIBILITY
    issues = list(result.get("integrity_issues") or [])
    existing = {item.get("code") for item in issues if isinstance(item, Mapping)}
    for reason in reasons:
        if reason not in existing:
            issues.append(_issue(reason, source="outcome_verification"))
    result["integrity_issues"] = issues
    return result


def _is_context_budget_failure(
    category: str | None,
    error_type: str | None,
    termination: str | None,
) -> bool:
    combined = " ".join(item or "" for item in (error_type, termination)).casefold()
    return (
        "contextbudget" in combined
        or "context_budget" in combined
        or "context_hard_limit" in combined
        or "max_assistant_turns" in combined
        or category == "policy" and "context" in combined
    )


def _termination_reason(trajectory: Mapping[str, Any]) -> str | None:
    terminal = trajectory.get("terminal_result")
    if isinstance(terminal, Mapping):
        detail = terminal.get("reward_detail")
        if isinstance(detail, Mapping) and detail.get("termination_reason"):
            return _optional_text(detail.get("termination_reason"))
        if terminal.get("termination_reason"):
            return _optional_text(terminal.get("termination_reason"))
    return _optional_text(trajectory.get("termination_reason"))


def _issue(code: str, *, source: str, detail: object = None) -> dict[str, Any]:
    return {
        "code": code,
        "source": source,
        "detail": deepcopy(detail),
    }


def _result(
    status: str,
    responsibility: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": WLX_EVAL_ELIGIBILITY_VERSION,
        "status": status,
        "responsibility": responsibility,
        "integrity_issues": deepcopy(issues),
    }


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = ["apply_outcome_review", "evaluate_eligibility"]
