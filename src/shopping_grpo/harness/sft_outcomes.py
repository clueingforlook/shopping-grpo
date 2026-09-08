"""把 Core 轨迹翻译成 SFT 流水线能直接使用的结果标签。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from shopping_grpo.harness.contracts import Trajectory
from shopping_grpo.harness.sft_contracts import (
    AttemptDecision,
    OutcomeType,
    SftDisposition,
)


_INVALID_ERROR_CATEGORIES = {
    "policy",
    "environment",
    "protocol",
    "infrastructure",
    "release",
    "internal",
}

_REWARD_OUTCOMES = {
    item.value: item
    for item in (
        OutcomeType.GOLD_PURCHASE,
        OutcomeType.VALID_ALTERNATIVE_PURCHASE,
        OutcomeType.PARTIAL_ALTERNATIVE_PURCHASE,
        OutcomeType.WRONG_PURCHASE,
        OutcomeType.GRACEFUL_STOP,
        OutcomeType.EARLY_ABSTAIN,
        OutcomeType.REPEAT_LOOP,
        OutcomeType.MAX_STEPS,
        OutcomeType.REWARD_UNVERIFIABLE,
    )
}

_SFT_MECHANICAL_REJECTION_FIELDS = (
    ("blocked_tool_calls", "has_guard_rejection"),
    ("tool_call_truncations", "has_parallel_tool_truncation"),
    ("context_compactions", "context_was_compacted"),
)


def trajectory_mapping(trajectory: Trajectory | Mapping[str, Any]) -> dict[str, Any]:
    """把 Core 轨迹对象或普通字典统一变成一份可安全修改的字典副本。"""

    if isinstance(trajectory, Trajectory):
        return trajectory.to_dict()
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory 必须是 Trajectory 或字典")
    return deepcopy(dict(trajectory))


def classify_attempt(trajectory: Trajectory | Mapping[str, Any]) -> AttemptDecision:
    """判断一次尝试是否有效、任务是否成功，以及它能否进入主 SFT。"""

    row = trajectory_mapping(trajectory)
    invalid_reason = _technical_invalid_reason(row)
    if invalid_reason is not None:
        outcome = (
            OutcomeType.REWARD_UNVERIFIABLE
            if invalid_reason == "reward_unverifiable"
            else _technical_outcome(row)
        )
        return AttemptDecision(
            outcome_type=outcome,
            attempt_valid=False,
            task_success=None,
            sft_disposition=SftDisposition.RETRY,
            reasons=(invalid_reason,),
        )

    reward_type = _reward_type(row)
    if reward_type == OutcomeType.GOLD_PURCHASE.value:
        rejection_reasons = sft_mechanical_rejection_reasons(row)
        return AttemptDecision(
            outcome_type=OutcomeType.GOLD_PURCHASE,
            attempt_valid=True,
            task_success=True,
            sft_disposition=(
                SftDisposition.REJECTED
                if rejection_reasons
                else SftDisposition.ACCEPTED_GOLD
            ),
            reasons=rejection_reasons,
        )
    if reward_type == OutcomeType.VALID_ALTERNATIVE_PURCHASE.value:
        return AttemptDecision(
            outcome_type=OutcomeType.VALID_ALTERNATIVE_PURCHASE,
            attempt_valid=True,
            task_success=True,
            # 仓库规定训练成功必须是 gold；替代商品只留作审计和难度统计。
            sft_disposition=SftDisposition.ALTERNATIVE_AUDIT,
            reasons=("strict_sft_requires_gold_purchase",),
        )
    if reward_type in _REWARD_OUTCOMES:
        return AttemptDecision(
            outcome_type=_REWARD_OUTCOMES[reward_type],
            attempt_valid=True,
            task_success=False,
            sft_disposition=SftDisposition.REJECTED,
            reasons=(f"terminal_outcome:{reward_type}",),
        )

    status = str(row.get("status") or "")
    termination_reason = str(row.get("termination_reason") or "")
    if status == "max_steps" or termination_reason == "max_steps":
        outcome = OutcomeType.MAX_STEPS
    elif status in {"invalid_action", "invalid_action_limit", "parallel_tool_calls"}:
        outcome = OutcomeType.INVALID_ACTION
    elif status == "assistant_final":
        outcome = OutcomeType.NO_PURCHASE
    elif _error_category(row) == "model":
        outcome = OutcomeType.MODEL_ERROR
    else:
        # 没有技术错误、也没有 Reward 终局时，把它当作一次正常但未完成的尝试。
        outcome = OutcomeType.NO_PURCHASE
    return AttemptDecision(
        outcome_type=outcome,
        attempt_valid=True,
        task_success=False,
        sft_disposition=SftDisposition.REJECTED,
        reasons=(f"non_success_status:{status or 'unknown'}",),
    )


def is_strict_gold(decision: AttemptDecision) -> bool:
    """返回这次结果是否满足仓库规定的严格 Gold 训练标准。"""

    return (
        decision.attempt_valid
        and decision.outcome_type == OutcomeType.GOLD_PURCHASE
        and decision.sft_disposition == SftDisposition.ACCEPTED_GOLD
    )


def sft_mechanical_rejection_reasons(
    trajectory: Trajectory | Mapping[str, Any],
) -> tuple[str, ...]:
    """返回采样完成时已经能确定会被主 SFT 机械拒绝的原因。"""

    row = trajectory_mapping(trajectory)
    return tuple(
        reason
        for field, reason in _SFT_MECHANICAL_REJECTION_FIELDS
        if row.get(field)
    )


def _technical_invalid_reason(row: Mapping[str, Any]) -> str | None:
    """找出导致本次尝试不公平的技术原因；模型自己做错动作不算技术故障。"""

    if row.get("release_error"):
        return "environment_release_failed"
    if row.get("infrastructure_invalid") is True:
        if _reward_type(row) == OutcomeType.REWARD_UNVERIFIABLE.value:
            return "reward_unverifiable"
        return "trajectory_marked_infrastructure_invalid"
    category = _error_category(row)
    if category in _INVALID_ERROR_CATEGORIES:
        return f"error_category:{category}"
    reward_type = _reward_type(row)
    if reward_type == OutcomeType.REWARD_UNVERIFIABLE.value:
        return "reward_unverifiable"
    if row.get("done") is True and row.get("reward_valid") is not True:
        return "terminal_reward_not_valid"
    if str(row.get("status") or "") in {"error", "environment_release_failed"}:
        return "unclassified_runtime_error"
    return None


def _technical_outcome(row: Mapping[str, Any]) -> OutcomeType:
    """根据 Core 的错误类别区分协议错误和普通基础设施错误。"""

    category = _error_category(row)
    if category in {"protocol", "policy", "internal"}:
        return OutcomeType.PROTOCOL_ERROR
    return OutcomeType.INFRASTRUCTURE_ERROR


def _error_category(row: Mapping[str, Any]) -> str | None:
    """从轨迹错误对象中读取稳定的错误类别。"""

    error = row.get("error")
    if not isinstance(error, Mapping):
        return None
    category = error.get("category")
    return str(category) if category is not None else None


def _reward_type(row: Mapping[str, Any]) -> str | None:
    """从终局结果中读取 Reward-v3 的 outcome 名称。"""

    terminal = row.get("terminal_result")
    if not isinstance(terminal, Mapping):
        return None
    detail = terminal.get("reward_detail")
    if not isinstance(detail, Mapping):
        return None
    value = detail.get("reward_type")
    return str(value) if value is not None else None


__all__ = [
    "classify_attempt",
    "is_strict_gold",
    "sft_mechanical_rejection_reasons",
    "trajectory_mapping",
]
