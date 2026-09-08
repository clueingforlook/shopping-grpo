"""基于确定性证据给 轨迹生成第一版失败归因。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from shopping_grpo.harness.eval_contracts import (
    ELIGIBLE,
    MODEL_RESPONSIBILITY,
    NON_MODEL_RESPONSIBILITY,
    UNDETERMINED_RESPONSIBILITY,
    EVAL_ATTRIBUTION_VERSION,
)


def evaluate_deterministic_attribution(
    *,
    eligibility: Mapping[str, Any],
    format_panel: Mapping[str, Any],
    outcome: Mapping[str, Any],
    behavior: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """确定性规则只给有直接证据的归因，其余明确等待 Judge/人工。"""

    terminal_event = _event_ref(behavior, "termination")
    purchase_event = _event_ref(behavior, "first_purchase")
    eligibility_status = eligibility.get("status")
    responsibility = str(
        eligibility.get("responsibility") or UNDETERMINED_RESPONSIBILITY
    )

    if eligibility_status != ELIGIBLE:
        issues = [
            str(item.get("code"))
            for item in eligibility.get("integrity_issues") or []
            if isinstance(item, Mapping) and item.get("code")
        ]
        return _result(
            status="review_required" if eligibility_status == "review_required" else "complete",
            responsibility=responsibility,
            primary=(issues[0] if issues else "insufficient_evidence"),
            evidence=[terminal_event],
            explanation=(
                "轨迹需要先处理非模型完整性或结果分歧，"
                "暂不归因到模型策略。"
            ),
            confidence=1.0 if eligibility_status == "non_model_invalid" else 0.7,
            review_required=eligibility_status == "review_required",
        )

    normalized = outcome.get("normalized_outcome")
    if normalized in {"gold_purchase", "valid_alternative_purchase"}:
        return _result(
            status="not_applicable",
            responsibility=UNDETERMINED_RESPONSIBILITY,
            primary=None,
            evidence=[purchase_event, terminal_event],
            explanation="结果验证通过，没有失败需要归因。",
            confidence=1.0,
            review_required=False,
        )

    if format_panel.get("format_correct") is False:
        error = format_panel.get("first_format_error")
        turn_index = error.get("turn_index") if isinstance(error, Mapping) else None
        event_id = _assistant_event_id(events, turn_index)
        reason = error.get("reason") if isinstance(error, Mapping) else "format_error"
        return _result(
            status="complete",
            responsibility=MODEL_RESPONSIBILITY,
            primary="invalid_action",
            evidence=[event_id or terminal_event],
            explanation=f"首个工具调用格式错误：{reason}。",
            confidence=1.0,
            review_required=False,
        )

    termination = str(behavior.get("termination_reason") or "")
    if behavior.get("environment_repeat_loop") is True or termination == "repeat_loop":
        return _result(
            status="complete",
            responsibility=MODEL_RESPONSIBILITY,
            primary="repeat_loop_no_progress",
            evidence=[terminal_event],
            explanation="环境确认轨迹因无进展重复循环结束。",
            confidence=1.0,
            review_required=False,
        )
    if termination in {"invalid_action_limit", "parallel_tool_calls"} or str(
        behavior.get("termination_category") or ""
    ) == "invalid_action":
        return _result(
            status="complete",
            responsibility=MODEL_RESPONSIBILITY,
            primary="invalid_action",
            evidence=[terminal_event],
            explanation="模型非法动作或 Guard 拒绝达到终止条件。",
            confidence=1.0,
            review_required=False,
        )
    if _budget_exhausted(termination):
        return _result(
            status="complete",
            responsibility=MODEL_RESPONSIBILITY,
            primary="context_or_action_budget_exhausted",
            evidence=[terminal_event],
            explanation="模型未在动作或上下文预算内完成任务。",
            confidence=0.95,
            review_required=False,
        )

    failed_checks = _failed_checks(outcome)
    failed_names = [str(item.get("name") or "") for item in failed_checks]
    if "price_upper" in failed_names:
        return _result(
            status="complete",
            responsibility=MODEL_RESPONSIBILITY,
            primary="over_budget_purchase",
            evidence=[purchase_event, terminal_event],
            explanation="实际购买价格超过明确上限。",
            confidence=1.0,
            review_required=False,
        )
    if "category" in failed_names:
        return _result(
            status="complete",
            responsibility=MODEL_RESPONSIBILITY,
            primary="wrong_product_selection",
            evidence=[purchase_event, terminal_event],
            explanation="实际购买商品类别不满足任务要求。",
            confidence=0.95,
            review_required=False,
        )
    if any(name.startswith("option:") for name in failed_names):
        return _result(
            status="complete",
            responsibility=MODEL_RESPONSIBILITY,
            primary="wrong_option_selection",
            evidence=[purchase_event, terminal_event],
            explanation="实际购买商品的至少一个必需选项不正确。",
            confidence=0.9,
            review_required=False,
        )
    if normalized in {"partial_purchase", "wrong_purchase"}:
        return _result(
            status="partial",
            responsibility=MODEL_RESPONSIBILITY,
            primary="wrong_product_selection",
            evidence=[purchase_event, terminal_event],
            explanation=(
                "购买结果违反需求，但仅凭确定性证据不能确认"
                "更早的策略原因。"
            ),
            confidence=0.65,
            review_required=True,
        )
    if normalized == "unverifiable":
        return _result(
            status="review_required",
            responsibility=NON_MODEL_RESPONSIBILITY,
            primary="insufficient_evidence",
            evidence=[terminal_event],
            explanation="购买证据不足，不能可靠评价模型结果。",
            confidence=1.0,
            review_required=True,
        )

    return _result(
        status="partial",
        responsibility=MODEL_RESPONSIBILITY,
        primary="undetermined_model_failure",
        evidence=[terminal_event],
        explanation=(
            "模型没有完成购买；需要过程 Judge 或人工定位第一处策略错误。"
        ),
        confidence=0.4,
        review_required=True,
    )


def _result(
    *,
    status: str,
    responsibility: str,
    primary: str | None,
    evidence: Sequence[str | None],
    explanation: str,
    confidence: float,
    review_required: bool,
) -> dict[str, Any]:
    evidence_ids = [item for item in evidence if item]
    return {
        "version": EVAL_ATTRIBUTION_VERSION,
        "status": status,
        "responsibility": responsibility,
        "primary_failure": primary,
        "secondary_failures": [],
        "first_error_event_id": evidence_ids[0] if evidence_ids else None,
        "evidence_event_ids": evidence_ids,
        "explanation": explanation,
        "recoverable_at_event": None if primary is None else True,
        "confidence": float(confidence),
        "review_required": bool(review_required),
    }


def _failed_checks(outcome: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    verifier = outcome.get("wlx_verifier")
    if not isinstance(verifier, Mapping):
        return []
    return [
        item
        for item in verifier.get("checks") or []
        if isinstance(item, Mapping) and item.get("passed") is not True
    ]


def _assistant_event_id(
    events: Sequence[Mapping[str, Any]], turn_index: object
) -> str | None:
    assistant_messages: list[int] = []
    for event in events:
        if event.get("role") != "assistant":
            continue
        message_index = event.get("message_index")
        if isinstance(message_index, int) and message_index not in assistant_messages:
            assistant_messages.append(message_index)
    try:
        selected_message = assistant_messages[int(turn_index)]
    except (IndexError, TypeError, ValueError):
        return None
    return next(
        (
            str(event.get("event_id"))
            for event in events
            if event.get("message_index") == selected_message
            and event.get("role") == "assistant"
        ),
        None,
    )


def _event_ref(behavior: Mapping[str, Any], key: str) -> str | None:
    refs = behavior.get("event_refs")
    if not isinstance(refs, Mapping):
        return None
    value = refs.get(key)
    return str(value) if value else None


def _budget_exhausted(termination: str) -> bool:
    value = termination.casefold()
    return any(
        marker in value
        for marker in (
            "contextbudget",
            "context_budget",
            "context_hard_limit",
            "max_steps",
            "max_assistant_turns",
        )
    )


__all__ = ["evaluate_deterministic_attribution"]
