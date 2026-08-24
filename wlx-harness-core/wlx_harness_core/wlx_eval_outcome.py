"""WLX Eval 最终结果、需求检查和环境/验证器分歧处理。"""

from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from typing import Any, Mapping, Sequence

from wlx_harness_core.wlx_eval_contracts import WLX_EVAL_OUTCOME_VERSION
from wlx_harness_core.wlx_purchase_verifier import (
    GOLD_PURCHASE,
    VALID_ALTERNATIVE_PURCHASE,
    verify_environment_terminal,
)


ENVIRONMENT_SUCCESS_OUTCOMES = {GOLD_PURCHASE, VALID_ALTERNATIVE_PURCHASE}


def evaluate_outcome(
    trajectory: Mapping[str, Any],
    *,
    terminal_event_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回最终结果面板和临时需求检查面板。"""

    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    terminal = trajectory.get("terminal_result")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    purchase = terminal.get("purchase")
    purchase = purchase if isinstance(purchase, Mapping) and purchase else None
    verification = verify_environment_terminal(terminal)
    environment = _environment_outcome(trajectory, terminal, purchase)
    normalized = _normalized_wlx_outcome(verification.to_dict(), purchase)
    wlx_success = normalized in ENVIRONMENT_SUCCESS_OUTCOMES
    environment_success = environment["success"]

    review_reasons: list[str] = []
    if normalized == "unverifiable":
        review_reasons.append("verifier_unverifiable")
    if environment["reward_type"] == "reward_unverifiable":
        review_reasons.append("environment_reward_unverifiable")
    if environment["reward_valid"] is False:
        review_reasons.append("environment_reward_invalid")
    if environment_success is not None and environment_success != wlx_success:
        review_reasons.append("outcome_success_disagreement")
    type_agreement = _type_agreement(environment["reward_type"], normalized)
    if type_agreement is False and "outcome_success_disagreement" not in review_reasons:
        review_reasons.append("outcome_type_disagreement")

    outcome = {
        "version": WLX_EVAL_OUTCOME_VERSION,
        "normalized_outcome": normalized,
        "purchase_correct": (
            None if normalized == "unverifiable" else bool(wlx_success)
        ),
        "purchase_type": (
            normalized if normalized in ENVIRONMENT_SUCCESS_OUTCOMES else None
        ),
        "environment": environment,
        "wlx_verifier": verification.to_dict(),
        "success_agreement": (
            None
            if environment_success is None or normalized == "unverifiable"
            else environment_success == wlx_success
        ),
        "type_agreement": type_agreement,
        "review_required": bool(review_reasons),
        "review_reasons": review_reasons,
        "evidence_event_ids": [terminal_event_id] if terminal_event_id else [],
    }
    requirements = _provisional_requirements(
        terminal,
        verification.to_dict().get("checks") or [],
        terminal_event_id=terminal_event_id,
    )
    return outcome, requirements


def _environment_outcome(
    trajectory: Mapping[str, Any],
    terminal: Mapping[str, Any],
    purchase: Mapping[str, Any] | None,
) -> dict[str, Any]:
    detail = terminal.get("reward_detail")
    detail = detail if isinstance(detail, Mapping) else {}
    reward_type = _optional_text(
        detail.get("reward_type")
        or detail.get("termination_reason")
        or terminal.get("termination_reason")
        or trajectory.get("termination_reason")
    )
    if reward_type in ENVIRONMENT_SUCCESS_OUTCOMES:
        success: bool | None = True
    elif reward_type == "reward_unverifiable" or reward_type is None:
        success = None
    else:
        success = False
    reward_valid = terminal.get("reward_valid")
    if not isinstance(reward_valid, bool):
        reward_valid = trajectory.get("reward_valid")
    if not isinstance(reward_valid, bool):
        reward_valid = None
    return {
        "reward_type": reward_type,
        "reward_valid": reward_valid,
        "sampling_invalid": _optional_bool(
            detail.get("sampling_invalid"), trajectory.get("sampling_invalid")
        ),
        "success": success,
        "purchase_present": purchase is not None,
        "purchase": _purchase_summary(purchase),
        "reward": _first_not_none(terminal.get("reward"), trajectory.get("final_reward")),
    }


def _normalized_wlx_outcome(
    verification: Mapping[str, Any],
    purchase: Mapping[str, Any] | None,
) -> str:
    if purchase is None:
        return "no_purchase"
    if verification.get("verifier_valid") is not True:
        return "unverifiable"
    if verification.get("purchase_correct") is True:
        if verification.get("purchase_type") == GOLD_PURCHASE:
            return GOLD_PURCHASE
        return VALID_ALTERNATIVE_PURCHASE

    checks = [
        item
        for item in verification.get("checks") or []
        if isinstance(item, Mapping)
    ]
    failed = [item for item in checks if item.get("passed") is not True]
    if any(str(item.get("name") or "") in {"category", "price_upper"} for item in failed):
        return "wrong_purchase"
    if checks and any(item.get("passed") is True for item in checks):
        return "partial_purchase"
    return "wrong_purchase"


def _type_agreement(environment_type: str | None, normalized: str) -> bool | None:
    if environment_type is None or environment_type == "reward_unverifiable":
        return None
    mapped = {
        "gold_purchase": "gold_purchase",
        "valid_alternative_purchase": "valid_alternative_purchase",
        "partial_alternative_purchase": "partial_purchase",
        "wrong_purchase": "wrong_purchase",
        "graceful_stop": "no_purchase",
        "early_abstain": "no_purchase",
        "repeat_loop": "no_purchase",
        "max_steps": "no_purchase",
    }.get(environment_type)
    return None if mapped is None else mapped == normalized


def _provisional_requirements(
    terminal: Mapping[str, Any],
    checks: Sequence[object],
    *,
    terminal_event_id: str | None,
) -> dict[str, Any]:
    goal = terminal.get("goal")
    goal = goal if isinstance(goal, Mapping) else {}
    instruction = _optional_text(
        goal.get("instruction_text") or terminal.get("instruction")
    )
    items: list[dict[str, Any]] = []
    for index, check in enumerate(checks, start=1):
        if not isinstance(check, Mapping):
            continue
        required = deepcopy(check.get("required"))
        item_type = _requirement_type(str(check.get("name") or ""))
        if check.get("verifiable") is not True:
            status = "unknown"
        elif check.get("passed") is True:
            status = "satisfied"
        else:
            status = "violated"
        query_evidence = _literal_query_evidence(instruction, required)
        items.append(
            {
                "requirement_id": f"R{index:03d}",
                "name": str(check.get("name") or f"requirement:{index}"),
                "type": item_type,
                "priority": "hard",
                "status": status,
                "required": required,
                "actual": deepcopy(check.get("actual")),
                "query_evidence": query_evidence,
                "source": "environment_goal_allowlist",
                "eligible_for_frozen_rubric": query_evidence is not None,
                "evidence_event_ids": (
                    [terminal_event_id] if terminal_event_id else []
                ),
            }
        )
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("satisfied", "violated", "unknown", "not_applicable", "conflicting")
    }
    return {
        "status": "provisional",
        "included_in_final_metrics": False,
        "reason": "user-query rubric has not been frozen and human-audited",
        "instruction": instruction,
        "items": items,
        "counts": counts,
    }


def _requirement_type(name: str) -> str:
    if name == "category":
        return "category"
    if name == "price_upper":
        return "budget"
    for prefix in ("brand", "model", "attribute", "option"):
        if name.startswith(prefix + ":"):
            return prefix
    return "other"


def _literal_query_evidence(instruction: str | None, required: object) -> str | None:
    if not instruction:
        return None
    values = (
        required
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes))
        else [required]
    )
    normalized_instruction = _normalize(instruction)
    for value in values:
        text = _optional_text(value)
        if text and _normalize(text) in normalized_instruction:
            return text
    return None


def _purchase_summary(purchase: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if purchase is None:
        return None
    return {
        key: deepcopy(purchase.get(key))
        for key in ("asin", "name", "category", "attributes", "options", "price")
        if key in purchase
    }


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s/|,，。:：;；、\-]+", "", text)


def _optional_bool(*values: object) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _first_not_none(*values: object) -> object:
    return next((deepcopy(value) for value in values if value is not None), None)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = ["ENVIRONMENT_SUCCESS_OUTCOMES", "evaluate_outcome"]
