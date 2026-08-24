"""WLX Reward v4: deterministic outcome and process signals for ShopSimulator."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


WLX_REWARD_VERSION = "wlx-reward-v4"
WLX_PRM_SELECTION_MARKER = 1.0e-4
_PURCHASE_REWARD_TYPES = {
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
    "reward_unverifiable",
}


def _normal_terminal(state: Mapping[str, Any]) -> bool:
    terminal = state.get("terminal_result") or {}
    return (
        state.get("done") is True
        and isinstance(terminal, Mapping)
        and terminal.get("done") is True
        and terminal.get("over") is True
    )


def _status(detail: Mapping[str, Any], gate: str) -> str:
    gates = detail.get("hard_gates") or {}
    value = gates.get(gate) if isinstance(gates, Mapping) else None
    return str(value.get("status") or "") if isinstance(value, Mapping) else ""


def _ratio(detail: Mapping[str, Any], names: Sequence[str]) -> float:
    dimensions = detail.get("dimension_details") or {}
    passed = 0
    required = 0
    for name in names:
        value = dimensions.get(name) if isinstance(dimensions, Mapping) else None
        if not isinstance(value, Mapping):
            continue
        required_count = int(value.get("required_count", 0))
        passed_count = int(value.get("passed_count", 0))
        if required_count < 0 or not 0 <= passed_count <= required_count:
            raise ValueError(f"invalid {name} pass counts")
        required += required_count
        passed += passed_count
    return passed / required if required else 1.0


def compute_wlx_orm(state: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the v4 terminal outcome from public structured reward evidence."""
    infrastructure_invalid = bool(state.get("infrastructure_invalid"))
    detail = state.get("reward_detail") or {}
    if not isinstance(detail, Mapping):
        detail = {}
    reward_type = str(state.get("reward_type") or "")
    purchased = reward_type in _PURCHASE_REWARD_TYPES
    normal_terminal = _normal_terminal(state)

    if infrastructure_invalid:
        return _result(
            score=0.0,
            outcome="infrastructure_invalid",
            sampling_invalid=True,
            purchased=purchased,
        )

    # Model-caused early termination and unfinished trajectories are valid no-purchase outcomes.
    if not purchased:
        return _result(
            score=-0.25,
            outcome="no_purchase",
            sampling_invalid=False,
            purchased=False,
        )

    if not normal_terminal:
        return _result(
            score=0.0,
            outcome="invalid_terminal_purchase",
            sampling_invalid=True,
            purchased=True,
        )

    category_status = _status(detail, "category")
    price_status = _status(detail, "budget")
    # Gold ASIN + gold options is already an exact environment-side match.
    if reward_type == "gold_purchase":
        return _result(
            score=1.0,
            outcome="gold",
            sampling_invalid=False,
            purchased=True,
            category_status=category_status or "pass",
            attribute_ratio=1.0,
            option_ratio=1.0,
            price_status=price_status or "pass",
        )
    if category_status == "unverifiable" or not category_status:
        return _result(
            score=0.0,
            outcome="unverifiable_category",
            sampling_invalid=True,
            purchased=True,
            category_status=category_status,
            price_status=price_status,
        )
    if category_status == "fail":
        return _result(
            score=-0.50,
            outcome="wrong_category",
            sampling_invalid=False,
            purchased=True,
            category_status=category_status,
            price_status=price_status,
        )
    if category_status != "pass":
        return _result(
            score=0.0,
            outcome="invalid_category_status",
            sampling_invalid=True,
            purchased=True,
            category_status=category_status,
            price_status=price_status,
        )

    try:
        attribute_ratio = _ratio(detail, ("brand", "model", "core_functions"))
        option_ratio = _ratio(detail, ("key_options",))
    except (TypeError, ValueError):
        return _result(
            score=0.0,
            outcome="invalid_dimension_counts",
            sampling_invalid=True,
            purchased=True,
            category_status=category_status,
            price_status=price_status,
        )
    price_pass = float(price_status == "pass")
    score = 0.50 * attribute_ratio * option_ratio * price_pass
    return _result(
        score=score,
        outcome="partial_purchase",
        sampling_invalid=False,
        purchased=True,
        category_status=category_status,
        attribute_ratio=attribute_ratio,
        option_ratio=option_ratio,
        price_status=price_status,
    )


def _result(
    *,
    score: float,
    outcome: str,
    sampling_invalid: bool,
    purchased: bool,
    category_status: str = "",
    attribute_ratio: float = 0.0,
    option_ratio: float = 0.0,
    price_status: str = "",
) -> dict[str, Any]:
    if not math.isfinite(score):
        raise ValueError("WLX ORM score must be finite")
    return {
        "version": WLX_REWARD_VERSION,
        "terminal_utility": float(score),
        "total": float(score),
        "outcome": outcome,
        "gold": float(outcome == "gold"),
        "partial_purchase": float(outcome == "partial_purchase"),
        "no_purchase": float(outcome == "no_purchase"),
        "wrong_category": float(outcome == "wrong_category"),
        "purchased": bool(purchased),
        "category_status": category_status,
        "attribute_ratio": float(attribute_ratio),
        "option_ratio": float(option_ratio),
        "price_status": price_status,
        "price_pass": float(price_status == "pass"),
        "sampling_invalid": bool(sampling_invalid),
        "infrastructure_invalid": bool(outcome == "infrastructure_invalid"),
    }


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", text)


def _last_tool_index(model_steps: Sequence[Mapping[str, Any]], tool: str) -> int | None:
    for index in range(len(model_steps) - 1, -1, -1):
        if str(model_steps[index].get("tool") or "") == tool:
            return index
    return None


def _wrong_option_values(detail: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    failures = detail.get("option_failures") or {}
    wrong_values = failures.get("wrong_values") if isinstance(failures, Mapping) else ()
    for item in wrong_values or ():
        if isinstance(item, Mapping) and _normalize(item.get("selected")):
            values.add(_normalize(item["selected"]))
    return values


def compute_wlx_prm(state: Mapping[str, Any]) -> dict[str, Any]:
    """Locate conservative negative process events on model-generation steps."""
    model_steps = list(state.get("model_steps") or ())
    adjustments = [0.0] * len(model_steps)
    rules: list[list[str]] = [[] for _ in model_steps]
    kinds: list[str] = ["normal"] * len(model_steps)

    def apply(index: int | None, value: float, rule: str, kind: str) -> None:
        if index is None or not 0 <= index < len(adjustments):
            return
        # One step uses only the strongest (most negative) process event.
        if value < adjustments[index]:
            adjustments[index] = float(value)
            kinds[index] = kind
        if rule not in rules[index]:
            rules[index].append(rule)

    # Penalize only sustained no-progress search/pagination, never positive progress.
    for index, step in enumerate(model_steps):
        progress = step.get("progress") or {}
        if not isinstance(progress, Mapping):
            continue
        if (
            str(step.get("tool") or "") in {"search_products", "next_page"}
            and not list(progress.get("runtime_progress_added") or ())
            and int(progress.get("no_progress_steps", 0)) >= 3
        ):
            apply(index, -0.25, "sustained_no_progress", "no_progress")

    orm = compute_wlx_orm(state)
    detail = state.get("reward_detail") or {}
    if not isinstance(detail, Mapping):
        detail = {}
    if orm["purchased"] and not orm["sampling_invalid"]:
        buy_index = _last_tool_index(model_steps, "buy_now")
        category_bad = orm["category_status"] == "fail"
        price_bad = orm["price_status"] in {"fail", "unverifiable"}
        failures = detail.get("option_failures") or {}
        missing_axes = (
            list(failures.get("missing_axes") or ())
            if isinstance(failures, Mapping)
            else []
        )
        wrong_values = _wrong_option_values(detail)
        localized_wrong = False
        for value in wrong_values:
            for index in range(len(model_steps) - 1, -1, -1):
                step = model_steps[index]
                if str(step.get("tool") or "") != "select_option":
                    continue
                parameters = step.get("parameters") or {}
                selected = parameters.get("value") if isinstance(parameters, Mapping) else None
                if _normalize(selected) == value:
                    apply(index, -0.50, "wrong_option", "wrong_option")
                    localized_wrong = True
                    break
        if category_bad:
            apply(buy_index, -1.0, "wrong_category_buy", "bad_buy")
        if price_bad:
            apply(buy_index, -1.0, "invalid_price_buy", "bad_buy")
        if missing_axes:
            apply(buy_index, -1.0, "missing_required_option_buy", "bad_buy")
        if wrong_values and not localized_wrong:
            apply(buy_index, -0.50, "unlocated_wrong_option", "wrong_option")

    return {
        "version": WLX_REWARD_VERSION,
        "step_prm": adjustments,
        "step_rules": rules,
        "step_kinds": kinds,
        "has_signal": any(value != 0.0 for value in adjustments),
    }


def reward_score_for_sampling(orm: Mapping[str, Any], prm: Mapping[str, Any]) -> float:
    """Encode PRM presence for the existing bounded dynamic sampler only."""
    score = float(orm["terminal_utility"])
    if bool(prm.get("has_signal")):
        score += WLX_PRM_SELECTION_MARKER
    return score
