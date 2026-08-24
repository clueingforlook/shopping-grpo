"""从完整 WLX 轨迹计算不依赖主观判断的行为指标。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

from wlx_harness_core.wlx_eval_contracts import WLX_EVAL_BEHAVIOR_VERSION


DETAIL_TOOLS = {
    "view_description",
    "view_features",
    "view_reviews",
    "view_attributes",
}
PAGE_TOOLS = {"next_page", "prev_page"}


def evaluate_deterministic_behavior(
    trajectory: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """统计动作、搜索、候选、详情、循环、Token 和终止事实。"""

    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    attempted = [event for event in events if event.get("kind") == "assistant_tool_call"]
    executed = [event for event in attempted if event.get("accepted") is True]
    attempted_names = [_text(event.get("tool_name")) for event in attempted]
    executed_names = [_text(event.get("tool_name")) for event in executed]
    attempted_names = [name for name in attempted_names if name]
    executed_names = [name for name in executed_names if name]

    signatures = [_action_signature(event) for event in attempted]
    exact_repeat_count, max_exact_repeat_run = _repeat_stats(signatures)
    search_queries = [
        _argument_text(event, "query")
        for event in attempted
        if event.get("tool_name") == "search_products"
    ]
    search_queries = [query for query in search_queries if query]
    opened_asins = [
        _argument_text(event, "asin")
        for event in attempted
        if event.get("tool_name") == "open_product"
    ]
    opened_asins = [asin for asin in opened_asins if asin]
    selected_options = [
        _argument_text(event, "value")
        for event in attempted
        if event.get("tool_name") == "select_option"
    ]
    selected_options = [value for value in selected_options if value]

    steps = _objects(trajectory.get("steps"))
    products_seen: set[str] = set()
    result_pages: set[tuple[str, int]] = set()
    visible_observation_tokens = 0
    raw_observation_tokens = 0
    truncated_observations = 0
    for step in steps:
        projection = step.get("projection")
        if isinstance(projection, Mapping):
            visible_observation_tokens += _nonnegative_int(
                projection.get("visible_tokens")
            )
            raw_observation_tokens += _nonnegative_int(projection.get("raw_tokens"))
            truncated_observations += int(projection.get("truncated") is True)
        result = step.get("result")
        if not isinstance(result, Mapping):
            continue
        state = result.get("observation_state")
        if not isinstance(state, Mapping):
            continue
        products = state.get("products")
        for product in _objects(products):
            asin = _text(product.get("asin"))
            if asin:
                products_seen.add(asin)
        if state.get("page_type") == "search_results":
            query = _text(state.get("normalized_query") or state.get("query"))
            page = _nonnegative_int(state.get("page"))
            result_pages.add((query, page))

    turn_tokens = _objects(trajectory.get("context_turn_tokens"))
    input_tokens = [_nonnegative_int(row.get("input_tokens")) for row in turn_tokens]
    blocked = _objects(trajectory.get("blocked_tool_calls"))
    truncations = _objects(trajectory.get("tool_call_truncations"))
    compactions = _objects(trajectory.get("context_compactions"))
    termination = _text(trajectory.get("termination_reason"))
    terminal = trajectory.get("terminal_result")
    if isinstance(terminal, Mapping):
        detail = terminal.get("reward_detail")
        if isinstance(detail, Mapping):
            termination = _text(detail.get("termination_reason")) or termination
        termination = _text(terminal.get("termination_reason")) or termination

    event_refs = {
        "first_search": _first_event_id(attempted, "search_products"),
        "first_product_open": _first_event_id(attempted, "open_product"),
        "first_option_selection": _first_event_id(attempted, "select_option"),
        "first_purchase": _first_event_id(attempted, "buy_now"),
        "termination": next(
            (
                _text(event.get("event_id"))
                for event in reversed(events)
                if event.get("kind") == "termination"
            ),
            None,
        ),
    }
    return {
        "version": WLX_EVAL_BEHAVIOR_VERSION,
        "total_assistant_turns": len(
            {
                event.get("message_index")
                for event in events
                if event.get("role") == "assistant"
                and event.get("kind") in {"assistant_tool_call", "assistant_message"}
                and isinstance(event.get("message_index"), int)
            }
        ),
        "attempted_tool_calls": len(attempted),
        "executed_tool_calls": len(steps),
        "valid_action_count": len(steps),
        "invalid_action_count": len(blocked)
        + int(str(trajectory.get("termination_category") or "") == "invalid_action"),
        "tool_counts_attempted": dict(sorted(Counter(attempted_names).items())),
        "tool_counts_executed": dict(sorted(Counter(executed_names).items())),
        "guard_rejection_count": len(blocked),
        "parallel_tool_truncation_count": len(truncations),
        "context_compaction_count": len(compactions),
        "search_count": sum(name == "search_products" for name in attempted_names),
        "unique_search_query_count": len({_normalize_text(item) for item in search_queries}),
        "search_queries": search_queries,
        "page_navigation_count": sum(name in PAGE_TOOLS for name in attempted_names),
        "unique_result_page_count": len(result_pages),
        "products_seen_count": len(products_seen),
        "product_open_count": sum(name == "open_product" for name in attempted_names),
        "unique_product_open_count": len(set(opened_asins)),
        "opened_asins": opened_asins,
        "detail_view_count": sum(name in DETAIL_TOOLS for name in attempted_names),
        "option_selection_count": sum(name == "select_option" for name in attempted_names),
        "selected_options": selected_options,
        "purchase_call_count": sum(name == "buy_now" for name in attempted_names),
        "finish_without_purchase_count": sum(
            name == "finish_without_purchase" for name in attempted_names
        ),
        "exact_consecutive_repeat_count": exact_repeat_count,
        "max_exact_repeat_run": max_exact_repeat_run,
        "environment_repeat_loop": termination == "repeat_loop",
        "input_token_sum": sum(input_tokens),
        "max_turn_input_tokens": max(input_tokens, default=0),
        "visible_observation_token_sum": visible_observation_tokens,
        "raw_observation_token_sum": raw_observation_tokens,
        "truncated_observation_count": truncated_observations,
        "termination_category": _text(trajectory.get("termination_category")),
        "termination_reason": termination,
        "status": _text(trajectory.get("status")),
        "event_refs": event_refs,
    }


def _action_signature(event: Mapping[str, Any]) -> str:
    payload = {
        "name": event.get("tool_name"),
        "arguments": event.get("arguments"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _repeat_stats(signatures: Sequence[str]) -> tuple[int, int]:
    if not signatures:
        return 0, 0
    repeats = 0
    current_run = 1
    longest = 1
    for previous, current in zip(signatures, signatures[1:]):
        if current == previous:
            repeats += 1
            current_run += 1
            longest = max(longest, current_run)
        else:
            current_run = 1
    return repeats, longest


def _argument_text(event: Mapping[str, Any], name: str) -> str | None:
    arguments = event.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    return _text(arguments.get(name))


def _first_event_id(events: Sequence[Mapping[str, Any]], tool_name: str) -> str | None:
    return next(
        (
            _text(event.get("event_id"))
            for event in events
            if event.get("tool_name") == tool_name
        ),
        None,
    )


def _objects(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


__all__ = ["DETAIL_TOOLS", "PAGE_TOOLS", "evaluate_deterministic_behavior"]
