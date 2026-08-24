"""把 WLX Harness 轨迹转换成带稳定事件 ID 的评测视图。"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence

from wlx_harness_core.wlx_eval_contracts import WLX_EVAL_EVENT_VERSION


def standardize_trajectory_events(
    trajectory: Mapping[str, Any],
    *,
    include_content: bool = False,
) -> tuple[dict[str, Any], ...]:
    """按消息顺序创建事件；默认只保存评测需要的紧凑元数据。"""

    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    messages = trajectory.get("messages") or []
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        messages = []

    steps_by_call_id = _steps_by_call_id(trajectory.get("steps"))
    blocked_by_call_id = _blocked_by_call_id(trajectory.get("blocked_tool_calls"))
    events: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            events.append(
                _event(
                    events,
                    kind="malformed_message",
                    message_index=message_index,
                )
            )
            continue
        role = str(message.get("role") or "unknown")
        calls = message.get("tool_calls")
        if role == "assistant" and isinstance(calls, Sequence) and not isinstance(
            calls, (str, bytes)
        ) and calls:
            for call_index, call in enumerate(calls):
                call_id, name, arguments = _openai_call(call)
                step = steps_by_call_id.get(call_id)
                blocked = blocked_by_call_id.get(call_id)
                payload = {
                    "role": role,
                    "message_index": message_index,
                    "call_index": call_index,
                    "call_id": call_id,
                    "tool_name": name,
                    "arguments": arguments,
                    "step_index": _optional_int(
                        (step or blocked or {}).get("step_index")
                    ),
                    "accepted": step is not None,
                    "guard_rejection": (
                        deepcopy(blocked.get("reason")) if blocked else None
                    ),
                }
                if include_content:
                    payload["assistant_content"] = deepcopy(message.get("content"))
                events.append(_event(events, kind="assistant_tool_call", **payload))
            continue

        kind = {
            "user": "user_request",
            "tool": "tool_observation",
            "system": "system_message",
            "assistant": "assistant_message",
        }.get(role, "message")
        payload = {
            "role": role,
            "message_index": message_index,
            "call_id": _optional_text(message.get("tool_call_id")),
            "tool_name": _optional_text(message.get("name")),
        }
        if include_content:
            payload["content"] = deepcopy(message.get("content"))
        events.append(_event(events, kind=kind, **payload))

    events.append(
        _event(
            events,
            kind="termination",
            termination_category=_optional_text(trajectory.get("termination_category")),
            termination_reason=_optional_text(trajectory.get("termination_reason")),
            status=_optional_text(trajectory.get("status")),
            done=bool(trajectory.get("done")),
        )
    )
    return tuple(events)


def event_id_for_step(
    events: Sequence[Mapping[str, Any]],
    step_index: int,
) -> str | None:
    """返回执行某个 step 的工具调用事件 ID。"""

    for event in events:
        if (
            event.get("kind") == "assistant_tool_call"
            and event.get("step_index") == step_index
        ):
            return _optional_text(event.get("event_id"))
    return None


def termination_event_id(events: Sequence[Mapping[str, Any]]) -> str | None:
    """返回终止事件 ID。"""

    for event in reversed(events):
        if event.get("kind") == "termination":
            return _optional_text(event.get("event_id"))
    return None


def _event(events: Sequence[object], *, kind: str, **payload: Any) -> dict[str, Any]:
    return {
        "event_version": WLX_EVAL_EVENT_VERSION,
        "event_id": f"E{len(events):03d}",
        "sequence_index": len(events),
        "kind": kind,
        **payload,
    }


def _steps_by_call_id(value: object) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for step in _objects(value):
        call = step.get("tool_call")
        if not isinstance(call, Mapping):
            continue
        call_id = _optional_text(call.get("call_id") or call.get("id"))
        if call_id:
            result[call_id] = deepcopy(dict(step))
    return result


def _blocked_by_call_id(value: object) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for blocked in _objects(value):
        call = blocked.get("tool_call")
        call_id = None
        if isinstance(call, Mapping):
            call_id = _optional_text(call.get("id") or call.get("call_id"))
        if call_id:
            result[call_id] = deepcopy(dict(blocked))
    return result


def _openai_call(value: object) -> tuple[str | None, str | None, object]:
    if not isinstance(value, Mapping):
        return None, None, None
    function = value.get("function")
    if not isinstance(function, Mapping):
        return _optional_text(value.get("id")), None, None
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, str):
        try:
            arguments: object = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = raw_arguments
    else:
        arguments = deepcopy(raw_arguments)
    return (
        _optional_text(value.get("id")),
        _optional_text(function.get("name")),
        arguments,
    )


def _objects(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "event_id_for_step",
    "standardize_trajectory_events",
    "termination_event_id",
]
