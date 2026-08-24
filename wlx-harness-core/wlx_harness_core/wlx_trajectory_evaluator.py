"""WLX Eval v2 逐轨迹入口，同时保留 v1 的顶层兼容字段。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from wlx_harness_core.wlx_eval_attribution import evaluate_deterministic_attribution
from wlx_harness_core.wlx_eval_behavior import evaluate_deterministic_behavior
from wlx_harness_core.wlx_eval_contracts import (
    NON_MODEL_INVALID,
    WLX_EVAL_SCHEMA_VERSION,
    pending_section,
    validate_evaluation_record,
)
from wlx_harness_core.wlx_eval_eligibility import (
    apply_outcome_review,
    evaluate_eligibility,
)
from wlx_harness_core.wlx_eval_events import (
    standardize_trajectory_events,
    termination_event_id,
)
from wlx_harness_core.wlx_eval_outcome import evaluate_outcome
from wlx_harness_core.wlx_eval_requirements import evaluate_frozen_requirements
from wlx_harness_core.wlx_tools import ToolRegistry, WLX_SFT_TOOL_REGISTRY


WLX_TRAJECTORY_EVALUATION_VERSION = WLX_EVAL_SCHEMA_VERSION
FORMAT_ERROR_REASONS = (
    "missing_tool_call",
    "multiple_tool_calls",
    "malformed_arguments",
    "unknown_tool",
    "schema_invalid",
)


@dataclass(frozen=True)
class FormatEvaluation:
    format_correct: bool
    first_format_error: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_correct": self.format_correct,
            "first_format_error": (
                deepcopy(dict(self.first_format_error))
                if self.first_format_error is not None
                else None
            ),
        }


@dataclass(frozen=True)
class WlxTrajectoryEvaluation:
    task_id: int | None
    trajectory_id: str | None
    trajectory_valid: bool
    invalid_reason: str | None
    format_correct: bool
    first_format_error: Mapping[str, Any] | None
    purchase_correct: bool | None
    purchase_type: str | None
    eligibility: Mapping[str, Any]
    format_panel: Mapping[str, Any]
    outcome: Mapping[str, Any]
    requirements: Mapping[str, Any]
    deterministic_behavior: Mapping[str, Any]
    process_quality: Mapping[str, Any]
    failure_attribution: Mapping[str, Any]
    review: Mapping[str, Any]
    event_index: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        row = {
            "schema_version": WLX_TRAJECTORY_EVALUATION_VERSION,
            "task_id": self.task_id,
            "trajectory_id": self.trajectory_id,
            "trajectory_valid": self.trajectory_valid,
            "invalid_reason": self.invalid_reason,
            "format_correct": self.format_correct,
            "first_format_error": (
                deepcopy(dict(self.first_format_error))
                if self.first_format_error is not None
                else None
            ),
            "purchase_correct": self.purchase_correct,
            "purchase_type": self.purchase_type,
            "eligibility": deepcopy(dict(self.eligibility)),
            "format": deepcopy(dict(self.format_panel)),
            "outcome": deepcopy(dict(self.outcome)),
            "requirements": deepcopy(dict(self.requirements)),
            "deterministic_behavior": deepcopy(dict(self.deterministic_behavior)),
            "process_quality": deepcopy(dict(self.process_quality)),
            "failure_attribution": deepcopy(dict(self.failure_attribution)),
            "review": deepcopy(dict(self.review)),
            "event_index": [deepcopy(dict(item)) for item in self.event_index],
        }
        return validate_evaluation_record(row)


def evaluate_format(
    trajectory: object,
    *,
    tool_registry: ToolRegistry = WLX_SFT_TOOL_REGISTRY,
) -> FormatEvaluation:
    """按 Assistant 回合顺序返回整条轨迹的首个格式错误。"""

    raw = _trajectory_mapping(trajectory)
    messages = raw.get("messages") or []
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return _format_failure(0, "malformed_arguments")

    assistant_index = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if calls is None or calls == []:
            return _format_failure(assistant_index, "missing_tool_call")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            return _format_failure(assistant_index, "malformed_arguments")
        if len(calls) > 1:
            return _format_failure(assistant_index, "multiple_tool_calls")
        call = calls[0]
        if not isinstance(call, Mapping):
            return _format_failure(assistant_index, "malformed_arguments")
        function = call.get("function")
        if not isinstance(function, Mapping):
            return _format_failure(assistant_index, "malformed_arguments")
        name = function.get("name")
        if not isinstance(name, str) or name not in tool_registry.tool_names:
            return _format_failure(assistant_index, "unknown_tool")
        arguments = _decode_arguments(function.get("arguments", {}))
        if arguments is None:
            return _format_failure(assistant_index, "malformed_arguments")
        parameters = tool_registry.schema(name)["function"].get("parameters") or {}
        if not _schema_accepts(arguments, parameters):
            return _format_failure(assistant_index, "schema_invalid")
        assistant_index += 1

    if assistant_index == 0:
        return _format_failure(0, "missing_tool_call")
    if raw.get("tool_call_truncations"):
        return _format_failure(0, "multiple_tool_calls")
    return FormatEvaluation(format_correct=True)


def evaluate_trajectory(
    trajectory: object,
    *,
    tool_registry: ToolRegistry = WLX_SFT_TOOL_REGISTRY,
    rubric: Mapping[str, Any] | None = None,
) -> WlxTrajectoryEvaluation:
    """生成 WLX Eval v2 确定性面板；语义 Judge 面板明确标为未运行。"""

    raw = _trajectory_mapping(trajectory)
    format_result = evaluate_format(raw, tool_registry=tool_registry)
    events = standardize_trajectory_events(raw)
    behavior = evaluate_deterministic_behavior(raw, events)
    outcome, requirements = evaluate_outcome(
        raw,
        terminal_event_id=termination_event_id(events),
    )
    if rubric is not None:
        requirements = evaluate_frozen_requirements(
            raw,
            rubric,
            terminal_event_id=termination_event_id(events),
        )
    eligibility = apply_outcome_review(
        evaluate_eligibility(raw),
        review_reasons=outcome.get("review_reasons") or [],
    )
    invalid_reason = _legacy_invalid_reason(eligibility)
    trajectory_valid = eligibility.get("status") != NON_MODEL_INVALID
    format_panel = format_result.to_dict()
    failure = evaluate_deterministic_attribution(
        eligibility=eligibility,
        format_panel=format_panel,
        outcome=outcome,
        behavior=behavior,
        events=events,
    )
    review_reasons = list(outcome.get("review_reasons") or [])
    if failure.get("review_required") and failure.get("primary_failure"):
        review_reasons.append(f"attribution:{failure['primary_failure']}")
    review_reasons = list(dict.fromkeys(str(item) for item in review_reasons))
    return WlxTrajectoryEvaluation(
        task_id=_optional_int(raw.get("task_id")),
        trajectory_id=_optional_text(raw.get("trajectory_id")),
        trajectory_valid=trajectory_valid,
        invalid_reason=invalid_reason,
        format_correct=format_result.format_correct,
        first_format_error=format_result.first_format_error,
        purchase_correct=(
            outcome.get("purchase_correct")
            if eligibility.get("status") != NON_MODEL_INVALID
            else None
        ),
        purchase_type=(
            outcome.get("purchase_type")
            if eligibility.get("status") != NON_MODEL_INVALID
            else None
        ),
        eligibility=eligibility,
        format_panel=format_panel,
        outcome=outcome,
        requirements=requirements,
        deterministic_behavior=behavior,
        process_quality=pending_section(
            "calibrated LLM judge and frozen human gold set are not configured"
        ),
        failure_attribution=failure,
        review={
            "required": bool(review_reasons),
            "reasons": review_reasons,
            "automatic_revision_applied": False,
        },
        event_index=events,
    )


def _legacy_invalid_reason(eligibility: Mapping[str, Any]) -> str | None:
    if eligibility.get("status") != NON_MODEL_INVALID:
        return None
    for issue in eligibility.get("integrity_issues") or []:
        if isinstance(issue, Mapping) and issue.get("code"):
            return str(issue["code"])
    return "non_model_invalid"


def _format_failure(turn_index: int, reason: str) -> FormatEvaluation:
    if reason not in FORMAT_ERROR_REASONS:
        raise ValueError(f"unsupported format error reason: {reason}")
    return FormatEvaluation(
        format_correct=False,
        first_format_error={"turn_index": int(turn_index), "reason": reason},
    )


def _decode_arguments(value: object) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value or "{}")
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _schema_accepts(value: object, schema: Mapping[str, Any]) -> bool:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, Mapping):
            return False
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        if any(name not in value for name in required):
            return False
        if schema.get("additionalProperties") is False and any(
            name not in properties for name in value
        ):
            return False
        return all(
            name not in value or _schema_accepts(value[name], child_schema)
            for name, child_schema in properties.items()
            if isinstance(child_schema, Mapping)
        )
    if expected_type == "string" and not isinstance(value, str):
        return False
    if expected_type == "integer" and (
        isinstance(value, bool) or not isinstance(value, int)
    ):
        return False
    if expected_type == "number" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        return False
    if expected_type == "boolean" and not isinstance(value, bool):
        return False
    if expected_type == "array" and (
        not isinstance(value, Sequence) or isinstance(value, (str, bytes))
    ):
        return False
    enum = schema.get("enum")
    return not isinstance(enum, Sequence) or value in enum


def _trajectory_mapping(trajectory: object) -> dict[str, Any]:
    converter = getattr(trajectory, "to_dict", None)
    if callable(converter):
        trajectory = converter()
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object or provide to_dict()")
    return deepcopy(dict(trajectory))


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "FORMAT_ERROR_REASONS",
    "FormatEvaluation",
    "WLX_TRAJECTORY_EVALUATION_VERSION",
    "WlxTrajectoryEvaluation",
    "evaluate_format",
    "evaluate_trajectory",
]
