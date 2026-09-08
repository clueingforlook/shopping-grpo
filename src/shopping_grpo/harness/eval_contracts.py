"""Eval v2 的稳定枚举、Schema 版本和轻量校验。"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping, Sequence


EVAL_SCHEMA_VERSION = "wlx-eval-v2"
EVAL_METHOD_VERSION = "wlx-eval-v2"
EVAL_EVENT_VERSION = "wlx-eval-events-v1"
EVAL_ELIGIBILITY_VERSION = "wlx-eval-eligibility-v1"
EVAL_OUTCOME_VERSION = "wlx-eval-outcome-v1"
EVAL_BEHAVIOR_VERSION = "wlx-eval-behavior-v1"
EVAL_ATTRIBUTION_VERSION = "wlx-eval-attribution-v1"
EVAL_REPORT_VERSION = "wlx-eval-report-v1"
EVAL_OFFLINE_RUN_VERSION = "wlx-offline-evaluation-v1"

ELIGIBLE = "eligible"
REVIEW_REQUIRED = "review_required"
NON_MODEL_INVALID = "non_model_invalid"
EVALUATION_ELIGIBILITIES = (ELIGIBLE, REVIEW_REQUIRED, NON_MODEL_INVALID)

MODEL_RESPONSIBILITY = "model"
NON_MODEL_RESPONSIBILITY = "non_model"
MIXED_RESPONSIBILITY = "mixed"
UNDETERMINED_RESPONSIBILITY = "undetermined"
RESPONSIBILITIES = (
    MODEL_RESPONSIBILITY,
    NON_MODEL_RESPONSIBILITY,
    MIXED_RESPONSIBILITY,
    UNDETERMINED_RESPONSIBILITY,
)

OUTCOME_TYPES = (
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_purchase",
    "wrong_purchase",
    "no_purchase",
    "unverifiable",
)

REQUIREMENT_STATUSES = (
    "satisfied",
    "violated",
    "unknown",
    "not_applicable",
    "conflicting",
)

PROCESS_DIMENSIONS = (
    "search_strategy",
    "candidate_utilization",
    "evidence_verification",
    "decision_quality",
    "termination_efficiency",
)


class EvalContractError(ValueError):
    """Eval 记录不满足已经冻结的数据契约。"""


def pending_section(reason: str) -> dict[str, Any]:
    """生成一个不会与缺失字段混淆的“尚未执行”面板。"""

    text = str(reason).strip()
    if not text:
        raise ValueError("pending section reason must be non-empty")
    return {"status": "not_run", "reason": text}


def ensure_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    """返回 JSON 安全副本，同时拒绝 NaN、集合等不稳定值。"""

    if not isinstance(value, Mapping):
        raise TypeError("evaluation value must be an object")
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - 编码入口已经保证
        raise TypeError("evaluation value must encode to an object")
    return decoded


def validate_evaluation_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """检查离线管线写盘前最重要的跨面板约束。"""

    row = ensure_json_object(value)
    if row.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise EvalContractError("unexpected Eval schema_version")
    task_id = row.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise EvalContractError("task_id must be a non-negative integer")

    eligibility = row.get("eligibility")
    if not isinstance(eligibility, Mapping):
        raise EvalContractError("eligibility must be an object")
    status = eligibility.get("status")
    if status not in EVALUATION_ELIGIBILITIES:
        raise EvalContractError(f"unsupported eligibility status: {status!r}")
    responsibility = eligibility.get("responsibility")
    if responsibility not in RESPONSIBILITIES:
        raise EvalContractError(f"unsupported responsibility: {responsibility!r}")

    format_panel = row.get("format")
    if not isinstance(format_panel, Mapping) or not isinstance(
        format_panel.get("format_correct"), bool
    ):
        raise EvalContractError("format.format_correct must be boolean")

    outcome = row.get("outcome")
    if not isinstance(outcome, Mapping):
        raise EvalContractError("outcome must be an object")
    normalized = outcome.get("normalized_outcome")
    if normalized not in OUTCOME_TYPES:
        raise EvalContractError(f"unsupported normalized outcome: {normalized!r}")

    behavior = row.get("deterministic_behavior")
    if not isinstance(behavior, Mapping):
        raise EvalContractError("deterministic_behavior must be an object")
    if behavior.get("total_assistant_turns", 0) < 0:
        raise EvalContractError("behavior counts must be non-negative")

    requirements = row.get("requirements")
    if not isinstance(requirements, Mapping):
        raise EvalContractError("requirements must be an object")
    items = requirements.get("items") or []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise EvalContractError("requirements.items must be an array")
    for item in items:
        if not isinstance(item, Mapping):
            raise EvalContractError("each requirement item must be an object")
        if item.get("status") not in REQUIREMENT_STATUSES:
            raise EvalContractError(
                f"unsupported requirement status: {item.get('status')!r}"
            )

    return deepcopy(row)


__all__ = [
    "ELIGIBLE",
    "EVALUATION_ELIGIBILITIES",
    "EvalContractError",
    "MIXED_RESPONSIBILITY",
    "MODEL_RESPONSIBILITY",
    "NON_MODEL_INVALID",
    "NON_MODEL_RESPONSIBILITY",
    "OUTCOME_TYPES",
    "PROCESS_DIMENSIONS",
    "REQUIREMENT_STATUSES",
    "RESPONSIBILITIES",
    "REVIEW_REQUIRED",
    "UNDETERMINED_RESPONSIBILITY",
    "EVAL_ATTRIBUTION_VERSION",
    "EVAL_BEHAVIOR_VERSION",
    "EVAL_ELIGIBILITY_VERSION",
    "EVAL_EVENT_VERSION",
    "EVAL_METHOD_VERSION",
    "EVAL_OFFLINE_RUN_VERSION",
    "EVAL_OUTCOME_VERSION",
    "EVAL_REPORT_VERSION",
    "EVAL_SCHEMA_VERSION",
    "ensure_json_object",
    "pending_section",
    "validate_evaluation_record",
]
