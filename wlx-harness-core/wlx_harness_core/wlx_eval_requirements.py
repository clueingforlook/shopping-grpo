"""Eval v2 的 Rubric 待判定面板与 Judge 结果合并。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from wlx_harness_core.wlx_eval_contracts import REQUIREMENT_STATUSES
from wlx_harness_core.wlx_eval_rubric import validate_rubric_bundle


WLX_EVAL_REQUIREMENTS_VERSION = "wlx-eval-v2-requirements-v1"


def evaluate_frozen_requirements(
    trajectory: Mapping[str, Any],
    rubric: Mapping[str, Any],
    *,
    terminal_event_id: str | None = None,
) -> dict[str, Any]:
    """为全部 Rubric 项建立待 Judge 面板，不用字符串规则作语义结论。

    保留原函数名是为了让现有离线入口继续工作。Eval v2 中，程序只提供
    客观轨迹/环境事实；每条自然语言需求均由同一个校准后的 Judge 判断。
    """

    frozen = validate_rubric_bundle(rubric)
    if trajectory.get("task_id") != frozen.get("task_id"):
        raise ValueError("trajectory and rubric task_id mismatch")
    evidence = [terminal_event_id] if terminal_event_id else []
    items = [
        {
            **deepcopy(dict(item)),
            "status": "unknown",
            "method": "calibrated_llm_judge_pending",
            "reason": "all query requirements await the calibrated Eval v2 judge",
            "actual": None,
            "evidence_event_ids": evidence,
            "confidence": 0.0,
        }
        for item in frozen["items"]
    ]
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in REQUIREMENT_STATUSES
    }
    return {
        "version": WLX_EVAL_REQUIREMENTS_VERSION,
        "status": "llm_judge_pending",
        "included_in_final_metrics": False,
        "reason": "all query requirements await calibrated Eval v2 judge",
        "rubric_set_version": frozen["rubric_set_version"],
        "instruction": frozen["instruction"],
        "items": items,
        "counts": counts,
    }


def merge_judged_requirements(
    deterministic: Mapping[str, Any],
    judgments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """以 requirement_id 合并 Judge 对全部 Rubric 项的判断。"""

    judged = {str(item.get("requirement_id")): item for item in judgments}
    expected = {
        str(item.get("requirement_id")) for item in deterministic.get("items") or []
    }
    if len(judged) != len(judgments) or set(judged) != expected:
        raise ValueError("Judge requirements must exactly cover every frozen Rubric item")
    result = deepcopy(dict(deterministic))
    merged = []
    for raw in deterministic.get("items") or []:
        item = deepcopy(dict(raw))
        decision = judged.get(str(item.get("requirement_id")))
        if not isinstance(decision, Mapping):  # guarded by the exact-ID check above
            raise ValueError("Judge requirement result is missing")
        item.update(
            {
                "status": decision["status"],
                "method": "calibrated_llm_judge",
                "reason": decision["reason"],
                "evidence_event_ids": list(decision["evidence_event_ids"]),
                "confidence": float(decision["confidence"]),
            }
        )
        merged.append(item)
    result["items"] = merged
    result["counts"] = {
        status: sum(item["status"] == status for item in merged)
        for status in REQUIREMENT_STATUSES
    }
    result["status"] = "complete"
    result["included_in_final_metrics"] = True
    result["reason"] = None
    return result


__all__ = [
    "WLX_EVAL_REQUIREMENTS_VERSION",
    "evaluate_frozen_requirements",
    "merge_judged_requirements",
]
