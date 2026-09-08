"""用户需求 Rubric 的冻结契约；本模块不从隐藏目标补充需求。"""

from __future__ import annotations

from copy import deepcopy
import re
import unicodedata
from typing import Any, Mapping, Sequence


EVAL_RUBRIC_VERSION = "wlx-eval-rubric-v1"
RUBRIC_PRIORITIES = {"hard", "soft"}
RUBRIC_TYPES = {
    "category",
    "brand",
    "model",
    "function",
    "attribute",
    "option",
    "quantity",
    "budget",
    "preference",
    "other",
}


class RubricContractError(ValueError):
    """需求 Rubric 不可追溯到用户原始文本。"""


def freeze_rubric_bundle(
    *,
    task_id: int,
    instruction: str,
    items: Sequence[Mapping[str, Any]],
    rubric_set_version: str,
) -> dict[str, Any]:
    """校验并返回可冻结的单任务 Rubric。"""

    if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
        raise RubricContractError("task_id must be a non-negative integer")
    instruction = str(instruction).strip()
    if not instruction:
        raise RubricContractError("instruction must be non-empty")
    version = str(rubric_set_version).strip()
    if not version:
        raise RubricContractError("rubric_set_version must be non-empty")
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or not items:
        raise RubricContractError("rubric items must be a non-empty array")

    frozen_items = []
    seen: set[str] = set()
    normalized_instruction = _normalize(instruction)
    for index, raw in enumerate(items, start=1):
        if not isinstance(raw, Mapping):
            raise RubricContractError(f"rubric item {index} must be an object")
        requirement_id = str(raw.get("requirement_id") or f"R{index:03d}").strip()
        if not requirement_id or requirement_id in seen:
            raise RubricContractError(f"duplicate or empty requirement_id: {requirement_id!r}")
        seen.add(requirement_id)
        requirement = str(raw.get("requirement") or "").strip()
        query_evidence = str(raw.get("query_evidence") or "").strip()
        if not requirement or not query_evidence:
            raise RubricContractError(
                f"{requirement_id} must contain requirement and query_evidence"
            )
        if _normalize(query_evidence) not in normalized_instruction:
            raise RubricContractError(
                f"{requirement_id}.query_evidence is not present in user instruction"
            )
        priority = str(raw.get("priority") or "").strip()
        item_type = str(raw.get("type") or "").strip()
        if priority not in RUBRIC_PRIORITIES:
            raise RubricContractError(f"unsupported priority: {priority!r}")
        if item_type not in RUBRIC_TYPES:
            raise RubricContractError(f"unsupported requirement type: {item_type!r}")
        verifier = raw.get("verifier")
        if not isinstance(verifier, Mapping) or not verifier.get("kind"):
            raise RubricContractError(f"{requirement_id}.verifier must name a kind")
        frozen_items.append(
            {
                "requirement_id": requirement_id,
                "requirement": requirement,
                "type": item_type,
                "priority": priority,
                "query_evidence": query_evidence,
                "verifier": deepcopy(dict(verifier)),
            }
        )
    return {
        "schema_version": EVAL_RUBRIC_VERSION,
        "rubric_set_version": version,
        "task_id": task_id,
        "instruction": instruction,
        "items": frozen_items,
    }


def validate_rubric_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    """读取已有 Rubric 时复用冻结校验。"""

    if not isinstance(value, Mapping):
        raise TypeError("rubric bundle must be an object")
    if value.get("schema_version") != EVAL_RUBRIC_VERSION:
        raise RubricContractError("unexpected rubric schema_version")
    return freeze_rubric_bundle(
        task_id=value.get("task_id"),
        instruction=value.get("instruction"),
        items=value.get("items"),
        rubric_set_version=value.get("rubric_set_version"),
    )


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", "", text)


__all__ = [
    "RUBRIC_PRIORITIES",
    "RUBRIC_TYPES",
    "RubricContractError",
    "EVAL_RUBRIC_VERSION",
    "freeze_rubric_bundle",
    "validate_rubric_bundle",
]
