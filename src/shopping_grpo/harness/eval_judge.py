"""过程 Judge 的无泄漏输入和严格输出契约。

这里故意不绑定具体模型服务。只有完成人工 Gold Set 校准后，调用方才应把
这个 payload 发给 Judge。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping, Sequence

from shopping_grpo.harness.eval_contracts import (
    PROCESS_DIMENSIONS,
    REQUIREMENT_STATUSES,
    RESPONSIBILITIES,
    EVAL_METHOD_VERSION,
)
from shopping_grpo.harness.eval_events import standardize_trajectory_events
from shopping_grpo.harness.eval_rubric import validate_rubric_bundle


EVAL_JUDGE_INPUT_VERSION = "wlx-eval-v2-judge-input-v1"
EVAL_JUDGE_OUTPUT_VERSION = "wlx-eval-v2-judge-process-output-v1"
EVAL_JUDGE_FULL_OUTPUT_VERSION = "wlx-eval-v2-judge-output-v1"
EVAL_JUDGE_PROMPT_VERSION = "wlx-eval-v2-judge-prompt-v2"
FAILURE_TYPES = {
    "search_strategy_error",
    "candidate_utilization_error",
    "insufficient_verification",
    "wrong_product_selection",
    "wrong_option_selection",
    "over_budget_purchase",
    "premature_purchase",
    "premature_abstention",
    "repeat_loop_no_progress",
    "invalid_action",
    "context_or_action_budget_exhausted",
    "task_requirement_conflict",
    "verifier_mismatch",
    "semantic_normalization_gap",
    "environment_unverifiable",
    "environment_error",
    "harness_error",
    "model_service_error",
    "insufficient_evidence",
    "none",
}


class JudgeContractError(ValueError):
    """Judge 输入或输出违反盲评、证据或分数契约。"""


def build_process_judge_payload(
    *,
    trajectory: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    """只投影模型当时可见的消息，不包含 Reward、Gold ASIN 或隐藏字段。"""

    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    frozen = validate_rubric_bundle(rubric)
    if trajectory.get("task_id") != frozen.get("task_id"):
        raise JudgeContractError("trajectory and rubric task_id mismatch")
    events = standardize_trajectory_events(trajectory, include_content=True)
    public_events = [
        _public_event(event)
        for event in events
        if event.get("kind") != "termination"
    ]
    event_ids = [str(event["event_id"]) for event in public_events]
    return {
        "schema_version": EVAL_JUDGE_INPUT_VERSION,
        "task_id": trajectory.get("task_id"),
        "user_instruction": frozen["instruction"],
        "rubric": deepcopy(frozen),
        "events": public_events,
        "rule_gate_facts": _rule_gate_facts(trajectory, public_events),
        "allowed_evidence_event_ids": event_ids,
        "dimensions": list(PROCESS_DIMENSIONS),
        "blind_fields": [
            "model_identity",
            "reward",
            "gold_asin",
            "hidden_goal",
            "raw_unprojected_observation",
        ],
    }


def build_full_judge_payload(
    *,
    trajectory: Mapping[str, Any],
    rubric: Mapping[str, Any],
) -> dict[str, Any]:
    """构造全部需求、过程评分和第一错误归因的统一盲评输入。"""

    payload = build_process_judge_payload(trajectory=trajectory, rubric=rubric)
    requirements = [deepcopy(dict(item)) for item in payload["rubric"]["items"]]
    payload.update(
        {
            "schema_version": "wlx-eval-v2-judge-full-input-v1",
            "method_version": EVAL_METHOD_VERSION,
            "requirements_to_judge": requirements,
            "required_requirement_ids": [
                str(item["requirement_id"]) for item in requirements
            ],
            "failure_types": sorted(FAILURE_TYPES),
            "responsibilities": list(RESPONSIBILITIES),
        }
    )
    return payload


def build_full_judge_messages(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """生成只包含盲评 payload 的严格 JSON Judge Prompt。"""

    system = f"""你是购物 Agent 轨迹评测 Judge，只依据给定用户需求、Rubric 和模型当时可见事件。

禁止使用或猜测参考商品、隐藏目标、Reward、模型身份和轨迹外信息。

任务：
1. 对 requirements_to_judge 中每一项都输出 satisfied/violated/unknown/conflicting；不得跳过价格、数量、类别、品牌或规格等看似可用字符串规则判断的项目。证据不足必须 unknown。
2. 对五个过程维度分别给 0/1/2 分：search_strategy、candidate_utilization、evidence_verification、decision_quality、termination_efficiency。
3. 判断责任来源与第一处真正导致结果的错误；成功且无失败时 primary_failure="none"。
4. 所有证据只能引用 allowed_evidence_event_ids。不要因为最终结果正确就倒推过程一定正确。
5. 低置信度、信息缺失或确定性事实可能冲突时设置 review_required=true。
6. Rubric 中 verifier 只描述需求来源与建议核验方式，不是程序已经给出的答案；你必须根据用户原文和可见轨迹独立判断全部 Rubric。
7. rule_gate_facts 只是由公开动作计算的次数与中性终止事实，不包含成功答案；不能用它替代 Rubric 证据。

Rubric 的评价对象：
- 每条需求只评价“最终实际购买的商品和最终选项”，不是评价搜索过程中见过的任意候选。
- 只有 accepted=true 的 buy_now 才表示完成了购买尝试。若没有 accepted buy_now，所有商品需求一律为 unknown；不得因为候选列表或 available_options 中出现过合适商品就判 satisfied。
- 有最终购买时，只使用该次 buy_now 之前最近的已选商品状态。available_options 只代表可选，不代表模型实际选择；以 selected_options 为准。
- 商品标题、key_attributes、价格、selected_options 等结构化可见字段都是有效证据。宽泛或明显噪声较大的 category 不应推翻更具体且一致的标题、属性和选项。

严格按以下边界评分，五个维度相互独立，不要因为最终成功或失败反推其他维度：
- search_strategy：只评价查询理解和检索收敛，不要求把全部约束塞进一个查询。2=简洁查询直接命中强候选，或查询覆盖多数关键约束，或改写后有效收敛；1=方向正确但查询较宽/窄、命中较弱或调整有限；0=误解需求、查询无关或持续无效。
- candidate_utilization：只评价候选选择和利用，导航冗余另归 termination_efficiency。2=优先打开并充分利用强候选；首个候选已足够强时不强制比较。1=看过相关候选但利用或筛选不充分。0=忽略已经出现的强候选、选择明显较差候选，或主要查看无关商品。
- evidence_verification：“核对”指关键事实在购买/停止前真实出现在模型可见 Observation 中，不要求 Assistant 显式复述，也不要求点击特定子页面。2=可见信息覆盖全部决策关键要求；1=只覆盖部分，仍有重要遗漏；0=基本没有关键证据。空白 information_subpage 不增加证据，但也不能抹去商品详情页已经显示的标题、属性、价格和选项。
- decision_quality：2=最终购买或主动放弃与充分证据一致；1=确实形成了有一定依据但证据不完整的决定，或在充分探索后合理主动放弃；0=无据错购、明显错过强候选、在已有反证时仍购买，或因循环/非法动作/预算耗尽而被动结束。仅仅做过一些探索不能自动得到1分。
- termination_efficiency：2=证据足够后及时正确结束；1=正确或可辩护地结束但有明显冗余；0=错误或证据不足时过早购买/放弃，或因循环、非法动作、上下文/动作预算耗尽而结束。动作少不代表终止有效。

需求与归因边界：
- unknown 只表示证据不足，不等于 violated，也不能仅凭 unknown 判定模型失败。
- attribution 只描述“失败责任”。若 primary_failure="none"，responsibility 必须是 "undetermined"，first_error_event_id 必须为 null。
- 若有模型失败，优先定位最早且可恢复的因果错误，例如错过强候选、未核对关键选项或选错规格；不要把后来的预算耗尽、循环结束或放弃症状机械地当作第一错误。
- context_or_action_budget_exhausted、premature_abstention 或 repeat_loop_no_progress 仅在找不到更早的策略性原因时作为 primary_failure，否则放入 secondary_failures。

只输出 JSON 对象：
{{
  "schema_version": "{EVAL_JUDGE_FULL_OUTPUT_VERSION}",
  "requirements": [{{"requirement_id":"R001","status":"unknown","reason":"...","evidence_event_ids":["E001"],"confidence":0.8}}],
  "dimensions": {{
    "search_strategy": {{"score":0,"reason":"...","evidence_event_ids":["E001"],"confidence":0.8}},
    "candidate_utilization": {{"score":0,"reason":"...","evidence_event_ids":[],"confidence":0.8}},
    "evidence_verification": {{"score":0,"reason":"...","evidence_event_ids":[],"confidence":0.8}},
    "decision_quality": {{"score":0,"reason":"...","evidence_event_ids":[],"confidence":0.8}},
    "termination_efficiency": {{"score":0,"reason":"...","evidence_event_ids":[],"confidence":0.8}}
  }},
  "attribution": {{"responsibility":"model","primary_failure":"none","secondary_failures":[],"first_error_event_id":null,"reason":"...","confidence":0.8}},
  "review_required": false,
  "review_reasons": []
}}

Prompt 版本：{EVAL_JUDGE_PROMPT_VERSION}"""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


EVAL_JUDGE_SYSTEM_PROMPT_SHA256 = hashlib.sha256(
    build_full_judge_messages({})[0]["content"].encode("utf-8")
).hexdigest()


def validate_process_judgment(
    value: Mapping[str, Any],
    *,
    allowed_event_ids: Sequence[str],
) -> dict[str, Any]:
    """验证 Judge 的五维评分、理由、证据和置信度。"""

    if not isinstance(value, Mapping):
        raise TypeError("judge output must be an object")
    if value.get("schema_version") != EVAL_JUDGE_OUTPUT_VERSION:
        raise JudgeContractError("unexpected judge output schema_version")
    allowed = set(str(item) for item in allowed_event_ids)
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise JudgeContractError("judge dimensions must be an object")
    if set(dimensions) != set(PROCESS_DIMENSIONS):
        raise JudgeContractError("judge must return exactly the five process dimensions")
    normalized: dict[str, Any] = {}
    for name in PROCESS_DIMENSIONS:
        item = dimensions[name]
        if not isinstance(item, Mapping):
            raise JudgeContractError(f"dimension {name} must be an object")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or score not in {0, 1, 2}:
            raise JudgeContractError(f"dimension {name}.score must be 0, 1 or 2")
        reason = str(item.get("reason") or "").strip()
        if not reason:
            raise JudgeContractError(f"dimension {name}.reason must be non-empty")
        evidence = item.get("evidence_event_ids") or []
        if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
            raise JudgeContractError(f"dimension {name}.evidence_event_ids must be an array")
        evidence = [str(event_id) for event_id in evidence]
        unknown = sorted(set(evidence).difference(allowed))
        if unknown:
            raise JudgeContractError(f"dimension {name} cites unknown events: {unknown}")
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise JudgeContractError(f"dimension {name}.confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise JudgeContractError(f"dimension {name}.confidence must be in [0, 1]")
        normalized[name] = {
            "score": score,
            "reason": reason,
            "evidence_event_ids": evidence,
            "confidence": float(confidence),
        }
    return {
        "schema_version": EVAL_JUDGE_OUTPUT_VERSION,
        "dimensions": normalized,
        "review_required": bool(value.get("review_required")),
        "review_reasons": [
            str(item) for item in value.get("review_reasons") or []
        ],
    }


def validate_full_judgment(
    value: Mapping[str, Any],
    *,
    allowed_event_ids: Sequence[str],
    required_requirement_ids: Sequence[str],
) -> dict[str, Any]:
    """验证完整 Judge 输出，不允许遗漏需求项或引用不可见证据。"""

    if not isinstance(value, Mapping):
        raise TypeError("judge output must be an object")
    if value.get("schema_version") != EVAL_JUDGE_FULL_OUTPUT_VERSION:
        raise JudgeContractError("unexpected full judge output schema_version")
    process = validate_process_judgment(
        {
            "schema_version": EVAL_JUDGE_OUTPUT_VERSION,
            "dimensions": value.get("dimensions"),
            "review_required": value.get("review_required"),
            "review_reasons": value.get("review_reasons"),
        },
        allowed_event_ids=allowed_event_ids,
    )
    allowed = set(str(item) for item in allowed_event_ids)
    expected = [str(item) for item in required_requirement_ids]
    requirements = value.get("requirements")
    if isinstance(requirements, (str, bytes)) or not isinstance(requirements, Sequence):
        raise JudgeContractError("judge requirements must be an array")
    normalized_requirements = []
    seen: set[str] = set()
    for item in requirements:
        if not isinstance(item, Mapping):
            raise JudgeContractError("judge requirement result must be an object")
        requirement_id = str(item.get("requirement_id") or "")
        if not requirement_id or requirement_id in seen:
            raise JudgeContractError("judge requirement_id is empty or duplicated")
        seen.add(requirement_id)
        status = str(item.get("status") or "")
        if status not in REQUIREMENT_STATUSES:
            raise JudgeContractError(f"unsupported requirement status: {status}")
        normalized_requirements.append(
            {
                "requirement_id": requirement_id,
                "status": status,
                "reason": _required_reason(item),
                "evidence_event_ids": _evidence_ids(item, allowed),
                "confidence": _confidence(item),
            }
        )
    if seen != set(expected) or len(normalized_requirements) != len(expected):
        raise JudgeContractError("judge requirements do not exactly cover all Rubric ids")

    attribution = value.get("attribution")
    if not isinstance(attribution, Mapping):
        raise JudgeContractError("judge attribution must be an object")
    responsibility = str(attribution.get("responsibility") or "")
    primary = str(attribution.get("primary_failure") or "")
    if responsibility not in RESPONSIBILITIES:
        raise JudgeContractError("unsupported attribution responsibility")
    if primary not in FAILURE_TYPES:
        raise JudgeContractError("unsupported primary_failure")
    secondary = attribution.get("secondary_failures") or []
    if isinstance(secondary, (str, bytes)) or not isinstance(secondary, Sequence):
        raise JudgeContractError("secondary_failures must be an array")
    secondary = [str(item) for item in secondary]
    if any(item not in FAILURE_TYPES or item == "none" for item in secondary):
        raise JudgeContractError("unsupported secondary_failure")
    first_error = attribution.get("first_error_event_id")
    if first_error is not None and str(first_error) not in allowed:
        raise JudgeContractError("first_error_event_id is not visible to judge")
    return {
        "schema_version": EVAL_JUDGE_FULL_OUTPUT_VERSION,
        "requirements": normalized_requirements,
        "dimensions": process["dimensions"],
        "attribution": {
            "responsibility": responsibility,
            "primary_failure": primary,
            "secondary_failures": secondary,
            "first_error_event_id": str(first_error) if first_error is not None else None,
            "reason": _required_reason(attribution),
            "confidence": _confidence(attribution),
        },
        "review_required": process["review_required"],
        "review_reasons": process["review_reasons"],
    }


def _required_reason(item: Mapping[str, Any]) -> str:
    reason = str(item.get("reason") or "").strip()
    if not reason:
        raise JudgeContractError("judge reason must be non-empty")
    return reason


def _evidence_ids(item: Mapping[str, Any], allowed: set[str]) -> list[str]:
    raw = item.get("evidence_event_ids") or []
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise JudgeContractError("evidence_event_ids must be an array")
    result = [str(event_id) for event_id in raw]
    unknown = sorted(set(result).difference(allowed))
    if unknown:
        raise JudgeContractError(f"judge cites unknown events: {unknown}")
    return result


def _confidence(item: Mapping[str, Any]) -> float:
    confidence = item.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise JudgeContractError("judge confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise JudgeContractError("judge confidence must be in [0, 1]")
    return float(confidence)


def _public_event(event: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "event_id",
        "sequence_index",
        "kind",
        "role",
        "message_index",
        "call_index",
        "call_id",
        "tool_name",
        "arguments",
        "step_index",
        "accepted",
        "guard_rejection",
        "assistant_content",
        "content",
    }
    return {
        key: deepcopy(value)
        for key, value in event.items()
        if key in allowed
    }


def _rule_gate_facts(
    trajectory: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    calls = [event for event in events if event.get("kind") == "assistant_tool_call"]
    signatures = [
        json.dumps(
            [event.get("tool_name"), event.get("arguments")],
            ensure_ascii=False,
            sort_keys=True,
        )
        for event in calls
    ]
    accepted_purchase_event_ids = [
        str(event["event_id"])
        for event in calls
        if event.get("tool_name") == "buy_now" and event.get("accepted") is True
    ]
    return {
        "tool_call_count": len(calls),
        "accepted_tool_call_count": sum(event.get("accepted") is True for event in calls),
        "guard_rejection_count": sum(bool(event.get("guard_rejection")) for event in calls),
        "search_count": sum(event.get("tool_name") == "search_products" for event in calls),
        "product_open_count": sum(event.get("tool_name") == "open_product" for event in calls),
        "purchase_call_count": sum(event.get("tool_name") == "buy_now" for event in calls),
        "accepted_purchase_event_ids": accepted_purchase_event_ids,
        "has_accepted_purchase": bool(accepted_purchase_event_ids),
        "exact_repeated_call_count": len(signatures) - len(set(signatures)),
        "context_compaction_count": len(trajectory.get("context_compactions") or []),
        "tool_call_truncation_count": len(trajectory.get("tool_call_truncations") or []),
        "terminal_done": bool(trajectory.get("done")),
    }


__all__ = [
    "JudgeContractError",
    "FAILURE_TYPES",
    "EVAL_JUDGE_FULL_OUTPUT_VERSION",
    "EVAL_JUDGE_INPUT_VERSION",
    "EVAL_JUDGE_OUTPUT_VERSION",
    "EVAL_JUDGE_PROMPT_VERSION",
    "EVAL_JUDGE_SYSTEM_PROMPT_SHA256",
    "build_full_judge_messages",
    "build_full_judge_payload",
    "build_process_judge_payload",
    "validate_full_judgment",
    "validate_process_judgment",
]
