"""Rubric 的规则预提取、DeepSeek 生成和确定性审计。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import RemoteDisconnected
import hashlib
import json
import math
import re
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shopping_grpo.harness.eval_rubric import (
    RUBRIC_PRIORITIES,
    RUBRIC_TYPES,
    RubricContractError,
    freeze_rubric_bundle,
)


RUBRIC_PROMPT_VERSION = "wlx-rubric-prompt-v1"
RUBRIC_GENERATOR_VERSION = "wlx-rubric-generator-v1"
RUBRIC_RULES_VERSION = "wlx-rubric-rules-v1"

RETRYABLE_HTTP_STATUSES = {408, 409, 429, 500, 502, 503, 504}
VERIFIER_KINDS = {"deterministic", "semantic"}
_PROMPT_DETERMINISTIC_OPERATORS = {
    "equals",
    "contains",
    "category_match",
    "option_match",
    "lte",
    "gte",
    "approx",
}
DETERMINISTIC_OPERATORS = _PROMPT_DETERMINISTIC_OPERATORS | {
    "lt",
    "gt",
    "between",
}
_OPERATOR_ALIASES = {
    "<": "lt",
    "less_than": "lt",
    "<=": "lte",
    "less_than_or_equal": "lte",
    ">": "gt",
    "greater_than": "gt",
    ">=": "gte",
    "greater_than_or_equal": "gte",
    "range": "between",
    "in_range": "between",
    "within_range": "between",
}
SOFT_MARKERS = ("最好", "优先", "尽量", "左右", "上下", "大约", "约", "差不多")
HARD_MARKERS = ("必须", "需要", "要", "不能", "不要", "不超过", "以内", "以下")
NEGATIVE_MARKERS = ("不要", "不能", "不含", "无", "避免")

_NUMBER_PATTERN = re.compile(
    r"(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>人民币|块钱|元|块|厘米|毫米|cm|mm|kg|公斤|克|g|ml|毫升|"
    r"升|l|米|m|英寸|寸|个|件|套|盒|包|袋|支|瓶|片|对|双)"
    r"(?P<qualifier>以内|以下|不超过|至多|最多|以上|至少|左右|上下|大约|约)?",
    flags=re.IGNORECASE,
)
_ARABIC_CURRENCY_RANGE = re.compile(
    r"(?P<minimum>\d+(?:\.\d+)?)\s*"
    r"(?P<minimum_unit>人民币|块钱|元|块)?\s*"
    r"(?:到|至|[-~～—–])\s*"
    r"(?P<maximum>\d+(?:\.\d+)?)\s*"
    r"(?P<maximum_unit>人民币|块钱|元|块)"
)
_ARABIC_BUDGET_CONTEXT = re.compile(
    r"(?P<prefix>预算(?:控制)?|价格|价位|售价|心里价格)\s*"
    r"(?:控制)?\s*(?:大概|大约|约)?\s*(?:在|为|是)?\s*"
    r"(?P<precompare>不超过|至多|最多|低于|小于)?\s*"
    r"(?P<number>\d+(?:\.\d+)?)\s*(?P<magnitude>千|万)?\s*"
    r"(?P<currency>人民币|块钱|元|块)?\s*"
    r"(?P<qualifier>以内|以下|不超过|至多|最多|之内|及以内|左右|上下|"
    r"大约|约|多|出头)?"
)
_CHINESE_CURRENCY_BUDGET = re.compile(
    r"(?P<number>[零〇一二两三四五六七八九十百千万]+)\s*"
    r"(?P<prequalifier>多|出头)?\s*"
    r"(?P<currency>人民币|块钱|元|块)\s*"
    r"(?P<qualifier>以内|以下|不超过|至多|最多|之内|及以内|左右|上下|"
    r"大约|约|多|出头)?"
)


class RubricGenerationError(RuntimeError):
    """DeepSeek 调用或候选 Rubric 无法满足生成契约。"""


@dataclass(frozen=True)
class RubricTask:
    """Rubric 生成只允许看到的公开任务字段。"""

    task_id: int
    instruction: str

    def __post_init__(self) -> None:
        if isinstance(self.task_id, bool) or not isinstance(self.task_id, int):
            raise TypeError("task_id must be an integer")
        if self.task_id < 0:
            raise ValueError("task_id must be non-negative")
        if not str(self.instruction).strip():
            raise ValueError("instruction must be non-empty")

    @property
    def instruction_sha256(self) -> str:
        return hashlib.sha256(self.instruction.encode("utf-8")).hexdigest()


class DeepSeekRubricClient:
    """仅用于 Rubric JSON 的 OpenAI-compatible DeepSeek 客户端。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 4096,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        response_format_json: bool = True,
        transport: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        if not str(model).strip() or not str(base_url).strip() or not str(api_key):
            raise ValueError("model、base_url 和 api_key 均不能为空")
        if max_tokens < 1 or timeout_s <= 0 or max_retries < 0:
            raise ValueError("max_tokens/timeout_s/max_retries 配置不合法")
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self._api_key = str(api_key)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.response_format_json = bool(response_format_json)
        self.transport = transport

    def complete_json(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """请求一次严格 JSON，并返回不含凭证的响应与调用元数据。"""

        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise TypeError("messages must be an array")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [deepcopy(dict(message)) for message in messages],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
        }
        if self.model.casefold().startswith("deepseek-v4"):
            payload["thinking"] = {"type": "disabled"}
        if self.response_format_json:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "wlx-rubric-generator/1",
        }
        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        retries: list[dict[str, Any]] = []
        response: Mapping[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.transport is not None:
                    candidate = self.transport(
                        url,
                        deepcopy(payload),
                        deepcopy(headers),
                        self.timeout_s,
                    )
                else:
                    request = Request(
                        url,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout_s) as raw:
                        candidate = json.loads(raw.read().decode("utf-8"))
                if not isinstance(candidate, Mapping):
                    raise RubricGenerationError("DeepSeek 响应必须是 JSON 对象")
                response = candidate
                break
            except HTTPError as exc:
                if exc.code not in RETRYABLE_HTTP_STATUSES or attempt >= self.max_retries:
                    raise RubricGenerationError(
                        f"DeepSeek HTTP 状态码 {exc.code}"
                    ) from exc
                delay = max(float(attempt + 1), _retry_after_seconds(exc) or 0.0)
                retries.append({"category": "http", "status": exc.code, "wait": delay})
                time.sleep(delay)
            except (RemoteDisconnected, TimeoutError, URLError) as exc:
                if attempt >= self.max_retries:
                    raise RubricGenerationError("DeepSeek 连接失败并已用完重试") from exc
                delay = float(attempt + 1)
                retries.append(
                    {"category": "connection", "error_type": type(exc).__name__, "wait": delay}
                )
                time.sleep(delay)
        if response is None:  # pragma: no cover - 循环的成功或异常保证不会到这里
            raise RubricGenerationError("DeepSeek 请求没有得到响应")
        choices = response.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise RubricGenerationError("DeepSeek 响应缺少 choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise RubricGenerationError("DeepSeek choices[0].message.content 为空")
        result = _decode_json_content(content)
        usage = response.get("usage")
        return {
            "result": result,
            "metadata": {
                "provider_request_id": response.get("id"),
                "provider_model": response.get("model") or self.model,
                "requested_model": self.model,
                "attempts": len(retries) + 1,
                "retries": retries,
                "latency_seconds": time.monotonic() - started,
                "usage": deepcopy(dict(usage)) if isinstance(usage, Mapping) else {},
            },
        }


def extract_rule_signals(instruction: str) -> dict[str, Any]:
    """提取高精度数值和语气信号；规则不尝试理解所有语义要求。"""

    text = str(instruction).strip()
    if not text:
        raise ValueError("instruction must be non-empty")
    numeric_spans = []
    budget_facts: list[dict[str, Any]] = []
    occupied: set[tuple[int, int]] = set()
    for match in _ARABIC_CURRENCY_RANGE.finditer(text):
        minimum = _canonical_number(match.group("minimum"))
        maximum = _canonical_number(match.group("maximum"))
        if minimum > maximum:
            minimum, maximum = maximum, minimum
        fact = _budget_fact(
            {
                "text": match.group(0),
                "value": {"min": minimum, "max": maximum},
                "unit": match.group("maximum_unit"),
                "qualifier": None,
                "start": match.start(),
                "end": match.end(),
            },
            operator="between",
            priority="hard",
        )
        budget_facts.append(fact)
        occupied.add((match.start(), match.end()))
    for match in _NUMBER_PATTERN.finditer(text):
        number = _canonical_number(match.group("number"))
        item = {
            "text": match.group(0),
            "value": number,
            "unit": match.group("unit"),
            "qualifier": match.group("qualifier"),
            "start": match.start(),
            "end": match.end(),
        }
        numeric_spans.append(item)
        unit = str(match.group("unit")).casefold()
        if unit in {"人民币", "块钱", "元", "块"} and not _span_overlaps(
            match.start(), match.end(), occupied
        ):
            qualifier = str(match.group("qualifier") or "")
            operator, priority = _budget_operator_priority(
                text,
                start=match.start(),
                end=match.end(),
                qualifier=qualifier,
            )
            budget_facts.append(
                _budget_fact(item, operator=operator, priority=priority)
            )
            occupied.add((match.start(), match.end()))
    for match in _ARABIC_BUDGET_CONTEXT.finditer(text):
        magnitude = {None: 1, "": 1, "千": 1000, "万": 10000}[
            match.group("magnitude")
        ]
        value = float(match.group("number")) * magnitude
        value = int(value) if value.is_integer() else value
        fact = _budget_fact_from_match(match, value=value, source_text=text)
        if not _overlaps_existing(fact, occupied):
            budget_facts.append(fact)
            occupied.add((fact["start"], fact["end"]))
    for match in _CHINESE_CURRENCY_BUDGET.finditer(text):
        value = _chinese_number(match.group("number"))
        if value is None:
            continue
        fact = _budget_fact_from_match(match, value=value, source_text=text)
        if not _overlaps_existing(fact, occupied):
            budget_facts.append(fact)
            occupied.add((fact["start"], fact["end"]))
    budget_facts.sort(key=lambda item: (item["start"], item["end"]))
    return {
        "rules_version": RUBRIC_RULES_VERSION,
        "numeric_spans": numeric_spans,
        "budget_facts": budget_facts,
        "hard_markers": _marker_spans(text, HARD_MARKERS),
        "soft_markers": _marker_spans(text, SOFT_MARKERS),
        "negative_markers": _marker_spans(text, NEGATIVE_MARKERS),
    }


def _rubric_system_prompt() -> str:
    return f"""你负责为购物 Agent 评测提取用户需求 Rubric。

严格规则：
1. 只能使用 user_instruction 中明确表达的需求，
   禁止补充常识、目标商品属性或隐藏答案。
2. 将复合需求拆成原子要求；每项必须能单独判断。
3. query_evidence 必须逐字复制 user_instruction 中一个非空、连续的原文片段。
4. priority 只能是 hard 或 soft。明确品类、品牌、型号、功能、规格、数量、
   排除条件通常是 hard；含软化词的表达允许妥协，应标为 soft。
5. type 只能是：{', '.join(sorted(RUBRIC_TYPES))}。
6. verifier.kind 只能是 deterministic 或 semantic。明确数值、类别、品牌、型号、
   选项、数量用 deterministic；需要主观语义理解的体验和适用性用 semantic。
7. deterministic verifier 使用 field、operator、value、unit；
   operator 只能是 {', '.join(sorted(_PROMPT_DETERMINISTIC_OPERATORS))}。
8. semantic verifier 使用 field、operator="semantic_entailment"、criterion。
9. 规则预提取的明确预算必须保留，数值与比较方向不得修改。
10. 不要把用户背景本身当成商品要求，除非它明确限定了商品适用性。

只输出 JSON 对象，格式为：
{{
  "items": [
    {{
      "requirement": "规范化后的原子要求",
      "type": "枚举值",
      "priority": "hard或soft",
      "query_evidence": "用户原文连续片段",
      "verifier": {{
        "kind": "deterministic或semantic", "field": "...", "operator": "...",
        "value": "...", "unit": "...", "criterion": "..."
      }}
    }}
  ],
  "review_required": false,
  "review_reasons": []
}}

如果原文存在真正歧义或互相冲突，仍提取可确认的要求，
并设置 review_required=true，写出简短原因。
Prompt 版本：{RUBRIC_PROMPT_VERSION}"""


RUBRIC_SYSTEM_PROMPT = _rubric_system_prompt()
RUBRIC_SYSTEM_PROMPT_SHA256 = hashlib.sha256(
    RUBRIC_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()


def build_rubric_messages(
    task: RubricTask,
    rule_signals: Mapping[str, Any],
) -> list[dict[str, str]]:
    """构造只含用户需求和规则信号的无目标泄漏 Prompt。"""

    user = json.dumps(
        {
            "task_id": task.task_id,
            "user_instruction": task.instruction,
            "deterministic_rule_signals": deepcopy(dict(rule_signals)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return [
        {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_rubric_candidate(
    task: RubricTask,
    *,
    client: DeepSeekRubricClient,
    rubric_set_version: str,
) -> dict[str, Any]:
    """规则 + DeepSeek + 程序审计，返回可冻结候选或复核候选。"""

    signals = extract_rule_signals(task.instruction)
    response = client.complete_json(build_rubric_messages(task, signals))
    return audit_rubric_candidate_result(
        task,
        raw_result=response["result"],
        llm_metadata=response.get("metadata") or {},
        rubric_set_version=rubric_set_version,
    )


def audit_rubric_candidate_result(
    task: RubricTask,
    *,
    raw_result: object,
    llm_metadata: Mapping[str, Any],
    rubric_set_version: str,
) -> dict[str, Any]:
    """不调用 API，按当前规则重新规范化并审计一份 LLM 原始结果。"""

    signals = extract_rule_signals(task.instruction)
    candidate = raw_result
    if not isinstance(candidate, Mapping):
        raise RubricGenerationError("Rubric JSON 根必须是对象")
    items = candidate.get("items")
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or not items:
        raise RubricGenerationError("Rubric items 必须是非空数组")
    if len(items) > 30:
        raise RubricGenerationError("Rubric items 超过 30，疑似没有原子化或发生幻觉")

    normalized_items, audit_reasons = normalize_generated_rubric_items(
        items,
        instruction=task.instruction,
    )
    audit_reasons.extend(_budget_audit_reasons(normalized_items, signals))
    llm_review_reasons = candidate.get("review_reasons") or []
    if isinstance(llm_review_reasons, Sequence) and not isinstance(
        llm_review_reasons, (str, bytes)
    ):
        audit_reasons.extend(str(item).strip() for item in llm_review_reasons if str(item).strip())
    elif llm_review_reasons:
        audit_reasons.append("llm_review_reasons_not_array")
    if candidate.get("review_required") is True and not llm_review_reasons:
        audit_reasons.append("llm_marked_review_without_reason")
    audit_reasons = list(dict.fromkeys(audit_reasons))

    try:
        bundle = freeze_rubric_bundle(
            task_id=task.task_id,
            instruction=task.instruction,
            items=normalized_items,
            rubric_set_version=rubric_set_version,
        )
    except RubricContractError as exc:
        raise RubricGenerationError(f"候选 Rubric 未通过冻结契约：{exc}") from exc
    return {
        "generator_version": RUBRIC_GENERATOR_VERSION,
        "prompt_version": RUBRIC_PROMPT_VERSION,
        "prompt_sha256": RUBRIC_SYSTEM_PROMPT_SHA256,
        "task_id": task.task_id,
        "instruction_sha256": task.instruction_sha256,
        "rule_signals": signals,
        "rubric": bundle,
        "review_required": bool(audit_reasons),
        "review_reasons": audit_reasons,
        "llm_raw_result": deepcopy(dict(candidate)),
        "llm_raw_result_sha256": hashlib.sha256(
            json.dumps(
                dict(candidate),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "llm_metadata": deepcopy(dict(llm_metadata)),
    }


def normalize_generated_rubric_items(
    items: Sequence[object],
    *,
    instruction: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """对 DeepSeek 或人工替换项应用同一套严格结构与原文证据校验。"""

    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or not items:
        raise RubricGenerationError("Rubric items 必须是非空数组")
    normalized_items = []
    audit_reasons: list[str] = []
    for index, item in enumerate(items, start=1):
        normalized, reasons = _normalize_candidate_item(
            item,
            instruction=instruction,
            requirement_id=f"R{index:03d}",
        )
        normalized_items.append(normalized)
        audit_reasons.extend(reasons)
    audit_reasons.extend(_duplicate_reasons(normalized_items))
    return normalized_items, audit_reasons


def _normalize_candidate_item(
    item: object,
    *,
    instruction: str,
    requirement_id: str,
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(item, Mapping):
        raise RubricGenerationError(f"{requirement_id} 必须是对象")
    requirement = str(item.get("requirement") or "").strip()
    item_type = str(item.get("type") or "").strip()
    priority = str(item.get("priority") or "").strip()
    evidence = str(item.get("query_evidence") or "").strip()
    verifier = item.get("verifier")
    if not requirement or not evidence:
        raise RubricGenerationError(f"{requirement_id} 缺少 requirement/query_evidence")
    reasons: list[str] = []
    if _normalize_text(evidence) not in _normalize_text(instruction):
        # 不猜测 LLM 想引用哪一段：用完整原文保住可追溯性，并强制人工复核。
        evidence = instruction
        reasons.append(f"{requirement_id}:query_evidence_not_exact")
    if item_type not in RUBRIC_TYPES or priority not in RUBRIC_PRIORITIES:
        raise RubricGenerationError(f"{requirement_id} type/priority 不在允许枚举中")
    if not isinstance(verifier, Mapping):
        raise RubricGenerationError(f"{requirement_id}.verifier 必须是对象")
    verifier = deepcopy(dict(verifier))
    kind = str(verifier.get("kind") or "").strip()
    operator = str(verifier.get("operator") or "").strip()
    if kind not in VERIFIER_KINDS:
        raise RubricGenerationError(f"{requirement_id}.verifier.kind 不受支持")
    if kind == "deterministic":
        normalized_operator = _OPERATOR_ALIASES.get(operator.casefold(), operator)
        if normalized_operator != operator:
            verifier["operator"] = normalized_operator
            operator = normalized_operator
        if operator not in DETERMINISTIC_OPERATORS:
            verifier = _semantic_fallback(verifier, requirement=requirement)
            reasons.append(f"{requirement_id}:unsupported_operator:{operator or 'empty'}")
        elif "value" not in verifier:
            verifier = _semantic_fallback(verifier, requirement=requirement)
            reasons.append(f"{requirement_id}:deterministic_value_missing")
        elif operator in {"lt", "lte", "gt", "gte", "approx", "between"}:
            normalized_value = _normalize_numeric_value(verifier.get("value"))
            if normalized_value is None:
                verifier = _semantic_fallback(verifier, requirement=requirement)
                reasons.append(f"{requirement_id}:numeric_value_unusable")
            else:
                value, is_range = normalized_value
                if is_range:
                    verifier["operator"] = "between"
                verifier["value"] = value
    else:
        if operator != "semantic_entailment" or not str(verifier.get("criterion") or "").strip():
            raise RubricGenerationError(f"{requirement_id} 语义 verifier 契约不完整")
    if not str(verifier.get("field") or "").strip():
        raise RubricGenerationError(f"{requirement_id}.verifier.field 不能为空")
    if len(evidence) < 2:
        reasons.append(f"{requirement_id}:query_evidence_too_short")
    return (
        {
            "requirement_id": requirement_id,
            "requirement": requirement,
            "type": item_type,
            "priority": priority,
            "query_evidence": evidence,
            "verifier": verifier,
        },
        reasons,
    )


def _budget_audit_reasons(
    items: Sequence[Mapping[str, Any]],
    signals: Mapping[str, Any],
) -> list[str]:
    reasons = []
    budget_items = [item for item in items if item.get("type") == "budget"]
    for index, fact in enumerate(signals.get("budget_facts") or [], start=1):
        matched = []
        for item in budget_items:
            verifier = item.get("verifier") or {}
            evidence = str(item.get("query_evidence") or "")
            if str(fact.get("text") or "") not in evidence and evidence not in str(
                fact.get("text") or ""
            ):
                continue
            if (
                verifier.get("kind") == "deterministic"
                and verifier.get("operator") == fact.get("operator")
                and _same_budget_value(verifier.get("value"), fact.get("value"))
            ):
                matched.append(item)
        if not matched:
            reasons.append(f"budget_rule_conflict:{index}")
    return reasons


def _duplicate_reasons(items: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: dict[tuple[str, str], str] = {}
    reasons = []
    for item in items:
        key = (
            str(item.get("type") or ""),
            _normalize_text(item.get("requirement")),
        )
        requirement_id = str(item.get("requirement_id") or "")
        if key in seen:
            reasons.append(f"duplicate_requirements:{seen[key]}:{requirement_id}")
        else:
            seen[key] = requirement_id
    return reasons


def _decode_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RubricGenerationError("DeepSeek content 不是严格 JSON") from exc
    if not isinstance(value, dict):
        raise RubricGenerationError("DeepSeek content JSON 根必须是对象")
    return value


def _marker_spans(text: str, markers: Sequence[str]) -> list[dict[str, Any]]:
    result = []
    for marker in markers:
        for match in re.finditer(re.escape(marker), text):
            result.append({"text": marker, "start": match.start(), "end": match.end()})
    return sorted(result, key=lambda item: (item["start"], item["end"]))


def _budget_fact_from_match(
    match: re.Match[str],
    *,
    value: int | float,
    source_text: str,
) -> dict[str, Any]:
    qualifier = str(
        match.groupdict().get("precompare")
        or
        match.groupdict().get("qualifier")
        or match.groupdict().get("prequalifier")
        or ""
    )
    operator, priority = _budget_operator_priority(
        source_text,
        start=match.start(),
        end=match.end(),
        qualifier=qualifier,
    )
    return _budget_fact(
        {
            "text": match.group(0),
            "value": value,
            "unit": match.groupdict().get("currency") or "元",
            "qualifier": qualifier or None,
            "start": match.start(),
            "end": match.end(),
        },
        operator=operator,
        priority=priority,
    )


def _budget_fact(
    item: Mapping[str, Any],
    *,
    operator: str,
    priority: str,
) -> dict[str, Any]:
    return {
        **deepcopy(dict(item)),
        "operator": operator,
        "priority": priority,
        "currency": "CNY",
    }


def _budget_operator_priority(
    text: str,
    *,
    start: int,
    end: int,
    qualifier: str,
) -> tuple[str, str]:
    context = text[max(0, start - 8) : min(len(text), end + 8)]
    if qualifier in {
        "以内",
        "以下",
        "不超过",
        "至多",
        "最多",
        "低于",
        "小于",
        "之内",
        "及以内",
    }:
        return "lte", "hard"
    if qualifier in {"左右", "上下", "大约", "约", "多", "出头"}:
        return "approx", "soft"
    if any(
        marker in context
        for marker in (
            "不超过",
            "不得超过",
            "别超过",
            "不要超",
            "不超",
            "至多",
            "最多",
            "以内",
            "以下",
            "之内",
            "低于",
            "小于",
            "不到",
            "控在",
        )
    ):
        priority = "soft" if any(marker in context for marker in SOFT_MARKERS) else "hard"
        return "lte", priority
    return "approx", "soft"


def _overlaps_existing(
    fact: Mapping[str, Any],
    occupied: set[tuple[int, int]],
) -> bool:
    start = int(fact["start"])
    end = int(fact["end"])
    return any(
        start < previous_end and previous_start < end
        for previous_start, previous_end in occupied
    )


def _span_overlaps(start: int, end: int, occupied: set[tuple[int, int]]) -> bool:
    return any(
        start < previous_end and previous_start < end
        for previous_start, previous_end in occupied
    )


def _canonical_number(value: object) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _normalize_numeric_value(
    value: object,
) -> tuple[int | float | dict[str, int | float], bool] | None:
    """接受数字、数字字符串或常见区间表达，输出无 NaN 的结构。"""

    if _finite_number(value):
        return _canonical_number(value), False
    if isinstance(value, Mapping):
        minimum = value.get("min", value.get("minimum"))
        maximum = value.get("max", value.get("maximum"))
        if _finite_number(minimum) and _finite_number(maximum):
            lower = _canonical_number(minimum)
            upper = _canonical_number(maximum)
            if lower > upper:
                lower, upper = upper, lower
            return {"min": lower, "max": upper}, True
        return None
    if not isinstance(value, str):
        return None
    numbers = re.findall(r"(?<!\d)\d+(?:\.\d+)?", value)
    if len(numbers) == 1:
        return _canonical_number(numbers[0]), False
    if len(numbers) == 2 and re.search(r"到|至|[-~～—–]", value):
        lower = _canonical_number(numbers[0])
        upper = _canonical_number(numbers[1])
        if lower > upper:
            lower, upper = upper, lower
        return {"min": lower, "max": upper}, True
    return None


def _semantic_fallback(
    verifier: Mapping[str, Any],
    *,
    requirement: str,
) -> dict[str, Any]:
    return {
        "kind": "semantic",
        "field": str(verifier.get("field") or "product").strip() or "product",
        "operator": "semantic_entailment",
        "criterion": requirement,
    }


def _chinese_number(text: str) -> int | None:
    """解析评测需求中常见的万以内中文整数。"""

    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    if not text or any(char not in digits and char not in units for char in text):
        return None
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in digits:
            number = digits[char]
            continue
        unit = units[char]
        if unit == 10000:
            section += number
            total += (section or 1) * unit
            section = 0
            number = 0
        else:
            section += (number or 1) * unit
            number = 0
    return total + section + number


def _retry_after_seconds(error: HTTPError) -> float | None:
    value = error.headers.get("Retry-After") if error.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _same_number(left: object, right: object) -> bool:
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-9
    )


def _same_budget_value(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return _same_number(left.get("min"), right.get("min")) and _same_number(
            left.get("max"), right.get("max")
        )
    return _same_number(left, right)


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


__all__ = [
    "DETERMINISTIC_OPERATORS",
    "DeepSeekRubricClient",
    "RubricGenerationError",
    "RubricTask",
    "VERIFIER_KINDS",
    "RUBRIC_GENERATOR_VERSION",
    "RUBRIC_PROMPT_VERSION",
    "RUBRIC_RULES_VERSION",
    "RUBRIC_SYSTEM_PROMPT",
    "RUBRIC_SYSTEM_PROMPT_SHA256",
    "audit_rubric_candidate_result",
    "build_rubric_messages",
    "extract_rule_signals",
    "generate_rubric_candidate",
    "normalize_generated_rubric_items",
]
