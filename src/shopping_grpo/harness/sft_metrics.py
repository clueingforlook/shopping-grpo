"""计算 SFT 采样过程中的 Token、步骤和搜索行为指标。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from shopping_grpo.harness.sft_outcomes import trajectory_mapping


class TrainingTokenCounter(Protocol):
    """约定最终训练 Token 计数器要返回总长度、Loss 长度和版本信息。"""

    def __call__(self, row: Mapping[str, Any]) -> tuple[int, int, Mapping[str, Any]]:
        """使用真实训练 tokenizer 和 chat template 计算一条 SFT 数据。"""

        ...


class TrainingSequenceUnrenderable(ValueError):
    """表示单条样本无法被最终训练模板无损渲染，可按行拒绝。"""


@dataclass(frozen=True)
class TrainingLength:
    """保存最终 SFT 序列的真实长度，以及它是否能放进训练上下文。"""

    total_tokens: int
    loss_tokens: int
    within_limit: bool
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    chat_template_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """把训练长度结果转成适合附加到 SFT 行的元数据。"""

        return {
            "sft_total_tokens": self.total_tokens,
            "sft_loss_tokens": self.loss_tokens,
            "sft_within_context_limit": self.within_limit,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_version": self.chat_template_version,
        }


def aggregate_token_metrics(
    trajectory: Mapping[str, Any] | object,
    *,
    request_usage_history: Iterable[Mapping[str, Any]] = (),
    raw_trajectory_counter: Callable[[Mapping[str, Any]], int] | None = None,
) -> dict[str, Any]:
    """汇总每轮 API 用量、上下文峰值和 Observation Projection 的压缩量。"""

    row = trajectory_mapping(trajectory)
    request_usages = [_normalise_usage(item) for item in request_usage_history]
    request_usages = [item for item in request_usages if item is not None]
    context_inputs = [
        int(item.get("input_tokens", 0) or 0)
        for item in row.get("context_turn_tokens") or []
        if isinstance(item, Mapping)
    ]
    if request_usages:
        prompt_sum = sum(item["input_tokens"] for item in request_usages)
        completion_sum = sum(item["output_tokens"] for item in request_usages)
        maximum = max(item["request_total_tokens"] for item in request_usages)
        source = "api_usage"
    else:
        prompt_sum = sum(context_inputs)
        completion_sum = 0
        maximum = max(context_inputs, default=0)
        source = "local_context_counter" if context_inputs else "unavailable"

    projection_rows = [
        step.get("projection")
        for step in row.get("steps") or []
        if isinstance(step, Mapping) and isinstance(step.get("projection"), Mapping)
    ]
    raw_projection_tokens = sum(int(item.get("raw_tokens", 0) or 0) for item in projection_rows)
    visible_projection_tokens = sum(
        int(item.get("visible_tokens", 0) or 0) for item in projection_rows
    )
    quality = trajectory_cost_metrics(row)
    raw_tokens = (
        int(raw_trajectory_counter(row))
        if raw_trajectory_counter is not None
        else None
    )
    return {
        "request_usage": deepcopy(request_usages),
        "request_usage_source": source,
        "prompt_tokens_sum": prompt_sum,
        "completion_tokens_sum": completion_sum,
        "max_request_tokens": maximum,
        "raw_trajectory_tokens": raw_tokens,
        "tokens_before_projection": raw_projection_tokens,
        "tokens_after_projection": visible_projection_tokens,
        **quality,
    }


def measure_training_length(
    row: Mapping[str, Any],
    counter: TrainingTokenCounter,
    *,
    context_limit: int = 24_576,
) -> TrainingLength:
    """用最终训练模型的计数器检查整条 SFT 数据，而不是拿字符数猜长度。"""

    if context_limit < 1:
        raise ValueError("context_limit 必须至少为 1")
    total, loss, metadata = counter(row)
    total = int(total)
    loss = int(loss)
    if total < 1 or loss < 0 or loss > total:
        raise ValueError("训练 Token 计数器返回了不合理的长度")
    metadata = dict(metadata or {})
    return TrainingLength(
        total_tokens=total,
        loss_tokens=loss,
        within_limit=total <= context_limit,
        tokenizer_name=_optional_text(metadata.get("tokenizer_name")),
        tokenizer_revision=_optional_text(metadata.get("tokenizer_revision")),
        chat_template_version=_optional_text(metadata.get("chat_template_version")),
    )


def trajectory_cost_metrics(trajectory: Mapping[str, Any] | object) -> dict[str, Any]:
    """统计环境 Step、有效探索步、唯一搜索词和打开过的不同商品数量。"""

    row = trajectory_mapping(trajectory)
    steps = [item for item in row.get("steps") or [] if isinstance(item, Mapping)]
    searches: set[str] = set()
    products: set[str] = set()
    seen_evidence: set[str] = set()
    productive_steps = 0
    for step in steps:
        tool_name, arguments = _tool_name_and_arguments(step)
        if tool_name == "search_products":
            query = normalise_search_query(arguments.get("query", ""))
            if query:
                searches.add(query)
        if tool_name == "open_product":
            asin = str(arguments.get("asin") or "").strip()
            if asin:
                products.add(asin)
        if _is_productive_step(step, tool_name, arguments, seen_evidence):
            productive_steps += 1
    assistant_turns = sum(
        1
        for message in row.get("messages") or []
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    )
    return {
        "num_env_steps": len(steps),
        "num_assistant_turns": assistant_turns,
        "num_messages": len(row.get("messages") or []),
        "productive_steps": productive_steps,
        "trajectory_length_bucket": productive_step_bucket(productive_steps),
        "unique_search_queries": len(searches),
        "opened_product_count": len(products),
    }


def productive_step_bucket(productive_steps: int) -> str:
    """按已确定的 3～10、11～20、21～35 规则给成功轨迹分长度桶。"""

    steps = int(productive_steps)
    if steps < 3:
        return "too_short_review"
    if steps <= 10:
        return "short"
    if steps <= 20:
        return "medium"
    if steps <= 35:
        return "long"
    return "over_limit"


def normalise_search_query(query: object) -> str:
    """统一大小写、全半角、标点和空格，让同一个搜索词只统计一次。"""

    text = unicodedata.normalize("NFKC", str(query or "")).casefold()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _normalise_usage(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """兼容常见 API Usage 字段名，并统一成 input、output、total 三个数字。"""

    if not isinstance(value, Mapping):
        return None
    input_tokens = value.get("input_tokens", value.get("prompt_tokens"))
    output_tokens = value.get("output_tokens", value.get("completion_tokens"))
    total_tokens = value.get("request_total_tokens", value.get("total_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)
    total_tokens = int(total_tokens) if total_tokens is not None else input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_total_tokens": total_tokens,
        "source": str(value.get("source") or "api_usage"),
    }


def _tool_name_and_arguments(step: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """同时兼容 Core 和旧格式，从一步轨迹里读出工具名与参数。"""

    call = step.get("tool_call")
    if isinstance(call, Mapping) and isinstance(call.get("name"), str):
        arguments = call.get("arguments")
        return str(call["name"]), dict(arguments) if isinstance(arguments, Mapping) else {}
    if isinstance(call, Mapping):
        function = call.get("function")
        if isinstance(function, Mapping):
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            return str(function.get("name") or ""), (
                dict(arguments) if isinstance(arguments, Mapping) else {}
            )
    return str(step.get("tool_name") or ""), dict(step.get("parameters") or {})


def _is_productive_step(
    step: Mapping[str, Any],
    tool_name: str,
    arguments: Mapping[str, Any],
    seen_evidence: set[str],
) -> bool:
    """判断一步是否真正带来新页面、新证据、新规格选择或终局结果。"""

    if not step.get("env_action") or tool_name == "think" or step.get("error"):
        return False
    observation = str(step.get("observation") or "")
    evidence = json.dumps(
        {
            "tool": tool_name,
            "arguments": dict(arguments),
            "observation_sha": hashlib.sha256(observation.encode("utf-8")).hexdigest(),
            "done": bool(step.get("done")),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if evidence in seen_evidence:
        return False
    seen_evidence.add(evidence)
    return bool(observation.strip() or step.get("done"))


def _optional_text(value: object) -> str | None:
    """把可选版本字段转成字符串，空值继续保持为空。"""

    return str(value) if value is not None else None


__all__ = [
    "TrainingLength",
    "TrainingSequenceUnrenderable",
    "TrainingTokenCounter",
    "aggregate_token_metrics",
    "measure_training_length",
    "normalise_search_query",
    "productive_step_bucket",
    "trajectory_cost_metrics",
]
