"""提供 自己的 OpenAI-compatible Teacher 策略，不调用原项目 rollout Harness。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from copy import deepcopy
from http.client import RemoteDisconnected
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from shopping_grpo.environment.context import ContextBudgetError, compact_chat_messages
from shopping_grpo.environment.projection import project_observation

from shopping_grpo.harness.config import (
    ConfigValidationError,
    ContextPolicy,
    HarnessConfig,
    ObservationPolicy,
)
from shopping_grpo.harness.contracts import ModelRequest


logger = logging.getLogger(__name__)


class TeacherApiError(RuntimeError):
    """表示 Teacher 接口返回了无法使用的状态码或响应结构。"""

    pass


class TeacherOutputTruncatedError(TeacherApiError):
    """表示 Teacher 因输出上限停止，尚未生成可执行的完整回复。"""

    pass


class OpenAICompatibleTeacherPolicy:
    """直接调用兼容 Chat Completions 的 Teacher，并保存每一轮 API Token Usage。"""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        context_policy: ContextPolicy,
        observation_policy: ObservationPolicy,
        count_chat_tokens: Callable[[object, object], int],
        count_text_tokens: Callable[[object], int],
        temperature: float = 0.2,
        top_p: float = 1.0,
        timeout_s: float = 180.0,
        max_retries: int = 2,
        thinking_mode: str | None = None,
        reasoning_effort: str = "high",
        extra_body: Mapping[str, Any] | None = None,
        transport: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        """保存接口和长度配置；API Key 只留在内存中，不会进入轨迹或元数据。"""

        if not model:
            raise ValueError("model 不能为空")
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        if not callable(count_chat_tokens) or not callable(count_text_tokens):
            raise TypeError("必须提供 Teacher tokenizer 的聊天和文本计数函数")
        if timeout_s <= 0 or max_retries < 0:
            raise ValueError("timeout_s 必须大于 0，max_retries 不能小于 0")
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ValueError("thinking_mode 只能是 enabled、disabled 或 None")
        if reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort 只能是 high 或 max")
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self._api_key = str(api_key)
        self.context_policy = context_policy
        self.observation_policy = observation_policy
        self.count_chat_tokens = count_chat_tokens
        self.count_text_tokens = count_text_tokens
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.extra_body = deepcopy(dict(extra_body or {}))
        self.transport = transport
        self.last_context_event: dict[str, Any] | None = None
        self.last_context_tokens: int | None = None
        self.request_usage_history: list[dict[str, Any]] = []

    def validate_harness_config(self, config: HarnessConfig) -> None:
        """确认 Runner 与策略使用完全相同的上下文和 Observation Projection 规则。"""

        if config.context != self.context_policy:
            raise ConfigValidationError("Teacher 的 ContextPolicy 与 HarnessConfig 不一致")
        if config.observation != self.observation_policy:
            raise ConfigValidationError("Teacher 的 ObservationPolicy 与 HarnessConfig 不一致")

    async def generate(self, request: ModelRequest) -> dict[str, Any]:
        """检查上下文预算后在线程中请求 Teacher，避免阻塞其他并发任务。"""

        messages, input_tokens = self._bounded_messages(request.messages, request.tools)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": list(request.tools),
            "max_tokens": request.max_output_tokens
            or self.context_policy.generation_reserve_tokens,
        }
        if self.thinking_mode is None:
            payload.update({"temperature": self.temperature, "top_p": self.top_p})
        elif self.thinking_mode == "enabled":
            payload.update(
                {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": self.reasoning_effort,
                }
            )
        else:
            payload.update(
                {
                    "thinking": {"type": "disabled"},
                    "tool_choice": "auto",
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
            )
        if self.thinking_mode is None:
            payload["tool_choice"] = "auto"
        payload.update(deepcopy(self.extra_body))
        response = await asyncio.to_thread(self._request_with_retries, payload)
        message = _response_message(response)
        usage = _response_usage(response, input_tokens, message, self.count_text_tokens)
        self.request_usage_history.append(usage)
        if _response_finish_reason(response) == "length":
            raise TeacherOutputTruncatedError(
                "Teacher 输出达到 max_tokens，当前尝试必须按技术失败重试"
            )
        return message

    async def project_observation(
        self,
        tool_name: str,
        observation: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """按页面结构压缩长 Observation，并保证商品 ID、规格和按钮仍可操作。"""

        visible, metadata = await asyncio.to_thread(
            project_observation,
            tool_name,
            observation,
            parameters=dict(parameters or {}),
            count_tokens=self.count_text_tokens,
            token_budget=self.observation_policy.token_budget,
            detail_token_budget=self.observation_policy.detail_token_budget,
            generic_token_budget=self.observation_policy.generic_token_budget,
            search_top_k=self.observation_policy.search_top_k,
        )
        return visible, metadata.to_dict()

    def _bounded_messages(
        self,
        messages: object,
        tools: object,
    ) -> tuple[list[dict[str, Any]], int]:
        """计算本轮输入长度；默认超限就停止，只有明确开启时才压缩旧交互。"""

        copied_messages = _normalise_teacher_messages(messages)
        copied_tools = [deepcopy(dict(item)) for item in tools]
        original_tokens = int(self.count_chat_tokens(copied_messages, copied_tools))
        self.last_context_tokens = original_tokens
        self.last_context_event = None
        maximum = (
            self.context_policy.window_tokens
            - self.context_policy.generation_reserve_tokens
            - self.context_policy.safety_margin_tokens
        )
        input_budget = self.context_policy.input_budget_tokens or maximum
        if original_tokens <= input_budget:
            return copied_messages, original_tokens
        if not self.context_policy.compaction_enabled:
            raise ContextBudgetError(
                f"当前输入 {original_tokens} Token，超过允许的 {input_budget} Token"
            )
        compacted, stats = compact_chat_messages(
            copied_messages,
            copied_tools,
            count_tokens=self.count_chat_tokens,
            max_input_tokens=input_budget,
        )
        self.last_context_event = stats.to_dict()
        return compacted, original_tokens

    def _request_with_retries(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """对连接失败做有限重试；已经拿到业务响应后不会盲目重复请求。"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "wlx-sft-pipeline/1",
        }
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            try:
                if self.transport is not None:
                    response = self.transport(url, dict(payload), headers, self.timeout_s)
                else:
                    request = Request(
                        url,
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                        method="POST",
                    )
                    with urlopen(request, timeout=self.timeout_s) as raw:
                        response = json.loads(raw.read().decode("utf-8"))
                if not isinstance(response, Mapping):
                    raise TeacherApiError("Teacher 响应必须是 JSON 对象")
                return response
            except HTTPError as exc:
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    backoff_s = attempt + 1
                    logger.warning(
                        "Teacher request retry: category=http status=%s "
                        "next_attempt=%s/%s backoff_s=%s",
                        exc.code,
                        attempt + 2,
                        self.max_retries + 1,
                        backoff_s,
                    )
                    time.sleep(backoff_s)
                    continue
                raise TeacherApiError(f"Teacher HTTP 状态码 {exc.code}") from exc
            except (RemoteDisconnected, TimeoutError, URLError) as exc:
                if attempt >= self.max_retries:
                    raise TeacherApiError("Teacher 连接失败并已用完重试次数") from exc
                backoff_s = attempt + 1
                logger.warning(
                    "Teacher request retry: category=connection error_type=%s "
                    "next_attempt=%s/%s backoff_s=%s",
                    type(exc).__name__,
                    attempt + 2,
                    self.max_retries + 1,
                    backoff_s,
                )
                time.sleep(backoff_s)
        raise TeacherApiError("Teacher 请求没有得到响应")


def _response_message(response: Mapping[str, Any]) -> dict[str, Any]:
    """从 Chat Completions 响应中取出第一条 assistant 消息并检查基本结构。"""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TeacherApiError("Teacher 响应缺少 choices")
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
        raise TeacherApiError("Teacher 响应缺少 choices[0].message")
    message = deepcopy(dict(choice["message"]))
    message["role"] = "assistant"
    return message


def _response_finish_reason(response: Mapping[str, Any]) -> str | None:
    """读取第一条回复的停止原因；结构错误仍由 ``_response_message`` 报告。"""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("finish_reason") is None:
        return None
    return str(choice["finish_reason"])


def _response_usage(
    response: Mapping[str, Any],
    local_input_tokens: int,
    message: Mapping[str, Any],
    count_text_tokens: Callable[[object], int],
) -> dict[str, Any]:
    """优先采用 API Usage；接口没返回时用本地 tokenizer 给出可辨认的估算来源。"""

    usage = response.get("usage")
    if isinstance(usage, Mapping):
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "request_total_tokens": total_tokens,
            "source": "api_usage",
        }
    output_tokens = int(count_text_tokens(json.dumps(message, ensure_ascii=False)))
    return {
        "input_tokens": int(local_input_tokens),
        "output_tokens": output_tokens,
        "request_total_tokens": int(local_input_tokens) + output_tokens,
        "source": "local_teacher_tokenizer",
    }


def _normalise_teacher_messages(messages: object) -> list[dict[str, Any]]:
    """只保留 DeepSeek 接受的消息字段，并把工具调用中的空 content 改成空串。"""

    normalised: list[dict[str, Any]] = []
    for raw_message in messages:
        message = dict(raw_message)
        role = message.get("role")
        if role in {"system", "user"}:
            normalised.append(
                {"role": role, "content": _normalise_content(message.get("content"))}
            )
            continue
        if role == "assistant":
            cleaned: dict[str, Any] = {
                "role": "assistant",
                "content": _normalise_content(message.get("content")),
            }
            if message.get("reasoning_content") is not None:
                cleaned["reasoning_content"] = str(message["reasoning_content"])
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                cleaned["tool_calls"] = [
                    _normalise_teacher_tool_call(item) for item in tool_calls
                ]
            normalised.append(cleaned)
            continue
        if role == "tool":
            normalised.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id") or ""),
                    "content": _normalise_content(message.get("content")),
                }
            )
            continue
        raise TeacherApiError(f"Teacher 不支持消息角色：{role!r}")
    return normalised


def _normalise_teacher_tool_call(tool_call: object) -> dict[str, Any]:
    """把历史函数调用收紧成 OpenAI 标准形状，避免附加字段触发接口 400。"""

    if not isinstance(tool_call, Mapping):
        raise TeacherApiError("历史 tool_call 必须是对象")
    function = tool_call.get("function")
    if not isinstance(function, Mapping) or not function.get("name"):
        raise TeacherApiError("历史 tool_call 缺少 function.name")
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return {
        "id": str(tool_call.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(function["name"]),
            "arguments": arguments,
        },
    }


def _normalise_content(content: object) -> str:
    """把 API 不接受的空 content 统一成空字符串，其余内容保持可读文字。"""

    return "" if content is None else str(content)


__all__ = [
    "OpenAICompatibleTeacherPolicy",
    "TeacherApiError",
    "TeacherOutputTruncatedError",
]
