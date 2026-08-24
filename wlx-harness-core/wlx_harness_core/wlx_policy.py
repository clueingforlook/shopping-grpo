"""把现有聊天模型客户端接到 WLX Harness 的统一策略接口上。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Mapping

from shopping_grpo.environment.context import ContextBudgetError, compact_chat_messages

from wlx_harness_core.wlx_config import (
    ConfigValidationError,
    ContextPolicy,
    HarnessConfig,
)


_MISSING = object()


class LegacyChatPolicyAdapter:
    """把原有的 ``complete(messages, tools)`` 模型客户端接入 Harness Core。

    模型密钥仍由原客户端自己管理。每次运行前，这个适配器会核对旧客户端和
    Harness 的配置是否一致；请求太长时，也会按 Core 规定的上限先做压缩，避免
    不同训练阶段各自处理上下文后产生不一致的结果。
    """

    def __init__(self, client: object, *, call_in_thread: bool = True) -> None:
        """保存旧模型客户端，并确认它至少提供最基本的 ``complete`` 方法。

        ``call_in_thread`` 用来决定是否把同步模型调用放进线程，防止它卡住异步任务。
        """

        if not callable(getattr(client, "complete", None)):
            raise TypeError("client must provide complete(messages, tools)")
        self.client = client
        self.call_in_thread = bool(call_in_thread)
        self.last_context_event: dict[str, Any] | None = None
        self.last_context_tokens: int | None = None
        self._context_policy: ContextPolicy | None = None

    def validate_harness_config(self, config: HarnessConfig) -> None:
        """核对旧客户端能否真正执行 Harness 选定的上下文和页面压缩规则。

        这一步不是只检查参数能不能读取，而是要求两边的值完全相同。这样可以避免
        配置看起来已经开启，实际生成时却仍按旧规则运行。
        """

        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be HarnessConfig")
        context = config.context
        client_window = getattr(self.client, "context_window", None)
        if context is None:
            self._context_policy = None
            if client_window is not None:
                raise ConfigValidationError(
                    "legacy client enables context control but HarnessConfig.context is None"
                )
        else:
            _require_equal(self.client, "context_window", context.window_tokens)
            _require_equal(
                self.client,
                "max_tokens",
                context.generation_reserve_tokens,
            )
            _require_equal(
                self.client,
                "context_safety_margin",
                context.safety_margin_tokens,
            )
            _require_equal(
                self.client,
                "context_compaction_enable",
                context.compaction_enabled,
            )
            if not callable(getattr(self.client, "token_counter", None)):
                raise ConfigValidationError(
                    "legacy client must expose token_counter for ContextPolicy"
                )
            if context.preserve_recent_groups != 1:
                raise ConfigValidationError(
                    "legacy message compaction currently supports "
                    "preserve_recent_groups=1 only"
                )
            self._context_policy = context

        observation = config.observation
        client_observation_budget = getattr(
            self.client,
            "observation_token_budget",
            None,
        )
        if observation is None:
            if client_observation_budget is not None:
                raise ConfigValidationError(
                    "legacy client enables observation projection but "
                    "HarnessConfig.observation is None"
                )
        else:
            if not callable(getattr(self.client, "project_observation", None)):
                raise ConfigValidationError(
                    "legacy client cannot enforce HarnessConfig.observation"
                )
            _require_equal(
                self.client,
                "observation_token_budget",
                observation.token_budget,
            )
            _require_equal(
                self.client,
                "observation_detail_token_budget",
                observation.detail_token_budget,
            )
            _require_equal(
                self.client,
                "observation_generic_token_budget",
                observation.generic_token_budget,
            )
            _require_equal(
                self.client,
                "observation_search_top_k",
                observation.search_top_k,
            )

    async def generate(self, request: object) -> dict[str, Any]:
        """根据消息和工具列表调用模型，返回一条统一格式的助手消息。

        调用前会先处理上下文长度，调用后会保存本次压缩和 token 数信息，方便训练
        或评测阶段知道模型实际看到了什么规模的输入。
        """

        messages = deepcopy(_field(request, "messages", []))
        tools = deepcopy(_field(request, "tools", []))
        messages, context_tokens, context_event = self._bounded_messages(
            messages,
            tools,
        )
        if self.call_in_thread:
            assistant = await asyncio.to_thread(self.client.complete, messages, tools)
        else:
            assistant = self.client.complete(messages, tools)
        if not isinstance(assistant, Mapping):
            raise TypeError("policy response must be an assistant message object")
        client_event = getattr(self.client, "last_context_event", None)
        self.last_context_event = deepcopy(context_event or client_event)
        client_tokens = getattr(self.client, "last_context_tokens", None)
        self.last_context_tokens = (
            context_tokens
            if context_tokens is not None
            else (int(client_tokens) if client_tokens is not None else None)
        )
        return deepcopy(dict(assistant))

    def _bounded_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int | None, dict[str, Any] | None]:
        """检查输入是否超出预算，必要时压缩较早的聊天记录。

        最近的交互会被尽量保留，因为模型下一步决策最依赖这些内容；如果不允许压缩
        或压缩后仍无法满足规则，就明确报错，而不是把过长请求直接发给模型。
        """

        policy = self._context_policy
        if policy is None:
            return messages, None, None
        counter = self.client.token_counter
        original_tokens = int(counter(messages, tools))
        maximum = (
            policy.window_tokens
            - policy.generation_reserve_tokens
            - policy.safety_margin_tokens
        )
        input_budget = policy.input_budget_tokens or maximum
        if original_tokens <= input_budget:
            return messages, original_tokens, None
        if not policy.compaction_enabled:
            raise ContextBudgetError(
                f"prompt uses {original_tokens} tokens, above input budget "
                f"{input_budget}"
            )
        compacted, stats = compact_chat_messages(
            messages,
            tools,
            count_tokens=counter,
            max_input_tokens=input_budget,
        )
        return compacted, original_tokens, stats.to_dict()

    async def project_observation(
        self,
        tool_name: str,
        observation: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """把很长的环境页面压缩成模型能看到的版本，并返回压缩记录。

        如果旧客户端没有提供页面压缩能力，就原样返回页面。这样 Harness 可以兼容
        简单客户端，同时在启用投影配置时通过前面的配置检查保证能力确实存在。
        """

        projector = getattr(self.client, "project_observation", None)
        if not callable(projector):
            return str(observation), None
        args = (str(tool_name), str(observation), dict(parameters or {}))
        if self.call_in_thread:
            visible, meta = await asyncio.to_thread(projector, *args)
        else:
            visible, meta = projector(*args)
        if meta is not None and hasattr(meta, "to_dict"):
            meta = meta.to_dict()
        if meta is not None and not isinstance(meta, Mapping):
            raise TypeError("observation projection metadata must be an object")
        return str(visible), deepcopy(dict(meta)) if meta is not None else None


def _require_equal(client: object, name: str, expected: object) -> None:
    """要求旧客户端某项设置存在且与 Harness 完全相同，否则立即报配置错误。"""

    actual = getattr(client, name, _MISSING)
    if actual is _MISSING:
        raise ConfigValidationError(
            f"legacy client does not expose required setting {name!r}"
        )
    if actual != expected:
        raise ConfigValidationError(
            f"legacy client {name}={actual!r} differs from HarnessConfig "
            f"value {expected!r}"
        )


def _field(value: object, name: str, default: Any) -> Any:
    """同时兼容字典和普通对象，从中读取字段；没有该字段时返回默认值。"""

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = ["LegacyChatPolicyAdapter"]
