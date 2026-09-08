"""集中管理所有阶段共用的购物工具定义、检查规则和动作转换。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping, Sequence

from shopping_grpo.environment.actions import action_reject_reason
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action

from shopping_grpo.harness.contracts import TOOL_SCHEMA_VERSION, ToolAction, ToolCall


class ToolRegistryError(ValueError):
    """表示工具不存在、工具定义冲突，或工具检查和转换失败。"""

    pass


def tool_schema_fingerprint(schemas: Sequence[Mapping[str, Any]]) -> str:
    """接收一组工具定义，返回稳定的哈希值，用来发现各阶段配置是否漂移。"""

    payload = json.dumps(
        list(schemas),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ToolRegistry:
    """把工具定义、页面安全检查和环境动作转换集中放在一个地方。"""

    def __init__(
        self,
        schemas: Sequence[Mapping[str, Any]],
        *,
        version: str = TOOL_SCHEMA_VERSION,
        guard: Callable[[str, Mapping[str, Any], str], str | None] = action_reject_reason,
        action_mapper: Callable[[str, Mapping[str, Any]], str | None] = tool_call_to_action,
    ) -> None:
        """创建工具注册表；输入工具定义和转换函数，同时检查重名和缺项。"""

        copied = tuple(deepcopy(dict(item)) for item in schemas)
        if not copied:
            raise ToolRegistryError("tool registry must not be empty")
        by_name: dict[str, dict[str, Any]] = {}
        for index, schema in enumerate(copied):
            function = schema.get("function")
            if not isinstance(function, Mapping):
                raise ToolRegistryError(f"tool schema {index} is missing function")
            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ToolRegistryError(f"tool schema {index} is missing name")
            if name in by_name:
                raise ToolRegistryError(f"duplicate tool schema: {name}")
            by_name[name] = schema
        self.version = str(version)
        self._schemas = copied
        self._by_name = by_name
        self._guard = guard
        self._action_mapper = action_mapper
        self._fingerprint = tool_schema_fingerprint(copied)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """返回全部工具定义的深拷贝，外部修改它不会影响注册表。"""

        return deepcopy(list(self._schemas))

    @property
    def tool_names(self) -> tuple[str, ...]:
        """返回当前注册表中的全部工具名。"""

        return tuple(self._by_name)

    @property
    def fingerprint(self) -> str:
        """返回工具定义的指纹，用它可以快速比较两份配置是否完全一致。"""

        return self._fingerprint

    def schema(self, name: str) -> dict[str, Any]:
        """按工具名取出一份工具定义；工具不存在时给出明确错误。"""

        try:
            return deepcopy(self._by_name[str(name)])
        except KeyError as exc:
            raise ToolRegistryError(f"unknown tool: {name!r}") from exc

    def parse_tool_call(self, value: Mapping[str, Any] | ToolCall) -> ToolCall:
        """接收工具调用对象或 OpenAI 字典，检查工具存在后返回统一对象。"""

        call = value if isinstance(value, ToolCall) else ToolCall.from_openai(value)
        if call.name not in self._by_name:
            raise ToolRegistryError(f"unknown tool: {call.name!r}")
        return call

    def guard(self, call: ToolCall, observation: str) -> str | None:
        """结合当前页面检查工具调用能否执行；返回 None 表示允许。"""

        try:
            return self._guard(call.name, dict(call.arguments), str(observation))
        except Exception as exc:
            raise ToolRegistryError(
                f"tool guard failed for {call.name!r}: {exc}"
            ) from exc

    def to_action(self, call: ToolCall) -> str | None:
        """把模型给出的工具调用转换成 ShopSimulator 能执行的动作字符串。"""

        try:
            return self._action_mapper(call.name, dict(call.arguments))
        except Exception as exc:
            raise ToolRegistryError(
                f"cannot map tool {call.name!r} to an environment action: {exc}"
            ) from exc

    def resolve(self, call: ToolCall, observation: str) -> ToolAction:
        """一次完成页面检查和动作转换，返回允许执行或被拒绝的结果。"""

        rejection = self.guard(call, observation)
        if rejection:
            return ToolAction(
                tool_call=call,
                env_action=None,
                guard_rejection=str(rejection),
            )
        return ToolAction(tool_call=call, env_action=self.to_action(call))

    def assert_compatible(
        self,
        candidate: "ToolRegistry | Sequence[Mapping[str, Any]]",
    ) -> None:
        """比较候选工具配置和当前配置的指纹；不一致就立即报错。"""

        fingerprint = (
            candidate.fingerprint
            if isinstance(candidate, ToolRegistry)
            else tool_schema_fingerprint(candidate)
        )
        if fingerprint != self.fingerprint:
            raise ToolRegistryError(
                "tool schema fingerprint mismatch: "
                f"expected {self.fingerprint}, got {fingerprint}"
            )


SHOPPING_TOOL_REGISTRY = ToolRegistry(SHOP_TOOL_SCHEMAS)

# SFT 不把“思考”做成环境动作；模型可以在 assistant.content 中写简短公开推理。
SFT_TOOL_SCHEMAS = tuple(
    deepcopy(schema)
    for schema in SHOP_TOOL_SCHEMAS
    if (schema.get("function") or {}).get("name") != "think"
)
SFT_TOOL_REGISTRY = ToolRegistry(SFT_TOOL_SCHEMAS)


__all__ = [
    "SHOPPING_TOOL_REGISTRY",
    "ToolRegistry",
    "ToolRegistryError",
    "SFT_TOOL_REGISTRY",
    "SFT_TOOL_SCHEMAS",
    "tool_schema_fingerprint",
]
