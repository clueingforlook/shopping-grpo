"""定义 WLX Harness Core 各部分交换数据时共同遵守的数据格式。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


HARNESS_CONTRACT_VERSION = "wlx-harness-contract-v1"
TRAJECTORY_SCHEMA_VERSION = "wlx-harness-trajectory-v1"
ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
REWARD_VERSION = "shopsimulator-reward-v3"
TOOL_SCHEMA_VERSION = "shopping-tools-v2"


class TerminationCategory(str, Enum):
    """说明一条购物轨迹为什么结束，便于训练和评测统一统计。"""

    RUNNING = "running"
    ENVIRONMENT_DONE = "environment_done"
    MODEL_STOPPED = "model_stopped"
    LIMIT_REACHED = "limit_reached"
    INVALID_ACTION = "invalid_action"
    PROTOCOL_ERROR = "protocol_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class ErrorCategory(str, Enum):
    """给运行中的错误分组，方便判断问题来自模型、环境还是基础设施。"""

    MODEL = "model"
    POLICY = "policy"
    ENVIRONMENT = "environment"
    PROTOCOL = "protocol"
    INFRASTRUCTURE = "infrastructure"
    RELEASE = "release"
    INTERNAL = "internal"


@dataclass(frozen=True)
class EpisodeRequest:
    """表示要让购物 Agent 执行的一道题，以及这次尝试的附加信息。"""

    task_id: int
    instruction: str | None = None
    prompt: tuple[dict[str, Any], ...] = ()
    attempt_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """整理任务编号、提示词和附加信息，并检查尝试次数是否合法。"""

        if isinstance(self.task_id, bool):
            raise TypeError("task_id must be an integer")
        object.__setattr__(self, "task_id", int(self.task_id))
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")
        object.__setattr__(
            self,
            "prompt",
            tuple(deepcopy(dict(item)) for item in self.prompt),
        )
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeRequest":
        """从普通字典读取任务；兼容项目已有的多种 task_id 存放位置。"""

        extra = value.get("extra_info") or {}
        interaction = extra.get("interaction_kwargs") or {} if isinstance(extra, Mapping) else {}
        task_id = value.get("task_id")
        if task_id is None and isinstance(extra, Mapping):
            task_id = extra.get("task_id")
        if task_id is None and isinstance(interaction, Mapping):
            task_id = interaction.get("task_id")
        if task_id is None:
            raise ValueError("episode request is missing task_id")
        prompt = value.get("prompt") or ()
        return cls(
            task_id=int(task_id),
            instruction=value.get("instruction"),
            prompt=tuple(prompt),
            attempt_index=int(value.get("attempt_index", 0) or 0),
            metadata=deepcopy(dict(value.get("metadata") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """把任务请求转成可写入 JSON 的普通字典，并返回一份安全副本。"""

        return _json_safe(self)


@dataclass(frozen=True)
class ToolCall:
    """表示模型想调用一次购物工具，包括工具名、参数和调用编号。"""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_type: str = "function"

    def __post_init__(self) -> None:
        """创建工具调用时检查工具名，并复制参数，避免外部修改原数据。"""

        if not self.name:
            raise ValueError("tool call name must be non-empty")
        object.__setattr__(self, "call_id", str(self.call_id or ""))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "arguments", deepcopy(dict(self.arguments)))

    @classmethod
    def from_openai(cls, value: Mapping[str, Any]) -> "ToolCall":
        """把 OpenAI 风格的工具调用字典解析成统一的 ToolCall 对象。"""

        if not isinstance(value, Mapping):
            raise TypeError("tool call must be an object")
        function = value.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call is missing function object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call is missing function name")
        raw_arguments = function.get("arguments", {})
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError("tool call arguments are invalid JSON") from exc
        elif isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            raise ValueError("tool call arguments must be an object or JSON string")
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must decode to an object")
        return cls(
            call_id=str(value.get("id") or ""),
            name=name,
            arguments=arguments,
            call_type=str(value.get("type") or "function"),
        )

    def to_openai_dict(self) -> dict[str, Any]:
        """把统一工具调用转回 OpenAI 接口认识的字典格式。"""

        return {
            "id": self.call_id,
            "type": self.call_type,
            "function": {
                "name": self.name,
                "arguments": json.dumps(
                    dict(self.arguments),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """把工具调用转成 Harness 自己保存轨迹时使用的字典格式。"""

        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": _json_safe(self.arguments),
            "call_type": self.call_type,
        }


@dataclass(frozen=True)
class ToolAction:
    """保存工具调用经过页面规则检查后，真正要交给环境的动作。"""

    tool_call: ToolCall
    env_action: str | None
    guard_rejection: str | None = None

    @property
    def allowed(self) -> bool:
        """返回这个动作是否通过检查；没有拒绝原因就表示可以执行。"""

        return self.guard_rejection is None

    def to_dict(self) -> dict[str, Any]:
        """把检查后的工具动作转成可以安全写入 JSON 的字典。"""

        return _json_safe(self)


@dataclass(frozen=True)
class ModelRequest:
    """汇总一次发给模型的消息、工具列表和生成参数。"""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """复制请求内容并检查输出长度，防止调用方之后意外改动请求。"""

        object.__setattr__(self, "messages", tuple(deepcopy(self.messages)))
        object.__setattr__(self, "tools", tuple(deepcopy(self.tools)))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        """把模型请求转成可以记录或传递的普通字典。"""

        return _json_safe(self)


@dataclass(frozen=True)
class AssistantTurn:
    """表示模型的一轮回答，可以是文字，也可以包含一个或多个工具调用。"""

    content: Any = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: Any = None
    additional_fields: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """固定工具调用列表并复制附加字段，让这一轮回答保持不变。"""

        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(
            self,
            "additional_fields",
            deepcopy(dict(self.additional_fields)),
        )

    @classmethod
    def from_mapping(cls, message: Mapping[str, Any]) -> "AssistantTurn":
        """从模型返回的消息字典中拆出正文、思考内容和工具调用。"""

        if not isinstance(message, Mapping):
            raise TypeError("assistant turn must be an object")
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
            raise ValueError("assistant tool_calls must be a list")
        known = {"role", "content", "tool_calls", "reasoning_content"}
        return cls(
            content=deepcopy(message.get("content")),
            tool_calls=tuple(ToolCall.from_openai(item) for item in raw_calls),
            reasoning_content=deepcopy(message.get("reasoning_content")),
            additional_fields={
                str(key): deepcopy(value)
                for key, value in message.items()
                if key not in known
            },
        )

    def to_message_dict(self) -> dict[str, Any]:
        """把这一轮回答还原成聊天消息字典，供后续模型调用继续使用。"""

        message = {"role": "assistant", "content": deepcopy(self.content)}
        if self.reasoning_content is not None:
            message["reasoning_content"] = deepcopy(self.reasoning_content)
        if self.tool_calls:
            message["tool_calls"] = [item.to_openai_dict() for item in self.tool_calls]
        message.update(deepcopy(dict(self.additional_fields)))
        return message

    def to_dict(self) -> dict[str, Any]:
        """把这一轮回答转成 Harness 轨迹里使用的普通字典。"""

        return {
            "content": _json_safe(self.content),
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "reasoning_content": _json_safe(self.reasoning_content),
            "additional_fields": _json_safe(self.additional_fields),
        }


@dataclass(frozen=True)
class ProjectionMetadata:
    """记录页面从原始内容裁剪成模型可见内容时发生了什么。"""

    raw_tokens: int = 0
    visible_tokens: int = 0
    truncated: bool = False
    critical_footer_preserved: bool = True
    visible_asin_count: int = 0
    visible_button_count: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ProjectionMetadata | None":
        """从投影信息字典创建对象；输入为空时直接返回 None。"""

        if value is None:
            return None
        known = {
            "raw_tokens",
            "visible_tokens",
            "truncated",
            "critical_footer_preserved",
            "visible_asin_count",
            "visible_button_count",
        }
        return cls(
            raw_tokens=int(value.get("raw_tokens", 0) or 0),
            visible_tokens=int(value.get("visible_tokens", 0) or 0),
            truncated=bool(value.get("truncated", False)),
            critical_footer_preserved=bool(value.get("critical_footer_preserved", True)),
            visible_asin_count=int(value.get("visible_asin_count", 0) or 0),
            visible_button_count=int(value.get("visible_button_count", 0) or 0),
            extra={str(key): deepcopy(item) for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        """把页面裁剪统计转成普通字典，并保留不认识的扩展字段。"""

        result = {
            "raw_tokens": self.raw_tokens,
            "visible_tokens": self.visible_tokens,
            "truncated": self.truncated,
            "critical_footer_preserved": self.critical_footer_preserved,
            "visible_asin_count": self.visible_asin_count,
            "visible_button_count": self.visible_button_count,
        }
        result.update(_json_safe(self.extra))
        return result


@dataclass(frozen=True)
class InitialState:
    """保存购物环境刚启动时给出的首页内容和原始返回值。"""

    observation: str
    raw_result: Mapping[str, Any]
    environment_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """把环境初始状态转成可以安全写入 JSON 的字典。"""

        return _json_safe(self)


@dataclass(frozen=True)
class EnvironmentStep:
    """保存环境执行一个动作后返回的页面、分数和结束状态。"""

    observation: str
    raw_result: Mapping[str, Any]
    reward: float = 0.0
    done: bool = False
    over: bool = False

    def to_dict(self) -> dict[str, Any]:
        """把一步环境结果转成可以安全写入 JSON 的字典。"""

        return _json_safe(self)


@dataclass(frozen=True)
class HarnessError:
    """用统一格式保存错误类型、说明、堆栈和额外细节。"""

    category: ErrorCategory
    error_type: str
    message: str
    traceback: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        category: ErrorCategory,
        exc: BaseException,
        *,
        traceback_text: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> "HarnessError":
        """把捕获到的 Python 异常包装成统一错误对象，方便保存到轨迹。"""

        return cls(
            category=category,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback=traceback_text,
            details=deepcopy(dict(details or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """把错误转成字典；只有确实存在的堆栈和细节才会写进去。"""

        result = {
            "category": self.category.value,
            "type": self.error_type,
            "message": self.message,
        }
        if self.traceback:
            result["traceback"] = self.traceback
        if self.details:
            result["details"] = _json_safe(self.details)
        return result


@dataclass(frozen=True)
class TrajectoryStep:
    """记录 Agent 的一次工具操作，以及环境对这次操作的反馈。"""

    step_index: int
    tool_call: ToolCall
    env_action: str | None
    observation: str
    reward: float
    done: bool
    result: Mapping[str, Any]
    raw_observation: str | None = None
    projection: ProjectionMetadata | None = None
    error: HarnessError | None = None

    def to_dict(self) -> dict[str, Any]:
        """把单步记录转成新版 Core 轨迹使用的字典格式。"""

        return {
            "step_index": self.step_index,
            "tool_call": self.tool_call.to_dict(),
            "env_action": self.env_action,
            "observation": self.observation,
            "raw_observation": self.raw_observation,
            "projection": self.projection.to_dict() if self.projection else None,
            "reward": self.reward,
            "done": self.done,
            "result": _json_safe(self.result),
            "error": self.error.to_dict() if self.error else None,
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """把单步记录转成旧版 SFT 和评测代码能够读取的格式。"""

        result = {
            "step_index": self.step_index,
            "tool_call": self.tool_call.to_openai_dict(),
            "tool_name": self.tool_call.name,
            "parameters": _json_safe(self.tool_call.arguments),
            "env_action": self.env_action,
            "observation": self.observation,
            "reward": float(self.reward),
            "done": bool(self.done),
            "result": _json_safe(self.result),
        }
        if self.raw_observation is not None:
            result["raw_observation"] = self.raw_observation
        if self.projection is not None:
            result["projection"] = self.projection.to_dict()
        if self.error is not None:
            result["error"] = self.error.to_dict()
        return result


@dataclass(frozen=True)
class Trajectory:
    """保存一道购物任务从开始到结束的完整过程和最终结果。"""

    trajectory_id: str
    task_id: int
    attempt_index: int
    created_at: str
    finished_at: str
    status: str
    termination_category: TerminationCategory
    termination_reason: str | None
    messages: tuple[dict[str, Any], ...]
    steps: tuple[TrajectoryStep, ...]
    initial_result: Mapping[str, Any]
    terminal_result: Mapping[str, Any]
    final_reward: float
    done: bool
    reward_valid: bool | None = None
    sampling_invalid: bool = True
    blocked_tool_calls: tuple[dict[str, Any], ...] = ()
    tool_call_truncations: tuple[dict[str, Any], ...] = ()
    context_compactions: tuple[dict[str, Any], ...] = ()
    context_turn_tokens: tuple[dict[str, Any], ...] = ()
    infrastructure_invalid: bool = False
    error: HarnessError | None = None
    release_error: HarnessError | None = None
    stage_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """把完整轨迹转成带版本号的新版 Core 字典，便于审计和存档。"""

        return {
            "schema_version": self.schema_version,
            "contract_version": HARNESS_CONTRACT_VERSION,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "attempt_index": self.attempt_index,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "termination_category": self.termination_category.value,
            "termination_reason": self.termination_reason,
            "messages": _json_safe(self.messages),
            "steps": [item.to_dict() for item in self.steps],
            "initial_result": _json_safe(self.initial_result),
            "terminal_result": _json_safe(self.terminal_result),
            "final_reward": float(self.final_reward),
            "done": bool(self.done),
            "reward_valid": self.reward_valid,
            "sampling_invalid": bool(self.sampling_invalid),
            "blocked_tool_calls": _json_safe(self.blocked_tool_calls),
            "tool_call_truncations": _json_safe(self.tool_call_truncations),
            "context_compactions": _json_safe(self.context_compactions),
            "context_turn_tokens": _json_safe(self.context_turn_tokens),
            "infrastructure_invalid": bool(self.infrastructure_invalid),
            "error": self.error.to_dict() if self.error else None,
            "release_error": self.release_error.to_dict() if self.release_error else None,
            "stage_metadata": _json_safe(self.stage_metadata),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """把完整轨迹转成现有 SFT 和评测模块仍在使用的旧格式。"""

        return {
            "trajectory_schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "attempt_index": self.attempt_index,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "termination_category": self.termination_category.value,
            "termination_reason": self.termination_reason,
            "messages": _json_safe(self.messages),
            "steps": [item.to_legacy_dict() for item in self.steps],
            "reward_valid": self.reward_valid,
            "sampling_invalid": bool(self.sampling_invalid),
            "blocked_tool_calls": _json_safe(self.blocked_tool_calls),
            "tool_call_truncations": _json_safe(self.tool_call_truncations),
            "context_compactions": _json_safe(self.context_compactions),
            "context_turn_tokens": _json_safe(self.context_turn_tokens),
            "initial_result": _json_safe(self.initial_result),
            "terminal_result": _json_safe(self.terminal_result),
            "final_reward": float(self.final_reward),
            "done": bool(self.done),
            "infrastructure_invalid": bool(self.infrastructure_invalid),
            "error": self.error.to_dict() if self.error else None,
            "release_error": self.release_error.to_dict() if self.release_error else None,
            "stage_metadata": _json_safe(self.stage_metadata),
        }


def _json_safe(value: Any) -> Any:
    """递归复制数据并把枚举、数据类等转换成 JSON 能保存的基础类型。"""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}
    return deepcopy(value)


__all__ = [
    "AssistantTurn",
    "ENVIRONMENT_VERSION",
    "EnvironmentStep",
    "EpisodeRequest",
    "ErrorCategory",
    "HARNESS_CONTRACT_VERSION",
    "HarnessError",
    "InitialState",
    "ModelRequest",
    "ProjectionMetadata",
    "REWARD_VERSION",
    "TOOL_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "TerminationCategory",
    "ToolAction",
    "ToolCall",
    "Trajectory",
    "TrajectoryStep",
]
