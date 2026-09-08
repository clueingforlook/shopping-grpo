"""把统一 Harness 的设置接到 veRL GRPO 购物训练上。

可以把这个文件理解成一个“转接头”：统一 Harness 和 veRL 使用的配置格式不同，
这里负责在两者之间转换，但不亲自运行模型，也不亲自执行购物任务。

导入本文件时不会立刻导入 ``verl`` 或 ``shopping_grpo``。只有真正创建 GRPO
循环或 ShopSimulator 会话时，才会加载那些较重的运行依赖。因此，只检查配置、
生成配置文件或运行普通单元测试时，不需要先启动 veRL。

每条购物轨迹只使用一个 ``ShopSimulatorSession``。真正的购物循环负责在开始时
申请环境、结束时释放环境；这个转接层只传递配置，绝不会自己占用环境。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields
from importlib import import_module
from types import MappingProxyType
from typing import Any, Final

from shopping_grpo.harness.config import HarnessConfig
from shopping_grpo.harness.contracts import (
    ENVIRONMENT_VERSION,
    REWARD_VERSION,
)


DEFAULT_AGENT_LOOP_NAME: Final = "shopping_tool_agent"
DEFAULT_AGENT_LOOP_TARGET: Final = (
    "shopping_grpo.training.grpo.adapter.agent_loop.ShoppingToolAgentLoop"
)
DEFAULT_SESSION_TARGET: Final = (
    "shopping_grpo.training.grpo.adapter.session.ShopSimulatorSession"
)
DEFAULT_TOOL_CLASS_TARGET: Final = (
    "shopping_grpo.training.grpo.adapter.tools.ShopSimulatorTool"
)
DEFAULT_TOOL_REGISTRY_TARGET: Final = (
    "shopping_grpo.harness.tools.SHOPPING_TOOL_REGISTRY"
)
DEFAULT_MAX_ASSISTANT_TURNS: Final = 40
REQUIRED_MAX_GUARD_REJECTIONS: Final = 3
REQUIRED_PARALLEL_TOOL_CALL_POLICY: Final = "reject"
SHOPPING_EXTRA_FIELDS_NAMESPACE: Final = "shopping"

# 这里的顺序要保持稳定，因为它同时也是给人看的字段清单。
SHOPPING_EXTRA_FIELDS: Final = (
    "task_id",
    "steps",
    "done",
    "termination_reason",
    "error",
    "infrastructure_invalid",
    "action_attempts",
    "repeat_actions",
    "reward_mode",
    "reward_version",
    "reward_type",
    "reward_valid",
    "reward_unverifiable",
    "reward",
    "context_compactions",
    "context_tokens_removed",
    "context_max_input_tokens",
    "observation_projection_count",
    "observation_truncated_count",
    "observation_raw_tokens",
    "observation_visible_tokens",
    "observation_max_raw_tokens",
    "observation_max_visible_tokens",
    "observation_visible_asin_count",
    "observation_visible_button_count",
    "observation_any_truncated",
    "observation_footer_failures",
    "guard_rejections",
    "guard_rejections_after_truncation",
    "action_attempts_after_truncation",
)
SHOPPING_EXTRA_FIELDS_REQUIRED: Final = frozenset(SHOPPING_EXTRA_FIELDS)

_INTEGER_EXTRA_FIELDS: Final = frozenset(
    {
        "task_id",
        "steps",
        "action_attempts",
        "repeat_actions",
        "context_compactions",
        "context_tokens_removed",
        "context_max_input_tokens",
        "observation_projection_count",
        "observation_truncated_count",
        "observation_raw_tokens",
        "observation_visible_tokens",
        "observation_max_raw_tokens",
        "observation_max_visible_tokens",
        "observation_visible_asin_count",
        "observation_visible_button_count",
        "observation_footer_failures",
        "guard_rejections",
        "guard_rejections_after_truncation",
        "action_attempts_after_truncation",
    }
)
_BOOLEAN_EXTRA_FIELDS: Final = frozenset(
    {
        "done",
        "infrastructure_invalid",
        "reward_valid",
        "reward_unverifiable",
        "observation_any_truncated",
    }
)

_AGENT_LOOP_CONFIG_FIELDS: Final = (
    "base_url",
    "timeout",
    "max_steps",
    "required_environment_version",
    "reward_mode",
    "context_window_tokens",
    "context_generation_reserve_tokens",
    "context_safety_margin_tokens",
    "context_input_budget_tokens",
    "context_preserve_recent_groups",
    "context_compaction_enable",
    "observation_token_budget",
    "observation_detail_token_budget",
    "observation_generic_token_budget",
    "observation_search_top_k",
)


_HARNESS_CONFIG_MAPPING_KEYS: Final = frozenset(
    {
        "environment_base_url",
        "environment_timeout_s",
        "required_reward_version",
        "max_guard_rejections",
        "max_assistant_turns",
        "parallel_tool_call_policy",
        "context",
        "observation",
        "validate_terminal_reward",
    }
)


class BridgeDependencyError(ImportError):
    """真正加载 veRL 或购物运行类失败时抛出的错误。"""


class ExtraFieldsContractError(ValueError):
    """veRL 返回的购物附加信息不符合约定时抛出的错误。"""


class ToolConfigContractError(ValueError):
    """veRL 的工具配置和统一工具清单不一致时抛出的错误。"""


@dataclass(frozen=True)
class GrpoStageConfig:
    """集中保存 GRPO 购物循环要用的设置。

    这个类把环境地址、最大步数、上下文长度和观察页面长度等参数放在一起，
    后面可以直接转换成 veRL 能读懂的配置。``extra_agent_loop_kwargs`` 用来
    暂存项目以后新增的参数，但不允许它偷偷覆盖 Harness 已经管住的关键参数。"""

    agent_loop_name: str = DEFAULT_AGENT_LOOP_NAME
    agent_loop_target: str = DEFAULT_AGENT_LOOP_TARGET
    session_target: str = DEFAULT_SESSION_TARGET
    tool_class_target: str = DEFAULT_TOOL_CLASS_TARGET
    base_url: str = "http://127.0.0.1:5700"
    timeout: int = 60
    max_steps: int = 35
    max_assistant_turns: int = DEFAULT_MAX_ASSISTANT_TURNS
    required_environment_version: str = ENVIRONMENT_VERSION
    required_reward_version: str = REWARD_VERSION
    max_guard_rejections: int = REQUIRED_MAX_GUARD_REJECTIONS
    parallel_tool_call_policy: str = REQUIRED_PARALLEL_TOOL_CALL_POLICY
    validate_terminal_reward: bool = True
    reward_mode: str = "constraint_aware"
    context_window_tokens: int = 24576
    context_generation_reserve_tokens: int = 512
    context_safety_margin_tokens: int = 512
    context_input_budget_tokens: int = 16384
    context_preserve_recent_groups: int = 1
    context_compaction_enable: bool = False
    observation_token_budget: int = 1536
    observation_detail_token_budget: int = 4096
    observation_generic_token_budget: int = 768
    observation_search_top_k: int = 20
    extra_agent_loop_kwargs: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """配置对象创建后立即检查参数，尽早告诉使用者哪里填错了。

        平时不需要手动调用它；执行 ``GrpoStageConfig(...)`` 时 Python 会自动调用。
        这里只做校验和只读化处理，不会连接环境，也不会开始训练。"""

        for label, value in (
            ("agent_loop_name", self.agent_loop_name),
            ("agent_loop_target", self.agent_loop_target),
            ("session_target", self.session_target),
            ("tool_class_target", self.tool_class_target),
            ("base_url", self.base_url),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        for label, target in (
            ("agent_loop_target", self.agent_loop_target),
            ("session_target", self.session_target),
            ("tool_class_target", self.tool_class_target),
        ):
            if not _is_dotted_target(target):
                raise ValueError(f"{label} must be a dotted module attribute")

        positive_values = {
            "timeout": self.timeout,
            "max_steps": self.max_steps,
            "max_assistant_turns": self.max_assistant_turns,
            "context_window_tokens": self.context_window_tokens,
            "context_generation_reserve_tokens": self.context_generation_reserve_tokens,
            "context_input_budget_tokens": self.context_input_budget_tokens,
            "context_preserve_recent_groups": self.context_preserve_recent_groups,
            "observation_token_budget": self.observation_token_budget,
            "observation_detail_token_budget": self.observation_detail_token_budget,
            "observation_generic_token_budget": self.observation_generic_token_budget,
            "observation_search_top_k": self.observation_search_top_k,
        }
        for label, value in positive_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if (
            isinstance(self.context_safety_margin_tokens, bool)
            or not isinstance(self.context_safety_margin_tokens, int)
            or self.context_safety_margin_tokens < 0
        ):
            raise ValueError("context_safety_margin_tokens must be a non-negative integer")
        if not isinstance(self.context_compaction_enable, bool):
            raise ValueError("context_compaction_enable must be boolean")
        maximum_input = (
            self.context_window_tokens
            - self.context_generation_reserve_tokens
            - self.context_safety_margin_tokens
        )
        if self.context_input_budget_tokens > maximum_input:
            raise ValueError("context_input_budget_tokens must fit the model context window")
        if min(
            self.observation_token_budget,
            self.observation_detail_token_budget,
            self.observation_generic_token_budget,
        ) < 64:
            raise ValueError("all observation token budgets must be at least 64")
        if self.reward_mode not in {"native", "constraint_aware"}:
            raise ValueError(f"unknown shopping reward mode: {self.reward_mode!r}")

        if self.required_environment_version != ENVIRONMENT_VERSION:
            raise ValueError(f"GRPO requires environment version {ENVIRONMENT_VERSION!r}")
        if self.required_reward_version != REWARD_VERSION:
            raise ValueError(f"GRPO requires reward version {REWARD_VERSION!r}")
        if self.max_guard_rejections != REQUIRED_MAX_GUARD_REJECTIONS:
            raise ValueError(
                f"GRPO requires max_guard_rejections={REQUIRED_MAX_GUARD_REJECTIONS}"
            )
        if self.parallel_tool_call_policy != REQUIRED_PARALLEL_TOOL_CALL_POLICY:
            raise ValueError(
                "GRPO requires parallel_tool_call_policy="
                f"{REQUIRED_PARALLEL_TOOL_CALL_POLICY!r}"
            )
        if self.validate_terminal_reward is not True:
            raise ValueError("GRPO requires validate_terminal_reward=True")
        extensions = dict(self.extra_agent_loop_kwargs)
        reserved = {
            "name",
            "_target_",
            "session_target",
            "tool_class_target",
            "max_assistant_turns",
            "required_reward_version",
            "max_guard_rejections",
            "parallel_tool_call_policy",
            "validate_terminal_reward",
            *_AGENT_LOOP_CONFIG_FIELDS,
        }
        overlap = reserved.intersection(extensions)
        if overlap:
            labels = ", ".join(sorted(overlap))
            raise ValueError(f"extra_agent_loop_kwargs overrides reserved fields: {labels}")
        object.__setattr__(
            self,
            "extra_agent_loop_kwargs",
            MappingProxyType(extensions),
        )

    @classmethod
    def from_harness_config(
        cls,
        config: HarnessConfig,
        **overrides: Any,
    ) -> "GrpoStageConfig":
        """把统一的 ``HarnessConfig`` 翻译成 GRPO 阶段配置。

        当 SFT、评测和 GRPO 共用一份 Harness 配置时调用它。它会把通用字段放到
        GRPO 对应的位置，并确认奖励版本、工具调用规则等关键约定没有被改坏。"""

        if not isinstance(config, HarnessConfig):
            raise TypeError("config must be HarnessConfig")
        fixed_semantics = {
            "required_environment_version": ENVIRONMENT_VERSION,
            "required_reward_version": REWARD_VERSION,
            "max_guard_rejections": REQUIRED_MAX_GUARD_REJECTIONS,
            "parallel_tool_call_policy": REQUIRED_PARALLEL_TOOL_CALL_POLICY,
            "validate_terminal_reward": True,
        }
        for name, expected in fixed_semantics.items():
            actual = getattr(config, name)
            if actual != expected:
                raise ValueError(
                    f"HarnessConfig {name} must be {expected!r} for GRPO, got {actual!r}"
                )

        timeout = float(config.environment_timeout_s)
        if not timeout.is_integer():
            raise ValueError("GRPO environment_timeout_s must be a whole number of seconds")

        values: dict[str, Any] = {
            "base_url": config.environment_base_url,
            "timeout": int(timeout),
            "max_steps": config.max_steps,
            "required_environment_version": config.required_environment_version,
            "required_reward_version": config.required_reward_version,
            "max_guard_rejections": config.max_guard_rejections,
            "parallel_tool_call_policy": config.parallel_tool_call_policy,
            "validate_terminal_reward": config.validate_terminal_reward,
        }
        if config.max_assistant_turns is not None:
            values["max_assistant_turns"] = config.max_assistant_turns

        context = config.context
        if context is not None:
            input_budget = context.input_budget_tokens
            if input_budget is None:
                input_budget = (
                    context.window_tokens
                    - context.generation_reserve_tokens
                    - context.safety_margin_tokens
                )
            values.update(
                {
                    "context_window_tokens": context.window_tokens,
                    "context_generation_reserve_tokens": context.generation_reserve_tokens,
                    "context_safety_margin_tokens": context.safety_margin_tokens,
                    "context_input_budget_tokens": input_budget,
                    "context_preserve_recent_groups": context.preserve_recent_groups,
                    "context_compaction_enable": context.compaction_enabled,
                }
            )

        observation = config.observation
        if observation is not None:
            values.update(
                {
                    "observation_token_budget": observation.token_budget,
                    "observation_detail_token_budget": observation.detail_token_budget,
                    "observation_generic_token_budget": observation.generic_token_budget,
                    "observation_search_top_k": observation.search_top_k,
                }
            )

        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_agent_loop_entry(
        cls,
        entry: Mapping[str, Any],
        *,
        session_target: str = DEFAULT_SESSION_TARGET,
    ) -> "GrpoStageConfig":
        """读取 ``configs/agent_loop.yaml`` 中的一项并变成配置对象。

        这个入口适合继续使用项目原有的 veRL 配置。认识的字段会放到固定位置，
        暂时不认识的扩展字段会保留下来，交给原购物循环处理。"""

        if not isinstance(entry, Mapping):
            raise TypeError("agent loop entry must be a mapping")
        values = dict(entry)
        constructor: dict[str, Any] = {
            "agent_loop_name": values.pop("name", DEFAULT_AGENT_LOOP_NAME),
            "agent_loop_target": values.pop("_target_", DEFAULT_AGENT_LOOP_TARGET),
            "session_target": session_target,
        }
        for field_name in _AGENT_LOOP_CONFIG_FIELDS:
            if field_name in values:
                constructor[field_name] = values.pop(field_name)
        constructor["extra_agent_loop_kwargs"] = values
        return cls(**constructor)

    @classmethod
    def from_mapping(
        cls,
        entry: Mapping[str, Any],
        *,
        session_target: str = DEFAULT_SESSION_TARGET,
    ) -> "GrpoStageConfig":
        """把普通字典当成 veRL 的 agent-loop 配置来读取。

        这里故意不接受 ``HarnessConfig`` 直接转出的字典，避免两套长得相似的配置
        被误用。通用 Harness 配置应先建成对象，再调用 ``from_harness_config``。"""

        if not isinstance(entry, Mapping):
            raise TypeError("agent loop entry must be a mapping")
        core_keys = _HARNESS_CONFIG_MAPPING_KEYS.intersection(entry)
        if core_keys:
            labels = ", ".join(sorted(core_keys))
            raise TypeError(
                "HarnessConfig mapping passed as agent-loop config; construct "
                f"HarnessConfig and use from_harness_config() instead ({labels})"
            )
        return cls.from_agent_loop_entry(entry, session_target=session_target)

    def agent_loop_kwargs(self) -> dict[str, Any]:
        """整理创建购物 Agent 循环时要传入的参数。

        ``VerlHarnessBridge.make_agent_loop`` 会调用它。返回的是一份新字典，包含
        固定参数以及允许保留的扩展参数。"""

        values = {
            field_name: getattr(self, field_name)
            for field_name in _AGENT_LOOP_CONFIG_FIELDS
        }
        values.update(self.extra_agent_loop_kwargs)
        return values

    def session_kwargs(self) -> dict[str, Any]:
        """整理创建 ShopSimulator 会话真正需要的那部分参数。

        它只挑出环境地址、超时、最大步数和环境版本，不会把模型上下文之类与环境
        无关的配置传进去。"""

        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_steps": self.max_steps,
            "required_environment_version": self.required_environment_version,
        }

    def to_agent_loop_entry(self) -> dict[str, Any]:
        """生成一项 Hydra/veRL 能直接读取的 Agent 循环配置。

        在准备 veRL 启动配置时调用，结果中会带上循环名字、Python 类路径和构造
        循环所需的全部参数。"""

        return {
            "name": self.agent_loop_name,
            "_target_": self.agent_loop_target,
            **self.agent_loop_kwargs(),
        }

    def to_dict(self) -> dict[str, Any]:
        """把配置复制成普通、可修改、可序列化的字典。

        适合打印、保存或交给配置系统。返回值与当前对象分开，修改它不会反过来
        改动这个配置对象，也不会暴露内部使用的只读字典包装。"""

        values: dict[str, Any] = {}
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if descriptor.name == "extra_agent_loop_kwargs":
                value = dict(value)
            values[descriptor.name] = deepcopy(value)
        return values


class VerlHarnessBridge:
    """连接 Harness 配置和现有 veRL 购物循环的转接器。

    创建这个对象时只检查和保存配置，所以没有 GPU、没有 veRL 的机器也能用它。
    只有调用 ``resolve_*`` 或 ``make_*`` 方法时才会加载运行依赖。它没有 ``run``
    方法，因为模型生成和逐 token 的购物循环仍由 veRL 负责。"""

    def __init__(
        self,
        config: GrpoStageConfig | HarnessConfig | Mapping[str, Any] | None = None,
        *,
        module_loader: Callable[[str], Any] = import_module,
    ) -> None:
        """创建 GRPO 转接器，并把传入配置统一成 ``GrpoStageConfig``。

        准备生成 veRL 配置、检查输出或创建运行对象前会调用这个构造函数。此时不会
        连接 ShopSimulator，也不会加载 veRL；``module_loader`` 主要方便测试时替换导入器。"""

        if config is None:
            config = GrpoStageConfig()
        elif isinstance(config, HarnessConfig):
            config = GrpoStageConfig.from_harness_config(config)
        elif isinstance(config, Mapping):
            config = GrpoStageConfig.from_mapping(config)
        elif not isinstance(config, GrpoStageConfig):
            raise TypeError("config must be GrpoStageConfig, HarnessConfig, a mapping, or None")
        if not callable(module_loader):
            raise TypeError("module_loader must be callable")
        self.config = config
        self._module_loader = module_loader

    def to_agent_loop_config(self) -> list[dict[str, Any]]:
        """生成 veRL 配置加载器需要的 Agent 循环列表。

        veRL 即使只配置一个购物循环也要求外层是列表，所以这里把当前配置包装成
        一项列表，通常用于写入或拼装启动配置。"""

        return [self.config.to_agent_loop_entry()]

    def to_verl_tool_config(self, *, registry: Any | None = None) -> dict[str, Any]:
        """根据统一工具清单生成 veRL 能读的工具配置。

        准备 GRPO 训练配置时调用。这样搜索、查看商品和购买等工具只维护一份定义，
        避免 Harness 与 veRL 各写一份后慢慢变得不一致。"""

        registry = self._resolve_tool_registry(registry)
        schemas = getattr(registry, "schemas", None)
        if not isinstance(schemas, Sequence) or isinstance(schemas, (str, bytes)):
            raise ToolConfigContractError("tool registry schemas must be a sequence")
        tools = []
        for index, schema in enumerate(schemas):
            if not isinstance(schema, Mapping):
                raise ToolConfigContractError(f"tool registry schema {index} must be a mapping")
            tools.append(
                {
                    "class_name": self.config.tool_class_target,
                    "config": {"type": "native"},
                    "tool_schema": deepcopy(dict(schema)),
                }
            )
        if not tools:
            raise ToolConfigContractError("tool registry must not be empty")
        return {"tools": tools}

    def assert_tool_config_compatible(
        self,
        candidate: Mapping[str, Any],
        *,
        registry: Any | None = None,
    ) -> None:
        """检查现有 veRL 工具配置是否仍和统一工具清单一致。

        在启动训练前做预检时调用。它会逐项检查工具类、调用类型和参数结构；有漂移
        就立即报错，防止训练跑到一半才发现模型调用的工具对不上。"""

        if not isinstance(candidate, Mapping):
            raise ToolConfigContractError("veRL tool config must be a mapping")
        entries = candidate.get("tools")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ToolConfigContractError("veRL tool config 'tools' must be a sequence")

        schemas = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ToolConfigContractError(f"veRL tool entry {index} must be a mapping")
            if entry.get("class_name") != self.config.tool_class_target:
                raise ToolConfigContractError(
                    f"veRL tool entry {index} has an unexpected class_name"
                )
            if entry.get("config") != {"type": "native"}:
                raise ToolConfigContractError(
                    f"veRL tool entry {index} must use native tool config"
                )
            schema = entry.get("tool_schema")
            if not isinstance(schema, Mapping):
                raise ToolConfigContractError(
                    f"veRL tool entry {index} is missing tool_schema"
                )
            schemas.append(deepcopy(dict(schema)))

        registry = self._resolve_tool_registry(registry)
        checker = getattr(registry, "assert_compatible", None)
        if not callable(checker):
            raise ToolConfigContractError("tool registry must provide assert_compatible()")
        try:
            checker(schemas)
        except (TypeError, ValueError) as exc:
            raise ToolConfigContractError(str(exc)) from exc

    def _resolve_tool_registry(self, registry: Any | None) -> Any:
        """取得要使用的统一工具清单。

        调用者已经传入清单时直接使用；没有传入时才按默认 Python 路径加载。这个
        内部辅助函数供工具配置生成和检查方法调用。"""

        if registry is not None:
            return registry
        return self._resolve_target(DEFAULT_TOOL_REGISTRY_TARGET, role="tool registry")

    def to_verl_overrides(
        self,
        *,
        agent_loop_config_path: str | None = None,
        tool_config_path: str | None = None,
    ) -> dict[str, Any]:
        """生成 Harness 需要覆盖到 veRL rollout 配置中的少量参数。

        项目原来的 PPO/GRPO 训练配方仍由 Hydra 配置负责。这里仅补上 Agent 循环路径、
        工具路径、多轮开关以及一次只调用一个工具等 Harness 运行约定。"""

        agent: dict[str, Any] = {"default_agent_loop": self.config.agent_loop_name}
        multi_turn: dict[str, Any] = {
            "enable": True,
            "max_parallel_calls": 1,
            "max_assistant_turns": self.config.max_assistant_turns,
        }
        if agent_loop_config_path is not None:
            agent["agent_loop_config_path"] = str(agent_loop_config_path)
        if tool_config_path is not None:
            multi_turn["tool_config_path"] = str(tool_config_path)
        return {
            "actor_rollout_ref": {
                "rollout": {
                    "agent": agent,
                    "multi_turn": multi_turn,
                }
            }
        }

    def lease_contract(self) -> dict[str, str]:
        """用普通字典说明 ShopSimulator 环境由谁申请和释放。

        这个方法用于文档、日志或预检，不会真的操作环境。它明确一条轨迹的购物循环
        持有环境，避免工具层和转接层重复释放同一个环境。"""

        return {
            "scope": "trajectory",
            "owner": f"{self.config.agent_loop_target}.run",
            "acquire": f"{self.config.session_target}.start",
            "release": f"{self.config.session_target}.close",
            "tool_release": "no-op; the trajectory owner releases the shared lease",
            "bridge": "configuration-only; never acquires or releases a lease",
        }

    def extra_fields_contract(self) -> dict[str, Any]:
        """说明购物循环必须写出哪些公开诊断信息。

        评测或训练记录需要知道步数、结束原因和奖励是否有效等信息。这个方法返回这些
        字段放在哪里、由谁写、如何处理未知字段，也强调不能泄露模拟器隐藏目标。"""

        return {
            "container": "AgentLoopOutput.extra_fields",
            "namespace": SHOPPING_EXTRA_FIELDS_NAMESPACE,
            "writer": f"{self.config.agent_loop_target}.run",
            "stage_metadata_path": "Trajectory.stage_metadata['shopping']",
            "required_fields": SHOPPING_EXTRA_FIELDS,
            "unknown_fields": "preserve",
            "privacy": "public diagnostics only; never copy the hidden ShopSimulator goal",
        }

    def resolve_agent_loop_class(self) -> Any:
        """需要运行 GRPO 时，才加载真正的购物 Agent 循环类。

        配置生成阶段不会调用它；``make_agent_loop`` 要实例化运行对象时才调用，因此
        缺少 veRL 依赖的问题会在这里以更清楚的错误形式报告。"""

        return self._resolve_target(self.config.agent_loop_target, role="agent loop")

    def resolve_session_class(self) -> Any:
        """需要创建环境会话时，才加载 ``ShopSimulatorSession`` 类。

        这里只找到并返回 Python 类，不会创建会话，更不会申请 ShopSimulator 环境。"""

        return self._resolve_target(self.config.session_target, role="session")

    def make_agent_loop(self, *args: Any, **overrides: Any) -> Any:
        """创建现有的 veRL 购物循环对象，但不开始执行任务。

        通常由 veRL 的启动流程或测试代码调用。额外参数会覆盖普通循环参数，实例化后
        何时真正运行仍由 veRL 决定。"""

        kwargs = self.config.agent_loop_kwargs()
        kwargs.update(overrides)
        return self.resolve_agent_loop_class()(*args, **kwargs)

    def make_session(self, **overrides: Any) -> Any:
        """创建一个尚未启动的 ShopSimulator 会话对象。

        主要用于启动前检查或依赖注入。创建对象不等于占用环境；只有真正的购物循环
        调用 ``session.start`` 后才会申请环境，调用者不要绕过循环抢走管理权。"""

        kwargs = self.config.session_kwargs()
        kwargs.update(overrides)
        return self.resolve_session_class()(**kwargs)

    def validate_extra_fields(
        self,
        extra_fields: Mapping[str, Any],
        *,
        require_complete: bool = True,
    ) -> dict[str, Any]:
        """检查并复制 ``extra_fields['shopping']`` 中的购物诊断信息。

        收到一次 veRL 运行结果后调用。它会检查必填字段和常见字段类型，同时保留项目
        以后新增的字段；返回副本，避免后续处理意外改动原始输出。"""

        if not isinstance(extra_fields, Mapping):
            raise ExtraFieldsContractError("AgentLoopOutput.extra_fields must be a mapping")
        payload = extra_fields.get(SHOPPING_EXTRA_FIELDS_NAMESPACE)
        if not isinstance(payload, Mapping):
            raise ExtraFieldsContractError(
                "AgentLoopOutput.extra_fields['shopping'] must be a mapping"
            )
        if require_complete:
            missing = SHOPPING_EXTRA_FIELDS_REQUIRED.difference(payload)
            if missing:
                labels = ", ".join(sorted(missing))
                raise ExtraFieldsContractError(f"shopping extra_fields missing: {labels}")
        self._validate_extra_field_types(payload)
        return dict(payload)

    def extract_extra_fields(
        self,
        output: Any,
        *,
        require_complete: bool = True,
    ) -> dict[str, Any]:
        """从 veRL 输出对象或字典中取出购物诊断信息并检查。

        调用者不必关心输出是对象还是字典。这个方法先找到 ``extra_fields``，再交给
        ``validate_extra_fields`` 做统一校验。"""

        if isinstance(output, Mapping):
            extra_fields = output.get("extra_fields", output)
        else:
            extra_fields = getattr(output, "extra_fields", None)
        return self.validate_extra_fields(
            extra_fields,
            require_complete=require_complete,
        )

    def to_stage_metadata(
        self,
        output: Any,
        *,
        existing: Mapping[str, Any] | None = None,
        require_complete: bool = True,
    ) -> dict[str, Any]:
        """把 veRL 的购物诊断信息放进轨迹的固定 metadata 位置。

        GRPO 结果要交给统一 Trajectory 时调用。已有的其他阶段信息会保留，但如果里面
        已经有 ``shopping``，就拒绝覆盖，以免悄悄丢失旧数据。"""

        if existing is not None and not isinstance(existing, Mapping):
            raise ExtraFieldsContractError("existing stage_metadata must be a mapping")
        metadata = deepcopy(dict(existing or {}))
        if SHOPPING_EXTRA_FIELDS_NAMESPACE in metadata:
            raise ExtraFieldsContractError(
                "existing stage_metadata already contains the 'shopping' namespace"
            )
        metadata[SHOPPING_EXTRA_FIELDS_NAMESPACE] = self.extract_extra_fields(
            output,
            require_complete=require_complete,
        )
        return metadata

    def _resolve_target(self, target: str, *, role: str) -> Any:
        """根据“模块.属性”形式的路径懒加载一个运行对象。

        Agent 循环、环境会话和工具清单都通过这个内部函数加载。它把普通导入错误改成
        更容易看懂的依赖错误，告诉使用者缺的是哪类 GRPO 运行组件。"""

        module_name, _, attribute = target.rpartition(".")
        try:
            module = self._module_loader(module_name)
        except ImportError as exc:
            raise BridgeDependencyError(
                f"cannot load GRPO {role} {target!r}; install the shopping project "
                "and its GRPO/veRL runtime before resolving runtime classes"
            ) from exc
        try:
            return getattr(module, attribute)
        except AttributeError as exc:
            raise BridgeDependencyError(
                f"GRPO {role} target {target!r} does not exist"
            ) from exc

    @staticmethod
    def _validate_extra_field_types(payload: Mapping[str, Any]) -> None:
        """检查购物诊断字段的值是不是约定的类型和范围。

        这是 ``validate_extra_fields`` 使用的内部检查步骤，例如步数不能是负数、布尔值
        不能写成数字、奖励详情必须是字典。它只检查已出现的字段。"""

        for key in _INTEGER_EXTRA_FIELDS.intersection(payload):
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExtraFieldsContractError(
                    f"shopping extra_fields {key!r} must be a non-negative integer"
                )
        for key in _BOOLEAN_EXTRA_FIELDS.intersection(payload):
            if not isinstance(payload[key], bool):
                raise ExtraFieldsContractError(
                    f"shopping extra_fields {key!r} must be boolean"
                )
        if "reward" in payload and not isinstance(payload["reward"], Mapping):
            raise ExtraFieldsContractError("shopping extra_fields 'reward' must be a mapping")
        if "reward_mode" in payload and payload["reward_mode"] not in {
            "native",
            "constraint_aware",
        }:
            raise ExtraFieldsContractError(
                "shopping extra_fields 'reward_mode' must be native or constraint_aware"
            )
        for key in ("termination_reason", "error", "reward_version", "reward_type"):
            if key in payload and payload[key] is not None and not isinstance(payload[key], str):
                raise ExtraFieldsContractError(
                    f"shopping extra_fields {key!r} must be a string or None"
                )


def _is_dotted_target(value: str) -> bool:
    """判断一个字符串是否像“模块.属性”这样的 Python 加载路径。

    创建 GRPO 配置时用它提前拦住明显错误的类路径；它只检查基本形状，不会真的
    导入模块。"""

    module_name, separator, attribute = value.rpartition(".")
    return bool(separator and module_name and attribute)


__all__ = [
    "BridgeDependencyError",
    "DEFAULT_AGENT_LOOP_NAME",
    "DEFAULT_AGENT_LOOP_TARGET",
    "DEFAULT_MAX_ASSISTANT_TURNS",
    "DEFAULT_SESSION_TARGET",
    "DEFAULT_TOOL_CLASS_TARGET",
    "ExtraFieldsContractError",
    "GrpoStageConfig",
    "REQUIRED_MAX_GUARD_REJECTIONS",
    "REQUIRED_PARALLEL_TOOL_CALL_POLICY",
    "SHOPPING_EXTRA_FIELDS",
    "SHOPPING_EXTRA_FIELDS_NAMESPACE",
    "SHOPPING_EXTRA_FIELDS_REQUIRED",
    "ToolConfigContractError",
    "VerlHarnessBridge",
]
