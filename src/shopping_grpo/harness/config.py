"""Harness Core 的配置定义和检查规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shopping_grpo.harness.contracts import ENVIRONMENT_VERSION, REWARD_VERSION


class ConfigValidationError(ValueError):
    """表示配置写错了，继续运行可能得到不可靠结果。"""

    pass


@dataclass(frozen=True)
class ContextPolicy:
    """规定模型一次最多能看到多少上下文，以及要为回答预留多少空间。"""

    window_tokens: int
    generation_reserve_tokens: int = 512
    safety_margin_tokens: int = 512
    input_budget_tokens: int | None = None
    compaction_enabled: bool = False
    preserve_recent_groups: int = 1

    def __post_init__(self) -> None:
        """创建配置后立即检查各项数字，配置不合理时直接报错。"""

        _positive("window_tokens", self.window_tokens)
        _positive("generation_reserve_tokens", self.generation_reserve_tokens)
        _non_negative("safety_margin_tokens", self.safety_margin_tokens)
        _positive("preserve_recent_groups", self.preserve_recent_groups)
        maximum = (
            self.window_tokens
            - self.generation_reserve_tokens
            - self.safety_margin_tokens
        )
        if maximum < 1:
            raise ConfigValidationError(
                "context window must exceed generation reserve plus safety margin"
            )
        if self.input_budget_tokens is not None:
            _positive("input_budget_tokens", self.input_budget_tokens)
            if self.input_budget_tokens > maximum:
                raise ConfigValidationError(
                    "input_budget_tokens must fit the model context window"
                )


@dataclass(frozen=True)
class ObservationPolicy:
    """规定购物页面怎样裁剪，让模型看到的信息既够用又不会太长。"""

    token_budget: int
    detail_token_budget: int = 4096
    generic_token_budget: int = 768
    search_top_k: int = 20

    def __post_init__(self) -> None:
        """创建配置后检查页面预算和搜索结果数量是否为合理的正数。"""

        for name, value in (
            ("token_budget", self.token_budget),
            ("detail_token_budget", self.detail_token_budget),
            ("generic_token_budget", self.generic_token_budget),
        ):
            _positive(name, value)
            if value < 64:
                raise ConfigValidationError(f"{name} must be at least 64")
        _positive("search_top_k", self.search_top_k)


@dataclass(frozen=True)
class HarnessConfig:
    """汇总一次运行要用的设置；不启用的能力默认保持为 None。"""

    max_steps: int
    environment_base_url: str = "http://127.0.0.1:5700"
    environment_timeout_s: float = 60
    required_environment_version: str | None = ENVIRONMENT_VERSION
    required_reward_version: str | None = REWARD_VERSION
    max_guard_rejections: int = 3
    max_assistant_turns: int | None = None
    parallel_tool_call_policy: Literal["truncate", "reject"] = "truncate"
    context: ContextPolicy | None = None
    observation: ObservationPolicy | None = None
    validate_terminal_reward: bool = True

    def __post_init__(self) -> None:
        """配置对象创建完成后，统一调用检查函数确认它可以安全使用。"""

        validate_config(self)


def validate_config(config: HarnessConfig) -> HarnessConfig:
    """检查整套 Harness 配置；检查通过后原样返回，方便调用方接着使用。"""

    if not isinstance(config, HarnessConfig):
        raise TypeError("config must be HarnessConfig")
    _positive("max_steps", config.max_steps)
    _positive("environment_timeout_s", config.environment_timeout_s)
    _positive("max_guard_rejections", config.max_guard_rejections)
    if config.max_assistant_turns is not None:
        _positive("max_assistant_turns", config.max_assistant_turns)
    if not isinstance(config.environment_base_url, str) or not config.environment_base_url:
        raise ConfigValidationError("environment_base_url must be non-empty")
    if config.parallel_tool_call_policy not in {"truncate", "reject"}:
        raise ConfigValidationError(
            "parallel_tool_call_policy must be 'truncate' or 'reject'"
        )
    if not isinstance(config.validate_terminal_reward, bool):
        raise ConfigValidationError("validate_terminal_reward must be boolean")
    return config


def _positive(name: str, value: float) -> None:
    """检查某个配置值必须大于零；不符合要求时指出对应的配置名。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigValidationError(f"{name} must be positive")


def _non_negative(name: str, value: float) -> None:
    """检查某个配置值不能小于零；零在这里是允许的。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigValidationError(f"{name} must be non-negative")


__all__ = [
    "ConfigValidationError",
    "ContextPolicy",
    "HarnessConfig",
    "ObservationPolicy",
    "validate_config",
]
