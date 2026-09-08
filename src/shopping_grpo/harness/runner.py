"""SFT 数据采集和评测共用的“单条购物任务运行器”。"""

from __future__ import annotations

import inspect
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from shopping_grpo.environment.actions import action_guard_tool_message
from shopping_grpo.evaluation.rollout import SYSTEM_PROMPT
from shopping_grpo.environment.client import (
    ShopEnvironmentError,
    ShopHttpError,
    ShopProtocolError,
)
from shopping_grpo.environment.context import ContextBudgetError
from shopping_grpo.environment.observation import (
    HEADER as STRUCTURED_OBSERVATION_HEADER,
    OBSERVATION_VERSION,
    StructuredObservationError,
)

from shopping_grpo.harness.config import ConfigValidationError, HarnessConfig, validate_config
from shopping_grpo.harness.contracts import (
    AssistantTurn,
    EpisodeRequest,
    ErrorCategory,
    HarnessError,
    ModelRequest,
    ProjectionMetadata,
    TerminationCategory,
    Trajectory,
    TrajectoryStep,
)
from shopping_grpo.harness.environment import (
    EnvironmentResult,
    ShopSimulatorEnvironmentAdapter,
)
from shopping_grpo.harness.reward import RewardContractError, validate_terminal_result
from shopping_grpo.harness.tools import (
    SHOPPING_TOOL_REGISTRY,
    ToolRegistry,
    ToolRegistryError,
)


class PolicyAdapter(Protocol):
    """模型适配器约定：Runner 只管提请求，不关心底层模型来自哪里。"""
    async def generate(self, request: ModelRequest) -> AssistantTurn | Mapping[str, Any]:
        """向模型要下一步回复；具体调用哪个模型，由实现这个接口的适配器决定。"""
        ...


class EnvironmentAdapter(Protocol):
    """环境适配器约定：用统一方法启动、执行和释放购物环境。"""
    async def start(self, request: EpisodeRequest) -> EnvironmentResult:
        """启动一条购物任务并拿到初始指令和页面；调用后就占用一个环境租约。"""
        ...
    async def execute(self, action: str) -> EnvironmentResult:
        """把已经通过检查的商店动作交给环境执行，并返回新的页面状态。"""
        ...
    async def close(self) -> None:
        """结束本次任务并归还环境资源；Runner 无论成功失败都会调用它。"""
        ...


class RunObserver(Protocol):
    """运行过程的旁观者接口，用来接收安全清洗后的开始、步骤和结束事件。"""
    async def on_start(self, event: Mapping[str, Any]) -> None:
        """任务刚开始时接收一份不含隐藏答案的通知。"""
        ...
    async def on_step(self, event: Mapping[str, Any]) -> None:
        """每执行完一个公开步骤就接收一次通知，方便记录日志或指标。"""
        ...
    async def on_finish(self, event: Mapping[str, Any]) -> None:
        """整条任务结束时接收最终公开结果，不包含审计专用的原始数据。"""
        ...


class EpisodeRunner:
    """串起模型与 ShopSimulator 的核心状态机，并保证环境最终被释放。"""

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry = SHOPPING_TOOL_REGISTRY,
        environment_factory: Callable[[HarnessConfig], EnvironmentAdapter] | None = None,
        system_prompt: str | None = SYSTEM_PROMPT,
        include_initial_observation: bool = False,
    ) -> None:
        """保存工具表、环境创建方式和系统提示词，供后续每条任务复用。"""
        self.tool_registry = tool_registry
        self.environment_factory = environment_factory or _default_environment_factory
        self.system_prompt = system_prompt
        self.include_initial_observation = bool(include_initial_observation)

    async def run(
        self,
        request: EpisodeRequest | Mapping[str, Any],
        policy: PolicyAdapter,
        config: HarnessConfig,
        observers: Sequence[RunObserver] = (),
    ) -> Trajectory:
        """完整跑完一条购物任务：问模型、查工具、操作商店、验奖励，最后一定释放环境。"""
        config = validate_config(config)
        if isinstance(request, Mapping):
            request = EpisodeRequest.from_mapping(request)
        if not isinstance(request, EpisodeRequest):
            raise TypeError("request must be EpisodeRequest or a mapping")
        if not callable(getattr(policy, "generate", None)):
            raise TypeError("policy must provide generate(request)")
        await _validate_policy_config(policy, config)

        trajectory_id = str(uuid4())
        created_at = _now()
        messages: list[dict[str, Any]] = []
        steps: list[TrajectoryStep] = []
        blocked_tool_calls: list[dict[str, Any]] = []
        truncations: list[dict[str, Any]] = []
        context_compactions: list[dict[str, Any]] = []
        context_turn_tokens: list[dict[str, Any]] = []
        initial_result: dict[str, Any] = {}
        terminal_result: dict[str, Any] = {}
        final_reward = 0.0
        done = False
        infrastructure_invalid = False
        reward_valid: bool | None = None
        sampling_invalid = True
        audit_final_reward = 0.0
        status = "running"
        termination_category = TerminationCategory.RUNNING
        termination_reason: str | None = None
        error: HarnessError | None = None
        release_error: HarnessError | None = None
        latest_observation = ""
        latest_observation_truncated = False
        assistant_turns = 0
        consecutive_guard_rejections = 0
        environment = self.environment_factory(config)
        environment_started = False

        try:
            initial = await environment.start(request)
            environment_started = True
            initial_result = initial.audit_dict()
            latest_observation = initial.observation
            if self.include_initial_observation:
                _require_structured_initial_observation(initial)
            messages = _initial_messages(
                request,
                _initial_instruction(request, initial, initial_result),
                self.system_prompt,
                initial_observation=(
                    initial.observation if self.include_initial_observation else None
                ),
            )
            await _emit(
                observers,
                "on_start",
                {
                    "trajectory_id": trajectory_id,
                    "task_id": request.task_id,
                    "attempt_index": request.attempt_index,
                },
            )

            while len(steps) < config.max_steps:
                if (
                    config.max_assistant_turns is not None
                    and assistant_turns >= config.max_assistant_turns
                ):
                    status = "max_assistant_turns"
                    termination_category = TerminationCategory.LIMIT_REACHED
                    termination_reason = status
                    break
                assistant_turns += 1
                model_request = ModelRequest(
                    messages=tuple(deepcopy(messages)),
                    tools=tuple(self.tool_registry.schemas),
                    max_output_tokens=(
                        config.context.generation_reserve_tokens
                        if config.context is not None
                        else None
                    ),
                    metadata={
                        "trajectory_id": trajectory_id,
                        "task_id": request.task_id,
                        "assistant_turn": assistant_turns,
                    },
                )
                raw_turn = await _maybe_await(policy.generate(model_request))
                try:
                    turn = (
                        raw_turn
                        if isinstance(raw_turn, AssistantTurn)
                        else AssistantTurn.from_mapping(raw_turn)
                    )
                except (TypeError, ValueError):
                    # 格式评测必须看到触发解析错误的原始 Assistant
                    # 回合，否则 malformed_arguments 只会留下一个异常名。
                    if isinstance(raw_turn, Mapping):
                        audit_turn = deepcopy(dict(raw_turn))
                        audit_turn["role"] = "assistant"
                        messages.append(audit_turn)
                    raise
                _record_context(policy, len(steps), context_turn_tokens, context_compactions)

                if len(turn.tool_calls) > 1:
                    if config.parallel_tool_call_policy == "reject":
                        messages.append(turn.to_message_dict())
                        status = "parallel_tool_calls"
                        termination_category = TerminationCategory.INVALID_ACTION
                        termination_reason = status
                        break
                    kept = turn.tool_calls[0]
                    dropped = turn.tool_calls[1:]
                    truncations.append(
                        {
                            "message_index": len(messages),
                            "kept_tool_call_id": kept.call_id,
                            "dropped_tool_calls": [
                                item.to_openai_dict() for item in dropped
                            ],
                        }
                    )
                    turn = AssistantTurn(
                        content=turn.content,
                        tool_calls=(kept,),
                        reasoning_content=turn.reasoning_content,
                        additional_fields=turn.additional_fields,
                    )

                if not turn.tool_calls:
                    messages.append(turn.to_message_dict())
                    status = "assistant_final"
                    termination_category = TerminationCategory.MODEL_STOPPED
                    termination_reason = status
                    break

                try:
                    call = self.tool_registry.parse_tool_call(turn.tool_calls[0])
                    resolved = self.tool_registry.resolve(call, latest_observation)
                except (TypeError, ValueError):
                    # unknown tool 或参数转换错误同样需要保留当轮输出。
                    messages.append(turn.to_message_dict())
                    raise
                if not resolved.allowed:
                    consecutive_guard_rejections += 1
                    blocked_tool_calls.append(
                        {
                            "step_index": len(steps),
                            "tool_call": call.to_openai_dict(),
                            "reason": resolved.guard_rejection,
                            "consecutive_count": consecutive_guard_rejections,
                            "latest_observation_truncated": latest_observation_truncated,
                        }
                    )
                    messages.append(turn.to_message_dict())
                    messages.append(
                        action_guard_tool_message(
                            call.to_openai_dict(),
                            str(resolved.guard_rejection),
                            latest_observation,
                        )
                    )
                    if consecutive_guard_rejections >= config.max_guard_rejections:
                        status = "invalid_action_limit"
                        termination_category = TerminationCategory.INVALID_ACTION
                        termination_reason = status
                        break
                    continue

                messages.append(turn.to_message_dict())
                consecutive_guard_rejections = 0
                if resolved.env_action is None:
                    note = str(call.arguments.get("note", ""))
                    visible_observation = (
                        "Reasoning recorded. Continue with one environment tool call."
                    )
                    result = EnvironmentResult(
                        observation=visible_observation,
                        raw_result={
                            "instruction": note,
                            "reward": 0.0,
                            "done": False,
                        },
                    )
                    projection = None
                    raw_observation = visible_observation
                else:
                    try:
                        result = await environment.execute(resolved.env_action)
                    except Exception as exc:
                        step_error = _environment_error_from_exception(exc)
                        steps.append(
                            TrajectoryStep(
                                step_index=len(steps),
                                tool_call=call,
                                env_action=resolved.env_action,
                                observation="",
                                reward=0.0,
                                done=False,
                                result={},
                                error=step_error,
                            )
                        )
                        error = step_error
                        sampling_invalid = True
                        final_reward = 0.0
                        infrastructure_invalid = _invalid_sample_error(step_error)
                        status = "error"
                        termination_category = (
                            TerminationCategory.PROTOCOL_ERROR
                            if step_error.category == ErrorCategory.PROTOCOL
                            else TerminationCategory.INFRASTRUCTURE_ERROR
                        )
                        termination_reason = f"tool_error:{step_error.error_type}"
                        break
                    raw_observation = result.observation
                    try:
                        visible_observation, raw_projection = await _project_observation(
                            policy,
                            call.name,
                            raw_observation,
                            call.arguments,
                        )
                        projection = ProjectionMetadata.from_mapping(raw_projection)
                    except Exception as exc:
                        step_error = HarnessError.from_exception(
                            ErrorCategory.PROTOCOL,
                            exc,
                            traceback_text=traceback.format_exc(),
                        )
                        steps.append(
                            TrajectoryStep(
                                step_index=len(steps),
                                tool_call=call,
                                env_action=resolved.env_action,
                                observation="",
                                raw_observation=raw_observation,
                                reward=result.reward,
                                done=result.done,
                                result=result.audit_dict(),
                                error=step_error,
                            )
                        )
                        error = step_error
                        infrastructure_invalid = True
                        sampling_invalid = True
                        final_reward = 0.0
                        status = "error"
                        termination_category = TerminationCategory.PROTOCOL_ERROR
                        termination_reason = "observation_projection_failed"
                        break

                step = TrajectoryStep(
                    step_index=len(steps),
                    tool_call=call,
                    env_action=resolved.env_action,
                    observation=visible_observation,
                    raw_observation=(
                        raw_observation
                        if projection is not None and visible_observation != raw_observation
                        else None
                    ),
                    projection=projection,
                    reward=result.reward,
                    done=result.done,
                    result=result.audit_dict(),
                )
                steps.append(step)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "content": visible_observation,
                    }
                )
                if resolved.env_action is not None:
                    latest_observation = visible_observation
                    latest_observation_truncated = bool(
                        projection.truncated if projection is not None else False
                    )
                await _emit(
                    observers,
                    "on_step",
                    {
                        "trajectory_id": trajectory_id,
                        "task_id": request.task_id,
                        "step": _public_step_event(step),
                    },
                )

                if result.done:
                    done = True
                    terminal_result = result.audit_dict()
                    audit_final_reward = result.reward
                    status = "done"
                    termination_category = TerminationCategory.ENVIRONMENT_DONE
                    reward_detail = terminal_result.get("reward_detail")
                    reward_reason = (
                        reward_detail.get("termination_reason")
                        if isinstance(reward_detail, Mapping)
                        else terminal_result.get("termination_reason")
                    )
                    termination_reason = str(reward_reason or "environment_done")
                    if config.validate_terminal_reward:
                        try:
                            validated_reward = validate_terminal_result(
                                terminal_result,
                                required_reward_version=config.required_reward_version,
                            )
                        except RewardContractError as exc:
                            error = HarnessError.from_exception(
                                ErrorCategory.PROTOCOL,
                                exc,
                                traceback_text=traceback.format_exc(),
                            )
                            reward_valid = False
                            sampling_invalid = True
                            final_reward = 0.0
                            infrastructure_invalid = True
                            status = "error"
                            termination_category = TerminationCategory.PROTOCOL_ERROR
                            termination_reason = "invalid_terminal_reward"
                        else:
                            reward_valid = validated_reward.get("reward_valid")
                            sampling_invalid = bool(
                                validated_reward.get("sampling_invalid", True)
                            )
                            final_reward = (
                                result.reward if not sampling_invalid else 0.0
                            )
                    else:
                        reward_valid = None
                        sampling_invalid = False
                        final_reward = result.reward
                    break
            else:
                status = "max_steps"
                termination_category = TerminationCategory.LIMIT_REACHED
                termination_reason = status

            if status == "running":
                status = "max_steps"
                termination_category = TerminationCategory.LIMIT_REACHED
                termination_reason = status
            if steps and not done:
                final_reward = 0.0
                sampling_invalid = True
        except Exception as exc:
            error = (
                _error_from_exception(exc)
                if environment_started
                else _environment_error_from_exception(exc)
            )
            sampling_invalid = True
            final_reward = 0.0
            infrastructure_invalid = _invalid_sample_error(error)
            status = (
                "invalid_action"
                if error.category == ErrorCategory.MODEL
                else "error"
            )
            if error.category == ErrorCategory.MODEL:
                termination_category = TerminationCategory.INVALID_ACTION
            elif error.category == ErrorCategory.PROTOCOL:
                termination_category = TerminationCategory.PROTOCOL_ERROR
            else:
                termination_category = TerminationCategory.INFRASTRUCTURE_ERROR
            termination_reason = f"{error.category.value}:{error.error_type}"
        finally:
            try:
                await environment.close()
            except Exception as exc:
                release_error = HarnessError.from_exception(
                    ErrorCategory.RELEASE,
                    exc,
                    traceback_text=traceback.format_exc(),
                )
                infrastructure_invalid = True
                sampling_invalid = True
                final_reward = 0.0
                status = "environment_release_failed"
                termination_category = TerminationCategory.INFRASTRUCTURE_ERROR
                termination_reason = "environment_release_failed"

        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            task_id=request.task_id,
            attempt_index=request.attempt_index,
            created_at=created_at,
            finished_at=_now(),
            status=status,
            termination_category=termination_category,
            termination_reason=termination_reason,
            messages=tuple(messages),
            steps=tuple(steps),
            blocked_tool_calls=tuple(blocked_tool_calls),
            tool_call_truncations=tuple(truncations),
            context_compactions=tuple(context_compactions),
            context_turn_tokens=tuple(context_turn_tokens),
            initial_result=initial_result,
            terminal_result=terminal_result,
            final_reward=final_reward,
            done=done,
            reward_valid=reward_valid,
            sampling_invalid=sampling_invalid,
            infrastructure_invalid=infrastructure_invalid,
            error=error,
            release_error=release_error,
            stage_metadata={
                "request": deepcopy(dict(request.metadata)),
                "harness": {
                    "audit_final_reward": audit_final_reward,
                    "tool_schema_version": self.tool_registry.version,
                    "tool_schema_fingerprint": self.tool_registry.fingerprint,
                    "required_environment_version": config.required_environment_version,
                    "required_reward_version": config.required_reward_version,
                    "max_steps": config.max_steps,
                    "max_assistant_turns": config.max_assistant_turns,
                    "parallel_tool_call_policy": config.parallel_tool_call_policy,
                },
            },
        )
        await _emit(observers, "on_finish", _public_finish_event(trajectory))
        return trajectory


def _default_environment_factory(config: HarnessConfig) -> EnvironmentAdapter:
    """根据 Harness 配置创建默认的 ShopSimulator 适配器。"""
    return ShopSimulatorEnvironmentAdapter(
        base_url=config.environment_base_url,
        timeout_s=config.environment_timeout_s,
        required_environment_version=config.required_environment_version,
    )


def _initial_instruction(
    request: EpisodeRequest,
    initial: EnvironmentResult,
    initial_result: Mapping[str, Any],
) -> str:
    """按优先级找出真正要交给模型的购物指令，避免把页面内容错当任务要求。"""
    raw_instruction = initial_result.get("instruction")
    return str(
        initial.instruction
        or request.instruction
        or (raw_instruction if isinstance(raw_instruction, str) else "")
    )


def _initial_messages(
    request: EpisodeRequest,
    instruction: str,
    system_prompt: str | None,
    *,
    initial_observation: str | None = None,
) -> list[dict[str, Any]]:
    """把系统提示、已有 prompt 和购物指令整理成模型能理解的消息列表。"""
    messages = [deepcopy(dict(item)) for item in request.prompt]
    if system_prompt and not any(item.get("role") == "system" for item in messages):
        messages.insert(0, {"role": "system", "content": system_prompt})
    user_instruction = instruction or request.instruction or ""
    if initial_observation:
        user_content = (
            "【购物任务】\n"
            f"{user_instruction}\n\n"
            "【初始 ShopSimulator observation（页面状态，不是额外需求）】\n"
            f"{initial_observation}"
        )
    else:
        user_content = user_instruction
    messages.append({"role": "user", "content": user_content})
    return messages


def _require_structured_initial_observation(initial: EnvironmentResult) -> None:
    """首轮页面必须来自 answer-free Observation v2 渲染，不能接受任意文本。"""

    if initial.observation_version != OBSERVATION_VERSION:
        raise StructuredObservationError(
            "SFT requires an initial observation_state rendered as Observation v2"
        )
    if not initial.observation.startswith(STRUCTURED_OBSERVATION_HEADER + "\n"):
        raise StructuredObservationError(
            "SFT initial observation is missing the Observation v2 header"
        )


async def _project_observation(
    policy: object,
    tool_name: str,
    observation: str,
    parameters: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None]:
    """如果策略支持页面压缩，就压缩环境页面；不支持时原样返回。"""
    projector = getattr(policy, "project_observation", None)
    if not callable(projector):
        return str(observation), None
    value = await _maybe_await(projector(tool_name, observation, parameters))
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("policy projector must return (visible_observation, metadata)")
    visible, metadata = value
    if metadata is not None and not isinstance(metadata, Mapping):
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        else:
            raise TypeError("projection metadata must be an object")
    return str(visible), metadata


def _record_context(
    policy: object,
    step_index: int,
    turn_tokens: list[dict[str, Any]],
    compactions: list[dict[str, Any]],
) -> None:
    """把本轮用了多少上下文、是否做过压缩记录下来，便于排查超长轨迹。"""
    raw_tokens = getattr(policy, "last_context_tokens", None)
    if raw_tokens is not None:
        turn_tokens.append({"step_index": step_index, "input_tokens": int(raw_tokens)})
    event = getattr(policy, "last_context_event", None)
    if isinstance(event, Mapping):
        compactions.append({"step_index": step_index, **deepcopy(dict(event))})


async def _validate_policy_config(policy: object, config: HarnessConfig) -> None:
    """在任务开始前确认策略真的能执行所选配置，防止参数看似开启但实际没生效。"""

    validator = getattr(policy, "validate_harness_config", None)
    if callable(validator):
        await _maybe_await(validator(config))
        return
    if config.context is not None or config.observation is not None:
        raise ConfigValidationError(
            "policy must provide validate_harness_config(config) when context or "
            "observation policies are enabled"
        )


def _public_step_event(step: TrajectoryStep) -> dict[str, Any]:
    """只提取外部日志可以看的步骤信息，主动去掉环境原始结果和隐藏字段。"""

    return {
        "step_index": step.step_index,
        "tool_call": step.tool_call.to_dict(),
        "env_action": step.env_action,
        "observation": step.observation,
        "projection": step.projection.to_dict() if step.projection else None,
        "done": bool(step.done),
        "error": _public_error(step.error),
    }


def _public_finish_event(trajectory: Trajectory) -> dict[str, Any]:
    """整理任务结束通知；完整原始轨迹仍只留在审计对象里。"""

    harness_metadata = trajectory.stage_metadata.get("harness")
    return {
        "schema_version": trajectory.schema_version,
        "trajectory_id": trajectory.trajectory_id,
        "task_id": trajectory.task_id,
        "attempt_index": trajectory.attempt_index,
        "created_at": trajectory.created_at,
        "finished_at": trajectory.finished_at,
        "status": trajectory.status,
        "termination_category": trajectory.termination_category.value,
        "termination_reason": trajectory.termination_reason,
        "messages": [_public_message(item) for item in trajectory.messages],
        "steps": [_public_step_event(item) for item in trajectory.steps],
        "blocked_tool_calls": deepcopy(list(trajectory.blocked_tool_calls)),
        "tool_call_truncations": deepcopy(list(trajectory.tool_call_truncations)),
        "context_compactions": deepcopy(list(trajectory.context_compactions)),
        "context_turn_tokens": deepcopy(list(trajectory.context_turn_tokens)),
        "final_reward": float(trajectory.final_reward),
        "done": bool(trajectory.done),
        "reward_valid": trajectory.reward_valid,
        "sampling_invalid": bool(trajectory.sampling_invalid),
        "infrastructure_invalid": bool(trajectory.infrastructure_invalid),
        "error": _public_error(trajectory.error),
        "release_error": _public_error(trajectory.release_error),
        "harness": _public_harness_metadata(harness_metadata),
    }


def _public_harness_metadata(value: object) -> dict[str, Any]:
    """从 Harness 元数据中挑出可以公开的配置项，未验证的审计奖励不会被带出去。"""
    if not isinstance(value, Mapping):
        return {}
    allowed = (
        "tool_schema_version",
        "tool_schema_fingerprint",
        "required_environment_version",
        "required_reward_version",
        "max_steps",
        "max_assistant_turns",
        "parallel_tool_call_policy",
    )
    return {name: deepcopy(value[name]) for name in allowed if name in value}


def _public_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """清洗一条对话消息，只保留角色、正文和工具调用等公开字段。"""
    return {
        name: deepcopy(message[name])
        for name in ("role", "content", "tool_calls", "tool_call_id", "name")
        if name in message
    }


def _public_error(error: HarnessError | None) -> dict[str, Any] | None:
    """把内部错误缩成适合日志展示的类别、名称和说明，不暴露 traceback。"""
    if error is None:
        return None
    return {
        "category": error.category.value,
        "type": error.error_type,
        "message": error.message,
    }


def _environment_error_from_exception(exc: BaseException) -> HarnessError:
    """把环境启动或执行时的异常分成协议问题、基础设施问题或一般环境问题。"""
    if isinstance(
        exc,
        (ShopProtocolError, StructuredObservationError, ValueError, TypeError, KeyError),
    ):
        category = ErrorCategory.PROTOCOL
    elif isinstance(
        exc,
        (ShopHttpError, ShopEnvironmentError, TimeoutError, OSError),
    ):
        category = ErrorCategory.INFRASTRUCTURE
    else:
        category = ErrorCategory.ENVIRONMENT
    return HarnessError.from_exception(
        category,
        exc,
        traceback_text=traceback.format_exc(),
    )


def _error_from_exception(exc: BaseException) -> HarnessError:
    """把模型、策略和通用运行异常归到稳定类别，阶段层不用猜异常文本。"""
    if isinstance(
        exc,
        (RewardContractError, ShopProtocolError, StructuredObservationError),
    ):
        category = ErrorCategory.PROTOCOL
    elif isinstance(exc, (ToolRegistryError, ValueError, TypeError, KeyError)):
        category = ErrorCategory.MODEL
    elif isinstance(exc, ContextBudgetError):
        category = ErrorCategory.POLICY
    elif isinstance(exc, (ShopHttpError, ShopEnvironmentError, TimeoutError, OSError)):
        category = ErrorCategory.INFRASTRUCTURE
    else:
        category = ErrorCategory.INTERNAL
    return HarnessError.from_exception(
        category,
        exc,
        traceback_text=traceback.format_exc(),
    )


def _invalid_sample_error(error: HarnessError) -> bool:
    """判断某类错误是否意味着这条采样不能用于训练。"""
    return error.category in {
        ErrorCategory.POLICY,
        ErrorCategory.ENVIRONMENT,
        ErrorCategory.PROTOCOL,
        ErrorCategory.INFRASTRUCTURE,
        ErrorCategory.RELEASE,
        ErrorCategory.INTERNAL,
    }


async def _emit(
    observers: Sequence[RunObserver],
    method_name: str,
    event: Mapping[str, Any],
) -> None:
    """依次通知所有观察者，并同时兼容普通函数和异步函数。"""
    for observer in observers:
        method = getattr(observer, method_name, None)
        if callable(method):
            await _maybe_await(method(deepcopy(dict(event))))


async def _maybe_await(value: Any) -> Any:
    """如果返回值需要 await 就等待，否则直接返回，统一同步与异步调用。"""
    return await value if inspect.isawaitable(value) else value


def _now() -> str:
    """生成带时区的当前 UTC 时间字符串，用于记录任务开始和结束时间。"""
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "EnvironmentAdapter",
    "EpisodeRunner",
    "PolicyAdapter",
    "RunObserver",
]
