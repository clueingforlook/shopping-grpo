"""编排 WLX SFT 的校准采样和正式采样，并处理并发、重试与续采。"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from wlx_harness_core.wlx_config import HarnessConfig
from wlx_harness_core.wlx_contracts import Trajectory
from wlx_harness_core.wlx_runner import EpisodeRunner
from wlx_harness_core.wlx_sft_contracts import (
    AttemptEnvelope,
    SamplingConfig,
    SamplingMode,
    SftDisposition,
    SftTask,
)
from wlx_harness_core.wlx_sft_metrics import aggregate_token_metrics
from wlx_harness_core.wlx_sft_outcomes import (
    classify_attempt,
    is_strict_gold,
    trajectory_mapping,
)
from wlx_harness_core.wlx_sft_storage import RawTrajectoryStore


@dataclass(frozen=True)
class AttemptExecution:
    """保存一次 Core 执行结果，以及策略逐轮积累的模型 API Usage。"""

    trajectory: Trajectory | Mapping[str, Any]
    request_usage_history: tuple[Mapping[str, Any], ...] = ()
    raw_trajectory_tokens: int | None = None
    metric_warnings: tuple[str, ...] = ()


class AttemptExecutor(Protocol):
    """约定采样器如何请求 Core 跑一条独立轨迹。"""

    async def execute(
        self,
        task: SftTask,
        attempt_index: int,
        technical_retry_index: int,
    ) -> AttemptExecution:
        """运行一次任务；每次调用必须使用独立的模型客户端和环境 Session。"""

        ...


class BatchSafetyPause(RuntimeError):
    """表示环境状态可能不明确，整批采样必须停下，由人检查后再续采。"""

    pass


class CoreAttemptExecutor:
    """把 SFT 调度器接到 WLX EpisodeRunner，而不是原项目采样 Harness。"""

    def __init__(
        self,
        *,
        runner: EpisodeRunner,
        harness_config: HarnessConfig,
        policy_factory: Callable[
            [SftTask, int, int], object | Awaitable[object]
        ],
    ) -> None:
        """保存 Core、统一配置和策略工厂；策略工厂每次都要返回新实例。"""

        self.runner = runner
        self.harness_config = harness_config
        self.policy_factory = policy_factory

    async def execute(
        self,
        task: SftTask,
        attempt_index: int,
        technical_retry_index: int,
    ) -> AttemptExecution:
        """创建独立策略并让 Core 跑完一条轨迹，然后取出每轮 Token Usage。"""

        policy = self.policy_factory(task, attempt_index, technical_retry_index)
        if inspect.isawaitable(policy):
            policy = await policy
        trajectory = await self.runner.run(
            task.to_episode_request(attempt_index),
            policy,
            self.harness_config,
        )
        history = _policy_usage_history(policy)
        raw_tokens, token_warning = _policy_raw_trajectory_tokens(
            policy,
            trajectory,
            self.runner.tool_registry.schemas,
        )
        return AttemptExecution(
            trajectory=trajectory,
            request_usage_history=tuple(history),
            raw_trajectory_tokens=raw_tokens,
            metric_warnings=((token_warning,) if token_warning else ()),
        )


class SftSamplingScheduler:
    """控制多个任务的派发顺序，同一任务仍然严格按尝试编号依次运行。"""

    def __init__(self, config: SamplingConfig) -> None:
        """保存采样模式、有效尝试数、并发和技术重试上限。"""

        if not isinstance(config, SamplingConfig):
            raise TypeError("config 必须是 SamplingConfig")
        self.config = config

    async def collect(
        self,
        tasks: Sequence[SftTask],
        *,
        executor: AttemptExecutor,
        store: RawTrajectoryStore,
    ) -> dict[str, Any]:
        """并发采不同任务、串行采同一任务，并把每次结果立即追加到 raw.jsonl。"""

        _validate_unique_tasks(tasks)
        semaphore = asyncio.Semaphore(self.config.concurrency)
        stop_event = asyncio.Event()
        state_lock = asyncio.Lock()
        state = {
            "new_rows": 0,
            "new_valid_attempts": 0,
            "new_gold": 0,
            "technical_failures": 0,
            "paused_reason": None,
        }
        existing_gold = _existing_gold_count(store)

        async def run_task(task: SftTask) -> None:
            """让一个任务按逻辑 attempt_index 运行，结束后才会处理它的下一次。"""

            await self._collect_task(
                task,
                executor=executor,
                store=store,
                semaphore=semaphore,
                stop_event=stop_event,
                state=state,
                state_lock=state_lock,
                existing_gold=existing_gold,
            )

        results = await asyncio.gather(
            *(run_task(task) for task in tasks),
            return_exceptions=True,
        )
        unexpected = [item for item in results if isinstance(item, BaseException)]
        if unexpected and state["paused_reason"] is None:
            raise unexpected[0]
        if state["paused_reason"] is not None:
            raise BatchSafetyPause(str(state["paused_reason"]))
        return {
            "mode": self.config.mode.value,
            "tasks_considered": len(tasks),
            **state,
            "existing_gold_before_run": existing_gold,
            "raw_path": str(store.path),
        }

    async def _collect_task(
        self,
        task: SftTask,
        *,
        executor: AttemptExecutor,
        store: RawTrajectoryStore,
        semaphore: asyncio.Semaphore,
        stop_event: asyncio.Event,
        state: dict[str, Any],
        state_lock: asyncio.Lock,
        existing_gold: int,
    ) -> None:
        """执行单个任务的三次有效尝试；正式模式拿到 Gold 后立刻停止该任务。"""

        prior_rows = store.decisions_for_task(task.task_id)
        if self.config.mode == SamplingMode.FORMAL and _rows_have_gold(prior_rows):
            return
        completed = {
            int(row.get("attempt_index", 0))
            for row in prior_rows
            if row.get("attempt_valid") is True
        }
        for attempt_index in range(self.config.valid_attempts_per_task):
            if attempt_index in completed:
                if (
                    self.config.mode == SamplingMode.FORMAL
                    and _attempt_is_gold(prior_rows, attempt_index)
                ):
                    return
                continue
            if stop_event.is_set():
                return
            valid, gold = await self._run_logical_attempt(
                task,
                attempt_index,
                executor=executor,
                store=store,
                semaphore=semaphore,
                stop_event=stop_event,
                state=state,
                state_lock=state_lock,
                existing_gold=existing_gold,
            )
            if not valid:
                return
            if self.config.mode == SamplingMode.FORMAL and gold and self.config.stop_after_gold:
                return

    async def _run_logical_attempt(
        self,
        task: SftTask,
        attempt_index: int,
        *,
        executor: AttemptExecutor,
        store: RawTrajectoryStore,
        semaphore: asyncio.Semaphore,
        stop_event: asyncio.Event,
        state: dict[str, Any],
        state_lock: asyncio.Lock,
        existing_gold: int,
    ) -> tuple[bool, bool]:
        """技术故障时复用同一个 attempt_index，直到得到有效结果或用完重试额度。"""

        first_retry = store.technical_retry_count(task.task_id, attempt_index)
        maximum_runs = self.config.max_technical_retries_per_attempt + 1
        if first_retry >= maximum_runs:
            return False, False
        for technical_retry_index in range(first_retry, maximum_runs):
            if stop_event.is_set():
                return False, False
            async with semaphore:
                if stop_event.is_set():
                    return False, False
                try:
                    execution = await executor.execute(
                        task,
                        attempt_index,
                        technical_retry_index,
                    )
                except Exception as exc:
                    execution = AttemptExecution(
                        trajectory=_exception_trajectory(task, attempt_index, exc)
                    )
            trajectory = trajectory_mapping(execution.trajectory)
            decision = classify_attempt(trajectory)
            metrics = aggregate_token_metrics(
                trajectory,
                request_usage_history=execution.request_usage_history,
            )
            if execution.raw_trajectory_tokens is not None:
                metrics["raw_trajectory_tokens"] = int(execution.raw_trajectory_tokens)
            if execution.metric_warnings:
                metrics["metric_warnings"] = list(execution.metric_warnings)
            envelope = AttemptEnvelope(
                trajectory=trajectory,
                decision=decision,
                technical_retry_index=technical_retry_index,
                token_metrics=metrics,
            )
            store.append(envelope)
            async with state_lock:
                state["new_rows"] += 1
                if decision.attempt_valid:
                    state["new_valid_attempts"] += 1
                else:
                    state["technical_failures"] += 1
                if decision.sft_disposition == SftDisposition.ACCEPTED_GOLD:
                    state["new_gold"] += 1
                if _requires_batch_pause(trajectory):
                    state["paused_reason"] = (
                        f"task_id={task.task_id} 的环境状态可能不明确；"
                        "已保存现场并停止继续派发，请先检查 ShopSimulator"
                    )
                    stop_event.set()
                target = self.config.target_gold_trajectories
                if target is not None and existing_gold + state["new_gold"] >= target:
                    stop_event.set()
            if decision.attempt_valid:
                return True, decision.sft_disposition == SftDisposition.ACCEPTED_GOLD
            if stop_event.is_set():
                return False, False
        return False, False


def _policy_usage_history(policy: object) -> list[Mapping[str, Any]]:
    """从新策略或它包着的旧客户端读取整条轨迹的逐轮 API Usage。"""

    direct = getattr(policy, "request_usage_history", None)
    if isinstance(direct, (list, tuple)):
        return [dict(item) for item in direct if isinstance(item, Mapping)]
    client = getattr(policy, "client", None)
    nested = getattr(client, "request_usage_history", None)
    if isinstance(nested, (list, tuple)):
        return [dict(item) for item in nested if isinstance(item, Mapping)]
    return []


def _policy_raw_trajectory_tokens(
    policy: object,
    trajectory: Trajectory | Mapping[str, Any],
    tools: object,
) -> tuple[int | None, str | None]:
    """用 Teacher tokenizer 统计最终完整消息轨迹；计数器不可用时明确返回空值。"""

    counter = getattr(policy, "count_chat_tokens", None)
    if not callable(counter):
        client = getattr(policy, "client", None)
        counter = getattr(client, "token_counter", None)
    if not callable(counter):
        return None, "teacher_chat_token_counter_unavailable"
    row = trajectory_mapping(trajectory)
    try:
        return int(counter(row.get("messages") or [], tools)), None
    except Exception as exc:
        return None, f"raw_trajectory_token_count_failed:{exc.__class__.__name__}"


def _validate_unique_tasks(tasks: Sequence[SftTask]) -> None:
    """拒绝同一批计划里重复出现 task_id，避免两个协程同时写同一道题。"""

    task_ids = [int(task.task_id) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("采样计划中存在重复 task_id")


def _existing_gold_count(store: RawTrajectoryStore) -> int:
    """统计 raw.jsonl 中已有多少个不同任务获得过严格 Gold。"""

    return len(
        {
            int(row["task_id"])
            for row in store.rows()
            if _row_is_strict_gold(row)
        }
    )


def _rows_have_gold(rows: Sequence[Mapping[str, Any]]) -> bool:
    """判断一个任务的历史记录中是否已经存在严格 Gold 轨迹。"""

    return any(
        _row_is_strict_gold(row)
        for row in rows
    )


def _attempt_is_gold(rows: Sequence[Mapping[str, Any]], attempt_index: int) -> bool:
    """判断指定逻辑尝试是否已经得到严格 Gold 结果。"""

    return any(
        int(row.get("attempt_index", -1)) == int(attempt_index)
        and _row_is_strict_gold(row)
        for row in rows
    )


def _row_is_strict_gold(row: Mapping[str, Any]) -> bool:
    """重新分类历史 raw 行，防止旧的脏 Gold 标签让断点续采提前停止。"""

    return is_strict_gold(classify_attempt(row))


def _requires_batch_pause(trajectory: Mapping[str, Any]) -> bool:
    """识别环境动作可能已经执行但回包不确定的情况，此时不能自动接着采。"""

    if trajectory.get("release_error") or trajectory.get("status") == "environment_release_failed":
        return True
    reason = str(trajectory.get("termination_reason") or "")
    return reason.startswith("tool_error:")


def _exception_trajectory(
    task: SftTask,
    attempt_index: int,
    exc: BaseException,
) -> dict[str, Any]:
    """把 Core 外层的异常包装成可审计 raw 行，同时避免把密钥带进错误文字。"""

    now = datetime.now(timezone.utc).isoformat()
    return {
        "trajectory_id": str(uuid4()),
        "task_id": task.task_id,
        "attempt_index": int(attempt_index),
        "created_at": now,
        "finished_at": now,
        "status": "error",
        "termination_category": "infrastructure_error",
        "termination_reason": "executor_exception",
        "messages": [],
        "steps": [],
        "initial_result": {},
        "terminal_result": {},
        "final_reward": 0.0,
        "done": False,
        "reward_valid": None,
        "sampling_invalid": True,
        "infrastructure_invalid": True,
        "error": {
            "category": "infrastructure",
            "type": exc.__class__.__name__,
            "message": _safe_error_message(exc),
        },
        "release_error": None,
        "stage_metadata": {},
    }


def _safe_error_message(exc: BaseException) -> str:
    """清掉常见 Authorization 和 Key 片段，只保留最多 500 字符的诊断信息。"""

    message = str(exc)
    message = re.sub(r"(?i)bearer\s+[a-z0-9._-]+", "Bearer [REDACTED]", message)
    message = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)\S+", r"\1[REDACTED]", message)
    return message[:500]


__all__ = [
    "AttemptExecution",
    "AttemptExecutor",
    "BatchSafetyPause",
    "CoreAttemptExecutor",
    "SftSamplingScheduler",
]
