"""单模型评测批次：逐题运行、立即落盘并支持安全续跑。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from shopping_grpo.harness.config import HarnessConfig
from shopping_grpo.harness.contracts import EpisodeRequest
from shopping_grpo.harness.sft_storage import append_jsonl_row, read_jsonl
from shopping_grpo.harness.trajectory_evaluator import evaluate_trajectory


EVALUATION_BATCH_VERSION = "wlx-evaluation-batch-v1"
EVALUATION_MANIFEST_VERSION = "wlx-evaluation-run-manifest-v1"


class EvaluationBatchContractError(ValueError):
    """已有输出与本次 Run 不兼容，继续会混合两套评测。"""


class EvaluationBatchSafetyPause(RuntimeError):
    """环境释放状态不确定，当前批次不能继续派发新任务。"""


class EvaluationBatchPaths:
    """一个单模型 Run 的三个固定输出位置。"""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.trajectories = self.output_dir / "trajectories.jsonl"
        self.task_results = self.output_dir / "task-results.jsonl"
        self.manifest = self.output_dir / "run-manifest.json"


async def run_evaluation_batch(
    *,
    task_ids: Sequence[int],
    output_dir: str | Path,
    runner: object,
    harness_config: HarnessConfig,
    policy_factory: Callable[[int], object | Awaitable[object]],
    run_contract: Mapping[str, Any],
    concurrency: int = 1,
    progress_callback: (
        Callable[[Mapping[str, Any]], object | Awaitable[object]] | None
    ) = None,
) -> dict[str, Any]:
    """对一个模型运行一批任务，每条轨迹和评测结论分别追加到 JSONL。"""

    selected = _validated_task_ids(task_ids)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise ValueError("concurrency 必须是正整数")
    if not callable(getattr(runner, "run", None)):
        raise TypeError("runner 必须提供异步 run()")
    if not callable(policy_factory):
        raise TypeError("policy_factory 必须可调用")

    paths = EvaluationBatchPaths(output_dir)
    contract = _normalise_contract(run_contract, selected)
    manifest = _prepare_manifest(paths, contract)
    trajectories = _rows_by_task(paths.trajectories, "轨迹")
    results = _rows_by_task(paths.task_results, "逐题评测")
    selected_set = set(selected)
    unexpected = sorted((set(trajectories) | set(results)).difference(selected_set))
    if unexpected:
        raise EvaluationBatchContractError(
            f"已有输出含本 Run 之外的 task_id：{unexpected[:10]}"
        )
    orphan_results = sorted(set(results).difference(trajectories))
    if orphan_results:
        raise EvaluationBatchContractError(
            f"逐题结果缺少对应完整轨迹：{orphan_results[:10]}"
        )

    # 如果上次恰好在“轨迹落盘”和“评测落盘”之间中断，直接由轨迹补齐结果，
    # 不能重新调用模型造成同一道题出现第二条 rollout。
    reconciled = 0
    for task_id in selected:
        if task_id in trajectories and task_id not in results:
            result = evaluate_trajectory(trajectories[task_id]).to_dict()
            append_jsonl_row(paths.task_results, result)
            results[task_id] = result
            reconciled += 1

    completed = set(trajectories).intersection(results)
    pending = [task_id for task_id in selected if task_id not in completed]
    progress = {
        "event": "start",
        "total": len(selected),
        "completed": len(completed),
        "format_correct": sum(
            row.get("format_correct") is True for row in results.values()
        ),
        "purchase_correct": sum(
            row.get("purchase_correct") is True for row in results.values()
        ),
        "trajectory_invalid": sum(
            row.get("trajectory_valid") is False for row in results.values()
        ),
        "task_id": None,
    }
    await _notify_progress(progress_callback, progress)
    queue: asyncio.Queue[int] = asyncio.Queue()
    for task_id in pending:
        queue.put_nowait(task_id)
    stop_event = asyncio.Event()
    write_lock = asyncio.Lock()
    state = {
        "new_trajectories": 0,
        "new_results": 0,
        "safety_pause_reason": None,
    }

    async def worker() -> None:
        while not stop_event.is_set():
            try:
                task_id = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if stop_event.is_set():
                queue.task_done()
                return
            try:
                policy = policy_factory(task_id)
                if inspect.isawaitable(policy):
                    policy = await policy
                trajectory = await runner.run(
                    EpisodeRequest(
                        task_id=task_id,
                        attempt_index=0,
                        metadata={
                            "evaluation_batch_version": EVALUATION_BATCH_VERSION,
                            "run_contract_sha256": manifest["run_contract_sha256"],
                        },
                    ),
                    policy,
                    harness_config,
                )
                trajectory_row = _trajectory_mapping(trajectory)
                if int(trajectory_row.get("task_id", -1)) != task_id:
                    raise EvaluationBatchContractError(
                        f"Runner 返回了错误 task_id：预期 {task_id}，"
                        f"实际 {trajectory_row.get('task_id')!r}"
                    )
                result_row = evaluate_trajectory(trajectory_row).to_dict()
                async with write_lock:
                    append_jsonl_row(paths.trajectories, trajectory_row)
                    state["new_trajectories"] += 1
                    append_jsonl_row(paths.task_results, result_row)
                    state["new_results"] += 1
                    progress["event"] = "advance"
                    progress["completed"] += 1
                    progress["format_correct"] += int(
                        result_row.get("format_correct") is True
                    )
                    progress["purchase_correct"] += int(
                        result_row.get("purchase_correct") is True
                    )
                    progress["trajectory_invalid"] += int(
                        result_row.get("trajectory_valid") is False
                    )
                    progress["task_id"] = task_id
                    await _notify_progress(progress_callback, progress)
                    if _requires_safety_pause(trajectory_row):
                        state["safety_pause_reason"] = (
                            f"task_id={task_id} 的环境释放或动作回包状态不确定"
                        )
                        stop_event.set()
            except BaseException:
                stop_event.set()
                raise
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    gathered = await asyncio.gather(*workers, return_exceptions=True)
    failures = [item for item in gathered if isinstance(item, BaseException)]
    if failures:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        raise failures[0]
    if state["safety_pause_reason"] is not None:
        raise EvaluationBatchSafetyPause(str(state["safety_pause_reason"]))

    final_trajectories = _rows_by_task(paths.trajectories, "轨迹")
    final_results = _rows_by_task(paths.task_results, "逐题评测")
    final_completed = set(final_trajectories).intersection(final_results)
    return {
        "batch_version": EVALUATION_BATCH_VERSION,
        "tasks_selected": len(selected),
        "completed_before_run": len(completed),
        "reconciled_results": reconciled,
        **state,
        "completed_after_run": len(final_completed),
        "remaining": len(selected_set.difference(final_completed)),
        "output_dir": str(paths.output_dir),
        "trajectories": str(paths.trajectories),
        "task_results": str(paths.task_results),
        "manifest": str(paths.manifest),
    }


def _validated_task_ids(task_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(task_ids, (str, bytes)) or not isinstance(task_ids, Sequence):
        raise TypeError("task_ids 必须是整数序列")
    values = []
    for task_id in task_ids:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("task_ids 只能包含非负整数")
        values.append(int(task_id))
    if not values:
        raise ValueError("评测任务集不能为空")
    if len(values) != len(set(values)):
        raise ValueError("评测任务集含重复 task_id")
    return tuple(values)


def _normalise_contract(
    value: Mapping[str, Any],
    task_ids: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("run_contract 必须是对象")
    contract = deepcopy(dict(value))
    contract["task_ids_sha256"] = hashlib.sha256(
        json.dumps(list(task_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    contract["task_count"] = len(task_ids)
    # 同时检查它确实能稳定序列化，避免运行到一半才发现 manifest 写不了。
    json.dumps(contract, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return contract


def _prepare_manifest(
    paths: EvaluationBatchPaths,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    encoded_contract = json.dumps(
        dict(contract),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    contract_sha = hashlib.sha256(encoded_contract.encode("utf-8")).hexdigest()
    if paths.manifest.exists():
        existing = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if not isinstance(existing, Mapping):
            raise EvaluationBatchContractError("已有 Run manifest 不是 JSON 对象")
        if (
            existing.get("schema_version") != EVALUATION_MANIFEST_VERSION
            or existing.get("run_contract_sha256") != contract_sha
            or existing.get("run_contract") != dict(contract)
        ):
            raise EvaluationBatchContractError(
                "已有 Run manifest 与本次模型、任务或 Harness 配置不一致"
            )
        return dict(existing)
    if _nonempty(paths.trajectories) or _nonempty(paths.task_results):
        raise EvaluationBatchContractError(
            "已有逐条输出但缺少 Run manifest，拒绝猜测来源后继续"
        )
    manifest = {
        "schema_version": EVALUATION_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_contract_sha256": contract_sha,
        "run_contract": deepcopy(dict(contract)),
        "outputs": {
            "trajectories": paths.trajectories.name,
            "task_results": paths.task_results.name,
        },
    }
    _atomic_write_json(paths.manifest, manifest)
    return manifest


def _rows_by_task(path: Path, label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise EvaluationBatchContractError(f"{label}存在无效 task_id")
        if task_id in result:
            raise EvaluationBatchContractError(f"{label}中 task_id={task_id} 重复")
        result[task_id] = row
    return result


def _trajectory_mapping(value: object) -> dict[str, Any]:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise TypeError("Runner 轨迹必须是对象或提供 to_dict()")
    return deepcopy(dict(value))


def _requires_safety_pause(trajectory: Mapping[str, Any]) -> bool:
    if trajectory.get("release_error"):
        return True
    return str(trajectory.get("termination_reason") or "").startswith("tool_error:")


async def _notify_progress(
    callback: Callable[[Mapping[str, Any]], object | Awaitable[object]] | None,
    progress: Mapping[str, Any],
) -> None:
    if callback is None:
        return
    result = callback(deepcopy(dict(progress)))
    if inspect.isawaitable(result):
        await result


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    text = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


__all__ = [
    "EvaluationBatchContractError",
    "EvaluationBatchPaths",
    "EvaluationBatchSafetyPause",
    "EVALUATION_BATCH_VERSION",
    "EVALUATION_MANIFEST_VERSION",
    "run_evaluation_batch",
]
