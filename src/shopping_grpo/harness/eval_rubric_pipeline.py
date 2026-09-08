"""Rubric 批量生成、断点续跑、复核队列和最终冻结。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from shopping_grpo.harness.eval_rubric import (
    freeze_rubric_bundle,
    validate_rubric_bundle,
)
from shopping_grpo.harness.eval_rubric_generator import (
    DeepSeekRubricClient,
    RubricGenerationError,
    RubricTask,
    RUBRIC_GENERATOR_VERSION,
    RUBRIC_PROMPT_VERSION,
    RUBRIC_SYSTEM_PROMPT_SHA256,
    audit_rubric_candidate_result,
    generate_rubric_candidate,
    normalize_generated_rubric_items,
)
from shopping_grpo.harness.sft_storage import (
    append_jsonl_row,
    file_sha256,
    read_jsonl,
    write_jsonl,
)


RUBRIC_BATCH_VERSION = "wlx-rubric-batch-v1"
RUBRIC_MANIFEST_VERSION = "wlx-rubric-generation-manifest-v1"
RUBRIC_FINALIZATION_VERSION = "wlx-rubric-finalization-v1"


class RubricBatchContractError(ValueError):
    """已有生成目录与当前任务、模型或 Prompt 不一致。"""


class RubricBatchPaths:
    """Rubric 生成目录中的固定产物。"""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.manifest = self.output_dir / "rubric-generation-manifest.json"
        self.attempts = self.output_dir / "rubric-attempts.jsonl"
        self.candidates = self.output_dir / "rubric-candidates.jsonl"
        self.auto_frozen = self.output_dir / "rubrics-auto-frozen.jsonl"
        self.review_queue = self.output_dir / "rubric-review-queue.jsonl"
        self.review_decisions_template = (
            self.output_dir / "rubric-review-decisions-template.jsonl"
        )
        self.summary = self.output_dir / "rubric-generation-summary.json"
        self.final_rubrics = self.output_dir / "rubrics.jsonl"
        self.applied_reviews = self.output_dir / "rubric-review-decisions-applied.jsonl"
        self.finalization_manifest = (
            self.output_dir / "rubric-finalization-manifest.json"
        )


def load_rubric_tasks(
    tasks_path: str | Path,
    *,
    trajectories_path: str | Path | None = None,
    limit: int | None = None,
) -> list[RubricTask]:
    """读取任务；缺少 instruction 时只从轨迹初始请求补齐。"""

    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")
    instruction_by_task = (
        _trajectory_instructions(Path(trajectories_path))
        if trajectories_path is not None
        else {}
    )
    result: list[RubricTask] = []
    seen: set[int] = set()
    for row in read_jsonl(tasks_path):
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("任务文件含无效 task_id")
        if task_id in seen:
            raise ValueError(f"任务文件中 task_id={task_id} 重复")
        seen.add(task_id)
        direct = _optional_text(row.get("instruction") or row.get("instruction_text"))
        from_trajectory = instruction_by_task.get(task_id)
        if direct and from_trajectory and _normalize_instruction(direct) != _normalize_instruction(
            from_trajectory
        ):
            raise ValueError(f"task_id={task_id} 的任务文本与轨迹初始请求不一致")
        instruction = direct or from_trajectory
        if not instruction:
            raise ValueError(
                f"task_id={task_id} 缺少 instruction；请传 --trajectories 补齐初始需求"
            )
        result.append(RubricTask(task_id=task_id, instruction=instruction))
        if limit is not None and len(result) >= limit:
            break
    if not result:
        raise ValueError("Rubric 任务集为空")
    return result


def run_rubric_generation(
    *,
    tasks: Sequence[RubricTask],
    output_dir: str | Path,
    client: DeepSeekRubricClient,
    rubric_set_version: str,
    source_contract: Mapping[str, Any],
    concurrency: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], object] | None = None,
) -> dict[str, Any]:
    """每个未完成任务调用一次 DeepSeek；技术失败可安全续跑。"""

    selected = _validate_tasks(tasks)
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    version = str(rubric_set_version).strip()
    if not version:
        raise ValueError("rubric_set_version must be non-empty")
    paths = RubricBatchPaths(output_dir)
    contract = {
        "batch_version": RUBRIC_BATCH_VERSION,
        "generator_version": RUBRIC_GENERATOR_VERSION,
        "prompt_version": RUBRIC_PROMPT_VERSION,
        "prompt_sha256": RUBRIC_SYSTEM_PROMPT_SHA256,
        "rubric_set_version": version,
        "task_ids": [task.task_id for task in selected],
        "task_instructions_sha256": _task_instructions_sha256(selected),
        "model": client.model,
        "base_url": client.base_url,
        "max_tokens": client.max_tokens,
        "response_format_json": client.response_format_json,
        "source": deepcopy(dict(source_contract)),
    }
    manifest = _prepare_generation_manifest(paths, contract)
    successful = _reaudit_successful_attempts(
        _latest_successful_attempts(paths.attempts, selected),
        selected,
        rubric_set_version=version,
    )
    pending = [task for task in selected if task.task_id not in successful]
    progress = {
        "event": "start",
        "total": len(selected),
        "completed": len(successful),
        "review_required": sum(
            attempt["candidate"].get("review_required") is True
            for attempt in successful.values()
        ),
        "technical_errors": 0,
        "task_id": None,
    }
    _notify(progress_callback, progress)

    def generate(task: RubricTask) -> dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        try:
            candidate = generate_rubric_candidate(
                task,
                client=client,
                rubric_set_version=version,
            )
            return {
                "schema_version": "wlx-rubric-attempt-v1",
                "task_id": task.task_id,
                "instruction_sha256": task.instruction_sha256,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "candidate_ready",
                "candidate": candidate,
                "error": None,
            }
        except Exception as exc:  # 每题技术失败要落盘，不能丢失批次进度
            return {
                "schema_version": "wlx-rubric-attempt-v1",
                "task_id": task.task_id,
                "instruction_sha256": task.instruction_sha256,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "technical_error",
                "candidate": None,
                "error": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(generate, task): task for task in pending}
            for future in as_completed(futures):
                attempt = future.result()
                append_jsonl_row(paths.attempts, attempt)
                if attempt["status"] == "candidate_ready":
                    successful[int(attempt["task_id"])] = attempt
                    progress["completed"] += 1
                    progress["review_required"] += int(
                        attempt["candidate"].get("review_required") is True
                    )
                else:
                    progress["technical_errors"] += 1
                progress["event"] = "advance"
                progress["task_id"] = int(attempt["task_id"])
                _notify(progress_callback, progress)

    remaining = [task.task_id for task in selected if task.task_id not in successful]
    if remaining:
        return {
            "batch_version": RUBRIC_BATCH_VERSION,
            "complete": False,
            "tasks_selected": len(selected),
            "completed": len(successful),
            "remaining": len(remaining),
            "remaining_task_ids": remaining,
            "output_dir": str(paths.output_dir),
            "attempts": str(paths.attempts),
            "manifest": str(paths.manifest),
        }

    ordered_attempts = [successful[task.task_id] for task in selected]
    if all(
        path.exists()
        for path in (
            paths.candidates,
            paths.auto_frozen,
            paths.review_queue,
            paths.review_decisions_template,
            paths.summary,
        )
    ):
        return _generation_result(paths, ordered_attempts, manifest, reused=True)
    _ensure_none_exist(
        paths.candidates,
        paths.auto_frozen,
        paths.review_queue,
        paths.review_decisions_template,
        paths.summary,
    )
    candidate_rows = [deepcopy(attempt["candidate"]) for attempt in ordered_attempts]
    auto_rows = [
        deepcopy(candidate["rubric"])
        for candidate in candidate_rows
        if candidate.get("review_required") is not True
    ]
    review_rows = [
        {
            "schema_version": "wlx-rubric-review-queue-v1",
            "task_id": candidate["task_id"],
            "instruction": candidate["rubric"]["instruction"],
            "review_reasons": deepcopy(candidate.get("review_reasons") or []),
            "proposed_rubric": deepcopy(candidate["rubric"]),
            "rule_signals": deepcopy(candidate["rule_signals"]),
            "llm_metadata": deepcopy(candidate["llm_metadata"]),
        }
        for candidate in candidate_rows
        if candidate.get("review_required") is True
    ]
    write_jsonl(paths.candidates, candidate_rows)
    write_jsonl(paths.auto_frozen, auto_rows)
    write_jsonl(paths.review_queue, review_rows)
    write_jsonl(
        paths.review_decisions_template,
        (
            {
                "task_id": row["task_id"],
                "decision": "REQUIRED: approve or replace",
                "reviewer": "",
                "reason": "",
                "items": deepcopy(row["proposed_rubric"]["items"]),
            }
            for row in review_rows
        ),
    )
    summary = {
        "schema_version": "wlx-rubric-generation-summary-v1",
        "tasks": len(selected),
        "candidate_ready": len(candidate_rows),
        "auto_frozen": len(auto_rows),
        "review_required": len(review_rows),
        "rubric_set_version": version,
        "manifest_sha256": file_sha256(paths.manifest),
    }
    _write_new_json(paths.summary, summary)
    return _generation_result(paths, ordered_attempts, manifest, reused=False)


def finalize_rubric_set(
    *,
    generation_dir: str | Path,
    review_decisions_path: str | Path | None = None,
) -> dict[str, Any]:
    """应用人工决定并冻结一份覆盖全部任务、顺序稳定的 Rubric 集。"""

    paths = RubricBatchPaths(generation_dir)
    if paths.final_rubrics.exists() or paths.finalization_manifest.exists():
        raise FileExistsError(
            "最终 Rubric 已存在，拒绝覆盖；修改后请使用新版本目录"
        )
    manifest = _read_json_object(paths.manifest)
    contract = manifest.get("generation_contract")
    if not isinstance(contract, Mapping):
        raise RubricBatchContractError("生成 manifest 缺少 generation_contract")
    task_ids = contract.get("task_ids")
    if not isinstance(task_ids, list) or not task_ids:
        raise RubricBatchContractError("生成 manifest 缺少 task_ids")
    candidates = {int(row["task_id"]): row for row in read_jsonl(paths.candidates)}
    if set(candidates) != set(task_ids):
        raise RubricBatchContractError("Rubric candidates 没有覆盖完整任务集")
    decisions = _load_review_decisions(review_decisions_path)
    review_ids = {
        task_id
        for task_id, candidate in candidates.items()
        if candidate.get("review_required") is True
    }
    missing_decisions = sorted(review_ids.difference(decisions))
    if missing_decisions:
        raise ValueError(
            f"仍有 {len(missing_decisions)} 个复核任务没有决定："
            f"{missing_decisions[:10]}"
        )
    unexpected = sorted(set(decisions).difference(review_ids))
    if unexpected:
        raise ValueError(f"复核决定包含不在队列中的任务：{unexpected[:10]}")

    rubrics = []
    applied = []
    for task_id in task_ids:
        candidate = candidates[int(task_id)]
        proposed = validate_rubric_bundle(candidate["rubric"])
        if int(task_id) not in review_ids:
            rubrics.append(proposed)
            continue
        decision = decisions[int(task_id)]
        action = str(decision.get("decision") or "").strip()
        reviewer = str(decision.get("reviewer") or "").strip()
        reason = str(decision.get("reason") or "").strip()
        if action not in {"approve", "replace"} or not reviewer or not reason:
            raise ValueError(
                f"task_id={task_id} 的复核决定需要 approve/replace、reviewer 和 reason"
            )
        if action == "approve":
            final = proposed
        else:
            replacement_items, replacement_warnings = normalize_generated_rubric_items(
                decision.get("items"),
                instruction=proposed["instruction"],
            )
            if replacement_warnings:
                raise ValueError(
                    f"task_id={task_id} 的人工替换仍有审计问题："
                    f"{replacement_warnings}"
                )
            final = freeze_rubric_bundle(
                task_id=int(task_id),
                instruction=proposed["instruction"],
                items=replacement_items,
                rubric_set_version=contract["rubric_set_version"],
            )
        rubrics.append(final)
        applied.append(
            {
                "schema_version": "wlx-rubric-review-decision-applied-v1",
                "task_id": int(task_id),
                "decision": action,
                "reviewer": reviewer,
                "reason": reason,
                "final_rubric": deepcopy(final),
            }
        )

    _ensure_none_exist(paths.final_rubrics, paths.applied_reviews)
    write_jsonl(paths.final_rubrics, rubrics)
    write_jsonl(paths.applied_reviews, applied)
    final_manifest = {
        "schema_version": RUBRIC_FINALIZATION_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rubric_set_version": contract["rubric_set_version"],
        "tasks": len(rubrics),
        "auto_frozen": len(rubrics) - len(applied),
        "human_reviewed": len(applied),
        "source_generation_manifest": str(paths.manifest),
        "source_generation_manifest_sha256": file_sha256(paths.manifest),
        "review_decisions": (
            str(review_decisions_path) if review_decisions_path is not None else None
        ),
        "review_decisions_sha256": (
            file_sha256(review_decisions_path)
            if review_decisions_path is not None
            else None
        ),
        "rubrics": str(paths.final_rubrics),
        "rubrics_sha256": file_sha256(paths.final_rubrics),
        "applied_reviews": str(paths.applied_reviews),
        "applied_reviews_sha256": file_sha256(paths.applied_reviews),
    }
    _write_new_json(paths.finalization_manifest, final_manifest)
    return {
        "finalization_version": RUBRIC_FINALIZATION_VERSION,
        "tasks": len(rubrics),
        "human_reviewed": len(applied),
        "rubrics": str(paths.final_rubrics),
        "manifest": str(paths.finalization_manifest),
    }


def _trajectory_instructions(path: Path) -> dict[int, str]:
    if not path.is_file():
        raise FileNotFoundError(f"轨迹文件不存在：{path}")
    result: dict[int, str] = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError("轨迹含无效 task_id")
        instruction = _instruction_from_trajectory(row)
        if not instruction:
            raise ValueError(f"task_id={task_id} 的轨迹缺少初始用户需求")
        previous = result.get(task_id)
        if previous and _normalize_instruction(previous) != _normalize_instruction(instruction):
            raise ValueError(f"task_id={task_id} 存在不同的初始需求")
        result[task_id] = instruction
    return result


def _instruction_from_trajectory(trajectory: Mapping[str, Any]) -> str | None:
    initial = trajectory.get("initial_result")
    if isinstance(initial, Mapping):
        instruction = _strip_instruction_prefix(initial.get("instruction"))
        if instruction:
            return instruction
    messages = trajectory.get("messages") or []
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            match = re_search_instruction_block(content)
            if match:
                return match
    terminal = trajectory.get("terminal_result")
    if isinstance(terminal, Mapping):
        goal = terminal.get("goal")
        if isinstance(goal, Mapping):
            return _optional_text(goal.get("instruction_text"))
    return None


def re_search_instruction_block(content: str) -> str | None:
    marker = "Instruction:"
    if marker not in content:
        return None
    value = content.split(marker, 1)[1]
    for boundary in ("\n\n【初始", "\n[SHOPPING_OBSERVATION_V2]"):
        value = value.split(boundary, 1)[0]
    return _optional_text(value)


def _strip_instruction_prefix(value: object) -> str | None:
    text = _optional_text(value)
    if not text:
        return None
    return _optional_text(re.sub(r"^Instruction\s*:\s*", "", text, flags=re.IGNORECASE))


def _validate_tasks(tasks: Sequence[RubricTask]) -> list[RubricTask]:
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence) or not tasks:
        raise ValueError("tasks must be a non-empty array")
    result = []
    seen: set[int] = set()
    for task in tasks:
        if not isinstance(task, RubricTask):
            raise TypeError("tasks must contain RubricTask")
        if task.task_id in seen:
            raise ValueError(f"duplicate Rubric task_id: {task.task_id}")
        seen.add(task.task_id)
        result.append(task)
    return result


def _prepare_generation_manifest(
    paths: RubricBatchPaths,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    encoded = json.dumps(
        dict(contract), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if paths.manifest.exists():
        manifest = _read_json_object(paths.manifest)
        if (
            manifest.get("schema_version") != RUBRIC_MANIFEST_VERSION
            or manifest.get("generation_contract_sha256") != fingerprint
            or manifest.get("generation_contract") != dict(contract)
        ):
            raise RubricBatchContractError("已有 Rubric manifest 与本次生成不一致")
        return manifest
    if paths.attempts.exists() and paths.attempts.stat().st_size:
        raise RubricBatchContractError("已有 Rubric attempts 但缺少 manifest")
    manifest = {
        "schema_version": RUBRIC_MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation_contract_sha256": fingerprint,
        "generation_contract": deepcopy(dict(contract)),
    }
    _write_new_json(paths.manifest, manifest)
    return manifest


def _latest_successful_attempts(
    path: Path,
    tasks: Sequence[RubricTask],
) -> dict[int, dict[str, Any]]:
    expected = {task.task_id: task for task in tasks}
    result = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if task_id not in expected:
            raise RubricBatchContractError(f"attempts 含本批次外 task_id={task_id}")
        if row.get("instruction_sha256") != expected[int(task_id)].instruction_sha256:
            raise RubricBatchContractError(f"task_id={task_id} instruction 哈希不一致")
        if row.get("status") == "candidate_ready":
            result[int(task_id)] = row
    return result


def _reaudit_successful_attempts(
    attempts: Mapping[int, Mapping[str, Any]],
    tasks: Sequence[RubricTask],
    *,
    rubric_set_version: str,
) -> dict[int, dict[str, Any]]:
    """用当前本地规则重审已有 LLM 原始结果，不再消耗 API。"""

    task_by_id = {task.task_id: task for task in tasks}
    result: dict[int, dict[str, Any]] = {}
    for task_id, raw_attempt in attempts.items():
        attempt = deepcopy(dict(raw_attempt))
        candidate = attempt.get("candidate")
        if not isinstance(candidate, Mapping):
            continue
        raw_result = candidate.get("llm_raw_result")
        metadata = candidate.get("llm_metadata") or {}
        if not isinstance(raw_result, Mapping) or not isinstance(metadata, Mapping):
            continue
        try:
            attempt["candidate"] = audit_rubric_candidate_result(
                task_by_id[task_id],
                raw_result=raw_result,
                llm_metadata=metadata,
                rubric_set_version=rubric_set_version,
            )
        except RubricGenerationError:
            # 已有原始结果若不再符合当前契约，让它回到 pending 重新生成。
            continue
        result[task_id] = attempt
    return result


def _load_review_decisions(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    result = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError("复核决定含无效 task_id")
        if task_id in result:
            raise ValueError(f"复核决定中 task_id={task_id} 重复")
        result[task_id] = row
    return result


def _generation_result(
    paths: RubricBatchPaths,
    attempts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    reused: bool,
) -> dict[str, Any]:
    reviews = sum(
        attempt["candidate"].get("review_required") is True for attempt in attempts
    )
    return {
        "batch_version": RUBRIC_BATCH_VERSION,
        "complete": True,
        "reused_existing_outputs": reused,
        "tasks_selected": len(attempts),
        "auto_frozen": len(attempts) - reviews,
        "review_required": reviews,
        "output_dir": str(paths.output_dir),
        "manifest": str(paths.manifest),
        "manifest_sha256": file_sha256(paths.manifest),
        "candidates": str(paths.candidates),
        "auto_frozen_rubrics": str(paths.auto_frozen),
        "review_queue": str(paths.review_queue),
        "review_decisions_template": str(paths.review_decisions_template),
        "summary": str(paths.summary),
    }


def _task_instructions_sha256(tasks: Sequence[RubricTask]) -> str:
    value = [
        {"task_id": task.task_id, "instruction_sha256": task.instruction_sha256}
        for task in tasks
    ]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize_instruction(value: str) -> str:
    return "".join(str(value).split()).casefold()


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _notify(
    callback: Callable[[Mapping[str, Any]], object] | None,
    progress: Mapping[str, Any],
) -> None:
    if callback is not None:
        callback(deepcopy(dict(progress)))


def _ensure_none_exist(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有 Rubric 产物：{existing[0]}")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"文件必须是 JSON 对象：{path}")
    return deepcopy(dict(value))


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"临时文件已经存在：{temporary}")
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


__all__ = [
    "RubricBatchContractError",
    "RubricBatchPaths",
    "RUBRIC_BATCH_VERSION",
    "RUBRIC_FINALIZATION_VERSION",
    "RUBRIC_MANIFEST_VERSION",
    "finalize_rubric_set",
    "load_rubric_tasks",
    "re_search_instruction_block",
    "run_rubric_generation",
]
