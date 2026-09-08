"""把 ShopSimulator canonical goals 整理成不含标准答案的公开 SFT 任务计划。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from shopping_grpo.harness.sft_contracts import SftTask
from shopping_grpo.harness.sft_difficulty import (
    LogisticDifficultyModel,
    TaskDifficultyFeatures,
    select_calibration_tasks,
)
from shopping_grpo.harness.sft_storage import write_jsonl


def public_tasks_from_canonical_goals(
    goals: Sequence[Mapping[str, Any]],
    *,
    held_out_task_ids: Iterable[int] = (),
) -> list[SftTask]:
    """按完整 goals 的原始下标生成 task_id，并在输出前去掉 ASIN、Reward 等隐藏字段。"""

    held_out = {int(task_id) for task_id in held_out_task_ids}
    tasks = []
    for task_id, goal in enumerate(goals):
        if task_id in held_out:
            continue
        tasks.append(
            SftTask(
                task_id=task_id,
                instruction=str(goal.get("instruction_text") or ""),
                official_split="train",
                category=(str(goal["category"]) if goal.get("category") else None),
            )
        )
    return tasks


def calibration_task_plan(
    tasks: Sequence[SftTask],
    features: Sequence[TaskDifficultyFeatures],
    *,
    sample_size: int = 200,
    seed: int = 42,
) -> list[SftTask]:
    """按临时难度和类别选校准题，并把规则分数写进公开计划。"""

    by_task = {task.task_id: task for task in tasks}
    feature_by_task = {item.task_id: item for item in features}
    selected_ids = select_calibration_tasks(features, sample_size=sample_size, seed=seed)
    result = []
    for task_id in selected_ids:
        if task_id not in by_task:
            raise ValueError(f"难度特征中的 task_id={task_id} 不在公开任务池")
        task = by_task[task_id]
        feature = feature_by_task[task_id]
        result.append(
            SftTask(
                task_id=task.task_id,
                instruction=task.instruction,
                official_split=task.official_split,
                category=task.category,
                difficulty_label=feature.preliminary_label,
                difficulty_score=feature.preliminary_score,
                difficulty_version="preliminary-rule-v1",
            )
        )
    return result


def apply_difficulty_model(
    tasks: Sequence[SftTask],
    features: Mapping[int, TaskDifficultyFeatures],
    model: LogisticDifficultyModel,
) -> list[SftTask]:
    """用校准后的成功概率给全部正式采样任务写 easy、medium、hard 标签。"""

    result = []
    for task in tasks:
        try:
            task_features = features[task.task_id]
        except KeyError as exc:
            raise ValueError(f"task_id={task.task_id} 缺少难度特征") from exc
        label, _, score = model.classify(task_features)
        result.append(
            SftTask(
                task_id=task.task_id,
                instruction=task.instruction,
                official_split=task.official_split,
                category=task.category,
                difficulty_label=label,
                difficulty_score=score,
                difficulty_version=model.difficulty_version,
            )
        )
    return result


def write_public_task_plan(path: str | Path, tasks: Iterable[SftTask]) -> int:
    """把公开任务计划写成 JSONL；SftTask 本身没有 target_asin 等隐藏字段。"""

    return write_jsonl(path, (deepcopy(task.to_public_dict()) for task in tasks))


__all__ = [
    "apply_difficulty_model",
    "calibration_task_plan",
    "public_tasks_from_canonical_goals",
    "write_public_task_plan",
]
