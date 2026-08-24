"""把 WLX Core 轨迹整理成评测程序需要的安全格式，并计算固定指标。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.summary import summarize_trajectories
from shopping_grpo.evaluation.trajectory import normalize_trajectory

from wlx_harness_core.wlx_serialization import trajectory_to_legacy


@dataclass(frozen=True)
class EvaluationStageOutput:
    """保存清理后的评测轨迹和由代码直接算出的固定指标。

    轨迹先经过统一整理，避免把不该交给裁判模型的原始审计内容带进去；确定性指标
    则完全由代码计算，同一份轨迹每次都会得到同样的结果。
    """

    normalized_trajectory: dict[str, Any]
    deterministic_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """把评测结果转成普通字典，并复制内容，防止调用方改动内部结果。"""

        return {
            "normalized_trajectory": deepcopy(self.normalized_trajectory),
            "deterministic_metrics": deepcopy(self.deterministic_metrics),
        }


class EvaluationStageAdapter:
    """把评测专用的轨迹整理、指标计算和汇总留在独立阶段层。

    共享运行器只负责完成购物交互，不掺杂某一种评测报表的规则。这样以后调整指标
    或报告格式时，不需要修改 SFT、GRPO 共用的核心流程。
    """

    def prepare(
        self,
        trajectory: object,
        *,
        include_audit_raw_observations: bool = False,
    ) -> EvaluationStageOutput:
        """整理一条轨迹并计算它的固定指标，得到可供评测使用的结果。

        默认不包含原始页面审计内容；只有明确需要排查问题时，调用方才应开启该选项。
        """

        return prepare_evaluation_stage_output(
            trajectory,
            include_audit_raw_observations=include_audit_raw_observations,
        )

    def summarize(
        self,
        expected_task_ids: Iterable[int],
        trajectories: Iterable[object],
    ) -> dict[str, Any]:
        """把多条评测轨迹汇总成整体报告，并检查预期任务是否齐全。"""

        return summarize_evaluation_trajectories(expected_task_ids, trajectories)


def normalize_for_evaluation(
    trajectory: object,
    *,
    include_audit_raw_observations: bool = False,
) -> dict[str, Any]:
    """按项目统一规则，把 Core 轨迹整理成评测只需要看到的内容。

    先转成旧评测代码认识的格式，再调用项目已有的标准化逻辑，可以保证新旧流水线
    对同一条轨迹使用相同的可见范围。
    """

    return normalize_trajectory(
        trajectory_to_legacy(trajectory),
        include_audit_raw_observations=include_audit_raw_observations,
    )


def prepare_evaluation_stage_output(
    trajectory: object,
    *,
    include_audit_raw_observations: bool = False,
) -> EvaluationStageOutput:
    """一次完成单条轨迹的标准化和确定性指标计算。

    把这两个步骤绑在同一入口，可以避免调用方拿未经清理的轨迹直接计算或展示指标。
    """

    normalized = normalize_for_evaluation(
        trajectory,
        include_audit_raw_observations=include_audit_raw_observations,
    )
    return EvaluationStageOutput(
        normalized_trajectory=normalized,
        deterministic_metrics=compute_deterministic_metrics(normalized),
    )


def summarize_evaluation_trajectories(
    expected_task_ids: Iterable[int],
    trajectories: Iterable[object],
) -> dict[str, Any]:
    """把不同表示形式的多条轨迹统一转换后，交给现有评测汇总器生成报告。"""

    legacy = [trajectory_to_legacy(item) for item in trajectories]
    return summarize_trajectories(expected_task_ids, legacy)


__all__ = [
    "EvaluationStageAdapter",
    "EvaluationStageOutput",
    "normalize_for_evaluation",
    "prepare_evaluation_stage_output",
    "summarize_evaluation_trajectories",
]
