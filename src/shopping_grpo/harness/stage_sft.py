"""把统一轨迹交给现有 SFT 数据流水线，并负责挡住评测数据泄漏。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from shopping_grpo.collection.sft import (
    acceptance_reasons as _acceptance_reasons,
    build_collection_artifacts as _build_collection_artifacts,
    build_sft_row as _build_sft_row,
    task_ids_from_jsonl,
)

from shopping_grpo.harness.serialization import trajectory_to_legacy


DEFAULT_HELD_OUT_TASKS_PATH = (
    Path(__file__).resolve().parents[3] / "data/evaluation/tasks.jsonl"
)


class SFTDataLeakError(ValueError):
    """表示有人试图把专门留给评测的任务放进 SFT 训练数据。"""

    pass


@dataclass(frozen=True)
class SFTStageOutput:
    """保存一条轨迹是否能用于 SFT，以及不能使用时的具体原因。

    只有通过质量检查和评测集隔离检查的轨迹才会带有 ``training_row``，这样后续
    写数据集时不用再次猜测这条样本是否安全。
    """

    accepted: bool
    rejection_reasons: tuple[str, ...]
    training_row: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """把阶段结果转成普通字典，并复制训练行，避免调用方改动内部数据。"""

        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "training_row": deepcopy(self.training_row),
        }


class SFTStageAdapter:
    """集中处理 SFT 样本筛选，并强制排除所有留给评测的任务。

    交互运行器只负责产生轨迹，SFT 特有的质量门槛和防泄漏规则放在这一层，避免
    把训练数据规则塞进共享 Core，也避免不同数据脚本各自漏做检查。
    """

    def __init__(
        self,
        *,
        held_out_task_ids: Iterable[int] = (),
        held_out_tasks_path: str | Path = DEFAULT_HELD_OUT_TASKS_PATH,
    ) -> None:
        """读取标准评测任务编号，并合并调用方额外指定的保留任务编号。"""

        self.held_out_tasks_path = Path(held_out_tasks_path)
        self.held_out_task_ids = _resolve_held_out_ids(
            held_out_task_ids,
            self.held_out_tasks_path,
        )

    def prepare(self, trajectory: object) -> SFTStageOutput:
        """检查一条轨迹是否适合 SFT，并在通过时生成一条训练数据。"""

        return prepare_sft_stage_output(
            trajectory,
            held_out_task_ids=self.held_out_task_ids,
            held_out_tasks_path=self.held_out_tasks_path,
        )

    def build_artifacts(
        self,
        *,
        raw_path: str | Path,
        output_dir: str | Path,
        held_out_task_ids: Iterable[int] = (),
        validation_ratio: float = 0.1,
        seed: int = 42,
        collection_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """把原始轨迹批量整理成 SFT 训练集、验证集和对应统计信息。

        适配器初始化时的保留任务和本次额外传入的保留任务都会被排除，确保批量构建
        数据时也不会绕开单条轨迹使用的防泄漏规则。
        """

        return build_sft_artifacts(
            raw_path=raw_path,
            output_dir=output_dir,
            held_out_task_ids=self.held_out_task_ids.union(
                int(item) for item in held_out_task_ids
            ),
            held_out_tasks_path=self.held_out_tasks_path,
            validation_ratio=validation_ratio,
            seed=seed,
            collection_config=collection_config,
        )


def to_legacy_trajectory(trajectory: object) -> dict[str, Any]:
    """把 Core 轨迹转换成旧 SFT 流水线认识的字典格式。"""

    return trajectory_to_legacy(trajectory)


def acceptance_reasons(trajectory: object) -> tuple[bool, list[str]]:
    """调用现有质量规则，说明轨迹质量是否达标以及不达标的原因。

    这个函数只看样本质量；是否属于保留评测任务由 ``prepare`` 系列函数统一检查。
    """

    return _acceptance_reasons(to_legacy_trajectory(trajectory))


def build_sft_row(
    trajectory: object,
    *,
    held_out_task_ids: Iterable[int] = (),
    held_out_tasks_path: str | Path = DEFAULT_HELD_OUT_TASKS_PATH,
) -> dict[str, Any]:
    """把一条轨迹转换成 SFT 训练行，并在转换前阻止评测任务泄漏。

    这是给明确需要单条训练行的调用方使用的低层入口，所以发现保留任务时会直接
    抛错，而不是悄悄返回空值。
    """

    legacy = to_legacy_trajectory(trajectory)
    held_out = _resolve_held_out_ids(held_out_task_ids, held_out_tasks_path)
    if int(legacy["task_id"]) in held_out:
        raise SFTDataLeakError("held-out evaluation task cannot become an SFT row")
    return _build_sft_row(legacy)


def prepare_sft_stage_output(
    trajectory: object,
    *,
    held_out_task_ids: Iterable[int] = (),
    held_out_tasks_path: str | Path = DEFAULT_HELD_OUT_TASKS_PATH,
) -> SFTStageOutput:
    """依次执行严格的 Reward-v3 质量检查和评测任务隔离检查。

    返回值会明确告诉上层是否接收；只有全部通过时才真正生成训练行，防止低质量或
    留作评测的轨迹进入 SFT 数据。
    """

    legacy = to_legacy_trajectory(trajectory)
    accepted, reasons = _acceptance_reasons(legacy)
    held_out = _resolve_held_out_ids(held_out_task_ids, held_out_tasks_path)
    if accepted and int(legacy["task_id"]) in held_out:
        accepted = False
        reasons = ["held_out_task"]
    row = _build_sft_row(legacy) if accepted else None
    return SFTStageOutput(
        accepted=bool(accepted),
        rejection_reasons=tuple(str(reason) for reason in reasons),
        training_row=row,
    )


def build_sft_artifacts(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    held_out_task_ids: Iterable[int] = (),
    held_out_tasks_path: str | Path = DEFAULT_HELD_OUT_TASKS_PATH,
    validation_ratio: float = 0.1,
    seed: int = 42,
    collection_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """调用项目原有的稳定构建逻辑，生成可直接训练的 SFT 数据文件。

    标准评测任务会被自动加入排除名单；固定随机种子和验证集比例则让多次构建结果
    可以复现。
    """

    return _build_collection_artifacts(
        raw_path=raw_path,
        output_dir=output_dir,
        held_out_task_ids=_resolve_held_out_ids(
            held_out_task_ids,
            held_out_tasks_path,
        ),
        validation_ratio=validation_ratio,
        seed=seed,
        collection_config=collection_config,
    )


def _resolve_held_out_ids(
    additional_ids: Iterable[int],
    path: str | Path,
) -> set[int]:
    """读取标准评测任务文件，并与额外保留编号合并成一个集合。

    标准文件不存在时直接报错，因为缺少它就无法证明训练数据和评测数据已经隔离。
    """

    held_out_path = Path(path)
    if not held_out_path.is_file():
        raise FileNotFoundError(
            f"held-out evaluation task file is required: {held_out_path}"
        )
    canonical = task_ids_from_jsonl(held_out_path)
    return canonical.union(int(item) for item in additional_ids)


__all__ = [
    "DEFAULT_HELD_OUT_TASKS_PATH",
    "SFTDataLeakError",
    "SFTStageAdapter",
    "SFTStageOutput",
    "acceptance_reasons",
    "build_sft_artifacts",
    "build_sft_row",
    "prepare_sft_stage_output",
    "to_legacy_trajectory",
]
