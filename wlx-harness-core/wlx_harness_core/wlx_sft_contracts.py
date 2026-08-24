"""定义 WLX SFT 数据流水线各模块共同使用的数据格式。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from wlx_harness_core.wlx_contracts import EpisodeRequest


WLX_SFT_PIPELINE_VERSION = "wlx-sft-pipeline-v1"


class SamplingMode(str, Enum):
    """说明当前是在做难度校准，还是在正式收集 SFT 数据。"""

    CALIBRATION = "calibration"
    FORMAL = "formal"


class OutcomeType(str, Enum):
    """用一个稳定名称说明一次尝试最后发生了什么。"""

    GOLD_PURCHASE = "gold_purchase"
    VALID_ALTERNATIVE_PURCHASE = "valid_alternative_purchase"
    PARTIAL_ALTERNATIVE_PURCHASE = "partial_alternative_purchase"
    WRONG_PURCHASE = "wrong_purchase"
    GRACEFUL_STOP = "graceful_stop"
    EARLY_ABSTAIN = "early_abstain"
    REPEAT_LOOP = "repeat_loop"
    MAX_STEPS = "max_steps"
    NO_PURCHASE = "no_purchase"
    INVALID_ACTION = "invalid_action"
    MODEL_ERROR = "model_error"
    REWARD_UNVERIFIABLE = "reward_unverifiable"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    PROTOCOL_ERROR = "protocol_error"
    UNKNOWN = "unknown"


class SftDisposition(str, Enum):
    """说明这条轨迹在 SFT 构建阶段应该进入哪一个去向。"""

    ACCEPTED_GOLD = "accepted_gold"
    ALTERNATIVE_AUDIT = "alternative_audit"
    REJECTED = "rejected"
    RETRY = "retry"


@dataclass(frozen=True)
class SftTask:
    """保存 Teacher 真正需要看到的任务信息，不携带标准答案等隐藏字段。"""

    task_id: int
    instruction: str | None = None
    official_split: str = "train"
    category: str | None = None
    difficulty_label: str | None = None
    difficulty_score: float | None = None
    difficulty_version: str | None = None

    def __post_init__(self) -> None:
        """创建任务后检查编号和难度分数，尽早拦住写错的数据。"""

        if isinstance(self.task_id, bool):
            raise TypeError("task_id 必须是整数")
        object.__setattr__(self, "task_id", int(self.task_id))
        if self.task_id < 0:
            raise ValueError("task_id 不能小于 0")
        if self.difficulty_score is not None:
            score = float(self.difficulty_score)
            if not 0.0 <= score <= 1.0:
                raise ValueError("difficulty_score 必须在 0 到 1 之间")
            object.__setattr__(self, "difficulty_score", score)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SftTask":
        """从任务表读取公开字段，并故意忽略 target_asin、Reward 等隐藏答案。"""

        if not isinstance(value, Mapping):
            raise TypeError("任务必须是一个字典")
        task_id = value.get("task_id")
        extra = value.get("extra_info")
        if task_id is None and isinstance(extra, Mapping):
            task_id = extra.get("task_id")
        if task_id is None:
            raise ValueError("任务缺少 task_id")
        instruction = value.get("instruction")
        if instruction is None:
            instruction = _last_user_message(value.get("prompt"))
        return cls(
            task_id=int(task_id),
            instruction=str(instruction) if instruction is not None else None,
            official_split=str(value.get("official_split") or "train"),
            category=(str(value["category"]) if value.get("category") else None),
            difficulty_label=(
                str(value["difficulty_label"])
                if value.get("difficulty_label")
                else None
            ),
            difficulty_score=(
                float(value["difficulty_score"])
                if value.get("difficulty_score") is not None
                else None
            ),
            difficulty_version=(
                str(value["difficulty_version"])
                if value.get("difficulty_version")
                else None
            ),
        )

    def to_episode_request(self, attempt_index: int) -> EpisodeRequest:
        """把任务变成 Core 能执行的请求，只传公开的分组信息。"""

        metadata = {
            "official_split": self.official_split,
            "category": self.category,
            "difficulty_label": self.difficulty_label,
            "difficulty_score": self.difficulty_score,
            "difficulty_version": self.difficulty_version,
            "pipeline_version": WLX_SFT_PIPELINE_VERSION,
        }
        return EpisodeRequest(
            task_id=self.task_id,
            instruction=self.instruction,
            attempt_index=int(attempt_index),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def to_public_dict(self) -> dict[str, Any]:
        """导出任务的公开信息，方便写采样计划和统计文件。"""

        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "official_split": self.official_split,
            "category": self.category,
            "difficulty_label": self.difficulty_label,
            "difficulty_score": self.difficulty_score,
            "difficulty_version": self.difficulty_version,
        }


@dataclass(frozen=True)
class AttemptDecision:
    """把一次轨迹是否有效、是否成功、能否训练这三件事分开记录。"""

    outcome_type: OutcomeType
    attempt_valid: bool
    task_success: bool | None
    sft_disposition: SftDisposition
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """把判断结果转成能直接写入 JSONL 的普通字典。"""

        return {
            "outcome_type": self.outcome_type.value,
            "attempt_valid": self.attempt_valid,
            "task_success": self.task_success,
            "sft_disposition": self.sft_disposition.value,
            "decision_reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SamplingConfig:
    """规定每个任务采几次、并发多少，以及技术故障最多重试几次。"""

    mode: SamplingMode
    valid_attempts_per_task: int = 3
    max_technical_retries_per_attempt: int = 2
    concurrency: int = 1
    stop_after_gold: bool = True
    target_gold_trajectories: int | None = None

    def __post_init__(self) -> None:
        """检查采样数字是否合理，并固定两种模式各自的停止规则。"""

        if not isinstance(self.mode, SamplingMode):
            object.__setattr__(self, "mode", SamplingMode(str(self.mode)))
        for name, value in (
            ("valid_attempts_per_task", self.valid_attempts_per_task),
            ("concurrency", self.concurrency),
        ):
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"{name} 必须至少为 1")
        if self.max_technical_retries_per_attempt < 0:
            raise ValueError("max_technical_retries_per_attempt 不能小于 0")
        if (
            self.target_gold_trajectories is not None
            and self.target_gold_trajectories < 1
        ):
            raise ValueError("target_gold_trajectories 必须至少为 1")
        if self.mode == SamplingMode.CALIBRATION and self.stop_after_gold:
            object.__setattr__(self, "stop_after_gold", False)
        if self.mode == SamplingMode.CALIBRATION and self.target_gold_trajectories:
            raise ValueError("难度校准必须固定跑满，不能设置 target_gold_trajectories")


@dataclass(frozen=True)
class AttemptEnvelope:
    """把 Core 原始轨迹、流水线判断和技术重试编号装在同一条记录里。"""

    trajectory: Mapping[str, Any]
    decision: AttemptDecision
    technical_retry_index: int = 0
    token_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """复制输入内容，避免落盘前被外部代码意外修改。"""

        object.__setattr__(self, "trajectory", deepcopy(dict(self.trajectory)))
        object.__setattr__(self, "token_metrics", deepcopy(dict(self.token_metrics)))
        if self.technical_retry_index < 0:
            raise ValueError("technical_retry_index 不能小于 0")

    def to_dict(self) -> dict[str, Any]:
        """生成 raw.jsonl 的一行，保留完整轨迹并添加流水线字段。"""

        row = deepcopy(dict(self.trajectory))
        row.update(self.decision.to_dict())
        row["technical_retry_index"] = self.technical_retry_index
        row["token_metrics"] = deepcopy(dict(self.token_metrics))
        row["wlx_sft_pipeline_version"] = WLX_SFT_PIPELINE_VERSION
        return row


def _last_user_message(raw_prompt: object) -> str | None:
    """从已有 Prompt 中找最后一条用户消息，找不到时返回空值。"""

    if not isinstance(raw_prompt, (list, tuple)):
        return None
    for message in reversed(raw_prompt):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            return str(content) if content is not None else None
    return None


__all__ = [
    "AttemptDecision",
    "AttemptEnvelope",
    "OutcomeType",
    "SamplingConfig",
    "SamplingMode",
    "SftDisposition",
    "SftTask",
    "WLX_SFT_PIPELINE_VERSION",
]
