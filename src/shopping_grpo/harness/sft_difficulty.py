"""实现 SFT 任务难度的规则初分和小型逻辑回归校准。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


DIFFICULTY_FEATURE_NAMES = (
    "constraint_count",
    "option_axis_count",
    "has_brand",
    "has_model",
    "has_budget",
    "retrieval_score",
    "near_miss_score",
)


@dataclass(frozen=True)
class RetrievalEvidence:
    """保存完整指令搜索前 20 名时，合格商品和相似干扰商品的机械证据。"""

    best_valid_rank_at_20: int | None
    target_asin_rank_at_20: int | None
    best_valid_outcome_type: str | None
    near_miss_count_at_20: int
    unverifiable_candidates_at_20: int = 0
    candidate_evaluation_reliable: bool = True
    near_miss_evidence: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """检查排名和计数，防止离线特征缓存出现不可能的值。"""

        for rank in (self.best_valid_rank_at_20, self.target_asin_rank_at_20):
            if rank is not None and not 1 <= int(rank) <= 20:
                raise ValueError("前 20 名中的 rank 必须在 1 到 20 之间")
        if self.near_miss_count_at_20 < 0 or self.unverifiable_candidates_at_20 < 0:
            raise ValueError("候选商品数量不能小于 0")


@dataclass(frozen=True)
class TaskDifficultyFeatures:
    """保存一道题的七个校准输入，以及便于审计的规则分数。"""

    task_id: int
    constraint_count: int
    option_axis_count: int
    has_brand: bool
    has_model: bool
    has_budget: bool
    retrieval_score: float
    near_miss_score: float
    constraint_score: float
    preliminary_score: float
    preliminary_label: str | None = None
    category: str | None = None
    evidence: Mapping[str, Any] | None = None

    def vector(self) -> tuple[float, ...]:
        """按固定顺序返回逻辑回归使用的七个数字。"""

        return (
            float(self.constraint_count),
            float(self.option_axis_count),
            float(self.has_brand),
            float(self.has_model),
            float(self.has_budget),
            float(self.retrieval_score),
            float(self.near_miss_score),
        )

    def to_dict(self) -> dict[str, Any]:
        """把难度特征转成可缓存、可复查的普通字典。"""

        return {
            "task_id": self.task_id,
            "constraint_count": self.constraint_count,
            "option_axis_count": self.option_axis_count,
            "has_brand": self.has_brand,
            "has_model": self.has_model,
            "has_budget": self.has_budget,
            "retrieval_score": self.retrieval_score,
            "near_miss_score": self.near_miss_score,
            "constraint_score": self.constraint_score,
            "preliminary_score": self.preliminary_score,
            "preliminary_label": self.preliminary_label,
            "category": self.category,
            "evidence": dict(self.evidence or {}),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskDifficultyFeatures":
        """从离线特征 JSONL 恢复一条任务难度特征。"""

        return cls(
            task_id=int(value["task_id"]),
            constraint_count=int(value["constraint_count"]),
            option_axis_count=int(value["option_axis_count"]),
            has_brand=bool(value["has_brand"]),
            has_model=bool(value["has_model"]),
            has_budget=bool(value["has_budget"]),
            retrieval_score=float(value["retrieval_score"]),
            near_miss_score=float(value["near_miss_score"]),
            constraint_score=float(value["constraint_score"]),
            preliminary_score=float(value["preliminary_score"]),
            preliminary_label=(
                str(value["preliminary_label"])
                if value.get("preliminary_label")
                else None
            ),
            category=str(value["category"]) if value.get("category") else None,
            evidence=dict(value.get("evidence") or {}),
        )


@dataclass(frozen=True)
class CalibrationSample:
    """表示一道任务的一次有效尝试，成功为 1，失败为 0。"""

    task_id: int
    features: TaskDifficultyFeatures
    success: bool

    def __post_init__(self) -> None:
        """保证样本编号与特征编号一致，避免把结果贴到另一道题上。"""

        if int(self.task_id) != int(self.features.task_id):
            raise ValueError("CalibrationSample 的 task_id 与 features 不一致")


@dataclass(frozen=True)
class LogisticDifficultyModel:
    """一个只接收七个数字的小模型，用来估计当前 Teacher 的成功概率。"""

    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    difficulty_version: str

    def predict_success(self, features: TaskDifficultyFeatures) -> float:
        """估计 Teacher 单次完成该任务的概率，返回 0 到 1 之间的数字。"""

        values = features.vector()
        standardised = [
            (value - mean) / scale
            for value, mean, scale in zip(values, self.means, self.scales, strict=True)
        ]
        logit = self.intercept + sum(
            weight * value
            for weight, value in zip(self.coefficients, standardised, strict=True)
        )
        return _sigmoid(logit)

    def classify(self, features: TaskDifficultyFeatures) -> tuple[str, float, float]:
        """返回 easy、medium 或 hard，同时给出成功概率和难度分数。"""

        success = self.predict_success(features)
        if success >= 0.75:
            label = "easy"
        elif success >= 0.35:
            label = "medium"
        else:
            label = "hard"
        return label, success, 1.0 - success

    def to_dict(self) -> dict[str, Any]:
        """导出模型权重和版本，使正式采样可以复现同一套难度标签。"""

        return {
            "model_type": "stdlib_logistic_regression",
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "difficulty_version": self.difficulty_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LogisticDifficultyModel":
        """从冻结的 JSON 权重恢复难度模型。"""

        return cls(
            feature_names=tuple(str(item) for item in value["feature_names"]),
            means=tuple(float(item) for item in value["means"]),
            scales=tuple(float(item) for item in value["scales"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            intercept=float(value["intercept"]),
            difficulty_version=str(value["difficulty_version"]),
        )


def independent_constraint_count(goal: Mapping[str, Any]) -> tuple[int, dict[str, int | bool]]:
    """按功能、规格轴、品牌、型号和预算计算互不重复的要求数量 A。"""

    core_count = len(set(goal.get("expected_core_functions") or ()))
    resolved_axes = len(set((goal.get("required_options_by_key") or {}).keys()))
    unresolved_count = len(set(goal.get("unresolved_option_requirements") or ()))
    has_brand = bool(goal.get("expected_brand"))
    has_model = bool(goal.get("expected_model"))
    has_budget = goal.get("price_upper") is not None or goal.get("max_price") is not None
    total = (
        core_count
        + resolved_axes
        + unresolved_count
        + int(has_brand)
        + int(has_model)
        + int(has_budget)
    )
    return total, {
        "core_function_count": core_count,
        "option_axis_count": resolved_axes,
        "unresolved_option_count": unresolved_count,
        "has_brand": has_brand,
        "has_model": has_model,
        "has_budget": has_budget,
    }


def percentile_95(values: Iterable[int | float]) -> float:
    """返回 nearest-rank 口径的 95 分位，用来压住极少数异常大值。"""

    ordered = sorted(float(item) for item in values)
    if not ordered:
        raise ValueError("计算 95 分位时至少需要一个数字")
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def constraint_score(constraint_count: int, a95: float) -> float:
    """把独立约束数 A 按文档公式压到 0 到 1 之间。"""

    if a95 <= 0:
        return 0.0 if int(constraint_count) <= 0 else 1.0
    return min(1.0, math.log1p(max(0, int(constraint_count))) / math.log1p(a95))


def retrieval_score(best_valid_rank_at_20: int | None) -> float:
    """把首个合格商品排名转成 R；前 20 名没有时按第 21 名处理。"""

    rank = 21 if best_valid_rank_at_20 is None else int(best_valid_rank_at_20)
    if not 1 <= rank <= 21:
        raise ValueError("best_valid_rank_at_20 必须是 1～20 或 None")
    return math.log1p(min(rank - 1, 20)) / math.log(21)


def near_miss_score(near_miss_count_at_20: int, n95: float) -> float:
    """把相似干扰商品数量按 N 的对数公式压到 0 到 1 之间。"""

    count = max(0, int(near_miss_count_at_20))
    if n95 <= 0:
        return 0.0 if count == 0 else 1.0
    return min(1.0, math.log1p(count) / math.log1p(n95))


def build_task_features(
    *,
    task_id: int,
    goal: Mapping[str, Any],
    retrieval: RetrievalEvidence,
    a95: float,
    n95: float,
    category: str | None = None,
) -> TaskDifficultyFeatures:
    """把 Reward 编译后的任务字段与前 20 名检索证据合成一行难度特征。"""

    count, parts = independent_constraint_count(goal)
    c_score = constraint_score(count, a95)
    r_score = retrieval_score(retrieval.best_valid_rank_at_20)
    n_score = near_miss_score(retrieval.near_miss_count_at_20, n95)
    preliminary = 0.45 * c_score + 0.35 * r_score + 0.20 * n_score
    return TaskDifficultyFeatures(
        task_id=int(task_id),
        constraint_count=count,
        option_axis_count=int(parts["option_axis_count"]),
        has_brand=bool(parts["has_brand"]),
        has_model=bool(parts["has_model"]),
        has_budget=bool(parts["has_budget"]),
        retrieval_score=r_score,
        near_miss_score=n_score,
        constraint_score=c_score,
        preliminary_score=preliminary,
        category=category,
        evidence={
            **parts,
            "best_valid_rank_at_20": retrieval.best_valid_rank_at_20,
            "target_asin_rank_at_20": retrieval.target_asin_rank_at_20,
            "best_valid_outcome_type": retrieval.best_valid_outcome_type,
            "near_miss_count_at_20": retrieval.near_miss_count_at_20,
            "unverifiable_candidates_at_20": retrieval.unverifiable_candidates_at_20,
            "candidate_evaluation_reliable": retrieval.candidate_evaluation_reliable,
            "near_miss_evidence": [dict(item) for item in retrieval.near_miss_evidence],
        },
    )


def assign_preliminary_labels(
    features: Sequence[TaskDifficultyFeatures],
) -> list[TaskDifficultyFeatures]:
    """把规则分数最低约 30%、中间约 50%、最高约 20% 标成临时三档。"""

    if not features:
        return []
    scores = sorted(item.preliminary_score for item in features)
    easy_cut = _quantile(scores, 0.30)
    hard_cut = _quantile(scores, 0.80)
    result = []
    for item in features:
        label = (
            "easy"
            if item.preliminary_score <= easy_cut
            else "hard" if item.preliminary_score >= hard_cut else "medium"
        )
        values = item.to_dict()
        values["preliminary_label"] = label
        result.append(TaskDifficultyFeatures(**values))
    return result


def select_calibration_tasks(
    features: Sequence[TaskDifficultyFeatures],
    *,
    sample_size: int = 200,
    seed: int = 42,
) -> list[int]:
    """按临时难度和类别轮流取题，让约 200 道校准题覆盖不同结构。"""

    if sample_size < 1:
        raise ValueError("sample_size 必须至少为 1")
    groups: dict[tuple[str, str], list[TaskDifficultyFeatures]] = {}
    for item in features:
        key = (item.preliminary_label or "unlabelled", item.category or "unknown")
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        group.sort(key=lambda item: _stable_hash(seed, item.task_id))
    selected: list[int] = []
    ordered_keys = sorted(groups)
    while len(selected) < min(sample_size, len(features)):
        made_progress = False
        for key in ordered_keys:
            group = groups[key]
            if group and len(selected) < sample_size:
                selected.append(group.pop(0).task_id)
                made_progress = True
        if not made_progress:
            break
    return selected


def fit_logistic_difficulty_model(
    samples: Sequence[CalibrationSample],
    *,
    learning_rate: float = 0.05,
    epochs: int = 2_000,
    l2_strength: float = 0.05,
    version_context: Mapping[str, Any] | None = None,
) -> LogisticDifficultyModel:
    """用纯 Python 梯度下降拟合小型逻辑回归，不引入 sklearn 等额外依赖。"""

    if len(samples) < 2:
        raise ValueError("逻辑回归至少需要两个有效样本")
    labels = [1.0 if item.success else 0.0 for item in samples]
    if min(labels) == max(labels):
        raise ValueError("校准结果必须同时包含成功和失败")
    matrix = [item.features.vector() for item in samples]
    width = len(DIFFICULTY_FEATURE_NAMES)
    means = tuple(sum(row[index] for row in matrix) / len(matrix) for index in range(width))
    scales = tuple(_safe_scale([row[index] for row in matrix], means[index]) for index in range(width))
    standardised = [
        tuple(
            (row[index] - means[index]) / scales[index]
            for index in range(width)
        )
        for row in matrix
    ]
    coefficients = [0.0] * width
    intercept = 0.0
    for _ in range(int(epochs)):
        probabilities = [
            _sigmoid(intercept + sum(weight * value for weight, value in zip(coefficients, row)))
            for row in standardised
        ]
        errors = [prediction - label for prediction, label in zip(probabilities, labels)]
        intercept -= learning_rate * sum(errors) / len(errors)
        for index in range(width):
            gradient = (
                sum(error * row[index] for error, row in zip(errors, standardised))
                / len(errors)
                + l2_strength * coefficients[index]
            )
            coefficients[index] -= learning_rate * gradient
    context = {
        "feature_names": DIFFICULTY_FEATURE_NAMES,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "l2_strength": l2_strength,
        "version_context": dict(version_context or {}),
        "calibration_samples": sorted(
            (
                item.task_id,
                int(item.success),
                tuple(round(value, 12) for value in item.features.vector()),
            )
            for item in samples
        ),
        "coefficients": [round(value, 12) for value in coefficients],
        "intercept": round(intercept, 12),
    }
    version = "wlx-difficulty-" + hashlib.sha256(
        json.dumps(context, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return LogisticDifficultyModel(
        feature_names=DIFFICULTY_FEATURE_NAMES,
        means=means,
        scales=scales,
        coefficients=tuple(coefficients),
        intercept=intercept,
        difficulty_version=version,
    )


def evaluate_difficulty_model(
    model: LogisticDifficultyModel,
    samples: Sequence[CalibrationSample],
) -> dict[str, Any]:
    """按 task_id 汇总校准模型的准确率和 Log Loss，给出数据不足警告。"""

    if not samples:
        raise ValueError("评估模型时至少需要一个样本")
    probabilities = [model.predict_success(item.features) for item in samples]
    labels = [1 if item.success else 0 for item in samples]
    correct = sum((probability >= 0.5) == bool(label) for probability, label in zip(probabilities, labels))
    log_loss = -sum(
        label * math.log(max(probability, 1e-12))
        + (1 - label) * math.log(max(1.0 - probability, 1e-12))
        for probability, label in zip(probabilities, labels)
    ) / len(labels)
    success_count = sum(labels)
    failure_count = len(labels) - success_count
    warnings = []
    if success_count < 100:
        warnings.append("有效成功少于 100 次，建议增加不同任务")
    if failure_count < 100:
        warnings.append("有效失败少于 100 次，建议增加不同任务")
    return {
        "samples": len(samples),
        "tasks": len({item.task_id for item in samples}),
        "successes": success_count,
        "failures": failure_count,
        "accuracy_at_0_5": correct / len(labels),
        "log_loss": log_loss,
        "warnings": warnings,
    }


def grouped_task_split(
    samples: Sequence[CalibrationSample],
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[CalibrationSample], list[CalibrationSample]]:
    """按 task_id 切开逻辑回归训练和验证，三次尝试绝不会跨到两边。"""

    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio 必须在 0 到 1 之间")
    validation_ids = {
        task_id
        for task_id in {item.task_id for item in samples}
        if _stable_fraction(seed, task_id) < validation_ratio
    }
    training = [item for item in samples if item.task_id not in validation_ids]
    validation = [item for item in samples if item.task_id in validation_ids]
    return training, validation


def calibration_samples_from_rows(
    rows: Iterable[Mapping[str, Any]],
    features_by_task: Mapping[int, TaskDifficultyFeatures],
) -> list[CalibrationSample]:
    """只把有效尝试变成校准样本；基础设施故障不会被错误地当成失败。"""

    samples: list[CalibrationSample] = []
    for row in rows:
        if row.get("attempt_valid") is not True:
            continue
        success = row.get("task_success")
        if not isinstance(success, bool):
            raise ValueError("有效尝试必须明确记录 task_success=true 或 false")
        task_id = int(row["task_id"])
        try:
            features = features_by_task[task_id]
        except KeyError as exc:
            raise ValueError(f"task_id={task_id} 缺少难度特征") from exc
        samples.append(CalibrationSample(task_id, features, success))
    return samples


def calibration_completeness(
    samples: Sequence[CalibrationSample],
    *,
    expected_attempts: int = 3,
    expected_task_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """检查每道校准题是否真的得到固定三次有效尝试，并列出不足或超出的任务。"""

    if expected_attempts < 1:
        raise ValueError("expected_attempts 必须至少为 1")
    counts: dict[int, int] = {
        int(task_id): 0 for task_id in (expected_task_ids or ())
    }
    for sample in samples:
        counts[sample.task_id] = counts.get(sample.task_id, 0) + 1
    incomplete = {
        task_id: count for task_id, count in counts.items() if count < expected_attempts
    }
    excessive = {
        task_id: count for task_id, count in counts.items() if count > expected_attempts
    }
    return {
        "tasks": len(counts),
        "valid_attempts": len(samples),
        "expected_attempts_per_task": expected_attempts,
        "complete": not incomplete and not excessive,
        "incomplete_tasks": incomplete,
        "excessive_tasks": excessive,
    }


def fit_with_grouped_validation(
    samples: Sequence[CalibrationSample],
    *,
    validation_ratio: float = 0.2,
    seed: int = 42,
    version_context: Mapping[str, Any] | None = None,
    expected_task_ids: Iterable[int] | None = None,
) -> tuple[LogisticDifficultyModel, dict[str, Any]]:
    """先按 task_id 验证方法是否可靠，再用全部样本拟合正式难度模型。"""

    completeness = calibration_completeness(
        samples,
        expected_task_ids=expected_task_ids,
    )
    if not completeness["complete"]:
        raise ValueError("校准数据没有让每道计划任务都得到恰好 3 次有效尝试")
    training, validation = grouped_task_split(
        samples,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    if not training or not validation:
        raise ValueError("校准任务太少，无法按 task_id 切出训练和验证两侧")
    validation_model = fit_logistic_difficulty_model(
        training,
        version_context={**dict(version_context or {}), "purpose": "validation"},
    )
    report = {
        "training": evaluate_difficulty_model(validation_model, training),
        "validation": evaluate_difficulty_model(validation_model, validation),
        "completeness": completeness,
    }
    final_model = fit_logistic_difficulty_model(
        samples,
        version_context={**dict(version_context or {}), "purpose": "final"},
    )
    report["final_training"] = evaluate_difficulty_model(final_model, samples)
    report["difficulty_version"] = final_model.difficulty_version
    return final_model, report


def _safe_scale(values: Sequence[float], mean: float) -> float:
    """计算标准差；某个特征完全不变化时用 1，避免除以零。"""

    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return scale if scale > 1e-12 else 1.0


def _sigmoid(value: float) -> float:
    """用数值稳定的写法把任意实数变成 0 到 1 的概率。"""

    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


def _quantile(ordered_values: Sequence[float], fraction: float) -> float:
    """在已排序数字中做线性插值，得到指定比例位置的分数。"""

    if not ordered_values:
        raise ValueError("quantile 不能接收空列表")
    position = (len(ordered_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered_values[lower])
    weight = position - lower
    return float(ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight)


def _stable_hash(seed: int, task_id: int) -> str:
    """把随机种子和任务编号变成稳定哈希，保证重复运行选到同一批题。"""

    return hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()


def _stable_fraction(seed: int, task_id: int) -> float:
    """把稳定哈希映射到 0 到 1，供按任务切分训练和验证。"""

    integer = int(_stable_hash(seed, task_id)[:16], 16)
    return integer / float(16**16)


__all__ = [
    "CalibrationSample",
    "DIFFICULTY_FEATURE_NAMES",
    "LogisticDifficultyModel",
    "RetrievalEvidence",
    "TaskDifficultyFeatures",
    "assign_preliminary_labels",
    "build_task_features",
    "calibration_completeness",
    "calibration_samples_from_rows",
    "constraint_score",
    "evaluate_difficulty_model",
    "fit_logistic_difficulty_model",
    "fit_with_grouped_validation",
    "grouped_task_split",
    "independent_constraint_count",
    "near_miss_score",
    "percentile_95",
    "retrieval_score",
    "select_calibration_tasks",
]
