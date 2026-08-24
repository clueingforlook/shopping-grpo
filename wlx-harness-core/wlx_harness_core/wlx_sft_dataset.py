"""从 raw.jsonl 独立构建严格 Gold、无泄漏、可复现的 SFT 数据集。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlx_harness_core.wlx_contracts import ToolCall
from wlx_harness_core.wlx_reward import RewardContractError, validate_terminal_result
from wlx_harness_core.wlx_sft_contracts import (
    SftDisposition,
    WLX_SFT_PIPELINE_VERSION,
)
from wlx_harness_core.wlx_sft_metrics import (
    TrainingSequenceUnrenderable,
    TrainingTokenCounter,
    measure_training_length,
    trajectory_cost_metrics,
)
from wlx_harness_core.wlx_sft_outcomes import (
    classify_attempt,
    sft_mechanical_rejection_reasons,
)
from wlx_harness_core.wlx_sft_storage import (
    file_sha256,
    jsonl_line_count,
    load_held_out_task_ids,
    read_jsonl,
    write_jsonl,
)
from wlx_harness_core.wlx_tools import (
    WLX_SFT_TOOL_REGISTRY,
    ToolRegistry,
)


WLX_SFT_DATASET_VERSION = "wlx-sft-dataset-v1"
_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}
_TOOL_CALL_KEYS = {"id", "type", "function"}
_FUNCTION_KEYS = {"name", "arguments"}
_SECRET_KEY_PARTS = ("api_key", "apikey", "password", "secret", "authorization")


@dataclass(frozen=True)
class DatasetBuildConfig:
    """规定最终长度、切分、任务配比和每题最少有效探索步。"""

    context_limit: int = 24_576
    validation_ratio: float = 0.1
    seed: int = 42
    min_productive_steps: int = 3
    target_size: int | None = None
    target_difficulty_ratios: Mapping[str, float] = field(
        default_factory=lambda: {"easy": 0.30, "medium": 0.50, "hard": 0.20}
    )

    def __post_init__(self) -> None:
        """检查构建参数，避免用错误比例或超小上下文生成数据集。"""

        if self.context_limit < 1:
            raise ValueError("context_limit 必须至少为 1")
        if not 0.0 <= self.validation_ratio < 1.0:
            raise ValueError("validation_ratio 必须在 0 到 1 之间")
        if self.min_productive_steps < 0:
            raise ValueError("min_productive_steps 不能小于 0")
        if self.target_size is not None and self.target_size < 1:
            raise ValueError("target_size 必须至少为 1")
        ratios = {str(key): float(value) for key, value in self.target_difficulty_ratios.items()}
        if not ratios or any(value < 0 for value in ratios.values()):
            raise ValueError("难度配比必须是非负数字")
        total = sum(ratios.values())
        if total <= 0:
            raise ValueError("难度配比之和必须大于 0")
        object.__setattr__(
            self,
            "target_difficulty_ratios",
            {key: value / total for key, value in ratios.items()},
        )


def strict_gold_rejection_reasons(
    trajectory: Mapping[str, Any],
    *,
    tool_registry: ToolRegistry = WLX_SFT_TOOL_REGISTRY,
) -> list[str]:
    """用纯机械规则列出轨迹不能进入主 SFT 的全部原因。"""

    reasons: list[str] = []
    decision = classify_attempt(trajectory)
    if decision.sft_disposition != SftDisposition.ACCEPTED_GOLD:
        reasons.append(f"not_strict_gold:{decision.outcome_type.value}")
    if trajectory.get("error"):
        reasons.append("has_error")
    if trajectory.get("release_error"):
        reasons.append("release_error")
    if trajectory.get("status") != "done" or trajectory.get("done") is not True:
        reasons.append("trajectory_not_done")
    reasons.extend(sft_mechanical_rejection_reasons(trajectory))

    terminal = trajectory.get("terminal_result")
    if not isinstance(terminal, Mapping):
        reasons.append("missing_terminal_result")
    else:
        try:
            reward = validate_terminal_result(terminal)
        except RewardContractError:
            reasons.append("invalid_reward_contract")
        else:
            if reward.get("reward_type") != "gold_purchase":
                reasons.append("reward_not_gold")
            if reward.get("reward_valid") is not True:
                reasons.append("reward_not_valid")
            if reward.get("purchase_success") is not True:
                reasons.append("purchase_not_successful")
            if reward.get("target_asin_match") is not True:
                reasons.append("target_asin_not_matched")

    steps = [item for item in trajectory.get("steps") or [] if isinstance(item, Mapping)]
    if not any(step.get("done") is True and step.get("env_action") == "click[Buy Now]" for step in steps):
        reasons.append("missing_terminal_buy_step")
    for index, step in enumerate(steps):
        step_reason = _step_rejection_reason(trajectory, step, tool_registry)
        if step_reason:
            reasons.append(f"step_{index}:{step_reason}")
    for index, message in enumerate(trajectory.get("messages") or []):
        if not isinstance(message, Mapping):
            reasons.append(f"message_{index}:not_object")
            continue
        if message.get("role") == "assistant" and len(message.get("tool_calls") or []) > 1:
            reasons.append(f"message_{index}:multiple_tool_calls")
    return list(dict.fromkeys(reasons))


def build_training_row(
    trajectory: Mapping[str, Any],
    *,
    tool_registry: ToolRegistry = WLX_SFT_TOOL_REGISTRY,
) -> dict[str, Any]:
    """删除私有推理和审计字段，保留公开推理、工具调用与模型可见 Observation。"""

    terminal_call_id = _terminal_tool_call_id(trajectory)
    messages = [
        _sanitise_message(message, terminal_call_id)
        for message in trajectory.get("messages") or []
        if isinstance(message, Mapping)
    ]
    request_metadata = _request_metadata(trajectory)
    token_metrics = dict(trajectory.get("token_metrics") or {})
    cost_metrics = trajectory_cost_metrics(trajectory)
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "task_id": int(trajectory["task_id"]),
        "messages": messages,
        "tools": tool_registry.schemas,
        "metadata": {
            "outcome_type": "gold_purchase",
            "difficulty_label": request_metadata.get("difficulty_label"),
            "difficulty_score": request_metadata.get("difficulty_score"),
            "difficulty_version": request_metadata.get("difficulty_version"),
            "category": request_metadata.get("category"),
            "attempt_index": int(trajectory.get("attempt_index", 0)),
            **token_metrics,
            **cost_metrics,
        },
    }


def build_dataset_artifacts(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    held_out_tasks_path: str | Path,
    training_token_counter: TrainingTokenCounter,
    config: DatasetBuildConfig | None = None,
    collection_config: Mapping[str, Any] | None = None,
    tool_registry: ToolRegistry = WLX_SFT_TOOL_REGISTRY,
    artifact_prefix: str = "",
) -> dict[str, Any]:
    """从唯一事实源 raw.jsonl 生成清洗、配比、切分、统计和哈希文件。"""

    config = config or DatasetBuildConfig()
    if not callable(training_token_counter):
        raise TypeError("必须提供最终训练模型的 training_token_counter")
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到原始轨迹：{raw_path}")
    output_dir = Path(output_dir)
    artifact_prefix = _validate_artifact_prefix(artifact_prefix)
    _prepare_new_output_directory(output_dir)
    held_out_ids = load_held_out_task_ids(held_out_tasks_path)
    safe_config = _safe_collection_config(collection_config or {})

    eligible_by_task: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    rejected: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    rejection_counter: Counter[str] = Counter()
    total = 0
    for trajectory in read_jsonl(raw_path):
        total += 1
        task_id = int(trajectory["task_id"])
        if (
            task_id not in held_out_ids
            and trajectory.get("sft_disposition")
            == SftDisposition.ALTERNATIVE_AUDIT.value
        ):
            alternatives.append(trajectory)
        reasons = strict_gold_rejection_reasons(trajectory, tool_registry=tool_registry)
        if task_id in held_out_ids:
            reasons.append("held_out_task")
        if reasons:
            _record_rejection(rejected, rejection_counter, trajectory, reasons)
            continue
        row = build_training_row(trajectory, tool_registry=tool_registry)
        try:
            training_length = measure_training_length(
                row,
                training_token_counter,
                context_limit=config.context_limit,
            )
        except TrainingSequenceUnrenderable:
            _record_rejection(
                rejected,
                rejection_counter,
                trajectory,
                ["sft_chat_template_unrenderable"],
            )
            continue
        row["metadata"].update(training_length.to_dict())
        productive_steps = int(row["metadata"].get("productive_steps", 0))
        length_reasons = []
        if not training_length.within_limit:
            length_reasons.append("sft_sequence_over_context_limit")
        if productive_steps < config.min_productive_steps:
            length_reasons.append("too_few_productive_steps")
        if length_reasons:
            _record_rejection(rejected, rejection_counter, trajectory, length_reasons)
            continue
        eligible_by_task.setdefault(task_id, []).append((trajectory, row))

    chosen_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id, pairs in eligible_by_task.items():
        pairs.sort(key=lambda pair: _trajectory_preference_key(pair[0]))
        chosen_pairs.append(pairs[0])
        for duplicate_trajectory, _ in pairs[1:]:
            _record_rejection(
                rejected,
                rejection_counter,
                duplicate_trajectory,
                ["duplicate_gold_for_same_task"],
            )

    selected_rows = _balance_rows(
        [pair[1] for pair in chosen_pairs],
        target_size=config.target_size,
        ratios=config.target_difficulty_ratios,
        seed=config.seed,
    )
    selected_ids = {str(row.get("trajectory_id")) for row in selected_rows}
    selected_trajectories = [
        trajectory
        for trajectory, row in chosen_pairs
        if str(row.get("trajectory_id")) in selected_ids
    ]
    for trajectory, row in chosen_pairs:
        if str(row.get("trajectory_id")) not in selected_ids:
            _record_rejection(
                rejected,
                rejection_counter,
                trajectory,
                ["not_selected_by_dataset_balance"],
            )

    train_rows, validation_rows = _stratified_task_split(
        selected_rows,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    paths = {
        "accepted": output_dir / f"{artifact_prefix}accepted.jsonl",
        "alternatives": output_dir / f"{artifact_prefix}alternatives.jsonl",
        "rejected": output_dir / f"{artifact_prefix}rejected.jsonl",
        "sft": output_dir / f"{artifact_prefix}sft.jsonl",
        "train": output_dir / f"{artifact_prefix}train.jsonl",
        "validation": output_dir / f"{artifact_prefix}validation.jsonl",
        "stats": output_dir / f"{artifact_prefix}reject-stats.json",
        "metadata": output_dir / f"{artifact_prefix}metadata.json",
    }
    write_jsonl(paths["accepted"], selected_trajectories)
    write_jsonl(paths["alternatives"], alternatives)
    write_jsonl(paths["rejected"], rejected)
    write_jsonl(paths["sft"], selected_rows)
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["validation"], validation_rows)

    summary = {
        "dataset_version": WLX_SFT_DATASET_VERSION,
        "pipeline_version": WLX_SFT_PIPELINE_VERSION,
        "raw_rows": total,
        "eligible_gold_tasks": len(eligible_by_task),
        "accepted": len(selected_rows),
        "alternatives_for_audit": len(alternatives),
        "rejected": len(rejected),
        "train": len(train_rows),
        "validation": len(validation_rows),
        "reject_reasons": dict(sorted(rejection_counter.items())),
        "distribution": _distribution_report(selected_rows),
    }
    _write_json(paths["stats"], summary)
    metadata = {
        **summary,
        "environment": "shopsimulator-environment-v2.1",
        "reward": "shopsimulator-reward-v3",
        "tool_schema_fingerprint": tool_registry.fingerprint,
        "context_limit": config.context_limit,
        "validation_ratio": config.validation_ratio,
        "split_seed": config.seed,
        "target_size": config.target_size,
        "artifact_prefix": artifact_prefix,
        "target_difficulty_ratios": dict(config.target_difficulty_ratios),
        "collection_config": safe_config,
        "raw": {
            "path": str(raw_path),
            "rows": total,
            "sha256": file_sha256(raw_path),
        },
        "files": {
            name: {
                "path": str(path),
                "rows": jsonl_line_count(path),
                "sha256": file_sha256(path),
            }
            for name, path in paths.items()
            if name not in {"stats", "metadata"}
        },
    }
    _write_json(paths["metadata"], metadata)
    return summary


def _step_rejection_reason(
    trajectory: Mapping[str, Any],
    step: Mapping[str, Any],
    registry: ToolRegistry,
) -> str | None:
    """核对工具名、参数和实际环境动作是否一一对应。"""

    try:
        call = _step_tool_call(step)
        call = registry.parse_tool_call(call)
    except (TypeError, ValueError, KeyError):
        return "invalid_tool_call"
    if call.name == "think":
        return "think_tool_not_allowed"
    try:
        expected = registry.to_action(call)
    except Exception:
        return "tool_action_mapping_failed"
    if expected != step.get("env_action"):
        return "tool_action_mismatch"
    previous_observation = _previous_observation(trajectory, call.call_id)
    if previous_observation:
        try:
            guard_rejection = registry.guard(call, previous_observation)
        except Exception:
            return "tool_guard_check_failed"
        if guard_rejection:
            return "tool_not_allowed_by_visible_observation"
    if step.get("error"):
        return "step_has_error"
    return None


def _step_tool_call(step: Mapping[str, Any]) -> ToolCall | Mapping[str, Any]:
    """把 Core step 里的简洁工具格式还原成统一 ToolCall 对象。"""

    raw = step.get("tool_call")
    if not isinstance(raw, Mapping):
        raise ValueError("step 缺少 tool_call")
    if raw.get("name"):
        return ToolCall(
            call_id=str(raw.get("call_id") or ""),
            name=str(raw["name"]),
            arguments=dict(raw.get("arguments") or {}),
            call_type=str(raw.get("call_type") or "function"),
        )
    return raw


def _terminal_tool_call_id(trajectory: Mapping[str, Any]) -> str | None:
    """找到真正让环境终止的购买工具编号，用它替换 Reward 工具回包。"""

    terminal_steps = [
        step
        for step in trajectory.get("steps") or []
        if isinstance(step, Mapping) and step.get("done") is True
    ]
    if not terminal_steps:
        return None
    call = terminal_steps[-1].get("tool_call")
    if not isinstance(call, Mapping):
        return None
    return str(call.get("call_id") or call.get("id") or "") or None


def _previous_observation(
    trajectory: Mapping[str, Any],
    tool_call_id: str,
) -> str:
    """找到某次 assistant 工具调用之前最后一条模型可见 Tool Observation。"""

    messages = trajectory.get("messages") or []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if not any(
            isinstance(call, Mapping) and str(call.get("id") or "") == tool_call_id
            for call in calls
        ):
            continue
        for previous in reversed(messages[:index]):
            if isinstance(previous, Mapping) and previous.get("role") == "tool":
                return str(previous.get("content") or "")
        return ""
    return ""


def _sanitise_message(message: Mapping[str, Any], terminal_call_id: str | None) -> dict[str, Any]:
    """按白名单保留模型真正看过的字段，并删除 reasoning_content 等私有内容。"""

    clean = {key: deepcopy(message[key]) for key in _MESSAGE_KEYS if key in message}
    if clean.get("role") == "tool" and clean.get("tool_call_id") == terminal_call_id:
        clean["content"] = "购买已完成。"
    if isinstance(clean.get("tool_calls"), list):
        clean["tool_calls"] = [_sanitise_tool_call(item) for item in clean["tool_calls"]]
    return clean


def _sanitise_tool_call(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    """只保留 OpenAI Tool Call 必需字段，去掉服务商返回的内部诊断数据。"""

    clean = {key: deepcopy(tool_call[key]) for key in _TOOL_CALL_KEYS if key in tool_call}
    function = clean.get("function")
    if isinstance(function, Mapping):
        clean["function"] = {
            key: deepcopy(function[key]) for key in _FUNCTION_KEYS if key in function
        }
    return clean


def _request_metadata(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """读取采样时写入的公开难度与类别，不读取环境隐藏目标。"""

    stage = trajectory.get("stage_metadata")
    if not isinstance(stage, Mapping):
        return {}
    request = stage.get("request")
    return dict(request) if isinstance(request, Mapping) else {}


def _record_rejection(
    rejected: list[dict[str, Any]],
    counter: Counter[str],
    trajectory: Mapping[str, Any],
    reasons: Sequence[str],
) -> None:
    """保存一条拒绝摘要，并把每个原因加入批次统计。"""

    unique_reasons = list(dict.fromkeys(str(reason) for reason in reasons))
    counter.update(unique_reasons)
    rejected.append(
        {
            "trajectory_id": trajectory.get("trajectory_id"),
            "task_id": int(trajectory["task_id"]),
            "attempt_index": int(trajectory.get("attempt_index", 0)),
            "outcome_type": trajectory.get("outcome_type"),
            "status": trajectory.get("status"),
            "reject_reasons": unique_reasons,
        }
    )


def _trajectory_preference_key(trajectory: Mapping[str, Any]) -> tuple[int, int, str]:
    """同一任务有多个 Gold 时稳定选择最早的有效尝试，不按偶然长度挑答案。"""

    return (
        int(trajectory.get("attempt_index", 0)),
        int(trajectory.get("technical_retry_index", 0)),
        str(trajectory.get("trajectory_id") or ""),
    )


def _balance_rows(
    rows: Sequence[dict[str, Any]],
    *,
    target_size: int | None,
    ratios: Mapping[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    """按 easy/medium/hard 目标数挑任务；没给目标总数时不强行丢弃数据。"""

    if target_size is None or target_size >= len(rows):
        return sorted((deepcopy(row) for row in rows), key=lambda row: int(row["task_id"]))
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str((row.get("metadata") or {}).get("difficulty_label") or "unlabelled")
        groups.setdefault(label, []).append(deepcopy(row))
    for group in groups.values():
        group.sort(key=lambda row: _stable_hash(seed, int(row["task_id"])))
    desired = _ratio_quotas(target_size, ratios)
    selected: list[dict[str, Any]] = []
    for label, quota in desired.items():
        selected.extend(groups.get(label, [])[:quota])
        groups[label] = groups.get(label, [])[quota:]
    leftovers = sorted(
        (row for group in groups.values() for row in group),
        key=lambda row: _stable_hash(seed, int(row["task_id"])),
    )
    selected.extend(leftovers[: max(0, target_size - len(selected))])
    return sorted(selected, key=lambda row: int(row["task_id"]))


def _ratio_quotas(target_size: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """用最大余数法把小数配比变成总和恰好等于 target_size 的整数名额。"""

    raw = {label: target_size * ratio for label, ratio in ratios.items()}
    quotas = {label: math.floor(value) for label, value in raw.items()}
    missing = target_size - sum(quotas.values())
    order = sorted(raw, key=lambda label: (-(raw[label] - quotas[label]), label))
    for label in order[:missing]:
        quotas[label] += 1
    return quotas


def _stratified_task_split(
    rows: Sequence[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """在每个难度档内按 task_id 稳定切分，绝不把同一道题拆到两侧。"""

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str((row.get("metadata") or {}).get("difficulty_label") or "unlabelled")
        groups.setdefault(label, []).append(deepcopy(row))
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group in groups.values():
        group.sort(key=lambda row: _stable_hash(seed, int(row["task_id"])))
        count = round(len(group) * validation_ratio)
        if validation_ratio > 0 and len(group) > 1:
            count = max(1, min(count, len(group) - 1))
        validation.extend(group[:count])
        train.extend(group[count:])
    train.sort(key=lambda row: int(row["task_id"]))
    validation.sort(key=lambda row: int(row["task_id"]))
    return train, validation


def _distribution_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """同时统计任务难度和轨迹长度桶，方便发现 easy+long 等异常偏斜。"""

    difficulty = Counter()
    length = Counter()
    cross = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        difficulty_label = str(metadata.get("difficulty_label") or "unlabelled")
        length_label = str(metadata.get("trajectory_length_bucket") or "unknown")
        difficulty[difficulty_label] += 1
        length[length_label] += 1
        cross[f"{difficulty_label}+{length_label}"] += 1
    return {
        "difficulty": dict(sorted(difficulty.items())),
        "trajectory_length": dict(sorted(length.items())),
        "difficulty_x_length": dict(sorted(cross.items())),
    }


def _safe_collection_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """拒绝把 API Key、密码或 Authorization 写进可分享的数据集元数据。"""

    result: dict[str, Any] = {}
    for key, item in value.items():
        lowered = str(key).casefold()
        if any(part in lowered for part in _SECRET_KEY_PARTS):
            raise ValueError(f"collection_config 不能包含秘密字段：{key}")
        if isinstance(item, Mapping):
            result[str(key)] = _safe_collection_config(item)
        else:
            result[str(key)] = deepcopy(item)
    return result


def _prepare_new_output_directory(path: Path) -> None:
    """要求每次冻结使用新目录，防止旧数据集被静默覆盖。"""

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录已经有内容，请换一个新版本目录：{path}")
    path.mkdir(parents=True, exist_ok=True)


def _validate_artifact_prefix(value: object) -> str:
    """只允许简单文件名前缀；非空前缀必须符合本机 WLX 文件约定。"""

    prefix = str(value or "")
    if not prefix:
        return ""
    if not prefix.startswith(("wlx_", "wlx-")):
        raise ValueError("artifact_prefix 必须以 wlx_ 或 wlx- 开头")
    if Path(prefix).name != prefix or any(character in prefix for character in ("/", "\\")):
        raise ValueError("artifact_prefix 不能包含路径分隔符")
    return prefix


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """先写临时文件再原子替换，避免进程中断留下半份统计 JSON。"""

    temporary = path.with_name(f"{path.name}.wlx_tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _stable_hash(seed: int, task_id: int) -> str:
    """为任务生成稳定排序键，让同样配置重复构建得到完全相同的切分。"""

    return hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()


__all__ = [
    "DatasetBuildConfig",
    "WLX_SFT_DATASET_VERSION",
    "build_dataset_artifacts",
    "build_training_row",
    "strict_gold_rejection_reasons",
]
