"""负责在新版 Core 轨迹和项目原有 JSONL 轨迹之间做明确转换。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from wlx_harness_core.wlx_contracts import TRAJECTORY_SCHEMA_VERSION, ToolCall


class TrajectorySerializationError(ValueError):
    """表示轨迹格式不受支持，或轨迹内容缺少转换所需的数据。"""

    pass


def trajectory_to_legacy(trajectory: object) -> dict[str, Any]:
    """接收轨迹对象或字典，返回旧版 SFT 和评测代码能读取的轨迹字典。"""

    converter = getattr(trajectory, "to_legacy_dict", None)
    if callable(converter):
        value = converter()
        if not isinstance(value, Mapping):
            raise TypeError("trajectory.to_legacy_dict() must return a mapping")
        return deepcopy(dict(value))
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be a mapping or provide to_legacy_dict()")
    schema_version = trajectory.get("schema_version")
    if schema_version == TRAJECTORY_SCHEMA_VERSION:
        return core_json_to_legacy(trajectory)
    if isinstance(schema_version, str) and schema_version.startswith("wlx-harness-"):
        raise TrajectorySerializationError(
            f"unsupported WLX trajectory schema: {schema_version!r}"
        )
    return deepcopy(dict(trajectory))


def core_json_to_legacy(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    """把一条带新版 schema_version 的 Core 字典转换成旧版轨迹字典。"""

    if trajectory.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise TrajectorySerializationError("mapping is not a supported Core trajectory")
    result = deepcopy(dict(trajectory))
    result["trajectory_schema_version"] = result.pop("schema_version")
    result.pop("contract_version", None)
    raw_steps = result.get("steps") or []
    if not isinstance(raw_steps, list):
        raise TrajectorySerializationError("Core trajectory steps must be a list")
    result["steps"] = [_core_step_to_legacy(item, index) for index, item in enumerate(raw_steps)]
    return result


def _core_step_to_legacy(value: object, index: int) -> dict[str, Any]:
    """转换 Core 轨迹中的一个步骤；输入步骤内容和位置，返回旧版步骤字典。"""

    if not isinstance(value, Mapping):
        raise TrajectorySerializationError(f"Core trajectory step {index} must be an object")
    raw_call = value.get("tool_call")
    if not isinstance(raw_call, Mapping):
        raise TrajectorySerializationError(f"Core trajectory step {index} is missing tool_call")
    try:
        call = ToolCall(
            call_id=str(raw_call.get("call_id") or ""),
            name=str(raw_call.get("name") or ""),
            arguments=dict(raw_call.get("arguments") or {}),
            call_type=str(raw_call.get("call_type") or "function"),
        )
    except (TypeError, ValueError) as exc:
        raise TrajectorySerializationError(
            f"Core trajectory step {index} has an invalid tool_call"
        ) from exc
    step = {
        "step_index": int(value.get("step_index", index)),
        "tool_call": call.to_openai_dict(),
        "tool_name": call.name,
        "parameters": deepcopy(dict(call.arguments)),
        "env_action": value.get("env_action"),
        "observation": str(value.get("observation") or ""),
        "reward": float(value.get("reward", 0.0) or 0.0),
        "done": bool(value.get("done", False)),
        "result": deepcopy(dict(value.get("result") or {})),
    }
    for name in ("raw_observation", "projection", "error"):
        if value.get(name) is not None:
            step[name] = deepcopy(value[name])
    return step


__all__ = [
    "TrajectorySerializationError",
    "core_json_to_legacy",
    "trajectory_to_legacy",
]
