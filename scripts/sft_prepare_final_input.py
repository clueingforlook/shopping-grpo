#!/usr/bin/env python3
"""把多批 raw 派生成统一使用最终难度标签的单一构建输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_RAW_NAME = "merged-raw-v3.jsonl"
OUTPUT_MANIFEST_NAME = "merge-manifest.json"
MERGE_VERSION = "wlx-sft-final-input-v1"
DIFFICULTY_FIELDS = (
    "difficulty_label",
    "difficulty_score",
    "difficulty_version",
    "category",
    "official_split",
)
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def build_parser() -> argparse.ArgumentParser:
    """创建只负责合并与 v3 重标的离线命令。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        action="append",
        required=True,
        help="按采样批次顺序重复传入；不会修改源文件",
    )
    parser.add_argument("--difficulty-plan", type=Path, required=True)
    parser.add_argument("--held-out-tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """执行合并并打印不含轨迹正文的摘要。"""

    args = build_parser().parse_args(argv)
    summary = prepare_final_input(
        raw_paths=args.raw,
        difficulty_plan_path=args.difficulty_plan,
        held_out_tasks_path=args.held_out_tasks,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def prepare_final_input(
    *,
    raw_paths: Sequence[str | Path],
    difficulty_plan_path: str | Path,
    held_out_tasks_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """复制所有 raw，并只用公开最终计划统一其难度元数据。"""

    sources = [Path(path) for path in raw_paths]
    if not sources:
        raise ValueError("至少需要一份 --raw")
    if len({path.resolve() for path in sources}) != len(sources):
        raise ValueError("--raw 不能重复传入同一个文件")
    for path in sources:
        _require_file(path, "raw")

    difficulty_plan_path = Path(difficulty_plan_path)
    held_out_tasks_path = Path(held_out_tasks_path)
    _require_file(difficulty_plan_path, "difficulty plan")
    _require_file(held_out_tasks_path, "held-out tasks")
    plan, plan_version = _load_difficulty_plan(difficulty_plan_path)
    held_out_ids = _load_task_ids(held_out_tasks_path)

    output_dir = Path(output_dir)
    _prepare_new_output_directory(output_dir)
    output_raw = output_dir / OUTPUT_RAW_NAME
    output_manifest = output_dir / OUTPUT_MANIFEST_NAME
    temporary_raw = output_dir / f"{OUTPUT_RAW_NAME}.tmp"
    temporary_manifest = output_dir / f"{OUTPUT_MANIFEST_NAME}.tmp"

    source_reports: list[dict[str, Any]] = []
    trajectory_ids: set[str] = set()
    first_source_by_task: dict[int, int] = {}
    unique_task_labels: dict[int, str] = {}
    relabel_transitions: Counter[str] = Counter()
    total_rows = 0

    try:
        with temporary_raw.open("w", encoding="utf-8") as output_handle:
            for source_index, source in enumerate(sources):
                source_rows = 0
                source_tasks: set[int] = set()
                for line_number, trajectory in _read_jsonl(source):
                    source_rows += 1
                    total_rows += 1
                    task_id = _required_int(trajectory, "task_id", source, line_number)
                    trajectory_id = str(trajectory.get("trajectory_id") or "").strip()
                    if not trajectory_id:
                        raise ValueError(f"{source}:{line_number}: 缺少 trajectory_id")
                    if trajectory_id in trajectory_ids:
                        raise ValueError(f"发现重复 trajectory_id：{trajectory_id}")
                    trajectory_ids.add(trajectory_id)
                    source_tasks.add(task_id)

                    previous_source = first_source_by_task.setdefault(task_id, source_index)
                    if previous_source != source_index:
                        raise ValueError(
                            f"task_id={task_id} 同时出现在两份 raw；请检查是否误加了已汇总分片"
                        )
                    if task_id in held_out_ids:
                        raise ValueError(f"task_id={task_id} 与 held-out evaluation 重叠")
                    difficulty = plan.get(task_id)
                    if difficulty is None:
                        raise ValueError(f"task_id={task_id} 不在最终 difficulty plan 中")

                    derived = _apply_difficulty(trajectory, difficulty, source, line_number)
                    request = derived["stage_metadata"]["request"]
                    original_label = _request_label(trajectory)
                    final_label = str(request["difficulty_label"])
                    relabel_transitions[f"{original_label}->{final_label}"] += 1
                    unique_task_labels[task_id] = final_label
                    output_handle.write(
                        json.dumps(derived, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                source_reports.append(
                    {
                        "path": _portable_path(source),
                        "rows": source_rows,
                        "unique_tasks": len(source_tasks),
                        "sha256": file_sha256(source),
                    }
                )
            output_handle.flush()
            os.fsync(output_handle.fileno())

        output_sha = file_sha256(temporary_raw)
        difficulty_distribution = Counter(unique_task_labels.values())
        manifest = {
            "merge_version": MERGE_VERSION,
            "operation": "copy raw rows and replace public difficulty metadata from final plan",
            "raw_sources": source_reports,
            "difficulty_plan": {
                "path": _portable_path(difficulty_plan_path),
                "rows": len(plan),
                "difficulty_version": plan_version,
                "sha256": file_sha256(difficulty_plan_path),
            },
            "held_out_tasks": {
                "path": _portable_path(held_out_tasks_path),
                "rows": len(held_out_ids),
                "sha256": file_sha256(held_out_tasks_path),
            },
            "overwritten_request_fields": list(DIFFICULTY_FIELDS),
            "rows": total_rows,
            "unique_trajectories": len(trajectory_ids),
            "unique_tasks": len(unique_task_labels),
            "held_out_overlap": 0,
            "difficulty_distribution_by_task": dict(sorted(difficulty_distribution.items())),
            "raw_to_final_label_transitions_by_row": dict(sorted(relabel_transitions.items())),
            "output": {
                "path": _portable_path(output_raw),
                "rows": total_rows,
                "sha256": output_sha,
            },
        }
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_raw.replace(output_raw)
        temporary_manifest.replace(output_manifest)
    except Exception:
        temporary_raw.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise

    return {
        "merge_version": MERGE_VERSION,
        "rows": total_rows,
        "unique_tasks": len(unique_task_labels),
        "difficulty_distribution_by_task": dict(sorted(difficulty_distribution.items())),
        "held_out_overlap": 0,
        "raw_path": _portable_path(output_raw),
        "raw_sha256": output_sha,
        "manifest_path": _portable_path(output_manifest),
    }


def _load_difficulty_plan(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    """读取最终公开难度计划并确认标签和版本完整一致。"""

    rows: dict[int, dict[str, Any]] = {}
    versions: set[str] = set()
    for line_number, value in _read_jsonl(path):
        task_id = _required_int(value, "task_id", path, line_number)
        if task_id in rows:
            raise ValueError(f"{path}:{line_number}: 重复 task_id={task_id}")
        missing = [field for field in DIFFICULTY_FIELDS if field not in value]
        if missing:
            raise ValueError(f"{path}:{line_number}: 缺少字段 {missing}")
        label = str(value["difficulty_label"])
        if label not in VALID_DIFFICULTIES:
            raise ValueError(f"{path}:{line_number}: 非法 difficulty_label={label}")
        if str(value["official_split"]) != "train":
            raise ValueError(f"{path}:{line_number}: 最终采样计划必须只包含 train")
        version = str(value["difficulty_version"] or "").strip()
        if not version:
            raise ValueError(f"{path}:{line_number}: difficulty_version 为空")
        float(value["difficulty_score"])
        versions.add(version)
        rows[task_id] = {field: deepcopy(value[field]) for field in DIFFICULTY_FIELDS}
    if not rows:
        raise ValueError(f"difficulty plan 为空：{path}")
    if len(versions) != 1:
        raise ValueError(f"difficulty plan 包含多个版本：{sorted(versions)}")
    return rows, next(iter(versions))


def _load_task_ids(path: Path) -> set[int]:
    """读取 held-out task_id；只用于阻止训练数据泄漏。"""

    task_ids: set[int] = set()
    for line_number, value in _read_jsonl(path):
        task_id = _required_int(value, "task_id", path, line_number)
        if task_id in task_ids:
            raise ValueError(f"{path}:{line_number}: 重复 task_id={task_id}")
        task_ids.add(task_id)
    return task_ids


def _apply_difficulty(
    trajectory: Mapping[str, Any],
    difficulty: Mapping[str, Any],
    source: Path,
    line_number: int,
) -> dict[str, Any]:
    """复制一条轨迹并覆盖 request 中的公开难度字段。"""

    derived = deepcopy(dict(trajectory))
    stage = derived.get("stage_metadata")
    if not isinstance(stage, Mapping):
        raise ValueError(f"{source}:{line_number}: stage_metadata 不是对象")
    request = stage.get("request")
    if not isinstance(request, Mapping):
        raise ValueError(f"{source}:{line_number}: stage_metadata.request 不是对象")
    clean_stage = deepcopy(dict(stage))
    clean_request = deepcopy(dict(request))
    clean_request.update({field: deepcopy(difficulty[field]) for field in DIFFICULTY_FIELDS})
    clean_stage["request"] = clean_request
    derived["stage_metadata"] = clean_stage
    return derived


def _request_label(trajectory: Mapping[str, Any]) -> str:
    """读取原始标签，仅用于 manifest 的迁移审计。"""

    stage = trajectory.get("stage_metadata")
    request = stage.get("request") if isinstance(stage, Mapping) else None
    if not isinstance(request, Mapping):
        return "missing"
    return str(request.get("difficulty_label") or "unlabelled")


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """逐行读取 JSONL，并对空行、坏 JSON 和非对象立即报错。"""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: JSONL 含空行")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 无效") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL 行必须是对象")
            yield line_number, value


def _required_int(value: Mapping[str, Any], key: str, path: Path, line_number: int) -> int:
    """读取必需整数，拒绝布尔值和缺失值。"""

    raw = value.get(key)
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"{path}:{line_number}: {key} 不是整数")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number}: {key} 不是整数") from exc


def _prepare_new_output_directory(path: Path) -> None:
    """只允许写入新目录，绝不覆盖已有筛选产物。"""

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录已经有内容，请换新版本目录：{path}")
    path.mkdir(parents=True, exist_ok=True)


def _require_file(path: Path, label: str) -> None:
    """确认输入是普通文件。"""

    if not path.is_file():
        raise FileNotFoundError(f"找不到 {label}：{path}")


def _portable_path(path: Path) -> str:
    """仓库内只记录相对路径，避免把本机用户名写入 manifest。"""

    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return path.name


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
