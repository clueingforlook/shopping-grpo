#!/usr/bin/env python3
"""把现有 原始轨迹和最终 SFT 冻结为两类清晰、可审计的数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
ARCHIVE_VERSION = "wlx-sft-data-archive-v1"
RECORD_SCHEMA_VERSION = "wlx-sft-data-record-v1"
DEFAULT_OUTPUT = REPOSITORY / "outputs" / "data-v1"
HELD_OUT_TASKS = REPOSITORY / "data" / "evaluation" / "tasks.jsonl"

CALIBRATION_CANONICAL = Path(
    "outputs/sft-calibration-final-v2/calibration-raw.jsonl"
)
CALIBRATION_PLAN = Path(
    "outputs/sft-calibration-final-v2/calibration-plan.jsonl"
)
CALIBRATION_MERGE_MANIFEST = Path(
    "outputs/sft-calibration-final-v2/calibration-merge-manifest.json"
)
CALIBRATION_BATCHES = (
    "sft-calibration-smoke-c2-v1",
    "sft-calibration-smoke-c2-v2",
    "sft-calibration-smoke-c2-v3",
    "sft-calibration-smoke-c4-v1",
    "sft-calibration-c4-remaining194-v1",
    "sft-calibration-replacements-c4-v1",
)
FORMAL_BATCHES = (
    "sft-formal-batch-001-v1",
    "sft-formal-hard100-run-v1",
    "sft-formal-easy63-medium256-run-v1",
    "sft-formal-easy-topup-run-v1",
    "sft-formal-medium-topup-run-v1",
    "sft-formal-hard-topup-run-v1",
)
FORMAL_PLAN_DIRS = (
    "sft-formal-batch-v1",
    "sft-formal-hard100-v1",
    "sft-formal-easy63-medium256-v1",
    "sft-formal-easy-topup-v1",
    "sft-formal-medium-topup-v1",
    "sft-formal-hard-topup-v1",
)
FINAL_DIRECTORY = Path("outputs/sft-final-v1")
FINAL_FILES = (
    "sft.jsonl",
    "train.jsonl",
    "validation.jsonl",
    "accepted.jsonl",
    "rejected.jsonl",
    "alternatives.jsonl",
    "metadata.json",
    "reject-stats.json",
)
FINAL_DIFFICULTY_PLAN = Path(
    "outputs/sft-formal-plan-v1/formal-all-v3.jsonl"
)
DIFFICULTY_PROVENANCE = Path("outputs/sft-difficulty-final-v3")
FILTER_MERGE_MANIFEST = Path(
    "outputs/sft-filter-input-v1/merge-manifest.json"
)
TASK_PLAN_PROVENANCE = Path("outputs/sft-task-plan-v1")


def _is_provenance_file(path: Path) -> bool:
    """Select planning artifacts, including files from historical collections."""
    return path.is_file() and (
        path.name.startswith("wlx-")
        or (
            path.suffix in {".json", ".jsonl"}
            and path.name.startswith(
                ("difficulty-", "calibration-", "formal-", "public-train-", "retrieval-")
            )
        )
    )

EXPECTED = {
    "calibration_canonical_rows": 942,
    "calibration_canonical_tasks": 200,
    "calibration_extra_rows": 49,
    "calibration_history_rows": 991,
    "calibration_history_tasks": 202,
    "formal_rows": 2389,
    "formal_tasks": 1257,
    "raw_rows": 3380,
    "raw_tasks": 1459,
    "evaluated_rows": 3331,
    "sft_rows": 428,
    "train_rows": 385,
    "validation_rows": 43,
    "rejected_rows": 2903,
    "alternatives_rows": 7,
    "selected_from_calibration": 106,
    "selected_from_formal": 322,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="新建归档；不会覆盖已有目录")
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    verify = subparsers.add_parser("verify", help="只读核对已生成归档")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument(
        "--verify-sources",
        action="store_true",
        help="同时核对原始来源仍与 manifest 哈希一致",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _absolute(args.output_dir)
    if args.command == "build":
        summary = build_archive(output_dir)
    else:
        summary = verify_archive(output_dir, verify_sources=args.verify_sources)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_archive(output_dir: Path) -> dict[str, Any]:
    """在临时目录完整生成和验证后，再原子放到目标位置。"""

    if output_dir.exists():
        raise FileExistsError(f"目标已存在，不会覆盖：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _require_under_repository(output_dir)
    required_sources = _all_required_sources()
    source_hashes_before = {path: file_sha256(path) for path in required_sources}

    temporary = Path(
        tempfile.mkdtemp(prefix="data-v1-tmp-", dir=output_dir.parent)
    )
    try:
        _build_archive_tree(temporary, output_dir)
        source_hashes_after = {path: file_sha256(path) for path in required_sources}
        if source_hashes_before != source_hashes_after:
            raise RuntimeError("整理过程中原始来源发生变化，已停止")
        verify_archive(
            temporary,
            verify_sources=True,
            logical_output_dir=output_dir,
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_archive(output_dir, verify_sources=True)


def _build_archive_tree(actual_root: Path, logical_root: Path) -> None:
    raw_root = actual_root / "raw"
    calibration_root = raw_root / "calibration-200"
    formal_root = raw_root / "formal"
    sft_root = actual_root / "sft"
    for directory in (calibration_root, formal_root, sft_root):
        directory.mkdir(parents=True, exist_ok=False)

    final = _load_final_decisions()
    difficulty = _load_unique_rows(_source(FINAL_DIFFICULTY_PLAN), "task_id")
    held_out_ids = set(_load_unique_rows(HELD_OUT_TASKS, "task_id"))

    calibration_records, calibration_sources = _build_calibration(
        calibration_root,
        logical_root,
        final,
        difficulty,
    )
    formal_records, formal_sources = _build_formal(
        formal_root,
        logical_root,
        final,
        difficulty,
    )
    all_records = calibration_records + formal_records
    all_ids = [record["trajectory_id"] for record in all_records]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("整理后的 raw 存在重复 trajectory_id")
    raw_task_ids = {int(record["task_id"]) for record in all_records}
    overlap = raw_task_ids & held_out_ids
    if overlap:
        raise ValueError(f"raw 与 evaluation 重叠：{sorted(overlap)}")

    raw_records_path = raw_root / "records.jsonl"
    _write_jsonl(raw_records_path, all_records)
    evaluated_records = [
        record for record in all_records if record["sft"]["evaluated"]
    ]
    _build_sft(
        sft_root,
        logical_root,
        evaluated_records,
        final,
        held_out_ids,
    )

    raw_manifest = {
        "archive_version": ARCHIVE_VERSION,
        "group": "raw",
        "description": "完整原始采样历史；成功、失败和技术异常均保留",
        "summary": {
            "trajectories": len(all_records),
            "unique_tasks": len(raw_task_ids),
            "result_class": dict(
                sorted(Counter(row["result_class"] for row in all_records).items())
            ),
            "sft_selection_status": dict(
                sorted(
                    Counter(
                        row["sft"]["selection_status"] for row in all_records
                    ).items()
                )
            ),
            "evaluation_overlap": 0,
        },
        "source_files": calibration_sources + formal_sources,
        "files": _tree_reports(
            actual_root=raw_root,
            logical_root=logical_root / "raw",
            exclude_names={"manifest.json"},
        ),
    }
    _write_json(raw_root / "manifest.json", raw_manifest)

    root_manifest = {
        "archive_version": ARCHIVE_VERSION,
        "layout": {
            "raw": "全部原始轨迹及逐条结果记录",
            "sft": "最终 SFT 数据及逐条接收/拒绝记录",
        },
        "summary": {
            "raw_trajectories": len(all_records),
            "sft_trajectories": len(final["accepted"]),
            "train": len(final["train_ids"]),
            "validation": len(final["validation_ids"]),
            "evaluation_overlap": 0,
        },
        "manifests": [
            {
                **_file_report(
                    raw_root / "manifest.json",
                    logical_root / "raw" / "manifest.json",
                ),
                "archive_relative_path": "raw/manifest.json",
            },
            {
                **_file_report(
                    sft_root / "manifest.json",
                    logical_root / "sft" / "manifest.json",
                ),
                "archive_relative_path": "sft/manifest.json",
            },
        ],
    }
    _write_json(actual_root / "manifest.json", root_manifest)


def _build_calibration(
    output: Path,
    logical_root: Path,
    final: Mapping[str, Any],
    difficulty: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical_source = _source(CALIBRATION_CANONICAL)
    canonical_output = output / "raw.jsonl"
    shutil.copy2(canonical_source, canonical_output)

    locators: dict[str, dict[str, Any]] = {}
    source_reports: list[dict[str, Any]] = []
    for batch in CALIBRATION_BATCHES:
        raw_path = _source(Path("outputs") / batch / "raw.jsonl")
        source_reports.append(_source_report(raw_path))
        for line_number, row, _ in _read_jsonl_bytes(raw_path):
            trajectory_id = _trajectory_id(row, raw_path, line_number)
            if trajectory_id in locators:
                raise ValueError(f"校准 source 重复 trajectory_id={trajectory_id}")
            locators[trajectory_id] = {
                "batch": batch,
                "source_file": _portable(raw_path),
                "source_row": line_number,
            }

    canonical_ids: set[str] = set()
    canonical_records: list[dict[str, Any]] = []
    logical_canonical = logical_root / "raw" / "calibration-200" / "raw.jsonl"
    for line_number, row, _ in _read_jsonl_bytes(canonical_source):
        trajectory_id = _trajectory_id(row, canonical_source, line_number)
        if trajectory_id in canonical_ids:
            raise ValueError(f"canonical 校准重复 trajectory_id={trajectory_id}")
        canonical_ids.add(trajectory_id)
        locator = locators.get(trajectory_id)
        if locator is None:
            raise ValueError(f"canonical 轨迹不在原始校准 source：{trajectory_id}")
        canonical_records.append(
            _record(
                row=row,
                source_group="calibration",
                locator=locator,
                raw_path=logical_canonical,
                raw_line=line_number,
                canonical=True,
                history_only=False,
                final=final,
                difficulty=difficulty,
            )
        )
    if not canonical_ids <= set(locators):
        raise ValueError("canonical 校准不是原始校准历史的子集")

    extras_output = output / "extra-history.jsonl"
    extra_records: list[dict[str, Any]] = []
    logical_extras = (
        logical_root
        / "raw"
        / "calibration-200"
        / "extra-history.jsonl"
    )
    with extras_output.open("wb") as handle:
        archive_line = 0
        for batch in CALIBRATION_BATCHES:
            raw_path = _source(Path("outputs") / batch / "raw.jsonl")
            for line_number, row, raw_line in _read_jsonl_bytes(raw_path):
                trajectory_id = _trajectory_id(row, raw_path, line_number)
                if trajectory_id in canonical_ids:
                    continue
                archive_line += 1
                handle.write(raw_line)
                extra_records.append(
                    _record(
                        row=row,
                        source_group="calibration",
                        locator=locators[trajectory_id],
                        raw_path=logical_extras,
                        raw_line=archive_line,
                        canonical=False,
                        history_only=True,
                        final=final,
                        difficulty=difficulty,
                    )
                )
        handle.flush()
        os.fsync(handle.fileno())

    records = canonical_records + extra_records
    _write_jsonl(output / "records.jsonl", records)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(_source(CALIBRATION_PLAN), provenance / "calibration-plan.jsonl")
    shutil.copy2(
        _source(CALIBRATION_MERGE_MANIFEST),
        provenance / "calibration-merge-manifest.json",
    )
    for source_path in sorted(_source(TASK_PLAN_PROVENANCE).iterdir()):
        if _is_provenance_file(source_path):
            shutil.copy2(source_path, provenance / source_path.name)
    _copy_batch_configs(CALIBRATION_BATCHES, provenance)

    manifest = {
        "archive_version": ARCHIVE_VERSION,
        "group": "calibration-200",
        "description": (
            "raw.jsonl 是 200 个校准任务的 canonical 942 行；"
            "extra-history.jsonl 单列未进入 canonical 的 49 条历史轨迹"
        ),
        "derivation": "extra = six source raw trajectory_id set minus canonical set",
        "summary": {
            "canonical_rows": len(canonical_records),
            "canonical_tasks": len({row["task_id"] for row in canonical_records}),
            "extra_history_rows": len(extra_records),
            "extra_history_tasks": len({row["task_id"] for row in extra_records}),
            "all_history_rows": len(records),
            "all_history_tasks": len({row["task_id"] for row in records}),
        },
        "source_files": source_reports + [_source_report(canonical_source)],
        "files": _tree_reports(
            actual_root=output,
            logical_root=logical_root / "raw" / "calibration-200",
            exclude_names={"manifest.json"},
        ),
    }
    _write_json(output / "manifest.json", manifest)
    return records, source_reports + [_source_report(canonical_source)]


def _build_formal(
    output: Path,
    logical_root: Path,
    final: Mapping[str, Any],
    difficulty: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged_output = output / "raw.jsonl"
    logical_raw = logical_root / "raw" / "formal" / "raw.jsonl"
    records: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    archive_line = 0
    with merged_output.open("wb") as handle:
        for batch in FORMAL_BATCHES:
            raw_path = _source(Path("outputs") / batch / "raw.jsonl")
            source_reports.append(_source_report(raw_path))
            for source_line, row, raw_line in _read_jsonl_bytes(raw_path):
                trajectory_id = _trajectory_id(row, raw_path, source_line)
                if trajectory_id in seen_ids:
                    raise ValueError(f"正式采样重复 trajectory_id={trajectory_id}")
                seen_ids.add(trajectory_id)
                archive_line += 1
                handle.write(raw_line)
                records.append(
                    _record(
                        row=row,
                        source_group="formal",
                        locator={
                            "batch": batch,
                            "source_file": _portable(raw_path),
                            "source_row": source_line,
                        },
                        raw_path=logical_raw,
                        raw_line=archive_line,
                        canonical=True,
                        history_only=False,
                        final=final,
                        difficulty=difficulty,
                    )
                )
        handle.flush()
        os.fsync(handle.fileno())
    _write_jsonl(output / "records.jsonl", records)

    provenance = output / "provenance"
    provenance.mkdir()
    _copy_batch_configs(FORMAL_BATCHES, provenance)
    for directory_name in FORMAL_PLAN_DIRS:
        directory = _source(Path("outputs") / directory_name)
        for source_path in sorted(directory.iterdir()):
            if _is_provenance_file(source_path):
                shutil.copy2(source_path, provenance / source_path.name)
    for batch in FORMAL_BATCHES:
        log_path = _source(Path("outputs") / batch / "collector.log")
        if log_path.exists():
            shutil.copy2(
                log_path,
                provenance / f"{batch}-collector.log",
            )
    shutil.copy2(
        _source(FINAL_DIFFICULTY_PLAN),
        provenance / "formal-all-v3.jsonl",
    )
    for source_path in sorted(_source(DIFFICULTY_PROVENANCE).iterdir()):
        if _is_provenance_file(source_path):
            shutil.copy2(source_path, provenance / source_path.name)

    manifest = {
        "archive_version": ARCHIVE_VERSION,
        "group": "formal",
        "description": "后续六批正式采样按固定批次顺序原样拼接",
        "merge_order": list(FORMAL_BATCHES),
        "summary": {
            "rows": len(records),
            "unique_tasks": len({row["task_id"] for row in records}),
        },
        "source_files": source_reports,
        "files": _tree_reports(
            actual_root=output,
            logical_root=logical_root / "raw" / "formal",
            exclude_names={"manifest.json"},
        ),
    }
    _write_json(output / "manifest.json", manifest)
    return records, source_reports


def _build_sft(
    output: Path,
    logical_root: Path,
    evaluated_records: Sequence[Mapping[str, Any]],
    final: Mapping[str, Any],
    held_out_ids: set[int],
) -> None:
    source_directory = _source(FINAL_DIRECTORY)
    source_reports: list[dict[str, Any]] = []
    for name in FINAL_FILES:
        source_path = source_directory / name
        target = output / name
        shutil.copy2(source_path, target)
        source_reports.append(_source_report(source_path))
    _write_jsonl(output / "records.jsonl", evaluated_records)

    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(
        _source(FILTER_MERGE_MANIFEST),
        provenance / "filter-input-merge-manifest.json",
    )
    for source_path in sorted(_source(DIFFICULTY_PROVENANCE).iterdir()):
        if _is_provenance_file(source_path):
            shutil.copy2(source_path, provenance / source_path.name)

    accepted = final["accepted"]
    accepted_tasks = {int(row["task_id"]) for row in accepted.values()}
    overlap = accepted_tasks & held_out_ids
    if overlap:
        raise ValueError(f"SFT 与 evaluation 重叠：{sorted(overlap)}")
    selected_source = Counter(
        record["source"]["group"]
        for record in evaluated_records
        if record["sft"]["selected"]
    )
    manifest = {
        "archive_version": ARCHIVE_VERSION,
        "group": "sft",
        "description": "最终正式 SFT 及全部 3331 条候选轨迹的接收/拒绝记录",
        "historical_metadata_note": (
            "metadata.json 原样保留生成时路径；整理后的定位以本 manifest 为准"
        ),
        "summary": {
            "selection_records": len(evaluated_records),
            "accepted": len(accepted),
            "rejected": len(final["rejected"]),
            "train": len(final["train_ids"]),
            "validation": len(final["validation_ids"]),
            "alternatives_for_audit": len(final["alternative_ids"]),
            "selected_source": dict(sorted(selected_source.items())),
            "evaluation_overlap": 0,
        },
        "source_files": source_reports,
        "files": _tree_reports(
            actual_root=output,
            logical_root=logical_root / "sft",
            exclude_names={"manifest.json"},
        ),
    }
    _write_json(output / "manifest.json", manifest)


def _record(
    *,
    row: Mapping[str, Any],
    source_group: str,
    locator: Mapping[str, Any],
    raw_path: Path,
    raw_line: int,
    canonical: bool,
    history_only: bool,
    final: Mapping[str, Any],
    difficulty: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    trajectory_id = str(row["trajectory_id"])
    task_id = int(row["task_id"])
    accepted = trajectory_id in final["accepted"]
    rejected = trajectory_id in final["rejected"]
    if accepted and rejected:
        raise ValueError(f"轨迹同时出现在 accepted/rejected：{trajectory_id}")
    if accepted:
        selection_status = "selected"
        split = "train" if trajectory_id in final["train_ids"] else "validation"
        reject_reasons: list[str] = []
    elif rejected:
        selection_status = "rejected"
        split = None
        reject_reasons = list(final["rejected"][trajectory_id].get("reject_reasons") or [])
    else:
        selection_status = "not_evaluated_history"
        split = None
        reject_reasons = []
    if history_only and selection_status != "not_evaluated_history":
        raise ValueError(f"历史 extra 意外进入最终筛选：{trajectory_id}")
    if not history_only and selection_status == "not_evaluated_history":
        raise ValueError(f"canonical/formal 轨迹缺少最终去留记录：{trajectory_id}")

    attempt_valid = row.get("attempt_valid") is True
    task_success = row.get("task_success") is True
    if attempt_valid and task_success:
        result_class = "success"
    elif attempt_valid:
        result_class = "valid_failure"
    else:
        result_class = "technical_failure"
    request = _mapping(_mapping(row.get("stage_metadata")).get("request"))
    final_difficulty = difficulty.get(task_id)
    if final_difficulty is None:
        raise ValueError(f"task_id={task_id} 不在最终难度表")
    token_metrics = _mapping(row.get("token_metrics"))
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "result_class": result_class,
        "source": {
            "group": source_group,
            "batch": locator["batch"],
            "source_file": locator["source_file"],
            "source_row": locator["source_row"],
            "raw_path": _portable(raw_path),
            "raw_row": raw_line,
            "canonical": canonical,
            "history_only": history_only,
        },
        "attempt": {
            "attempt_index": row.get("attempt_index"),
            "technical_retry_index": row.get("technical_retry_index"),
            "status": row.get("status"),
            "attempt_valid": row.get("attempt_valid"),
            "sampling_invalid": row.get("sampling_invalid"),
            "infrastructure_invalid": row.get("infrastructure_invalid"),
            "created_at": row.get("created_at"),
            "finished_at": row.get("finished_at"),
        },
        "result": {
            "done": row.get("done"),
            "task_success": row.get("task_success"),
            "reward_valid": row.get("reward_valid"),
            "final_reward": row.get("final_reward"),
            "outcome_type": row.get("outcome_type"),
            "termination_category": row.get("termination_category"),
            "termination_reason": row.get("termination_reason"),
            "sft_disposition": row.get("sft_disposition"),
        },
        "difficulty": {
            "source_label": request.get("difficulty_label"),
            "source_score": request.get("difficulty_score"),
            "source_version": request.get("difficulty_version"),
            "final_label": final_difficulty.get("difficulty_label"),
            "final_score": final_difficulty.get("difficulty_score"),
            "final_version": final_difficulty.get("difficulty_version"),
            "category": final_difficulty.get("category"),
            "official_split": final_difficulty.get("official_split"),
        },
        "diagnostics": {
            "decision_reasons": row.get("decision_reasons") or [],
            "error": row.get("error"),
            "release_error": row.get("release_error"),
            "blocked_tool_calls": row.get("blocked_tool_calls") or [],
            "tool_call_truncations": row.get("tool_call_truncations") or [],
        },
        "metrics": {
            "num_env_steps": token_metrics.get("num_env_steps"),
            "num_assistant_turns": token_metrics.get("num_assistant_turns"),
            "raw_trajectory_tokens": token_metrics.get("raw_trajectory_tokens"),
            "max_request_tokens": token_metrics.get("max_request_tokens"),
        },
        "sft": {
            "evaluated": accepted or rejected,
            "selection_status": selection_status,
            "selected": accepted,
            "split": split,
            "reject_reasons": reject_reasons,
        },
    }


def _load_final_decisions() -> dict[str, Any]:
    directory = _source(FINAL_DIRECTORY)
    accepted = _load_unique_rows(directory / "accepted.jsonl", "trajectory_id")
    rejected = _load_unique_rows(directory / "rejected.jsonl", "trajectory_id")
    sft = _load_unique_rows(directory / "sft.jsonl", "trajectory_id")
    train = _load_unique_rows(directory / "train.jsonl", "trajectory_id")
    validation = _load_unique_rows(directory / "validation.jsonl", "trajectory_id")
    alternatives = _load_unique_rows(directory / "alternatives.jsonl", "trajectory_id")
    accepted_ids = set(accepted)
    rejected_ids = set(rejected)
    train_ids = set(train)
    validation_ids = set(validation)
    if accepted_ids & rejected_ids:
        raise ValueError("final accepted/rejected 存在交集")
    if set(sft) != accepted_ids:
        raise ValueError("final sft 与 accepted trajectory_id 不一致")
    if train_ids & validation_ids or train_ids | validation_ids != accepted_ids:
        raise ValueError("final train/validation 不能精确覆盖 accepted")
    for trajectory_id, row in accepted.items():
        required = {
            "outcome_type": "gold_purchase",
            "reward_valid": True,
            "attempt_valid": True,
            "task_success": True,
            "sft_disposition": "accepted_gold",
            "done": True,
            "status": "done",
            "termination_category": "environment_done",
        }
        bad = {key: (row.get(key), value) for key, value in required.items() if row.get(key) != value}
        if bad:
            raise ValueError(f"accepted 非严格 Gold：{trajectory_id} {bad}")
    return {
        "accepted": accepted,
        "rejected": rejected,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "alternative_ids": set(alternatives),
    }


def verify_archive(
    output_dir: Path,
    *,
    verify_sources: bool = False,
    logical_output_dir: Path | None = None,
) -> dict[str, Any]:
    """只读验证归档的计数、集合关系、严格 Gold 和 manifest 哈希。"""

    output_dir = _absolute(output_dir)
    logical_output_dir = _absolute(logical_output_dir or output_dir)
    raw_root = output_dir / "raw"
    calibration = raw_root / "calibration-200"
    formal = raw_root / "formal"
    sft = output_dir / "sft"
    required = (
        output_dir / "manifest.json",
        raw_root / "records.jsonl",
        calibration / "raw.jsonl",
        calibration / "extra-history.jsonl",
        calibration / "records.jsonl",
        formal / "raw.jsonl",
        formal / "records.jsonl",
        sft / "sft.jsonl",
        sft / "train.jsonl",
        sft / "validation.jsonl",
        sft / "accepted.jsonl",
        sft / "rejected.jsonl",
        sft / "alternatives.jsonl",
        sft / "records.jsonl",
    )
    for path in required:
        _require_file(path)

    canonical = _load_unique_rows(calibration / "raw.jsonl", "trajectory_id")
    extras = _load_unique_rows(calibration / "extra-history.jsonl", "trajectory_id")
    formal_rows = _load_unique_rows(formal / "raw.jsonl", "trajectory_id")
    raw_records = _load_unique_rows(raw_root / "records.jsonl", "trajectory_id")
    calibration_records = _load_unique_rows(calibration / "records.jsonl", "trajectory_id")
    formal_records = _load_unique_rows(formal / "records.jsonl", "trajectory_id")
    accepted = _load_unique_rows(sft / "accepted.jsonl", "trajectory_id")
    rejected = _load_unique_rows(sft / "rejected.jsonl", "trajectory_id")
    sft_rows = _load_unique_rows(sft / "sft.jsonl", "trajectory_id")
    train = _load_unique_rows(sft / "train.jsonl", "trajectory_id")
    validation = _load_unique_rows(sft / "validation.jsonl", "trajectory_id")
    alternatives = _load_unique_rows(sft / "alternatives.jsonl", "trajectory_id")
    selection_records = _load_unique_rows(sft / "records.jsonl", "trajectory_id")

    canonical_ids = set(canonical)
    extra_ids = set(extras)
    formal_ids = set(formal_rows)
    raw_ids = canonical_ids | extra_ids | formal_ids
    if canonical_ids & extra_ids or (canonical_ids | extra_ids) & formal_ids:
        raise ValueError("归档 raw 分组存在 trajectory_id 交集")
    if set(calibration_records) != canonical_ids | extra_ids:
        raise ValueError("校准 records 没有精确覆盖校准 raw")
    if set(formal_records) != formal_ids:
        raise ValueError("formal records 没有精确覆盖 formal raw")
    if set(raw_records) != raw_ids:
        raise ValueError("raw 总 records 没有精确覆盖全部 raw")

    accepted_ids = set(accepted)
    rejected_ids = set(rejected)
    evaluated_ids = canonical_ids | formal_ids
    if accepted_ids & rejected_ids or accepted_ids | rejected_ids != evaluated_ids:
        raise ValueError("accepted/rejected 没有精确覆盖 canonical+formal")
    if set(selection_records) != evaluated_ids:
        raise ValueError("SFT records 没有精确覆盖全部筛选候选")
    if set(sft_rows) != accepted_ids:
        raise ValueError("SFT 与 accepted trajectory_id 不一致")
    if set(train) & set(validation) or set(train) | set(validation) != accepted_ids:
        raise ValueError("train/validation 没有互斥且完整覆盖 SFT")

    expected_counts = {
        "calibration_canonical_rows": len(canonical),
        "calibration_canonical_tasks": len({int(row["task_id"]) for row in canonical.values()}),
        "calibration_extra_rows": len(extras),
        "calibration_history_rows": len(canonical_ids | extra_ids),
        "calibration_history_tasks": len(
            {int(row["task_id"]) for row in canonical.values()}
            | {int(row["task_id"]) for row in extras.values()}
        ),
        "formal_rows": len(formal_rows),
        "formal_tasks": len({int(row["task_id"]) for row in formal_rows.values()}),
        "raw_rows": len(raw_ids),
        "raw_tasks": len({int(row["task_id"]) for row in raw_records.values()}),
        "evaluated_rows": len(evaluated_ids),
        "sft_rows": len(sft_rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "rejected_rows": len(rejected),
        "alternatives_rows": len(alternatives),
        "selected_from_calibration": len(accepted_ids & canonical_ids),
        "selected_from_formal": len(accepted_ids & formal_ids),
    }
    mismatches = {
        key: {"actual": value, "expected": EXPECTED[key]}
        for key, value in expected_counts.items()
        if value != EXPECTED[key]
    }
    if mismatches:
        raise ValueError(f"归档计数不符合冻结基准：{mismatches}")

    statuses = Counter(
        row["sft"]["selection_status"] for row in raw_records.values()
    )
    if statuses != Counter(
        {
            "selected": EXPECTED["sft_rows"],
            "rejected": EXPECTED["rejected_rows"],
            "not_evaluated_history": EXPECTED["calibration_extra_rows"],
        }
    ):
        raise ValueError(f"raw records 去留状态异常：{dict(statuses)}")
    for trajectory_id, row in accepted.items():
        strict = (
            row.get("outcome_type") == "gold_purchase"
            and row.get("reward_valid") is True
            and row.get("attempt_valid") is True
            and row.get("task_success") is True
            and row.get("sft_disposition") == "accepted_gold"
            and row.get("done") is True
            and row.get("status") == "done"
            and row.get("termination_category") == "environment_done"
        )
        if not strict:
            raise ValueError(f"SFT accepted 非严格 Gold：{trajectory_id}")

    held_out_ids = set(_load_unique_rows(HELD_OUT_TASKS, "task_id"))
    raw_task_ids = {int(row["task_id"]) for row in raw_records.values()}
    sft_task_ids = {int(row["task_id"]) for row in sft_rows.values()}
    if raw_task_ids & held_out_ids or sft_task_ids & held_out_ids:
        raise ValueError("整理后数据与 evaluation task_id 重叠")

    for manifest_path in (
        calibration / "manifest.json",
        formal / "manifest.json",
        raw_root / "manifest.json",
        sft / "manifest.json",
    ):
        _verify_manifest_files(output_dir, manifest_path)
        if verify_sources:
            _verify_manifest_sources(manifest_path)
    root_manifest = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for report in root_manifest.get("manifests", []):
        path = output_dir / report["archive_relative_path"]
        _require_file(path)
        if file_sha256(path) != report["sha256"]:
            raise ValueError(f"子 manifest 哈希不匹配：{path}")

    return {
        "archive_version": ARCHIVE_VERSION,
        "output_dir": _portable(logical_output_dir),
        "raw": {
            "trajectories": len(raw_ids),
            "tasks": len(raw_task_ids),
            "calibration_canonical": len(canonical_ids),
            "calibration_extra_history": len(extra_ids),
            "formal": len(formal_ids),
        },
        "sft": {
            "selection_records": len(selection_records),
            "accepted": len(accepted_ids),
            "train": len(train),
            "validation": len(validation),
            "selected_from_calibration": len(accepted_ids & canonical_ids),
            "selected_from_formal": len(accepted_ids & formal_ids),
        },
        "evaluation_overlap": 0,
        "strict_gold_verified": len(accepted_ids),
        "source_hashes_verified": verify_sources,
    }


def _copy_batch_configs(batch_names: Sequence[str], target: Path) -> None:
    for batch in batch_names:
        source_path = _source(Path("outputs") / batch / "collection-config.json")
        shutil.copy2(
            source_path,
            target / f"{batch}-collection-config.json",
        )


def _all_required_sources() -> list[Path]:
    paths = [
        HELD_OUT_TASKS,
        _source(CALIBRATION_CANONICAL),
        _source(CALIBRATION_PLAN),
        _source(CALIBRATION_MERGE_MANIFEST),
        _source(FINAL_DIFFICULTY_PLAN),
        _source(FILTER_MERGE_MANIFEST),
    ]
    for batch in CALIBRATION_BATCHES + FORMAL_BATCHES:
        paths.extend(
            [
                _source(Path("outputs") / batch / "raw.jsonl"),
                _source(Path("outputs") / batch / "collection-config.json"),
            ]
        )
        optional_log = _source(Path("outputs") / batch / "collector.log")
        if optional_log.exists():
            paths.append(optional_log)
    for directory in (
        _source(TASK_PLAN_PROVENANCE),
        _source(DIFFICULTY_PROVENANCE),
        *(_source(Path("outputs") / name) for name in FORMAL_PLAN_DIRS),
    ):
        paths.extend(path for path in directory.iterdir() if path.is_file())
    paths.extend(_source(FINAL_DIRECTORY) / name for name in FINAL_FILES)
    unique = sorted(set(paths))
    for path in unique:
        _require_file(path)
    return unique


def _source(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def _require_under_repository(path: Path) -> None:
    try:
        path.resolve().relative_to(REPOSITORY.resolve())
    except ValueError as error:
        raise ValueError(f"输出必须位于仓库内：{path}") from error


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _trajectory_id(row: Mapping[str, Any], path: Path, line_number: int) -> str:
    value = str(row.get("trajectory_id") or "").strip()
    if not value:
        raise ValueError(f"{path}:{line_number}: 缺少 trajectory_id")
    return value


def _read_jsonl_bytes(
    path: Path,
) -> Iterator[tuple[int, dict[str, Any], bytes]]:
    _require_file(path)
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: JSONL 空行")
            if not raw_line.endswith(b"\n"):
                raise ValueError(f"{path}:{line_number}: JSONL 末行缺少换行")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: 非法 JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL 行必须是对象")
            yield line_number, row, raw_line


def _load_unique_rows(path: Path, key: str) -> dict[Any, dict[str, Any]]:
    rows: dict[Any, dict[str, Any]] = {}
    for line_number, row, _ in _read_jsonl_bytes(path):
        if key not in row:
            raise ValueError(f"{path}:{line_number}: 缺少 {key}")
        value = row[key]
        if value in rows:
            raise ValueError(f"{path}:{line_number}: 重复 {key}={value}")
        rows[value] = row
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    absolute = _absolute(path)
    try:
        return absolute.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def _source_report(path: Path) -> dict[str, Any]:
    report = {
        "path": _portable(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if path.suffix == ".jsonl":
        report["rows"] = _line_count(path)
    return report


def _file_report(actual: Path, logical: Path) -> dict[str, Any]:
    report = {
        "path": _portable(logical),
        "archive_relative_path": logical.name,
        "bytes": actual.stat().st_size,
        "sha256": file_sha256(actual),
    }
    if actual.suffix == ".jsonl":
        report["rows"] = _line_count(actual)
    return report


def _tree_reports(
    *,
    actual_root: Path,
    logical_root: Path,
    exclude_names: set[str],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for actual in sorted(path for path in actual_root.rglob("*") if path.is_file()):
        if actual.name in exclude_names:
            continue
        relative = actual.relative_to(actual_root)
        report = _file_report(actual, logical_root / relative)
        report["archive_relative_path"] = relative.as_posix()
        reports.append(report)
    return reports


def _verify_manifest_files(archive_root: Path, manifest_path: Path) -> None:
    _require_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    for report in manifest.get("files", []):
        path = base / report["archive_relative_path"]
        _require_file(path)
        if file_sha256(path) != report["sha256"]:
            raise ValueError(f"归档文件哈希不匹配：{path}")
        if path.stat().st_size != report["bytes"]:
            raise ValueError(f"归档文件大小不匹配：{path}")
        if "rows" in report and _line_count(path) != report["rows"]:
            raise ValueError(f"归档文件行数不匹配：{path}")


def _verify_manifest_sources(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for report in manifest.get("source_files", []):
        path = _source(Path(report["path"]))
        _require_file(path)
        if file_sha256(path) != report["sha256"]:
            raise ValueError(f"原始来源哈希变化：{path}")
        if "rows" in report and _line_count(path) != report["rows"]:
            raise ValueError(f"原始来源行数变化：{path}")


if __name__ == "__main__":
    raise SystemExit(main())
