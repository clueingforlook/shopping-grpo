#!/usr/bin/env python3
"""Build the frozen 20/60/20 GRPO task split from existing tasks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shopping_grpo.harness.sft_difficulty import (  # noqa: E402
    LogisticDifficultyModel,
    TaskDifficultyFeatures,
)
from shopping_grpo.harness.sft_prompt import (  # noqa: E402
    SFT_SYSTEM_PROMPT,
    SFT_SYSTEM_PROMPT_SHA256,
    SFT_SYSTEM_PROMPT_VERSION,
)


DEFAULT_PUBLIC_TASKS = (
    ROOT
    / "outputs/data-v1/raw/calibration-200/provenance"
    / "public-train-tasks.jsonl"
)
DEFAULT_FEATURES = DEFAULT_PUBLIC_TASKS.with_name("difficulty-features.jsonl")
DEFAULT_MODEL = (
    ROOT / "outputs/data-v1/sft/provenance/difficulty-model.json"
)
DEFAULT_EVAL = ROOT / "data/evaluation/tasks.jsonl"
DEFAULT_OUTPUT = ROOT / "data/step-grpo"
TRAIN_QUOTAS = {"easy": 200, "medium": 600, "hard": 200}
VALIDATION_QUOTAS = {"easy": 10, "medium": 30, "hard": 10}
MIX_PATTERN = ("easy", "medium", "medium", "hard", "medium") * 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or isinstance(value.get("task_id"), bool):
            raise ValueError(f"{path}:{line_number} has an invalid task_id")
        value["task_id"] = int(value["task_id"])
        rows.append(value)
    return rows


def _sft_task_ids() -> set[int]:
    result: set[int] = set()
    for path in (ROOT / "data/sft/train.jsonl", ROOT / "data/sft/validation.jsonl"):
        for row in _read_jsonl(path):
            result.add(int(row["task_id"]))
    return result


def _select_and_interleave(
    pools: dict[str, list[dict[str, Any]]], quotas: dict[str, int]
) -> list[dict[str, Any]]:
    selected: dict[str, deque[dict[str, Any]]] = {}
    for label, count in quotas.items():
        if len(pools[label]) < count:
            raise ValueError(f"not enough {label} tasks: need {count}, got {len(pools[label])}")
        selected[label] = deque(pools[label][:count])
        del pools[label][:count]
    rows = []
    while any(selected.values()):
        progressed = False
        for label in MIX_PATTERN:
            if selected[label]:
                rows.append(selected[label].popleft())
                progressed = True
        if not progressed:
            break
    if len(rows) != sum(quotas.values()):
        raise AssertionError("difficulty interleave lost tasks")
    return rows


def _training_row(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    instruction = str(row.get("instruction") or "").strip()
    if not instruction:
        raise ValueError(f"task_id={row['task_id']} has an empty instruction")
    return {
        "data_source": "shopsimulator",
        "prompt": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Instruction: {instruction}"},
        ],
        "ability": "shopping",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "split": split,
            "index": index,
            "task_id": int(row["task_id"]),
            "difficulty_label": row["difficulty_label"],
            "difficulty_score": float(row["difficulty_score"]),
            "difficulty_version": row["difficulty_version"],
        },
    }


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.public_tasks, args.features, args.model, args.evaluation_tasks):
        if not path.is_file():
            raise FileNotFoundError(path)
    public_rows = _read_jsonl(args.public_tasks)
    features = {
        row["task_id"]: TaskDifficultyFeatures.from_dict(row)
        for row in _read_jsonl(args.features)
    }
    public_by_id = {row["task_id"]: row for row in public_rows}
    if len(public_by_id) != len(public_rows) or set(public_by_id) != set(features):
        raise ValueError("public tasks and difficulty features do not have identical task IDs")
    model = LogisticDifficultyModel.from_dict(
        json.loads(args.model.read_text(encoding="utf-8"))
    )
    excluded = {row["task_id"] for row in _read_jsonl(args.evaluation_tasks)}
    excluded.update(_sft_task_ids())
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_id, task in public_by_id.items():
        if task_id in excluded:
            continue
        label, predicted_success, difficulty_score = model.classify(features[task_id])
        pools[label].append(
            {
                **task,
                "difficulty_label": label,
                "difficulty_score": difficulty_score,
                "predicted_teacher_success": predicted_success,
                "difficulty_version": model.difficulty_version,
            }
        )
    rng = random.Random(args.seed)
    for label in ("easy", "medium", "hard"):
        pools[label].sort(key=lambda row: row["task_id"])
        rng.shuffle(pools[label])
    train_selected = _select_and_interleave(pools, TRAIN_QUOTAS)
    validation_selected = _select_and_interleave(pools, VALIDATION_QUOTAS)
    train = [_training_row(row, "train", i) for i, row in enumerate(train_selected)]
    validation = [
        _training_row(row, "validation", i)
        for i, row in enumerate(validation_selected)
    ]
    train_ids = {row["extra_info"]["task_id"] for row in train}
    validation_ids = {row["extra_info"]["task_id"] for row in validation}
    if train_ids & validation_ids or (train_ids | validation_ids) & excluded:
        raise AssertionError("RL task split overlap detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.parquet"
    validation_path = args.output_dir / "validation.parquet"
    metadata_path = args.output_dir / "metadata.json"
    if any(path.exists() for path in (train_path, validation_path, metadata_path)) and not args.force:
        raise FileExistsError(f"RL data already exists: {args.output_dir}; use --force to rebuild")
    _atomic_parquet(train_path, train)
    _atomic_parquet(validation_path, validation)
    metadata = {
        "schema_version": "wlx-grpo-data-v1",
        "seed": args.seed,
        "difficulty_version": model.difficulty_version,
        "system_prompt_version": SFT_SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": SFT_SYSTEM_PROMPT_SHA256,
        "excluded_sft_and_evaluation_tasks": len(excluded),
        "train": {
            "rows": len(train),
            "difficulty_counts": dict(Counter(row["extra_info"]["difficulty_label"] for row in train)),
            "sha256": _sha256(train_path),
        },
        "validation": {
            "rows": len(validation),
            "difficulty_counts": dict(
                Counter(row["extra_info"]["difficulty_label"] for row in validation)
            ),
            "sha256": _sha256(validation_path),
        },
        "sources": {
            "public_tasks_sha256": _sha256(args.public_tasks),
            "features_sha256": _sha256(args.features),
            "model_sha256": _sha256(args.model),
            "evaluation_tasks_sha256": _sha256(args.evaluation_tasks),
        },
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, metadata_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-tasks", type=Path, default=DEFAULT_PUBLIC_TASKS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluation-tasks", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        metadata = build(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
