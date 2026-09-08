"""Deterministic, outcome-only evaluation for the frozen Final-200."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


EVAL_V3_TASK_SCHEMA = "wlx-eval-v3-task-result-v1"
EVAL_V3_SUMMARY_SCHEMA = "wlx-eval-v3-summary-v1"
EVAL_V3_MANIFEST_SCHEMA = "wlx-eval-v3-manifest-v1"
EXPECTED_REWARD_VERSION = "shopsimulator-reward-v3"
EXPECTED_FINAL200_SHA256 = "2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208"
DIMENSION_NAMES = ("brand", "model", "core_functions", "key_options")
ATTRIBUTE_DIMENSIONS = ("brand", "model", "core_functions")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task_ids(path: str | Path, *, limit: int | None = None) -> list[int]:
    task_ids = []
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError(f"invalid task_id in {path}: {task_id!r}")
        task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task_id in {path}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        task_ids = task_ids[:limit]
    if not task_ids:
        raise ValueError(f"task file is empty: {path}")
    return task_ids


def load_difficulty(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    result = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        label = row.get("difficulty_label")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError(f"invalid difficulty task_id: {task_id!r}")
        if label not in {"easy", "medium", "hard"}:
            raise ValueError(f"invalid difficulty label for task {task_id}: {label!r}")
        if task_id in result:
            raise ValueError(f"duplicate difficulty task_id: {task_id}")
        result[task_id] = dict(row)
    return result


def evaluate_trajectory(
    trajectory: Mapping[str, Any],
    *,
    difficulty: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read only ShopSimulator terminal facts; never call an LLM or Rubric."""

    task_id = trajectory.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise ValueError(f"trajectory has invalid task_id: {task_id!r}")
    terminal = _mapping(trajectory.get("terminal_result"))
    detail = _mapping(terminal.get("reward_detail"))
    purchase = _mapping(terminal.get("purchase"))
    purchased = bool(_text(purchase.get("asin")))
    reward_type = _text(detail.get("reward_type"))
    reward_version = _text(detail.get("reward_version"))
    reward_valid = _first_bool(
        terminal.get("reward_valid"),
        detail.get("reward_valid"),
        trajectory.get("reward_valid"),
    )
    infrastructure_invalid = bool(trajectory.get("infrastructure_invalid"))
    normal_terminal = bool(
        trajectory.get("done") is True
        and terminal.get("done") is True
        and terminal.get("over") is True
    )

    integrity_issues = []
    if not terminal:
        integrity_issues.append("terminal_result_missing")
    if reward_version != EXPECTED_REWARD_VERSION:
        integrity_issues.append("unexpected_reward_version")
    if purchased and not detail:
        integrity_issues.append("purchase_reward_detail_missing")

    dimensions = _dimension_results(detail, purchased=purchased)
    attributes = _combine_dimensions(dimensions, ATTRIBUTE_DIMENSIONS, purchased)
    options = _combine_dimensions(dimensions, ("key_options",), purchased)
    category = _gate_result(detail, "category", purchased=purchased)
    price = _gate_result(detail, "budget", purchased=purchased)

    asin_match = purchased and detail.get("target_asin_match") is True
    environment_gold = reward_type == "gold_purchase"
    purchase_success = detail.get("purchase_success") is True
    gold = bool(
        purchased
        and normal_terminal
        and reward_valid is True
        and environment_gold
        and purchase_success
    )
    if environment_gold and not gold:
        integrity_issues.append("inconsistent_gold_terminal")
    if gold and not asin_match:
        integrity_issues.append("gold_without_asin_match")
    if gold and options["ratio"] != 1.0:
        integrity_issues.append("gold_without_full_option_match")

    dimension_unverifiable = any(
        value["verifiable_count"] < value["required_count"]
        for value in dimensions.values()
    )
    unverifiable = bool(
        infrastructure_invalid
        or reward_valid is not True
        or reward_type == "reward_unverifiable"
        or (
            purchased
            and (
                category["status"] == "unverifiable"
                or price["status"] == "unverifiable"
            )
        )
        or (purchased and dimension_unverifiable)
        or integrity_issues
    )
    all_constraints = bool(
        purchased
        and not unverifiable
        and category["passed"]
        and attributes["ratio"] == 1.0
        and options["ratio"] == 1.0
        and price["passed"]
    )

    difficulty_payload = dict(difficulty or {})
    difficulty_label = difficulty_payload.get("difficulty_label")
    if difficulty_label not in {None, "easy", "medium", "hard"}:
        raise ValueError(f"invalid difficulty for task {task_id}: {difficulty_label!r}")

    return {
        "schema_version": EVAL_V3_TASK_SCHEMA,
        "task_id": task_id,
        "trajectory_id": _text(trajectory.get("trajectory_id")),
        "difficulty": {
            "label": difficulty_label,
            "score": difficulty_payload.get("difficulty_score"),
            "version": difficulty_payload.get("difficulty_version"),
        },
        "valid_for_primary": not infrastructure_invalid,
        "infrastructure_invalid": infrastructure_invalid,
        "integrity_issues": integrity_issues,
        "reward": {
            "version": reward_version,
            "valid": reward_valid,
            "type": reward_type,
        },
        "purchased": purchased,
        "no_purchase": not purchased,
        "gold": gold,
        "asin_match": asin_match,
        "category": category,
        "attributes": attributes,
        "options": options,
        "price": price,
        "all_constraints": all_constraints,
        "unverifiable": unverifiable,
        "purchase": {
            "asin": _text(purchase.get("asin")),
            "category": _text(purchase.get("category")),
            "price": _finite_or_none(purchase.get("price")),
            "options": dict(purchase.get("options"))
            if isinstance(purchase.get("options"), Mapping)
            else {},
        },
        "dimensions": dimensions,
    }


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_task_ids: Iterable[int],
    model_role: str,
    model_name: str,
) -> dict[str, Any]:
    expected = [int(task_id) for task_id in expected_task_ids]
    if len(expected) != len(set(expected)):
        raise ValueError("expected task ids contain duplicates")
    by_task = {}
    for record in records:
        if record.get("schema_version") != EVAL_V3_TASK_SCHEMA:
            raise ValueError("unsupported Eval v3 task schema")
        task_id = int(record["task_id"])
        if task_id in by_task:
            raise ValueError(f"duplicate trajectory for task_id {task_id}")
        by_task[task_id] = record
    expected_set = set(expected)
    missing = sorted(expected_set.difference(by_task))
    unexpected = sorted(set(by_task).difference(expected_set))
    if missing or unexpected:
        raise ValueError(f"task mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    ordered = [by_task[task_id] for task_id in expected]
    grouped = {}
    for label in ("easy", "medium", "hard"):
        subset = [row for row in ordered if row.get("difficulty", {}).get("label") == label]
        if subset:
            grouped[label] = _summary_panel(subset)
    return {
        "schema_version": EVAL_V3_SUMMARY_SCHEMA,
        "model": {"role": model_role, "name": model_name},
        "task_count": len(expected),
        "overall": _summary_panel(ordered),
        "by_difficulty": grouped,
    }


def run_offline_evaluation(
    *,
    trajectories_path: str | Path,
    tasks_path: str | Path,
    output_dir: str | Path,
    model_role: str,
    model_name: str,
    difficulty_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if model_role not in {"base", "sft", "grpo"}:
        raise ValueError("model_role must be base, sft, or grpo")
    trajectories_path = Path(trajectories_path)
    tasks_path = Path(tasks_path)
    output_dir = Path(output_dir)
    difficulty_path = Path(difficulty_path) if difficulty_path is not None else None
    run_manifest_path = Path(run_manifest_path) if run_manifest_path is not None else None
    expected_task_ids = load_task_ids(tasks_path, limit=limit)
    if limit is None and len(expected_task_ids) != 200:
        raise ValueError("formal Eval v3 requires exactly 200 tasks; use --limit for smoke")
    if limit is None and file_sha256(tasks_path) != EXPECTED_FINAL200_SHA256:
        raise ValueError("formal Eval v3 must use the frozen Final-200 task file")
    difficulty = load_difficulty(difficulty_path)
    trajectories = read_jsonl(trajectories_path)
    by_task = {}
    for trajectory in trajectories:
        task_id = trajectory.get("task_id")
        if task_id in by_task:
            raise ValueError(f"duplicate trajectory task_id: {task_id}")
        by_task[task_id] = trajectory
    selected = []
    for task_id in expected_task_ids:
        if task_id not in by_task:
            raise ValueError(f"trajectory missing for task_id {task_id}")
        selected.append(
            evaluate_trajectory(by_task[task_id], difficulty=difficulty.get(task_id))
        )
    unexpected = sorted(set(by_task).difference(expected_task_ids))
    if unexpected:
        raise ValueError(f"trajectory file contains unexpected task_ids: {unexpected[:10]}")

    summary = summarize(
        selected,
        expected_task_ids=expected_task_ids,
        model_role=model_role,
        model_name=model_name,
    )
    sources = {
        "tasks": {"path": str(tasks_path), "sha256": file_sha256(tasks_path)},
        "trajectories": {
            "path": str(trajectories_path),
            "sha256": file_sha256(trajectories_path),
        },
        "difficulty": _source_record(difficulty_path),
        "run_manifest": _source_record(run_manifest_path),
    }
    manifest = {
        "schema_version": EVAL_V3_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {"role": model_role, "name": model_name},
        "protocol": {
            "llm_judge": False,
            "rubric": False,
            "gold_source": "ShopSimulator reward_detail.gold_purchase",
            "reward_version": EXPECTED_REWARD_VERSION,
            "fixed_denominator": len(expected_task_ids),
        },
        "sources": sources,
        "outputs": {
            "task_results": "eval-v3-task-results.jsonl",
            "summary_json": "eval-v3-summary.json",
            "summary_markdown": "eval-v3-summary.md",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "task_results": output_dir / "eval-v3-task-results.jsonl",
        "summary_json": output_dir / "eval-v3-summary.json",
        "summary_markdown": output_dir / "eval-v3-summary.md",
        "manifest": output_dir / "eval-v3-manifest.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        if len(existing) == len(paths) and _existing_matches(paths["manifest"], manifest):
            return json.loads(paths["summary_json"].read_text(encoding="utf-8"))
        raise FileExistsError(
            "Eval v3 output already exists or is incomplete; use --overwrite only after checking it"
        )
    _atomic_text(paths["task_results"], _jsonl_text(selected))
    _atomic_json(paths["summary_json"], summary)
    _atomic_text(paths["summary_markdown"], render_markdown(summary))
    _atomic_json(paths["manifest"], manifest)
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    model = summary["model"]
    overall = summary["overall"]
    lines = [
        "# Eval v3 结果",
        "",
        f"模型：`{model['role']}` / `{model['name']}`",
        "",
        "本报告只使用 ShopSimulator 结构化终局；不使用 Rubric 或 LLM Judge。",
        "",
        "## 总体",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
    ]
    for key, label in (
        ("gold", "Gold"),
        ("purchase", "完成购买"),
        ("no_purchase", "未购买"),
        ("asin_match", "ASIN 正确"),
        ("category_pass", "品类通过"),
        ("attribute_ratio_mean", "属性平均满足率"),
        ("option_ratio_mean", "规格平均满足率"),
        ("price_pass", "价格通过"),
        ("all_constraints", "全部结构化约束通过"),
        ("unverifiable", "无法验证"),
    ):
        value = overall[key]
        if key.endswith("_mean"):
            rendered = f"{value:.1%}"
        else:
            rendered = f"{value['count']}/{overall['tasks']} ({value['rate']:.1%})"
        lines.append(f"| {label} | {rendered} |")
    if summary.get("by_difficulty"):
        lines.extend(
            [
                "",
                "## 按难度",
                "",
                "| 难度 | 任务 | Gold | 属性满足率 | 规格满足率 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label in ("easy", "medium", "hard"):
            panel = summary["by_difficulty"].get(label)
            if panel:
                lines.append(
                    f"| {label} | {panel['tasks']} | {panel['gold']['rate']:.1%} | "
                    f"{panel['attribute_ratio_mean']:.1%} | {panel['option_ratio_mean']:.1%} |"
                )
    lines.extend(["", "## 终局类型", ""])
    for name, count in sorted(overall["reward_type_counts"].items()):
        lines.append(f"- `{name}`：{count}")
    lines.append("")
    return "\n".join(lines)


def _dimension_results(detail: Mapping[str, Any], *, purchased: bool) -> dict[str, Any]:
    preference = _mapping(_mapping(detail.get("evidence")).get("preference_scoring"))
    raw_dimensions = _mapping(preference.get("dimensions"))
    results = {}
    for name in DIMENSION_NAMES:
        raw = _mapping(raw_dimensions.get(name))
        required = _nonnegative_int(raw.get("required_count"))
        passed = _nonnegative_int(raw.get("passed_count"))
        verifiable = _nonnegative_int(raw.get("verifiable_count"))
        if passed > required or verifiable > required:
            raise ValueError(f"invalid {name} dimension counts")
        ratio = passed / required if required else (1.0 if purchased else 0.0)
        results[name] = {
            "active": bool(raw.get("active")),
            "required_count": required,
            "passed_count": passed,
            "verifiable_count": verifiable,
            "ratio": ratio,
        }
    return results


def _combine_dimensions(
    dimensions: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    purchased: bool,
) -> dict[str, Any]:
    required = sum(int(dimensions[name]["required_count"]) for name in names)
    passed = sum(int(dimensions[name]["passed_count"]) for name in names)
    verifiable = sum(int(dimensions[name]["verifiable_count"]) for name in names)
    ratio = passed / required if required else (1.0 if purchased else 0.0)
    return {
        "required_count": required,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "ratio": ratio,
        "all_passed": bool(purchased and ratio == 1.0),
    }


def _gate_result(detail: Mapping[str, Any], name: str, *, purchased: bool) -> dict[str, Any]:
    gate = _mapping(_mapping(detail.get("hard_gates")).get(name))
    status = _text(gate.get("status")) if purchased else "not_purchased"
    if purchased and status not in {"pass", "fail", "unverifiable"}:
        status = "unverifiable"
    return {
        "status": status,
        "passed": bool(purchased and status == "pass"),
        "verifiable": bool(purchased and status != "unverifiable"),
        "required": gate.get("required"),
        "actual": gate.get("actual"),
    }


def _summary_panel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)

    def ratio(key: str) -> dict[str, Any]:
        numerator = sum(bool(row.get(key)) for row in rows)
        return {"count": numerator, "rate": numerator / count if count else 0.0}

    return {
        "tasks": count,
        "valid_for_primary": ratio("valid_for_primary"),
        "infrastructure_invalid": ratio("infrastructure_invalid"),
        "gold": ratio("gold"),
        "purchase": ratio("purchased"),
        "no_purchase": ratio("no_purchase"),
        "asin_match": ratio("asin_match"),
        "category_pass": {
            "count": sum(row["category"]["passed"] for row in rows),
            "rate": sum(row["category"]["passed"] for row in rows) / count if count else 0.0,
        },
        "attribute_ratio_mean": _mean(row["attributes"]["ratio"] for row in rows),
        "option_ratio_mean": _mean(row["options"]["ratio"] for row in rows),
        "price_pass": {
            "count": sum(row["price"]["passed"] for row in rows),
            "rate": sum(row["price"]["passed"] for row in rows) / count if count else 0.0,
        },
        "all_constraints": ratio("all_constraints"),
        "unverifiable": ratio("unverifiable"),
        "reward_type_counts": dict(
            sorted(Counter(row["reward"]["type"] or "missing" for row in rows).items())
        ),
    }


def _source_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": file_sha256(path)}


def _existing_matches(path: Path, manifest: Mapping[str, Any]) -> bool:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        existing.get("schema_version") == manifest.get("schema_version")
        and existing.get("model") == manifest.get("model")
        and existing.get("protocol") == manifest.get("protocol")
        and existing.get("sources") == manifest.get("sources")
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _first_bool(*values: object) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError("dimension count cannot be bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dimension count: {value!r}") from exc
    if result < 0:
        raise ValueError("dimension count cannot be negative")
    return result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "EXPECTED_FINAL200_SHA256",
    "EVAL_V3_MANIFEST_SCHEMA",
    "EVAL_V3_SUMMARY_SCHEMA",
    "EVAL_V3_TASK_SCHEMA",
    "evaluate_trajectory",
    "file_sha256",
    "load_task_ids",
    "render_markdown",
    "run_offline_evaluation",
    "summarize",
]
