"""Eval Judge 的生成、断点续跑、校准比较和冻结门禁。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from shopping_grpo.harness.eval_judge import (
    EVAL_JUDGE_PROMPT_VERSION,
    EVAL_JUDGE_SYSTEM_PROMPT_SHA256,
    build_full_judge_messages,
    build_full_judge_payload,
    validate_full_judgment,
)
from shopping_grpo.harness.eval_rubric import validate_rubric_bundle
from shopping_grpo.harness.eval_rubric_generator import DeepSeekRubricClient
from shopping_grpo.harness.sft_storage import (
    append_jsonl_row,
    file_sha256,
    read_jsonl,
    write_jsonl,
)


EVAL_JUDGE_BATCH_VERSION = "wlx-eval-v2-judge-batch-v1"
EVAL_JUDGE_CALIBRATION_VERSION = "wlx-eval-v2-judge-calibration-v1"


@dataclass(frozen=True)
class JudgeCase:
    case_id: str
    trajectory: Mapping[str, Any]
    rubric: Mapping[str, Any]

    @property
    def task_id(self) -> int:
        return int(self.trajectory["task_id"])

    @property
    def content_sha256(self) -> str:
        return _json_sha256(
            {
                "case_id": self.case_id,
                "trajectory": self.trajectory,
                "rubric": self.rubric,
            }
        )


class JudgeBatchPaths:
    def __init__(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        self.root = root
        self.manifest = root / "judge-manifest.json"
        self.attempts = root / "judge-attempts.jsonl"
        self.results = root / "judge-results.jsonl"
        self.review_queue = root / "judge-review-queue.jsonl"
        self.summary = root / "judge-summary.json"


def load_judge_cases(
    *,
    trajectories_path: str | Path,
    rubrics_path: str | Path,
    case_prefix: str,
    selected_task_ids: Sequence[int] | None = None,
) -> list[JudgeCase]:
    """按 task_id 对齐轨迹与冻结 Rubric，并构造稳定 case_id。"""

    trajectories = read_jsonl(trajectories_path)
    rubrics = {
        int(row["task_id"]): validate_rubric_bundle(row)
        for row in read_jsonl(rubrics_path)
    }
    selected = set(int(item) for item in selected_task_ids) if selected_task_ids else None
    result = []
    seen: set[int] = set()
    for trajectory in trajectories:
        task_id = trajectory.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError("trajectory contains invalid task_id")
        if selected is not None and task_id not in selected:
            continue
        if task_id in seen:
            raise ValueError(f"duplicate trajectory task_id={task_id}")
        seen.add(task_id)
        rubric = rubrics.get(task_id)
        if rubric is None:
            raise ValueError(f"frozen rubric missing task_id={task_id}")
        result.append(
            JudgeCase(
                case_id=f"{case_prefix}:{task_id}",
                trajectory=trajectory,
                rubric=rubric,
            )
        )
    if selected is not None and seen != selected:
        raise ValueError(f"selected tasks not covered: {sorted(selected.difference(seen))}")
    if not result:
        raise ValueError("judge cases are empty")
    return result


def judge_case(
    case: JudgeCase,
    *,
    client: DeepSeekRubricClient,
) -> dict[str, Any]:
    payload = build_full_judge_payload(
        trajectory=case.trajectory,
        rubric=case.rubric,
    )
    response = client.complete_json(build_full_judge_messages(payload))
    judgment = validate_full_judgment(
        response["result"],
        allowed_event_ids=payload["allowed_evidence_event_ids"],
        required_requirement_ids=payload["required_requirement_ids"],
    )
    metadata = deepcopy(response.get("metadata") or {})
    return {
        "schema_version": "wlx-eval-v2-judge-result-v1",
        "case_id": case.case_id,
        "task_id": case.task_id,
        "case_sha256": case.content_sha256,
        "prompt_version": EVAL_JUDGE_PROMPT_VERSION,
        "prompt_sha256": EVAL_JUDGE_SYSTEM_PROMPT_SHA256,
        "rubric_requirements": [
            {
                "requirement_id": item["requirement_id"],
                "type": item["type"],
                "priority": item["priority"],
            }
            for item in case.rubric["items"]
        ],
        "judgment": judgment,
        "metadata": metadata,
    }


def run_judge_batch(
    *,
    cases: Sequence[JudgeCase],
    output_dir: str | Path,
    client: DeepSeekRubricClient,
    mode: str,
    calibration_manifest_path: str | Path | None = None,
    concurrency: int = 1,
    progress_callback: Callable[[Mapping[str, Any]], object] | None = None,
) -> dict[str, Any]:
    """运行 calibration 或 production Judge；production 强制要求冻结校准。"""

    if mode not in {"calibration", "production"}:
        raise ValueError("judge mode must be calibration or production")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    calibration = _load_calibration_gate(
        calibration_manifest_path,
        mode=mode,
        model=client.model,
    )
    selected = _validate_cases(cases)
    paths = JudgeBatchPaths(output_dir)
    contract = {
        "batch_version": EVAL_JUDGE_BATCH_VERSION,
        "mode": mode,
        "model": client.model,
        "base_url": client.base_url,
        "max_tokens": client.max_tokens,
        "prompt_version": EVAL_JUDGE_PROMPT_VERSION,
        "prompt_sha256": EVAL_JUDGE_SYSTEM_PROMPT_SHA256,
        "case_ids": [case.case_id for case in selected],
        "cases_sha256": _json_sha256(
            [{"case_id": case.case_id, "sha256": case.content_sha256} for case in selected]
        ),
        "calibration_manifest": str(calibration_manifest_path) if calibration else None,
        "calibration_manifest_sha256": (
            file_sha256(calibration_manifest_path) if calibration is not None else None
        ),
    }
    _prepare_manifest(paths, contract)
    successful = _latest_successes(paths.attempts, selected)
    pending = [case for case in selected if case.case_id not in successful]
    _notify(progress_callback, len(successful), len(selected), None)

    def execute(case: JudgeCase) -> dict[str, Any]:
        started = datetime.now(timezone.utc).isoformat()
        try:
            result = judge_case(case, client=client)
            return {
                "schema_version": "wlx-eval-v2-judge-attempt-v1",
                "case_id": case.case_id,
                "case_sha256": case.content_sha256,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "ready",
                "result": result,
                "error": None,
            }
        except Exception as exc:
            return {
                "schema_version": "wlx-eval-v2-judge-attempt-v1",
                "case_id": case.case_id,
                "case_sha256": case.content_sha256,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "technical_error",
                "result": None,
                "error": {"error_type": type(exc).__name__, "message": str(exc)},
            }

    if pending:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(execute, case): case for case in pending}
            for future in as_completed(futures):
                attempt = future.result()
                append_jsonl_row(paths.attempts, attempt)
                if attempt["status"] == "ready":
                    successful[attempt["case_id"]] = attempt
                _notify(
                    progress_callback,
                    len(successful),
                    len(selected),
                    attempt["case_id"],
                )
    remaining = [case.case_id for case in selected if case.case_id not in successful]
    if remaining:
        return {
            "complete": False,
            "completed": len(successful),
            "remaining": len(remaining),
            "remaining_case_ids": remaining,
            "output_dir": str(paths.root),
        }
    if paths.results.exists() and paths.summary.exists() and paths.review_queue.exists():
        return _batch_result(paths, len(selected), reused=True)
    if any(path.exists() for path in (paths.results, paths.review_queue, paths.summary)):
        raise FileExistsError("partial derived judge outputs already exist")
    ordered = [successful[case.case_id]["result"] for case in selected]
    reviews = [row for row in ordered if row["judgment"]["review_required"]]
    write_jsonl(paths.results, ordered)
    write_jsonl(paths.review_queue, reviews)
    _write_json(
        paths.summary,
        {
            "schema_version": "wlx-eval-v2-judge-summary-v1",
            "cases": len(ordered),
            "review_required": len(reviews),
            "mode": mode,
        },
    )
    return _batch_result(paths, len(selected), reused=False)


def calibrate_judge(
    *,
    gold_labels_path: str | Path,
    judge_results_path: str | Path,
    output_path: str | Path,
    model: str,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """比较人工 Gold 与 Judge 输出；达到门槛时生成 production 冻结清单。"""

    limits = {
        "dimension_exact": 0.75,
        "requirement_exact": 0.80,
        "responsibility_exact": 0.80,
        "primary_failure_exact": 0.70,
    }
    if thresholds:
        limits.update({key: float(value) for key, value in thresholds.items()})
    gold = {str(row["case_id"]): row for row in read_jsonl(gold_labels_path)}
    results = {str(row["case_id"]): row for row in read_jsonl(judge_results_path)}
    if not gold or set(gold) != set(results):
        raise ValueError("gold labels and judge results must cover identical case_ids")
    for result in results.values():
        metadata = result.get("metadata") or {}
        if (
            result.get("prompt_version") != EVAL_JUDGE_PROMPT_VERSION
            or result.get("prompt_sha256") != EVAL_JUDGE_SYSTEM_PROMPT_SHA256
            or metadata.get("requested_model") != model
        ):
            raise ValueError("judge results do not match the current prompt/model contract")
    dimension_hits = dimension_total = 0
    requirement_hits = requirement_total = 0
    requirement_available = 0
    requirement_type_counts: dict[str, int] = {}
    responsibility_hits = primary_hits = 0
    per_dimension: dict[str, dict[str, int]] = {}
    for case_id, label in gold.items():
        predicted = results[case_id]["judgment"]
        predicted_requirements = {
            item["requirement_id"]: item["status"]
            for item in predicted.get("requirements") or []
        }
        contract_rows = results[case_id].get("rubric_requirements") or []
        contract_by_id = {
            str(item.get("requirement_id")): item
            for item in contract_rows
            if isinstance(item, Mapping)
        }
        if set(predicted_requirements) != set(contract_by_id):
            raise ValueError("judge result does not expose its complete Rubric contract")
        requirement_available += len(contract_by_id)
        seen_gold_ids: set[str] = set()
        for expected in label.get("requirements") or []:
            requirement_id = str(expected["requirement_id"])
            if requirement_id in seen_gold_ids or requirement_id not in contract_by_id:
                raise ValueError("gold requirement IDs are duplicate or outside the Rubric")
            seen_gold_ids.add(requirement_id)
            requirement_total += 1
            requirement_hits += int(
                predicted_requirements.get(requirement_id)
                == expected["status"]
            )
            requirement_type = str(contract_by_id[requirement_id].get("type") or "")
            requirement_type_counts[requirement_type] = (
                requirement_type_counts.get(requirement_type, 0) + 1
            )
        for name, expected in label["dimensions"].items():
            hit = int(predicted["dimensions"][name]["score"] == expected["score"])
            dimension_hits += hit
            dimension_total += 1
            bucket = per_dimension.setdefault(name, {"correct": 0, "total": 0})
            bucket["correct"] += hit
            bucket["total"] += 1
        responsibility_hits += int(
            predicted["attribution"]["responsibility"]
            == label["attribution"]["responsibility"]
        )
        primary_hits += int(
            predicted["attribution"]["primary_failure"]
            == label["attribution"]["primary_failure"]
        )
    metrics = {
        "dimension_exact": dimension_hits / dimension_total,
        "requirement_exact": (
            requirement_hits / requirement_total if requirement_total else 1.0
        ),
        "gold_requirement_labels": requirement_total,
        "available_gold_case_requirements": requirement_available,
        "gold_requirement_coverage": (
            requirement_total / requirement_available if requirement_available else 0.0
        ),
        "gold_requirement_types": dict(sorted(requirement_type_counts.items())),
        "responsibility_exact": responsibility_hits / len(gold),
        "primary_failure_exact": primary_hits / len(gold),
        "per_dimension": {
            name: {**counts, "rate": counts["correct"] / counts["total"]}
            for name, counts in sorted(per_dimension.items())
        },
    }
    required_types = {
        str(item.get("type") or "")
        for result in results.values()
        for item in result.get("rubric_requirements") or []
        if isinstance(item, Mapping)
    }
    uncovered_types = sorted(required_types.difference(requirement_type_counts))
    metrics["uncovered_requirement_types"] = uncovered_types
    passed = (
        requirement_total > 0
        and not uncovered_types
        and all(metrics[name] >= limit for name, limit in limits.items())
    )
    manifest = {
        "schema_version": EVAL_JUDGE_CALIBRATION_VERSION,
        "status": "frozen" if passed else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_version": EVAL_JUDGE_PROMPT_VERSION,
        "prompt_sha256": EVAL_JUDGE_SYSTEM_PROMPT_SHA256,
        "cases": len(gold),
        "thresholds": limits,
        "metrics": metrics,
        "gold_labels": str(gold_labels_path),
        "gold_labels_sha256": file_sha256(gold_labels_path),
        "judge_results": str(judge_results_path),
        "judge_results_sha256": file_sha256(judge_results_path),
    }
    _write_json(Path(output_path), manifest)
    return manifest


def _load_calibration_gate(
    path: str | Path | None,
    *,
    mode: str,
    model: str,
) -> dict[str, Any] | None:
    if mode == "calibration":
        if path is not None:
            raise ValueError("calibration mode must not use a calibration manifest")
        return None
    if path is None:
        raise ValueError("production judge requires --calibration-manifest")
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != EVAL_JUDGE_CALIBRATION_VERSION
        or value.get("status") != "frozen"
        or value.get("model") != model
        or value.get("prompt_sha256") != EVAL_JUDGE_SYSTEM_PROMPT_SHA256
    ):
        raise ValueError("judge calibration manifest is not compatible or frozen")
    return value


def _prepare_manifest(paths: JudgeBatchPaths, contract: Mapping[str, Any]) -> None:
    fingerprint = _json_sha256(contract)
    if paths.manifest.exists():
        current = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if current.get("contract_sha256") != fingerprint or current.get("contract") != dict(contract):
            raise ValueError("existing judge manifest does not match this run")
        return
    if paths.attempts.exists() and paths.attempts.stat().st_size:
        raise ValueError("judge attempts exist without manifest")
    _write_json(
        paths.manifest,
        {
            "schema_version": "wlx-eval-v2-judge-manifest-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "contract_sha256": fingerprint,
            "contract": deepcopy(dict(contract)),
        },
    )


def _latest_successes(
    path: Path,
    cases: Sequence[JudgeCase],
) -> dict[str, dict[str, Any]]:
    expected = {case.case_id: case.content_sha256 for case in cases}
    result = {}
    for row in read_jsonl(path):
        case_id = str(row.get("case_id") or "")
        if case_id not in expected or row.get("case_sha256") != expected[case_id]:
            raise ValueError("judge attempt is outside the current batch contract")
        if row.get("status") == "ready":
            result[case_id] = row
    return result


def _validate_cases(cases: Sequence[JudgeCase]) -> list[JudgeCase]:
    if not cases:
        raise ValueError("judge cases must be non-empty")
    result = []
    seen = set()
    for case in cases:
        if not isinstance(case, JudgeCase) or not case.case_id or case.case_id in seen:
            raise ValueError("judge cases contain invalid or duplicate case_id")
        seen.add(case.case_id)
        result.append(case)
    return result


def _notify(callback, completed: int, total: int, case_id: str | None) -> None:
    if callback is not None:
        callback({"completed": completed, "total": total, "case_id": case_id})


def _batch_result(paths: JudgeBatchPaths, cases: int, *, reused: bool) -> dict[str, Any]:
    return {
        "complete": True,
        "cases": cases,
        "reused_existing_outputs": reused,
        "output_dir": str(paths.root),
        "results": str(paths.results),
        "review_queue": str(paths.review_queue),
        "summary": str(paths.summary),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


__all__ = [
    "JudgeCase",
    "EVAL_JUDGE_BATCH_VERSION",
    "EVAL_JUDGE_CALIBRATION_VERSION",
    "calibrate_judge",
    "judge_case",
    "load_judge_cases",
    "run_judge_batch",
]
