"""Eval v2 离线流水线：已有轨迹 -> 评测、复核队列和 Run 报告。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from shopping_grpo.harness.eval_contracts import (
    EVAL_METHOD_VERSION,
    EVAL_OFFLINE_RUN_VERSION,
    EVAL_SCHEMA_VERSION,
    validate_evaluation_record,
)
from shopping_grpo.harness.eval_report import (
    render_summary_markdown,
    summarize_evaluations,
)
from shopping_grpo.harness.eval_rubric import validate_rubric_bundle
from shopping_grpo.harness.eval_requirements import merge_judged_requirements
from shopping_grpo.harness.sft_storage import read_jsonl
from shopping_grpo.harness.trajectory_evaluator import evaluate_trajectory


class OfflineEvaluationOutputExistsError(FileExistsError):
    """目标目录已有 Eval 产物，拒绝静默覆盖。"""


class OfflineEvaluationPaths:
    """一个离线 Eval Run 的固定产物。"""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.manifest = self.output_dir / "eval-manifest.json"
        self.evaluations = self.output_dir / "trajectory-evaluations.jsonl"
        self.review_queue = self.output_dir / "review-queue.jsonl"
        self.summary_json = self.output_dir / "run-summary.json"
        self.summary_markdown = self.output_dir / "run-summary.md"

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.manifest,
            self.evaluations,
            self.review_queue,
            self.summary_json,
            self.summary_markdown,
        )


def run_offline_evaluation(
    *,
    trajectories_path: str | Path,
    output_dir: str | Path,
    run_manifest_path: str | Path | None = None,
    rubrics_path: str | Path | None = None,
    judge_results_path: str | Path | None = None,
    judge_case_prefix: str | None = None,
) -> dict[str, Any]:
    """对完整轨迹一次性生成 Eval v2 产物。"""

    source = Path(trajectories_path)
    if not source.is_file():
        raise FileNotFoundError(f"轨迹文件不存在：{source}")
    paths = OfflineEvaluationPaths(output_dir)
    existing = [str(path) for path in paths.outputs if path.exists()]
    if existing:
        raise OfflineEvaluationOutputExistsError(
            f"拒绝覆盖已有 Eval 产物：{existing[0]}"
        )

    manifest_source = (
        Path(run_manifest_path)
        if run_manifest_path is not None
        else source.parent / "run-manifest.json"
    )
    run_manifest = _read_optional_json_object(manifest_source)
    rubrics, rubric_source = _load_rubrics(rubrics_path)
    judgments, judge_source = _load_judgments(
        judge_results_path,
        case_prefix=judge_case_prefix,
    )
    if judgments is not None and rubrics is None:
        raise ValueError("Judge results require frozen Rubrics")
    run_name = source.parent.name
    model = _manifest_model(run_manifest)
    records: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line_number, raw_line in _jsonl_lines(source):
        try:
            trajectory = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"轨迹第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(trajectory, Mapping):
            raise ValueError(f"轨迹第 {line_number} 行不是 JSON 对象")
        task_id = trajectory.get("task_id")
        rubric = rubrics.get(task_id) if rubrics is not None else None
        if rubrics is not None and rubric is None:
            raise ValueError(f"冻结 Rubric 缺少 task_id={task_id}")
        evaluation = evaluate_trajectory(trajectory, rubric=rubric).to_dict()
        if judgments is not None:
            judgment = judgments.get(int(task_id))
            if judgment is None:
                raise ValueError(f"Judge results missing task_id={task_id}")
            evaluation = _apply_judgment(evaluation, judgment)
        task_id = evaluation["task_id"]
        if task_id in seen:
            raise ValueError(f"轨迹中 task_id={task_id} 重复")
        seen.add(task_id)
        evaluation["run_id"] = run_name
        evaluation["method_version"] = EVAL_METHOD_VERSION
        evaluation["model"] = deepcopy(model)
        evaluation["versions"] = _evaluation_versions(trajectory, run_manifest)
        evaluation["versions"]["rubric"] = (
            rubric.get("rubric_set_version") if rubric is not None else None
        )
        evaluation["versions"]["judge"] = (
            judgment.get("prompt_version") if judgments is not None else None
        )
        evaluation["source_refs"] = {
            "trajectory_file": str(source),
            "trajectory_line": line_number,
            "trajectory_line_sha256": hashlib.sha256(
                raw_line.encode("utf-8")
            ).hexdigest(),
        }
        evaluation = validate_evaluation_record(evaluation)
        records.append(evaluation)
        if _requires_review(evaluation):
            review_rows.append(_review_row(evaluation))
    if not records:
        raise ValueError(f"轨迹文件为空：{source}")

    summary = summarize_evaluations(records, run_name=run_name)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_jsonl(paths.evaluations, records)
    _atomic_write_jsonl(paths.review_queue, review_rows)
    _atomic_write_json(paths.summary_json, summary)
    _atomic_write_text(paths.summary_markdown, render_summary_markdown(summary))

    output_hashes = {
        path.name: _file_sha256(path)
        for path in (
            paths.evaluations,
            paths.review_queue,
            paths.summary_json,
            paths.summary_markdown,
        )
    }
    manifest = {
        "schema_version": EVAL_OFFLINE_RUN_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "method_version": EVAL_METHOD_VERSION,
        "evaluator_schema_version": EVAL_SCHEMA_VERSION,
        "source": {
            "trajectories": str(source),
            "trajectories_sha256": _file_sha256(source),
            "run_manifest": str(manifest_source) if run_manifest is not None else None,
            "run_manifest_sha256": (
                _file_sha256(manifest_source) if run_manifest is not None else None
            ),
        },
        "counts": {
            "evaluations": len(records),
            "review_queue": len(review_rows),
        },
        "outputs": output_hashes,
        "limitations": {
            "requirements": (
                "all Rubric requirements await calibrated Eval v2 judge"
                if rubrics is not None
                else "provisional; frozen user-query rubrics not configured"
            ),
            "process_quality": "not run; calibrated LLM judge not configured",
        },
        "rubrics": (
            {
                "path": str(rubric_source),
                "sha256": _file_sha256(rubric_source),
                "count": len(rubrics),
                "rubric_set_version": next(iter(rubrics.values()))[
                    "rubric_set_version"
                ],
            }
            if rubrics is not None and rubric_source is not None
            else None
        ),
        "judge": (
            {
                "path": str(judge_source),
                "sha256": _file_sha256(judge_source),
                "count": len(judgments),
                "case_prefix": judge_case_prefix,
            }
            if judgments is not None and judge_source is not None
            else None
        ),
    }
    if judgments is not None:
        manifest["limitations"]["requirements"] = None
        manifest["limitations"]["process_quality"] = None
    _atomic_write_json(paths.manifest, manifest)
    return {
        "offline_evaluation_version": EVAL_OFFLINE_RUN_VERSION,
        "evaluated_trajectories": len(records),
        "review_queue_size": len(review_rows),
        "output_dir": str(paths.output_dir),
        "manifest": str(paths.manifest),
        "evaluations": str(paths.evaluations),
        "review_queue": str(paths.review_queue),
        "summary_json": str(paths.summary_json),
        "summary_markdown": str(paths.summary_markdown),
    }


def _load_rubrics(
    path: str | Path | None,
) -> tuple[dict[int, dict[str, Any]] | None, Path | None]:
    if path is None:
        return None, None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Rubric 文件不存在：{source}")
    result: dict[int, dict[str, Any]] = {}
    versions: set[str] = set()
    for line_number, raw_line in _jsonl_lines(source):
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Rubric 第 {line_number} 行不是合法 JSON") from exc
        frozen = validate_rubric_bundle(raw)
        task_id = int(frozen["task_id"])
        if task_id in result:
            raise ValueError(f"Rubric 中 task_id={task_id} 重复")
        result[task_id] = frozen
        versions.add(str(frozen["rubric_set_version"]))
    if not result:
        raise ValueError("Rubric 文件为空")
    if len(versions) != 1:
        raise ValueError("Rubric 文件包含多个 rubric_set_version")
    return result, source


def _load_judgments(
    path: str | Path | None,
    *,
    case_prefix: str | None,
) -> tuple[dict[int, dict[str, Any]] | None, Path | None]:
    if path is None:
        if case_prefix is not None:
            raise ValueError("--judge-case-prefix requires --judge-results")
        return None, None
    if not case_prefix or ":" in case_prefix:
        raise ValueError("Judge results require a case prefix without ':'")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Judge 结果不存在：{source}")
    batch_manifest_path = source.parent / "judge-manifest.json"
    if not batch_manifest_path.is_file():
        raise ValueError("Judge results are missing their batch manifest")
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    contract = batch_manifest.get("contract")
    if not isinstance(contract, Mapping) or contract.get("mode") != "production":
        raise ValueError("offline Eval only accepts production Judge results")
    calibration_path = contract.get("calibration_manifest")
    calibration_sha = contract.get("calibration_manifest_sha256")
    if not calibration_path or not calibration_sha:
        raise ValueError("production Judge manifest is missing frozen calibration")
    calibration_source = Path(str(calibration_path))
    if not calibration_source.is_file() or _file_sha256(calibration_source) != calibration_sha:
        raise ValueError("Judge calibration manifest is missing or has changed")
    result = {}
    prefix = f"{case_prefix}:"
    for row in read_jsonl(source):
        case_id = str(row.get("case_id") or "")
        if not case_id.startswith(prefix):
            continue
        task_id = row.get("task_id")
        judgment = row.get("judgment")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or not isinstance(
            judgment, Mapping
        ):
            raise ValueError("Judge result contains invalid task_id/judgment")
        if task_id in result:
            raise ValueError(f"Judge results duplicate task_id={task_id}")
        result[task_id] = deepcopy(dict(row))
    if not result:
        raise ValueError(f"Judge results contain no cases for prefix={case_prefix}")
    return result, source


def _apply_judgment(
    evaluation: Mapping[str, Any],
    judge_result: Mapping[str, Any],
) -> dict[str, Any]:
    row = deepcopy(dict(evaluation))
    judgment = judge_result["judgment"]
    row["requirements"] = merge_judged_requirements(
        row["requirements"],
        judgment.get("requirements") or [],
    )
    row["process_quality"] = {
        "status": "complete",
        "version": judge_result.get("prompt_version"),
        "dimensions": deepcopy(judgment["dimensions"]),
    }
    deterministic = row.get("failure_attribution") or {}
    if deterministic.get("status") in {"partial", "review_required"}:
        attribution = judgment["attribution"]
        row["failure_attribution"] = {
            **deepcopy(dict(deterministic)),
            "status": "complete",
            "responsibility": attribution["responsibility"],
            "primary_failure": (
                None
                if attribution["primary_failure"] == "none"
                else attribution["primary_failure"]
            ),
            "secondary_failures": deepcopy(attribution["secondary_failures"]),
            "first_error_event_id": attribution["first_error_event_id"],
            "evidence_event_ids": (
                [attribution["first_error_event_id"]]
                if attribution["first_error_event_id"]
                else []
            ),
            "explanation": attribution["reason"],
            "confidence": attribution["confidence"],
            "review_required": judgment["review_required"],
        }
    review = row.get("review") or {}
    # 确定性阶段的 attribution:* 只是“等待 Judge”的临时原因。Judge 已给出
    # 完整归因后必须清除，否则正式 Eval 会把几乎所有失败轨迹误报为待人工复核。
    reasons = [
        str(item)
        for item in review.get("reasons") or []
        if not str(item).startswith("attribution:")
    ]
    reasons.extend(str(item) for item in judgment.get("review_reasons") or [])
    reasons = list(dict.fromkeys(reasons))
    row["review"] = {
        **deepcopy(dict(review)),
        "required": bool(reasons or judgment["review_required"]),
        "reasons": reasons,
    }
    return validate_evaluation_record(row)


def _review_row(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "wlx-eval-review-queue-v1",
        "task_id": evaluation.get("task_id"),
        "trajectory_id": evaluation.get("trajectory_id"),
        "run_id": evaluation.get("run_id"),
        "eligibility": deepcopy(evaluation.get("eligibility")),
        "outcome": {
            key: deepcopy((evaluation.get("outcome") or {}).get(key))
            for key in (
                "normalized_outcome",
                "success_agreement",
                "type_agreement",
                "review_reasons",
            )
        },
        "failure_attribution": deepcopy(evaluation.get("failure_attribution")),
        "review": deepcopy(evaluation.get("review")),
        "source_refs": deepcopy(evaluation.get("source_refs")),
    }


def _requires_review(evaluation: Mapping[str, Any]) -> bool:
    return (
        (evaluation.get("review") or {}).get("required") is True
        or (evaluation.get("failure_attribution") or {}).get("review_required") is True
        or (evaluation.get("eligibility") or {}).get("status") == "review_required"
    )


def _evaluation_versions(
    trajectory: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract = manifest.get("run_contract") if isinstance(manifest, Mapping) else {}
    contract = contract if isinstance(contract, Mapping) else {}
    return {
        "trajectory_schema": trajectory.get("schema_version"),
        "harness_contract": trajectory.get("contract_version"),
        "environment": contract.get("environment_version"),
        "tool_schema": contract.get("tool_schema_version"),
        "purchase_verifier": contract.get("purchase_verifier_version"),
        "source_trajectory_evaluator": contract.get("trajectory_evaluator_version"),
        "offline_evaluator": EVAL_SCHEMA_VERSION,
        "rubric": None,
        "judge": None,
    }


def _manifest_model(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        return {}
    contract = manifest.get("run_contract")
    if not isinstance(contract, Mapping):
        return {}
    model = contract.get("model")
    return deepcopy(dict(model)) if isinstance(model, Mapping) else {}


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Run manifest 不是 JSON 对象：{path}")
    return deepcopy(dict(value))


def _jsonl_lines(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            raw = line.strip()
            if raw:
                yield line_number, raw


def _atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_write_text(path, text, ensure_newline=False)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        ensure_newline=False,
    )


def _atomic_write_text(path: Path, text: str, *, ensure_newline: bool = True) -> None:
    if path.exists():
        raise OfflineEvaluationOutputExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise OfflineEvaluationOutputExistsError(f"临时文件已存在：{temporary}")
    payload = text + ("\n" if ensure_newline and not text.endswith("\n") else "")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "OfflineEvaluationOutputExistsError",
    "OfflineEvaluationPaths",
    "run_offline_evaluation",
]
