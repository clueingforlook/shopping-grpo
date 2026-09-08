"""从 Base/SFT 轨迹分层构造待人工确认的 Judge Gold Set。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from shopping_grpo.harness.eval_contracts import PROCESS_DIMENSIONS
from shopping_grpo.harness.eval_events import standardize_trajectory_events
from shopping_grpo.harness.eval_rubric import validate_rubric_bundle
from shopping_grpo.harness.sft_storage import file_sha256, read_jsonl, write_jsonl
from shopping_grpo.harness.trajectory_evaluator import evaluate_trajectory


JUDGE_GOLD_VERSION = "wlx-eval-judge-gold-v1"


def build_judge_gold_draft(
    *,
    baseline_trajectories_path: str | Path,
    candidate_trajectories_path: str | Path,
    rubrics_path: str | Path,
    output_dir: str | Path,
    pairs_per_stratum: int = 3,
) -> dict[str, Any]:
    """按购买成功转移分层抽样，并写出机器草稿与人读 Markdown。"""

    if isinstance(pairs_per_stratum, bool) or pairs_per_stratum < 1:
        raise ValueError("pairs_per_stratum must be positive")
    output = Path(output_dir)
    paths = {
        "manifest": output / "judge-gold-manifest.json",
        "labels": output / "judge-gold-labels-draft.jsonl",
        "markdown": output / "judge-gold-review.md",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite Gold Set output: {existing[0]}")
    rubrics = {
        int(row["task_id"]): validate_rubric_bundle(row)
        for row in read_jsonl(rubrics_path)
    }
    baseline = _evaluate_run(baseline_trajectories_path, rubrics)
    candidate = _evaluate_run(candidate_trajectories_path, rubrics)
    if set(baseline) != set(candidate):
        raise ValueError("Base and SFT trajectories must cover identical task_ids")
    strata: dict[str, list[int]] = {
        "both_success": [],
        "base_only_success": [],
        "sft_only_success": [],
        "both_failure": [],
    }
    for task_id in sorted(baseline):
        before = baseline[task_id]["evaluation"].get("purchase_correct") is True
        after = candidate[task_id]["evaluation"].get("purchase_correct") is True
        key = (
            "both_success"
            if before and after
            else "base_only_success"
            if before
            else "sft_only_success"
            if after
            else "both_failure"
        )
        strata[key].append(task_id)
    ranked_by_stratum: dict[str, list[int]] = {}
    for name, task_ids in strata.items():
        ranked_by_stratum[name] = sorted(
            task_ids,
            key=lambda task_id: _selection_key(
                name,
                task_id,
                baseline[task_id]["evaluation"],
                candidate[task_id]["evaluation"],
            ),
        )
    targets = {
        name: min(pairs_per_stratum, len(task_ids))
        for name, task_ids in ranked_by_stratum.items()
    }
    remaining = pairs_per_stratum * len(strata) - sum(targets.values())
    while remaining:
        candidates = [
            name
            for name, task_ids in ranked_by_stratum.items()
            if targets[name] < len(task_ids)
        ]
        if not candidates:
            raise ValueError("not enough paired tasks to build requested Gold Set")
        selected_name = max(
            candidates,
            key=lambda name: (len(ranked_by_stratum[name]) - targets[name], name),
        )
        targets[selected_name] += 1
        remaining -= 1
    selected_tasks = [
        (name, task_id)
        for name, task_ids in ranked_by_stratum.items()
        for task_id in task_ids[: targets[name]]
    ]

    labels = []
    selected_cases = []
    markdown_cases = []
    case_number = 1
    for stratum, task_id in selected_tasks:
        for run_prefix, source in (
            ("base", baseline[task_id]),
            ("sft", candidate[task_id]),
        ):
            blind_id = f"G{case_number:03d}"
            draft = _draft_label(
                blind_id=blind_id,
                run_prefix=run_prefix,
                stratum=stratum,
                trajectory=source["trajectory"],
                rubric=rubrics[task_id],
                evaluation=source["evaluation"],
            )
            labels.append(draft)
            selected_cases.append(
                {
                    "blind_id": blind_id,
                    "case_id": f"{run_prefix}:{task_id}",
                    "run_prefix": run_prefix,
                    "task_id": task_id,
                    "stratum": stratum,
                    "case_sha256": _json_sha256(
                        {
                            "trajectory": source["trajectory"],
                            "rubric": rubrics[task_id],
                        }
                    ),
                }
            )
            markdown_cases.append(
                _render_case(
                    draft,
                    trajectory=source["trajectory"],
                    rubric=rubrics[task_id],
                    evaluation=source["evaluation"],
                )
            )
            case_number += 1

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["labels"], labels)
    manifest = {
        "schema_version": JUDGE_GOLD_VERSION,
        "status": "awaiting_human_confirmation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pairs_per_stratum": pairs_per_stratum,
        "strata_available": {name: len(values) for name, values in strata.items()},
        "selected_tasks": len(selected_tasks),
        "selected_trajectories": len(selected_cases),
        "selected_cases": selected_cases,
        "sources": {
            "baseline": _source_report(baseline_trajectories_path),
            "candidate": _source_report(candidate_trajectories_path),
            "rubrics": _source_report(rubrics_path),
        },
        "draft_labels": str(paths["labels"]),
        "review_markdown": str(paths["markdown"]),
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        _render_header(manifest) + "\n\n" + "\n\n".join(markdown_cases) + "\n",
        encoding="utf-8",
    )
    return {
        "status": manifest["status"],
        "selected_tasks": len(selected_tasks),
        "selected_trajectories": len(selected_cases),
        **{name: str(path) for name, path in paths.items()},
    }


def _evaluate_run(
    path: str | Path,
    rubrics: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    result = {}
    for trajectory in read_jsonl(path):
        task_id = int(trajectory["task_id"])
        if task_id in result or task_id not in rubrics:
            raise ValueError(f"invalid/duplicate/missing-rubric task_id={task_id}")
        result[task_id] = {
            "trajectory": trajectory,
            "evaluation": evaluate_trajectory(
                trajectory, rubric=rubrics[task_id]
            ).to_dict(),
        }
    return result


def _selection_key(
    stratum: str,
    task_id: int,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[Any, ...]:
    """优先选择失败类型和行为差异明显的样本，再用稳定哈希破同分。"""

    before_behavior = before["deterministic_behavior"]
    after_behavior = after["deterministic_behavior"]
    diversity = (
        before["failure_attribution"].get("primary_failure")
        != after["failure_attribution"].get("primary_failure")
    )
    action_gap = abs(
        int(before_behavior.get("attempted_tool_calls", 0))
        - int(after_behavior.get("attempted_tool_calls", 0))
    )
    digest = hashlib.sha256(f"{stratum}:{task_id}".encode()).hexdigest()
    return (-int(diversity), -action_gap, digest)


def _draft_label(
    *,
    blind_id: str,
    run_prefix: str,
    stratum: str,
    trajectory: Mapping[str, Any],
    rubric: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    behavior = evaluation["deterministic_behavior"]
    events = standardize_trajectory_events(trajectory, include_content=True)
    refs = behavior.get("event_refs") or {}
    search = 0 if behavior.get("search_count", 0) == 0 else (
        2 if behavior.get("unique_search_query_count", 0) >= 2 else 1
    )
    candidate = 0 if behavior.get("product_open_count", 0) == 0 else (
        2 if behavior.get("unique_product_open_count", 0) >= 2 else 1
    )
    evidence = 0 if behavior.get("product_open_count", 0) == 0 else (
        2
        if behavior.get("detail_view_count", 0) > 0
        or behavior.get("option_selection_count", 0) > 0
        else 1
    )
    if evaluation.get("purchase_correct") is True:
        decision = 2 if evidence >= 1 else 1
    elif behavior.get("purchase_call_count", 0) > 0:
        decision = 0
    else:
        decision = 1
    termination = 0 if (
        behavior.get("environment_repeat_loop")
        or behavior.get("invalid_action_count", 0) > 0
        or "max_" in str(behavior.get("termination_reason") or "")
    ) else (1 if behavior.get("exact_consecutive_repeat_count", 0) else 2)
    score_map = {
        "search_strategy": (search, refs.get("first_search")),
        "candidate_utilization": (candidate, refs.get("first_product_open")),
        "evidence_verification": (evidence, refs.get("first_product_open")),
        "decision_quality": (decision, refs.get("first_purchase") or refs.get("termination")),
        "termination_efficiency": (termination, refs.get("termination")),
    }
    unresolved = [
        {
            "requirement_id": item["requirement_id"],
            "status": "unknown",
            "reason": "draft placeholder; human must inspect visible evidence",
            "evidence_event_ids": [],
            "confidence": 0.0,
        }
        for item in evaluation["requirements"]["items"]
        if item["status"] == "unknown"
    ]
    failure = evaluation["failure_attribution"]
    primary = failure.get("primary_failure")
    if evaluation.get("purchase_correct") is True:
        responsibility, primary = "undetermined", "none"
    else:
        responsibility = failure.get("responsibility") or "undetermined"
        if primary in {None, "undetermined_model_failure"}:
            primary = "insufficient_evidence"
    return {
        "schema_version": "wlx-eval-judge-gold-label-draft-v1",
        "confirmation_status": "needs_human_confirmation",
        "blind_id": blind_id,
        "case_id": f"{run_prefix}:{trajectory['task_id']}",
        "task_id": int(trajectory["task_id"]),
        "stratum": stratum,
        "requirements": unresolved,
        "dimensions": {
            name: {
                "score": score,
                "reason": "heuristic draft; human must confirm against trajectory",
                "evidence_event_ids": [event_id] if event_id else [],
                "confidence": 0.0,
            }
            for name, (score, event_id) in score_map.items()
        },
        "attribution": {
            "responsibility": responsibility,
            "primary_failure": primary,
            "secondary_failures": [],
            "first_error_event_id": failure.get("first_error_event_id"),
            "reason": "deterministic draft; human must identify the causal first error",
            "confidence": 0.0,
        },
        "reviewer": "",
        "review_notes": "",
    }


def _render_header(manifest: Mapping[str, Any]) -> str:
    return f"""# Judge Gold Set 人工确认

> 状态：**待人工确认，不是已冻结 Gold**<br>
> 样本：{manifest['selected_tasks']} 个任务、{manifest['selected_trajectories']} 条轨迹<br>
> 盲评：正文不显示 Base/SFT 身份和隐藏目标。

## 如何确认

每条轨迹先阅读用户需求、Rubric 和操作轨迹，再确认：

1. 未解决需求是 `satisfied / violated / unknown / conflicting`。
2. 五个过程维度是 `0 / 1 / 2`。
3. 责任来源和第一处关键错误。
4. 建议标签只是程序草稿，**不要直接默认批准**。

在每条的“人工结论”中勾选通过或写下修改。全部确认后，再把修改同步到 JSONL 并冻结。

## 评分速查

| 维度 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| 搜索策略 | 误解、无关或反复无效 | 方向基本正确，但约束或调整不足 | 抓住关键需求并能有效调整 |
| 候选利用 | 忽略明显候选 | 有查看，但筛选或比较不足 | 优先检查有希望候选并合理比较 |
| 证据检查 | 未核对关键要求 | 检查了部分，但有重要遗漏 | 核对类别、硬约束、选项和价格 |
| 购买决策 | 无据错购或放弃 | 部分合理，但证据不足 | 购买或停止都有充分证据 |
| 终止效率 | 过早终止、循环或耗尽预算 | 能结束但明显冗余 | 证据足够后及时结束 |"""


def _render_case(
    draft: Mapping[str, Any],
    *,
    trajectory: Mapping[str, Any],
    rubric: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    lines = [
        f"## {draft['blind_id']}",
        "",
        f"- 用户需求：{rubric['instruction']}",
        f"- 工具调用：{evaluation['deterministic_behavior']['attempted_tool_calls']} 次",
        "",
        "### Rubric",
        "",
        "| ID | 优先级 | 要求 | 程序状态 |",
        "|---|---|---|---|",
    ]
    evaluated_by_id = {
        item["requirement_id"]: item for item in evaluation["requirements"]["items"]
    }
    for item in rubric["items"]:
        current = evaluated_by_id[item["requirement_id"]]
        lines.append(
            f"| {item['requirement_id']} | {item['priority']} | "
            f"{_cell(item['requirement'])} | `{current['status']}` |"
        )
    lines.extend(
        [
            "",
            "### 模型可见轨迹",
            "",
            "| 事件 | 动作 | 参数 | 执行后可见信息摘要 |",
            "|---|---|---|---|",
        ]
    )
    events = standardize_trajectory_events(trajectory, include_content=True)
    steps = {
        int(step["step_index"]): step
        for step in trajectory.get("steps") or []
        if isinstance(step, Mapping) and isinstance(step.get("step_index"), int)
    }
    for event in events:
        if event.get("kind") != "assistant_tool_call":
            continue
        step = steps.get(event.get("step_index"))
        lines.append(
            f"| {event['event_id']} | `{event.get('tool_name')}` | "
            f"`{_cell(json.dumps(event.get('arguments'), ensure_ascii=False))}` | "
            f"{_cell(_observation_summary(step))} |"
        )
    lines.extend(["", "### 建议标签（必须人工核对）", ""])
    if draft["requirements"]:
        lines.extend(
            [
                "| 未解决需求 | 建议 | 人工确认 |",
                "|---|---|---|",
            ]
        )
        for item in draft["requirements"]:
            lines.append(
                f"| {item['requirement_id']} | `{item['status']}` | `待填` |"
            )
        lines.append("")
    lines.extend(
        [
            "| 过程维度 | 建议分数 | 人工确认 |",
            "|---|---:|---:|",
        ]
    )
    for name in PROCESS_DIMENSIONS:
        lines.append(f"| `{name}` | {draft['dimensions'][name]['score']} | `待填` |")
    attribution = draft["attribution"]
    lines.extend(
        [
            "",
            f"- 建议责任：`{attribution['responsibility']}`",
            f"- 建议第一错误：`{attribution['primary_failure']}`",
            "- 人工结论：[ ] 通过建议标签　[ ] 需要修改",
            "- 修改说明：",
            "",
            "---",
        ]
    )
    return "\n".join(lines)


def _observation_summary(step: Mapping[str, Any] | None) -> str:
    if not isinstance(step, Mapping):
        return "未执行或无可见观察"
    text = str(step.get("observation") or "")
    keep = []
    markers = (
        "page_type:",
        "query:",
        "asin:",
        "title:",
        "brand:",
        "category:",
        "price:",
        "key_attributes:",
        "selected_options:",
        "products:",
        "error",
    )
    for raw in text.splitlines():
        line = raw.strip()
        if line and (line.startswith(markers) or re.match(r"^\d+\.\s", line)):
            keep.append(line)
        if len(keep) >= 10:
            break
    summary = "; ".join(keep) or " ".join(text.split())[:400]
    return summary[:900]


def _cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def _source_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return {"path": str(source), "sha256": file_sha256(source)}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


__all__ = ["JUDGE_GOLD_VERSION", "build_judge_gold_draft"]
