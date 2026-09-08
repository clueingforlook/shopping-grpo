#!/usr/bin/env python3
"""把简短的人工/辅助审核决定应用到 Judge Gold 草稿。

输出仍然是“待用户确认”，不会冒充已经冻结的人工 Gold。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIMENSIONS = (
    "search_strategy",
    "candidate_utilization",
    "evidence_verification",
    "decision_quality",
    "termination_efficiency",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-labels", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = _parser().parse_args()
    draft = _read_jsonl(args.draft_labels)
    bundle = json.loads(args.decisions.read_text(encoding="utf-8"))
    decisions = bundle["cases"]
    blind_ids = {row["blind_id"] for row in draft}
    if blind_ids != set(decisions):
        raise ValueError("review decisions must exactly cover draft blind_ids")

    reviewed = []
    for row in draft:
        decision = decisions[row["blind_id"]]
        expected_ids = [item["requirement_id"] for item in row["requirements"]]
        statuses = decision["requirements"]
        if statuses == "all_unknown":
            statuses = {requirement_id: "unknown" for requirement_id in expected_ids}
        if set(statuses) != set(expected_ids):
            raise ValueError(f"{row['blind_id']} requirement decisions do not match draft")
        requirements = []
        for requirement_id in expected_ids:
            status = statuses[requirement_id]
            requirements.append(
                {
                    "requirement_id": requirement_id,
                    "status": status,
                    "reason": _requirement_reason(status, decision["note"]),
                    "evidence_event_ids": [],
                    "confidence": 0.85 if status == "unknown" else 0.95,
                }
            )
        scores = decision["dimensions"]
        if len(scores) != len(DIMENSIONS) or any(score not in {0, 1, 2} for score in scores):
            raise ValueError(f"{row['blind_id']} has invalid dimension scores")
        dimensions = {
            name: {
                "score": score,
                "reason": _dimension_reason(name, score, decision["note"]),
                "evidence_event_ids": [],
                "confidence": 0.9,
            }
            for name, score in zip(DIMENSIONS, scores)
        }
        attr = decision["attribution"]
        reviewed.append(
            {
                **row,
                "schema_version": "wlx-eval-judge-gold-label-codex-reviewed-v1",
                "confirmation_status": "codex_reviewed_awaiting_user_confirmation",
                "requirements": requirements,
                "dimensions": dimensions,
                "attribution": {
                    "responsibility": attr["responsibility"],
                    "primary_failure": attr["primary_failure"],
                    "secondary_failures": attr.get("secondary_failures", []),
                    "first_error_event_id": attr.get("first_error_event_id"),
                    "reason": decision["note"],
                    "confidence": 0.9,
                },
                "reviewer": "Codex assisted review (not independent human annotation)",
                "review_notes": decision["note"],
            }
        )
    _write_jsonl(args.output_labels, reviewed)
    if args.output_summary.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_summary}")
    args.output_summary.write_text(_summary(bundle, reviewed), encoding="utf-8")
    print(json.dumps({"cases": len(reviewed), "labels": str(args.output_labels), "summary": str(args.output_summary)}, ensure_ascii=False, indent=2))


def _requirement_reason(status: str, note: str) -> str:
    if status == "unknown":
        return f"可见轨迹不足以确认该项；{note}"
    if status == "satisfied":
        return f"商品标题、属性、选项或价格中的可见证据支持该项；{note}"
    if status == "violated":
        return f"最终选择中的可见证据与该项冲突；{note}"
    return f"可见证据相互冲突；{note}"


def _dimension_reason(name: str, score: int, note: str) -> str:
    level = {0: "存在关键缺陷", 1: "基本合理但有明显不足", 2: "表现充分"}[score]
    return f"{name}：{level}。{note}"


def _summary(bundle: dict, rows: list[dict]) -> str:
    lines = [
        "# Judge Gold：Codex 辅助审核摘要",
        "",
        "> 这不是独立人工双标。以下 24 条已由 Codex 逐条复核，等待用户对这一页做最终确认。",
        "",
        "## 审核中发现并修正的问题",
        "",
    ]
    lines.extend(f"- {item}" for item in bundle["findings"])
    lines.extend(["", "## 12 组任务的结论", "", "| 任务对 | 结论 |", "|---|---|"])
    for item in bundle["pair_summary"]:
        lines.append(f"| {item['pair']} | {item['conclusion']} |")
    lines.extend(
        [
            "",
            "## 请用户只确认这一件事",
            "",
            "如果你认可上述判断，我再把这份辅助审核标记为“经用户确认”并冻结；若不认可，只需指出哪一组任务有问题，不必阅读 24 条长轨迹。",
            "",
            f"- 待确认标签：`{len(rows)}` 条",
            "- 当前状态：`codex_reviewed_awaiting_user_confirmation`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
