#!/usr/bin/env python3
"""对已确认 Gold 应用有审计记录的一致性修订，保留原文件不变。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
    updates = corrections["search_strategy_scores"]
    rows = []
    seen = set()
    for line in args.labels.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row["case_id"]
        if case_id in updates:
            score = updates[case_id]
            if score not in {0, 1, 2}:
                raise ValueError(f"invalid score for {case_id}")
            row["dimensions"]["search_strategy"] = {
                "score": score,
                "reason": corrections["reason"],
                "evidence_event_ids": [],
                "confidence": 0.95,
            }
            seen.add(case_id)
        row["schema_version"] = "wlx-eval-judge-gold-label-consistency-v1"
        row["confirmation_status"] = "confirmed"
        row["reviewer"] = "User-confirmed conclusions; Codex consistency correction"
        row["review_notes"] = (
            f"{row.get('review_notes', '')} {corrections['audit_note']}"
        ).strip()
        rows.append(row)
    if seen != set(updates):
        raise ValueError("corrections contain unknown case_ids")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "corrected": len(seen), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
