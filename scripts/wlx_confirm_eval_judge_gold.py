#!/usr/bin/env python3
"""记录用户对 Codex 辅助审核摘要的确认，生成可冻结标签。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    args = parser.parse_args()
    if args.output_labels.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_labels}")
    rows = []
    for line in args.reviewed_labels.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("confirmation_status") != "codex_reviewed_awaiting_user_confirmation":
            raise ValueError(f"unexpected status for {row.get('case_id')}")
        row["schema_version"] = "wlx-eval-judge-gold-label-user-confirmed-v1"
        row["confirmation_status"] = "confirmed"
        row["reviewer"] = "User-confirmed Codex-assisted review"
        row["review_notes"] = (
            f"{row.get('review_notes', '')} 用户已确认辅助审核摘要；"
            "该标签不是独立双人标注。"
        ).strip()
        rows.append(row)
    if not rows:
        raise ValueError("reviewed labels are empty")
    args.output_labels.parent.mkdir(parents=True, exist_ok=True)
    args.output_labels.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"confirmed": len(rows), "output": str(args.output_labels)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
