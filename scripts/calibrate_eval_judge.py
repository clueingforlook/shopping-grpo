#!/usr/bin/env python3
"""将人工确认的 Gold 标签与 Judge 结果比较并生成校准门禁。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.eval_judge_pipeline import calibrate_judge  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-labels", type=Path, required=True)
    parser.add_argument("--judge-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--dimension-exact", type=float, default=0.75)
    parser.add_argument("--requirement-exact", type=float, default=0.80)
    parser.add_argument("--responsibility-exact", type=float, default=0.80)
    parser.add_argument("--primary-failure-exact", type=float, default=0.70)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = calibrate_judge(
        gold_labels_path=args.gold_labels,
        judge_results_path=args.judge_results,
        output_path=args.output,
        model=args.model,
        thresholds={
            "dimension_exact": args.dimension_exact,
            "requirement_exact": args.requirement_exact,
            "responsibility_exact": args.responsibility_exact,
            "primary_failure_exact": args.primary_failure_exact,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
