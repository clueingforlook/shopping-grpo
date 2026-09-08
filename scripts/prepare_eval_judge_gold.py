#!/usr/bin/env python3
"""从现有 Base/SFT 轨迹生成待人工确认的 Judge Gold Set。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.eval_judge_gold import build_judge_gold_draft  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trajectories", type=Path, required=True)
    parser.add_argument("--candidate-trajectories", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-stratum", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = build_judge_gold_draft(
            baseline_trajectories_path=args.baseline_trajectories,
            candidate_trajectories_path=args.candidate_trajectories,
            rubrics_path=args.rubrics,
            output_dir=args.output_dir,
            pairs_per_stratum=args.pairs_per_stratum,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
