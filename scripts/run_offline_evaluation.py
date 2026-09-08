#!/usr/bin/env python3
"""合并已保存轨迹与 Eval v2 Judge 结果，不重新运行 Actor 或环境。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.eval_pipeline import (  # noqa: E402
    OfflineEvaluationOutputExistsError,
    run_offline_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="默认写到轨迹所在 Run 的 wlx-eval-v2 子目录",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="默认自动读取轨迹同目录的 run-manifest.json",
    )
    parser.add_argument(
        "--rubrics",
        type=Path,
        help="已冻结的 rubrics.jsonl；提供后启用正式 Rubric 需求检查",
    )
    parser.add_argument("--judge-results", type=Path)
    parser.add_argument(
        "--judge-case-prefix",
        help="Judge case_id 的前缀，例如 base 或 sft",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir or args.trajectories.parent / "eval-v2"
    return run_offline_evaluation(
        trajectories_path=args.trajectories,
        output_dir=output_dir,
        run_manifest_path=args.run_manifest,
        rubrics_path=args.rubrics,
        judge_results_path=args.judge_results,
        judge_case_prefix=args.judge_case_prefix,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run_from_args(args)
    except (FileNotFoundError, OfflineEvaluationOutputExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
