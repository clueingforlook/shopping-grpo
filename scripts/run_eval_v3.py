#!/usr/bin/env python3
"""Run or offline-score one Base, SFT, or GRPO model with Eval v3."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (
    REPOSITORY,
    REPOSITORY / "src",
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from scripts import run_model_evaluation  # noqa: E402
from shopping_grpo.evaluation.eval_v3 import (  # noqa: E402
    EXPECTED_FINAL200_SHA256,
    file_sha256,
    run_offline_evaluation,
)
from shopping_grpo.harness.evaluation_batch import (  # noqa: E402
    EvaluationBatchSafetyPause,
)


DEFAULT_TASKS = REPOSITORY / "data/evaluation/tasks.jsonl"
DEFAULT_DIFFICULTY = (
    REPOSITORY
    / "outputs/evaluation-difficulty-v1/evaluation-difficulty.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="运行一个模型并立即生成 Eval v3 报告")
    _add_identity(run)
    run.add_argument("--served-model", required=True)
    run.add_argument("--model-artifact", type=Path, required=True)
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--tokenizer-revision")
    run.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    run.add_argument("--difficulty", type=Path, default=DEFAULT_DIFFICULTY)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--model-base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--shopsim-base-url", default="http://127.0.0.1:5700")
    run.add_argument("--api-key-file", type=Path)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--timeout", type=float, default=180.0)
    run.add_argument("--api-retries", type=int, default=2)
    run.add_argument("--no-progress", action="store_true")
    run.add_argument("--overwrite-score", action="store_true")

    score = subparsers.add_parser("score", help="只对已有轨迹生成 Eval v3 报告")
    _add_identity(score)
    score.add_argument("--trajectories", type=Path, required=True)
    score.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    score.add_argument("--difficulty", type=Path, default=DEFAULT_DIFFICULTY)
    score.add_argument("--run-manifest", type=Path)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--limit", type=int)
    score.add_argument("--overwrite", action="store_true")
    return parser


def _add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-role", choices=("base", "sft", "grpo"), required=True)
    parser.add_argument(
        "--model-name",
        required=True,
        help="报告中使用的模型/checkpoint 名称",
    )


def _score(args: argparse.Namespace, *, trajectories: Path, run_manifest: Path | None) -> dict:
    difficulty = args.difficulty if args.difficulty and args.difficulty.is_file() else None
    return run_offline_evaluation(
        trajectories_path=trajectories,
        tasks_path=args.tasks,
        output_dir=args.output_dir,
        model_role=args.model_role,
        model_name=args.model_name,
        difficulty_path=difficulty,
        run_manifest_path=run_manifest,
        limit=args.limit,
        overwrite=bool(
            getattr(args, "overwrite", False)
            or getattr(args, "overwrite_score", False)
        ),
    )


async def _run(args: argparse.Namespace) -> dict:
    if args.limit is None and file_sha256(args.tasks) != EXPECTED_FINAL200_SHA256:
        raise ValueError("正式 Eval v3 必须使用冻结的 Final-200 任务文件")
    collection_args = argparse.Namespace(
        model_role=args.model_role,
        served_model=args.served_model,
        model_artifact=args.model_artifact,
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        tasks=args.tasks,
        output_dir=args.output_dir,
        model_base_url=args.model_base_url,
        shopsim_base_url=args.shopsim_base_url,
        api_key_file=args.api_key_file,
        concurrency=args.concurrency,
        limit=args.limit,
        seed=args.seed,
        timeout=args.timeout,
        api_retries=args.api_retries,
        no_progress=args.no_progress,
    )
    batch = await run_model_evaluation.run_from_args(collection_args)
    if int(batch["remaining"]) != 0:
        raise RuntimeError(f"evaluation batch is incomplete: remaining={batch['remaining']}")
    summary = _score(
        args,
        trajectories=Path(batch["trajectories"]),
        run_manifest=Path(batch["manifest"]),
    )
    return {"batch": batch, "eval_v3": summary}


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            result = asyncio.run(_run(args))
        else:
            result = _score(
                args,
                trajectories=args.trajectories,
                run_manifest=args.run_manifest,
            )
    except EvaluationBatchSafetyPause as exc:
        raise SystemExit(
            f"环境安全状态暂停；修复环境后重跑同一命令即可续跑：{exc}"
        ) from exc
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
