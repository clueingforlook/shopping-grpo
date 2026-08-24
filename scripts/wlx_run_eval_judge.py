#!/usr/bin/env python3
"""对已保存轨迹运行可续跑的 DeepSeek Eval Judge。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from wlx_harness_core.wlx_eval_judge_pipeline import (  # noqa: E402
    load_judge_cases,
    run_judge_batch,
)
from wlx_harness_core.wlx_eval_rubric_generator import (  # noqa: E402
    DeepSeekRubricClient,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-run",
        action="append",
        required=True,
        metavar="PREFIX=PATH",
        help="可重复，例如 base=.../wlx-trajectories.jsonl",
    )
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("calibration", "production"), required=True)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument(
        "--selected-cases",
        type=Path,
        help="Gold Set manifest；只运行其中 selected_cases",
    )
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--api-key-file", type=Path)
    return parser


def run_from_args(args: argparse.Namespace) -> dict:
    selected = _selected_cases(args.selected_cases)
    cases = []
    prefixes = set()
    for raw in args.trajectory_run:
        prefix, path = _trajectory_run(raw)
        if prefix in prefixes:
            raise ValueError(f"duplicate trajectory prefix: {prefix}")
        prefixes.add(prefix)
        task_ids = (
            [task_id for selected_prefix, task_id in selected if selected_prefix == prefix]
            if selected is not None
            else None
        )
        if selected is not None and not task_ids:
            continue
        cases.extend(
            load_judge_cases(
                trajectories_path=path,
                rubrics_path=args.rubrics,
                case_prefix=prefix,
                selected_task_ids=task_ids,
            )
        )
    if selected is not None and {case.case_id for case in cases} != {
        f"{prefix}:{task_id}" for prefix, task_id in selected
    }:
        raise ValueError("trajectory runs do not cover selected Gold cases")
    api_key = _load_api_key(args.api_key_file)
    client = DeepSeekRubricClient(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
    )

    def progress(value):
        print(
            f"WLX JUDGE: {value['completed']}/{value['total']} "
            f"case={value['case_id'] or '-'}",
            file=sys.stderr,
            flush=True,
        )

    return run_judge_batch(
        cases=cases,
        output_dir=args.output_dir,
        client=client,
        mode=args.mode,
        calibration_manifest_path=args.calibration_manifest,
        concurrency=args.concurrency,
        progress_callback=progress,
    )


def _trajectory_run(value: str) -> tuple[str, Path]:
    prefix, separator, path = str(value).partition("=")
    if not separator or not prefix.strip() or not path.strip() or ":" in prefix:
        raise ValueError("--trajectory-run must use PREFIX=PATH and PREFIX cannot contain ':'")
    return prefix.strip(), Path(path.strip())


def _selected_cases(path: Path | None) -> set[tuple[str, int]] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("selected_cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Gold manifest missing selected_cases")
    result = set()
    for row in rows:
        prefix = str(row.get("run_prefix") or "")
        task_id = row.get("task_id")
        if not prefix or isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError("Gold manifest contains invalid selected case")
        result.add((prefix, task_id))
    return result


def _load_api_key(path: Path | None) -> str:
    if path is None:
        value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not value:
            raise ValueError("set DEEPSEEK_API_KEY or provide --api-key-file")
        return value
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError("API key file permissions must be 600 or stricter")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("API key file is empty")
    return value


def main() -> None:
    try:
        result = run_from_args(build_parser().parse_args())
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
