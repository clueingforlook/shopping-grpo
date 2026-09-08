#!/usr/bin/env python3
"""按 task_id 对齐两个 Eval v2 Run 并生成成对比较。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
CORE = REPOSITORY / "src"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from shopping_grpo.harness.eval_report import (  # noqa: E402
    compare_evaluation_runs,
    render_comparison_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-name", default="base")
    parser.add_argument("--candidate-name", default="sft")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def compare_from_args(args: argparse.Namespace) -> dict:
    baseline = _read_jsonl(args.baseline)
    candidate = _read_jsonl(args.candidate)
    summary, paired = compare_evaluation_runs(
        baseline,
        candidate,
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
    )
    outputs = {
        "summary_json": args.output_dir / "comparison-summary.json",
        "summary_markdown": args.output_dir / "comparison-summary.md",
        "paired_results": args.output_dir / "paired-task-results.jsonl",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有比较产物：{existing[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_new(
        outputs["summary_json"],
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        outputs["summary_markdown"],
        render_comparison_markdown(summary),
    )
    _write_new(
        outputs["paired_results"],
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in paired
        ),
    )
    return {
        "paired_tasks": len(paired),
        **{name: str(path) for name, path in outputs.items()},
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"评测文件不存在：{path}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON 对象")
            rows.append(row)
    return rows


def _write_new(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text if text.endswith("\n") else text + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = compare_from_args(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
