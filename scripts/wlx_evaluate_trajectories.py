#!/usr/bin/env python3
"""对已保存的 WLX Harness 轨迹生成确定性的 Eval v2 逐题记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
WLX_CORE = REPOSITORY / "wlx-harness-core"
if str(WLX_CORE) not in sys.path:
    sys.path.insert(0, str(WLX_CORE))

from wlx_harness_core.wlx_trajectory_evaluator import evaluate_trajectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="评估 WLX 轨迹的格式、购买、资格、行为与失败归因"
    )
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def evaluate_jsonl(input_path: Path, output_path: Path, *, force: bool = False) -> int:
    if not input_path.is_file():
        raise FileNotFoundError(f"轨迹文件不存在：{input_path}")
    if output_path.exists() and not force:
        raise FileExistsError(f"拒绝覆盖已有输出：{output_path}")
    rows = []
    with input_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                trajectory = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是合法 JSON") from exc
            rows.append(evaluate_trajectory(trajectory).to_dict())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return len(rows)


def main() -> None:
    args = parse_args()
    try:
        count = evaluate_jsonl(args.trajectories, args.output, force=args.force)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"evaluated_trajectories": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
