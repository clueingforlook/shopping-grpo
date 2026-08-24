#!/usr/bin/env python3
"""使用规则与 DeepSeek 为 WLX Eval 任务生成、复核并冻结 Rubric。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping
from urllib.request import Request, urlopen

from tqdm.auto import tqdm


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from wlx_harness_core.wlx_eval_rubric_generator import (  # noqa: E402
    DeepSeekRubricClient,
)
from wlx_harness_core.wlx_eval_rubric_pipeline import (  # noqa: E402
    finalize_rubric_set,
    load_rubric_tasks,
    run_rubric_generation,
)
from wlx_harness_core.wlx_sft_storage import file_sha256  # noqa: E402


DEFAULT_TASKS = REPOSITORY / "data/evaluation/tasks.jsonl"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-config",
        help="只验证 DeepSeek Key、接口和模型，不生成 Rubric",
    )
    _add_api_arguments(check)

    generate = subparsers.add_parser(
        "generate",
        help="为每个任务生成一次 Rubric 候选并建立复核队列",
    )
    generate.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    generate.add_argument(
        "--trajectories",
        type=Path,
        help="tasks 只有 task_id 时，从完整轨迹的初始请求读取用户需求",
    )
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--rubric-set-version", required=True)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--concurrency", type=int, default=1)
    generate.add_argument("--no-progress", action="store_true")
    _add_api_arguments(generate)

    finalize = subparsers.add_parser(
        "finalize",
        help="应用人工复核决定，冻结覆盖全部任务的 Rubric 集",
    )
    finalize.add_argument("--generation-dir", type=Path, required=True)
    finalize.add_argument("--review-decisions", type=Path)
    return parser


def _add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=(
            Path(os.environ["WLX_DEEPSEEK_API_KEY_FILE"])
            if os.environ.get("WLX_DEEPSEEK_API_KEY_FILE")
            else None
        ),
        help="可选：只含 Key 且权限为 600 的文件；不会写入产物",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
    )
    parser.add_argument(
        "--deepseek-model",
        default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--response-format-json",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "finalize":
        result = finalize_rubric_set(
            generation_dir=args.generation_dir,
            review_decisions_path=args.review_decisions,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    api_key, api_key_source = _load_api_key(args.api_key_file)
    if args.command == "check-config":
        result = _check_config(
            base_url=args.deepseek_base_url,
            model=args.deepseek_model,
            api_key=api_key,
            timeout=args.timeout,
            api_key_source=api_key_source,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.concurrency < 1 or args.max_tokens < 1:
        raise SystemExit("--concurrency 和 --max-tokens 必须是正整数")
    if args.timeout <= 0 or args.max_retries < 0:
        raise SystemExit("--timeout 必须大于0，--max-retries 不能小于0")
    tasks = load_rubric_tasks(
        args.tasks,
        trajectories_path=args.trajectories,
        limit=args.limit,
    )
    client = DeepSeekRubricClient(
        model=args.deepseek_model,
        base_url=args.deepseek_base_url,
        api_key=api_key,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        response_format_json=args.response_format_json,
    )
    progress_bar = None

    def update_progress(snapshot: Mapping[str, Any]) -> None:
        nonlocal progress_bar
        if progress_bar is None:
            progress_bar = tqdm(
                total=int(snapshot["total"]),
                initial=int(snapshot["completed"]),
                desc="WLX RUBRIC",
                unit="task",
                dynamic_ncols=True,
            )
        else:
            progress_bar.update(
                max(0, int(snapshot["completed"]) - int(progress_bar.n))
            )
        progress_bar.set_postfix(
            {
                "review": int(snapshot["review_required"]),
                "errors": int(snapshot["technical_errors"]),
                "task_id": snapshot.get("task_id"),
            },
            refresh=True,
        )

    try:
        result = run_rubric_generation(
            tasks=tasks,
            output_dir=args.output_dir,
            client=client,
            rubric_set_version=args.rubric_set_version,
            source_contract={
                "tasks": str(args.tasks),
                "tasks_sha256": file_sha256(args.tasks),
                "trajectories": (
                    str(args.trajectories) if args.trajectories is not None else None
                ),
                "trajectories_sha256": (
                    file_sha256(args.trajectories)
                    if args.trajectories is not None
                    else None
                ),
                "instruction_source": (
                    "task_file_or_initial_trajectory_request"
                    if args.trajectories is not None
                    else "task_file"
                ),
                "limit": args.limit,
                "api_key_source": api_key_source,
            },
            concurrency=args.concurrency,
            progress_callback=(None if args.no_progress else update_progress),
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("complete") is True else 2


def _load_api_key(path: Path | None) -> tuple[str, str]:
    """复用 SFT 采样约定：私有 Key 文件优先，环境变量后备。"""

    if path is not None:
        if not path.is_file():
            raise SystemExit(f"找不到 DeepSeek Key 文件：{path}")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise SystemExit("DeepSeek Key 文件权限过宽，请先执行 chmod 600")
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise SystemExit("DeepSeek Key 文件为空")
        return value, "permission-checked key file"
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if value:
        return value, "DEEPSEEK_API_KEY environment variable"
    raise SystemExit(
        "缺少 DeepSeek Key：设置 DEEPSEEK_API_KEY，或通过 "
        "WLX_DEEPSEEK_API_KEY_FILE/--api-key-file 提供权限为600的文件"
    )


def _check_config(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    api_key_source: str,
) -> dict[str, Any]:
    request = Request(
        f"{str(base_url).rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "wlx-rubric-generator/1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("DeepSeek /models 响应必须是 JSON 对象")
    model_ids = sorted(
        str(item["id"])
        for item in payload.get("data") or []
        if isinstance(item, Mapping) and item.get("id")
    )
    if model not in model_ids:
        raise ValueError(f"DeepSeek 服务未暴露模型 {model!r}；当前可用：{model_ids}")
    return {
        "deepseek_api": "ok",
        "base_url": str(base_url).rstrip("/"),
        "model": model,
        "api_key_source": api_key_source,
        "available_models": model_ids,
    }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
