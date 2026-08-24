#!/usr/bin/env python3
"""运行 WLX SFT 的难度预计算、采样计划、在线采样和数据集构建。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from wlx_harness_core.wlx_sft_contracts import (  # noqa: E402
    SamplingConfig,
    SamplingMode,
)
from wlx_harness_core.wlx_sft_dataset import DatasetBuildConfig  # noqa: E402
from wlx_harness_core.wlx_sft_difficulty import (  # noqa: E402
    TaskDifficultyFeatures,
    assign_preliminary_labels,
    build_task_features,
    calibration_samples_from_rows,
    fit_with_grouped_validation,
    independent_constraint_count,
    LogisticDifficultyModel,
    percentile_95,
)
from wlx_harness_core.wlx_sft_pipeline import (  # noqa: E402
    WlxSftDataPipeline,
    default_sft_harness_config,
    sft_harness_contract,
    sft_harness_contract_fingerprint,
)
from wlx_harness_core.wlx_sft_policy import OpenAICompatibleTeacherPolicy  # noqa: E402
from wlx_harness_core.wlx_sft_prompt import (  # noqa: E402
    WLX_SFT_SYSTEM_PROMPT_SHA256,
    WLX_SFT_SYSTEM_PROMPT_VERSION,
)
from wlx_harness_core.wlx_sft_shopsim_difficulty import (  # noqa: E402
    precompute_retrieval_cache,
    retrieval_evidence_from_cache,
)
from wlx_harness_core.wlx_sft_storage import (  # noqa: E402
    assert_no_held_out_tasks,
    load_held_out_task_ids,
    load_sft_tasks,
    read_jsonl,
    write_jsonl,
)
from wlx_harness_core.wlx_sft_tasks import (  # noqa: E402
    apply_difficulty_model,
    calibration_task_plan,
    public_tasks_from_canonical_goals,
    write_public_task_plan,
)
from wlx_harness_core.wlx_sft_tokenizer import (  # noqa: E402
    DEFAULT_DEEPSEEK_V4_TOKENIZER,
    DEFAULT_DEEPSEEK_V4_TOKENIZER_REVISION,
    DeepSeekV4RuntimeTokenCounter,
    TransformersTrainingTokenCounter,
)
from wlx_harness_core.wlx_tools import WLX_SFT_TOOL_REGISTRY  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """创建分阶段 CLI；每个子命令只做一件事，便于一步一步执行。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-config",
        help="不生成轨迹，只检查 DeepSeek Key、模型列表和 Teacher tokenizer",
    )
    check.add_argument(
        "--api-key-file",
        type=Path,
        default=(
            Path(os.environ["WLX_DEEPSEEK_API_KEY_FILE"])
            if os.environ.get("WLX_DEEPSEEK_API_KEY_FILE")
            else None
        ),
    )
    check.add_argument(
        "--teacher-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    check.add_argument(
        "--teacher-model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    )
    check.add_argument(
        "--teacher-tokenizer",
        default=os.environ.get(
            "WLX_TEACHER_TOKENIZER",
            DEFAULT_DEEPSEEK_V4_TOKENIZER,
        ),
    )
    check.add_argument(
        "--teacher-tokenizer-revision",
        default=os.environ.get(
            "WLX_TEACHER_TOKENIZER_REVISION",
            DEFAULT_DEEPSEEK_V4_TOKENIZER_REVISION,
        ),
    )
    check.add_argument("--timeout", type=float, default=30.0)

    precompute = subparsers.add_parser(
        "precompute-difficulty",
        help="复用 ShopSimulator 搜索和 Reward 生成公开任务表与 C/R/N 特征",
    )
    precompute.add_argument("--output-dir", type=Path, required=True)
    precompute.add_argument(
        "--held-out-tasks",
        type=Path,
        default=Path("data/evaluation/tasks.jsonl"),
    )
    precompute.add_argument(
        "--product-file",
        type=Path,
        default=Path("environments/ShopSimulator/shop_env/data/items_eval_train.json"),
    )
    precompute.add_argument(
        "--search-index",
        type=Path,
        default=Path("environments/ShopSimulator/shop_env/search_engine/products.sqlite3"),
    )

    plan_calibration = subparsers.add_parser(
        "plan-calibration",
        help="从规则三档中选择约 200 道校准题",
    )
    plan_calibration.add_argument("--tasks", type=Path, required=True)
    plan_calibration.add_argument("--features", type=Path, required=True)
    plan_calibration.add_argument("--output", type=Path, required=True)
    plan_calibration.add_argument("--size", type=int, default=200)
    plan_calibration.add_argument("--seed", type=int, default=42)

    fit = subparsers.add_parser(
        "fit-difficulty",
        help="用约 600 次有效结果拟合并验证小型逻辑回归",
    )
    fit.add_argument("--plan", type=Path, required=True)
    fit.add_argument("--raw", type=Path, required=True)
    fit.add_argument("--features", type=Path, required=True)
    fit.add_argument("--manifest", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--validation-ratio", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=42)

    plan_formal = subparsers.add_parser(
        "plan-formal",
        help="把校准后的难度标签写入正式任务计划",
    )
    plan_formal.add_argument("--tasks", type=Path, required=True)
    plan_formal.add_argument("--features", type=Path, required=True)
    plan_formal.add_argument("--model", type=Path, required=True)
    plan_formal.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser(
        "collect",
        help="使用 WLX Harness Core 在线采集 calibration 或 formal 轨迹",
    )
    collect.add_argument("--tasks", type=Path, required=True)
    collect.add_argument("--raw", type=Path, required=True)
    collect.add_argument(
        "--held-out-tasks",
        type=Path,
        default=Path("data/evaluation/tasks.jsonl"),
    )
    collect.add_argument("--mode", choices=("calibration", "formal"), required=True)
    collect.add_argument(
        "--api-key-file",
        type=Path,
        default=(
            Path(os.environ["WLX_DEEPSEEK_API_KEY_FILE"])
            if os.environ.get("WLX_DEEPSEEK_API_KEY_FILE")
            else None
        ),
        help="可选：只含 Key 且权限为 600 的文件；不会写入轨迹",
    )
    collect.add_argument("--limit", type=int)
    collect.add_argument("--concurrency", type=int, default=1)
    collect.add_argument("--technical-retries", type=int, default=2)
    collect.add_argument("--target-gold", type=int)
    collect.add_argument(
        "--shopsim-base-url",
        default=os.environ.get("SHOPSIM_BASE_URL", "http://127.0.0.1:5700"),
    )
    collect.add_argument(
        "--teacher-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    collect.add_argument(
        "--teacher-model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    )
    collect.add_argument(
        "--teacher-tokenizer",
        default=os.environ.get(
            "WLX_TEACHER_TOKENIZER",
            DEFAULT_DEEPSEEK_V4_TOKENIZER,
        ),
    )
    collect.add_argument(
        "--teacher-tokenizer-revision",
        default=os.environ.get(
            "WLX_TEACHER_TOKENIZER_REVISION",
            DEFAULT_DEEPSEEK_V4_TOKENIZER_REVISION,
        ),
    )
    collect.add_argument("--temperature", type=float, default=0.2)
    collect.add_argument("--top-p", type=float, default=1.0)
    collect.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default="enabled",
    )
    collect.add_argument(
        "--reasoning-effort",
        choices=("high", "max"),
        default="high",
    )
    collect.add_argument("--timeout", type=float, default=180.0)
    collect.add_argument("--api-retries", type=int, default=2)

    build = subparsers.add_parser(
        "build",
        help="用最终训练 tokenizer 清洗、切分并冻结严格 Gold SFT 数据",
    )
    build.add_argument("--raw", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument(
        "--held-out-tasks",
        type=Path,
        default=Path("data/evaluation/tasks.jsonl"),
    )
    build.add_argument(
        "--training-tokenizer",
        default=os.environ.get("WLX_TRAINING_TOKENIZER", "Qwen/Qwen3.5-2B"),
    )
    build.add_argument("--training-tokenizer-revision")
    build.add_argument("--context-limit", type=int, default=24_576)
    build.add_argument("--validation-ratio", type=float, default=0.1)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--target-size", type=int)
    build.add_argument(
        "--artifact-prefix",
        default="wlx-",
        help="构建产物文件名前缀；默认满足 WLX 新文件命名约定",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析一个子命令并调用对应阶段；不会自动跨阶段执行昂贵操作。"""

    args = build_parser().parse_args(argv)
    if args.command == "check-config":
        return _check_config(args)
    if args.command == "precompute-difficulty":
        return _precompute_difficulty(args)
    if args.command == "plan-calibration":
        return _plan_calibration(args)
    if args.command == "fit-difficulty":
        return _fit_difficulty(args)
    if args.command == "plan-formal":
        return _plan_formal(args)
    if args.command == "collect":
        return asyncio.run(_collect(args))
    if args.command == "build":
        return _build_dataset(args)
    raise AssertionError(f"未知子命令：{args.command}")


def _check_config(args: argparse.Namespace) -> int:
    """读取模型列表并加载 tokenizer；不发 Chat Completion，因此不产生采样轨迹。"""

    from urllib.request import Request, urlopen

    api_key, _ = _load_api_key(args.api_key_file)
    if not api_key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY 或 --api-key-file")
    request = Request(
        f"{str(args.teacher_base_url).rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "wlx-sft-pipeline/1",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"DeepSeek 配置检查失败：{exc.__class__.__name__}") from exc
    models = {
        str(item.get("id"))
        for item in payload.get("data") or []
        if isinstance(item, Mapping) and item.get("id")
    }
    if args.teacher_model not in models:
        raise SystemExit(
            f"模型 {args.teacher_model!r} 不在当前账号返回的模型列表：{sorted(models)}"
        )
    counter = DeepSeekV4RuntimeTokenCounter.from_pretrained(
        args.teacher_tokenizer,
        revision=args.teacher_tokenizer_revision,
        thinking_mode="enabled",
        reasoning_effort="high",
    )
    smoke_tokens = counter.count_chat(
        [{"role": "user", "content": "配置检查"}],
        [],
    )
    print(
        json.dumps(
            {
                "deepseek_api": "ok",
                "teacher_model": args.teacher_model,
                "teacher_tokenizer": args.teacher_tokenizer,
                "teacher_tokenizer_revision": args.teacher_tokenizer_revision,
                "tokenizer_smoke_tokens": smoke_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _precompute_difficulty(args: argparse.Namespace) -> int:
    """加载冻结商品库，生成公开任务表、检索缓存和带临时分档的特征表。"""

    shopsim_root = REPOSITORY / "environments" / "ShopSimulator" / "shop_env"
    if str(shopsim_root) not in sys.path:
        sys.path.insert(0, str(shopsim_root))
    os.environ["SHOP_SEARCH_INDEX"] = str(args.search_index.resolve())
    try:
        from web_agent_site.engine.engine import init_search_engine, load_products
        from web_agent_site.engine.goal import get_goals
    except ImportError as exc:
        raise SystemExit(
            "无法导入 ShopSimulator；请使用它的 Python 环境运行这个子命令"
        ) from exc
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_products, product_items, product_prices, _ = load_products(
        filepath=str(args.product_file),
        num_products=None,
    )
    goals = get_goals(all_products, product_prices)
    held_out = load_held_out_task_ids(args.held_out_tasks)
    train_task_ids = set(range(len(goals))).difference(held_out)
    searcher = init_search_engine(product_filepath=str(args.product_file))
    cache_path = output_dir / "wlx-retrieval-cache.jsonl"
    try:
        precompute_retrieval_cache(
            goals=goals,
            product_item_dict=product_items,
            searcher=searcher,
            output_path=cache_path,
            task_ids=train_task_ids,
        )
    finally:
        searcher.close()
    public_tasks = public_tasks_from_canonical_goals(goals, held_out_task_ids=held_out)
    write_public_task_plan(output_dir / "wlx-public-train-tasks.jsonl", public_tasks)
    retrieval_rows = {int(row["task_id"]): row for row in read_jsonl(cache_path)}
    train_goal_pairs = [
        (task_id, goal)
        for task_id, goal in enumerate(goals)
        if task_id not in held_out
    ]
    a95 = percentile_95(independent_constraint_count(goal)[0] for _, goal in train_goal_pairs)
    n95 = percentile_95(
        int(retrieval_rows[task_id].get("near_miss_count_at_20", 0))
        for task_id, _ in train_goal_pairs
    )
    features = [
        build_task_features(
            task_id=task_id,
            goal=goal,
            retrieval=retrieval_evidence_from_cache(retrieval_rows[task_id]),
            a95=a95,
            n95=n95,
            category=str(goal.get("category") or "") or None,
        )
        for task_id, goal in train_goal_pairs
    ]
    features = assign_preliminary_labels(features)
    write_jsonl(
        output_dir / "wlx-difficulty-features.jsonl",
        (item.to_dict() for item in features),
    )
    _write_json(
        output_dir / "wlx-difficulty-stats.json",
        {"train_tasks": len(features), "a95": a95, "n95": n95},
    )
    print(f"完成：{len(features)} 道 Train 任务，A95={a95}，N95={n95}")
    return 0


def _plan_calibration(args: argparse.Namespace) -> int:
    """读取公开任务与规则特征，输出不含隐藏答案的校准任务计划。"""

    tasks = load_sft_tasks(args.tasks)
    features = [TaskDifficultyFeatures.from_dict(row) for row in read_jsonl(args.features)]
    plan = calibration_task_plan(tasks, features, sample_size=args.size, seed=args.seed)
    write_public_task_plan(args.output, plan)
    print(f"完成：校准计划共 {len(plan)} 道任务")
    return 0


def _fit_difficulty(args: argparse.Namespace) -> int:
    """从校准 raw 和静态特征训练逻辑回归，并保存验证报告与正式模型。"""

    plan_rows = list(read_jsonl(args.plan))
    raw_rows = list(read_jsonl(args.raw))
    expected_task_ids = _validate_fit_attempts(plan_rows, raw_rows)
    version_context = _fit_version_context(
        plan_path=args.plan,
        raw_path=args.raw,
        features_path=args.features,
        manifest_path=args.manifest,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )
    features = {
        item.task_id: item
        for item in (
            TaskDifficultyFeatures.from_dict(row) for row in read_jsonl(args.features)
        )
    }
    samples = calibration_samples_from_rows(raw_rows, features)
    model, report = fit_with_grouped_validation(
        samples,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        version_context=version_context,
        expected_task_ids=expected_task_ids,
    )
    report["fit_provenance"] = version_context
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"输出目录已有内容，请换新目录：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "wlx-difficulty-model.json", model.to_dict())
    _write_json(output_dir / "wlx-difficulty-report.json", report)
    print(f"完成：{model.difficulty_version}")
    return 0


def _validate_fit_attempts(
    plan_rows: list[Mapping[str, Any]],
    raw_rows: list[Mapping[str, Any]],
) -> set[int]:
    """要求计划内每题恰有 0/1/2 三次有效尝试，技术重试不参与计数。"""

    if not plan_rows:
        raise SystemExit("校准计划为空，不能拟合难度模型")
    plan_task_ids = [_required_task_id(row, source="校准计划") for row in plan_rows]
    duplicate_plan_ids = sorted(
        task_id for task_id in set(plan_task_ids) if plan_task_ids.count(task_id) > 1
    )
    if duplicate_plan_ids:
        raise SystemExit(f"校准计划包含重复 task_id：{duplicate_plan_ids}")
    expected_task_ids = set(plan_task_ids)
    valid_attempts: dict[int, list[int]] = {
        task_id: [] for task_id in expected_task_ids
    }
    unexpected_task_ids: set[int] = set()
    for row in raw_rows:
        task_id = _required_task_id(row, source="校准 raw")
        if task_id not in expected_task_ids:
            unexpected_task_ids.add(task_id)
            continue
        if row.get("attempt_valid") is not True:
            continue
        attempt_index = row.get("attempt_index")
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
            raise SystemExit(
                f"task_id={task_id} 的有效尝试缺少整数 attempt_index"
            )
        valid_attempts[task_id].append(attempt_index)
    if unexpected_task_ids:
        raise SystemExit(
            "校准 raw 包含计划外 task_id：" + str(sorted(unexpected_task_ids))
        )
    invalid_attempts = {
        task_id: sorted(indices)
        for task_id, indices in valid_attempts.items()
        if sorted(indices) != [0, 1, 2]
    }
    if invalid_attempts:
        preview = "; ".join(
            f"task_id={task_id}: {indices}"
            for task_id, indices in sorted(invalid_attempts.items())[:20]
        )
        suffix = "" if len(invalid_attempts) <= 20 else "；其余省略"
        raise SystemExit(
            "每道计划任务必须恰有 attempt_index=0,1,2 三次有效尝试；"
            + preview
            + suffix
        )
    return expected_task_ids


def _required_task_id(row: Mapping[str, Any], *, source: str) -> int:
    """从计划或 raw 中读取非布尔整数 task_id，并给出明确错误。"""

    task_id = row.get("task_id")
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        raise SystemExit(f"{source} 中存在缺少整数 task_id 的记录")
    return task_id


def _fit_version_context(
    *,
    plan_path: Path,
    raw_path: Path,
    features_path: Path,
    manifest_path: Path,
    seed: int,
    validation_ratio: float,
) -> dict[str, Any]:
    """核对冻结清单，并返回不含路径、可进入 difficulty_version 的语义上下文。"""

    manifest = _read_json_object(manifest_path, description="校准合并清单")
    semantic_contract = manifest.get("semantic_contract")
    if not isinstance(semantic_contract, Mapping):
        raise SystemExit("校准合并清单缺少 semantic_contract 对象")
    _assert_no_secret_fields(semantic_contract)
    final_artifacts = manifest.get("final_artifacts")
    if not isinstance(final_artifacts, Mapping):
        raise SystemExit("校准合并清单缺少 final_artifacts 对象")
    actual_hashes = {
        "plan_sha256": _sha256_file(plan_path),
        "raw_sha256": _sha256_file(raw_path),
        "features_sha256": _sha256_file(features_path),
        "manifest_sha256": _sha256_file(manifest_path),
    }
    for name in ("plan_sha256", "raw_sha256"):
        recorded = final_artifacts.get(name)
        if not isinstance(recorded, str) or recorded != actual_hashes[name]:
            raise SystemExit(
                f"校准合并清单的 {name} 与实际文件不一致，拒绝拟合"
            )
    return {
        "fit_contract_version": "wlx-difficulty-fit-contract-v1",
        "semantic_contract": dict(semantic_contract),
        **actual_hashes,
        "seed": int(seed),
        "validation_ratio": float(validation_ratio),
    }


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    """读取一个 JSON 对象；拟合来源文件损坏时不继续使用模糊配置。"""

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取{description}：{path}") from exc
    if not isinstance(value, Mapping):
        raise SystemExit(f"{description}必须是 JSON 对象：{path}")
    return dict(value)


def _assert_no_secret_fields(value: Mapping[str, Any]) -> None:
    """拒绝把疑似凭据字段复制进模型版本或公开拟合报告。"""

    forbidden_markers = (
        "api_key",
        "apikey",
        "secret",
        "authorization",
        "credential",
        "access_token",
        "bearer",
    )
    pending: list[Any] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                normalized = str(key).lower()
                if any(marker in normalized for marker in forbidden_markers):
                    raise SystemExit(
                        f"semantic_contract 含疑似凭据字段 {key!r}，拒绝写入拟合产物"
                    )
                pending.append(item)
        elif isinstance(current, (list, tuple)):
            pending.extend(current)


def _plan_formal(args: argparse.Namespace) -> int:
    """把正式难度模型应用到全部公开 Train 任务并输出派发计划。"""

    tasks = load_sft_tasks(args.tasks)
    features = {
        item.task_id: item
        for item in (
            TaskDifficultyFeatures.from_dict(row) for row in read_jsonl(args.features)
        )
    }
    model = LogisticDifficultyModel.from_dict(
        json.loads(Path(args.model).read_text(encoding="utf-8"))
    )
    plan = apply_difficulty_model(tasks, features, model)
    write_public_task_plan(args.output, plan)
    print(f"完成：正式任务计划共 {len(plan)} 道")
    return 0


async def _collect(args: argparse.Namespace) -> int:
    """检查泄漏和安全密钥来源后，用独立策略实例并发采集轨迹。"""

    api_key, api_key_source = _load_api_key(args.api_key_file)
    missing = [
        name
        for name, value in (
            ("DEEPSEEK_API_KEY 或 --api-key-file", api_key),
            ("--teacher-base-url 或 DEEPSEEK_BASE_URL", args.teacher_base_url),
            ("--teacher-model 或 DEEPSEEK_MODEL", args.teacher_model),
            ("--teacher-tokenizer 或 WLX_TEACHER_TOKENIZER", args.teacher_tokenizer),
        )
        if not value
    ]
    if missing:
        raise SystemExit("缺少配置：" + "；".join(missing))
    tasks = load_sft_tasks(args.tasks)
    held_out = load_held_out_task_ids(args.held_out_tasks)
    assert_no_held_out_tasks(tasks, held_out)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit 必须至少为 1")
        tasks = tasks[: args.limit]
    runtime_counter = DeepSeekV4RuntimeTokenCounter.from_pretrained(
        args.teacher_tokenizer,
        revision=args.teacher_tokenizer_revision,
        thinking_mode=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )
    harness_config = default_sft_harness_config(
        environment_base_url=args.shopsim_base_url
    )
    pipeline = WlxSftDataPipeline(harness_config=harness_config)

    def policy_factory(task, attempt_index, technical_retry_index):
        """每条 in-flight 轨迹创建独立策略，避免并发覆盖 Token 与上下文状态。"""

        return OpenAICompatibleTeacherPolicy(
            model=args.teacher_model,
            base_url=args.teacher_base_url,
            api_key=api_key,
            context_policy=harness_config.context,
            observation_policy=harness_config.observation,
            count_chat_tokens=runtime_counter.count_chat,
            count_text_tokens=runtime_counter.count_text,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout_s=args.timeout,
            max_retries=args.api_retries,
            thinking_mode=args.thinking,
            reasoning_effort=args.reasoning_effort,
        )

    metadata_path = Path(args.raw).with_name("wlx-collection-config.json")
    harness_contract = sft_harness_contract(harness_config, pipeline.runner)
    collection_config = {
        "teacher_base_url": str(args.teacher_base_url).rstrip("/"),
        "teacher_model": args.teacher_model,
        "teacher_tokenizer": args.teacher_tokenizer,
        "teacher_tokenizer_revision": args.teacher_tokenizer_revision,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "temperature_effective": args.thinking == "disabled",
        "shopsim_base_url": str(args.shopsim_base_url).rstrip("/"),
        "mode": args.mode,
        "task_plan_sha256": _sha256_file(Path(args.tasks)),
        "held_out_tasks_sha256": _sha256_file(Path(args.held_out_tasks)),
        "task_limit": args.limit,
        "tasks_selected": len(tasks),
        "concurrency": args.concurrency,
        "technical_retries": args.technical_retries,
        "api_retries": args.api_retries,
        "timeout_s": args.timeout,
        "api_key_source": api_key_source,
        "system_prompt_version": WLX_SFT_SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": WLX_SFT_SYSTEM_PROMPT_SHA256,
        "tool_schema_version": WLX_SFT_TOOL_REGISTRY.version,
        "tool_schema_fingerprint": WLX_SFT_TOOL_REGISTRY.fingerprint,
        "generation_reserve_tokens": harness_config.context.generation_reserve_tokens,
        "harness_contract": harness_contract,
        "harness_contract_sha256": sft_harness_contract_fingerprint(
            harness_config,
            pipeline.runner,
        ),
    }
    raw_path = Path(args.raw)
    if metadata_path.exists():
        _assert_collection_config_compatible(metadata_path, collection_config)
    elif raw_path.exists() and raw_path.stat().st_size:
        raise SystemExit(
            "raw 文件已有内容但缺少 wlx-collection-config.json；"
            "无法证明 Prompt 一致，请换一个新的采样目录"
        )
    else:
        _write_json(metadata_path, collection_config)
    report = await pipeline.collect(
        tasks=tasks,
        raw_path=args.raw,
        sampling_config=SamplingConfig(
            mode=SamplingMode(args.mode),
            concurrency=args.concurrency,
            max_technical_retries_per_attempt=args.technical_retries,
            target_gold_trajectories=args.target_gold,
        ),
        policy_factory=policy_factory,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _build_dataset(args: argparse.Namespace) -> int:
    """加载最终训练 tokenizer，构建不截断且严格 Gold 的冻结数据集。"""

    if not args.training_tokenizer:
        raise SystemExit("缺少 --training-tokenizer 或 WLX_TRAINING_TOKENIZER")
    counter = TransformersTrainingTokenCounter.from_pretrained(
        args.training_tokenizer,
        revision=args.training_tokenizer_revision,
    )
    summary = WlxSftDataPipeline().build_dataset(
        raw_path=args.raw,
        output_dir=args.output_dir,
        held_out_tasks_path=args.held_out_tasks,
        training_token_counter=counter,
        build_config=DatasetBuildConfig(
            context_limit=args.context_limit,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            target_size=args.target_size,
        ),
        artifact_prefix=args.artifact_prefix,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_api_key(api_key_file: Path | None) -> tuple[str | None, str | None]:
    """显式 Key 文件优先，环境变量只作后备；永远不回显 Key 内容。"""

    if api_key_file is not None:
        path = Path(api_key_file)
        if not path.is_file():
            raise SystemExit(f"找不到 DeepSeek Key 文件：{path}")
        if path.stat().st_mode & 0o077:
            raise SystemExit("DeepSeek Key 文件权限过宽，请先执行 chmod 600")
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise SystemExit("DeepSeek Key 文件是空的")
        return value, "permission-checked key file"
    environment_value = os.environ.get("DEEPSEEK_API_KEY")
    if environment_value:
        return environment_value, "DEEPSEEK_API_KEY environment variable"
    return None, None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """把小型配置或报告写成易读 JSON；调用方负责使用新的版本目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_collection_config_compatible(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    """续采前确认影响轨迹语义的配置一致，防止新旧 Prompt 混入同一 raw。"""

    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"无法读取既有采样配置：{path}") from exc
    if not isinstance(existing, Mapping):
        raise SystemExit(f"既有采样配置不是 JSON 对象：{path}")
    contract_keys = (
        "teacher_base_url",
        "teacher_model",
        "teacher_tokenizer",
        "teacher_tokenizer_revision",
        "thinking",
        "reasoning_effort",
        "shopsim_base_url",
        "mode",
        "task_plan_sha256",
        "held_out_tasks_sha256",
        "task_limit",
        "tasks_selected",
        "system_prompt_version",
        "system_prompt_sha256",
        "tool_schema_version",
        "tool_schema_fingerprint",
        "generation_reserve_tokens",
        "harness_contract_sha256",
    )
    if expected.get("temperature_effective"):
        contract_keys += ("temperature", "top_p")
    missing_expected = [key for key in contract_keys if key not in expected]
    if missing_expected:
        raise ValueError(
            "当前采样配置缺少契约字段：" + "、".join(missing_expected)
        )
    mismatches = [
        key for key in contract_keys if existing.get(key) != expected.get(key)
    ]
    if mismatches:
        raise SystemExit(
            "既有采样目录与当前 WLX 契约不一致（"
            + "、".join(mismatches)
            + "）；请换一个新的 raw/output 目录，不能混合续采"
        )


def _sha256_file(path: Path) -> str:
    """流式计算公开任务或 held-out 文件指纹，不读取或记录任何密钥。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SystemExit(f"无法计算文件 SHA-256：{path}") from exc
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
