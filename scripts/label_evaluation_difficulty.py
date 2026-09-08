#!/usr/bin/env python3
"""用冻结的 难度模型给 Final-200 添加难度标签。

脚本只输出 task_id、难度标签和预测成功率。为了避免把 Final-200 的
私有需求或目标商品写入公开产物，检索证据只在内存中存在。
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.sft_difficulty import (  # noqa: E402
    LogisticDifficultyModel,
)
from shopping_grpo.harness.sft_shopsim_difficulty import (  # noqa: E402
    evaluate_instruction_retrieval,
)

# 正式 Train 难度特征实际使用了这里修复后的字典形规格计数器。
from sft_task_planner import build_task_features  # noqa: E402


SCHEMA_VERSION = "wlx-evaluation-difficulty-v1"
DEFAULT_EVALUATION = REPOSITORY / "data/evaluation/tasks.jsonl"
DEFAULT_PRODUCT_FILE = (
    REPOSITORY
    / "environments/ShopSimulator/shop_env/data/items_eval_train.json"
)
DEFAULT_SEARCH_INDEX = (
    REPOSITORY
    / "environments/ShopSimulator/shop_env/search_engine/products.sqlite3"
)
DEFAULT_MODEL = (
    REPOSITORY
    / "outputs/data-v1/sft/provenance/difficulty-model.json"
)
DEFAULT_STATS = (
    REPOSITORY
    / "outputs/data-v1/raw/calibration-200/provenance"
    / "difficulty-stats.json"
)
DEFAULT_FIT_REPORT = (
    REPOSITORY
    / "outputs/data-v1/sft/provenance/difficulty-report.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY / "outputs/evaluation-difficulty-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-tasks", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--product-file", type=Path, default=DEFAULT_PRODUCT_FILE)
    parser.add_argument("--search-index", type=Path, default=DEFAULT_SEARCH_INDEX)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--difficulty-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--fit-report", type=Path, default=DEFAULT_FIT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def iter_json_array(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """流式读取顶层 JSON 数组，避免一次加载完整商品库。"""

    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    ended = False
    with Path(path).open(encoding="utf-8") as stream:
        while not ended:
            buffer = buffer.lstrip()
            if not buffer:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                buffer = chunk
                continue
            if not started:
                if not buffer.startswith("["):
                    raise ValueError(f"商品文件不是顶层 JSON 数组：{path}")
                buffer = buffer[1:]
                started = True
                continue
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                ended = True
                buffer = buffer[1:]
                continue
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError(f"商品 JSON 在数组元素中途结束：{path}")
                buffer += chunk
                continue
            yield value
            buffer = buffer[end:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
            elif buffer.startswith("]"):
                ended = True
                buffer = buffer[1:]
            elif not buffer:
                continue
            else:
                raise ValueError(f"商品 JSON 数组元素之间缺少逗号：{path}")
    if not started or not ended:
        raise ValueError(f"商品 JSON 数组不完整：{path}")


def _evaluation_task_ids(path: Path) -> list[int]:
    task_ids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id") if isinstance(row, Mapping) else None
        if isinstance(task_id, bool) or not isinstance(task_id, int):
            raise ValueError(f"{path}:{line_number}: task_id 必须是整数")
        task_ids.append(task_id)
    if len(task_ids) != 200 or len(set(task_ids)) != 200:
        raise ValueError("Final-200 必须恰好包含 200 个不同 task_id")
    return task_ids


def _goal_from_item(
    item: Mapping[str, Any],
    instruction: Mapping[str, Any],
    *,
    compile_reward_features: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    explicit_budget_from_instruction: Callable[[str], float | None],
) -> dict[str, Any]:
    persona = item.get("user_persona")
    persona = dict(persona) if isinstance(persona, Mapping) else {}
    if "__reasoning__" in persona:
        reasoning = persona.pop("__reasoning__")
        persona = {"__reasoning__": reasoning, **persona}
    goal = {
        "asin": item["asin"],
        "category": item["category"],
        "query": str(item.get("query") or "").lower().strip(),
        "name": item["title"],
        "instruction_text": instruction["instruction"],
        "instruction_simple": instruction.get("instruction_simple"),
        "attributes": instruction.get("attributes") or [],
        "price_upper": explicit_budget_from_instruction(instruction["instruction"]),
        "goal_options": instruction.get("instruction_options") or [],
        "user_persona": persona,
        "reason_key": item.get("reason_key"),
        "weight": 1,
    }
    goal.update(compile_reward_features(dict(instruction), dict(item)))
    return goal


def _extract_evaluation_goals(
    product_path: Path,
    task_ids: Sequence[int],
    *,
    compile_reward_features: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    explicit_budget_from_instruction: Callable[[str], float | None],
) -> tuple[dict[int, dict[str, Any]], int]:
    wanted = set(task_ids)
    goals: dict[int, dict[str, Any]] = {}
    seen_asins: set[str] = set()
    goal_index = 0
    for raw in iter_json_array(product_path):
        if not isinstance(raw, Mapping):
            continue
        asin = str(raw.get("asin") or "")
        if asin == "nan" or len(asin) > 20 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        instructions = raw.get("instructions") or []
        if isinstance(instructions, (str, bytes)) or not isinstance(
            instructions, Sequence
        ):
            continue
        for instruction in instructions:
            if not isinstance(instruction, Mapping):
                continue
            if not instruction.get("attributes"):
                continue
            if goal_index in wanted:
                goals[goal_index] = _goal_from_item(
                    raw,
                    instruction,
                    compile_reward_features=compile_reward_features,
                    explicit_budget_from_instruction=explicit_budget_from_instruction,
                )
            goal_index += 1
    missing = sorted(wanted.difference(goals))
    if missing:
        raise ValueError(f"无法从 canonical goals 找到 Evaluation task_id：{missing[:10]}")
    return goals, goal_index


def _minimal_reward_product(item: Mapping[str, Any]) -> dict[str, Any]:
    """仅保留 Reward v3 比较器需要的字段，避免图片等内容占用内存。"""

    pricing = item.get("pricing") or [100.0]
    return {
        "asin": str(item.get("asin") or ""),
        "title": item.get("title"),
        "Title": item.get("title"),
        "shop_name": item.get("shop_name"),
        "category": item.get("category"),
        "attribute": item.get("attribute") or [],
        "Attributes": item.get("attribute") or [],
        "small_description": item.get("small_description") or "",
        "BulletPoints": (
            item.get("small_description")
            if isinstance(item.get("small_description"), list)
            else [item.get("small_description") or ""]
        ),
        "full_description": item.get("full_description") or "",
        "Description": item.get("full_description") or "",
        "customization_options": item.get("customization_options") or {},
        "variant_combinations": item.get("variant_combinations") or [],
        "pricing": pricing,
    }


def _candidate_products(
    product_path: Path,
    wanted_asins: set[str],
) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    seen_asins: set[str] = set()
    for raw in iter_json_array(product_path):
        if not isinstance(raw, Mapping):
            continue
        asin = str(raw.get("asin") or "")
        if asin == "nan" or len(asin) > 20 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        if asin in wanted_asins:
            products[asin] = _minimal_reward_product(raw)
    missing = wanted_asins.difference(products)
    if missing:
        raise ValueError(f"BM25 候选不在冻结商品库中：{sorted(missing)[:10]}")
    return products


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"需要 JSON 对象：{path}")
    return dict(value)


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"拒绝覆盖已有文件：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def label_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    shop_root = REPOSITORY / "environments/ShopSimulator/shop_env"
    if str(shop_root) not in sys.path:
        sys.path.insert(0, str(shop_root))
    try:
        from web_agent_site.engine.constraints import explicit_budget_from_instruction
        from web_agent_site.engine.reward_features import compile_reward_features
        from web_agent_site.engine.search import MultiFieldBM25Searcher
    except ImportError as exc:
        raise RuntimeError("请使用 ShopSimulator 的 Python 环境运行本脚本") from exc

    evaluation_sha = _sha256(args.evaluation_tasks)
    product_sha = _sha256(args.product_file)
    stats = _read_object(args.difficulty_stats)
    fit_report = _read_object(args.fit_report)
    expected_evaluation_sha = fit_report["fit_provenance"]["semantic_contract"][
        "held_out_tasks_sha256"
    ]
    if evaluation_sha != expected_evaluation_sha or evaluation_sha != stats.get(
        "evaluation_sha256"
    ):
        raise ValueError("Evaluation 文件与冻结难度模型的 held-out 来源不一致")
    if product_sha != stats.get("product_sha256"):
        raise ValueError("商品文件与冻结难度统计不一致")

    task_ids = _evaluation_task_ids(args.evaluation_tasks)
    goals, canonical_goal_count = _extract_evaluation_goals(
        args.product_file,
        task_ids,
        compile_reward_features=compile_reward_features,
        explicit_budget_from_instruction=explicit_budget_from_instruction,
    )
    searcher = MultiFieldBM25Searcher(
        args.search_index,
        expected_product_sha256=product_sha,
    )
    try:
        hits_by_task = {
            task_id: searcher.search(goals[task_id]["instruction_text"], k=20)
            for task_id in task_ids
        }
        wanted_asins = {
            str(hit.asin)
            for hits in hits_by_task.values()
            for hit in hits
        }
        products = _candidate_products(args.product_file, wanted_asins)

        class FrozenHitSearcher:
            manifest = searcher.manifest

            def search(self, query: object, k: int = 20) -> list[Any]:
                del query
                if k != 20:
                    raise ValueError("难度定义固定使用 BM25 前 20 名")
                return list(self.hits)

        frozen_searcher = FrozenHitSearcher()
        model = LogisticDifficultyModel.from_dict(_read_object(args.model))
        output_rows = []
        unreliable = 0
        for task_id in task_ids:
            frozen_searcher.hits = hits_by_task[task_id]
            evidence = evaluate_instruction_retrieval(
                goal=goals[task_id],
                product_item_dict=products,
                searcher=frozen_searcher,
            )
            unreliable += int(not evidence.candidate_evaluation_reliable)
            features = build_task_features(
                task_id=task_id,
                goal=goals[task_id],
                retrieval=evidence,
                a95=float(stats["a95"]),
                n95=float(stats["n95"]),
                category=str(goals[task_id].get("category") or "") or None,
            )
            label, predicted_success, difficulty_score = model.classify(features)
            output_rows.append(
                {
                    "task_id": task_id,
                    "difficulty_label": label,
                    "difficulty_score": difficulty_score,
                    "predicted_teacher_success": predicted_success,
                    "difficulty_version": model.difficulty_version,
                }
            )
    finally:
        searcher.close()

    counts = Counter(row["difficulty_label"] for row in output_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_count": len(output_rows),
        "canonical_goal_count": canonical_goal_count,
        "difficulty_counts": {
            label: counts.get(label, 0) for label in ("easy", "medium", "hard")
        },
        "thresholds": {
            "easy": "predicted_success >= 0.75",
            "medium": "0.35 <= predicted_success < 0.75",
            "hard": "predicted_success < 0.35",
        },
        "interpretation": (
            "predicted DeepSeek V4 Flash success under the frozen SFT "
            "harness; not an observed Base/SFT evaluation result"
        ),
        "difficulty_version": model.difficulty_version,
        "a95": stats["a95"],
        "n95": stats["n95"],
        "candidate_evaluation_unreliable_tasks": unreliable,
        "provenance": {
            "evaluation_tasks_sha256": evaluation_sha,
            "product_data_sha256": product_sha,
            "search_index_sha256": _sha256(args.search_index),
            "difficulty_model_sha256": _sha256(args.model),
            "difficulty_stats_sha256": _sha256(args.difficulty_stats),
        },
    }
    labels_path = args.output_dir / "evaluation-difficulty.jsonl"
    summary_path = args.output_dir / "evaluation-difficulty-summary.json"
    _atomic_write(
        labels_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in output_rows
        ),
    )
    _atomic_write(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = label_evaluation(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
