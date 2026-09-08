"""复用 ShopSimulator 的 BM25 和 Reward-v3 离线计算检索难度证据。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shopping_grpo.harness.sft_difficulty import RetrievalEvidence
from shopping_grpo.harness.sft_storage import append_jsonl_row, read_jsonl


def evaluate_instruction_retrieval(
    *,
    goal: Mapping[str, Any],
    product_item_dict: Mapping[str, Mapping[str, Any]],
    searcher: object,
    top_k: int = 20,
) -> RetrievalEvidence:
    """用完整 instruction 搜首屏，并用环境同一 Reward 判断合格与近似商品。"""

    if top_k != 20:
        raise ValueError("第一版难度定义固定使用前 20 名")
    try:
        from web_agent_site.engine.reward import evaluate_purchase
        from web_agent_site.engine.variant_price import candidate_options_for_evaluation
    except ImportError as exc:
        raise RuntimeError(
            "请在 ShopSimulator 的 Python 环境中运行离线难度预计算"
        ) from exc
    search = getattr(searcher, "search", None)
    if not callable(search):
        raise TypeError("searcher 必须提供 search(query, k)")
    hits = search(goal.get("instruction_text") or "", k=top_k)
    best_rank: int | None = None
    best_type: str | None = None
    target_rank: int | None = None
    near_misses: list[dict[str, Any]] = []
    unverifiable = 0
    reliable = not bool(goal.get("unresolved_option_requirements"))
    for position, hit in enumerate(hits, start=1):
        asin = str(getattr(hit, "asin", "") or "")
        rank = int(getattr(hit, "rank", position) or position)
        if asin == str(goal.get("asin")):
            target_rank = rank
        product = product_item_dict.get(asin)
        if not isinstance(product, Mapping):
            continue
        selected, _ = candidate_options_for_evaluation(
            dict(product),
            goal.get("required_options_by_key"),
        )
        result = evaluate_purchase(
            dict(product),
            dict(goal),
            selected_options=selected,
        )
        reward_type = str(result.reward_type)
        if not result.reward_valid or reward_type == "reward_unverifiable":
            unverifiable += 1
            continue
        if reward_type in {"gold_purchase", "valid_alternative_purchase"}:
            if best_rank is None:
                best_rank = rank
                best_type = reward_type
            continue
        if asin == str(goal.get("asin")):
            continue
        near_miss = _near_miss_evidence(
            asin=asin,
            rank=rank,
            result=result,
            has_budget=goal.get("price_upper") is not None,
        )
        if near_miss is not None:
            near_misses.append(near_miss)
    return RetrievalEvidence(
        best_valid_rank_at_20=best_rank,
        target_asin_rank_at_20=target_rank,
        best_valid_outcome_type=best_type,
        near_miss_count_at_20=len(near_misses),
        unverifiable_candidates_at_20=unverifiable,
        candidate_evaluation_reliable=reliable,
        near_miss_evidence=tuple(near_misses),
    )


def precompute_retrieval_cache(
    *,
    goals: Sequence[Mapping[str, Any]],
    product_item_dict: Mapping[str, Mapping[str, Any]],
    searcher: object,
    output_path: str | Path,
    task_ids: set[int] | None = None,
) -> dict[str, int]:
    """按完整 canonical 下标缓存指定 Train 任务的 R/N，并支持按 task_id 断点续算。"""

    output_path = Path(output_path)
    expected_versions = _difficulty_versions(searcher, goals[0]) if goals else {}
    completed: set[int] = set()
    for row in read_jsonl(output_path):
        if row.get("versions") != expected_versions:
            raise ValueError("已有 R/N 缓存版本不同，请换一个新的输出目录重新计算")
        if row.get("task_id") is not None:
            completed.add(int(row["task_id"]))
    written = 0
    for task_id, goal in enumerate(goals):
        if task_ids is not None and task_id not in task_ids:
            continue
        if task_id in completed:
            continue
        evidence = evaluate_instruction_retrieval(
            goal=goal,
            product_item_dict=product_item_dict,
            searcher=searcher,
        )
        append_jsonl_row(
            output_path,
            {
                "task_id": task_id,
                "instruction_bm25_best_valid_rank_at_20": evidence.best_valid_rank_at_20,
                "instruction_bm25_target_asin_rank_at_20": evidence.target_asin_rank_at_20,
                "instruction_bm25_best_valid_outcome_type": evidence.best_valid_outcome_type,
                "near_miss_count_at_20": evidence.near_miss_count_at_20,
                "unverifiable_candidates_at_20": evidence.unverifiable_candidates_at_20,
                "candidate_evaluation_reliable": evidence.candidate_evaluation_reliable,
                "near_miss_evidence": [dict(item) for item in evidence.near_miss_evidence],
                "versions": _difficulty_versions(searcher, goal),
            },
            durable=False,
        )
        written += 1
    planned = len(goals) if task_ids is None else len(task_ids)
    return {"goals": planned, "already_completed": len(completed), "new_rows": written}


def retrieval_evidence_from_cache(row: Mapping[str, Any]) -> RetrievalEvidence:
    """把离线 JSONL 缓存的一行恢复成通用 RetrievalEvidence。"""

    return RetrievalEvidence(
        best_valid_rank_at_20=_optional_int(
            row.get("instruction_bm25_best_valid_rank_at_20")
        ),
        target_asin_rank_at_20=_optional_int(
            row.get("instruction_bm25_target_asin_rank_at_20")
        ),
        best_valid_outcome_type=(
            str(row["instruction_bm25_best_valid_outcome_type"])
            if row.get("instruction_bm25_best_valid_outcome_type")
            else None
        ),
        near_miss_count_at_20=int(row.get("near_miss_count_at_20", 0)),
        unverifiable_candidates_at_20=int(
            row.get("unverifiable_candidates_at_20", 0)
        ),
        candidate_evaluation_reliable=bool(
            row.get("candidate_evaluation_reliable", True)
        ),
        near_miss_evidence=tuple(row.get("near_miss_evidence") or ()),
    )


def _near_miss_evidence(
    *,
    asin: str,
    rank: int,
    result: object,
    has_budget: bool,
) -> dict[str, Any] | None:
    """按独立要求通过率判断近似干扰商品，不拿 Reward 加权分数冒充 70%。"""

    hard_gates = getattr(result, "hard_gates", {}) or {}
    category = hard_gates.get("category") or {}
    if category.get("status") != "pass":
        return None
    evidence = getattr(result, "evidence", {}) or {}
    preferences = evidence.get("preference_scoring") or {}
    dimensions = preferences.get("dimensions") or {}
    required = sum(int(item.get("required_count", 0)) for item in dimensions.values())
    passed = sum(int(item.get("passed_count", 0)) for item in dimensions.values())
    verifiable = sum(
        int(item.get("verifiable_count", 0)) for item in dimensions.values()
    )
    if has_budget:
        budget = hard_gates.get("budget") or {}
        required += 1
        passed += int(budget.get("status") == "pass")
        verifiable += int(budget.get("status") != "unverifiable")
    match = passed / required if required else 1.0
    coverage = verifiable / required if required else 1.0
    if match < 0.70 or coverage < 0.75:
        return None
    return {
        "asin": asin,
        "rank": int(rank),
        "reward_type": str(getattr(result, "reward_type", "")),
        "flat_match": match,
        "flat_coverage": coverage,
        "passed_requirements": passed,
        "verifiable_requirements": verifiable,
        "total_requirements": required,
    }


def _difficulty_versions(searcher: object, goal: Mapping[str, Any]) -> dict[str, Any]:
    """保存商品库、搜索和 Reward 特征版本，任一变化都说明 R/N 需要重算。"""

    try:
        from web_agent_site.engine.comparators import COMPARATOR_VERSION
        from web_agent_site.engine.reward import REWARD_VERSION
        from web_agent_site.engine.reward_features import (
            OPTION_AXIS_VERSION,
            REWARD_FEATURE_VERSION,
        )
        from web_agent_site.engine.variant_price import VARIANT_PRICE_VERSION
    except ImportError as exc:
        raise RuntimeError("无法读取 ShopSimulator 难度依赖版本") from exc
    manifest = getattr(searcher, "manifest", {})
    manifest = dict(manifest) if isinstance(manifest, Mapping) else {}
    return {
        "difficulty_feature_code_version": "wlx-static-difficulty-v1",
        "product_data_sha256": manifest.get("product_data_sha256"),
        "search_version": manifest.get("search_version"),
        "index_schema_version": manifest.get("index_schema_version"),
        "reward_version": REWARD_VERSION,
        "reward_feature_version": goal.get("reward_feature_version")
        or REWARD_FEATURE_VERSION,
        "comparator_version": COMPARATOR_VERSION,
        "variant_price_version": VARIANT_PRICE_VERSION,
        "option_axis_version": OPTION_AXIS_VERSION,
    }


def _optional_int(value: object) -> int | None:
    """把缓存中的可选排名转成整数，空值继续保持为空。"""

    return int(value) if value is not None else None


__all__ = [
    "evaluate_instruction_retrieval",
    "precompute_retrieval_cache",
    "retrieval_evidence_from_cache",
]
