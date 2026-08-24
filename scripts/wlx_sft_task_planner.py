#!/usr/bin/env python3
"""Build deterministic, public-only WLX calibration and formal sampling plans.

This sidecar keeps the existing pipeline untouched.  It consumes the canonical
public task table and retrieval cache, fixes dictionary-shaped unresolved option
requirements, selects a deterministic 30/50/20 calibration sample, and can select
the first category-rich 30/40/30 formal batch without calling a model API or the
ShopSimulator HTTP service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from wlx_harness_core.wlx_sft_contracts import SftTask  # noqa: E402
from wlx_harness_core.wlx_sft_difficulty import (  # noqa: E402
    LogisticDifficultyModel,
    TaskDifficultyFeatures,
    assign_preliminary_labels,
    constraint_score,
    near_miss_score,
    percentile_95,
    retrieval_score,
    select_calibration_tasks,
)
from wlx_harness_core.wlx_sft_shopsim_difficulty import (  # noqa: E402
    retrieval_evidence_from_cache,
)
from wlx_harness_core.wlx_sft_storage import (  # noqa: E402
    load_held_out_task_ids,
    load_sft_tasks,
    read_jsonl,
)
from wlx_harness_core.wlx_sft_tasks import (  # noqa: E402
    apply_difficulty_model,
    public_tasks_from_canonical_goals,
)


LABELS = ("easy", "medium", "hard")
LABEL_WEIGHTS = {"easy": 0.30, "medium": 0.50, "hard": 0.20}
CONSTRAINT_COUNT_VERSION = "wlx-unresolved-option-value-v1"
PLAN_VERSION = "wlx-preliminary-rule-v2"
SELECTOR_VERSION = "wlx-stratified-category-mincost-v1"
FORMAL_BATCH_SELECTOR_VERSION = "wlx-formal-batch-category-mincost-v1"
FORMAL_BATCH_QUOTAS = {"easy": 30, "medium": 40, "hard": 30}
PUBLIC_TASK_FIELDS = {
    "task_id",
    "instruction",
    "official_split",
    "category",
    "difficulty_label",
    "difficulty_score",
    "difficulty_version",
}


def _normalise_option_value(value: object) -> str:
    """Match ShopSimulator's option-value identity without importing its runtime."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("/", "|")
    return re.sub(r"\s+", "", text)


def independent_constraint_count(
    goal: Mapping[str, Any],
    *,
    task_id: int | None = None,
) -> tuple[int, dict[str, int | bool]]:
    """Count independent requirements while accepting canonical unresolved dicts."""

    context = f"task_id={task_id}: " if task_id is not None else ""
    core = goal.get("expected_core_functions") or ()
    if isinstance(core, (str, bytes)) or not isinstance(core, Sequence):
        raise ValueError(context + "expected_core_functions must be a sequence")
    core_values = {str(item).strip() for item in core if str(item).strip()}

    required = goal.get("required_options_by_key") or {}
    if not isinstance(required, Mapping):
        raise ValueError(context + "required_options_by_key must be a mapping")
    resolved_axes = {str(key).strip().casefold() for key in required if str(key).strip()}

    unresolved = goal.get("unresolved_option_requirements") or ()
    if isinstance(unresolved, (str, bytes)) or not isinstance(unresolved, Sequence):
        raise ValueError(context + "unresolved_option_requirements must be a sequence")
    unresolved_values: set[str] = set()
    for item in unresolved:
        if not isinstance(item, Mapping):
            raise ValueError(context + "every unresolved option requirement must be an object")
        value = _normalise_option_value(item.get("value"))
        if not value:
            raise ValueError(context + "unresolved option requirement has an empty value")
        unresolved_values.add(value)

    has_brand = bool(goal.get("expected_brand"))
    has_model = bool(goal.get("expected_model"))
    has_budget = goal.get("price_upper") is not None or goal.get("max_price") is not None
    total = (
        len(core_values)
        + len(resolved_axes)
        + len(unresolved_values)
        + int(has_brand)
        + int(has_model)
        + int(has_budget)
    )
    return total, {
        "core_function_count": len(core_values),
        "option_axis_count": len(resolved_axes),
        "unresolved_option_count": len(unresolved_values),
        "has_brand": has_brand,
        "has_model": has_model,
        "has_budget": has_budget,
    }


def build_task_features(
    *,
    task_id: int,
    goal: Mapping[str, Any],
    retrieval: object,
    a95: float,
    n95: float,
    category: str | None,
) -> TaskDifficultyFeatures:
    """Build the existing feature contract with the corrected constraint counter."""

    count, parts = independent_constraint_count(goal, task_id=task_id)
    c_score = constraint_score(count, a95)
    r_score = retrieval_score(retrieval.best_valid_rank_at_20)
    n_score = near_miss_score(retrieval.near_miss_count_at_20, n95)
    preliminary = 0.45 * c_score + 0.35 * r_score + 0.20 * n_score
    return TaskDifficultyFeatures(
        task_id=int(task_id),
        constraint_count=count,
        option_axis_count=int(parts["option_axis_count"]),
        has_brand=bool(parts["has_brand"]),
        has_model=bool(parts["has_model"]),
        has_budget=bool(parts["has_budget"]),
        retrieval_score=r_score,
        near_miss_score=n_score,
        constraint_score=c_score,
        preliminary_score=preliminary,
        category=category,
        evidence={
            **parts,
            "constraint_count_version": CONSTRAINT_COUNT_VERSION,
            "best_valid_rank_at_20": retrieval.best_valid_rank_at_20,
            "target_asin_rank_at_20": retrieval.target_asin_rank_at_20,
            "best_valid_outcome_type": retrieval.best_valid_outcome_type,
            "near_miss_count_at_20": retrieval.near_miss_count_at_20,
            "unverifiable_candidates_at_20": retrieval.unverifiable_candidates_at_20,
            "candidate_evaluation_reliable": retrieval.candidate_evaluation_reliable,
            "near_miss_evidence": [dict(item) for item in retrieval.near_miss_evidence],
        },
    )


def difficulty_quotas(sample_size: int) -> dict[str, int]:
    """Allocate the documented 30/50/20 mix with deterministic largest remainder."""

    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    raw = {label: sample_size * LABEL_WEIGHTS[label] for label in LABELS}
    quotas = {label: math.floor(raw[label]) for label in LABELS}
    remaining = sample_size - sum(quotas.values())
    order = sorted(
        LABELS,
        key=lambda label: (-(raw[label] - quotas[label]), LABELS.index(label)),
    )
    for label in order[:remaining]:
        quotas[label] += 1
    return quotas


def _stable_digest(seed: int, namespace: str, value: object) -> str:
    payload = f"{seed}:{namespace}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _top_category(category: str | None) -> str:
    value = str(category or "unknown").strip() or "unknown"
    return value.split("›", 1)[0].strip() or "unknown"


@dataclass
class _FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: int


class _MinCostFlow:
    """Small deterministic integer min-cost flow used only for task stratification."""

    def __init__(self, nodes: int) -> None:
        self.graph: list[list[_FlowEdge]] = [[] for _ in range(nodes)]

    def add_edge(self, source: int, target: int, capacity: int, cost: int) -> _FlowEdge:
        forward = _FlowEdge(target, len(self.graph[target]), int(capacity), int(cost))
        reverse = _FlowEdge(source, len(self.graph[source]), 0, -int(cost))
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def send(self, source: int, sink: int, target_flow: int) -> tuple[int, int]:
        flow = 0
        total_cost = 0
        nodes = len(self.graph)
        infinity = 10**18
        while flow < target_flow:
            distance = [infinity] * nodes
            previous_node = [-1] * nodes
            previous_edge = [-1] * nodes
            distance[source] = 0
            for _ in range(nodes - 1):
                changed = False
                for node, edges in enumerate(self.graph):
                    if distance[node] == infinity:
                        continue
                    for edge_index, edge in enumerate(edges):
                        candidate = distance[node] + edge.cost
                        if edge.capacity > 0 and candidate < distance[edge.to]:
                            distance[edge.to] = candidate
                            previous_node[edge.to] = node
                            previous_edge[edge.to] = edge_index
                            changed = True
                if not changed:
                    break
            if distance[sink] == infinity:
                break
            amount = target_flow - flow
            node = sink
            while node != source:
                parent = previous_node[node]
                if parent < 0:
                    raise RuntimeError("broken min-cost-flow path")
                edge = self.graph[parent][previous_edge[node]]
                amount = min(amount, edge.capacity)
                node = parent
            node = sink
            while node != source:
                parent = previous_node[node]
                edge = self.graph[parent][previous_edge[node]]
                edge.capacity -= amount
                self.graph[node][edge.reverse].capacity += amount
                node = parent
            flow += amount
            total_cost += amount * distance[sink]
        return flow, total_cost


def _interleave_by_target_ratio(
    buckets: Mapping[str, Sequence[SftTask]],
) -> list[SftTask]:
    """Order selected tasks so every prefix stays near the 30/50/20 target."""

    queues = {label: list(buckets.get(label, ())) for label in LABELS}
    chosen = {label: 0 for label in LABELS}
    result: list[SftTask] = []
    total = sum(len(items) for items in queues.values())
    for position in range(1, total + 1):
        available = [label for label in LABELS if queues[label]]
        label = max(
            available,
            key=lambda item: (
                LABEL_WEIGHTS[item] * position - chosen[item],
                -LABELS.index(item),
            ),
        )
        result.append(queues[label].pop(0))
        chosen[label] += 1
    return result


def _interleave_by_quotas(
    buckets: Mapping[str, Sequence[SftTask]],
    quotas: Mapping[str, int],
) -> list[SftTask]:
    """Order a fixed-quota batch so early prefixes remain close to its final mix."""

    total = sum(int(quotas[label]) for label in LABELS)
    if total < 1:
        raise ValueError("formal batch quotas must select at least one task")
    queues = {label: list(buckets.get(label, ())) for label in LABELS}
    chosen = {label: 0 for label in LABELS}
    result: list[SftTask] = []
    weights = {label: int(quotas[label]) / total for label in LABELS}
    for position in range(1, total + 1):
        available = [label for label in LABELS if queues[label]]
        if not available:
            raise AssertionError("formal batch queues ended before quotas were filled")
        label = max(
            available,
            key=lambda item: (
                weights[item] * position - chosen[item],
                -LABELS.index(item),
            ),
        )
        result.append(queues[label].pop(0))
        chosen[label] += 1
    return result


def balanced_calibration_task_plan(
    tasks: Sequence[SftTask],
    features: Sequence[TaskDifficultyFeatures],
    *,
    sample_size: int = 200,
    seed: int = 42,
) -> tuple[list[SftTask], dict[str, Any]]:
    """Select exact difficulty quotas while maximizing hierarchical category coverage."""

    by_task = {task.task_id: task for task in tasks}
    if len(by_task) != len(tasks):
        raise ValueError("public task table contains duplicate task_id values")
    feature_by_task = {item.task_id: item for item in features}
    if len(feature_by_task) != len(features):
        raise ValueError("difficulty feature table contains duplicate task_id values")
    if set(by_task) != set(feature_by_task):
        raise ValueError("public tasks and difficulty features have different task_id sets")

    quotas = difficulty_quotas(sample_size)
    pool_counts = Counter(str(item.preliminary_label) for item in features)
    for label in LABELS:
        if pool_counts[label] < quotas[label]:
            raise ValueError(
                f"difficulty pool {label!r} has {pool_counts[label]} tasks, "
                f"but quota requires {quotas[label]}"
            )
    unexpected = set(pool_counts).difference(LABELS)
    if unexpected:
        raise ValueError(f"unexpected preliminary labels: {sorted(unexpected)}")

    cells: dict[tuple[str, str], list[TaskDifficultyFeatures]] = defaultdict(list)
    for item in features:
        label = str(item.preliminary_label)
        category = str(item.category or "unknown")
        cells[(label, category)].append(item)
    for (label, category), items in cells.items():
        items.sort(key=lambda item: _stable_digest(seed, f"task:{label}:{category}", item.task_id))

    full_categories = sorted(
        {category for _, category in cells},
        key=lambda value: _stable_digest(seed, "full-category", value),
    )
    top_categories = sorted(
        {_top_category(category) for category in full_categories},
        key=lambda value: _stable_digest(seed, "top-category", value),
    )

    source = 0
    next_node = 1
    label_nodes = {label: next_node + index for index, label in enumerate(LABELS)}
    next_node += len(label_nodes)
    full_nodes = {
        category: next_node + index for index, category in enumerate(full_categories)
    }
    next_node += len(full_nodes)
    top_nodes = {category: next_node + index for index, category in enumerate(top_categories)}
    next_node += len(top_nodes)
    sink = next_node
    flow_network = _MinCostFlow(sink + 1)

    for label in LABELS:
        flow_network.add_edge(source, label_nodes[label], quotas[label], 0)
    cell_edges: dict[tuple[str, str], _FlowEdge] = {}
    ordered_cells = sorted(
        cells,
        key=lambda item: _stable_digest(seed, "cell", f"{item[0]}:{item[1]}"),
    )
    for label, category in ordered_cells:
        cell_edges[(label, category)] = flow_network.add_edge(
            label_nodes[label], full_nodes[category], 1, 0
        )
    for category in full_categories:
        flow_network.add_edge(full_nodes[category], top_nodes[_top_category(category)], 1, 0)
    for category in top_categories:
        flow_network.add_edge(top_nodes[category], sink, 1, -1)
        flow_network.add_edge(top_nodes[category], sink, sample_size, 0)

    flow, _ = flow_network.send(source, sink, sample_size)
    if flow != sample_size:
        raise ValueError(
            "not enough distinct full categories to satisfy the requested difficulty quotas"
        )

    selected_features = [
        cells[cell][0]
        for cell, edge in cell_edges.items()
        if edge.capacity == 0
    ]
    if len(selected_features) != sample_size:
        raise AssertionError("min-cost flow selected an unexpected number of task cells")

    buckets: dict[str, list[SftTask]] = {label: [] for label in LABELS}
    for feature in selected_features:
        task = by_task[feature.task_id]
        label = str(feature.preliminary_label)
        buckets[label].append(
            SftTask(
                task_id=task.task_id,
                instruction=task.instruction,
                official_split=task.official_split,
                category=task.category,
                difficulty_label=label,
                difficulty_score=feature.preliminary_score,
                difficulty_version=PLAN_VERSION,
            )
        )
    for label in LABELS:
        buckets[label].sort(
            key=lambda task: _stable_digest(seed, f"plan-task:{label}", task.task_id)
        )
        if len(buckets[label]) != quotas[label]:
            raise AssertionError(f"selected {len(buckets[label])} {label} tasks")

    plan = _interleave_by_target_ratio(buckets)
    selected_categories = {str(task.category or "unknown") for task in plan}
    selected_top = {_top_category(task.category) for task in plan}
    expected_top = min(len(top_categories), sample_size)
    if len(selected_top) != expected_top:
        raise ValueError(
            f"category flow covered {len(selected_top)} top categories; expected {expected_top}"
        )
    prefix_counts = {
        str(prefix): dict(Counter(task.difficulty_label for task in plan[:prefix]))
        for prefix in (10, 20, 50, 100, sample_size)
        if prefix <= sample_size
    }
    report = {
        "selector_version": SELECTOR_VERSION,
        "difficulty_version": PLAN_VERSION,
        "seed": int(seed),
        "sample_size": int(sample_size),
        "quotas": quotas,
        "pool_counts": {label: pool_counts[label] for label in LABELS},
        "selected_counts": dict(Counter(task.difficulty_label for task in plan)),
        "pool_full_categories": len(full_categories),
        "selected_full_categories": len(selected_categories),
        "pool_top_categories": len(top_categories),
        "selected_top_categories": len(selected_top),
        "prefix_counts": prefix_counts,
    }
    return plan, report


def balanced_formal_batch_plan(
    candidates: Sequence[SftTask],
    *,
    calibration_task_ids: set[int],
    held_out_task_ids: set[int],
    prior_formal_task_ids: set[int] | None = None,
    quotas: Mapping[str, int] = FORMAL_BATCH_QUOTAS,
    seed: int = 42,
) -> tuple[list[SftTask], dict[str, Any]]:
    """Select an exact final-difficulty mix while maximizing category coverage."""

    normalized_quotas = {label: int(quotas.get(label, 0)) for label in LABELS}
    if set(quotas) != set(LABELS) or any(value < 0 for value in normalized_quotas.values()):
        raise ValueError("formal quotas must contain non-negative easy/medium/hard counts")
    sample_size = sum(normalized_quotas.values())
    if sample_size < 1:
        raise ValueError("formal batch must contain at least one task")

    by_task = {task.task_id: task for task in candidates}
    if len(by_task) != len(candidates):
        raise ValueError("formal candidate plan contains duplicate task_id values")
    versions = {task.difficulty_version for task in candidates}
    if None in versions or len(versions) != 1:
        raise ValueError("formal candidates must share one non-empty difficulty_version")
    for task in candidates:
        if task.official_split != "train":
            raise ValueError(f"task_id={task.task_id} is not in the Train split")
        if not task.instruction:
            raise ValueError(f"task_id={task.task_id} has no public instruction")
        if task.difficulty_label not in LABELS or task.difficulty_score is None:
            raise ValueError(f"task_id={task.task_id} lacks final difficulty metadata")

    calibration_ids = {int(task_id) for task_id in calibration_task_ids}
    held_out_ids = {int(task_id) for task_id in held_out_task_ids}
    prior_ids = {int(task_id) for task_id in (prior_formal_task_ids or set())}
    excluded_ids = calibration_ids | held_out_ids | prior_ids
    eligible = [task for task in candidates if task.task_id not in excluded_ids]
    pool_counts = Counter(str(task.difficulty_label) for task in candidates)
    eligible_counts = Counter(str(task.difficulty_label) for task in eligible)
    for label in LABELS:
        if eligible_counts[label] < normalized_quotas[label]:
            raise ValueError(
                f"eligible formal pool {label!r} has {eligible_counts[label]} tasks, "
                f"but quota requires {normalized_quotas[label]}"
            )

    cells: dict[tuple[str, str], list[SftTask]] = defaultdict(list)
    for task in eligible:
        cells[(str(task.difficulty_label), str(task.category or "unknown"))].append(task)
    for (label, category), items in cells.items():
        items.sort(
            key=lambda task: _stable_digest(
                seed,
                f"formal-task:{label}:{category}",
                task.task_id,
            )
        )

    full_categories = sorted(
        {category for _, category in cells},
        key=lambda value: _stable_digest(seed, "formal-full-category", value),
    )
    top_categories = sorted(
        {_top_category(category) for category in full_categories},
        key=lambda value: _stable_digest(seed, "formal-top-category", value),
    )
    source = 0
    next_node = 1
    label_nodes = {label: next_node + index for index, label in enumerate(LABELS)}
    next_node += len(label_nodes)
    full_nodes = {
        category: next_node + index for index, category in enumerate(full_categories)
    }
    next_node += len(full_nodes)
    top_nodes = {category: next_node + index for index, category in enumerate(top_categories)}
    next_node += len(top_nodes)
    sink = next_node
    flow_network = _MinCostFlow(sink + 1)

    for label in LABELS:
        flow_network.add_edge(source, label_nodes[label], normalized_quotas[label], 0)
    cell_edges: dict[tuple[str, str], tuple[_FlowEdge, int]] = {}
    ordered_cells = sorted(
        cells,
        key=lambda item: _stable_digest(seed, "formal-cell", f"{item[0]}:{item[1]}"),
    )
    for label, category in ordered_cells:
        capacity = len(cells[(label, category)])
        edge = flow_network.add_edge(
            label_nodes[label],
            full_nodes[category],
            capacity,
            0,
        )
        cell_edges[(label, category)] = (edge, capacity)

    # One use of a full category is worth more than every possible top-category
    # tie-break combined.  Extra uses remain possible when a quota requires them.
    full_category_reward = sample_size + 1
    tasks_per_full = Counter(str(task.category or "unknown") for task in eligible)
    for category in full_categories:
        capacity = tasks_per_full[category]
        top_node = top_nodes[_top_category(category)]
        flow_network.add_edge(
            full_nodes[category], top_node, 1, -full_category_reward
        )
        if capacity > 1:
            flow_network.add_edge(full_nodes[category], top_node, capacity - 1, 0)
    tasks_per_top = Counter(_top_category(task.category) for task in eligible)
    for category in top_categories:
        capacity = tasks_per_top[category]
        flow_network.add_edge(top_nodes[category], sink, 1, -1)
        if capacity > 1:
            flow_network.add_edge(top_nodes[category], sink, capacity - 1, 0)

    flow, _ = flow_network.send(source, sink, sample_size)
    if flow != sample_size:
        raise ValueError("eligible formal pool cannot satisfy the requested quotas")

    buckets: dict[str, list[SftTask]] = {label: [] for label in LABELS}
    for cell, (edge, initial_capacity) in cell_edges.items():
        selected_count = initial_capacity - edge.capacity
        if selected_count < 0 or selected_count > initial_capacity:
            raise AssertionError("formal category flow returned an invalid cell count")
        for task in cells[cell][:selected_count]:
            buckets[str(task.difficulty_label)].append(
                SftTask(
                    task_id=task.task_id,
                    instruction=task.instruction,
                    official_split=task.official_split,
                    category=task.category,
                    difficulty_label=task.difficulty_label,
                    difficulty_score=task.difficulty_score,
                    difficulty_version=task.difficulty_version,
                )
            )
    for label in LABELS:
        buckets[label].sort(
            key=lambda task: _stable_digest(
                seed, f"formal-plan-task:{label}", task.task_id
            )
        )
        if len(buckets[label]) != normalized_quotas[label]:
            raise AssertionError(f"selected {len(buckets[label])} {label} formal tasks")

    plan = _interleave_by_quotas(buckets, normalized_quotas)
    plan_ids = {task.task_id for task in plan}
    if len(plan_ids) != sample_size:
        raise AssertionError("formal batch contains duplicate task IDs")
    if (
        plan_ids.intersection(calibration_ids)
        or plan_ids.intersection(held_out_ids)
        or plan_ids.intersection(prior_ids)
    ):
        raise AssertionError("formal batch contains excluded task IDs")
    selected_counts = Counter(str(task.difficulty_label) for task in plan)
    if any(selected_counts[label] != normalized_quotas[label] for label in LABELS):
        raise AssertionError("formal batch difficulty quotas drifted")

    selected_categories = {str(task.category or "unknown") for task in plan}
    selected_top = {_top_category(task.category) for task in plan}
    report = {
        "selector_version": FORMAL_BATCH_SELECTOR_VERSION,
        "difficulty_version": next(iter(versions)),
        "seed": int(seed),
        "sample_size": sample_size,
        "quotas": normalized_quotas,
        "pool_counts": {label: pool_counts[label] for label in LABELS},
        "eligible_counts": {label: eligible_counts[label] for label in LABELS},
        "selected_counts": {label: selected_counts[label] for label in LABELS},
        "candidate_tasks": len(candidates),
        "eligible_tasks": len(eligible),
        "calibration_ids_supplied": len(calibration_ids),
        "prior_formal_ids_supplied": len(prior_ids),
        "held_out_ids_supplied": len(held_out_ids),
        "candidate_calibration_overlap_excluded": len(set(by_task).intersection(calibration_ids)),
        "candidate_prior_formal_overlap_excluded": len(set(by_task).intersection(prior_ids)),
        "candidate_held_out_overlap_excluded": len(set(by_task).intersection(held_out_ids)),
        "calibration_prior_formal_overlap": len(calibration_ids.intersection(prior_ids)),
        "calibration_held_out_overlap": len(calibration_ids.intersection(held_out_ids)),
        "prior_formal_held_out_overlap": len(prior_ids.intersection(held_out_ids)),
        "selected_calibration_overlap": 0,
        "selected_prior_formal_overlap": 0,
        "selected_held_out_overlap": 0,
        "pool_full_categories": len(full_categories),
        "selected_full_categories": len(selected_categories),
        "pool_top_categories": len(top_categories),
        "selected_top_categories": len(selected_top),
        "prefix_counts": {
            str(prefix): {
                label: Counter(
                    str(task.difficulty_label) for task in plan[:prefix]
                )[label]
                for label in LABELS
            }
            for prefix in (10, 20, 50, sample_size)
            if prefix <= sample_size
        },
    }
    return plan, report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows_by_task_id(
    path: Path,
    label: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    by_task: dict[int, dict[str, Any]] = {}
    for row in rows:
        if type(row.get("task_id")) is not int or int(row["task_id"]) < 0:
            raise ValueError(f"{label} contains an invalid task_id")
        task_id = int(row["task_id"])
        if task_id in by_task:
            raise ValueError(f"{label} contains duplicate task_id={task_id}")
        by_task[task_id] = row
    return rows, by_task


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    temporary = path.with_name(path.name + ".wlx_tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        if _sha256_file(path) == _sha256_file(temporary):
            temporary.unlink()
            return "unchanged"
        temporary.unlink()
        raise FileExistsError(f"refusing to replace different existing output: {path}")
    temporary.replace(path)
    return "created"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    text = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".wlx_tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        if _sha256_file(path) == _sha256_file(temporary):
            temporary.unlink()
            return "unchanged"
        temporary.unlink()
        raise FileExistsError(f"refusing to replace different existing output: {path}")
    temporary.replace(path)
    return "created"


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return dict(value)


def _load_formal_candidates(
    args: argparse.Namespace,
) -> tuple[list[SftTask], dict[str, Any]]:
    """Load an existing public formal plan or derive it from public tasks/features/model."""

    formal_plan = getattr(args, "formal_plan", None)
    generated_paths = {
        "tasks": getattr(args, "tasks", None),
        "features": getattr(args, "features", None),
        "model": getattr(args, "model", None),
    }
    if formal_plan is not None:
        if any(value is not None for value in generated_paths.values()):
            raise SystemExit(
                "use either --formal-plan or --tasks/--features/--model, not both"
            )
        candidates = load_sft_tasks(formal_plan)
        source = {
            "source_mode": "formal_plan",
            "formal_plan_sha256": _sha256_file(Path(formal_plan)),
        }
    else:
        missing = [name for name, value in generated_paths.items() if value is None]
        if missing:
            raise SystemExit(
                "without --formal-plan, these arguments are required: "
                + ", ".join(f"--{name}" for name in missing)
            )
        tasks_path = Path(generated_paths["tasks"])
        features_path = Path(generated_paths["features"])
        model_path = Path(generated_paths["model"])
        public_tasks = load_sft_tasks(tasks_path)
        feature_rows = [
            TaskDifficultyFeatures.from_dict(row) for row in read_jsonl(features_path)
        ]
        features_by_task = {item.task_id: item for item in feature_rows}
        if len(features_by_task) != len(feature_rows):
            raise ValueError("difficulty feature table contains duplicate task_id values")
        public_ids = {task.task_id for task in public_tasks}
        if public_ids != set(features_by_task):
            raise ValueError("public tasks and difficulty features have different task_id sets")
        model = LogisticDifficultyModel.from_dict(
            _read_json_object(model_path, "difficulty model")
        )
        candidates = apply_difficulty_model(public_tasks, features_by_task, model)
        source = {
            "source_mode": "features_and_model",
            "tasks_sha256": _sha256_file(tasks_path),
            "features_sha256": _sha256_file(features_path),
            "model_sha256": _sha256_file(model_path),
        }
    if not candidates:
        raise SystemExit("formal candidate pool is empty")
    return candidates, source


def _load_prior_formal_exclusions(
    paths: Sequence[Path],
) -> tuple[set[int], dict[str, Any]]:
    """Load all prior public plans and retain only safe hashes and aggregate counts."""

    references: list[int] = []
    plan_records: list[dict[str, Any]] = []
    for path in paths:
        tasks = load_sft_tasks(path)
        if not tasks:
            raise SystemExit(f"prior formal plan is empty: {path}")
        task_ids = [task.task_id for task in tasks]
        references.extend(task_ids)
        plan_records.append(
            {
                "sha256": _sha256_file(Path(path)),
                "task_count": len(task_ids),
            }
        )
    counts = Counter(references)
    return set(counts), {
        "prior_formal_plan_count": len(plan_records),
        "prior_formal_plans": plan_records,
        "prior_formal_task_references": len(references),
        "prior_formal_unique_tasks": len(counts),
        "prior_formal_duplicate_task_references": sum(
            count - 1 for count in counts.values()
        ),
    }


def plan_formal_batch(args: argparse.Namespace) -> int:
    """Freeze the first 100-task formal batch after calibration/evaluation exclusions."""

    candidates, source = _load_formal_candidates(args)
    calibration_tasks = load_sft_tasks(args.calibration_plan)
    if not calibration_tasks:
        raise SystemExit("calibration plan is empty; exclusions cannot be verified")
    calibration_ids = {task.task_id for task in calibration_tasks}
    held_out_ids = load_held_out_task_ids(args.held_out_tasks)
    prior_ids, prior_metadata = _load_prior_formal_exclusions(
        tuple(getattr(args, "prior_formal_plan", ()) or ())
    )
    quotas = {
        "easy": int(getattr(args, "easy_count", FORMAL_BATCH_QUOTAS["easy"])),
        "medium": int(
            getattr(args, "medium_count", FORMAL_BATCH_QUOTAS["medium"])
        ),
        "hard": int(getattr(args, "hard_count", FORMAL_BATCH_QUOTAS["hard"])),
    }
    plan, report = balanced_formal_batch_plan(
        candidates,
        calibration_task_ids=calibration_ids,
        held_out_task_ids=held_out_ids,
        prior_formal_task_ids=prior_ids,
        quotas=quotas,
        seed=args.seed,
    )
    plan_rows = [task.to_public_dict() for task in plan]
    if any(set(row) != PUBLIC_TASK_FIELDS for row in plan_rows):
        raise AssertionError("formal batch public field whitelist drifted")
    if any(
        task.task_id in calibration_ids
        or task.task_id in prior_ids
        or task.task_id in held_out_ids
        for task in plan
    ):
        raise AssertionError("formal batch exclusion check drifted before write")

    output = Path(args.output)
    output_status = _atomic_write_jsonl(output, plan_rows)
    plan_sha256 = _sha256_file(output)
    metadata_path = Path(args.metadata)
    recorded_output_status = output_status
    if metadata_path.is_file():
        existing_metadata = _read_json_object(metadata_path, "formal batch metadata")
        if existing_metadata.get("plan_sha256") == plan_sha256:
            recorded_output_status = str(
                existing_metadata.get("output_status", output_status)
            )
    metadata = {
        **report,
        **source,
        **prior_metadata,
        "output_status": recorded_output_status,
        "public_fields": sorted(PUBLIC_TASK_FIELDS),
        "calibration_plan_sha256": _sha256_file(Path(args.calibration_plan)),
        "evaluation_sha256": _sha256_file(Path(args.held_out_tasks)),
        "plan_sha256": plan_sha256,
    }
    metadata_status = _atomic_write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "plan": output_status,
                "metadata": metadata_status,
                "difficulty_version": report["difficulty_version"],
                "selected_counts": report["selected_counts"],
                "selected_full_categories": report["selected_full_categories"],
                "selected_top_categories": report["selected_top_categories"],
                "calibration_overlap": 0,
                "prior_formal_overlap": 0,
                "held_out_overlap": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def materialize_features(args: argparse.Namespace) -> int:
    """Reuse the complete cache and materialize corrected difficulty features."""

    shop_root = REPOSITORY / "environments" / "ShopSimulator" / "shop_env"
    if str(shop_root) not in sys.path:
        sys.path.insert(0, str(shop_root))
    try:
        from web_agent_site.engine.engine import load_products
        from web_agent_site.engine.goal import get_goals
    except ImportError as exc:
        raise SystemExit(
            "run materialize-features with the ShopSimulator Python environment"
        ) from exc

    output_dir = Path(args.output_dir)
    cache_path = output_dir / "wlx-retrieval-cache.jsonl"
    public_path = output_dir / "wlx-public-train-tasks.jsonl"
    if not cache_path.is_file() or not public_path.is_file():
        raise SystemExit("the complete retrieval cache and public task table are required")

    products, _, prices, _ = load_products(filepath=str(args.product_file), num_products=None)
    goals = get_goals(products, prices)
    held_out = load_held_out_task_ids(args.held_out_tasks)
    train_ids = set(range(len(goals))).difference(held_out)
    expected_tasks = public_tasks_from_canonical_goals(goals, held_out_task_ids=held_out)
    expected_public = {task.task_id: task.to_public_dict() for task in expected_tasks}

    public_rows, public_by_task = _rows_by_task_id(public_path, "public task table")
    cache_rows, cache_by_task = _rows_by_task_id(cache_path, "retrieval cache")
    if set(public_by_task) != train_ids or set(cache_by_task) != train_ids:
        raise SystemExit("public tasks/cache do not exactly match the canonical Train task IDs")
    if public_by_task != expected_public:
        raise SystemExit("public task table no longer matches canonical public goal fields")
    if any(set(row) != PUBLIC_TASK_FIELDS for row in public_rows):
        raise SystemExit("public task table contains unexpected or missing fields")

    versions = {json.dumps(row.get("versions"), sort_keys=True) for row in cache_rows}
    if len(versions) != 1:
        raise SystemExit("retrieval cache contains mixed version manifests")
    cache_versions = dict(cache_rows[0].get("versions") or {})
    product_sha256 = _sha256_file(Path(args.product_file))
    if cache_versions.get("product_data_sha256") != product_sha256:
        raise SystemExit("retrieval cache product SHA-256 does not match the frozen product file")

    train_goal_pairs = [(task_id, goals[task_id]) for task_id in sorted(train_ids)]
    a95 = percentile_95(
        independent_constraint_count(goal, task_id=task_id)[0]
        for task_id, goal in train_goal_pairs
    )
    n95 = percentile_95(
        int(cache_by_task[task_id].get("near_miss_count_at_20", 0))
        for task_id in sorted(train_ids)
    )
    features = [
        build_task_features(
            task_id=task_id,
            goal=goal,
            retrieval=retrieval_evidence_from_cache(cache_by_task[task_id]),
            a95=a95,
            n95=n95,
            category=str(goal.get("category") or "") or None,
        )
        for task_id, goal in train_goal_pairs
    ]
    features = assign_preliminary_labels(features)
    feature_rows = [item.to_dict() for item in features]
    features_path = output_dir / "wlx-difficulty-features.jsonl"
    feature_status = _atomic_write_jsonl(features_path, feature_rows)

    label_counts = Counter(str(item.preliminary_label) for item in features)
    stats = {
        "train_tasks": len(features),
        "a95": a95,
        "n95": n95,
        "constraint_count_version": CONSTRAINT_COUNT_VERSION,
        "label_counts": {label: label_counts[label] for label in LABELS},
        "full_categories": len({str(item.category or "unknown") for item in features}),
        "top_categories": len({_top_category(item.category) for item in features}),
        "product_sha256": product_sha256,
        "evaluation_sha256": _sha256_file(Path(args.held_out_tasks)),
        "public_tasks_sha256": _sha256_file(public_path),
        "retrieval_cache_sha256": _sha256_file(cache_path),
        "features_sha256": _sha256_file(features_path),
        "cache_versions": cache_versions,
    }
    stats_path = output_dir / "wlx-difficulty-stats.json"
    stats_status = _atomic_write_json(stats_path, stats)
    print(
        json.dumps(
            {
                "features": feature_status,
                "stats": stats_status,
                "train_tasks": len(features),
                "a95": a95,
                "n95": n95,
                "label_counts": stats["label_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def plan_calibration(args: argparse.Namespace) -> int:
    """Build and atomically freeze the corrected public calibration task plan."""

    tasks = load_sft_tasks(args.tasks)
    features = [TaskDifficultyFeatures.from_dict(row) for row in read_jsonl(args.features)]
    held_out = load_held_out_task_ids(args.held_out_tasks)
    task_ids = {task.task_id for task in tasks}
    if task_ids.intersection(held_out):
        raise SystemExit("public task table overlaps the canonical Evaluation split")

    plan, report = balanced_calibration_task_plan(
        tasks,
        features,
        sample_size=args.size,
        seed=args.seed,
    )
    plan_rows = [task.to_public_dict() for task in plan]
    if any(set(row) != PUBLIC_TASK_FIELDS for row in plan_rows):
        raise AssertionError("calibration plan public field whitelist drifted")
    plan_ids = {task.task_id for task in plan}
    if len(plan_ids) != len(plan) or plan_ids.intersection(held_out):
        raise AssertionError("calibration plan contains duplicate or held-out task IDs")

    output = Path(args.output)
    output_status = _atomic_write_jsonl(output, plan_rows)
    feature_by_task = {item.task_id: item for item in features}
    default_ids = select_calibration_tasks(features, sample_size=args.size, seed=args.seed)
    default_distribution = Counter(
        str(feature_by_task[task_id].preliminary_label) for task_id in default_ids
    )
    plan_sha256 = _sha256_file(output)
    metadata_path = Path(args.metadata)
    recorded_output_status = output_status
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata.get("plan_sha256") == plan_sha256:
            recorded_output_status = str(
                existing_metadata.get("output_status", output_status)
            )
    metadata = {
        **report,
        # Preserve the first freeze status so an identical rerun is byte-for-byte stable.
        "output_status": recorded_output_status,
        "held_out_overlap": 0,
        "public_fields": sorted(PUBLIC_TASK_FIELDS),
        "default_selector_distribution_for_audit": {
            label: default_distribution[label] for label in LABELS
        },
        "tasks_sha256": _sha256_file(Path(args.tasks)),
        "features_sha256": _sha256_file(Path(args.features)),
        "evaluation_sha256": _sha256_file(Path(args.held_out_tasks)),
        "plan_sha256": plan_sha256,
    }
    metadata_status = _atomic_write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "plan": output_status,
                "metadata": metadata_status,
                "selected_counts": report["selected_counts"],
                "selected_full_categories": report["selected_full_categories"],
                "selected_top_categories": report["selected_top_categories"],
                "held_out_overlap": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    features = subparsers.add_parser(
        "materialize-features",
        help="reuse a complete retrieval cache and build corrected C/R/N features",
    )
    features.add_argument("--output-dir", type=Path, required=True)
    features.add_argument(
        "--held-out-tasks", type=Path, default=Path("data/evaluation/tasks.jsonl")
    )
    features.add_argument(
        "--product-file",
        type=Path,
        default=Path("environments/ShopSimulator/shop_env/data/items_eval_train.json"),
    )

    plan = subparsers.add_parser(
        "plan-calibration",
        help="select a deterministic category-rich 30/50/20 calibration plan",
    )
    plan.add_argument("--tasks", type=Path, required=True)
    plan.add_argument("--features", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--metadata", type=Path, required=True)
    plan.add_argument(
        "--held-out-tasks", type=Path, default=Path("data/evaluation/tasks.jsonl")
    )
    plan.add_argument("--size", type=int, default=200)
    plan.add_argument("--seed", type=int, default=42)

    formal = subparsers.add_parser(
        "plan-formal-batch",
        help="select the first public-only 100-task formal batch with a 30/40/30 mix",
    )
    formal.add_argument(
        "--formal-plan",
        type=Path,
        help="existing full plan-formal JSONL; mutually exclusive with feature inputs",
    )
    formal.add_argument("--tasks", type=Path, help="full public Train task JSONL")
    formal.add_argument("--features", type=Path, help="full difficulty feature JSONL")
    formal.add_argument("--model", type=Path, help="frozen fitted difficulty model JSON")
    formal.add_argument("--calibration-plan", type=Path, required=True)
    formal.add_argument(
        "--prior-formal-plan",
        type=Path,
        action="append",
        default=[],
        help="repeat for every earlier formal batch whose task IDs must be excluded",
    )
    formal.add_argument(
        "--held-out-tasks", type=Path, default=Path("data/evaluation/tasks.jsonl")
    )
    formal.add_argument("--output", type=Path, required=True)
    formal.add_argument("--metadata", type=Path, required=True)
    formal.add_argument("--easy-count", type=int, default=FORMAL_BATCH_QUOTAS["easy"])
    formal.add_argument(
        "--medium-count", type=int, default=FORMAL_BATCH_QUOTAS["medium"]
    )
    formal.add_argument("--hard-count", type=int, default=FORMAL_BATCH_QUOTAS["hard"])
    formal.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "materialize-features":
        return materialize_features(args)
    if args.command == "plan-calibration":
        return plan_calibration(args)
    if args.command == "plan-formal-batch":
        return plan_formal_batch(args)
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
