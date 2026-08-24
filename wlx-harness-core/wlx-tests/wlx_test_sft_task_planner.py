"""Offline tests for the WLX task-planning sidecar."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from wlx_harness_core.wlx_sft_contracts import SftTask
from wlx_harness_core.wlx_sft_difficulty import (
    DIFFICULTY_FEATURE_NAMES,
    LogisticDifficultyModel,
    TaskDifficultyFeatures,
)


REPOSITORY = Path(__file__).resolve().parents[2]
PLANNER_PATH = REPOSITORY / "scripts" / "wlx_sft_task_planner.py"
SPEC = importlib.util.spec_from_file_location("wlx_sft_task_planner", PLANNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {PLANNER_PATH}")
PLANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLANNER
SPEC.loader.exec_module(PLANNER)


def _feature(task_id: int, label: str, category: str) -> TaskDifficultyFeatures:
    score = {"easy": 0.1, "medium": 0.5, "hard": 0.9}[label]
    return TaskDifficultyFeatures(
        task_id=task_id,
        constraint_count=2,
        option_axis_count=1,
        has_brand=False,
        has_model=False,
        has_budget=True,
        retrieval_score=score,
        near_miss_score=score,
        constraint_score=score,
        preliminary_score=score,
        preliminary_label=label,
        category=category,
        evidence={},
    )


def _formal_task(task_id: int, label: str, category: str) -> SftTask:
    score = {"easy": 0.1, "medium": 0.5, "hard": 0.9}[label]
    return SftTask(
        task_id=task_id,
        instruction=f"public formal task {task_id}",
        official_split="train",
        category=category,
        difficulty_label=label,
        difficulty_score=score,
        difficulty_version="wlx-difficulty-test-v3",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class ConstraintCountTest(unittest.TestCase):
    def test_dictionary_requirements_are_normalised_and_deduplicated(self):
        goal = {
            "expected_core_functions": ["功能 A", "功能 A", "功能 B"],
            "required_options_by_key": {"Color": {}, "size": {}},
            "unresolved_option_requirements": [
                {"value": "Ａ / B", "reason": "axis_not_found", "axes": []},
                {"axes": ["颜色"], "reason": "axis_ambiguous", "value": "a| b"},
                {"value": "C", "reason": "axis_not_found", "axes": []},
            ],
            "expected_brand": ["品牌"],
            "expected_model": [],
            "price_upper": 100,
        }
        count, parts = PLANNER.independent_constraint_count(goal, task_id=7)
        self.assertEqual(count, 8)
        self.assertEqual(parts["core_function_count"], 2)
        self.assertEqual(parts["option_axis_count"], 2)
        self.assertEqual(parts["unresolved_option_count"], 2)

    def test_malformed_unresolved_requirement_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "task_id=8"):
            PLANNER.independent_constraint_count(
                {"unresolved_option_requirements": [{"reason": "missing value"}]},
                task_id=8,
            )


class BalancedPlanTest(unittest.TestCase):
    def setUp(self):
        labels = ["easy"] * 30 + ["medium"] * 50 + ["hard"] * 20
        self.features = [
            _feature(task_id, label, f"Top{task_id % 8}›Category{task_id}")
            for task_id, label in enumerate(labels)
        ]
        self.tasks = [
            SftTask(
                task_id=item.task_id,
                instruction=f"public task {item.task_id}",
                official_split="train",
                category=item.category,
            )
            for item in self.features
        ]

    def test_exact_quotas_category_coverage_and_prefix_balance(self):
        plan, report = PLANNER.balanced_calibration_task_plan(
            self.tasks, self.features, sample_size=20, seed=42
        )
        self.assertEqual(len(plan), 20)
        self.assertEqual(len({task.task_id for task in plan}), 20)
        self.assertEqual(Counter(task.difficulty_label for task in plan), {
            "easy": 6,
            "medium": 10,
            "hard": 4,
        })
        self.assertEqual(report["selected_full_categories"], 20)
        self.assertEqual(report["selected_top_categories"], 8)
        self.assertEqual(report["prefix_counts"]["10"], {
            "easy": 3,
            "medium": 5,
            "hard": 2,
        })

    def test_selection_is_stable_under_input_reordering(self):
        first, _ = PLANNER.balanced_calibration_task_plan(
            self.tasks, self.features, sample_size=20, seed=42
        )
        second, _ = PLANNER.balanced_calibration_task_plan(
            list(reversed(self.tasks)), list(reversed(self.features)), sample_size=20, seed=42
        )
        self.assertEqual(
            [task.to_public_dict() for task in first],
            [task.to_public_dict() for task in second],
        )

    def test_different_seed_changes_tasks_but_not_contract(self):
        first, _ = PLANNER.balanced_calibration_task_plan(
            self.tasks, self.features, sample_size=20, seed=1
        )
        second, _ = PLANNER.balanced_calibration_task_plan(
            self.tasks, self.features, sample_size=20, seed=2
        )
        self.assertNotEqual(
            {task.task_id for task in first},
            {task.task_id for task in second},
        )
        for plan in (first, second):
            self.assertEqual(Counter(task.difficulty_label for task in plan), {
                "easy": 6,
                "medium": 10,
                "hard": 4,
            })
            self.assertTrue(
                all(set(task.to_public_dict()) == PLANNER.PUBLIC_TASK_FIELDS for task in plan)
            )

    def test_frozen_plan_and_metadata_are_idempotent(self):
        with tempfile.TemporaryDirectory(prefix="wlx_planner_") as temporary:
            directory = Path(temporary)
            tasks_path = directory / "wlx_tasks.jsonl"
            features_path = directory / "wlx_features.jsonl"
            held_out_path = directory / "wlx_held_out.jsonl"
            plan_path = directory / "wlx_plan.jsonl"
            metadata_path = directory / "wlx_metadata.json"

            def write_jsonl(path: Path, rows: list[dict]) -> None:
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            write_jsonl(tasks_path, [task.to_public_dict() for task in self.tasks])
            write_jsonl(features_path, [item.to_dict() for item in self.features])
            write_jsonl(held_out_path, [{"task_id": 999}])
            arguments = argparse.Namespace(
                tasks=tasks_path,
                features=features_path,
                held_out_tasks=held_out_path,
                size=20,
                seed=42,
                output=plan_path,
                metadata=metadata_path,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(PLANNER.plan_calibration(arguments), 0)
            first_plan = plan_path.read_bytes()
            first_metadata = metadata_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(PLANNER.plan_calibration(arguments), 0)

            self.assertEqual(plan_path.read_bytes(), first_plan)
            self.assertEqual(metadata_path.read_bytes(), first_metadata)
            self.assertEqual(
                json.loads(first_metadata)["output_status"],
                "created",
            )


class FormalBatchPlanTest(unittest.TestCase):
    def setUp(self):
        labels = ["easy"] * 12 + ["medium"] * 12 + ["hard"] * 12
        self.candidates = [
            _formal_task(
                task_id,
                label,
                f"Top{task_id % 6}›FormalCategory{task_id}",
            )
            for task_id, label in enumerate(labels)
        ]

    def test_exact_quotas_deduplication_and_explicit_exclusions(self):
        calibration_ids = {0, 12, 24}
        held_out_ids = {1, 13, 25}
        plan, report = PLANNER.balanced_formal_batch_plan(
            self.candidates,
            calibration_task_ids=calibration_ids,
            held_out_task_ids=held_out_ids,
            quotas={"easy": 3, "medium": 4, "hard": 3},
            seed=42,
        )

        selected_ids = {task.task_id for task in plan}
        self.assertEqual(len(plan), 10)
        self.assertEqual(len(selected_ids), 10)
        self.assertFalse(selected_ids.intersection(calibration_ids))
        self.assertFalse(selected_ids.intersection(held_out_ids))
        self.assertEqual(
            Counter(task.difficulty_label for task in plan),
            {"easy": 3, "medium": 4, "hard": 3},
        )
        self.assertEqual(report["selected_full_categories"], 10)
        self.assertEqual(report["selected_calibration_overlap"], 0)
        self.assertEqual(report["selected_held_out_overlap"], 0)

    def test_formal_selection_is_stable_under_input_reordering(self):
        arguments = {
            "calibration_task_ids": {0, 12, 24},
            "held_out_task_ids": {1, 13, 25},
            "quotas": {"easy": 3, "medium": 4, "hard": 3},
            "seed": 7,
        }
        first, _ = PLANNER.balanced_formal_batch_plan(
            self.candidates,
            **arguments,
        )
        second, _ = PLANNER.balanced_formal_batch_plan(
            list(reversed(self.candidates)),
            **arguments,
        )
        self.assertEqual(
            [task.to_public_dict() for task in first],
            [task.to_public_dict() for task in second],
        )

    def test_duplicate_formal_candidate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate task_id"):
            PLANNER.balanced_formal_batch_plan(
                self.candidates + [self.candidates[0]],
                calibration_task_ids=set(),
                held_out_task_ids=set(),
                quotas={"easy": 3, "medium": 4, "hard": 3},
                seed=42,
            )

    def test_hard_only_100_task_quota(self):
        candidates = [
            _formal_task(task_id, "hard", f"Top{task_id % 50}›HardCategory{task_id}")
            for task_id in range(120)
        ]
        calibration_ids = set(range(5))
        prior_ids = set(range(5, 10))
        held_out_ids = set(range(10, 15))

        plan, report = PLANNER.balanced_formal_batch_plan(
            candidates,
            calibration_task_ids=calibration_ids,
            prior_formal_task_ids=prior_ids,
            held_out_task_ids=held_out_ids,
            quotas={"easy": 0, "medium": 0, "hard": 100},
            seed=42,
        )

        selected_ids = {task.task_id for task in plan}
        self.assertEqual(len(plan), 100)
        self.assertEqual(len(selected_ids), 100)
        self.assertEqual(
            Counter(task.difficulty_label for task in plan),
            {"hard": 100},
        )
        self.assertFalse(
            selected_ids.intersection(calibration_ids | prior_ids | held_out_ids)
        )
        self.assertEqual(report["quotas"], {
            "easy": 0,
            "medium": 0,
            "hard": 100,
        })
        self.assertEqual(report["candidate_prior_formal_overlap_excluded"], 5)
        self.assertEqual(report["selected_prior_formal_overlap"], 0)

    def test_cli_accepts_hard_only_counts_and_repeatable_prior_plans(self):
        args = PLANNER.build_parser().parse_args(
            [
                "plan-formal-batch",
                "--formal-plan",
                "wlx-formal-all.jsonl",
                "--calibration-plan",
                "wlx-calibration.jsonl",
                "--prior-formal-plan",
                "wlx-prior-001.jsonl",
                "--prior-formal-plan",
                "wlx-prior-002.jsonl",
                "--output",
                "wlx-hard-plan.jsonl",
                "--metadata",
                "wlx-hard-metadata.json",
                "--easy-count",
                "0",
                "--medium-count",
                "0",
                "--hard-count",
                "100",
            ]
        )

        self.assertEqual((args.easy_count, args.medium_count, args.hard_count), (0, 0, 100))
        self.assertEqual(
            args.prior_formal_plan,
            [Path("wlx-prior-001.jsonl"), Path("wlx-prior-002.jsonl")],
        )

    def test_cli_writes_public_only_fixed_100_task_batch(self):
        with tempfile.TemporaryDirectory(prefix="wlx_formal_batch_") as temporary:
            root = Path(temporary)
            formal_path = root / "wlx_formal_all.jsonl"
            calibration_path = root / "wlx_calibration.jsonl"
            prior_one_path = root / "wlx_prior_formal_001.jsonl"
            prior_two_path = root / "wlx_prior_formal_002.jsonl"
            held_out_path = root / "wlx_evaluation.jsonl"
            output_path = root / "wlx_formal_batch_001.jsonl"
            metadata_path = root / "wlx_formal_batch_001_metadata.json"
            labels = ["easy"] * 36 + ["medium"] * 46 + ["hard"] * 36
            candidates = [
                _formal_task(
                    task_id,
                    label,
                    f"Top{task_id % 20}›UniqueCategory{task_id}",
                )
                for task_id, label in enumerate(labels)
            ]
            formal_rows = []
            for task in candidates:
                row = task.to_public_dict()
                row["target_asin"] = f"hidden-{task.task_id}"
                formal_rows.append(row)
            calibration_ids = {0, 36, 82}
            prior_one_ids = {2, 38, 84}
            prior_two_ids = {3, 39, 85}
            prior_ids = prior_one_ids | prior_two_ids
            held_out_ids = {1, 37, 83}
            _write_jsonl(formal_path, formal_rows)
            _write_jsonl(
                calibration_path,
                [
                    next(task for task in candidates if task.task_id == task_id).to_public_dict()
                    for task_id in sorted(calibration_ids)
                ],
            )
            for path, task_ids in (
                (prior_one_path, prior_one_ids),
                (prior_two_path, prior_two_ids),
            ):
                _write_jsonl(
                    path,
                    [
                        next(
                            task
                            for task in candidates
                            if task.task_id == task_id
                        ).to_public_dict()
                        for task_id in sorted(task_ids)
                    ],
                )
            _write_jsonl(
                held_out_path,
                [{"task_id": task_id} for task_id in sorted(held_out_ids)],
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    PLANNER.main(
                        [
                            "plan-formal-batch",
                            "--formal-plan",
                            str(formal_path),
                            "--calibration-plan",
                            str(calibration_path),
                            "--prior-formal-plan",
                            str(prior_one_path),
                            "--prior-formal-plan",
                            str(prior_two_path),
                            "--held-out-tasks",
                            str(held_out_path),
                            "--output",
                            str(output_path),
                            "--metadata",
                            str(metadata_path),
                            "--seed",
                            "42",
                        ]
                    ),
                    0,
                )

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            selected_ids = {int(row["task_id"]) for row in rows}
            self.assertEqual(len(rows), 100)
            self.assertEqual(len(selected_ids), 100)
            self.assertFalse(
                selected_ids.intersection(calibration_ids | prior_ids | held_out_ids)
            )
            self.assertEqual(
                Counter(str(row["difficulty_label"]) for row in rows),
                {"easy": 30, "medium": 40, "hard": 30},
            )
            self.assertTrue(
                all(set(row) == PLANNER.PUBLIC_TASK_FIELDS for row in rows)
            )
            self.assertNotIn("target_asin", output_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["candidate_calibration_overlap_excluded"], 3)
            self.assertEqual(metadata["candidate_prior_formal_overlap_excluded"], 6)
            self.assertEqual(metadata["candidate_held_out_overlap_excluded"], 3)
            self.assertEqual(metadata["prior_formal_plan_count"], 2)
            self.assertEqual(metadata["prior_formal_task_references"], 6)
            self.assertEqual(metadata["prior_formal_unique_tasks"], 6)
            self.assertEqual(metadata["prior_formal_duplicate_task_references"], 0)
            self.assertEqual(
                [item["task_count"] for item in metadata["prior_formal_plans"]],
                [3, 3],
            )
            self.assertTrue(
                all(set(item) == {"sha256", "task_count"} for item in metadata["prior_formal_plans"])
            )
            self.assertEqual(metadata["selected_prior_formal_overlap"], 0)
            self.assertEqual(metadata["selected_counts"], {
                "easy": 30,
                "medium": 40,
                "hard": 30,
            })

    def test_candidates_can_be_derived_from_public_features_and_model(self):
        with tempfile.TemporaryDirectory(prefix="wlx_formal_source_") as temporary:
            root = Path(temporary)
            tasks_path = root / "wlx_tasks.jsonl"
            features_path = root / "wlx_features.jsonl"
            model_path = root / "wlx_model.json"
            tasks = [
                SftTask(
                    task_id=task_id,
                    instruction=f"task {task_id}",
                    official_split="train",
                    category=f"Category{task_id}",
                )
                for task_id in range(3)
            ]
            features = [
                TaskDifficultyFeatures(
                    task_id=task_id,
                    constraint_count=task_id,
                    option_axis_count=0,
                    has_brand=False,
                    has_model=False,
                    has_budget=False,
                    retrieval_score=0.0,
                    near_miss_score=0.0,
                    constraint_score=0.0,
                    preliminary_score=0.0,
                    category=f"Category{task_id}",
                )
                for task_id in range(3)
            ]
            model = LogisticDifficultyModel(
                feature_names=DIFFICULTY_FEATURE_NAMES,
                means=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                scales=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                coefficients=(-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                intercept=0.0,
                difficulty_version="wlx-difficulty-test-v3",
            )
            _write_jsonl(tasks_path, [task.to_public_dict() for task in tasks])
            _write_jsonl(features_path, [feature.to_dict() for feature in features])
            model_path.write_text(json.dumps(model.to_dict()), encoding="utf-8")

            candidates, source = PLANNER._load_formal_candidates(
                argparse.Namespace(
                    formal_plan=None,
                    tasks=tasks_path,
                    features=features_path,
                    model=model_path,
                )
            )

            self.assertEqual(
                [task.difficulty_label for task in candidates],
                ["easy", "medium", "hard"],
            )
            self.assertEqual(source["source_mode"], "features_and_model")
            self.assertEqual(set(source), {
                "source_mode",
                "tasks_sha256",
                "features_sha256",
                "model_sha256",
            })


if __name__ == "__main__":
    unittest.main()
