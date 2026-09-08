"""CPU-only checks for deterministic Eval v3."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from shopping_grpo.evaluation.eval_v3 import (
    evaluate_trajectory,
    run_offline_evaluation,
    summarize,
)


def _trajectory(
    task_id: int,
    *,
    reward_type: str,
    asin_match: bool,
    option_passed: int,
    purchased: bool = True,
) -> dict:
    dimensions = {
        "brand": {
            "active": False,
            "required_count": 0,
            "passed_count": 0,
            "verifiable_count": 0,
        },
        "model": {
            "active": False,
            "required_count": 0,
            "passed_count": 0,
            "verifiable_count": 0,
        },
        "core_functions": {
            "active": True,
            "required_count": 2,
            "passed_count": 2,
            "verifiable_count": 2,
        },
        "key_options": {
            "active": True,
            "required_count": 1,
            "passed_count": option_passed,
            "verifiable_count": 1,
        },
    }
    terminal = {
        "done": True,
        "over": True,
        "reward_valid": True,
        "purchase": (
            {
                "asin": "gold" if asin_match else "other",
                "category": "服饰›上衣",
                "price": 99.0,
                "options": {"颜色": "米色"},
            }
            if purchased
            else None
        ),
        "reward_detail": {
            "reward_version": "shopsimulator-reward-v3",
            "reward_valid": True,
            "reward_type": reward_type,
            "purchase_success": reward_type == "gold_purchase",
            "target_asin_match": asin_match,
            "hard_gates": {
                "category": {"status": "pass", "passed": True},
                "budget": {"status": "pass", "passed": True},
            },
            "evidence": {"preference_scoring": {"dimensions": dimensions}},
        },
    }
    if not purchased:
        terminal["reward_detail"]["hard_gates"] = {}
        terminal["reward_detail"]["evidence"] = {}
    return {
        "task_id": task_id,
        "trajectory_id": f"trajectory-{task_id}",
        "done": True,
        "infrastructure_invalid": False,
        "reward_valid": True,
        "terminal_result": terminal,
    }


class EvalV3Test(unittest.TestCase):
    def test_gold_uses_environment_gold_and_exact_options(self):
        result = evaluate_trajectory(
            _trajectory(
                1,
                reward_type="gold_purchase",
                asin_match=True,
                option_passed=1,
            )
        )
        self.assertTrue(result["gold"])
        self.assertTrue(result["asin_match"])
        self.assertEqual(result["attributes"]["ratio"], 1.0)
        self.assertEqual(result["options"]["ratio"], 1.0)
        self.assertTrue(result["all_constraints"])

    def test_same_asin_with_wrong_option_is_not_gold(self):
        result = evaluate_trajectory(
            _trajectory(
                2,
                reward_type="partial_alternative_purchase",
                asin_match=True,
                option_passed=0,
            )
        )
        self.assertFalse(result["gold"])
        self.assertTrue(result["asin_match"])
        self.assertEqual(result["options"]["ratio"], 0.0)
        self.assertFalse(result["all_constraints"])

    def test_wrong_asin_can_still_pass_all_structured_constraints(self):
        result = evaluate_trajectory(
            _trajectory(
                4,
                reward_type="partial_alternative_purchase",
                asin_match=False,
                option_passed=1,
            )
        )
        self.assertFalse(result["gold"])
        self.assertFalse(result["asin_match"])
        self.assertTrue(result["category"]["passed"])
        self.assertEqual(result["attributes"]["ratio"], 1.0)
        self.assertEqual(result["options"]["ratio"], 1.0)
        self.assertTrue(result["price"]["passed"])
        self.assertTrue(result["all_constraints"])

    def test_no_purchase_scores_zero_on_all_outcome_dimensions(self):
        result = evaluate_trajectory(
            _trajectory(
                3,
                reward_type="repeat_loop",
                asin_match=False,
                option_passed=0,
                purchased=False,
            )
        )
        self.assertTrue(result["no_purchase"])
        self.assertFalse(result["gold"])
        self.assertFalse(result["category"]["passed"])
        self.assertEqual(result["attributes"]["ratio"], 0.0)
        self.assertEqual(result["options"]["ratio"], 0.0)
        self.assertFalse(result["price"]["passed"])

    def test_summary_uses_fixed_task_denominator(self):
        records = [
            evaluate_trajectory(
                _trajectory(1, reward_type="gold_purchase", asin_match=True, option_passed=1)
            ),
            evaluate_trajectory(
                _trajectory(
                    2,
                    reward_type="repeat_loop",
                    asin_match=False,
                    option_passed=0,
                    purchased=False,
                )
            ),
        ]
        result = summarize(
            records,
            expected_task_ids=[1, 2],
            model_role="grpo",
            model_name="step-300",
        )
        self.assertEqual(result["overall"]["gold"], {"count": 1, "rate": 0.5})
        self.assertEqual(result["overall"]["purchase"], {"count": 1, "rate": 0.5})
        self.assertEqual(result["overall"]["option_ratio_mean"], 0.5)

    def test_offline_run_writes_prefixed_outputs_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.jsonl"
            trajectories = root / "trajectories.jsonl"
            output = root / "output"
            tasks.write_text('{"task_id":1}\n{"task_id":2}\n', encoding="utf-8")
            trajectories.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in (
                        _trajectory(
                            1,
                            reward_type="gold_purchase",
                            asin_match=True,
                            option_passed=1,
                        ),
                        _trajectory(
                            2,
                            reward_type="repeat_loop",
                            asin_match=False,
                            option_passed=0,
                            purchased=False,
                        ),
                    )
                ),
                encoding="utf-8",
            )
            first = run_offline_evaluation(
                trajectories_path=trajectories,
                tasks_path=tasks,
                output_dir=output,
                model_role="grpo",
                model_name="step-300",
                limit=2,
            )
            second = run_offline_evaluation(
                trajectories_path=trajectories,
                tasks_path=tasks,
                output_dir=output,
                model_role="grpo",
                model_name="step-300",
                limit=2,
            )
            self.assertEqual(first, second)
            self.assertTrue((output / "eval-v3-manifest.json").is_file())
            self.assertTrue((output / "eval-v3-task-results.jsonl").is_file())
            self.assertTrue((output / "eval-v3-summary.md").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
