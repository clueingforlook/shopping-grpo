"""Unit tests for the project-side reward-group filter."""

import unittest

from shopping_grpo.training.grpo.dynamic_sampling import (
    aggregate_shopping_metrics,
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


class RewardGroupSelectionTest(unittest.TestCase):
    def test_all_zero_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(["a"] * 4, [0, 0, 0, 0])
        self.assertEqual(indices, [])
        self.assertEqual(stats["dropped_uids"], ("a",))
        self.assertEqual(stats["all_zero_utility_group_count"], 1)
        self.assertEqual(stats["all_purchase_success_group_count"], 0)

    def test_all_one_group_is_dropped(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [1, 1, 1, 1],
            terminal_utilities=[1.0, 1.0, 1.0, 1.0],
            purchase_success=[True] * 4,
        )
        self.assertEqual(indices, [])
        self.assertEqual(stats["kept_group_count"], 0)
        self.assertEqual(stats["all_purchase_success_group_count"], 1)

    def test_fractional_reward_variance_is_kept(self):
        rewards = [2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(["a"] * 4, rewards)
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(stats["kept_uids"], ("a",))

    def test_mixed_uids_preserve_trajectory_indices(self):
        uids = ["a", "b", "a", "b", "a", "b", "a", "b"]
        rewards = [0, 2 / 7, 0, 4 / 7, 0, 2 / 7, 0, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [1, 3, 5, 7])
        self.assertEqual(stats["kept_uids"], ("b",))
        self.assertEqual(stats["dropped_uids"], ("a",))

    def test_zero_and_varying_groups_keep_only_varying_group(self):
        uids = ["zero"] * 4 + ["signal"] * 4
        rewards = [0, 0, 0, 0, 2 / 7, 4 / 7, 2 / 7, 2 / 7]
        indices, stats = select_reward_varying_groups(uids, rewards)
        self.assertEqual(indices, [4, 5, 6, 7])
        self.assertEqual(stats["kept_group_count"], 1)
        self.assertEqual(stats["dropped_group_count"], 1)

    def test_tolerance_treats_tiny_roundoff_as_constant(self):
        indices, _ = select_reward_varying_groups(
            ["a"] * 4,
            [0.5, 0.5 + 1.0e-9, 0.5, 0.5],
            tolerance=1.0e-8,
        )
        self.assertEqual(indices, [])

    def test_varying_terminal_utility_is_kept_without_purchase_success(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.85, -0.65, -0.50, -0.35],
            terminal_utilities=[-0.85, -0.65, -0.50, -0.35],
            purchase_success=[False] * 4,
            sampling_invalid=[False] * 4,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])
        self.assertEqual(stats["no_purchase_success_group_count"], 1)

    def test_varying_group_with_purchase_success_is_kept(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [-0.5, 0.55, -0.5, -0.5],
            terminal_utilities=[-0.5, 0.55, -0.5, -0.5],
            purchase_success=[False, True, False, False],
            sampling_invalid=[False] * 4,
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])

    def test_sampling_invalid_member_is_ignored_when_two_valid_members_vary(self):
        indices, stats = select_reward_varying_groups(
            ["a"] * 4,
            [0.0, 99.0, 0.5, 0.0],
            terminal_utilities=[0.0, 99.0, 0.5, 0.0],
            purchase_success=[False, True, False, False],
            sampling_invalid=[False, True, False, False],
            sampling_invalid_reasons=[(), ("infrastructure_invalid",), (), ()],
        )

        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertIsNone(stats["groups"][0]["drop_reason"])
        self.assertEqual(stats["sampling_invalid_group_count"], 1)
        self.assertEqual(
            stats["sampling_invalid_reason_counts"]["infrastructure_invalid"],
            1,
        )

    def test_shopping_extra_fields_are_reduced_to_filter_signals(self):
        utility, success, invalid, reasons = extract_shopping_group_signals(
            [
                {
                    "infrastructure_invalid": False,
                    "reward": {
                        "terminal_utility": 0.55,
                        "gold": 1.0,
                        "sampling_invalid": False,
                    },
                },
                {
                    "infrastructure_invalid": True,
                    "reward": {
                        "terminal_utility": 0.0,
                        "gold": 0.0,
                        "sampling_invalid": True,
                    },
                },
            ]
        )

        self.assertEqual(utility, [0.55, 0.0])
        self.assertEqual(success, [True, False])
        self.assertEqual(invalid, [False, True])
        self.assertEqual(reasons, [(), ("infrastructure_invalid",)])

    def test_unverifiable_reward_is_sampling_invalid_but_not_infrastructure(self):
        utility, success, invalid, reasons = extract_shopping_group_signals(
            [
                {
                    "infrastructure_invalid": False,
                    "reward_unverifiable": True,
                    "reward": {
                        "terminal_utility": 0.0,
                        "gold": 0.0,
                        "sampling_invalid": True,
                        "outcome": "reward_unverifiable",
                    },
                }
            ]
        )
        self.assertEqual(utility, [0.0])
        self.assertEqual(success, [False])
        self.assertEqual(invalid, [True])
        self.assertEqual(reasons, [("reward_unverifiable",)])

    def test_missing_shopping_filter_signal_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "shopping"):
            extract_shopping_group_signals([None])

    def test_shopping_metrics_are_aggregated_for_a0_and_a1(self):
        infos = [
            {
                "steps": 10,
                "done": True,
                "termination_reason": "environment_done",
                "infrastructure_invalid": False,
                "reward": {
                    "version": "wlx-reward-v4",
                    "terminal_utility": 1.0,
                    "gold": 1.0,
                    "partial_purchase": 0.0,
                    "no_purchase": 0.0,
                    "wrong_category": 0.0,
                    "sampling_invalid": False,
                    "attribute_ratio": 1.0,
                    "option_ratio": 1.0,
                    "price_pass": 1.0,
                },
                "prm": {"step_kinds": ["normal"]},
            },
            {
                "steps": 35,
                "done": False,
                "termination_reason": "max_steps",
                "infrastructure_invalid": False,
                "reward": {
                    "version": "wlx-reward-v4",
                    "terminal_utility": -0.25,
                    "gold": 0.0,
                    "partial_purchase": 0.0,
                    "no_purchase": 1.0,
                    "wrong_category": 0.0,
                    "sampling_invalid": False,
                    "attribute_ratio": 0.0,
                    "option_ratio": 0.0,
                    "price_pass": 0.0,
                },
                "prm": {"step_kinds": ["no_progress"]},
            },
        ]

        metrics = aggregate_shopping_metrics(infos)

        self.assertEqual(metrics["wlx_reward/gold_rate"], 0.5)
        self.assertEqual(metrics["wlx_reward/orm_min"], -0.25)
        self.assertEqual(metrics["wlx_reward/orm_max"], 1.0)
        self.assertEqual(metrics["wlx_reward/no_purchase_rate"], 0.5)
        self.assertEqual(metrics["wlx_prm/no_progress_per_trajectory"], 0.5)
        self.assertEqual(metrics["trajectory/average_steps"], 22.5)
        self.assertEqual(metrics["trajectory/done_rate"], 0.5)
        self.assertEqual(metrics["trajectory/max_steps_rate"], 0.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
