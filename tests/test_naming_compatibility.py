"""Naming changes must keep historical records readable and outputs usable."""

import argparse
from pathlib import Path
import tempfile
import unittest

from scripts.plot_grpo_metrics import parse_log
from scripts.sft_organize_data import _is_provenance_file
from scripts.train_rl import _run_name
from shopping_grpo.harness.serialization import (
    TrajectorySerializationError,
    trajectory_to_legacy,
)


class NamingCompatibilityTest(unittest.TestCase):
    def test_historical_training_and_validation_metrics_remain_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "step:250 - training/global_step:250 - wlx_reward/orm_mean:0.375"
                " - wlx_reward/gold_rate:0.25\n"
                "step:250 - val-core/shopsimulator/reward/mean@1:0.5"
                " - val-aux/shopsimulator/wlx_gold/mean@1:0.52"
                " - val-aux/shopsimulator/wlx_orm/mean@1:0.625\n",
                encoding="utf-8",
            )
            training, validation = parse_log(log)
        self.assertEqual(training[0]["gold"], 0.25)
        self.assertEqual(training[0]["orm"], 0.375)
        self.assertEqual(validation[0]["gold"], 0.52)
        self.assertEqual(validation[0]["orm"], 0.625)

    def test_unknown_historical_trajectory_schema_is_still_rejected(self):
        for schema in ("wlx-harness-trajectory-v999", "harness-trajectory-v999"):
            with self.subTest(schema=schema), self.assertRaises(TrajectorySerializationError):
                trajectory_to_legacy({"schema_version": schema})

    def test_run_names_allow_normal_names_without_allowing_paths(self):
        for name in ("rl-v4-main-24k-noentropy", "experiment_01", "step500"):
            self.assertEqual(_run_name(name), name)
        for name in ("", "../run", "run/name", "run\\name", ".", "..", "--run"):
            with self.subTest(name=name), self.assertRaises(argparse.ArgumentTypeError):
                _run_name(name)

    def test_provenance_selection_does_not_collect_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("difficulty-model.json", "formal-all-v3.jsonl", "wlx-old-plan.json"):
                path = root / name
                path.touch()
                self.assertTrue(_is_provenance_file(path))
            for name in ("notes.txt", "credentials.json", "difficulty-model.safetensors"):
                path = root / name
                path.touch()
                self.assertFalse(_is_provenance_file(path))
