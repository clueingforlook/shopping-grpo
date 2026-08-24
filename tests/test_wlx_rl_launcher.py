"""CPU-only safety checks for the WLX one-command launcher."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts.wlx_train_rl import (
    _TerminalOutput,
    _compact_sampling_line,
    _compact_step_line,
    _complete_checkpoints,
    _fingerprint,
    _check_or_write_manifest,
    _resume_path,
    parse_args,
)
from unittest.mock import patch


class WlxRlLauncherTest(unittest.TestCase):
    def test_terminal_compacts_step_metrics(self):
        line = (
            "\x1b[36m(TaskRunner pid=42)\x1b[0m step:151"
            " - group/effective_ratio:0.5"
            " - wlx_reward/orm_mean:0.375"
            " - wlx_reward/gold_rate:0.25"
            " - wlx_reward/partial_purchase_rate:0.5"
            " - wlx_reward/no_purchase_rate:0.25"
            " - actor/ppo_kl:np.float64(0.0042)"
            " - actor/grad_norm:np.float64(0.31)"
            " - actor/perf/max_memory_allocated_gb:np.float64(53.0)"
            " - training/global_step:151"
            " - training/optimizer_updated:1"
            " - response_length/mean:6400.0"
            " - timing_s/step:92.4\n"
        )
        self.assertEqual(
            _compact_step_line(line),
            "[step 151] | ORM 0.375 | Gold 25% | 部分购买 50% | 未购买 25% | "
            "有效组 50% | KL 0.0042 | 梯度 0.310 | 长度 6400 | 耗时 92s | "
            "显存 53.0GiB\n",
        )

    def test_terminal_hides_sampling_batch_but_summarizes_ready(self):
        output = _TerminalOutput()
        self.assertIsNone(
            output.render("SHOPPING_GRPO_DYNAMIC_SAMPLING_BATCH {\"groups\": []}\n")
        )
        ready = (
            "SHOPPING_GRPO_DYNAMIC_SAMPLING_READY "
            '{"generation_batches": 2, "trained_groups": 2, '
            '"generated_groups": 4, "generated_trajectories": 16, '
            '"filtered_groups": 2}\n'
        )
        self.assertEqual(
            _compact_sampling_line(ready),
            "[采样] 批次 2 | 训练组 2/4 | 轨迹 16 | 过滤组 2\n",
        )

    def test_terminal_keeps_full_error_after_traceback(self):
        output = _TerminalOutput()
        first = "Error executing job with overrides: []\n"
        detail = "Traceback (most recent call last):\n"
        self.assertEqual(output.render(first), first)
        self.assertEqual(output.render(detail), detail)

    def test_documented_mode_first_cli_parses_launcher_options(self):
        with patch(
            "sys.argv",
            [
                "wlx_train_rl.py",
                "train",
                "--run-name",
                "wlx-test",
                "--",
                "trainer.total_training_steps=4",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.run_name, "wlx-test")
        self.assertEqual(args.hydra_overrides, ["--", "trainer.total_training_steps=4"])

    def test_resume_uses_latest_complete_checkpoint_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for step in (25, 50):
                checkpoint = output / f"global_step_{step}"
                (checkpoint / "actor").mkdir(parents=True)
                (checkpoint / "actor/state.bin").write_bytes(b"actor")
                (checkpoint / "data.pt").write_bytes(b"data")
            (output / "global_step_75/actor").mkdir(parents=True)
            args = SimpleNamespace(mode="resume", resume_from=None)
            self.assertEqual(_resume_path(args, output).name, "global_step_50")
            self.assertEqual(
                [path.name for path in _complete_checkpoints(output)],
                ["global_step_25", "global_step_50"],
            )

    def test_fingerprint_ignores_only_resume_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weight")
            train = root / "train.parquet"
            validation = root / "validation.parquet"
            metadata = root / "metadata.json"
            for path in (train, validation, metadata):
                path.write_bytes(path.name.encode())
            fresh = _fingerprint(
                model,
                train,
                validation,
                metadata,
                ["trainer.resume_mode=disable", "trainer.resume_from_path=null", "actor.lr=1e-6"],
            )
            resumed = _fingerprint(
                model,
                train,
                validation,
                metadata,
                [
                    "trainer.resume_mode=resume_path",
                    "trainer.resume_from_path=/tmp/global_step_25",
                    "actor.lr=1e-6",
                ],
            )
            self.assertEqual(fresh, resumed)

    def test_resume_manifest_allows_only_continuation_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = {
                "schema_version": "wlx-rl-run-v1",
                "hydra_overrides": ["actor.lr=1e-6"],
                "model": {"name": "same"},
            }
            (output / "wlx-run-manifest.json").write_text(
                json.dumps(original), encoding="utf-8"
            )
            continued = {
                **original,
                "hydra_overrides": [
                    "actor.lr=1e-6",
                    "trainer.total_training_steps=500",
                    "trainer.max_actor_ckpt_to_keep=2",
                ],
            }
            _check_or_write_manifest(output, continued, resume=True)

            changed_training = {
                **continued,
                "hydra_overrides": [
                    *continued["hydra_overrides"],
                    "actor.lr=2e-6",
                ],
            }
            with self.assertRaisesRegex(SystemExit, "指纹发生变化"):
                _check_or_write_manifest(output, changed_training, resume=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
