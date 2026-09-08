"""离线检查最终 raw 合并与难度重标，不调用模型或环境。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY / "scripts" / "sft_prepare_final_input.py"
SPEC = importlib.util.spec_from_file_location("sft_prepare_final_input", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载 {SCRIPT_PATH}")
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _plan_row(task_id: int, label: str) -> dict:
    return {
        "task_id": task_id,
        "instruction": f"task {task_id}",
        "official_split": "train",
        "category": f"category-{task_id}",
        "difficulty_label": label,
        "difficulty_score": {"easy": 0.1, "medium": 0.5, "hard": 0.9}[label],
        "difficulty_version": "difficulty-final-test",
    }


def _raw_row(task_id: int, trajectory_id: str, label: str = "easy") -> dict:
    return {
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "stage_metadata": {
            "request": {
                "task_id": task_id,
                "difficulty_label": label,
                "difficulty_score": 0.01,
                "difficulty_version": "old-label",
                "category": "old-category",
                "official_split": "train",
            }
        },
    }


class PrepareFinalInputTest(unittest.TestCase):
    def test_copies_rows_and_applies_final_difficulty_without_touching_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_a = root / "raw-a.jsonl"
            raw_b = root / "raw-b.jsonl"
            plan = root / "plan.jsonl"
            held_out = root / "evaluation.jsonl"
            output = root / "output"
            rows_a = [_raw_row(1, "a0"), _raw_row(1, "a1")]
            rows_b = [_raw_row(2, "b0", label="hard")]
            _write_jsonl(raw_a, rows_a)
            _write_jsonl(raw_b, rows_b)
            original_a = raw_a.read_bytes()
            original_b = raw_b.read_bytes()
            _write_jsonl(plan, [_plan_row(1, "medium"), _plan_row(2, "hard")])
            _write_jsonl(held_out, [{"task_id": 99}])

            summary = PREPARE.prepare_final_input(
                raw_paths=[raw_a, raw_b],
                difficulty_plan_path=plan,
                held_out_tasks_path=held_out,
                output_dir=output,
            )

            self.assertEqual(summary["rows"], 3)
            self.assertEqual(summary["unique_tasks"], 2)
            self.assertEqual(summary["difficulty_distribution_by_task"], {"hard": 1, "medium": 1})
            self.assertEqual(raw_a.read_bytes(), original_a)
            self.assertEqual(raw_b.read_bytes(), original_b)
            merged = [
                json.loads(line)
                for line in (output / PREPARE.OUTPUT_RAW_NAME).read_text(encoding="utf-8").splitlines()
            ]
            request = merged[0]["stage_metadata"]["request"]
            self.assertEqual(request["difficulty_label"], "medium")
            self.assertEqual(request["difficulty_version"], "difficulty-final-test")
            self.assertEqual(request["category"], "category-1")
            manifest = json.loads(
                (output / PREPARE.OUTPUT_MANIFEST_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["held_out_overlap"], 0)
            self.assertEqual(manifest["unique_trajectories"], 3)

    def test_rejects_task_shared_by_two_raw_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_a = root / "raw-a.jsonl"
            raw_b = root / "raw-b.jsonl"
            plan = root / "plan.jsonl"
            held_out = root / "evaluation.jsonl"
            _write_jsonl(raw_a, [_raw_row(1, "a")])
            _write_jsonl(raw_b, [_raw_row(1, "b")])
            _write_jsonl(plan, [_plan_row(1, "easy")])
            _write_jsonl(held_out, [{"task_id": 99}])
            with self.assertRaisesRegex(ValueError, "同时出现在两份 raw"):
                PREPARE.prepare_final_input(
                    raw_paths=[raw_a, raw_b],
                    difficulty_plan_path=plan,
                    held_out_tasks_path=held_out,
                    output_dir=root / "output",
                )

    def test_rejects_held_out_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            plan = root / "plan.jsonl"
            held_out = root / "evaluation.jsonl"
            _write_jsonl(raw, [_raw_row(1, "a")])
            _write_jsonl(plan, [_plan_row(1, "easy")])
            _write_jsonl(held_out, [{"task_id": 1}])
            with self.assertRaisesRegex(ValueError, "held-out evaluation"):
                PREPARE.prepare_final_input(
                    raw_paths=[raw],
                    difficulty_plan_path=plan,
                    held_out_tasks_path=held_out,
                    output_dir=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
