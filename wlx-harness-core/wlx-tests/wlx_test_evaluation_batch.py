"""WLX 单模型评测批次测试；不连接模型或 ShopSimulator。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core.wlx_config import HarnessConfig  # noqa: E402
from wlx_harness_core.wlx_evaluation_batch import (  # noqa: E402
    EvaluationBatchContractError,
    EvaluationBatchSafetyPause,
    run_evaluation_batch,
)


def _tool_call(name: str, arguments: str = "{}") -> dict:
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _trajectory(task_id: int, *, release_error=None) -> dict:
    return {
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("buy_now")],
            }
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "goal": {
                "asin": f"asin-{task_id}",
                "category": "杯具›保温杯",
                "expected_core_functions": ["保温"],
                "required_options_by_key": {},
                "expected_brand": [],
                "expected_model": [],
                "price_upper": 50,
            },
            "purchase": {
                "asin": f"asin-{task_id}",
                "category": "杯具›保温杯",
                "name": "保温杯",
                "attributes": ["保温"],
                "options": {},
                "price": 39,
            },
        },
        "infrastructure_invalid": False,
        "release_error": release_error,
        "error": None,
        "tool_call_truncations": [],
    }


class FakeRunner:
    def __init__(self, *, unsafe_task_id: int | None = None):
        self.calls = []
        self.unsafe_task_id = unsafe_task_id

    async def run(self, request, policy, config):
        del policy, config
        self.calls.append(request.task_id)
        return _trajectory(
            request.task_id,
            release_error=(
                {"category": "release", "message": "uncertain"}
                if request.task_id == self.unsafe_task_id
                else None
            ),
        )


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class EvaluationBatchTest(unittest.TestCase):
    def test_one_model_run_writes_per_task_outputs_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "wlx-base-run"
            runner = FakeRunner()
            progress_events = []
            first = asyncio.run(
                run_evaluation_batch(
                    task_ids=[11, 22],
                    output_dir=output,
                    runner=runner,
                    harness_config=HarnessConfig(max_steps=2),
                    policy_factory=lambda task_id: {"task_id": task_id},
                    run_contract={"model_role": "base", "served_model": "base"},
                    concurrency=2,
                    progress_callback=progress_events.append,
                )
            )
            second = asyncio.run(
                run_evaluation_batch(
                    task_ids=[11, 22],
                    output_dir=output,
                    runner=runner,
                    harness_config=HarnessConfig(max_steps=2),
                    policy_factory=lambda task_id: {"task_id": task_id},
                    run_contract={"model_role": "base", "served_model": "base"},
                    concurrency=2,
                )
            )

            trajectories = _rows(output / "wlx-trajectories.jsonl")
            results = _rows(output / "wlx-task-results.jsonl")
            manifest = json.loads((output / "wlx-run-manifest.json").read_text())

        self.assertEqual(sorted(runner.calls), [11, 22])
        self.assertEqual(first["completed_after_run"], 2)
        self.assertEqual(second["new_trajectories"], 0)
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(row["format_correct"] for row in results))
        self.assertTrue(all(row["purchase_correct"] for row in results))
        self.assertEqual(manifest["run_contract"]["task_count"], 2)
        self.assertEqual(progress_events[0]["event"], "start")
        self.assertEqual(progress_events[0]["completed"], 0)
        self.assertEqual(progress_events[-1]["completed"], 2)
        self.assertEqual(progress_events[-1]["format_correct"], 2)
        self.assertEqual(progress_events[-1]["purchase_correct"], 2)

    def test_existing_trajectory_is_scored_without_second_rollout(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "wlx-resume-run"
            first_runner = FakeRunner()
            asyncio.run(
                run_evaluation_batch(
                    task_ids=[7],
                    output_dir=output,
                    runner=first_runner,
                    harness_config=HarnessConfig(max_steps=2),
                    policy_factory=lambda task_id: object(),
                    run_contract={"model_role": "sft"},
                )
            )
            (output / "wlx-task-results.jsonl").write_text("", encoding="utf-8")
            resumed_runner = FakeRunner()
            result = asyncio.run(
                run_evaluation_batch(
                    task_ids=[7],
                    output_dir=output,
                    runner=resumed_runner,
                    harness_config=HarnessConfig(max_steps=2),
                    policy_factory=lambda task_id: object(),
                    run_contract={"model_role": "sft"},
                )
            )

            rows = _rows(output / "wlx-task-results.jsonl")

        self.assertEqual(resumed_runner.calls, [])
        self.assertEqual(result["reconciled_results"], 1)
        self.assertEqual(len(rows), 1)

    def test_incompatible_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "wlx-contract-run"
            asyncio.run(
                run_evaluation_batch(
                    task_ids=[1],
                    output_dir=output,
                    runner=FakeRunner(),
                    harness_config=HarnessConfig(max_steps=2),
                    policy_factory=lambda task_id: object(),
                    run_contract={"served_model": "base"},
                )
            )
            with self.assertRaises(EvaluationBatchContractError):
                asyncio.run(
                    run_evaluation_batch(
                        task_ids=[1],
                        output_dir=output,
                        runner=FakeRunner(),
                        harness_config=HarnessConfig(max_steps=2),
                        policy_factory=lambda task_id: object(),
                        run_contract={"served_model": "sft"},
                    )
                )

    def test_uncertain_release_pauses_before_next_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(unsafe_task_id=1)
            with self.assertRaises(EvaluationBatchSafetyPause):
                asyncio.run(
                    run_evaluation_batch(
                        task_ids=[1, 2],
                        output_dir=Path(temporary) / "wlx-unsafe-run",
                        runner=runner,
                        harness_config=HarnessConfig(max_steps=2),
                        policy_factory=lambda task_id: object(),
                        run_contract={"served_model": "base"},
                        concurrency=1,
                    )
                )

        self.assertEqual(runner.calls, [1])


if __name__ == "__main__":
    unittest.main()
