"""离线验证 SFT 和评测阶段适配器能兼容各种轨迹格式，并守住数据边界。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core.wlx_contracts import (  # noqa: E402
    TerminationCategory,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)
from wlx_harness_core.wlx_stage_evaluation import EvaluationStageAdapter  # noqa: E402
from wlx_harness_core.wlx_stage_sft import (  # noqa: E402
    SFTDataLeakError,
    SFTStageAdapter,
    build_sft_row,
)


HELD_OUT_TASK_ID = 8187
ALLOWED_TASK_ID = 999999


def _reward_detail() -> dict:
    """造一份字段完整、表示成功买到目标商品的 Reward-v3 奖励明细。"""

    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "gold_purchase",
        "termination_reason": "gold_purchase",
        "reward_valid": True,
        "sampling_invalid": False,
        "purchase_success": True,
        "target_asin_match": True,
        "terminal_utility": 1.0,
        "weighted_score": 1.0,
        "evidence_coverage": 1.0,
        "hard_gates": {
            name: {"status": "pass", "passed": True, "verifiable": True}
            for name in ("category", "budget")
        },
        "dimension_scores": {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
    }


def _trajectory(task_id: int = ALLOWED_TASK_ID) -> Trajectory:
    """造一条成功购买的完整轨迹，作为各类阶段适配测试的标准样本。

    默认使用不在评测保留集里的任务编号；测试需要验证数据隔离时，可以传入保留
    任务编号复用同一份轨迹结构。
    """

    call = ToolCall(call_id="buy", name="buy_now", arguments={})
    terminal = {
        "instruction": "Environment terminated.",
        "reward": 1.0,
        "done": True,
        "over": True,
        "purchase": {"asin": "100000000001"},
        "reward_detail": _reward_detail(),
    }
    previous = (
        "asin: 100000000001\n\n搜索功能是否可用: False\n\n"
        '可点击的按钮: ["back to search", "buy now"]'
    )
    return Trajectory(
        trajectory_id=f"trajectory-{task_id}",
        task_id=task_id,
        attempt_index=0,
        created_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        status="done",
        termination_category=TerminationCategory.ENVIRONMENT_DONE,
        termination_reason="gold_purchase",
        messages=(
            {"role": "user", "content": "Buy the target pillow."},
            {
                "role": "tool",
                "tool_call_id": "open",
                "name": "open_product",
                "content": previous,
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "must not enter SFT",
                "tool_calls": [call.to_openai_dict()],
            },
            {
                "role": "tool",
                "tool_call_id": "buy",
                "name": "buy_now",
                "content": "Environment terminated with private reward evidence.",
            },
        ),
        steps=(
            TrajectoryStep(
                step_index=0,
                tool_call=call,
                env_action="click[Buy Now]",
                observation="Environment terminated.",
                reward=1.0,
                done=True,
                result=terminal,
            ),
        ),
        initial_result={"instruction": "Buy the target pillow."},
        terminal_result=terminal,
        final_reward=1.0,
        done=True,
        reward_valid=True,
        sampling_invalid=False,
    )


class StageAdapterTest(unittest.TestCase):
    """检查 SFT 与评测阶段对轨迹格式、保留任务和批量构建的处理是否一致。"""

    def test_object_core_json_and_legacy_json_are_equivalent(self):
        """验证对象、Core 字典和旧字典三种输入会得到完全相同的阶段结果。"""

        trajectory = _trajectory()
        shapes = (
            trajectory,
            trajectory.to_dict(),
            trajectory.to_legacy_dict(),
        )

        sft_outputs = [SFTStageAdapter().prepare(item).to_dict() for item in shapes]
        evaluation_outputs = [
            EvaluationStageAdapter().prepare(item).to_dict() for item in shapes
        ]

        self.assertEqual(sft_outputs[0], sft_outputs[1])
        self.assertEqual(sft_outputs[1], sft_outputs[2])
        self.assertTrue(sft_outputs[0]["accepted"])
        self.assertEqual(evaluation_outputs[0], evaluation_outputs[1])
        self.assertEqual(evaluation_outputs[1], evaluation_outputs[2])
        self.assertEqual(
            evaluation_outputs[0]["deterministic_metrics"]["actions_and_efficiency"]
            ["buy_count"],
            1,
        )

    def test_canonical_held_out_gate_cannot_be_bypassed_by_input_shape(self):
        """验证无论轨迹用哪种格式传入，标准评测任务都不能混入 SFT 数据。"""

        trajectory = replace(
            _trajectory(),
            task_id=HELD_OUT_TASK_ID,
            trajectory_id="held-out",
        )
        shapes = (
            trajectory,
            trajectory.to_dict(),
            trajectory.to_legacy_dict(),
        )

        for item in shapes:
            with self.subTest(shape=type(item).__name__):
                output = SFTStageAdapter().prepare(item)
                self.assertFalse(output.accepted)
                self.assertEqual(output.rejection_reasons, ("held_out_task",))
                with self.assertRaises(SFTDataLeakError):
                    build_sft_row(item)

    def test_bulk_artifacts_always_exclude_canonical_held_out_tasks(self):
        """验证批量生成 SFT 文件时，也一定会排除标准评测任务。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            rows = [
                _trajectory().to_legacy_dict(),
                replace(
                    _trajectory(),
                    task_id=HELD_OUT_TASK_ID,
                    trajectory_id="held-out",
                ).to_legacy_dict(),
            ]
            raw.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = SFTStageAdapter().build_artifacts(
                raw_path=raw,
                output_dir=root / "output",
                validation_ratio=0.5,
            )

            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["held_out_excluded"], 1)

    def test_evaluation_summary_accepts_all_three_shapes(self):
        """验证评测汇总同样兼容对象、Core 字典和旧字典三种输入。"""

        trajectory = _trajectory()
        adapter = EvaluationStageAdapter()
        summaries = [
            adapter.summarize([ALLOWED_TASK_ID], [item])
            for item in (
                trajectory,
                trajectory.to_dict(),
                trajectory.to_legacy_dict(),
            )
        ]
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[1], summaries[2])


if __name__ == "__main__":
    unittest.main()
