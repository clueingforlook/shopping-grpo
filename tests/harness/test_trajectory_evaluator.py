"""轨迹评测的独立测试，不导入原项目评测模块。"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "src", REPOSITORY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shopping_grpo.harness.config import HarnessConfig  # noqa: E402
from shopping_grpo.harness.contracts import EpisodeRequest  # noqa: E402
from shopping_grpo.harness.environment import EnvironmentResult  # noqa: E402
from shopping_grpo.harness.purchase_verifier import (  # noqa: E402
    GOLD_PURCHASE,
    UNSUCCESSFUL_PURCHASE,
    VALID_ALTERNATIVE_PURCHASE,
    requirements_from_environment_goal,
    verify_purchase,
)
from shopping_grpo.harness.runner import EpisodeRunner  # noqa: E402
from shopping_grpo.harness.trajectory_evaluator import (  # noqa: E402
    evaluate_format,
    evaluate_trajectory,
)


_SCRIPT_PATH = REPOSITORY / "scripts" / "evaluate_trajectories.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "evaluate_trajectories_script",
    _SCRIPT_PATH,
)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"无法加载 评测入口：{_SCRIPT_PATH}")
_SCRIPT_MODULE = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(_SCRIPT_MODULE)
evaluate_jsonl = _SCRIPT_MODULE.evaluate_jsonl


def _tool_call(name: str, arguments: object = "{}") -> dict:
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _goal() -> dict:
    return {
        "asin": "gold-1",
        "category": "家居用品›杯具›保温杯",
        "name": "目标商品中用户没要求的额外特征",
        "expected_core_functions": ["保温", "白色"],
        "required_options_by_key": {
            "color": {"value": "白色", "source": "instruction"}
        },
        "price_upper": 50,
        "expected_brand": [],
        "expected_model": [],
    }


def _purchase(
    asin: str,
    *,
    attributes: list[str] | None = None,
    name: str = "白色保温杯，含候选商品自己的其他特征",
) -> dict:
    return {
        "asin": asin,
        "category": "家居用品›杯具›保温杯",
        "name": name,
        "attributes": attributes if attributes is not None else ["保温", "白色", "防滑"],
        "options": {"颜色": "白色"},
        "price": 39,
    }


def _trajectory(*, messages=None, purchase=None, goal=None, **overrides) -> dict:
    row = {
        "trajectory_id": "trajectory-1",
        "task_id": 1,
        "messages": messages
        if messages is not None
        else [
            {"role": "user", "content": "请购买白色保温杯"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("search_products", '{"query":"白色保温杯"}')],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("buy_now")],
            },
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "goal": goal if goal is not None else _goal(),
            "purchase": purchase if purchase is not None else _purchase("gold-1"),
        },
        "infrastructure_invalid": False,
        "release_error": None,
        "error": None,
        "tool_call_truncations": [],
    }
    row.update(overrides)
    return row


class PurchaseVerifierTest(unittest.TestCase):
    def test_gold_and_alternative_compare_only_allowlisted_requirements(self):
        requirements = requirements_from_environment_goal(_goal())
        gold = verify_purchase(requirements, _purchase("gold-1"))
        alternative = verify_purchase(requirements, _purchase("alternative-2"))

        self.assertTrue(gold.purchase_correct)
        self.assertEqual(gold.purchase_type, GOLD_PURCHASE)
        self.assertTrue(alternative.purchase_correct)
        self.assertEqual(alternative.purchase_type, VALID_ALTERNATIVE_PURCHASE)

    def test_missing_user_requirement_is_unsuccessful(self):
        requirements = requirements_from_environment_goal(_goal())
        result = verify_purchase(
            requirements,
            _purchase(
                "alternative-2",
                attributes=["白色"],
                name="白色普通杯子",
            ),
        )

        self.assertTrue(result.verifier_valid)
        self.assertFalse(result.purchase_correct)
        self.assertEqual(result.purchase_type, UNSUCCESSFUL_PURCHASE)

    def test_punctuation_and_frozen_brand_aliases_do_not_create_false_negative(self):
        goal = _goal()
        goal["expected_brand"] = ["Philips"]
        goal["required_options_by_key"] = {
            "model": {"value": "HX-12/Pro", "source": "instruction"}
        }
        purchase = _purchase("alternative-2", name="飞利浦 白色保温杯")
        purchase["options"] = {"型号": "HX 12|Pro"}

        result = verify_purchase(requirements_from_environment_goal(goal), purchase)

        self.assertTrue(result.purchase_correct)
        self.assertEqual(result.purchase_type, VALID_ALTERNATIVE_PURCHASE)


class FormatEvaluationTest(unittest.TestCase):
    def test_valid_tool_calls_pass(self):
        result = evaluate_format(_trajectory())
        self.assertTrue(result.format_correct)
        self.assertIsNone(result.first_format_error)

    def test_missing_assistant_turn_is_missing_tool_call(self):
        result = evaluate_format(
            _trajectory(messages=[{"role": "user", "content": "请购买白色保温杯"}])
        )

        self.assertFalse(result.format_correct)
        self.assertEqual(result.first_format_error["reason"], "missing_tool_call")

    def test_each_frozen_format_error_is_reported(self):
        cases = {
            "missing_tool_call": {"role": "assistant", "content": "完成"},
            "multiple_tool_calls": {
                "role": "assistant",
                "tool_calls": [_tool_call("buy_now"), _tool_call("buy_now")],
            },
            "malformed_arguments": {
                "role": "assistant",
                "tool_calls": [_tool_call("search_products", "{")],
            },
            "unknown_tool": {
                "role": "assistant",
                "tool_calls": [_tool_call("invented_tool")],
            },
            "schema_invalid": {
                "role": "assistant",
                "tool_calls": [_tool_call("search_products", '{"query":7}')],
            },
        }
        for expected, message in cases.items():
            with self.subTest(expected=expected):
                result = evaluate_format(_trajectory(messages=[message]))
                self.assertFalse(result.format_correct)
                self.assertEqual(result.first_format_error["reason"], expected)


class TrajectoryEvaluationTest(unittest.TestCase):
    def test_minimal_output_contains_the_three_frozen_decisions(self):
        result = evaluate_trajectory(
            _trajectory(purchase=_purchase("alternative-2"))
        ).to_dict()

        self.assertTrue(result["trajectory_valid"])
        self.assertTrue(result["format_correct"])
        self.assertTrue(result["purchase_correct"])
        self.assertEqual(result["purchase_type"], VALID_ALTERNATIVE_PURCHASE)

    def test_infrastructure_failure_is_not_counted_as_purchase_failure(self):
        result = evaluate_trajectory(
            _trajectory(infrastructure_invalid=True)
        ).to_dict()

        self.assertFalse(result["trajectory_valid"])
        self.assertIsNone(result["purchase_correct"])
        self.assertIsNone(result["purchase_type"])

    def test_jsonl_entrypoint_writes_one_result_per_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "trajectories.jsonl"
            output = root / "evaluation.jsonl"
            source.write_text(
                json.dumps(_trajectory(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            count = evaluate_jsonl(source, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["purchase_correct"])


class RunnerAuditTest(unittest.TestCase):
    def test_runner_preserves_malformed_assistant_turn_for_evaluation(self):
        class FakeEnvironment:
            async def start(self, request):
                return EnvironmentResult(
                    observation="搜索功能是否可用: True",
                    instruction="请购买保温杯",
                    raw_result={"instruction": "请购买保温杯"},
                )

            async def execute(self, action):  # pragma: no cover - 格式错误前不会调用
                raise AssertionError("environment action must not execute")

            async def close(self):
                return None

        class MalformedPolicy:
            async def generate(self, request):
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call("search_products", "{")],
                }

        runner = EpisodeRunner(environment_factory=lambda config: FakeEnvironment())
        trajectory = asyncio.run(
            runner.run(
                EpisodeRequest(task_id=1),
                MalformedPolicy(),
                HarnessConfig(max_steps=2, validate_terminal_reward=False),
            )
        )
        result = evaluate_format(trajectory)

        self.assertFalse(result.format_correct)
        self.assertEqual(
            result.first_format_error["reason"],
            "malformed_arguments",
        )


if __name__ == "__main__":
    unittest.main()
