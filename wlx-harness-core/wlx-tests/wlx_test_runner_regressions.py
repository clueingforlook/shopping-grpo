"""运行器关键安全边界的回归测试：防止后续改代码时重新引入已经修过的问题。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core import EpisodeRequest, EpisodeRunner, HarnessConfig  # noqa: E402
from wlx_harness_core.wlx_contracts import ErrorCategory  # noqa: E402
from wlx_harness_core.wlx_environment import (  # noqa: E402
    EnvironmentResult,
    ShopSimulatorEnvironmentAdapter,
)
from wlx_harness_core.wlx_sft_outcomes import classify_attempt  # noqa: E402


ASIN = "100000000001"


def assistant_tool(name: str, arguments: dict, call_id: str = "call") -> dict:
    """造一条调用指定商店工具的模型回复。"""
    return {
        "role": "assistant",
        "content": None,
        "reasoning_content": "private chain of thought",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def reward_detail(*, internally_consistent: bool = True) -> dict:
    """造一份测试奖励明细，并可指定它是否通过语义校验。"""
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "gold_purchase",
        "termination_reason": "gold_purchase",
        "reward_valid": True,
        "sampling_invalid": False,
        "purchase_success": internally_consistent,
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


class CapturingPolicy:
    """会保存请求的假模型，用来检查 Runner 到底给模型看了什么。"""
    def __init__(self, turns: list[dict]) -> None:
        """保存预设回复，并建立一个列表记录模型请求。"""
        self.turns = list(turns)
        self.requests = []

    async def generate(self, request):
        """返回预设回复，同时完整记录本次模型请求。"""
        self.requests.append(request)
        return self.turns.pop(0)


class StructuredSyncEnvironment:
    """同步风格的假环境，用来验证环境适配器能理解结构化结果。"""
    instances = []

    def __init__(self, **kwargs) -> None:
        """保存结构化的重置与执行结果，并准备释放标记。"""
        self.actions = []
        self.released = 0
        self.__class__.instances.append(self)

    def reset(self, task_id: int) -> dict:
        """模拟同步环境重置，返回结构化初始结果。"""
        return {
            "instruction": "Buy the requested latex pillow under budget.",
            "environment_version": "shopsimulator-environment-v2.1",
            "hidden_goal": "audit only",
            "observation_state": {
                "observation_version": "shopping-observation-v2",
                "page_type": "search_results",
                "search_available": False,
                "actions": ["back to search", ASIN],
                "query": "latex pillow",
                "normalized_query": "latex pillow",
                "page": 1,
                "total_pages": 1,
                "total_results": 1,
                "rank_start": 1,
                "rank_end": 1,
                "products": [
                    {
                        "rank": 1,
                        "asin": ASIN,
                        "title": "Latex pillow",
                        "brand": "Brand",
                        "category": "Pillow",
                        "price": 20,
                        "key_attributes": ["latex"],
                    }
                ],
            },
        }

    def step(self, action: str) -> dict:
        """模拟同步环境执行动作，返回结构化步骤结果。"""
        self.actions.append(action)
        detail = reward_detail()
        return {
            "instruction": "Environment terminated.",
            "reward": 1.0,
            "done": True,
            "over": True,
            "reward_detail": detail,
            "hidden_terminal_goal": "audit only",
        }

    def release(self) -> None:
        """模拟释放同步环境，并记录资源已经归还。"""
        self.released += 1


class OneStepEnvironment:
    """只允许执行一步的假环境，适合检查错误分类和公开事件。"""
    def __init__(self, detail: dict | None = None) -> None:
        """保存这一步将返回的环境结果，并准备调用记录。"""
        self.detail = detail or reward_detail()
        self.closed = False

    async def start(self, request) -> EnvironmentResult:
        """模拟启动一条只执行一步的购物任务。"""
        return EnvironmentResult(
            instruction="Buy a pillow.",
            observation="搜索功能是否可用: True\n\n可点击的按钮: []",
            raw_result={"instruction": "Buy a pillow.", "secret": "audit only"},
        )

    async def execute(self, action: str) -> EnvironmentResult:
        """记录收到的工具动作，然后返回预设的一步结果。"""
        terminal_reward = float(self.detail["terminal_utility"])
        raw = {
            "instruction": "Environment terminated.",
            "reward": terminal_reward,
            "done": True,
            "over": True,
            "reward_detail": self.detail,
            "secret_terminal": "audit only",
        }
        return EnvironmentResult(
            observation="Environment terminated.",
            raw_result=raw,
            reward=terminal_reward,
            done=True,
            over=True,
        )

    async def close(self) -> None:
        """记录 Runner 已经尝试释放这个假环境。"""
        self.closed = True


class RecordingObserver:
    """把 Runner 发出的公开事件存起来，供测试检查有没有泄露隐藏字段。"""
    def __init__(self) -> None:
        """准备三个列表，分别保存开始、步骤和结束事件。"""
        self.steps = []
        self.finished = None

    async def on_start(self, event) -> None:
        """记录任务开始时对外公开的事件。"""
        pass

    async def on_step(self, event) -> None:
        """记录每一步结束后对外公开的事件。"""
        self.steps.append(event)

    async def on_finish(self, event) -> None:
        """记录整条任务结束时对外公开的事件。"""
        self.finished = event


class RunnerRegressionTest(unittest.IsolatedAsyncioTestCase):
    """集中检查指令隔离、奖励可信度、错误分类和日志脱敏。"""
    async def test_instruction_and_guard_observation_are_separate(self):
        """验证任务指令和页面观察分开传递，页面文本不会被误当成用户要求。"""
        StructuredSyncEnvironment.instances.clear()
        adapter = ShopSimulatorEnvironmentAdapter(
            required_environment_version="shopsimulator-environment-v2.1",
            env_factory=StructuredSyncEnvironment,
            call_in_thread=False,
        )
        policy = CapturingPolicy([assistant_tool("open_product", {"asin": ASIN})])
        runner = EpisodeRunner(environment_factory=lambda config: adapter)

        trajectory = await runner.run(
            EpisodeRequest(task_id=1),
            policy,
            HarnessConfig(max_steps=2),
        )

        request_messages = policy.requests[0].messages
        user_messages = [item for item in request_messages if item["role"] == "user"]
        self.assertEqual(
            user_messages[-1]["content"],
            "Buy the requested latex pillow under budget.",
        )
        self.assertEqual(
            StructuredSyncEnvironment.instances[-1].actions,
            [f"click[{ASIN}]"],
        )
        self.assertEqual(trajectory.status, "done")

    async def test_semantically_invalid_reward_never_becomes_training_reward(self):
        """验证没通过语义校验的环境评分不能进入训练奖励。"""
        environment = OneStepEnvironment(
            reward_detail(internally_consistent=False)
        )
        runner = EpisodeRunner(environment_factory=lambda config: environment)

        trajectory = await runner.run(
            EpisodeRequest(task_id=2),
            CapturingPolicy([assistant_tool("search_products", {"query": "pillow"})]),
            HarnessConfig(max_steps=2),
        )

        self.assertEqual(trajectory.final_reward, 0.0)
        self.assertFalse(trajectory.reward_valid)
        self.assertTrue(trajectory.sampling_invalid)
        self.assertTrue(trajectory.infrastructure_invalid)
        self.assertEqual(trajectory.status, "error")

    async def test_repeat_loop_is_valid_model_failure_not_protocol_error(self):
        """环境判定重复循环时应保留负奖励并作为有效失败结束。"""

        detail = reward_detail()
        detail.update(
            {
                "reward_type": "repeat_loop",
                "termination_reason": "repeat_loop",
                "purchase_success": False,
                "target_asin_match": False,
                "terminal_utility": -0.65,
                "weighted_score": 0.0,
                "evidence_coverage": 0.0,
                "hard_gates": {},
                "dimension_scores": {},
            }
        )
        environment = OneStepEnvironment(detail)
        runner = EpisodeRunner(environment_factory=lambda config: environment)

        trajectory = await runner.run(
            EpisodeRequest(task_id=20),
            CapturingPolicy(
                [assistant_tool("search_products", {"query": "pillow"})]
            ),
            HarnessConfig(max_steps=2),
        )
        decision = classify_attempt(trajectory)

        self.assertEqual(trajectory.status, "done")
        self.assertEqual(trajectory.termination_reason, "repeat_loop")
        self.assertEqual(trajectory.final_reward, -0.65)
        self.assertTrue(trajectory.reward_valid)
        self.assertFalse(trajectory.sampling_invalid)
        self.assertFalse(trajectory.infrastructure_invalid)
        self.assertIsNone(trajectory.error)
        self.assertEqual(decision.outcome_type.value, "repeat_loop")
        self.assertTrue(decision.attempt_valid)
        self.assertFalse(decision.task_success)
        self.assertEqual(decision.sft_disposition.value, "rejected")

    async def test_unknown_tool_is_model_error_not_infrastructure_error(self):
        """验证模型调用不存在的工具时算模型错误，而不是服务器或环境故障。"""
        environment = OneStepEnvironment()
        runner = EpisodeRunner(environment_factory=lambda config: environment)

        trajectory = await runner.run(
            EpisodeRequest(task_id=3),
            CapturingPolicy([assistant_tool("unknown_tool", {})]),
            HarnessConfig(max_steps=2),
        )

        self.assertEqual(trajectory.error.category, ErrorCategory.MODEL)
        self.assertFalse(trajectory.infrastructure_invalid)
        self.assertEqual(trajectory.status, "invalid_action")

    async def test_public_observer_excludes_raw_and_private_fields(self):
        """验证公开观察事件会删除原始环境结果、隐藏奖励等私有字段。"""
        environment = OneStepEnvironment()
        observer = RecordingObserver()
        runner = EpisodeRunner(environment_factory=lambda config: environment)

        trajectory = await runner.run(
            EpisodeRequest(task_id=4),
            CapturingPolicy([assistant_tool("search_products", {"query": "pillow"})]),
            HarnessConfig(max_steps=2),
            observers=(observer,),
        )

        self.assertTrue(
            any("reasoning_content" in item for item in trajectory.messages)
        )
        serialized = json.dumps(observer.finished, ensure_ascii=False)
        for forbidden in (
            "reasoning_content",
            "raw_observation",
            "initial_result",
            "terminal_result",
            "audit_final_reward",
            "secret_terminal",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(observer.finished["final_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
