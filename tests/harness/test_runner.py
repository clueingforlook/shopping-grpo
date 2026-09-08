"""运行器的离线端到端测试：不用真实模型和商店，也能检查一条购物轨迹是否正确串起来。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shopping_grpo.harness import (  # noqa: E402
    EpisodeRequest,
    EpisodeRunner,
    EvaluationStageAdapter,
    HarnessConfig,
    SFTStageAdapter,
)
from shopping_grpo.harness.environment import EnvironmentResult  # noqa: E402


def reward_detail() -> dict:
    """造一份测试用奖励明细，方便模拟 ShopSimulator 的评分结果。"""
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "gold_purchase",
        "termination_reason": "gold_purchase",
        "reward_valid": True,
        "sampling_invalid": False,
        "purchase_success": True,
        "terminal_utility": 1.0,
        "weighted_score": 1.0,
        "evidence_coverage": 1.0,
        "target_asin_match": True,
        "hard_gates": {
            "category": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
            },
            "budget": {
                "status": "pass",
                "passed": True,
                "verifiable": True,
            },
        },
        "dimension_scores": {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
    }


def assistant_tool(name: str, arguments: dict, call_id: str) -> dict:
    """造一条带工具调用的模型回复，方便测试模型要操作商店的情况。"""
    return {
        "role": "assistant",
        "content": None,
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


class FakeEnvironmentAdapter:
    """假的购物环境：按测试预先准备的顺序返回页面和奖励。"""
    def __init__(self, *, release_error: Exception | None = None) -> None:
        """保存测试要用的初始页面、每步结果，并准备调用记录。"""
        self.actions = []
        self.release_error = release_error
        self.closed = False

    async def start(self, request):
        """模拟启动购物任务，并返回预设的任务指令和初始页面。"""
        return EnvironmentResult(
            observation="Instruction: buy a latex pillow\n搜索功能是否可用: True",
            raw_result={
                "instruction": "Instruction: buy a latex pillow\n搜索功能是否可用: True",
                "environment_version": "shopsimulator-environment-v2.1",
                "hidden_goal": "must never reach the actor",
            },
            environment_version="shopsimulator-environment-v2.1",
        )

    async def execute(self, action):
        """模拟执行一个商店动作，记录动作后返回下一份预设结果。"""
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return EnvironmentResult(
                observation="1|100000000001|乳胶枕",
                raw_result={"instruction": "1|100000000001|乳胶枕", "reward": 0.0},
            )
        if action == "click[100000000001]":
            observation = 'detail\n\n可点击的按钮: ["Buy Now"]'
            return EnvironmentResult(
                observation=observation,
                raw_result={"instruction": observation, "reward": 0.0},
            )
        if action == "click[Buy Now]":
            detail = reward_detail()
            return EnvironmentResult(
                observation="Environment terminated.",
                raw_result={
                    "instruction": "Environment terminated.",
                    "reward": 1.0,
                    "done": True,
                    "over": True,
                    "purchase": {"asin": "100000000001"},
                    "reward_detail": detail,
                },
                reward=1.0,
                done=True,
                over=True,
            )
        raise AssertionError(f"unexpected action: {action}")

    async def close(self):
        """模拟归还购物环境，并记录释放动作确实发生过。"""
        self.closed = True
        if self.release_error is not None:
            raise self.release_error


class FakePolicy:
    """假的模型策略：按测试预先准备的顺序返回模型回复。"""
    def __init__(self, turns):
        """保存模型将依次给出的假回复，并准备请求记录。"""
        self.turns = list(turns)
        self.requests = []
        self.last_context_tokens = None
        self.last_context_event = None

    async def generate(self, request):
        """模拟模型生成下一步回复，同时记下 Runner 传来的请求。"""
        self.requests.append(request)
        return self.turns.pop(0)


class RunnerTest(unittest.IsolatedAsyncioTestCase):
    """检查运行器主流程、动作拦截和环境释放是否符合约定。"""
    async def test_successful_core_trajectory_feeds_sft_and_evaluation(self):
        """验证成功轨迹既能转换成 SFT 样本，也能交给评测阶段统计。"""
        environment = FakeEnvironmentAdapter()
        policy = FakePolicy(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "open"),
                assistant_tool("buy_now", {}, "buy"),
            ]
        )
        runner = EpisodeRunner(
            environment_factory=lambda config: environment,
            system_prompt="shopping system",
        )

        trajectory = await runner.run(
            EpisodeRequest(task_id=7),
            policy,
            HarnessConfig(max_steps=5),
        )

        self.assertEqual(trajectory.status, "done")
        self.assertTrue(trajectory.done)
        self.assertFalse(trajectory.infrastructure_invalid)
        self.assertEqual(
            environment.actions,
            ["search[乳胶枕]", "click[100000000001]", "click[Buy Now]"],
        )
        self.assertTrue(environment.closed)
        self.assertNotIn(
            "hidden_goal",
            json.dumps(policy.requests[0].to_dict(), ensure_ascii=False),
        )
        json.dumps(trajectory.to_dict(), ensure_ascii=False)

        sft = SFTStageAdapter().prepare(trajectory)
        self.assertTrue(sft.accepted, sft.rejection_reasons)
        self.assertEqual(sft.training_row["task_id"], 7)

        evaluation = EvaluationStageAdapter().prepare(trajectory)
        self.assertTrue(
            evaluation.deterministic_metrics["reward_and_outcome"][
                "strict_gold_success"
            ]
        )

    async def test_guard_rejections_terminate_without_touching_environment(self):
        """验证危险或不合规动作会被提前拦住，而且不会误操作购物环境。"""
        environment = FakeEnvironmentAdapter()
        invalid = assistant_tool(
            "open_product",
            {"asin": "100000000001"},
            "invalid",
        )
        policy = FakePolicy([invalid, invalid, invalid])
        runner = EpisodeRunner(environment_factory=lambda config: environment)

        trajectory = await runner.run(
            EpisodeRequest(task_id=8),
            policy,
            HarnessConfig(max_steps=5, max_guard_rejections=3),
        )

        self.assertEqual(trajectory.status, "invalid_action_limit")
        self.assertEqual(len(trajectory.blocked_tool_calls), 3)
        self.assertEqual(environment.actions, [])
        self.assertFalse(trajectory.infrastructure_invalid)
        self.assertTrue(environment.closed)

    async def test_parallel_tool_policy_is_explicit(self):
        """验证一次发多个工具动作时，Runner 会按照明确配置决定允许还是拒绝。"""
        environment = FakeEnvironmentAdapter()
        parallel = assistant_tool("search_products", {"query": "乳胶枕"}, "first")
        parallel["tool_calls"].append(
            assistant_tool("search_products", {"query": "枕头"}, "second")["tool_calls"][0]
        )
        runner = EpisodeRunner(environment_factory=lambda config: environment)

        trajectory = await runner.run(
            EpisodeRequest(task_id=9),
            FakePolicy([parallel]),
            HarnessConfig(max_steps=5, parallel_tool_call_policy="reject"),
        )

        self.assertEqual(trajectory.status, "parallel_tool_calls")
        self.assertEqual(environment.actions, [])

    async def test_release_failure_is_never_reported_as_model_reward(self):
        """验证环境释放失败只算基础设施问题，绝不会冒充模型应得的奖励。"""
        environment = FakeEnvironmentAdapter(release_error=OSError("release failed"))
        runner = EpisodeRunner(environment_factory=lambda config: environment)

        trajectory = await runner.run(
            EpisodeRequest(task_id=10),
            FakePolicy([{"role": "assistant", "content": "stop"}]),
            HarnessConfig(max_steps=2),
        )

        self.assertEqual(trajectory.status, "environment_release_failed")
        self.assertTrue(trajectory.infrastructure_invalid)
        self.assertIsNotNone(trajectory.release_error)


if __name__ == "__main__":
    unittest.main()
