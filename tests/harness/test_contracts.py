"""离线检查 Harness 的数据格式、配置、工具和奖励规则。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shopping_grpo.harness.config import (  # noqa: E402
    ConfigValidationError,
    ContextPolicy,
    HarnessConfig,
    ObservationPolicy,
)
from shopping_grpo.harness.contracts import (  # noqa: E402
    AssistantTurn,
    EpisodeRequest,
    ToolCall,
)
from shopping_grpo.harness.reward import (  # noqa: E402
    RewardContractError,
    validate_reward_v3,
)
from shopping_grpo.harness.tools import (  # noqa: E402
    SHOPPING_TOOL_REGISTRY,
    ToolRegistryError,
)


def reward_detail() -> dict:
    """构造一份合规的成功购物奖励，供后面的奖励测试重复使用。"""

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


class ContractTest(unittest.TestCase):
    """检查 Harness 对任务请求和模型工具调用的统一数据格式。"""

    def test_episode_request_reads_current_grpo_extra_info_shape(self):
        """验证 GRPO 现有字典里的任务编号能被正确读取并写成 JSON。"""

        request = EpisodeRequest.from_mapping(
            {"extra_info": {"interaction_kwargs": {"task_id": 17}}}
        )
        self.assertEqual(request.task_id, 17)
        json.dumps(request.to_dict())

    def test_openai_tool_call_round_trip_is_stable(self):
        """验证工具调用来回转换后内容不变，模型回答也能正常解析。"""

        raw = {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "search_products",
                "arguments": '{"query":"latex pillow"}',
            },
        }
        call = ToolCall.from_openai(raw)
        self.assertEqual(call.arguments, {"query": "latex pillow"})
        rebuilt = ToolCall.from_openai(call.to_openai_dict())
        self.assertEqual(rebuilt, call)

        turn = AssistantTurn.from_mapping(
            {"role": "assistant", "content": None, "tool_calls": [raw]}
        )
        self.assertEqual(turn.tool_calls, (call,))
        json.dumps(turn.to_message_dict())


class ConfigTest(unittest.TestCase):
    """检查上下文和页面裁剪配置的默认值与数字限制。"""

    def test_disabled_context_and_observation_preserve_collection_defaults(self):
        """验证不启用上下文和页面策略时，仍保留原采集流程的默认行为。"""

        config = HarnessConfig(max_steps=35)
        self.assertIsNone(config.context)
        self.assertIsNone(config.observation)
        self.assertEqual(config.parallel_tool_call_policy, "truncate")

    def test_context_and_observation_budgets_are_validated(self):
        """验证合理预算可以创建，太小或超出上下文的预算会立刻报错。"""

        config = HarnessConfig(
            max_steps=30,
            context=ContextPolicy(
                window_tokens=24576,
                input_budget_tokens=16384,
            ),
            observation=ObservationPolicy(token_budget=1536),
        )
        self.assertEqual(config.context.input_budget_tokens, 16384)
        with self.assertRaises(ConfigValidationError):
            ContextPolicy(
                window_tokens=1000,
                generation_reserve_tokens=900,
                safety_margin_tokens=200,
            )
        with self.assertRaises(ConfigValidationError):
            ObservationPolicy(token_budget=63)


class ToolRegistryTest(unittest.TestCase):
    """检查工具定义、页面 Guard 和环境动作转换是否绑在一起。"""

    def test_registry_binds_schema_guard_and_action_mapping(self):
        """验证搜索工具通过页面检查后，会变成环境认识的搜索动作。"""

        call = ToolCall(
            call_id="search",
            name="search_products",
            arguments={"query": "乳胶枕"},
        )
        action = SHOPPING_TOOL_REGISTRY.resolve(
            call,
            "搜索功能是否可用: True",
        )
        self.assertTrue(action.allowed)
        self.assertEqual(action.env_action, "search[乳胶枕]")
        self.assertEqual(len(SHOPPING_TOOL_REGISTRY.fingerprint), 64)

    def test_registry_detects_schema_drift(self):
        """验证工具说明被私自改动后，指纹比较能发现配置漂移。"""

        changed = SHOPPING_TOOL_REGISTRY.schemas
        changed[0]["function"]["description"] = "drift"
        with self.assertRaises(ToolRegistryError):
            SHOPPING_TOOL_REGISTRY.assert_compatible(changed)


class RewardTest(unittest.TestCase):
    """检查环境给出的最终奖励是否符合统一的 Reward v3 规则。"""

    def test_valid_reward_is_copied_and_invalid_reward_is_rejected(self):
        """验证合法奖励会安全复制，而标成无效采样的奖励会被拒绝。"""

        detail = reward_detail()
        detail["private_evidence"] = "audit only"
        validated = validate_reward_v3(detail, terminal_reward=1.0)
        self.assertEqual(validated["reward_type"], "gold_purchase")
        self.assertNotIn("private_evidence", validated)
        validated["reward_type"] = "changed"
        self.assertEqual(detail["reward_type"], "gold_purchase")

        broken = reward_detail()
        broken["sampling_invalid"] = True
        with self.assertRaises(RewardContractError):
            validate_reward_v3(broken, terminal_reward=1.0)

    def test_unscored_terminal_rewards_accept_empty_dimension_scores(self):
        """没有评价具体商品的正常终局不应被误判成环境协议故障。"""

        rewards = {
            "graceful_stop": -0.15,
            "early_abstain": -0.35,
            "max_steps": -0.50,
            "repeat_loop": -0.65,
        }
        for reward_type, terminal_reward in rewards.items():
            with self.subTest(reward_type=reward_type):
                detail = reward_detail()
                detail.update(
                    {
                        "reward_type": reward_type,
                        "termination_reason": reward_type,
                        "purchase_success": False,
                        "target_asin_match": False,
                        "terminal_utility": terminal_reward,
                        "weighted_score": 0.0,
                        "evidence_coverage": 0.0,
                        "hard_gates": {},
                        "dimension_scores": {},
                    }
                )

                validated = validate_reward_v3(
                    detail,
                    terminal_reward=terminal_reward,
                )

                self.assertEqual(validated["reward_type"], reward_type)
                self.assertEqual(
                    validated["dimension_scores"],
                    {
                        "brand": 0.0,
                        "model": 0.0,
                        "core_functions": 0.0,
                        "key_options": 0.0,
                    },
                )

    def test_scored_reward_still_requires_every_dimension(self):
        """购买类奖励仍必须提供四个完整商品维度，不能借修复绕过校验。"""

        broken = reward_detail()
        del broken["dimension_scores"]["brand"]
        with self.assertRaisesRegex(
            RewardContractError,
            "dimension_scores is missing 'brand'",
        ):
            validate_reward_v3(broken, terminal_reward=1.0)


if __name__ == "__main__":
    unittest.main()
