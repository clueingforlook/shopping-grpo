"""检查 Core 明确指定的策略配置是否会在运行前得到严格执行。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core import EpisodeRequest, EpisodeRunner  # noqa: E402
from wlx_harness_core.wlx_config import (  # noqa: E402
    ConfigValidationError,
    ContextPolicy,
    HarnessConfig,
    ObservationPolicy,
)
from wlx_harness_core.wlx_policy import LegacyChatPolicyAdapter  # noqa: E402


class ConfigurableClient:
    """模拟一个配置完整的旧版客户端，用来测试各项参数能否对齐。"""

    token_counter = staticmethod(lambda messages, tools: len(messages))
    context_window = 1000
    max_tokens = 100
    context_safety_margin = 50
    context_compaction_enable = True
    observation_token_budget = 128
    observation_detail_token_budget = 256
    observation_generic_token_budget = 128
    observation_search_top_k = 10

    def complete(self, messages, tools):
        """模拟模型生成；接收消息和工具后固定返回停止回答。"""

        return {"role": "assistant", "content": "stop"}

    def project_observation(self, tool_name, observation, parameters):
        """模拟页面投影；原样返回页面，并表示没有额外统计信息。"""

        return observation, None


def configured_harness() -> HarnessConfig:
    """构造一份与假客户端参数完全一致的 Harness 配置。"""

    return HarnessConfig(
        max_steps=2,
        context=ContextPolicy(
            window_tokens=1000,
            generation_reserve_tokens=100,
            safety_margin_tokens=50,
            input_budget_tokens=850,
            compaction_enabled=True,
        ),
        observation=ObservationPolicy(
            token_budget=128,
            detail_token_budget=256,
            generic_token_budget=128,
            search_top_k=10,
        ),
    )


class PolicyConfigTest(unittest.IsolatedAsyncioTestCase):
    """检查策略适配器会接受匹配配置，并在不匹配时尽早报错。"""

    async def test_matching_legacy_client_proves_every_explicit_policy(self):
        """验证旧版客户端的每项参数都匹配时，配置检查可以通过。"""

        adapter = LegacyChatPolicyAdapter(ConfigurableClient(), call_in_thread=False)
        adapter.validate_harness_config(configured_harness())

    async def test_mismatch_fails_before_any_model_or_environment_call(self):
        """验证生成长度不匹配时，在模型或环境启动前就直接报错。"""

        client = ConfigurableClient()
        client.max_tokens = 99
        adapter = LegacyChatPolicyAdapter(client, call_in_thread=False)
        with self.assertRaisesRegex(ConfigValidationError, "max_tokens"):
            adapter.validate_harness_config(configured_harness())

    async def test_implicit_legacy_context_policy_is_rejected(self):
        """验证旧客户端暗中启用上下文策略，而 Core 未声明时会被拒绝。"""

        adapter = LegacyChatPolicyAdapter(ConfigurableClient(), call_in_thread=False)
        with self.assertRaisesRegex(ConfigValidationError, "context"):
            adapter.validate_harness_config(HarnessConfig(max_steps=2))

    async def test_custom_policy_must_acknowledge_explicit_core_policies(self):
        """验证自定义策略必须主动确认 Core 配置，否则环境不会启动。"""

        class BarePolicy:
            """模拟一个没有配置确认接口的自定义策略。"""

            async def generate(self, request):
                """如果错误地走到模型生成阶段，就立刻让测试失败。"""

                raise AssertionError("generation must not start")

        runner = EpisodeRunner(
            environment_factory=lambda config: (_ for _ in ()).throw(
                AssertionError("environment must not start")
            )
        )
        with self.assertRaisesRegex(ConfigValidationError, "validate_harness_config"):
            await runner.run(
                EpisodeRequest(task_id=1),
                BarePolicy(),
                configured_harness(),
            )


if __name__ == "__main__":
    unittest.main()
