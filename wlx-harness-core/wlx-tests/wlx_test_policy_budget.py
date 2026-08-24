"""检查旧版策略适配器是否严格执行 Harness 的上下文输入预算。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core.wlx_config import ContextPolicy, HarnessConfig  # noqa: E402
from wlx_harness_core.wlx_contracts import ModelRequest  # noqa: E402
from wlx_harness_core.wlx_policy import LegacyChatPolicyAdapter  # noqa: E402


class CountingClient:
    """用消息条数当作 token 数的假客户端，方便直观看出是否做了压缩。"""

    context_window = 100
    max_tokens = 10
    context_safety_margin = 10
    context_compaction_enable = True
    observation_token_budget = None

    def __init__(self) -> None:
        """创建假客户端，并准备记录实际收到的消息和上下文统计。"""

        self.received_messages = None
        self.last_context_event = None
        self.last_context_tokens = None

    @staticmethod
    def token_counter(messages, tools) -> int:
        """把消息条数当作 token 数返回，让测试结果简单且可以预测。"""

        return len(messages)

    def complete(self, messages, tools):
        """记录适配器最终传来的消息，并固定返回停止回答。"""

        self.received_messages = list(messages)
        return {"role": "assistant", "content": "stop"}


class PolicyBudgetTest(unittest.IsolatedAsyncioTestCase):
    """检查模型调用前是否按照 Core 的更严格预算压缩旧消息。"""

    async def test_stricter_core_input_budget_compacts_before_client_call(self):
        """验证超预算时只删除旧工具交互，最近一组消息仍会保留。"""

        client = CountingClient()
        adapter = LegacyChatPolicyAdapter(client, call_in_thread=False)
        adapter.validate_harness_config(
            HarnessConfig(
                max_steps=2,
                context=ContextPolicy(
                    window_tokens=100,
                    generation_reserve_tokens=10,
                    safety_margin_tokens=10,
                    input_budget_tokens=4,
                    compaction_enabled=True,
                ),
            )
        )
        messages = (
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "old", "function": {"name": "think"}}],
            },
            {"role": "tool", "tool_call_id": "old", "content": "old"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "new", "function": {"name": "think"}}],
            },
            {"role": "tool", "tool_call_id": "new", "content": "new"},
        )

        await adapter.generate(ModelRequest(messages=messages, tools=()))

        self.assertEqual(len(client.received_messages), 4)
        self.assertEqual(client.received_messages[-1]["content"], "new")
        self.assertEqual(adapter.last_context_tokens, 6)
        self.assertEqual(adapter.last_context_event["removed_groups"], 1)


if __name__ == "__main__":
    unittest.main()
