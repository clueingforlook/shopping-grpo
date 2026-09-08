"""验证环境协议出错时，Harness 不会把责任错误地算到模型头上。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shopping_grpo.harness import EpisodeRequest, EpisodeRunner, HarnessConfig  # noqa: E402
from shopping_grpo.harness.contracts import ErrorCategory  # noqa: E402
from shopping_grpo.harness.environment import EnvironmentResult  # noqa: E402


def search_call() -> dict:
    """造一条模型调用搜索工具的消息，供下面的假策略重复使用。"""

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "search",
                "type": "function",
                "function": {
                    "name": "search_products",
                    "arguments": json.dumps({"query": "pillow"}),
                },
            }
        ],
    }


class OneTurnPolicy:
    """假装一个只会发起一次商品搜索的简单模型策略。"""

    async def generate(self, request):
        """忽略请求内容，固定返回一条格式正确的搜索工具调用。"""

        return search_call()


class MalformedStartEnvironment:
    """假装一个在刚开始重置时就返回坏数据的环境。"""

    async def start(self, request):
        """用类型错误模拟 ShopSimulator 的 reset 返回格式不符合协议。"""

        raise TypeError("reset payload must be an object")

    async def close(self):
        """模拟关闭环境；这里没有实际资源需要释放。"""

        pass


class MalformedStepEnvironment:
    """假装一个能正常启动、但执行动作时返回坏数据的环境。"""

    async def start(self, request):
        """返回正常的初始任务和页面，让流程能够走到执行工具这一步。"""

        return EnvironmentResult(
            instruction="Buy a pillow.",
            observation="搜索功能是否可用: True\n\n可点击的按钮: []",
            raw_result={"instruction": "Buy a pillow."},
        )

    async def execute(self, action):
        """用类型错误模拟 ShopSimulator 的 step 返回格式不符合协议。"""

        raise TypeError("step payload must be an object")

    async def close(self):
        """模拟关闭环境；这里没有实际资源需要释放。"""

        pass


class EnvironmentErrorTest(unittest.IsolatedAsyncioTestCase):
    """检查环境错误在轨迹中会被归到协议或基础设施，而不是模型错误。"""

    async def test_malformed_reset_is_protocol_not_model_error(self):
        """验证 reset 数据损坏会被标为协议错误和无效基础设施样本。"""

        runner = EpisodeRunner(
            environment_factory=lambda config: MalformedStartEnvironment()
        )
        trajectory = await runner.run(
            EpisodeRequest(task_id=1),
            OneTurnPolicy(),
            HarnessConfig(max_steps=2),
        )
        self.assertEqual(trajectory.error.category, ErrorCategory.PROTOCOL)
        self.assertTrue(trajectory.infrastructure_invalid)

    async def test_malformed_step_is_protocol_not_model_error(self):
        """验证 step 数据损坏会同时写进整条轨迹和最后一步的协议错误。"""

        runner = EpisodeRunner(
            environment_factory=lambda config: MalformedStepEnvironment()
        )
        trajectory = await runner.run(
            EpisodeRequest(task_id=2),
            OneTurnPolicy(),
            HarnessConfig(max_steps=2),
        )
        self.assertEqual(trajectory.error.category, ErrorCategory.PROTOCOL)
        self.assertEqual(trajectory.steps[-1].error.category, ErrorCategory.PROTOCOL)
        self.assertTrue(trajectory.infrastructure_invalid)


if __name__ == "__main__":
    unittest.main()
