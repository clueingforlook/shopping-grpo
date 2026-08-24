"""离线检查环境租约适配器和旧版聊天客户端适配器。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core.wlx_contracts import EpisodeRequest, ModelRequest  # noqa: E402
from wlx_harness_core.wlx_environment import (  # noqa: E402
    ShopSimulatorEnvironmentAdapter,
)
from wlx_harness_core.wlx_policy import LegacyChatPolicyAdapter  # noqa: E402


class FakeSyncEnv:
    """模拟一个同步购物环境，用来记录动作和释放次数，不访问真实服务。"""

    instances = []

    def __init__(self, **kwargs):
        """创建假环境，保存传入配置，并准备记录动作和释放次数。"""

        self.kwargs = kwargs
        self.released = 0
        self.actions = []
        self.__class__.instances.append(self)

    def reset(self, task_id):
        """模拟重置环境；输入任务编号，返回任务说明和环境版本。"""

        return {
            "instruction": f"task {task_id}",
            "environment_version": "shopsimulator-environment-v2.1",
        }

    def step(self, action):
        """模拟执行环境动作；记录动作并返回一个尚未结束的新页面。"""

        self.actions.append(action)
        return {"instruction": "next", "reward": 0.0, "done": False}

    def release(self):
        """模拟释放环境租约，每调用一次就把释放次数加一。"""

        self.released += 1


class FakeClient:
    """模拟项目原来的模型客户端，让测试不需要真实模型和密钥。"""

    def __init__(self):
        """创建假客户端，并预先放入上下文统计结果。"""

        self.last_context_tokens = 123
        self.last_context_event = {"removed_groups": 1}

    def complete(self, messages, tools):
        """模拟模型回答；接收消息和工具，固定返回一条结束文字。"""

        return {"role": "assistant", "content": "done"}

    def project_observation(self, tool_name, observation, parameters):
        """模拟页面裁剪；返回前四个字符以及对应的裁剪统计。"""

        return observation[:4], {"raw_tokens": 10, "visible_tokens": 4, "truncated": True}


class EnvironmentAdapterTest(unittest.IsolatedAsyncioTestCase):
    """检查 ShopSimulator 适配器是否正确管理一个环境租约。"""

    async def test_adapter_owns_and_releases_exactly_one_lease(self):
        """验证环境只创建一次、动作能执行，而且重复关闭也只释放一次。"""

        FakeSyncEnv.instances.clear()
        adapter = ShopSimulatorEnvironmentAdapter(
            required_environment_version="shopsimulator-environment-v2.1",
            env_factory=FakeSyncEnv,
            call_in_thread=False,
        )

        initial = await adapter.start(EpisodeRequest(task_id=3))
        step = await adapter.execute("search[x]")
        await adapter.close()
        await adapter.close()

        env = FakeSyncEnv.instances[-1]
        self.assertEqual(initial.observation, "task 3")
        self.assertEqual(step.observation, "next")
        self.assertEqual(env.actions, ["search[x]"])
        self.assertEqual(env.released, 1)

    async def test_version_mismatch_releases_before_raising(self):
        """验证环境版本不符合要求时，会先释放租约再把错误抛出。"""

        FakeSyncEnv.instances.clear()
        adapter = ShopSimulatorEnvironmentAdapter(
            required_environment_version="different-version",
            env_factory=FakeSyncEnv,
            call_in_thread=False,
        )

        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            await adapter.start(EpisodeRequest(task_id=4))
        self.assertEqual(FakeSyncEnv.instances[-1].released, 1)


class PolicyAdapterTest(unittest.IsolatedAsyncioTestCase):
    """检查旧版聊天客户端经过适配后能否按统一接口工作。"""

    async def test_legacy_client_is_adapted_without_exposing_credentials(self):
        """验证模型回答、上下文统计和页面裁剪可用，且不会暴露凭据。"""

        adapter = LegacyChatPolicyAdapter(FakeClient(), call_in_thread=False)
        response = await adapter.generate(
            ModelRequest(messages=({"role": "user", "content": "x"},), tools=())
        )
        visible, metadata = await adapter.project_observation("search_products", "abcdefgh", {})

        self.assertEqual(response["content"], "done")
        self.assertEqual(adapter.last_context_tokens, 123)
        self.assertEqual(visible, "abcd")
        self.assertTrue(metadata["truncated"])


if __name__ == "__main__":
    unittest.main()
