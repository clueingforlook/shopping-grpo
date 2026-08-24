"""检查 ShopSimulator 只靠 env_idx 释放租约时的失败保护。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core import EpisodeRequest  # noqa: E402
from wlx_harness_core.wlx_environment import (  # noqa: E402
    ShopSimulatorEnvironmentAdapter,
)


class FailingReleaseEnvironment:
    """模拟释放响应丢失的环境，用来确认 Harness 不会危险地重复释放。"""

    instances = []

    def __init__(self, **kwargs) -> None:
        """创建假环境，并把释放尝试次数记为零。"""

        self.release_attempts = 0
        self.__class__.instances.append(self)

    def reset(self, task_id: int) -> dict:
        """模拟重置环境，并故意返回一个不符合要求的环境版本。"""

        return {
            "instruction": "task",
            "environment_version": "unexpected-version",
        }

    def release(self) -> None:
        """模拟释放请求已发出但响应丢失，每次调用都会抛出网络错误。"""

        self.release_attempts += 1
        raise OSError("release response lost")


class ReleaseSafetyTest(unittest.IsolatedAsyncioTestCase):
    """检查释放结果不确定后，适配器是否停止重试并禁止复用。"""

    async def test_failed_release_is_reported_without_network_retry(self):
        """验证释放只尝试一次，后续关闭不重试，环境也不能再次启动。"""

        FailingReleaseEnvironment.instances.clear()
        adapter = ShopSimulatorEnvironmentAdapter(
            required_environment_version="shopsimulator-environment-v2.1",
            env_factory=FailingReleaseEnvironment,
            call_in_thread=False,
        )

        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            await adapter.start(EpisodeRequest(task_id=1))
        environment = FailingReleaseEnvironment.instances[-1]
        self.assertEqual(environment.release_attempts, 1)
        self.assertTrue(adapter.release_uncertain)

        with self.assertRaisesRegex(OSError, "release response lost"):
            await adapter.close()
        self.assertEqual(environment.release_attempts, 1)
        await adapter.close()
        self.assertEqual(environment.release_attempts, 1)

        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            await adapter.start(EpisodeRequest(task_id=2))


if __name__ == "__main__":
    unittest.main()
