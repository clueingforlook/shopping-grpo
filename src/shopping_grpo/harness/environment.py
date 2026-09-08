"""把 ShopSimulator 包装成 Harness 可以统一调用的环境接口。

这个适配器一次只管理一个购物环境。它把原来会阻塞程序的同步接口包装成
异步调用，并且把“用户要买什么”和“智能体当前看到的页面”分开保存，避免
工具检查时把任务说明误当成网页内容。
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from shopping_grpo.environment.client import ShopAgentEnv, ShopProtocolError
from shopping_grpo.environment.observation import (
    OBSERVATION_VERSION,
    render_structured_observation,
)


@dataclass(frozen=True)
class EnvironmentResult:
    """保存环境重置或执行一步之后返回的完整结果。

    ``instruction`` 是用户任务，``observation`` 是智能体能看到的页面，
    ``raw_result`` 则留作排查问题和审计使用。把三者分开后，各阶段不会拿错数据。
    """

    observation: str
    raw_result: dict[str, Any]
    instruction: str = ""
    reward: float = 0.0
    done: bool = False
    over: bool = False
    environment_version: str | None = None
    observation_version: str | None = None

    def audit_dict(self) -> dict[str, Any]:
        """返回一份可供审计的原始结果副本，防止外部代码改坏内部记录。"""

        return deepcopy(self.raw_result)


class ShopSimulatorEnvironmentAdapter:
    """用异步、单环境的方式包装现有 ``ShopAgentEnv``。

    ShopSimulator 目前没有提供能唯一标记一次租用的令牌。因此，释放环境失败后
    这里不会再次通过网络重试：旧编号可能已经被别人重新使用，再释放一次反而可能
    关掉别人的环境。失败信息会保留下来供运行器记录，而当前适配器也不再继续复用。
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:5700",
        timeout_s: float = 60,
        required_environment_version: str | None = None,
        env_factory: Callable[..., Any] = ShopAgentEnv,
        call_in_thread: bool = True,
    ) -> None:
        """保存连接参数并准备环境适配器，但此时还不会真的申请环境。

        延迟到 ``start`` 再申请，能让一次任务的申请、使用和释放边界更清楚，也方便
        测试时传入假的环境工厂。
        """

        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.required_environment_version = required_environment_version
        self.env_factory = env_factory
        self.call_in_thread = bool(call_in_thread)
        self._env: Any | None = None
        self._pending_release_error: BaseException | None = None
        self._release_uncertain = False

    @property
    def active(self) -> bool:
        """告诉调用方当前是否正持有一个尚未释放的购物环境。"""

        return self._env is not None

    @property
    def release_uncertain(self) -> bool:
        """告诉调用方上次释放是否结果不确定，避免不安全地复用这个适配器。"""

        return self._release_uncertain

    async def start(self, request: object) -> EnvironmentResult:
        """根据任务编号申请一个环境，并返回任务说明和第一个可见页面。

        这里还会检查环境版本，防止训练或评测在不兼容的模拟器版本上悄悄运行。
        """

        if self._env is not None:
            raise RuntimeError("ShopSimulator environment adapter is already active")
        if self._release_uncertain:
            raise RuntimeError(
                "ShopSimulator adapter cannot be reused after an uncertain release"
            )
        task_id = _task_id(request)
        self._env = self.env_factory(base_url=self.base_url, timeout=self.timeout_s)
        try:
            raw = await self._invoke(self._env.reset, task_id)
            result = _environment_result(raw)
            required = self.required_environment_version
            if required is not None and result.environment_version != required:
                raise ShopProtocolError(
                    "ShopSimulator environment version mismatch: "
                    f"expected {required!r}, got {result.environment_version!r}"
                )
            return result
        except BaseException:
            try:
                await self.close()
            except BaseException:
                # 运行器最后会读取并上报已经保存的释放错误，不会再发第二次网络请求。
                pass
            raise

    async def execute(self, action: str) -> EnvironmentResult:
        """在当前环境里执行一个已经通过检查的动作，并返回执行结果。

        Harness 的工具层先负责校验动作，这一层只负责把合法动作交给模拟器。
        """

        if self._env is None:
            raise RuntimeError("ShopSimulator environment adapter has not been started")
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        raw = await self._invoke(self._env.step, action)
        return _environment_result(raw)

    async def close(self) -> None:
        """尝试释放当前环境，并把释放失败明确交给上层处理。

        为避免误释放后来复用同一编号的新环境，网络释放只尝试一次，不会自动重试。
        """

        env = self._env
        if env is None:
            if self._pending_release_error is not None:
                error = self._pending_release_error
                self._pending_release_error = None
                raise error
            return
        self._env = None
        try:
            await self._invoke(env.release)
        except BaseException as exc:
            self._release_uncertain = True
            self._pending_release_error = exc
            raise

    async def _invoke(self, function: Callable[..., Any], *args: object) -> Any:
        """调用旧的同步环境函数，必要时放到线程里，避免卡住异步运行器。"""

        if self.call_in_thread:
            return await asyncio.to_thread(function, *args)
        return function(*args)


def _task_id(request: object) -> int:
    """从字典或请求对象中取出任务编号，并统一转换成整数。"""

    if isinstance(request, Mapping):
        value = request.get("task_id")
    else:
        value = getattr(request, "task_id", None)
    if value is None:
        raise ValueError("episode request is missing task_id")
    return int(value)


def _environment_result(raw: object) -> EnvironmentResult:
    """把 ShopSimulator 的原始返回值整理成统一的 ``EnvironmentResult``。

    统一格式后，SFT、GRPO 和评测阶段不需要分别猜测环境字段，也能明确区分
    用户任务、可见页面与只用于审计的原始信息。
    """

    if not isinstance(raw, Mapping):
        raise TypeError("ShopSimulator result must be an object")
    copied = deepcopy(dict(raw))
    instruction = str(copied.get("instruction") or "")
    observation_state = copied.get("observation_state")
    if observation_state is not None:
        observation = render_structured_observation(observation_state)
        observation_version = OBSERVATION_VERSION
    else:
        observation = str(copied.get("observation", instruction))
        observation_version = None
    return EnvironmentResult(
        observation=observation,
        instruction=instruction,
        raw_result=copied,
        reward=float(copied.get("reward", 0.0) or 0.0),
        done=bool(copied.get("done", False)),
        over=bool(copied.get("over", False)),
        environment_version=(
            str(copied["environment_version"])
            if copied.get("environment_version") is not None
            else None
        ),
        observation_version=observation_version,
    )


__all__ = ["EnvironmentResult", "ShopSimulatorEnvironmentAdapter"]
