"""离线锁定 WLX SFT 的 Prompt、工具和首轮 DeepSeek 请求契约。"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[2]
for import_path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402
from shopping_grpo.environment.observation import (  # noqa: E402
    OBSERVATION_VERSION,
    render_structured_observation,
)
SCRIPT_PATH = REPOSITORY / "scripts" / "wlx_sft_data_pipeline.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "wlx_sft_prompt_contract_test_cli",
    SCRIPT_PATH,
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"无法加载 {SCRIPT_PATH}")
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)
_assert_collection_config_compatible = SCRIPT_MODULE._assert_collection_config_compatible
from wlx_harness_core.wlx_contracts import EpisodeRequest, ErrorCategory  # noqa: E402
from wlx_harness_core.wlx_environment import EnvironmentResult  # noqa: E402
from wlx_harness_core.wlx_runner import EpisodeRunner  # noqa: E402
from wlx_harness_core.wlx_sft_pipeline import (  # noqa: E402
    WlxSftDataPipeline,
    default_sft_harness_config,
    sft_harness_contract,
    sft_harness_contract_fingerprint,
)
from wlx_harness_core.wlx_sft_outcomes import classify_attempt  # noqa: E402
from wlx_harness_core.wlx_sft_policy import (  # noqa: E402
    OpenAICompatibleTeacherPolicy,
    TeacherOutputTruncatedError,
)
from wlx_harness_core.wlx_sft_prompt import (  # noqa: E402
    WLX_SFT_SYSTEM_PROMPT,
    WLX_SFT_SYSTEM_PROMPT_SHA256,
    WLX_SFT_SYSTEM_PROMPT_VERSION,
)
from wlx_harness_core.wlx_tools import (  # noqa: E402
    SHOPPING_TOOL_REGISTRY,
    ToolRegistryError,
    WLX_SFT_TOOL_REGISTRY,
    WLX_SFT_TOOL_SCHEMAS,
)


EXPECTED_SFT_TOOLS = (
    "search_products",
    "open_product",
    "select_option",
    "view_description",
    "view_features",
    "view_reviews",
    "view_attributes",
    "next_page",
    "prev_page",
    "back_to_search",
    "buy_now",
    "finish_without_purchase",
)


async def _inline_to_thread(function, /, *args, **kwargs):
    """Run fake transports inline so offline tests do not leak executor threads."""

    return function(*args, **kwargs)


class FirstTurnEnvironment:
    """返回一个 answer-free 搜索首页，用于捕获第一轮模型 payload。"""

    def __init__(self) -> None:
        self.closed = False

    async def start(self, request: EpisodeRequest) -> EnvironmentResult:
        state = {
            "observation_version": OBSERVATION_VERSION,
            "page_type": "search_home",
            "search_available": True,
            "actions": [],
        }
        return EnvironmentResult(
            instruction="Buy a latex pillow under $50.",
            observation=render_structured_observation(state),
            raw_result={
                "instruction": "Buy a latex pillow under $50.",
                "observation_state": state,
            },
            observation_version=OBSERVATION_VERSION,
        )

    async def execute(self, action: str) -> EnvironmentResult:
        raise AssertionError("首轮 payload 测试不应执行环境动作")

    async def close(self) -> None:
        self.closed = True


class UnsafeInitialEnvironment(FirstTurnEnvironment):
    """模拟标称可用却缺少结构化页面的服务，确认 Runner 会 fail closed。"""

    async def start(self, request: EpisodeRequest) -> EnvironmentResult:
        return EnvironmentResult(
            instruction="Buy a pillow.",
            observation="target_asin: SECRET-MUST-NOT-REACH-TEACHER",
            raw_result={"instruction": "Buy a pillow."},
        )


class SftPromptContractTest(unittest.IsolatedAsyncioTestCase):
    """检查专用 Prompt 与实际发给 Teacher 的工具请求完全一致。"""

    async def asyncSetUp(self) -> None:
        """Keep fake Teacher transports deterministic and let Python exit cleanly."""

        patcher = patch(
            "wlx_harness_core.wlx_sft_policy.asyncio.to_thread",
            new=_inline_to_thread,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_prompt_and_registry_have_one_exact_contract(self) -> None:
        """SFT 只删 canonical 的思考函数，剩余 schema 不做私有分叉。"""

        canonical_without_removed_function = tuple(
            schema
            for schema in SHOP_TOOL_SCHEMAS
            if schema["function"]["name"] != "think"
        )
        self.assertEqual(WLX_SFT_TOOL_SCHEMAS, canonical_without_removed_function)
        self.assertEqual(WLX_SFT_TOOL_REGISTRY.tool_names, EXPECTED_SFT_TOOLS)
        self.assertNotIn("think", WLX_SFT_SYSTEM_PROMPT.lower())
        for name in EXPECTED_SFT_TOOLS:
            self.assertIn(f"`{name}`", WLX_SFT_SYSTEM_PROMPT)
        for legacy_action in ("search[", "click[", "finish["):
            self.assertNotIn(legacy_action, WLX_SFT_SYSTEM_PROMPT)
        self.assertIn('`{"reason":"no_suitable_product"}`', WLX_SFT_SYSTEM_PROMPT)
        self.assertTrue(WLX_SFT_SYSTEM_PROMPT_VERSION.startswith("wlx-sft-"))
        self.assertEqual(len(WLX_SFT_SYSTEM_PROMPT_SHA256), 64)

        for schema in WLX_SFT_TOOL_SCHEMAS:
            parameters = schema["function"]["parameters"]
            self.assertIs(parameters["additionalProperties"], False)
            self.assertLessEqual(
                set(parameters["required"]),
                set(parameters["properties"]),
            )

        with self.assertRaisesRegex(ToolRegistryError, "unknown tool"):
            WLX_SFT_TOOL_REGISTRY.parse_tool_call(
                {
                    "id": "removed",
                    "type": "function",
                    "function": {
                        "name": "think",
                        "arguments": json.dumps({"note": "noop"}),
                    },
                }
            )

    def test_resume_rejects_a_raw_directory_from_the_old_prompt(self) -> None:
        """旧配置没有 Prompt 指纹时必须换目录，不能静默混合两种轨迹。"""

        expected = {
            "teacher_base_url": "https://teacher.example/v1",
            "teacher_model": "teacher",
            "teacher_tokenizer": "tokenizer",
            "teacher_tokenizer_revision": "revision",
            "thinking": "enabled",
            "reasoning_effort": "high",
            "shopsim_base_url": "http://127.0.0.1:5700",
            "mode": "calibration",
            "task_plan_sha256": "1" * 64,
            "held_out_tasks_sha256": "2" * 64,
            "task_limit": 2,
            "tasks_selected": 2,
            "system_prompt_version": WLX_SFT_SYSTEM_PROMPT_VERSION,
            "system_prompt_sha256": WLX_SFT_SYSTEM_PROMPT_SHA256,
            "tool_schema_version": WLX_SFT_TOOL_REGISTRY.version,
            "tool_schema_fingerprint": WLX_SFT_TOOL_REGISTRY.fingerprint,
            "generation_reserve_tokens": 1_536,
            "harness_contract_sha256": "3" * 64,
            "temperature_effective": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "wlx-collection-config.json"
            config_path.write_text(
                json.dumps(expected, ensure_ascii=False),
                encoding="utf-8",
            )
            _assert_collection_config_compatible(config_path, expected)

            old_config = dict(expected)
            old_config.pop("system_prompt_version")
            old_config.pop("system_prompt_sha256")
            config_path.write_text(
                json.dumps(old_config, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "system_prompt"):
                _assert_collection_config_compatible(config_path, expected)

            changed_url = dict(expected)
            changed_url["teacher_base_url"] = "https://other.example/v1"
            config_path.write_text(
                json.dumps(changed_url, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "teacher_base_url"):
                _assert_collection_config_compatible(config_path, expected)

            changed_harness = dict(expected)
            changed_harness["harness_contract_sha256"] = "4" * 64
            config_path.write_text(
                json.dumps(changed_harness, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "harness_contract"):
                _assert_collection_config_compatible(config_path, expected)

    def test_custom_runner_cannot_bypass_the_sft_contract(self) -> None:
        """注入 Runner 时也必须使用专用 Prompt、12 工具和首轮 v2 页面。"""

        config = default_sft_harness_config()
        with self.assertRaises(ToolRegistryError):
            WlxSftDataPipeline(
                harness_config=config,
                runner=EpisodeRunner(
                    tool_registry=SHOPPING_TOOL_REGISTRY,
                    system_prompt=WLX_SFT_SYSTEM_PROMPT,
                    include_initial_observation=True,
                ),
            )
        with self.assertRaisesRegex(ValueError, "WLX_SFT_SYSTEM_PROMPT"):
            WlxSftDataPipeline(
                harness_config=config,
                runner=EpisodeRunner(
                    tool_registry=WLX_SFT_TOOL_REGISTRY,
                    system_prompt="old prompt",
                    include_initial_observation=True,
                ),
            )
        with self.assertRaisesRegex(ValueError, "Observation v2"):
            WlxSftDataPipeline(
                harness_config=config,
                runner=EpisodeRunner(
                    tool_registry=WLX_SFT_TOOL_REGISTRY,
                    system_prompt=WLX_SFT_SYSTEM_PROMPT,
                    include_initial_observation=False,
                ),
            )

        pipeline = WlxSftDataPipeline(harness_config=config)
        contract = sft_harness_contract(config, pipeline.runner)
        self.assertTrue(contract["runner"]["include_initial_observation"])
        self.assertEqual(len(sft_harness_contract_fingerprint(config, pipeline.runner)), 64)

    async def test_default_pipeline_sends_dedicated_prompt_and_initial_state(self) -> None:
        """通过真实 Runner/Policy 组装路径捕获首轮请求，不进行任何网络调用。"""

        config = default_sft_harness_config()
        pipeline = WlxSftDataPipeline(harness_config=config)
        environment = FirstTurnEnvironment()
        pipeline.runner.environment_factory = lambda unused_config: environment
        captured: list[dict] = []

        def transport(url, payload, headers, timeout):
            captured.append(payload)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "stop"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "total_tokens": 101,
                },
            }

        policy = OpenAICompatibleTeacherPolicy(
            model="teacher",
            base_url="https://example.invalid/v1",
            api_key="offline-test-key",
            context_policy=config.context,
            observation_policy=config.observation,
            count_chat_tokens=lambda messages, tools: 100,
            count_text_tokens=lambda text: len(str(text)),
            thinking_mode="enabled",
            transport=transport,
        )
        trajectory = await pipeline.runner.run(
            EpisodeRequest(task_id=1),
            policy,
            config,
        )

        self.assertEqual(trajectory.status, "assistant_final")
        self.assertTrue(environment.closed)
        self.assertEqual(len(captured), 1)
        payload = captured[0]
        self.assertEqual(payload["messages"][0], {
            "role": "system",
            "content": WLX_SFT_SYSTEM_PROMPT,
        })
        first_user = payload["messages"][1]["content"]
        self.assertIn("【购物任务】", first_user)
        self.assertIn("Buy a latex pillow under $50.", first_user)
        self.assertIn("【初始 ShopSimulator observation", first_user)
        self.assertIn("搜索功能是否可用: True", first_user)
        self.assertEqual(
            tuple(item["function"]["name"] for item in payload["tools"]),
            EXPECTED_SFT_TOOLS,
        )
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["max_tokens"], 1_536)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("tool_choice", payload)

    async def test_unstructured_initial_text_never_reaches_the_teacher(self) -> None:
        """缺 Observation v2 provenance 时按协议失败，不把任意文本发给模型。"""

        config = default_sft_harness_config()
        pipeline = WlxSftDataPipeline(harness_config=config)
        environment = UnsafeInitialEnvironment()
        pipeline.runner.environment_factory = lambda unused_config: environment
        captured: list[dict] = []

        def transport(url, payload, headers, timeout):
            captured.append(payload)
            raise AssertionError("不安全的初始文本不得进入 Teacher transport")

        policy = OpenAICompatibleTeacherPolicy(
            model="teacher",
            base_url="https://example.invalid/v1",
            api_key="offline-test-key",
            context_policy=config.context,
            observation_policy=config.observation,
            count_chat_tokens=lambda messages, tools: 20,
            count_text_tokens=lambda text: len(str(text)),
            thinking_mode="enabled",
            transport=transport,
        )
        trajectory = await pipeline.runner.run(
            EpisodeRequest(task_id=3),
            policy,
            config,
        )

        self.assertEqual(captured, [])
        self.assertEqual(trajectory.error.category, ErrorCategory.PROTOCOL)
        self.assertTrue(trajectory.infrastructure_invalid)
        self.assertEqual(trajectory.messages, ())
        self.assertTrue(environment.closed)

    async def test_length_finish_reason_is_a_retryable_technical_failure(self) -> None:
        """输出上限截断不能被 Runner 当成一次正常的无购买结果。"""

        config = default_sft_harness_config()
        pipeline = WlxSftDataPipeline(harness_config=config)
        environment = FirstTurnEnvironment()
        pipeline.runner.environment_factory = lambda unused_config: environment

        def transport(url, payload, headers, timeout):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "partial",
                            "reasoning_content": "unfinished",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 1_536,
                    "total_tokens": 1_556,
                },
            }

        policy = OpenAICompatibleTeacherPolicy(
            model="teacher",
            base_url="https://example.invalid/v1",
            api_key="offline-test-key",
            context_policy=config.context,
            observation_policy=config.observation,
            count_chat_tokens=lambda messages, tools: 20,
            count_text_tokens=lambda text: len(str(text)),
            thinking_mode="enabled",
            transport=transport,
        )
        trajectory = await pipeline.runner.run(
            EpisodeRequest(task_id=2),
            policy,
            config,
        )
        decision = classify_attempt(trajectory)

        self.assertEqual(trajectory.status, "error")
        self.assertTrue(trajectory.infrastructure_invalid)
        self.assertEqual(
            trajectory.error.error_type,
            TeacherOutputTruncatedError.__name__,
        )
        self.assertFalse(decision.attempt_valid)
        self.assertTrue(environment.closed)
        self.assertEqual(
            policy.request_usage_history[0]["output_tokens"],
            1_536,
        )


if __name__ == "__main__":
    unittest.main()
