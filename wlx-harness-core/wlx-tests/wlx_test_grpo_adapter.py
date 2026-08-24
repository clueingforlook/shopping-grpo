"""GRPO Harness 转接层的离线测试。

这些测试不启动模型、veRL 或 ShopSimulator，而是用假对象和固定数据检查：通用
Harness 配置能否正确转成 GRPO 配置，工具定义有没有漂移，以及运行结果能否按
约定写入轨迹。
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from wlx_harness_core import (  # noqa: E402
    BridgeDependencyError,
    ContextPolicy,
    ExtraFieldsContractError,
    GrpoStageConfig,
    HarnessConfig,
    ObservationPolicy,
    ToolConfigContractError,
    VerlHarnessBridge,
)
from wlx_harness_core.wlx_stage_grpo import (  # noqa: E402
    DEFAULT_AGENT_LOOP_TARGET,
    DEFAULT_SESSION_TARGET,
    SHOPPING_EXTRA_FIELDS,
)


def complete_shopping_payload() -> dict:
    """造一份字段齐全的假购物结果，供多个测试重复使用。

    它模拟 Agent 已成功购买目标商品后的公开诊断数据，不会访问真实环境。需要测试
    字段缺失、类型错误或扩展字段时，调用者会复制并修改这份数据。
    """

    return {
        "task_id": 7,
        "steps": 3,
        "done": True,
        "termination_reason": "gold_purchase",
        "error": None,
        "infrastructure_invalid": False,
        "action_attempts": 3,
        "repeat_actions": 0,
        "reward_mode": "constraint_aware",
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "gold_purchase",
        "reward_valid": True,
        "reward_unverifiable": False,
        "reward": {
            "terminal_utility": 1.0,
            "purchase_success": 1.0,
            "sampling_invalid": False,
        },
        "context_compactions": 0,
        "context_tokens_removed": 0,
        "context_max_input_tokens": 1024,
        "observation_projection_count": 3,
        "observation_truncated_count": 0,
        "observation_raw_tokens": 900,
        "observation_visible_tokens": 800,
        "observation_max_raw_tokens": 400,
        "observation_max_visible_tokens": 350,
        "observation_visible_asin_count": 4,
        "observation_visible_button_count": 6,
        "observation_any_truncated": False,
        "observation_footer_failures": 0,
        "guard_rejections": 0,
        "guard_rejections_after_truncation": 0,
        "action_attempts_after_truncation": 0,
    }


class GrpoStageConfigTest(unittest.TestCase):
    """检查通用 Harness 配置转换成 GRPO 配置时是否准确、安全。"""

    def test_default_mapping_matches_the_existing_agent_loop_contract(self):
        """验证默认配置能直接满足项目现有购物循环的参数约定。

        这里重点检查循环名称、类路径、环境地址、步数和上下文参数，也确认只属于环境
        会话的内部字段没有误塞进 Agent 循环配置。
        """

        config = GrpoStageConfig()
        entry = config.to_agent_loop_entry()

        self.assertEqual(entry["name"], "shopping_tool_agent")
        self.assertEqual(entry["_target_"], DEFAULT_AGENT_LOOP_TARGET)
        self.assertEqual(entry["base_url"], "http://127.0.0.1:5700")
        self.assertEqual(entry["required_environment_version"], "shopsimulator-environment-v2.1")
        self.assertEqual(entry["reward_mode"], "constraint_aware")
        self.assertEqual(entry["max_steps"], 35)
        self.assertEqual(entry["context_window_tokens"], 24576)
        self.assertFalse(entry["context_compaction_enable"])
        self.assertNotIn("session_target", entry)

        self.assertEqual(
            config.session_kwargs(),
            {
                "base_url": "http://127.0.0.1:5700",
                "timeout": 60,
                "max_steps": 35,
                "required_environment_version": "shopsimulator-environment-v2.1",
            },
        )

    def test_mapping_round_trip_preserves_forward_compatible_kwargs(self):
        """验证从旧配置字典读入再写出时，未来新增参数不会丢失。

        这模拟项目升级后 YAML 多出 Harness 暂时不认识的字段，转接层应原样保留它，
        同时仍补齐环境版本等必需参数。
        """

        source = {
            "name": "custom_shop_loop",
            "_target_": "package.loop.CustomLoop",
            "base_url": "http://shop:5700",
            "timeout": 12,
            "max_steps": 9,
            "future_option": {"enabled": True},
        }

        config = GrpoStageConfig.from_mapping(
            source,
            session_target="package.session.CustomSession",
        )

        self.assertEqual(config.agent_loop_name, "custom_shop_loop")
        self.assertEqual(config.session_target, "package.session.CustomSession")
        rendered = config.to_agent_loop_entry()
        for key, value in source.items():
            self.assertEqual(rendered[key], value)
        self.assertEqual(
            rendered["required_environment_version"],
            "shopsimulator-environment-v2.1",
        )

    def test_to_dict_materializes_immutable_extensions(self):
        """验证只读配置转成普通字典后可以保存和独立修改。

        返回字典应该能被 JSON 处理，修改其中的嵌套扩展字段也不能反过来改坏原配置；
        同时字典还应能重新构造出相同配置。
        """

        config = GrpoStageConfig(
            extra_agent_loop_kwargs={"future_option": {"enabled": True}}
        )

        dumped = config.to_dict()

        self.assertEqual(
            json.loads(json.dumps(dumped))["extra_agent_loop_kwargs"],
            {"future_option": {"enabled": True}},
        )
        dumped["extra_agent_loop_kwargs"]["future_option"]["enabled"] = False
        self.assertTrue(
            config.extra_agent_loop_kwargs["future_option"]["enabled"]
        )
        self.assertEqual(GrpoStageConfig(**config.to_dict()), config)

    def test_config_rejects_values_the_real_loop_cannot_safely_accept(self):
        """验证真实购物循环无法安全使用的配置会被提前拒绝。

        测试覆盖上下文塞不下、页面预算太小，以及扩展参数试图覆盖关键字段三种常见错误。
        """

        with self.assertRaisesRegex(ValueError, "fit the model context window"):
            GrpoStageConfig(context_input_budget_tokens=24000)
        with self.assertRaisesRegex(ValueError, "at least 64"):
            GrpoStageConfig(observation_token_budget=63)
        with self.assertRaisesRegex(ValueError, "reserved fields"):
            GrpoStageConfig(extra_agent_loop_kwargs={"max_steps": 99})

    def test_harness_config_maps_every_shared_policy_explicitly(self):
        """验证统一 Harness 中每项共享策略都明确映射到 GRPO 配置。

        这会检查环境、轮数、上下文压缩和页面裁剪等参数，防止某个字段看似配置成功，
        实际上在 GRPO 阶段被悄悄忽略。
        """

        harness = HarnessConfig(
            max_steps=12,
            environment_base_url="http://shop:5700",
            environment_timeout_s=9.0,
            max_assistant_turns=17,
            parallel_tool_call_policy="reject",
            context=ContextPolicy(
                window_tokens=8192,
                generation_reserve_tokens=512,
                safety_margin_tokens=0,
                input_budget_tokens=None,
                compaction_enabled=True,
                preserve_recent_groups=2,
            ),
            observation=ObservationPolicy(
                token_budget=512,
                detail_token_budget=1024,
                generic_token_budget=256,
                search_top_k=7,
            ),
        )

        config = GrpoStageConfig.from_harness_config(harness)

        self.assertEqual(config.base_url, "http://shop:5700")
        self.assertEqual(config.timeout, 9)
        self.assertEqual(config.max_steps, 12)
        self.assertEqual(config.max_assistant_turns, 17)
        self.assertEqual(config.context_safety_margin_tokens, 0)
        self.assertEqual(config.context_input_budget_tokens, 7680)
        self.assertTrue(config.context_compaction_enable)
        self.assertEqual(config.context_preserve_recent_groups, 2)
        self.assertEqual(config.observation_token_budget, 512)
        self.assertEqual(config.observation_search_top_k, 7)
        self.assertEqual(VerlHarnessBridge(harness).config, config)

    def test_harness_config_rejects_unsupported_grpo_semantics(self):
        """验证 GRPO 不支持的关键规则会立即报错。

        它逐项模拟工具并行规则、奖励版本、Guard 次数、奖励校验和超时格式不符合约定，
        确认错误会在启动训练前暴露。
        """

        common = {"max_steps": 35, "parallel_tool_call_policy": "reject"}
        invalid = (
            (HarnessConfig(max_steps=35), "parallel_tool_call_policy"),
            (HarnessConfig(**common, required_reward_version=None), "required_reward_version"),
            (HarnessConfig(**common, max_guard_rejections=4), "max_guard_rejections"),
            (HarnessConfig(**common, validate_terminal_reward=False), "validate_terminal_reward"),
            (HarnessConfig(**common, environment_timeout_s=1.5), "whole number"),
        )
        for config, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    GrpoStageConfig.from_harness_config(config)

    def test_zero_safety_margin_is_valid_and_harness_mappings_fail_fast(self):
        """验证安全余量可以设为零，同时禁止把 Harness 字典误当成 veRL 字典。

        前半部分保护一个合法边界值，后半部分防止两套相似配置格式混用后静默采用错误默认值。
        """

        self.assertEqual(
            GrpoStageConfig(context_safety_margin_tokens=0).context_safety_margin_tokens,
            0,
        )
        dumped = asdict(
            HarnessConfig(max_steps=35, parallel_tool_call_policy="reject")
        )
        with self.assertRaisesRegex(TypeError, "from_harness_config"):
            GrpoStageConfig.from_mapping(dumped)


class VerlHarnessBridgeTest(unittest.TestCase):
    """检查 GRPO 转接器的懒加载、工具配置和结果字段契约。"""

    def test_construction_and_config_mapping_do_not_import_verl_or_shopping(self):
        """验证只创建转接器和生成配置时不会提前导入 veRL 或购物运行库。

        这保证普通配置检查能在没有 GPU 和重依赖的机器上运行。测试用一个禁止导入的
        假加载器记录调用，只要发生提前导入就立即失败。
        """

        imports = []

        def forbidden_loader(module_name):
            """模拟一个绝不允许被调用的模块加载器。

            如果转接器在只做配置工作时调用它，就说明重依赖被错误地提前加载，测试会马上失败。
            """

            imports.append(module_name)
            raise AssertionError("runtime import happened eagerly")

        bridge = VerlHarnessBridge(module_loader=forbidden_loader)

        self.assertEqual(bridge.to_agent_loop_config()[0]["name"], "shopping_tool_agent")
        self.assertEqual(bridge.lease_contract()["scope"], "trajectory")
        self.assertEqual(bridge.extra_fields_contract()["namespace"], "shopping")
        self.assertEqual(imports, [])

    def test_runtime_classes_are_resolved_lazily_and_only_constructed(self):
        """验证运行类只在真正创建对象时加载，而且创建后不会自动启动环境。

        测试用假的循环和会话代替 veRL、ShopSimulator，并记录导入顺序、构造参数以及
        会话是否被启动或关闭。
        """

        imports = []

        class FakeLoop:
            """模拟真实 veRL 购物循环，只保存收到的构造参数。"""

            def __init__(self, marker, **kwargs):
                """记录假循环的标记和参数，方便测试检查转接结果。

                它不运行模型，也不执行任何 token 生成。
                """

                self.marker = marker
                self.kwargs = kwargs

        class FakeSession:
            """模拟一个尚未连接 ShopSimulator 的环境会话。"""

            def __init__(self, **kwargs):
                """保存会话参数，并把启动、关闭状态都设为否。

                测试稍后用这些状态确认“创建会话”没有偷偷变成“占用环境”。
                """

                self.kwargs = kwargs
                self.started = False
                self.closed = False

            async def start(self, task_id):  # pragma: no cover - 测试中不应该调用这里
                """模拟会话启动；本测试预期永远不会真正调用它。

                如果被调用，状态会变成已启动，从而让断言发现转接层越权操作了环境。
                """

                self.started = True

            async def close(self):  # pragma: no cover - 测试中不应该调用这里
                """模拟会话关闭；本测试预期永远不会真正调用它。

                如果被调用，状态会变成已关闭，从而让断言发现转接层提前释放了环境。
                """

                self.closed = True

        modules = {
            "package.loop": SimpleNamespace(CustomLoop=FakeLoop),
            "package.session": SimpleNamespace(CustomSession=FakeSession),
        }

        def loader(module_name):
            """按模块名返回假的循环类或会话类，并记录加载顺序。

            它代替 Python 的真实导入机制，让测试在不安装 veRL 的情况下检查懒加载行为。
            """

            imports.append(module_name)
            return modules[module_name]

        config = GrpoStageConfig(
            agent_loop_target="package.loop.CustomLoop",
            session_target="package.session.CustomSession",
            max_steps=8,
        )
        bridge = VerlHarnessBridge(config, module_loader=loader)
        self.assertEqual(imports, [])

        loop = bridge.make_agent_loop("marker", timeout=17)
        session = bridge.make_session(timeout=19)

        self.assertEqual(imports, ["package.loop", "package.session"])
        self.assertEqual(loop.marker, "marker")
        self.assertEqual(loop.kwargs["max_steps"], 8)
        self.assertEqual(loop.kwargs["timeout"], 17)
        self.assertEqual(session.kwargs["max_steps"], 8)
        self.assertEqual(session.kwargs["timeout"], 19)
        self.assertFalse(session.started)
        self.assertFalse(session.closed)

    def test_missing_verl_is_reported_only_when_loop_resolution_is_requested(self):
        """验证缺少 veRL 时，只有请求真实循环类才会报告依赖错误。

        创建转接器本身应成功；真正解析运行类时，底层导入错误应被包装成更清楚的
        ``BridgeDependencyError``。
        """

        def missing_loader(module_name):
            """模拟运行环境没有安装 veRL，任何模块加载都会失败。"""

            raise ModuleNotFoundError("No module named 'verl'", name="verl")

        bridge = VerlHarnessBridge(module_loader=missing_loader)

        with self.assertRaisesRegex(BridgeDependencyError, "GRPO agent loop"):
            bridge.resolve_agent_loop_class()

    def test_verl_override_mapping_is_narrow_and_does_not_restate_ppo(self):
        """验证转接器只覆盖 Harness 必需的 rollout 参数。

        它检查 Agent 循环、工具路径和单工具调用限制，同时确认没有顺手重写项目原有的
        PPO/GRPO actor 训练配方。
        """

        bridge = VerlHarnessBridge()

        overrides = bridge.to_verl_overrides(
            agent_loop_config_path="configs/agent_loop.yaml",
            tool_config_path="configs/tools.json",
        )

        rollout = overrides["actor_rollout_ref"]["rollout"]
        self.assertEqual(rollout["agent"]["default_agent_loop"], "shopping_tool_agent")
        self.assertEqual(
            rollout["agent"]["agent_loop_config_path"],
            "configs/agent_loop.yaml",
        )
        self.assertEqual(rollout["multi_turn"]["max_parallel_calls"], 1)
        self.assertEqual(rollout["multi_turn"]["max_assistant_turns"], 40)
        self.assertEqual(rollout["multi_turn"]["tool_config_path"], "configs/tools.json")
        self.assertNotIn("actor", overrides["actor_rollout_ref"])

    def test_tool_config_is_generated_from_and_checked_against_shared_registry(self):
        """验证 veRL 工具配置来自统一清单，并能发现配置漂移。

        先比较生成配置和仓库保存文件，再故意改掉一个工具描述，确认指纹检查会拒绝
        这份已和统一定义不一致的配置。
        """

        repository = PACKAGE_ROOT.parent
        source = repository / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from wlx_harness_core.wlx_tools import SHOPPING_TOOL_REGISTRY

        bridge = VerlHarnessBridge()
        generated = bridge.to_verl_tool_config(registry=SHOPPING_TOOL_REGISTRY)
        persisted = json.loads(
            (PACKAGE_ROOT / "wlx-tools.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted, generated)
        bridge.assert_tool_config_compatible(
            persisted,
            registry=SHOPPING_TOOL_REGISTRY,
        )
        self.assertEqual(len(generated["tools"]), len(SHOPPING_TOOL_REGISTRY.schemas))
        self.assertTrue(all(item["config"] == {"type": "native"} for item in generated["tools"]))

        bridge.assert_tool_config_compatible(
            generated,
            registry=SHOPPING_TOOL_REGISTRY,
        )

        drifted = json.loads(json.dumps(generated))
        drifted["tools"][0]["tool_schema"]["function"]["description"] = "drift"
        with self.assertRaisesRegex(ToolConfigContractError, "fingerprint mismatch"):
            bridge.assert_tool_config_compatible(
                drifted,
                registry=SHOPPING_TOOL_REGISTRY,
            )


    def test_lease_contract_names_the_loop_as_the_only_owner(self):
        """验证一条轨迹的环境只能由购物循环统一申请和释放。

        这防止转接器或单个工具重复释放 ShopSimulator 环境，也确认转接器没有自行运行
        轨迹的 ``run`` 方法。
        """

        contract = VerlHarnessBridge().lease_contract()

        self.assertEqual(contract["owner"], f"{DEFAULT_AGENT_LOOP_TARGET}.run")
        self.assertEqual(contract["acquire"], f"{DEFAULT_SESSION_TARGET}.start")
        self.assertEqual(contract["release"], f"{DEFAULT_SESSION_TARGET}.close")
        self.assertIn("no-op", contract["tool_release"])
        self.assertIn("never acquires", contract["bridge"])
        self.assertFalse(hasattr(VerlHarnessBridge, "run"))

    def test_complete_extra_fields_are_copied_and_extensions_are_preserved(self):
        """验证完整购物诊断数据会被复制、保留扩展项并写到固定位置。

        测试同时覆盖对象输出、字典输出和已有阶段 metadata，确认转接过程不改原数据，
        也不会覆盖已经存在的 ``shopping`` 命名空间。
        """

        payload = complete_shopping_payload()
        payload["future_diagnostic"] = {"kept": True}
        extra_fields = {"shopping": payload, "another_stage": {"kept": True}}
        bridge = VerlHarnessBridge()

        normalized = bridge.validate_extra_fields(extra_fields)

        self.assertEqual(set(SHOPPING_EXTRA_FIELDS).difference(normalized), set())
        self.assertEqual(normalized["future_diagnostic"], {"kept": True})
        self.assertIsNot(normalized, payload)
        self.assertEqual(extra_fields["another_stage"], {"kept": True})

        output = SimpleNamespace(extra_fields=extra_fields)
        self.assertEqual(bridge.extract_extra_fields(output), normalized)
        self.assertEqual(
            bridge.extract_extra_fields({"extra_fields": extra_fields}),
            normalized,
        )

        existing = {"harness": {"max_steps": 35}}
        stage_metadata = bridge.to_stage_metadata(output, existing=existing)
        self.assertEqual(stage_metadata["shopping"], normalized)
        self.assertEqual(stage_metadata["harness"], {"max_steps": 35})
        self.assertEqual(existing, {"harness": {"max_steps": 35}})
        self.assertEqual(
            bridge.extra_fields_contract()["stage_metadata_path"],
            "Trajectory.stage_metadata['shopping']",
        )
        with self.assertRaisesRegex(ExtraFieldsContractError, "already contains"):
            bridge.to_stage_metadata(output, existing={"shopping": {}})


    def test_incomplete_or_mistyped_extra_fields_fail_at_the_bridge(self):
        """验证购物结果缺字段或字段类型错误时会在转接层被拦住。

        这里分别模拟缺少任务编号、布尔值误写成数字，以及整个 shopping 区块缺失。
        """

        bridge = VerlHarnessBridge()
        payload = complete_shopping_payload()
        del payload["task_id"]
        with self.assertRaisesRegex(ExtraFieldsContractError, "task_id"):
            bridge.validate_extra_fields({"shopping": payload})

        payload = complete_shopping_payload()
        payload["done"] = 1
        with self.assertRaisesRegex(ExtraFieldsContractError, "done"):
            bridge.validate_extra_fields({"shopping": payload})

        with self.assertRaisesRegex(ExtraFieldsContractError, r"\['shopping'\]"):
            bridge.validate_extra_fields({})

    def test_partial_validation_supports_pre_terminal_diagnostics(self):
        """验证任务尚未结束时可以只检查已有的少量诊断字段。

        关闭完整性要求后，转接器仍检查已出现字段的类型，并保留尚未认识的扩展信息。
        """

        bridge = VerlHarnessBridge()

        payload = bridge.validate_extra_fields(
            {"shopping": {"task_id": 3, "done": False, "extension": "kept"}},
            require_complete=False,
        )

        self.assertEqual(payload, {"task_id": 3, "done": False, "extension": "kept"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
