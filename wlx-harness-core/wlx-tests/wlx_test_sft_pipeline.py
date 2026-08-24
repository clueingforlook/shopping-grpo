"""离线验证 WLX SFT 采样、难度、清洗和 Teacher 策略的主要安全规则。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError


REPOSITORY = Path(__file__).resolve().parents[2]
for import_path in (
    REPOSITORY / "src",
    REPOSITORY / "wlx-harness-core",
    REPOSITORY / "environments" / "ShopSimulator" / "shop_env",
    REPOSITORY,
):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from wlx_harness_core.wlx_config import ContextPolicy, HarnessConfig, ObservationPolicy  # noqa: E402
from wlx_harness_core.wlx_contracts import ModelRequest  # noqa: E402
from wlx_harness_core.wlx_sft_contracts import (  # noqa: E402
    SamplingConfig,
    SamplingMode,
    SftTask,
)
from wlx_harness_core.wlx_sft_dataset import (  # noqa: E402
    DatasetBuildConfig,
    build_dataset_artifacts,
    build_training_row,
)
from wlx_harness_core.wlx_sft_difficulty import (  # noqa: E402
    CalibrationSample,
    RetrievalEvidence,
    TaskDifficultyFeatures,
    assign_preliminary_labels,
    build_task_features,
    fit_with_grouped_validation,
    grouped_task_split,
    percentile_95,
)
from wlx_harness_core.wlx_sft_metrics import TrainingSequenceUnrenderable  # noqa: E402
from wlx_harness_core.wlx_sft_outcomes import classify_attempt  # noqa: E402
from wlx_harness_core.wlx_sft_pipeline import default_sft_harness_config  # noqa: E402
from wlx_harness_core.wlx_sft_policy import OpenAICompatibleTeacherPolicy  # noqa: E402
from wlx_harness_core.wlx_sft_sampler import (  # noqa: E402
    AttemptExecution,
    SftSamplingScheduler,
)
from wlx_harness_core.wlx_sft_shopsim_difficulty import (  # noqa: E402
    evaluate_instruction_retrieval,
)
from wlx_harness_core.wlx_sft_storage import RawTrajectoryStore, read_jsonl  # noqa: E402
SCRIPT_PATH = REPOSITORY / "scripts" / "wlx_sft_data_pipeline.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "wlx_sft_data_pipeline_test_cli",
    SCRIPT_PATH,
)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"无法加载 {SCRIPT_PATH}")
SCRIPT_MODULE = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)
_fit_difficulty = SCRIPT_MODULE._fit_difficulty
_fit_version_context = SCRIPT_MODULE._fit_version_context
_validate_fit_attempts = SCRIPT_MODULE._validate_fit_attempts


def _reward_detail(reward_type: str = "gold_purchase") -> dict:
    """生成字段完整的 Reward-v3 结果，供不同终局测试复用。"""

    success = reward_type in {"gold_purchase", "valid_alternative_purchase"}
    utility = 1.0 if reward_type == "gold_purchase" else 0.55 if success else -0.85
    return {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": reward_type,
        "termination_reason": reward_type,
        "reward_valid": True,
        "sampling_invalid": False,
        "purchase_success": success,
        "target_asin_match": reward_type == "gold_purchase",
        "terminal_utility": utility,
        "weighted_score": 1.0 if success else 0.0,
        "evidence_coverage": 1.0,
        "hard_gates": {
            name: {"status": "pass", "passed": True, "verifiable": True}
            for name in ("category", "budget")
        },
        "dimension_scores": {
            "brand": 1.0,
            "model": 1.0,
            "core_functions": 1.0,
            "key_options": 1.0,
        },
    }


def _tool_call(call_id: str, name: str, arguments: dict | None = None) -> dict:
    """生成一条 OpenAI 风格工具调用。"""

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {}, ensure_ascii=False),
        },
    }


def _core_step(index: int, call_id: str, name: str, arguments: dict, action: str, *, done: bool) -> dict:
    """生成一条 Core 风格 TrajectoryStep 字典。"""

    return {
        "step_index": index,
        "tool_call": {
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "call_type": "function",
        },
        "env_action": action,
        "observation": "Environment terminated." if done else f"new evidence {index}",
        "raw_observation": None,
        "projection": None,
        "reward": 1.0 if done else 0.0,
        "done": done,
        "result": {},
        "error": None,
    }


def _gold_trajectory(task_id: int = 10, *, reward_type: str = "gold_purchase") -> dict:
    """生成包含搜索、打开商品和购买三步的完整 Core 轨迹。"""

    detail = _reward_detail(reward_type)
    terminal = {
        "instruction": "Environment terminated.",
        "reward": detail["terminal_utility"],
        "done": True,
        "over": True,
        "reward_detail": detail,
    }
    messages = [{"role": "system", "content": "shop"}, {"role": "user", "content": "买枕头"}]
    calls = [
        ("s", "search_products", {"query": "枕头"}, "search[枕头]"),
        ("o", "open_product", {"asin": "100000000001"}, "click[100000000001]"),
        ("b", "buy_now", {}, "click[Buy Now]"),
    ]
    steps = []
    for index, (call_id, name, arguments, action) in enumerate(calls):
        messages.append(
            {
                "role": "assistant",
                "content": "先核验。",
                "reasoning_content": "这是不进入训练的私有推理",
                "tool_calls": [_tool_call(call_id, name, arguments)],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": (
                    "1|100000000001|枕头\n\n可点击的按钮: "
                    '["100000000001"]'
                    if name == "search_products"
                    else "detail\n\n可点击的按钮: [\"Buy Now\"]"
                    if name == "open_product"
                    else "private reward evidence"
                ),
            }
        )
        steps.append(
            _core_step(
                index,
                call_id,
                name,
                arguments,
                action,
                done=name == "buy_now",
            )
        )
    steps[-1]["result"] = terminal
    disposition = (
        "accepted_gold" if reward_type == "gold_purchase" else "alternative_audit"
    )
    return {
        "schema_version": "wlx-harness-trajectory-v1",
        "trajectory_id": f"trajectory-{task_id}-{reward_type}",
        "task_id": task_id,
        "attempt_index": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "status": "done",
        "termination_category": "environment_done",
        "termination_reason": reward_type,
        "messages": messages,
        "steps": steps,
        "initial_result": {},
        "terminal_result": terminal,
        "final_reward": detail["terminal_utility"],
        "done": True,
        "reward_valid": True,
        "sampling_invalid": False,
        "blocked_tool_calls": [],
        "tool_call_truncations": [],
        "context_compactions": [],
        "context_turn_tokens": [],
        "infrastructure_invalid": False,
        "error": None,
        "release_error": None,
        "stage_metadata": {
            "request": {
                "difficulty_label": "medium",
                "difficulty_score": 0.5,
                "difficulty_version": "test-v1",
                "category": "home",
            }
        },
        "outcome_type": reward_type,
        "attempt_valid": True,
        "task_success": True,
        "sft_disposition": disposition,
        "token_metrics": {"prompt_tokens_sum": 100},
    }


def _failed_trajectory(task_id: int, attempt_index: int) -> dict:
    """生成环境正常但 Teacher 达到最大步数的有效失败。"""

    return {
        "trajectory_id": f"failed-{task_id}-{attempt_index}",
        "task_id": task_id,
        "attempt_index": attempt_index,
        "status": "max_steps",
        "termination_reason": "max_steps",
        "messages": [],
        "steps": [],
        "terminal_result": {},
        "done": False,
        "infrastructure_invalid": False,
        "error": None,
        "release_error": None,
    }


def _infrastructure_trajectory(task_id: int, attempt_index: int) -> dict:
    """生成一次可技术重试的模型接口故障。"""

    row = _failed_trajectory(task_id, attempt_index)
    row.update(
        {
            "trajectory_id": f"infra-{task_id}-{attempt_index}",
            "status": "error",
            "termination_reason": "executor_exception",
            "infrastructure_invalid": True,
            "error": {"category": "infrastructure", "type": "Timeout", "message": "timeout"},
        }
    )
    return row


def _repeat_loop_trajectory(task_id: int, attempt_index: int) -> dict:
    """生成 ShopSimulator 原生形状的正常重复循环终局。"""

    detail = {
        "reward_version": "shopsimulator-reward-v3",
        "reward_type": "repeat_loop",
        "termination_reason": "repeat_loop",
        "reward_valid": True,
        "sampling_invalid": False,
        "purchase_success": False,
        "target_asin_match": False,
        "terminal_utility": -0.65,
        "weighted_score": 0.0,
        "evidence_coverage": 0.0,
        "hard_gates": {},
        "dimension_scores": {},
    }
    return {
        "trajectory_id": f"repeat-{task_id}-{attempt_index}",
        "task_id": task_id,
        "attempt_index": attempt_index,
        "status": "done",
        "termination_category": "environment_done",
        "termination_reason": "repeat_loop",
        "messages": [],
        "steps": [],
        "terminal_result": {
            "reward": -0.65,
            "done": True,
            "over": True,
            "reward_detail": detail,
        },
        "final_reward": -0.65,
        "done": True,
        "reward_valid": True,
        "sampling_invalid": False,
        "infrastructure_invalid": False,
        "error": None,
        "release_error": None,
    }


class ScriptedExecutor:
    """按预设结果返回轨迹，并记录 scheduler 实际调用的逻辑与技术编号。"""

    def __init__(self, results: list[dict]) -> None:
        """保存将依次返回的轨迹。"""

        self.results = list(results)
        self.calls: list[tuple[int, int, int]] = []

    async def execute(self, task: SftTask, attempt_index: int, technical_retry_index: int):
        """模拟一次 Core 运行。"""

        self.calls.append((task.task_id, attempt_index, technical_retry_index))
        row = dict(self.results.pop(0))
        row["task_id"] = task.task_id
        row["attempt_index"] = attempt_index
        return AttemptExecution(row)


class OutcomeAndSchedulerTest(unittest.IsolatedAsyncioTestCase):
    """检查结果分类与两种采样模式的核心区别。"""

    def test_alternative_is_success_for_difficulty_but_not_for_sft(self):
        """验证替代商品计任务成功，却只能进入审计区。"""

        decision = classify_attempt(_gold_trajectory(reward_type="valid_alternative_purchase"))
        self.assertTrue(decision.attempt_valid)
        self.assertTrue(decision.task_success)
        self.assertEqual(decision.sft_disposition.value, "alternative_audit")

    def test_dirty_gold_remains_valid_success_but_is_rejected_for_sft(self):
        """Guard、截断或压缩不改变环境成功，只禁止轨迹进入主 SFT。"""

        dirty_fields = {
            "blocked_tool_calls": [{"reason": "guard_rejection"}],
            "tool_call_truncations": [{"dropped_tool_calls": []}],
            "context_compactions": [{"removed_messages": 2}],
        }
        expected_reasons = {
            "blocked_tool_calls": "has_guard_rejection",
            "tool_call_truncations": "has_parallel_tool_truncation",
            "context_compactions": "context_was_compacted",
        }
        for field, value in dirty_fields.items():
            with self.subTest(field=field):
                trajectory = _gold_trajectory()
                trajectory[field] = value
                decision = classify_attempt(trajectory)
                self.assertTrue(decision.attempt_valid)
                self.assertTrue(decision.task_success)
                self.assertEqual(decision.outcome_type.value, "gold_purchase")
                self.assertEqual(decision.sft_disposition.value, "rejected")
                self.assertIn(expected_reasons[field], decision.reasons)

    async def test_calibration_runs_all_three_valid_attempts(self):
        """验证难度校准第一次成功后仍会固定跑满三次。"""

        with tempfile.TemporaryDirectory() as temporary:
            executor = ScriptedExecutor(
                [_gold_trajectory(), _failed_trajectory(10, 1), _failed_trajectory(10, 2)]
            )
            scheduler = SftSamplingScheduler(
                SamplingConfig(mode=SamplingMode.CALIBRATION, concurrency=1)
            )
            report = await scheduler.collect(
                [SftTask(10)],
                executor=executor,
                store=RawTrajectoryStore(Path(temporary) / "raw.jsonl", durable=False),
            )
            self.assertEqual(report["new_valid_attempts"], 3)
            self.assertEqual([call[1] for call in executor.calls], [0, 1, 2])

    async def test_calibration_dirty_gold_still_runs_all_three_attempts(self):
        """脏 Gold 仍算校准成功，但校准模式无论如何都固定跑满三次。"""

        with tempfile.TemporaryDirectory() as temporary:
            dirty_gold = _gold_trajectory()
            dirty_gold["blocked_tool_calls"] = [{"reason": "guard_rejection"}]
            executor = ScriptedExecutor(
                [dirty_gold, _gold_trajectory(), _failed_trajectory(10, 2)]
            )
            scheduler = SftSamplingScheduler(
                SamplingConfig(mode=SamplingMode.CALIBRATION, concurrency=1)
            )
            store = RawTrajectoryStore(Path(temporary) / "raw.jsonl", durable=False)

            report = await scheduler.collect(
                [SftTask(10)], executor=executor, store=store
            )
            rows = list(store.rows())

            self.assertEqual(report["new_valid_attempts"], 3)
            self.assertEqual(report["new_gold"], 1)
            self.assertEqual([call[1] for call in executor.calls], [0, 1, 2])
            self.assertTrue(rows[0]["attempt_valid"])
            self.assertTrue(rows[0]["task_success"])
            self.assertEqual(rows[0]["sft_disposition"], "rejected")

    async def test_repeat_loops_fill_attempts_without_technical_retries(self):
        """重复循环是有效模型失败，三条应直接占满校准名额。"""

        with tempfile.TemporaryDirectory() as temporary:
            executor = ScriptedExecutor(
                [_repeat_loop_trajectory(10, index) for index in range(3)]
            )
            scheduler = SftSamplingScheduler(
                SamplingConfig(mode=SamplingMode.CALIBRATION, concurrency=1)
            )
            store = RawTrajectoryStore(
                Path(temporary) / "raw.jsonl",
                durable=False,
            )

            report = await scheduler.collect(
                [SftTask(10)],
                executor=executor,
                store=store,
            )
            rows = list(store.rows())

            self.assertEqual(report["new_valid_attempts"], 3)
            self.assertEqual(report["technical_failures"], 0)
            self.assertEqual(
                executor.calls,
                [(10, 0, 0), (10, 1, 0), (10, 2, 0)],
            )
            self.assertTrue(all(row["attempt_valid"] for row in rows))
            self.assertTrue(
                all(row["sft_disposition"] == "rejected" for row in rows)
            )

    async def test_formal_retries_technical_failure_and_stops_only_on_gold(self):
        """验证技术故障不吃名额、alternative 不停、Gold 才停止正式任务。"""

        with tempfile.TemporaryDirectory() as temporary:
            executor = ScriptedExecutor(
                [
                    _infrastructure_trajectory(10, 0),
                    _failed_trajectory(10, 0),
                    _gold_trajectory(reward_type="valid_alternative_purchase"),
                    _gold_trajectory(),
                ]
            )
            scheduler = SftSamplingScheduler(
                SamplingConfig(mode=SamplingMode.FORMAL, concurrency=1)
            )
            store = RawTrajectoryStore(Path(temporary) / "raw.jsonl", durable=False)
            report = await scheduler.collect([SftTask(10)], executor=executor, store=store)
            self.assertEqual(report["new_gold"], 1)
            self.assertEqual(
                executor.calls,
                [(10, 0, 0), (10, 0, 1), (10, 1, 0), (10, 2, 0)],
            )
            self.assertEqual(len(list(store.rows())), 4)

    async def test_formal_continues_after_dirty_gold_and_stops_on_clean_gold(self):
        """正式采样不把脏 Gold 计入目标，下一次拿到干净 Gold 才停止。"""

        with tempfile.TemporaryDirectory() as temporary:
            dirty_gold = _gold_trajectory()
            dirty_gold["blocked_tool_calls"] = [{"reason": "guard_rejection"}]
            executor = ScriptedExecutor(
                [dirty_gold, _gold_trajectory(), _failed_trajectory(10, 2)]
            )
            scheduler = SftSamplingScheduler(
                SamplingConfig(
                    mode=SamplingMode.FORMAL,
                    concurrency=1,
                    target_gold_trajectories=1,
                )
            )
            store = RawTrajectoryStore(Path(temporary) / "raw.jsonl", durable=False)

            report = await scheduler.collect(
                [SftTask(10)], executor=executor, store=store
            )
            rows = list(store.rows())

            self.assertEqual(executor.calls, [(10, 0, 0), (10, 1, 0)])
            self.assertEqual(report["new_valid_attempts"], 2)
            self.assertEqual(report["new_gold"], 1)
            self.assertEqual([row["task_success"] for row in rows], [True, True])
            self.assertEqual(
                [row["sft_disposition"] for row in rows],
                ["rejected", "accepted_gold"],
            )

    async def test_formal_resume_reclassifies_legacy_dirty_gold(self):
        """断点续采不能相信旧标签，历史脏 Gold 也必须继续下一次尝试。"""

        with tempfile.TemporaryDirectory() as temporary:
            raw_path = Path(temporary) / "raw.jsonl"
            legacy_dirty_gold = _gold_trajectory()
            legacy_dirty_gold["blocked_tool_calls"] = [
                {"reason": "guard_rejection"}
            ]
            legacy_dirty_gold["sft_disposition"] = "accepted_gold"
            raw_path.write_text(
                json.dumps(legacy_dirty_gold, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            executor = ScriptedExecutor([_gold_trajectory()])
            scheduler = SftSamplingScheduler(
                SamplingConfig(
                    mode=SamplingMode.FORMAL,
                    concurrency=1,
                    target_gold_trajectories=1,
                )
            )

            report = await scheduler.collect(
                [SftTask(10)],
                executor=executor,
                store=RawTrajectoryStore(raw_path, durable=False),
            )

            self.assertEqual(report["existing_gold_before_run"], 0)
            self.assertEqual(report["new_gold"], 1)
            self.assertEqual(executor.calls, [(10, 1, 0)])

    async def test_target_gold_does_not_drain_all_waiting_tasks(self):
        """验证达到目标后，等待并发槽的任务不会继续调用付费 Teacher。"""

        with tempfile.TemporaryDirectory() as temporary:
            executor = ScriptedExecutor([_gold_trajectory(task_id) for task_id in range(5)])
            scheduler = SftSamplingScheduler(
                SamplingConfig(
                    mode=SamplingMode.FORMAL,
                    concurrency=1,
                    target_gold_trajectories=1,
                )
            )
            await scheduler.collect(
                [SftTask(task_id) for task_id in range(5)],
                executor=executor,
                store=RawTrajectoryStore(Path(temporary) / "raw.jsonl", durable=False),
            )
            self.assertEqual(len(executor.calls), 1)


class DifficultyTest(unittest.TestCase):
    """检查规则分数、95 分位、分层和纯 Python 逻辑回归。"""

    def test_rule_features_follow_documented_formula(self):
        """验证品牌、型号、预算、功能和规格轴会分别增加独立约束。"""

        goal = {
            "expected_core_functions": ["防水", "降噪"],
            "required_options_by_key": {"color": "black"},
            "unresolved_option_requirements": [],
            "expected_brand": ["A"],
            "expected_model": ["M1"],
            "price_upper": 100,
        }
        evidence = RetrievalEvidence(2, 4, "valid_alternative_purchase", 3)
        features = build_task_features(
            task_id=1,
            goal=goal,
            retrieval=evidence,
            a95=6,
            n95=5,
        )
        self.assertEqual(features.constraint_count, 6)
        self.assertAlmostEqual(features.constraint_score, 1.0)
        self.assertEqual(percentile_95(range(1, 101)), 95.0)

    def test_grouped_validation_keeps_task_attempts_together(self):
        """验证同一 task_id 的三次尝试不会跨越逻辑回归训练和验证。"""

        samples = _calibration_samples()
        training, validation = grouped_task_split(samples, validation_ratio=0.25, seed=7)
        self.assertFalse(
            {item.task_id for item in training}.intersection(
                item.task_id for item in validation
            )
        )

    def test_logistic_model_can_fit_mixed_calibration_results(self):
        """验证不依赖 sklearn 的小模型能产出版本、报告和难度概率。"""

        samples = _calibration_samples()
        model, report = fit_with_grouped_validation(samples, seed=3)
        self.assertTrue(model.difficulty_version.startswith("wlx-difficulty-"))
        self.assertIn("validation", report)
        labels = assign_preliminary_labels([item.features for item in samples[::3]])
        self.assertEqual(len(labels), 40)

    def test_fit_attempt_validation_ignores_technical_retries(self):
        """技术重试可以复用编号，但每题仍必须有三个唯一有效尝试。"""

        plan = [{"task_id": 10}, {"task_id": 20}]
        raw = [
            {"task_id": task_id, "attempt_index": attempt, "attempt_valid": True}
            for task_id in (10, 20)
            for attempt in range(3)
        ]
        raw.extend(
            [
                {"task_id": 10, "attempt_index": 1, "attempt_valid": False},
                {"task_id": 10, "attempt_index": 1, "attempt_valid": False},
                {"task_id": 20, "attempt_index": 2, "attempt_valid": False},
            ]
        )

        self.assertEqual(_validate_fit_attempts(plan, raw), {10, 20})

    def test_fit_attempt_validation_rejects_duplicate_or_unplanned_rows(self):
        """有效编号重复以及任何计划外 raw 行都会阻止拟合。"""

        plan = [{"task_id": 10}]
        duplicate_valid = [
            {"task_id": 10, "attempt_index": attempt, "attempt_valid": True}
            for attempt in (0, 1, 2, 2)
        ]
        with self.assertRaisesRegex(SystemExit, "0,1,2"):
            _validate_fit_attempts(plan, duplicate_valid)

        with_unplanned_technical = [
            {"task_id": 10, "attempt_index": attempt, "attempt_valid": True}
            for attempt in range(3)
        ] + [{"task_id": 99, "attempt_index": 0, "attempt_valid": False}]
        with self.assertRaisesRegex(SystemExit, "计划外"):
            _validate_fit_attempts(plan, with_unplanned_technical)

    def test_fit_cli_binds_verified_provenance_and_reports_it(self):
        """拟合版本和报告都绑定语义契约、输入 SHA 与切分参数。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "wlx-plan.jsonl"
            raw_path = root / "wlx-raw.jsonl"
            features_path = root / "wlx-features.jsonl"
            manifest_path = root / "wlx-manifest.json"
            output_dir = root / "wlx-fit-output"
            plan_rows = [{"task_id": 10}, {"task_id": 20}]
            raw_rows = [
                {
                    "task_id": task_id,
                    "attempt_index": attempt,
                    "attempt_valid": True,
                    "task_success": attempt < 2,
                }
                for task_id in (10, 20)
                for attempt in range(3)
            ]
            feature_rows = [
                {
                    "task_id": task_id,
                    "constraint_count": 2,
                    "option_axis_count": 1,
                    "has_brand": False,
                    "has_model": False,
                    "has_budget": True,
                    "retrieval_score": 0.25,
                    "near_miss_score": 0.0,
                    "constraint_score": 0.2,
                    "preliminary_score": 0.2,
                    "preliminary_label": "easy",
                    "category": "测试",
                    "evidence": {},
                }
                for task_id in (10, 20)
            ]
            _write_test_jsonl(plan_path, plan_rows)
            _write_test_jsonl(raw_path, raw_rows)
            _write_test_jsonl(features_path, feature_rows)
            semantic_contract = {
                "teacher_model": "teacher-test",
                "system_prompt_sha256": "prompt-sha",
                "tool_schema_fingerprint": "tool-sha",
            }
            manifest_path.write_text(
                json.dumps(
                    {
                        "semantic_contract": semantic_contract,
                        "final_artifacts": {
                            "plan_sha256": _test_sha256(plan_path),
                            "raw_sha256": _test_sha256(raw_path),
                        },
                    }
                ),
                encoding="utf-8",
            )
            model = Mock(difficulty_version="wlx-difficulty-test")
            model.to_dict.return_value = {
                "difficulty_version": model.difficulty_version
            }
            report = {"validation": {"samples": 3}}
            args = SimpleNamespace(
                plan=plan_path,
                raw=raw_path,
                features=features_path,
                manifest=manifest_path,
                output_dir=output_dir,
                validation_ratio=0.25,
                seed=9,
            )

            with patch.object(
                SCRIPT_MODULE,
                "fit_with_grouped_validation",
                return_value=(model, report),
            ) as fit:
                self.assertEqual(_fit_difficulty(args), 0)

            context = fit.call_args.kwargs["version_context"]
            self.assertEqual(context["semantic_contract"], semantic_contract)
            self.assertEqual(context["plan_sha256"], _test_sha256(plan_path))
            self.assertEqual(context["raw_sha256"], _test_sha256(raw_path))
            self.assertEqual(context["features_sha256"], _test_sha256(features_path))
            self.assertEqual(context["manifest_sha256"], _test_sha256(manifest_path))
            self.assertEqual(context["seed"], 9)
            self.assertEqual(context["validation_ratio"], 0.25)
            self.assertEqual(fit.call_args.kwargs["expected_task_ids"], {10, 20})
            stored_report = json.loads(
                (output_dir / "wlx-difficulty-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stored_report["fit_provenance"], context)

            raw_path.write_text(
                raw_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "raw_sha256"):
                _fit_version_context(
                    plan_path=plan_path,
                    raw_path=raw_path,
                    features_path=features_path,
                    manifest_path=manifest_path,
                    seed=9,
                    validation_ratio=0.25,
                )

    def test_fit_provenance_rejects_secret_fields(self):
        """语义清单若误含 API Key 字段，不得写入模型或报告。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "wlx-plan.jsonl"
            raw_path = root / "wlx-raw.jsonl"
            features_path = root / "wlx-features.jsonl"
            manifest_path = root / "wlx-manifest.json"
            for path in (plan_path, raw_path, features_path):
                path.write_text("{}\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "semantic_contract": {"deepseek_api_key": "must-not-leak"},
                        "final_artifacts": {
                            "plan_sha256": _test_sha256(plan_path),
                            "raw_sha256": _test_sha256(raw_path),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SystemExit, "疑似凭据"):
                _fit_version_context(
                    plan_path=plan_path,
                    raw_path=raw_path,
                    features_path=features_path,
                    manifest_path=manifest_path,
                    seed=42,
                    validation_ratio=0.2,
                )

    def test_retrieval_features_reuse_real_shopsimulator_reward(self):
        """验证 R/N 会调用环境同一 Reward，并能识别首个替代商品和目标排名。"""

        from web_agent_site.engine.reward_features import compile_reward_features

        target = _difficulty_product("111111111111")
        alternative = _difficulty_product("222222222222")
        near_miss = _difficulty_product("333333333333", attributes=["智能洗地"])
        instruction = {
            "instruction": "购买石头 A20 支持热洗的白色 XL 洗地机，预算2200元",
            "attributes": ["洗地", "热洗"],
            "instruction_options": ["白色", "XL"],
        }
        goal = {
            "asin": target["asin"],
            "category": target["category"],
            "price_upper": 2200,
            "instruction_text": instruction["instruction"],
            **compile_reward_features(instruction, target),
        }
        products = {
            product["asin"]: product for product in (target, alternative, near_miss)
        }
        evidence = evaluate_instruction_retrieval(
            goal=goal,
            product_item_dict=products,
            searcher=FakeDifficultySearcher(
                [alternative["asin"], near_miss["asin"], target["asin"]]
            ),
        )
        self.assertEqual(evidence.best_valid_rank_at_20, 1)
        self.assertEqual(evidence.best_valid_outcome_type, "valid_alternative_purchase")
        self.assertEqual(evidence.target_asin_rank_at_20, 3)
        self.assertGreaterEqual(evidence.near_miss_count_at_20, 1)


def _calibration_samples() -> list[CalibrationSample]:
    """生成 40 道题各三次、同时含成功和失败的校准数据。"""

    samples = []
    for task_id in range(40):
        hard = task_id >= 20
        features = TaskDifficultyFeatures(
            task_id=task_id,
            constraint_count=8 if hard else 2,
            option_axis_count=3 if hard else 0,
            has_brand=hard,
            has_model=hard,
            has_budget=True,
            retrieval_score=0.9 if hard else 0.1,
            near_miss_score=0.8 if hard else 0.1,
            constraint_score=0.9 if hard else 0.2,
            preliminary_score=0.88 if hard else 0.15,
        )
        for attempt in range(3):
            success = (not hard and attempt < 2) or (hard and attempt == 0 and task_id % 5 == 0)
            samples.append(CalibrationSample(task_id, features, success))
    return samples


def _write_test_jsonl(path: Path, rows: list[dict]) -> None:
    """为 CLI 拟合测试写入带结尾换行的短 JSONL。"""

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _test_sha256(path: Path) -> str:
    """计算临时测试输入的 SHA-256。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _difficulty_product(asin: str, *, attributes: list[str] | None = None) -> dict:
    """生成能被真实 Reward-v3 检查的洗地机商品。"""

    return {
        "asin": asin,
        "title": "石头 A20 智能洗地机",
        "brand": "石头",
        "shop_name": "石头旗舰店",
        "category": "家电›清洁电器›洗地机",
        "attribute": attributes or ["智能洗地", "热洗"],
        "pricing": [1999],
        "customization_options": {
            "颜色分类": [
                {"value": "白色", "price": 1999},
                {"value": "黑色", "price": 1999},
            ],
            "尺码": [
                {"value": "L", "price": 1899},
                {"value": "XL", "price": 1999},
            ],
        },
    }


class FakeDifficultyHit:
    """模拟 BM25 返回的一条 ASIN 和排名。"""

    def __init__(self, asin: str, rank: int) -> None:
        """保存商品编号和从 1 开始的排名。"""

        self.asin = asin
        self.rank = rank


class FakeDifficultySearcher:
    """按固定顺序返回候选，方便验证真实 Reward 而不读取大索引。"""

    def __init__(self, asins: list[str]) -> None:
        """保存测试候选顺序。"""

        self.asins = list(asins)

    def search(self, query: object, k: int = 20):
        """模拟 BM25 search，并保留真实接口的 query 和 k 参数形状。"""

        return [
            FakeDifficultyHit(asin, rank)
            for rank, asin in enumerate(self.asins[:k], start=1)
        ]


class DatasetBuilderTest(unittest.TestCase):
    """检查最终 SFT 构建会清私有推理、排评测题、验长度并冻结目录。"""

    def test_builds_strict_gold_dataset_with_real_counter_contract(self):
        """验证 Gold 进入训练、alternative 只审计，并且工具表没有 think。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            rows = [_gold_trajectory(10), _gold_trajectory(11, reward_type="valid_alternative_purchase")]
            raw.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            held_out = root / "held-out.jsonl"
            held_out.write_text('{"task_id":999}\n', encoding="utf-8")

            def fake_counter(row):
                """模拟最终 tokenizer，返回可验证的总长度和 Loss 长度。"""

                return 1200, 180, {
                    "tokenizer_name": "fake",
                    "tokenizer_revision": "r1",
                    "chat_template_version": "t1",
                }

            output = root / "wlx-dataset-v1"
            summary = build_dataset_artifacts(
                raw_path=raw,
                output_dir=output,
                held_out_tasks_path=held_out,
                training_token_counter=fake_counter,
                config=DatasetBuildConfig(validation_ratio=0.0),
            )
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["alternatives_for_audit"], 1)
            row = list(read_jsonl(output / "sft.jsonl"))[0]
            serialised = json.dumps(row, ensure_ascii=False)
            self.assertNotIn("私有推理", serialised)
            self.assertIn("先核验", serialised)
            self.assertIn("购买已完成", serialised)
            self.assertNotIn('"name":"think"', serialised)
            self.assertEqual(row["metadata"]["sft_total_tokens"], 1200)
            with self.assertRaises(FileExistsError):
                build_dataset_artifacts(
                    raw_path=raw,
                    output_dir=output,
                    held_out_tasks_path=held_out,
                    training_token_counter=fake_counter,
                )

    def test_public_reasoning_is_kept_but_private_reasoning_is_removed(self):
        """验证公开 assistant.content 保留，reasoning_content 被白名单删除。"""

        row = build_training_row(_gold_trajectory())
        serialised = json.dumps(row, ensure_ascii=False)
        self.assertIn("先核验", serialised)
        self.assertNotIn("不进入训练的私有推理", serialised)

    def test_unrenderable_gold_is_rejected_without_aborting_the_batch(self):
        """一条模板坏样本应留下拒绝原因，其余 Gold 继续构建。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "wlx-raw.jsonl"
            _write_test_jsonl(raw, [_gold_trajectory(10), _gold_trajectory(12)])
            held_out = root / "wlx-held-out.jsonl"
            _write_test_jsonl(held_out, [{"task_id": 999}])

            def counter(row):
                if row["task_id"] == 10:
                    raise TrainingSequenceUnrenderable("bad template")
                return 1000, 100, {}

            output = root / "wlx-dataset"
            summary = build_dataset_artifacts(
                raw_path=raw,
                output_dir=output,
                held_out_tasks_path=held_out,
                training_token_counter=counter,
                config=DatasetBuildConfig(validation_ratio=0.0),
            )
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["reject_reasons"]["sft_chat_template_unrenderable"], 1)
            rejected = list(read_jsonl(output / "rejected.jsonl"))
            self.assertEqual(rejected[0]["task_id"], 10)

    def test_recomputed_cost_metrics_override_stale_raw_token_metrics(self):
        """旧 raw 中的步骤统计不能覆盖最终构建时的现场重算结果。"""

        trajectory = _gold_trajectory()
        trajectory["token_metrics"].update(
            {"productive_steps": 0, "trajectory_length_bucket": "too_short_review"}
        )
        row = build_training_row(trajectory)
        self.assertEqual(row["metadata"]["productive_steps"], 3)
        self.assertEqual(row["metadata"]["trajectory_length_bucket"], "short")


async def _inline_to_thread(function, /, *args, **kwargs):
    """在线程敏感的单测中同步执行，避免测试解释器等待后台线程退出。"""

    return function(*args, **kwargs)


class TeacherPolicyTest(unittest.IsolatedAsyncioTestCase):
    """检查 WLX Teacher 策略会记录 Usage 且与 Core 配置一致。"""

    async def test_policy_records_each_chat_completion_usage(self):
        """验证接口 Usage 会被保存，但 API Key 不会出现在轨迹指标。"""

        context = ContextPolicy(
            window_tokens=200,
            generation_reserve_tokens=32,
            safety_margin_tokens=16,
        )
        observation = ObservationPolicy(
            token_budget=64,
            detail_token_budget=64,
            generic_token_budget=64,
        )

        def transport(url, payload, headers, timeout):
            """模拟一次兼容 Chat Completions 的服务响应。"""

            self.assertTrue(url.endswith("/chat/completions"))
            self.assertIn("Bearer secret-test-key", headers["Authorization"])
            self.assertEqual(payload["thinking"], {"type": "enabled"})
            self.assertNotIn("temperature", payload)
            self.assertNotIn("tool_choice", payload)
            return {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            }

        policy = OpenAICompatibleTeacherPolicy(
            model="teacher",
            base_url="https://example.invalid/v1",
            api_key="secret-test-key",
            context_policy=context,
            observation_policy=observation,
            count_chat_tokens=lambda messages, tools: 12,
            count_text_tokens=lambda text: len(str(text).split()),
            thinking_mode="enabled",
            transport=transport,
        )
        policy.validate_harness_config(
            HarnessConfig(max_steps=35, context=context, observation=observation)
        )
        with patch(
            "wlx_harness_core.wlx_sft_policy.asyncio.to_thread",
            new=_inline_to_thread,
        ):
            message = await policy.generate(
                ModelRequest(messages=({"role": "user", "content": "hi"},), tools=())
            )
        self.assertEqual(message["content"], "ok")
        self.assertEqual(policy.request_usage_history[0]["request_total_tokens"], 15)
        self.assertNotIn(
            "secret-test-key",
            json.dumps(policy.request_usage_history),
        )

    def test_policy_logs_retry_status_without_request_or_key(self):
        """瞬时 429 成功恢复时留下安全日志，不记录请求正文或 Key。"""

        context = ContextPolicy(
            window_tokens=200,
            generation_reserve_tokens=32,
            safety_margin_tokens=16,
        )
        observation = ObservationPolicy(
            token_budget=64,
            detail_token_budget=64,
            generic_token_budget=64,
        )
        calls = 0

        def transport(url, payload, headers, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise HTTPError(url, 429, "rate limited", hdrs=None, fp=None)
            return {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }

        policy = OpenAICompatibleTeacherPolicy(
            model="teacher",
            base_url="https://example.invalid/v1",
            api_key="secret-test-key",
            context_policy=context,
            observation_policy=observation,
            count_chat_tokens=lambda messages, tools: 12,
            count_text_tokens=lambda text: len(str(text)),
            max_retries=1,
            transport=transport,
        )
        with patch("wlx_harness_core.wlx_sft_policy.time.sleep") as sleep:
            with self.assertLogs(
                "wlx_harness_core.wlx_sft_policy",
                level="WARNING",
            ) as captured:
                response = policy._request_with_retries(
                    {
                        "messages": [
                            {"role": "user", "content": "private prompt"}
                        ]
                    }
                )

        text = "\n".join(captured.output)
        self.assertEqual(response["choices"][0]["message"]["content"], "ok")
        self.assertEqual(calls, 2)
        sleep.assert_called_once_with(1)
        self.assertIn("status=429", text)
        self.assertNotIn("private prompt", text)
        self.assertNotIn("secret-test-key", text)

    async def test_policy_cleans_history_before_deepseek_request(self):
        """验证空 assistant content 和本地 Guard 字段不会原样发送给 DeepSeek。"""

        context = ContextPolicy(
            window_tokens=500,
            generation_reserve_tokens=32,
            safety_margin_tokens=16,
        )
        observation = ObservationPolicy(
            token_budget=64,
            detail_token_budget=64,
            generic_token_budget=64,
        )

        def transport(url, payload, headers, timeout):
            """检查真正的 HTTP payload 与本地计数前消息使用相同白名单。"""

            assistant = payload["messages"][1]
            tool = payload["messages"][2]
            self.assertEqual(assistant["content"], "")
            self.assertEqual(assistant["reasoning_content"], "完整私有推理")
            self.assertEqual(
                set(assistant),
                {"role", "content", "reasoning_content", "tool_calls"},
            )
            self.assertEqual(
                set(tool),
                {"role", "tool_call_id", "content"},
            )
            return {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1},
            }

        policy = OpenAICompatibleTeacherPolicy(
            model="teacher",
            base_url="https://example.invalid/v1",
            api_key="secret-test-key",
            context_policy=context,
            observation_policy=observation,
            count_chat_tokens=lambda messages, tools: 20,
            count_text_tokens=lambda text: len(str(text)),
            thinking_mode="enabled",
            transport=transport,
        )
        with patch(
            "wlx_harness_core.wlx_sft_policy.asyncio.to_thread",
            new=_inline_to_thread,
        ):
            await policy.generate(
                ModelRequest(
                    messages=(
                        {"role": "user", "content": "找商品"},
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "完整私有推理",
                            "provider_extra": "不要发送",
                            "tool_calls": [
                                _tool_call(
                                    "c1", "search_products", {"query": "枕头"}
                                )
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "c1",
                            "name": "search_products",
                            "runtime_action_guard": True,
                            "content": "环境结果",
                        },
                    ),
                    tools=(),
                )
            )

    def test_default_sft_config_enables_projection_not_context_deletion(self):
        """验证第一版默认开启 Observation Projection，但不自动删除旧上下文。"""

        config = default_sft_harness_config()
        self.assertIsNotNone(config.observation)
        self.assertFalse(config.context.compaction_enabled)
        self.assertEqual(config.context.window_tokens, 24_576)


if __name__ == "__main__":
    unittest.main()
