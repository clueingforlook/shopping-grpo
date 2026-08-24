"""WLX Eval v2 确定性管线、Judge 边界和报告测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[2]
for path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wlx_harness_core.wlx_eval_judge import (  # noqa: E402
    JudgeContractError,
    WLX_EVAL_JUDGE_FULL_OUTPUT_VERSION,
    WLX_EVAL_JUDGE_OUTPUT_VERSION,
    build_full_judge_payload,
    build_process_judge_payload,
    validate_full_judgment,
    validate_process_judgment,
)
from wlx_harness_core.wlx_eval_pipeline import (  # noqa: E402
    OfflineEvaluationOutputExistsError,
    _apply_judgment,
    run_offline_evaluation,
)
from wlx_harness_core.wlx_eval_judge_pipeline import (  # noqa: E402
    JudgeCase,
    run_judge_batch,
)
from wlx_harness_core.wlx_eval_report import compare_evaluation_runs  # noqa: E402
from wlx_harness_core.wlx_eval_rubric import freeze_rubric_bundle  # noqa: E402
from wlx_harness_core.wlx_eval_rubric_generator import (  # noqa: E402
    DeepSeekRubricClient,
)
from wlx_harness_core.wlx_trajectory_evaluator import evaluate_trajectory  # noqa: E402


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _step(index: int, call_id: str, name: str, arguments: dict) -> dict:
    return {
        "step_index": index,
        "tool_call": {
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "call_type": "function",
        },
        "env_action": name,
        "observation": "模型可见观察",
        "raw_observation": "模型不可见的完整原始观察",
        "projection": {
            "visible_tokens": 10,
            "raw_tokens": 20,
            "truncated": True,
        },
        "reward": 0,
        "done": name == "buy_now",
        "result": {
            "observation_state": {
                "page_type": "search_results" if name == "search_products" else "product_detail",
                "normalized_query": arguments.get("query"),
                "page": 1,
                "products": (
                    [{"asin": "gold-1"}, {"asin": "alt-2"}]
                    if name == "search_products"
                    else []
                ),
            }
        },
        "error": None,
    }


def _trajectory(
    task_id: int = 1,
    *,
    actual_option: str = "白色",
    environment_outcome: str = "gold_purchase",
    purchase: bool = True,
    **overrides,
) -> dict:
    calls = [
        ("c1", "search_products", {"query": "白色保温杯"}),
        ("c2", "open_product", {"asin": "gold-1"}),
        ("c3", "select_option", {"value": actual_option}),
        ("c4", "buy_now", {}),
    ]
    messages = [
        {"role": "system", "content": "使用购物工具"},
        {"role": "user", "content": "请购买50元以内的白色保温杯"},
    ]
    steps = []
    for index, (call_id, name, arguments) in enumerate(calls):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call(call_id, name, arguments)],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": "模型可见观察",
            }
        )
        steps.append(_step(index, call_id, name, arguments))
    terminal_purchase = (
        {
            "asin": "gold-1",
            "category": "杯具›保温杯",
            "name": "白色保温杯",
            "attributes": ["保温", "白色"],
            "options": {"颜色": actual_option},
            "price": 39,
        }
        if purchase
        else None
    )
    row = {
        "schema_version": "wlx-harness-trajectory-v1",
        "contract_version": "wlx-harness-contract-v1",
        "trajectory_id": f"trajectory-{task_id}",
        "task_id": task_id,
        "messages": messages,
        "steps": steps,
        "terminal_result": {
            "done": True,
            "over": True,
            "goal": {
                "asin": "gold-1",
                "category": "杯具›保温杯",
                "instruction_text": "请购买50元以内的白色保温杯",
                "expected_core_functions": ["保温", "白色"],
                "required_options_by_key": {"color": {"value": "白色"}},
                "price_upper": 50,
                "expected_brand": [],
                "expected_model": [],
            },
            "purchase": terminal_purchase,
            "reward": 1.0 if environment_outcome == "gold_purchase" else 0.0,
            "reward_valid": True,
            "termination_reason": environment_outcome,
            "reward_detail": {
                "reward_type": environment_outcome,
                "termination_reason": environment_outcome,
                "reward_valid": True,
                "sampling_invalid": False,
            },
        },
        "status": "done",
        "done": True,
        "termination_category": "environment_done",
        "termination_reason": environment_outcome,
        "reward_valid": True,
        "sampling_invalid": False,
        "infrastructure_invalid": False,
        "release_error": None,
        "error": None,
        "blocked_tool_calls": [],
        "tool_call_truncations": [],
        "context_compactions": [],
        "context_turn_tokens": [
            {"step_index": index, "input_tokens": 100 + index}
            for index in range(4)
        ],
    }
    row.update(overrides)
    return row


class DeterministicEvalV2Test(unittest.TestCase):
    def test_success_has_four_panels_and_stable_event_refs(self):
        result = evaluate_trajectory(_trajectory()).to_dict()

        self.assertEqual(result["schema_version"], "wlx-eval-v2")
        self.assertTrue(result["format_correct"])
        self.assertTrue(result["purchase_correct"])
        self.assertEqual(result["eligibility"]["status"], "eligible")
        self.assertEqual(result["outcome"]["normalized_outcome"], "gold_purchase")
        self.assertEqual(result["requirements"]["status"], "provisional")
        self.assertEqual(result["process_quality"]["status"], "not_run")
        self.assertEqual(result["failure_attribution"]["status"], "not_applicable")
        self.assertEqual(result["deterministic_behavior"]["search_count"], 1)
        self.assertEqual(result["deterministic_behavior"]["unique_product_open_count"], 1)
        self.assertEqual(result["deterministic_behavior"]["total_assistant_turns"], 4)
        self.assertEqual(result["event_index"][0]["event_id"], "E000")
        self.assertEqual(result["event_index"][-1]["kind"], "termination")

    def test_environment_verifier_disagreement_enters_review(self):
        result = evaluate_trajectory(_trajectory(actual_option="黑色")).to_dict()

        self.assertFalse(result["purchase_correct"])
        self.assertEqual(result["outcome"]["normalized_outcome"], "partial_purchase")
        self.assertFalse(result["outcome"]["success_agreement"])
        self.assertEqual(result["eligibility"]["status"], "review_required")
        self.assertIn("outcome_success_disagreement", result["review"]["reasons"])

    def test_model_invalid_action_remains_evaluable(self):
        trajectory = _trajectory(
            purchase=False,
            environment_outcome="invalid_action_limit",
            done=False,
            status="invalid_action_limit",
            termination_category="invalid_action",
            termination_reason="invalid_action_limit",
            terminal_result={},
        )
        result = evaluate_trajectory(trajectory).to_dict()

        self.assertTrue(result["trajectory_valid"])
        self.assertEqual(result["eligibility"]["status"], "eligible")
        self.assertEqual(
            result["failure_attribution"]["primary_failure"], "invalid_action"
        )

    def test_environment_failure_is_non_model_invalid(self):
        result = evaluate_trajectory(
            _trajectory(
                infrastructure_invalid=True,
                status="error",
                termination_category="infrastructure_error",
                termination_reason="tool_error:TimeoutError",
                error={"category": "environment", "error_type": "TimeoutError"},
            )
        ).to_dict()

        self.assertFalse(result["trajectory_valid"])
        self.assertIsNone(result["purchase_correct"])
        self.assertEqual(result["eligibility"]["status"], "non_model_invalid")

    def test_frozen_rubric_leaves_every_requirement_for_the_judge(self):
        rubric = freeze_rubric_bundle(
            task_id=1,
            instruction="请购买50元以内的白色保温杯",
            rubric_set_version="dev-v1",
            items=[
                {
                    "requirement_id": "R001",
                    "requirement": "颜色为白色",
                    "type": "option",
                    "priority": "hard",
                    "query_evidence": "白色",
                    "verifier": {
                        "kind": "deterministic",
                        "field": "color",
                        "operator": "option_match",
                        "value": "白色",
                    },
                },
                {
                    "requirement_id": "R002",
                    "requirement": "价格不超过50元",
                    "type": "budget",
                    "priority": "hard",
                    "query_evidence": "50元以内",
                    "verifier": {
                        "kind": "deterministic",
                        "field": "price",
                        "operator": "lte",
                        "value": 50,
                    },
                },
                {
                    "requirement_id": "R003",
                    "requirement": "容易清洗",
                    "type": "preference",
                    "priority": "soft",
                    "query_evidence": "保温杯",
                    "verifier": {
                        "kind": "semantic",
                        "field": "cleaning",
                        "operator": "semantic_entailment",
                        "criterion": "容易清洗",
                    },
                },
            ],
        )

        result = evaluate_trajectory(_trajectory(), rubric=rubric).to_dict()

        self.assertEqual(
            [item["status"] for item in result["requirements"]["items"]],
            ["unknown", "unknown", "unknown"],
        )
        self.assertEqual(
            result["requirements"]["status"],
            "llm_judge_pending",
        )

    def test_option_text_matches_punctuation_but_unrelated_option_is_not_counterevidence(self):
        rubric = freeze_rubric_bundle(
            task_id=1,
            instruction="购买赠送亲肤垫和收纳袋且包含安装服务的商品",
            rubric_set_version="dev-v1",
            items=[
                {
                    "requirement_id": "R001",
                    "requirement": "赠送亲肤垫和收纳袋",
                    "type": "option",
                    "priority": "hard",
                    "query_evidence": "赠送亲肤垫和收纳袋",
                    "verifier": {
                        "kind": "deterministic",
                        "field": "free_gifts",
                        "operator": "option_match",
                        "value": "亲肤垫,收纳袋",
                    },
                },
                {
                    "requirement_id": "R002",
                    "requirement": "包含安装服务",
                    "type": "option",
                    "priority": "hard",
                    "query_evidence": "包含安装服务",
                    "verifier": {
                        "kind": "deterministic",
                        "field": "installation_service",
                        "operator": "equals",
                        "value": "包含",
                    },
                },
            ],
        )
        trajectory = _trajectory()
        trajectory["terminal_result"]["purchase"]["options"] = {
            "颜色分类": "【黑色】送亲肤垫+收纳袋"
        }

        result = evaluate_trajectory(trajectory, rubric=rubric).to_dict()

        self.assertEqual(
            [item["status"] for item in result["requirements"]["items"]],
            ["unknown", "unknown"],
        )


class OfflinePipelineTest(unittest.TestCase):
    def test_judge_resolution_clears_temporary_attribution_review_reason(self):
        evaluation = evaluate_trajectory(
            _trajectory(environment_outcome="no_purchase", purchase=False)
        ).to_dict()
        self.assertTrue(
            any(
                reason.startswith("attribution:")
                for reason in evaluation["review"]["reasons"]
            )
        )
        judge_result = {
            "prompt_version": "test-judge-v1",
            "judgment": {
                "requirements": [],
                "dimensions": {
                    name: {
                        "score": 1,
                        "reason": "已判断",
                        "evidence_event_ids": [],
                        "confidence": 0.9,
                    }
                    for name in (
                        "search_strategy",
                        "candidate_utilization",
                        "evidence_verification",
                        "decision_quality",
                        "termination_efficiency",
                    )
                },
                "attribution": {
                    "responsibility": "model",
                    "primary_failure": "premature_abstention",
                    "secondary_failures": [],
                    "first_error_event_id": None,
                    "reason": "模型没有购买",
                    "confidence": 0.9,
                },
                "review_required": False,
                "review_reasons": [],
            },
        }

        resolved = _apply_judgment(evaluation, judge_result)

        self.assertFalse(resolved["review"]["required"])
        self.assertEqual(resolved["review"]["reasons"], [])

    def test_offline_pipeline_writes_derived_outputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "run-a" / "wlx-trajectories.jsonl"
            source.parent.mkdir()
            source.write_text(
                json.dumps(_trajectory(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output = source.parent / "wlx-eval-v2"
            result = run_offline_evaluation(
                trajectories_path=source,
                output_dir=output,
            )
            evaluations = [
                json.loads(line)
                for line in (output / "wlx-trajectory-evaluations.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            summary = json.loads(
                (output / "wlx-run-summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(result["evaluated_trajectories"], 1)
            self.assertEqual(len(evaluations), 1)
            self.assertEqual(summary["format"]["correct"]["numerator"], 1)
            self.assertIn("trajectory_line_sha256", evaluations[0]["source_refs"])
            with self.assertRaises(OfflineEvaluationOutputExistsError):
                run_offline_evaluation(
                    trajectories_path=source,
                    output_dir=output,
                )

    def test_paired_comparison_reports_candidate_only_success(self):
        baseline = evaluate_trajectory(
            _trajectory(environment_outcome="repeat_loop", purchase=False)
        ).to_dict()
        candidate = evaluate_trajectory(_trajectory()).to_dict()
        summary, paired = compare_evaluation_runs([baseline], [candidate])

        self.assertEqual(summary["paired_tasks"], 1)
        self.assertEqual(summary["success_transitions"]["candidate_only_success"], 1)
        self.assertEqual(paired[0]["task_id"], 1)

    def test_offline_pipeline_requires_matching_frozen_rubric(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "run-a" / "wlx-trajectories.jsonl"
            source.parent.mkdir()
            source.write_text(
                json.dumps(_trajectory(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rubrics = root / "wlx-rubrics.jsonl"
            bundle = freeze_rubric_bundle(
                task_id=1,
                instruction="请购买50元以内的白色保温杯",
                rubric_set_version="dev-v1",
                items=[
                    {
                        "requirement_id": "R001",
                        "requirement": "价格不超过50元",
                        "type": "budget",
                        "priority": "hard",
                        "query_evidence": "50元以内",
                        "verifier": {
                            "kind": "deterministic",
                            "field": "price",
                            "operator": "lte",
                            "value": 50,
                        },
                    }
                ],
            )
            rubrics.write_text(json.dumps(bundle, ensure_ascii=False) + "\n")

            result = run_offline_evaluation(
                trajectories_path=source,
                output_dir=root / "eval",
                rubrics_path=rubrics,
            )
            manifest = json.loads(Path(result["manifest"]).read_text())

        self.assertEqual(manifest["rubrics"]["count"], 1)
        self.assertEqual(manifest["rubrics"]["rubric_set_version"], "dev-v1")


class JudgeBoundaryTest(unittest.TestCase):
    def test_payload_excludes_hidden_outcome_and_validates_event_evidence(self):
        rubric = freeze_rubric_bundle(
            task_id=1,
            instruction="请购买50元以内的白色保温杯",
            rubric_set_version="dev-v1",
            items=[
                {
                    "requirement_id": "R001",
                    "requirement": "颜色为白色",
                    "type": "option",
                    "priority": "hard",
                    "query_evidence": "白色",
                    "verifier": {"kind": "semantic"},
                }
            ],
        )
        payload = build_process_judge_payload(
            trajectory=_trajectory(),
            rubric=rubric,
        )
        encoded = json.dumps(payload, ensure_ascii=False)

        self.assertNotIn("reward_detail", encoded)
        self.assertNotIn("target_asin", encoded)
        self.assertNotIn("模型不可见的完整原始观察", encoded)
        allowed = payload["allowed_evidence_event_ids"]
        output = {
            "schema_version": WLX_EVAL_JUDGE_OUTPUT_VERSION,
            "dimensions": {
                name: {
                    "score": 2,
                    "reason": "证据充分",
                    "evidence_event_ids": [allowed[0]],
                    "confidence": 0.9,
                }
                for name in (
                    "search_strategy",
                    "candidate_utilization",
                    "evidence_verification",
                    "decision_quality",
                    "termination_efficiency",
                )
            },
            "review_required": False,
            "review_reasons": [],
        }
        validated = validate_process_judgment(output, allowed_event_ids=allowed)
        self.assertEqual(validated["dimensions"]["search_strategy"]["score"], 2)

        output["dimensions"]["search_strategy"]["evidence_event_ids"] = ["E999"]
        with self.assertRaises(JudgeContractError):
            validate_process_judgment(output, allowed_event_ids=allowed)

    def test_full_judge_requires_every_rubric_item_and_valid_attribution(self):
        rubric = freeze_rubric_bundle(
            task_id=1,
            instruction="请购买50元以内的白色保温杯",
            rubric_set_version="dev-v1",
            items=[
                {
                    "requirement_id": "R001",
                    "requirement": "容易清洗",
                    "type": "preference",
                    "priority": "soft",
                    "query_evidence": "保温杯",
                    "verifier": {
                        "kind": "semantic",
                        "field": "cleaning",
                        "operator": "semantic_entailment",
                        "criterion": "容易清洗",
                    },
                }
            ],
        )
        payload = build_full_judge_payload(
            trajectory=_trajectory(),
            rubric=rubric,
        )
        allowed = payload["allowed_evidence_event_ids"]
        value = {
            "schema_version": WLX_EVAL_JUDGE_FULL_OUTPUT_VERSION,
            "requirements": [
                {
                    "requirement_id": "R001",
                    "status": "unknown",
                    "reason": "页面没有相关信息",
                    "evidence_event_ids": [],
                    "confidence": 0.9,
                }
            ],
            "dimensions": {
                name: {
                    "score": 2,
                    "reason": "过程合理",
                    "evidence_event_ids": [allowed[0]],
                    "confidence": 0.9,
                }
                for name in (
                    "search_strategy",
                    "candidate_utilization",
                    "evidence_verification",
                    "decision_quality",
                    "termination_efficiency",
                )
            },
            "attribution": {
                "responsibility": "undetermined",
                "primary_failure": "none",
                "secondary_failures": [],
                "first_error_event_id": None,
                "reason": "购买成功",
                "confidence": 0.9,
            },
            "review_required": False,
            "review_reasons": [],
        }

        normalized = validate_full_judgment(
            value,
            allowed_event_ids=allowed,
            required_requirement_ids=["R001"],
        )

        self.assertEqual(normalized["requirements"][0]["status"], "unknown")
        value["requirements"] = []
        with self.assertRaisesRegex(JudgeContractError, "exactly cover"):
            validate_full_judgment(
                value,
                allowed_event_ids=allowed,
                required_requirement_ids=["R001"],
            )

    def test_calibration_batch_is_resumable_without_frozen_gate(self):
        rubric = freeze_rubric_bundle(
            task_id=1,
            instruction="请购买50元以内的白色保温杯",
            rubric_set_version="dev-v1",
            items=[
                {
                    "requirement_id": "R001",
                    "requirement": "容易清洗",
                    "type": "preference",
                    "priority": "soft",
                    "query_evidence": "保温杯",
                    "verifier": {
                        "kind": "semantic",
                        "field": "cleaning",
                        "operator": "semantic_entailment",
                        "criterion": "容易清洗",
                    },
                }
            ],
        )
        calls = []

        def transport(url, payload, headers, timeout):
            del url, headers, timeout
            calls.append(1)
            judge_input = json.loads(payload["messages"][1]["content"])
            event_id = judge_input["allowed_evidence_event_ids"][0]
            value = {
                "schema_version": WLX_EVAL_JUDGE_FULL_OUTPUT_VERSION,
                "requirements": [
                    {
                        "requirement_id": "R001",
                        "status": "unknown",
                        "reason": "无证据",
                        "evidence_event_ids": [],
                        "confidence": 0.9,
                    }
                ],
                "dimensions": {
                    name: {
                        "score": 2,
                        "reason": "合理",
                        "evidence_event_ids": [event_id],
                        "confidence": 0.9,
                    }
                    for name in (
                        "search_strategy",
                        "candidate_utilization",
                        "evidence_verification",
                        "decision_quality",
                        "termination_efficiency",
                    )
                },
                "attribution": {
                    "responsibility": "undetermined",
                    "primary_failure": "none",
                    "secondary_failures": [],
                    "first_error_event_id": None,
                    "reason": "成功",
                    "confidence": 0.9,
                },
                "review_required": False,
                "review_reasons": [],
            }
            return {
                "id": "judge-1",
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": json.dumps(value)}}],
                "usage": {},
            }

        client = DeepSeekRubricClient(
            model="deepseek-v4-pro",
            base_url="https://teacher.example/v1",
            api_key="secret",
            transport=transport,
        )
        with tempfile.TemporaryDirectory() as temporary:
            first = run_judge_batch(
                cases=[JudgeCase("blind:1", _trajectory(), rubric)],
                output_dir=Path(temporary) / "judge",
                client=client,
                mode="calibration",
            )
            second = run_judge_batch(
                cases=[JudgeCase("blind:1", _trajectory(), rubric)],
                output_dir=Path(temporary) / "judge",
                client=client,
                mode="calibration",
            )

        self.assertTrue(first["complete"])
        self.assertTrue(second["reused_existing_outputs"])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
