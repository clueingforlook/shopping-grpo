"""Rubric 的规则、DeepSeek 客户端、批处理和最终冻结测试。"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[2]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.eval_rubric_generator import (  # noqa: E402
    DeepSeekRubricClient,
    RubricTask,
    RUBRIC_SYSTEM_PROMPT_SHA256,
    extract_rule_signals,
    generate_rubric_candidate,
)
from shopping_grpo.harness.eval_rubric_pipeline import (  # noqa: E402
    finalize_rubric_set,
    load_rubric_tasks,
    run_rubric_generation,
)


def _items(*, review: bool = False, hallucinated: bool = False) -> dict:
    evidence = "并不存在的红色" if hallucinated else "白色"
    return {
        "items": [
            {
                "requirement": "商品是保温杯",
                "type": "category",
                "priority": "hard",
                "query_evidence": "保温杯",
                "verifier": {
                    "kind": "deterministic",
                    "field": "category",
                    "operator": "category_match",
                    "value": "保温杯",
                },
            },
            {
                "requirement": "颜色为白色",
                "type": "option",
                "priority": "hard",
                "query_evidence": evidence,
                "verifier": {
                    "kind": "deterministic",
                    "field": "option",
                    "operator": "option_match",
                    "value": "白色",
                },
            },
            {
                "requirement": "价格不超过50元",
                "type": "budget",
                "priority": "hard",
                "query_evidence": "50元以内",
                "verifier": {
                    "kind": "deterministic",
                    "field": "price",
                    "operator": "lte",
                    "value": 50,
                    "unit": "CNY",
                },
            },
        ],
        "review_required": review,
        "review_reasons": ["颜色表达可能有歧义"] if review else [],
    }


def _response(result: dict) -> dict:
    return {
        "id": "request-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(result, ensure_ascii=False),
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 80},
    }


class RubricRuleAndClientTest(unittest.TestCase):
    def test_rules_extract_hard_and_approximate_budget(self):
        hard = extract_rule_signals("请买50元以内的白色保温杯")
        soft = extract_rule_signals("价格在20元左右")

        self.assertEqual(hard["budget_facts"][0]["operator"], "lte")
        self.assertEqual(hard["budget_facts"][0]["priority"], "hard")
        self.assertEqual(soft["budget_facts"][0]["operator"], "approx")
        self.assertEqual(soft["budget_facts"][0]["priority"], "soft")

    def test_rules_do_not_misclassify_dimensions_near_budget(self):
        signals = extract_rule_signals("直径17.5cm，容量180毫升，预算70元左右")

        self.assertEqual(len(signals["budget_facts"]), 1)
        self.assertEqual(signals["budget_facts"][0]["value"], 70)
        self.assertEqual(signals["budget_facts"][0]["operator"], "approx")

    def test_rules_extract_currency_range_as_one_budget(self):
        signals = extract_rule_signals("预算在100-200元，尺寸23厘米左右")

        self.assertEqual(len(signals["budget_facts"]), 1)
        self.assertEqual(signals["budget_facts"][0]["operator"], "between")
        self.assertEqual(
            signals["budget_facts"][0]["value"], {"min": 100, "max": 200}
        )

    def test_bugfix_keeps_existing_prompt_contract_compatible(self):
        self.assertEqual(
            RUBRIC_SYSTEM_PROMPT_SHA256,
            "885e17c15b8c7c698ee538923117eb9d5ee25e57929128272683be7269b9a052",
        )

    def test_deepseek_client_uses_json_mode_without_serializing_key(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update(
                {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
            )
            return _response(_items())

        client = DeepSeekRubricClient(
            model="deepseek-v4-flash",
            base_url="https://teacher.example/v1",
            api_key="secret-test-key",
            transport=transport,
        )
        result = client.complete_json([{"role": "user", "content": "test"}])

        self.assertEqual(
            captured["url"], "https://teacher.example/v1/chat/completions"
        )
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(
            captured["payload"]["response_format"], {"type": "json_object"}
        )
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret-test-key")
        self.assertNotIn("secret-test-key", json.dumps(result))

    def test_candidate_is_frozen_only_from_exact_query_evidence(self):
        client = DeepSeekRubricClient(
            model="deepseek-v4-flash",
            base_url="https://teacher.example/v1",
            api_key="secret",
            transport=lambda *_: _response(_items()),
        )
        task = RubricTask(1, "请购买50元以内的白色保温杯")
        candidate = generate_rubric_candidate(
            task,
            client=client,
            rubric_set_version="wlx-eval200-rubric-v1",
        )

        self.assertFalse(candidate["review_required"])
        self.assertEqual(candidate["rubric"]["task_id"], 1)
        self.assertEqual(candidate["rubric"]["items"][2]["verifier"]["value"], 50)
        self.assertNotIn("api_key", json.dumps(candidate))

        bad_client = DeepSeekRubricClient(
            model="deepseek-v4-flash",
            base_url="https://teacher.example/v1",
            api_key="secret",
            transport=lambda *_: _response(_items(hallucinated=True)),
        )
        repaired = generate_rubric_candidate(
            task,
            client=bad_client,
            rubric_set_version="wlx-eval200-rubric-v1",
        )
        self.assertTrue(repaired["review_required"])
        self.assertIn("R002:query_evidence_not_exact", repaired["review_reasons"])
        self.assertEqual(
            repaired["rubric"]["items"][1]["query_evidence"], task.instruction
        )

    def test_numeric_range_string_is_normalized_deterministically(self):
        result = _items()
        result["items"][2]["requirement"] = "价格在40到50元"
        result["items"][2]["query_evidence"] = "40-50元"
        result["items"][2]["verifier"].update(
            {"operator": "range", "value": "40-50元"}
        )
        client = DeepSeekRubricClient(
            model="deepseek-v4-flash",
            base_url="https://teacher.example/v1",
            api_key="secret",
            transport=lambda *_: _response(result),
        )

        candidate = generate_rubric_candidate(
            RubricTask(1, "请购买40-50元的白色保温杯"),
            client=client,
            rubric_set_version="wlx-eval200-rubric-v1",
        )

        verifier = candidate["rubric"]["items"][2]["verifier"]
        self.assertEqual(verifier["operator"], "between")
        self.assertEqual(verifier["value"], {"min": 40, "max": 50})
        self.assertFalse(candidate["review_required"])


class RubricBatchTest(unittest.TestCase):
    def test_loads_instruction_only_from_initial_trajectory_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks_path = root / "tasks.jsonl"
            trajectories = root / "trajectories.jsonl"
            tasks_path.write_text('{"task_id":7}\n', encoding="utf-8")
            trajectories.write_text(
                json.dumps(
                    {
                        "task_id": 7,
                        "initial_result": {
                            "instruction": "Instruction: 请购买50元以内的白色保温杯"
                        },
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "模型行为不能进入 Rubric",
                            }
                        ],
                        "terminal_result": {
                            "goal": {"instruction_text": "隐藏目标不应覆盖初始需求"}
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            tasks = load_rubric_tasks(tasks_path, trajectories_path=trajectories)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].instruction, "请购买50元以内的白色保温杯")
        self.assertNotIn("模型行为", tasks[0].instruction)

    def test_batch_is_resumable_and_finalization_requires_reviews(self):
        calls = []

        def transport(url, payload, headers, timeout):
            del url, headers, timeout
            user = json.loads(payload["messages"][1]["content"])
            calls.append(user["task_id"])
            return _response(_items(review=user["task_id"] == 2))

        client = DeepSeekRubricClient(
            model="deepseek-v4-flash",
            base_url="https://teacher.example/v1",
            api_key="secret",
            transport=transport,
        )
        tasks = [
            RubricTask(1, "请购买50元以内的白色保温杯"),
            RubricTask(2, "请购买50元以内的白色保温杯"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rubrics"
            first = run_rubric_generation(
                tasks=tasks,
                output_dir=output,
                client=client,
                rubric_set_version="rubric-v1",
                source_contract={"tasks_sha256": "test"},
                concurrency=2,
            )
            second = run_rubric_generation(
                tasks=tasks,
                output_dir=output,
                client=client,
                rubric_set_version="rubric-v1",
                source_contract={"tasks_sha256": "test"},
                concurrency=2,
            )

            self.assertTrue(first["complete"])
            self.assertEqual(first["auto_frozen"], 1)
            self.assertEqual(first["review_required"], 1)
            self.assertTrue(second["reused_existing_outputs"])
            self.assertEqual(sorted(calls), [1, 2])
            with self.assertRaisesRegex(ValueError, "没有决定"):
                finalize_rubric_set(generation_dir=output)

            decisions = output / "review-decisions.jsonl"
            decisions.write_text(
                json.dumps(
                    {
                        "task_id": 2,
                        "decision": "approve",
                        "reviewer": "tester",
                        "reason": "已核对用户原文",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            final = finalize_rubric_set(
                generation_dir=output,
                review_decisions_path=decisions,
            )
            rubric_rows = [
                json.loads(line)
                for line in (output / "rubrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(final["tasks"], 2)
        self.assertEqual(final["human_reviewed"], 1)
        self.assertEqual([row["task_id"] for row in rubric_rows], [1, 2])

    def test_resume_reuses_success_and_only_retries_failed_task(self):
        calls = {1: 0, 2: 0}

        def transport(url, payload, headers, timeout):
            del url, headers, timeout
            task_id = json.loads(payload["messages"][1]["content"])["task_id"]
            calls[task_id] += 1
            result = _items()
            if task_id == 2 and calls[task_id] == 1:
                result["items"][0]["type"] = "unsupported-type"
            return _response(result)

        client = DeepSeekRubricClient(
            model="deepseek-v4-flash",
            base_url="https://teacher.example/v1",
            api_key="secret",
            transport=transport,
        )
        tasks = [
            RubricTask(1, "请购买50元以内的白色保温杯"),
            RubricTask(2, "请购买50元以内的白色保温杯"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rubrics"
            first = run_rubric_generation(
                tasks=tasks,
                output_dir=output,
                client=client,
                rubric_set_version="rubric-v1",
                source_contract={"tasks_sha256": "test"},
            )
            second = run_rubric_generation(
                tasks=tasks,
                output_dir=output,
                client=client,
                rubric_set_version="rubric-v1",
                source_contract={"tasks_sha256": "test"},
            )

        self.assertFalse(first["complete"])
        self.assertEqual(first["remaining_task_ids"], [2])
        self.assertTrue(second["complete"])
        self.assertEqual(calls, {1: 1, 2: 2})


class RubricCliSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = REPOSITORY / "scripts" / "generate_eval_rubrics.py"
        spec = importlib.util.spec_from_file_location("generate_eval_rubrics", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("无法加载 Rubric CLI")
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def test_cli_has_no_plaintext_api_key_argument(self):
        parser = self.cli.build_parser()
        texts = [parser.format_help()]
        action = next(
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices
        )
        texts.extend(child.format_help() for child in action.choices.values())
        text = "\n".join(texts)

        self.assertNotIn("--api-key ", text)
        self.assertIn("--api-key-file", text)

    def test_cli_reuses_deepseek_environment_and_checks_key_permissions(self):
        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "deepseek-key"
            key_file.write_text("secret\n", encoding="utf-8")
            key_file.chmod(0o644)
            with self.assertRaises(SystemExit):
                self.cli._load_api_key(key_file)
            key_file.chmod(0o600)
            value, source = self.cli._load_api_key(key_file)
            self.assertEqual(value, "secret")
            self.assertIn("permission-checked", source)

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-secret"}):
            value, source = self.cli._load_api_key(None)
        self.assertEqual(value, "env-secret")
        self.assertIn("DEEPSEEK_API_KEY", source)


if __name__ == "__main__":
    unittest.main()
