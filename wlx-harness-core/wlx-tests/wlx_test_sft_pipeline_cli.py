"""离线检查 WLX SFT 分阶段命令行，不连接模型或 ShopSimulator。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[2]
for import_path in (REPOSITORY / "src", REPOSITORY / "wlx-harness-core"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def _load_cli_module():
    """按文件路径加载新 CLI，避免要求它先安装成 Python 包。"""

    path = REPOSITORY / "scripts" / "wlx_sft_data_pipeline.py"
    spec = importlib.util.spec_from_file_location("wlx_sft_data_pipeline_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载 WLX SFT CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = _load_cli_module()


class SftPipelineCliTest(unittest.TestCase):
    """检查子命令边界、公开计划输出和密钥配置保护。"""

    def test_parser_exposes_every_pipeline_stage(self):
        """验证难度、计划、采样和构建都有独立子命令。"""

        parser = CLI.build_parser()
        subparsers_action = next(
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        )
        expected = {
            "check-config",
            "precompute-difficulty",
            "plan-calibration",
            "fit-difficulty",
            "plan-formal",
            "collect",
            "build",
        }
        self.assertEqual(set(subparsers_action.choices), expected)

    def test_plan_calibration_writes_only_public_task_fields(self):
        """验证校准计划不会把输入中多余的 target_asin 写到输出。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = root / "wlx-tasks.jsonl"
            features = root / "wlx-features.jsonl"
            output = root / "wlx-plan.jsonl"
            task_rows = [
                {
                    "task_id": task_id,
                    "instruction": f"task {task_id}",
                    "official_split": "train",
                    "category": "test",
                    "target_asin": f"hidden-{task_id}",
                }
                for task_id in range(6)
            ]
            feature_rows = [
                {
                    "task_id": task_id,
                    "constraint_count": task_id + 1,
                    "option_axis_count": 0,
                    "has_brand": False,
                    "has_model": False,
                    "has_budget": True,
                    "retrieval_score": task_id / 10,
                    "near_miss_score": 0.1,
                    "constraint_score": task_id / 10,
                    "preliminary_score": task_id / 10,
                    "preliminary_label": ("easy" if task_id < 2 else "hard" if task_id > 3 else "medium"),
                    "category": "test",
                    "evidence": {},
                }
                for task_id in range(6)
            ]
            tasks.write_text(
                "".join(json.dumps(row) + "\n" for row in task_rows),
                encoding="utf-8",
            )
            features.write_text(
                "".join(json.dumps(row) + "\n" for row in feature_rows),
                encoding="utf-8",
            )
            exit_code = CLI.main(
                [
                    "plan-calibration",
                    "--tasks",
                    str(tasks),
                    "--features",
                    str(features),
                    "--output",
                    str(output),
                    "--size",
                    "4",
                ]
            )
            self.assertEqual(exit_code, 0)
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("target_asin", text)
            self.assertEqual(len([line for line in text.splitlines() if line]), 4)

    def test_collect_requires_key_from_environment_not_command_line(self):
        """验证在线采样缺少环境变量时会停下，而 CLI 本身没有明文 Key 参数。"""

        parser_text = parser_help_text(CLI.build_parser())
        self.assertNotIn("--api-key ", parser_text)
        self.assertIn("--api-key-file", parser_text)
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "",
                "DEEPSEEK_BASE_URL": "",
                "DEEPSEEK_MODEL": "",
                "WLX_TEACHER_TOKENIZER": "",
            },
            clear=False,
        ):
            with self.assertRaises(SystemExit) as captured:
                CLI.main(
                    [
                        "collect",
                        "--tasks",
                        "missing.jsonl",
                        "--raw",
                        "raw.jsonl",
                        "--mode",
                        "calibration",
                    ]
                )
        self.assertIn("DEEPSEEK_API_KEY", str(captured.exception))

    def test_key_file_must_have_private_permissions(self):
        """验证持久化 Key 文件权限不是 600 时，采样器会拒绝读取。"""

        with tempfile.TemporaryDirectory() as temporary:
            key_file = Path(temporary) / "wlx-deepseek-key"
            key_file.write_text("secret-value\n", encoding="utf-8")
            key_file.chmod(0o644)
            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "environment-must-not-win"},
                clear=False,
            ):
                with self.assertRaises(SystemExit):
                    CLI._load_api_key(key_file)
                key_file.chmod(0o600)
                value, source = CLI._load_api_key(key_file)
            self.assertEqual(value, "secret-value")
            self.assertEqual(source, "permission-checked key file")


def parser_help_text(parser) -> str:
    """把主命令和 collect 子命令帮助合在一起，检查是否暴露明文 Key 参数。"""

    parts = [parser.format_help()]
    subparsers_action = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    parts.append(subparsers_action.choices["collect"].format_help())
    return "\n".join(parts)


if __name__ == "__main__":
    unittest.main()
