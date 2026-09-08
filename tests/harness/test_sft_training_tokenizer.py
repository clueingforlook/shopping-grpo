"""离线检查最终训练 tokenizer 的轻量加载与按行拒绝契约。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[2]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness import sft_tokenizer as TOKENIZER  # noqa: E402
from shopping_grpo.harness.sft_metrics import TrainingSequenceUnrenderable  # noqa: E402


class _FakeTokenizer:
    """提供计数器构造所需的最小 tokenizer 接口。"""

    name_or_path = "Qwen/Qwen3.5-2B"
    chat_template = "official-qwen-template"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1]}

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"


class _QwenRuntimeTokenizer(_FakeTokenizer):
    def __init__(self):
        self.seen_messages = None

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.seen_messages = messages
        arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
        if not isinstance(arguments, dict):
            raise TypeError("Can only get item pairs from a mapping.")
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }


class TrainingTokenizerTest(unittest.TestCase):
    def test_runtime_counter_decodes_openai_tool_arguments_for_qwen_template(self):
        """第二轮计数临时解码参数，但不改写将发送给 API 的历史消息。"""

        tokenizer = _QwenRuntimeTokenizer()
        counter = TOKENIZER.TransformersRuntimeTokenCounter(tokenizer)
        messages = [
            {"role": "user", "content": "买一把木梳"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": json.dumps(
                                {"query": "木梳礼盒"}, ensure_ascii=False
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "搜索结果",
            },
        ]

        self.assertEqual(counter.count_chat(messages, []), 3)
        converted = tokenizer.seen_messages[1]["tool_calls"][0]["function"]
        self.assertEqual(converted["arguments"], {"query": "木梳礼盒"})
        original = messages[1]["tool_calls"][0]["function"]
        self.assertIsInstance(original["arguments"], str)

    def test_qwen_text_only_filtering_skips_multimodal_processor(self):
        """无 torch 时用官方 tokenizer 内嵌模板，不触发视觉 processor。"""

        fake = _FakeTokenizer()
        with (
            patch.object(TOKENIZER, "_multimodal_processor_runtime_available", return_value=False),
            patch(
                "transformers.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(model_type="qwen3_5"),
            ),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=fake) as auto_tokenizer,
            patch("transformers.AutoProcessor.from_pretrained") as auto_processor,
        ):
            counter = TOKENIZER.TransformersTrainingTokenCounter.from_pretrained(
                "Qwen/Qwen3.5-2B",
                revision="revision-test",
            )
        self.assertIs(counter.tokenizer, fake)
        self.assertIs(counter.chat_template, fake)
        auto_tokenizer.assert_called_once()
        auto_processor.assert_not_called()

    def test_unrenderable_example_raises_dedicated_sample_exception(self):
        """模板坏样本使用专用异常，构建器才能只拒绝该行。"""

        counter = TOKENIZER.TransformersTrainingTokenCounter(_FakeTokenizer())
        with patch.object(TOKENIZER, "build_supervised_example", return_value=None):
            with self.assertRaises(TrainingSequenceUnrenderable):
                counter({"messages": [], "tools": []})


if __name__ == "__main__":
    unittest.main()
