"""离线验证 DeepSeek V4 提示词渲染和轻量 Token 计数。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.sft_deepseek_v4 import (  # noqa: E402
    render_deepseek_v4_prompt,
)
from shopping_grpo.harness.sft_tokenizer import (  # noqa: E402
    DeepSeekV4RuntimeTokenCounter,
)


class _FakeEncoding:
    """模拟轻量 tokenizer 返回的 Encoding，只暴露测试需要的 ids。"""

    def __init__(self, text: str) -> None:
        """让每个字符对应一个假 Token，方便直接核对长度。"""

        self.ids = list(range(len(text)))


class _FakeTokenizer:
    """模拟 tokenizers.Tokenizer，测试时不下载 Hugging Face 文件。"""

    def encode(self, text: str, *, add_special_tokens: bool) -> _FakeEncoding:
        """返回按字符计数的假 Encoding，并检查没有重复添加 BOS。"""

        if add_special_tokens:
            raise AssertionError("提示词已经带 BOS，不应再次添加特殊 Token")
        return _FakeEncoding(text)


def _tools() -> list[dict]:
    """生成一项最小搜索工具 Schema，供所有渲染测试复用。"""

    return [
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "搜索商品",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]


class DeepSeekV4EncodingTest(unittest.TestCase):
    """检查工具、思考内容和环境返回不会在计数前丢失。"""

    def test_render_keeps_reasoning_tool_call_and_tool_result(self):
        """验证一轮完整购物交互会变成 DeepSeek 官方 DSML 结构。"""

        messages = [
            {"role": "system", "content": "你是购物助手"},
            {"role": "user", "content": "买一个枕头"},
            {
                "role": "assistant",
                "content": "先搜索。",
                "reasoning_content": "需要找到候选商品。",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query": "枕头"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "1|A1|枕头",
            },
        ]
        prompt = render_deepseek_v4_prompt(
            messages,
            _tools(),
            thinking_mode="thinking",
            reasoning_effort="high",
        )
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>你是购物助手"))
        self.assertIn("需要找到候选商品。</think>先搜索。", prompt)
        self.assertIn('<｜DSML｜invoke name="search_products">', prompt)
        self.assertIn("<tool_result>1|A1|枕头</tool_result>", prompt)
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))

    def test_non_thinking_mode_starts_direct_answer(self):
        """验证关闭私有推理时，新一轮回答直接从结束 think 标记后开始。"""

        prompt = render_deepseek_v4_prompt(
            [{"role": "user", "content": "搜索枕头"}],
            _tools(),
            thinking_mode="chat",
        )
        self.assertTrue(prompt.endswith("<｜Assistant｜></think>"))

    def test_runtime_counter_counts_rendered_chat_and_plain_text(self):
        """验证 Counter 的聊天计数包含工具说明，普通文本计数仍可单独使用。"""

        counter = DeepSeekV4RuntimeTokenCounter(
            _FakeTokenizer(),
            tokenizer_name="fake",
            tokenizer_revision="test",
            thinking_mode="enabled",
        )
        messages = [{"role": "user", "content": "搜索枕头"}]
        rendered = render_deepseek_v4_prompt(
            messages,
            _tools(),
            thinking_mode="thinking",
        )
        self.assertEqual(counter.count_chat(messages, _tools()), len(rendered))
        self.assertEqual(counter.count_text("环境返回"), 4)


if __name__ == "__main__":
    unittest.main()
