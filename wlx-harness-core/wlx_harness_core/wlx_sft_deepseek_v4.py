"""用 DeepSeek V4 官方格式把购物轨迹渲染成真正送入模型的提示词。

这里实现的是本项目会用到的官方 Encoding 子集：system、user、assistant、tool、
Thinking 和函数工具。格式改写自 DeepSeek-V4 官方实现：
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/
60d8d70770c6776ff598c94bb586a859a38244f1/encoding/encoding_dsv4.py

DeepSeek 衍生部分保留以下 MIT 许可声明：

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINK_START_TOKEN = "<think>"
THINK_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the "
    "problem to resolve the root cause, rigorously stress-testing your logic against "
    "all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every "
    "intermediate step, considered alternative, and rejected hypothesis to ensure "
    "absolutely no assumption is left unchecked.\n\n"
)

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}tool_calls>" block like the following:

<{dsml}tool_calls>
<{dsml}invoke name="$TOOL_NAME">
<{dsml}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}parameter>
...
</{dsml}invoke>
<{dsml}invoke name="$TOOL_NAME2">
...
</{dsml}invoke>
</{dsml}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def render_deepseek_v4_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: str,
    reasoning_effort: str = "high",
) -> str:
    """把 OpenAI 风格消息和工具表拼成 DeepSeek V4 实际计算 Token 的整段文字。"""

    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("thinking_mode 只能是 chat 或 thinking")
    if reasoning_effort not in {"high", "max"}:
        raise ValueError("reasoning_effort 只能是 high 或 max")
    copied = [deepcopy(dict(message)) for message in messages]
    if tools:
        _attach_tools(copied, tools)
    merged = _merge_tool_messages(copied)
    drop_old_thinking = not any(message.get("tools") for message in merged)
    prompt = BOS_TOKEN
    for index in range(len(merged)):
        prompt += _render_message(
            index,
            merged,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            drop_old_thinking=drop_old_thinking,
        )
    return prompt


def _attach_tools(
    messages: list[dict[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> None:
    """把 API 顶层的工具表放进 system 消息，模拟 DeepSeek 服务端的编码方式。"""

    copied_tools = [deepcopy(dict(tool)) for tool in tools]
    for message in messages:
        if message.get("role") == "system":
            message["tools"] = copied_tools
            return
    messages.insert(0, {"role": "system", "content": "", "tools": copied_tools})


def _merge_tool_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把独立 tool 消息变成 DeepSeek 使用的 user/tool_result 内容块。"""

    merged: list[dict[str, Any]] = []
    for original in messages:
        message = deepcopy(dict(original))
        role = message.get("role")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": message.get("content", ""),
            }
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(block)
            else:
                merged.append({"role": "user", "content_blocks": [block]})
            continue
        if role == "user":
            block = {"type": "text", "text": message.get("content", "")}
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(block)
            else:
                message["content_blocks"] = [block]
                merged.append(message)
            continue
        merged.append(message)
    return merged


def _render_message(
    index: int,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: str,
    reasoning_effort: str,
    drop_old_thinking: bool,
) -> str:
    """渲染一条消息，并在需要模型继续回答的位置补上 Assistant 起始标记。"""

    message = messages[index]
    role = message.get("role")
    prompt = ""
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
        prompt += REASONING_EFFORT_MAX
    if role == "system":
        prompt += str(message.get("content") or "")
        if message.get("tools"):
            prompt += "\n\n" + _render_tools(message["tools"])
    elif role == "user":
        prompt += USER_TOKEN + _render_user_content(message)
    elif role == "assistant":
        prompt += _render_assistant_content(
            index,
            messages,
            thinking_mode=thinking_mode,
            drop_old_thinking=drop_old_thinking,
        )
    else:
        raise ValueError(f"DeepSeek V4 计数器不支持消息角色：{role!r}")

    has_next = index + 1 < len(messages)
    next_role = messages[index + 1].get("role") if has_next else None
    if has_next and next_role != "assistant":
        return prompt
    if role == "user":
        prompt += ASSISTANT_TOKEN
        if thinking_mode == "thinking" and (
            not drop_old_thinking or index >= _last_user_index(messages)
        ):
            prompt += THINK_START_TOKEN
        else:
            prompt += THINK_END_TOKEN
    return prompt


def _render_user_content(message: Mapping[str, Any]) -> str:
    """把普通用户文字和环境工具返回按官方的 content block 格式拼起来。"""

    blocks = message.get("content_blocks")
    if not blocks:
        return str(message.get("content") or "")
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_result":
            parts.append(f"<tool_result>{block.get('content') or ''}</tool_result>")
        else:
            raise ValueError(f"不支持的 DeepSeek 内容块：{block.get('type')!r}")
    return "\n\n".join(parts)


def _render_assistant_content(
    index: int,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: str,
    drop_old_thinking: bool,
) -> str:
    """拼接一轮模型的私有推理、公开文字和函数调用，并补上本轮结束标记。"""

    message = messages[index]
    reasoning = ""
    if thinking_mode == "thinking" and (
        not drop_old_thinking or index > _last_user_index(messages)
    ):
        reasoning = str(message.get("reasoning_content") or "") + THINK_END_TOKEN
    content = str(message.get("content") or "")
    tool_calls = _render_tool_calls(message.get("tool_calls") or [])
    return reasoning + content + tool_calls + EOS_TOKEN


def _render_tool_calls(tool_calls: Sequence[Mapping[str, Any]]) -> str:
    """把 OpenAI 函数调用转成 DeepSeek V4 使用的 DSML 工具调用块。"""

    if not tool_calls:
        return ""
    rendered: list[str] = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        name = function.get("name")
        arguments = _render_arguments(function.get("arguments", "{}"))
        rendered.append(
            f'<{DSML_TOKEN}invoke name="{name}">\n{arguments}\n'
            f"</{DSML_TOKEN}invoke>"
        )
    body = "\n".join(rendered)
    return (
        f"\n\n<{DSML_TOKEN}tool_calls>\n{body}\n"
        f"</{DSML_TOKEN}tool_calls>"
    )


def _render_arguments(raw_arguments: object) -> str:
    """把函数参数拆成逐项 DSML；坏 JSON 也会原样保留，方便后续判定无效。"""

    if isinstance(raw_arguments, Mapping):
        arguments = dict(raw_arguments)
    else:
        try:
            parsed = json.loads(str(raw_arguments))
            arguments = parsed if isinstance(parsed, Mapping) else {"arguments": parsed}
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = {"arguments": str(raw_arguments)}
    parts: list[str] = []
    for key, value in arguments.items():
        is_string = isinstance(value, str)
        rendered_value = value if is_string else _json(value)
        parts.append(
            f'<{DSML_TOKEN}parameter name="{key}" '
            f'string="{str(is_string).lower()}">{rendered_value}'
            f"</{DSML_TOKEN}parameter>"
        )
    return "\n".join(parts)


def _render_tools(tools: Sequence[Mapping[str, Any]]) -> str:
    """把 OpenAI 工具 Schema 放进 DeepSeek 官方规定的工具说明模板。"""

    schemas = []
    for tool in tools:
        function = tool.get("function")
        schemas.append(_json(function if isinstance(function, Mapping) else tool))
    return TOOLS_TEMPLATE.format(
        dsml=DSML_TOKEN,
        think_start=THINK_START_TOKEN,
        think_end=THINK_END_TOKEN,
        tool_schemas="\n".join(schemas),
    )


def _last_user_index(messages: Sequence[Mapping[str, Any]]) -> int:
    """找到最后一条 user 消息，决定哪些旧推理需要保留。"""

    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return -1


def _json(value: object) -> str:
    """用稳定的中文友好 JSON 表示工具 Schema 和非字符串参数。"""

    return json.dumps(value, ensure_ascii=False)


__all__ = ["render_deepseek_v4_prompt"]
