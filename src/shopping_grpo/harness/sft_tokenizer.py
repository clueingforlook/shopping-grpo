"""把最终训练 tokenizer 接成 数据构建器需要的精确 Token 计数器。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
from typing import Any, Mapping

from shopping_grpo.training.sft.dataset import build_supervised_example
from shopping_grpo.harness.sft_deepseek_v4 import render_deepseek_v4_prompt
from shopping_grpo.harness.sft_metrics import TrainingSequenceUnrenderable


DEFAULT_DEEPSEEK_V4_TOKENIZER = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_DEEPSEEK_V4_TOKENIZER_REVISION = (
    "60d8d70770c6776ff598c94bb586a859a38244f1"
)
TRANSFORMERS_RUNTIME_COUNTER_VERSION = (
    "wlx-transformers-runtime-token-counter-v1"
)


class TransformersTrainingTokenCounter:
    """使用训练时同一 tokenizer 和 chat template 统计总 Token 与 Loss Token。"""

    def __init__(
        self,
        tokenizer: object,
        *,
        tokenizer_revision: str | None = None,
        chat_template: object | None = None,
    ) -> None:
        """保存已加载 tokenizer；这里只计数，不会加载模型权重或占用大量显存。"""

        if not callable(getattr(tokenizer, "__call__", None)):
            raise TypeError("tokenizer 必须可以把文本转换成 token")
        template = chat_template or tokenizer
        if not callable(getattr(template, "apply_chat_template", None)):
            raise TypeError("tokenizer 或 chat_template 必须提供 apply_chat_template")
        self.tokenizer = tokenizer
        self.chat_template = template
        self.tokenizer_name = str(getattr(tokenizer, "name_or_path", "unknown"))
        self.tokenizer_revision = tokenizer_revision
        self.chat_template_version = _template_fingerprint(template)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = True,
    ) -> "TransformersTrainingTokenCounter":
        """从模型名或本地目录加载 tokenizer；缺少 transformers 时给出清楚提示。"""

        try:
            from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "当前环境没有 transformers；请先安装项目的 sft 可选依赖"
            ) from exc
        load_kwargs = {"trust_remote_code": trust_remote_code}
        if revision:
            load_kwargs["revision"] = revision
        config = AutoConfig.from_pretrained(model_name_or_path, **load_kwargs)
        if str(getattr(config, "model_type", "")).startswith("qwen3_5"):
            if _multimodal_processor_runtime_available():
                # 训练环境使用完整 processor；其 tokenizer 和官方模板共同决定 labels。
                processor = AutoProcessor.from_pretrained(model_name_or_path, **load_kwargs)
                return cls(
                    processor.tokenizer,
                    tokenizer_revision=revision,
                    chat_template=processor,
                )
            # 轻量筛选环境不安装 torch/torchvision。Qwen3.5 官方仓库把同一份
            # chat template 同时写入 tokenizer_config.json 和 chat_template.jinja；
            # 当前数据只有文本和工具调用，因此可以只加载 tokenizer 进行精确计数。
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
            if not str(getattr(tokenizer, "chat_template", "") or "").strip():
                raise RuntimeError("Qwen3.5 tokenizer 缺少官方 chat template")
            return cls(tokenizer, tokenizer_revision=revision, chat_template=tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        return cls(tokenizer, tokenizer_revision=revision)

    def __call__(self, row: Mapping[str, Any]) -> tuple[int, int, Mapping[str, Any]]:
        """渲染完整训练序列，并统计全部 token 和真正参与 Loss 的 token。"""

        example = build_supervised_example(
            row.get("messages") or [],
            row.get("tools") or [],
            self.tokenizer,
            max_length=10**9,
            chat_template=self.chat_template,
        )
        if example is None:
            raise TrainingSequenceUnrenderable("这条轨迹无法被当前 chat template 无损渲染")
        input_ids = example["input_ids"]
        labels = example["labels"]
        return (
            len(input_ids),
            sum(1 for label in labels if int(label) != -100),
            {
                "tokenizer_name": self.tokenizer_name,
                "tokenizer_revision": self.tokenizer_revision,
                "chat_template_version": self.chat_template_version,
            },
        )


class TransformersRuntimeTokenCounter:
    """用 Teacher 自己的 tokenizer 计算每轮聊天输入和单个 Observation 长度。"""

    def __init__(self, tokenizer: object) -> None:
        """保存 tokenizer，并确认它同时支持普通编码和聊天模板。"""

        if not callable(getattr(tokenizer, "__call__", None)):
            raise TypeError("tokenizer 必须可以编码普通文本")
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer 必须提供 apply_chat_template")
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = True,
    ) -> "TransformersRuntimeTokenCounter":
        """只加载 Teacher tokenizer，不加载模型权重，因此通常不会占用 GPU 显存。"""

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "当前环境没有 transformers；请先安装项目的 sft 可选依赖"
            ) from exc
        load_kwargs = {"trust_remote_code": trust_remote_code}
        if revision:
            load_kwargs["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        return cls(tokenizer)

    def count_chat(self, messages: object, tools: object) -> int:
        """统计模型下一轮请求包含的历史消息、工具 Schema 和生成提示总长度。"""

        encoded = self.tokenizer.apply_chat_template(
            _messages_for_transformers_template(messages),
            tools=list(tools or []),
            tokenize=True,
            add_generation_prompt=True,
        )
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if (
            isinstance(token_ids, (list, tuple))
            and token_ids
            and isinstance(token_ids[0], (list, tuple))
        ):
            token_ids = token_ids[0]
        return len(token_ids)

    def count_text(self, text: object) -> int:
        """统计一段 Observation 或模型回复的 Token 数。"""

        encoded = self.tokenizer(str(text), add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        return len(input_ids)


def _messages_for_transformers_template(messages: object) -> list[dict[str, Any]]:
    """把 OpenAI 历史复制成 Transformers/Qwen 模板要求的形状。

    Chat Completions 协议中的 ``function.arguments`` 是 JSON 字符串；Qwen3.5
    的本地 Jinja 模板则会直接对它调用 ``items``，因此只在 token 计数副本中
    解码成对象。原始轨迹以及真正发送给模型服务的 OpenAI 消息都不能被修改。
    """

    normalised: list[dict[str, Any]] = []
    for raw_message in list(messages or []):
        if not isinstance(raw_message, Mapping):
            raise TypeError("聊天历史中的 message 必须是对象")
        message = deepcopy(dict(raw_message))
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            if not isinstance(tool_calls, list):
                raise TypeError("聊天历史中的 tool_calls 必须是列表")
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    raise TypeError("聊天历史中的 tool_call 必须是对象")
                function = tool_call.get("function")
                if not isinstance(function, Mapping):
                    raise TypeError("聊天历史中的 tool_call.function 必须是对象")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments or "{}")
                    except json.JSONDecodeError as exc:
                        raise ValueError("聊天历史中的工具参数不是合法 JSON") from exc
                if not isinstance(arguments, Mapping):
                    raise TypeError("聊天历史中的工具参数必须是 JSON 对象")
                function["arguments"] = deepcopy(dict(arguments))
        normalised.append(message)
    return normalised


class DeepSeekV4RuntimeTokenCounter:
    """用轻量 tokenizer 和 V4 官方 Encoding 统计 Teacher 的上下文长度。"""

    def __init__(
        self,
        tokenizer: object,
        *,
        tokenizer_name: str,
        tokenizer_revision: str | None,
        thinking_mode: str = "enabled",
        reasoning_effort: str = "high",
    ) -> None:
        """保存轻量 tokenizer，并固定本次采样采用的思考模式和推理强度。"""

        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError("tokenizer 必须提供 encode 方法")
        if thinking_mode not in {"enabled", "disabled"}:
            raise ValueError("thinking_mode 只能是 enabled 或 disabled")
        if reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort 只能是 high 或 max")
        self.tokenizer = tokenizer
        self.tokenizer_name = str(tokenizer_name)
        self.tokenizer_revision = tokenizer_revision
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = DEFAULT_DEEPSEEK_V4_TOKENIZER_REVISION,
        thinking_mode: str = "enabled",
        reasoning_effort: str = "high",
    ) -> "DeepSeekV4RuntimeTokenCounter":
        """只加载约几 MB 的 tokenizer.json，不碰 Teacher 模型权重和 GPU。"""

        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "当前环境没有 tokenizers；请先安装项目的 sft 可选依赖"
            ) from exc
        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            tokenizer_file = local_path / "tokenizer.json"
            if not tokenizer_file.is_file():
                raise FileNotFoundError(f"目录中找不到 tokenizer.json：{local_path}")
            tokenizer = Tokenizer.from_file(str(tokenizer_file))
        elif local_path.is_file():
            tokenizer = Tokenizer.from_file(str(local_path))
        else:
            tokenizer = Tokenizer.from_pretrained(
                model_name_or_path,
                revision=revision or "main",
            )
        return cls(
            tokenizer,
            tokenizer_name=model_name_or_path,
            tokenizer_revision=revision,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )

    def count_chat(self, messages: object, tools: object) -> int:
        """按 DeepSeek V4 的工具和 Thinking 格式渲染整段历史后统计 Token。"""

        prompt = render_deepseek_v4_prompt(
            list(messages or []),
            list(tools or []),
            thinking_mode=(
                "thinking" if self.thinking_mode == "enabled" else "chat"
            ),
            reasoning_effort=self.reasoning_effort,
        )
        return len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)

    def count_text(self, text: object) -> int:
        """统计一段环境返回或模型回复的 Token，供 Observation Projection 使用。"""

        return len(
            self.tokenizer.encode(str(text), add_special_tokens=False).ids
        )


def _template_fingerprint(template: object) -> str:
    """给 chat template 做短哈希，模板内容变化后数据版本也能被发现。"""

    raw = getattr(template, "chat_template", None)
    if raw is None:
        raw = template.__class__.__qualname__
    return "sha256:" + hashlib.sha256(str(raw).encode("utf-8")).hexdigest()


def _multimodal_processor_runtime_available() -> bool:
    """完整 Qwen processor 需要 torch 和 torchvision；文本筛选不安装它们。"""

    return find_spec("torch") is not None and find_spec("torchvision") is not None


__all__ = [
    "DEFAULT_DEEPSEEK_V4_TOKENIZER",
    "DEFAULT_DEEPSEEK_V4_TOKENIZER_REVISION",
    "DeepSeekV4RuntimeTokenCounter",
    "TransformersRuntimeTokenCounter",
    "TransformersTrainingTokenCounter",
    "TRANSFORMERS_RUNTIME_COUNTER_VERSION",
]
