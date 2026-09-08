#!/usr/bin/env python3
"""对验收后的 Shopping tool-calling 数据进行最小 LoRA SFT。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time as _time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

from shopping_grpo.training.sft.dataset import load_supervised_examples

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Qwen3.5 的大多数文本层是 Gated DeltaNet，不能遗漏其线性注意力投影。
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)

GIB = 1024**3
METRICS_FILENAME = "metrics.jsonl"
RUN_SUMMARY_FILENAME = "run-summary.json"
SWANLAB_DIRNAME = "swanlab"


def parse_args():
    parser = argparse.ArgumentParser(description="使用 Transformers + PEFT 执行 Shopping LoRA SFT")
    parser.add_argument("--model", required=True, help="Hugging Face 模型名或本地模型目录")
    parser.add_argument("--train", type=Path, required=True, help="训练 SFT JSONL")
    parser.add_argument("--validation", type=Path, default=None, help="可选验证 SFT JSONL")
    parser.add_argument("--output", type=Path, required=True, help="LoRA adapter 输出目录")
    # 24k 可保留当前真实轨迹的约 93%，48G 显存配合 batch=1 与梯度检查点可稳定训练。
    parser.add_argument("--max-length", type=int, default=24576)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="模型与训练精度；auto 在 CUDA 上优先 bf16，其次 fp16，CPU 使用 fp32。",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="兼容旧命令；等价于 --dtype bf16，不能与其他 --dtype 同时使用。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选模型 revision；本地路径通常不需要。",
    )
    parser.add_argument("--liger-kernel", action="store_true", help="启用 Liger 融合 loss，避免全序列 logits 常驻")
    parser.add_argument(
        "--attention-implementation",
        choices=("auto", "sdpa"),
        default="auto",
        help="注意力后端；sdpa 使用 PyTorch 原生内存高效实现，不要求编译 FlashAttention 2。",
    )
    parser.add_argument("--qlora", action="store_true", help="以 NF4 4-bit 加载基座，并按 PEFT 标准预处理")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-steps", type=int, default=-1, help="最大训练步数（-1=完整 epoch）；用于冒烟测试")
    parser.add_argument("--swanlab", action="store_true", help="启用 SwanLab 训练监控")
    parser.add_argument("--swanlab-project", default="shopping-grpo", help="SwanLab project 名")
    parser.add_argument("--swanlab-run-name", default=None, help="SwanLab run 名；默认自动生成")
    parser.add_argument(
        "--swanlab-mode",
        choices=("online", "local"),
        default="online",
        help="SwanLab 在线同步或只保存在本地；仅 --swanlab 时生效。",
    )
    return parser.parse_args()


def _training_dependencies():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("缺少训练依赖。请执行：uv sync --extra sft") from exc
    return (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )


def _model_load_kwargs(args, dtype, bits_and_bytes_config):
    """构造可审计的模型加载参数；加速功能必须显式开启。"""
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.attention_implementation != "auto":
        kwargs["attn_implementation"] = args.attention_implementation
    if args.qlora:
        kwargs["quantization_config"] = bits_and_bytes_config(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def _prepare_model_for_training(model, args, prepare_model_for_kbit_training):
    """按 PEFT 推荐顺序准备量化模型与梯度检查点。"""
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if not args.qlora and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def _validate_optional_training_dependencies(args):
    """仅在所选实验需要时检查可选加速包，保持基础 LoRA 环境轻量。"""
    if args.qlora:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--qlora 需要 bitsandbytes；请执行："
                "uv sync --extra sft --extra sft-accelerated"
            ) from exc
    if args.liger_kernel:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "--liger-kernel 需要 liger-kernel；请执行："
                "uv sync --extra sft --extra sft-accelerated"
            ) from exc


def _resolve_dtype(args, torch):
    """Resolve one explicit dtype for model loading and TrainingArguments."""

    requested = args.dtype
    if args.bf16:
        if requested not in {"auto", "bf16"}:
            raise SystemExit("--bf16 cannot be combined with a non-bf16 --dtype")
        requested = "bf16"
    if requested == "auto":
        if torch.cuda.is_available():
            requested = (
                "bf16"
                if torch.cuda.is_bf16_supported()
                else "fp16"
            )
        else:
            requested = "fp32"
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return requested, mapping[requested]


def _swanlab_config(args):
    """准备官方 Transformers 集成所需的最小 SwanLab 配置。"""
    if not args.swanlab:
        return "none", None
    try:
        import swanlab  # noqa: F401 - 仅验证可选依赖存在。
    except ImportError as exc:
        raise SystemExit("缺少 SwanLab。请执行：uv sync --extra sft") from exc
    if args.swanlab_mode == "local":
        try:
            import swanboard  # noqa: F401 - local dashboard 运行时依赖。
        except ImportError as exc:
            raise SystemExit(
                "SwanLab local 模式缺少 SwanBoard。请执行：uv sync --extra sft"
            ) from exc

    run_name = args.swanlab_run_name or (
        f"lora-r{args.lora_r}-bs{args.per_device_train_batch_size}"
        f"x{args.gradient_accumulation_steps}-lr{args.learning_rate}"
    )
    return "swanlab", run_name


def _loss_only_eval_trainer_class(trainer_base, enable_skip_logits):
    """构造只在 loss-only 验证时显式跳过完整词表 logits 的 Trainer。

    Qwen3.5 的 Liger forward 默认只在 ``model.training`` 时启用融合
    LM-head + cross-entropy；Trainer 验证会先调用 ``model.eval()``，即使最终
    只需要 eval_loss，也会物化 ``[batch, sequence, vocab]`` logits。20K 上下文
    和 248K 词表会因此产生约 20 GiB 的瞬时 FP32 张量。

    ``skip_logits`` 是 Liger Qwen3.5 forward 的公开参数。这里只在 Trainer 已经
    明确 ``prediction_loss_only=True`` 且输入含 labels 时传入，不改变训练前向，
    也不影响需要 predictions/metrics 的评估。
    """

    class LossOnlyEvalTrainer(trainer_base):
        def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=None,
        ):
            if enable_skip_logits and prediction_loss_only and inputs.get("labels") is not None:
                inputs = dict(inputs)
                inputs["skip_logits"] = True
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

    return LossOnlyEvalTrainer


def _metrics_logging_trainer_class(trainer_base, metrics_provider):
    """在 Trainer 持久化和上报日志前注入自定义训练指标。

    Transformers 的 ``Trainer.log`` 会先复制指标到 ``state.log_history``，再调用
    SwanLab 等 callback。因此仅在 callback 的 ``on_log`` 中修改 ``logs`` 会导致
    自定义指标既不进入 checkpoint，也不进入 SwanLab。这里在调用父类 ``log``
    之前注入，确保两处拿到完全相同的数据。
    """

    class MetricsLoggingTrainer(trainer_base):
        def log(self, logs, start_time=None):
            enriched_logs = dict(logs)
            if "loss" in enriched_logs:
                enriched_logs.update(metrics_provider())
            return super().log(enriched_logs, start_time=start_time)

    return MetricsLoggingTrainer


def _progress_callback_class(callback_base, torch, clock=_time.perf_counter):
    """构造按 optimizer step 采样耗时和显存的 Trainer callback。"""

    class ProgressCallback(callback_base):
        def __init__(self):
            self.step_start = None
            self.epoch_start = None
            self.latest_metrics = {}
            self.pending_step_time = 0.0
            self.pending_step_count = 0

        @staticmethod
        def _cuda_available():
            return bool(torch.cuda.is_available())

        def on_step_begin(self, args, state, control, **kwargs):
            if self._cuda_available():
                # CUDA kernel 默认异步；同步后再计时才能得到真实 optimizer-step 耗时。
                torch.cuda.synchronize()
            self.step_start = clock()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self._cuda_available():
                torch.cuda.synchronize()
            elapsed = clock() - self.step_start if self.step_start is not None else 0.0
            elapsed = max(elapsed, 0.0)
            self.pending_step_time += elapsed
            self.pending_step_count += 1
            metrics = {}
            if self._cuda_available():
                metrics.update(
                    {
                        "gpu_memory_allocated_gib": round(
                            torch.cuda.memory_allocated() / GIB, 3
                        ),
                        "gpu_memory_reserved_gib": round(
                            torch.cuda.memory_reserved() / GIB, 3
                        ),
                        "gpu_peak_memory_allocated_gib": round(
                            torch.cuda.max_memory_allocated() / GIB, 3
                        ),
                        "gpu_peak_memory_reserved_gib": round(
                            torch.cuda.max_memory_reserved() / GIB, 3
                        ),
                    }
                )
            self.latest_metrics = metrics
            return control

        def metrics_for_log(self):
            metrics = dict(self.latest_metrics)
            if self.pending_step_count:
                metrics["step_time_s"] = round(
                    self.pending_step_time / self.pending_step_count, 3
                )
                metrics["step_time_sample_count"] = self.pending_step_count
            self.pending_step_time = 0.0
            self.pending_step_count = 0
            return metrics

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs or "loss" not in logs:
                return control
            elapsed = float(logs.get("step_time_s", 0.0))
            gpu_mem = float(logs.get("gpu_memory_allocated_gib", 0.0))
            gpu_peak = float(logs.get("gpu_peak_memory_allocated_gib", 0.0))
            eta_seconds = (state.max_steps - state.global_step) * elapsed
            eta_str = f"{eta_seconds / 60:.0f}min" if eta_seconds > 0 else "0min"
            print(
                f"[step {state.global_step}/{state.max_steps}] "
                f"loss={float(logs['loss']):.4f} step_t={elapsed:.1f}s "
                f"GPU={gpu_mem:.1f}GiB peak={gpu_peak:.1f}GiB ETA={eta_str}"
            )
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_start = clock()
            epoch = int(state.epoch or 0) + 1
            print(f"\n{'=' * 60}\n  EPOCH {epoch} 开始  steps={state.max_steps}\n{'=' * 60}")
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch_time = clock() - self.epoch_start if self.epoch_start is not None else 0.0
            print(f"  EPOCH {int(state.epoch or 0)} 完成  耗时={epoch_time / 60:.1f}min")
            return control

    return ProgressCallback


def _metrics_jsonl_callback_class(callback_base, output_path):
    """将每次 Trainer 日志事件实时追加到可独立分析的 JSONL。"""

    class MetricsJsonlCallback(callback_base):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return control
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "step": state.global_step,
                **logs,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                stream.flush()
            return control

    return MetricsJsonlCallback


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_fingerprint(path):
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _model_fingerprint(model_name):
    """对本地模型的非隐藏文件生成逐文件和聚合 SHA-256。"""

    path = Path(model_name).expanduser()
    if not path.exists():
        return {
            "identifier": str(model_name),
            "local": False,
            "sha256": None,
            "note": "remote model identifier; local artifacts were not hashable",
        }
    if path.is_file():
        return {"local": True, **_file_fingerprint(path)}

    root = path.resolve()
    files = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and not any(part.startswith(".") for part in candidate.relative_to(root).parts)
    )
    manifest_digest = hashlib.sha256()
    file_records = {}
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        fingerprint = _file_fingerprint(candidate)
        file_records[relative] = {
            "bytes": fingerprint["bytes"],
            "sha256": fingerprint["sha256"],
        }
        manifest_digest.update(
            f"{relative}\0{fingerprint['bytes']}\0{fingerprint['sha256']}\n".encode()
        )
    return {
        "path": str(root),
        "local": True,
        "sha256": manifest_digest.hexdigest(),
        "sha256_type": "manifest(relative_path,bytes,file_sha256)",
        "files": file_records,
    }


def _adapter_fingerprint(output_dir):
    """只哈希 adapter 根目录产物，避免把 checkpoint 和监控日志递归算入。"""

    root = Path(output_dir).resolve()
    files = sorted(
        candidate
        for candidate in root.iterdir()
        if candidate.is_file()
        and candidate.name not in {METRICS_FILENAME, RUN_SUMMARY_FILENAME}
        and not candidate.name.startswith("wlx-")  # Historical run records.
    )
    manifest_digest = hashlib.sha256()
    file_records = {}
    for candidate in files:
        fingerprint = _file_fingerprint(candidate)
        file_records[candidate.name] = {
            "bytes": fingerprint["bytes"],
            "sha256": fingerprint["sha256"],
        }
        manifest_digest.update(
            f"{candidate.name}\0{fingerprint['bytes']}\0{fingerprint['sha256']}\n".encode()
        )
    return {
        "path": str(root),
        "sha256": manifest_digest.hexdigest(),
        "sha256_type": "manifest(filename,bytes,file_sha256)",
        "files": file_records,
    }


def _git_metadata(repo_root):
    def run_git(*arguments):
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *arguments],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run_git("status", "--porcelain")
    diff = run_git("diff", "--binary", "HEAD")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status": status.splitlines() if status else [],
        "tracked_diff_sha256": (
            hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None
        ),
    }


def _environment_metadata(torch):
    package_names = (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "swanlab",
        "swanboard",
    )
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / GIB, 3),
                    "compute_capability": f"{capability[0]}.{capability[1]}",
                }
            )

    driver_version = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            driver_version = sorted(set(result.stdout.split()))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    cudnn_version = None
    if getattr(torch.backends, "cudnn", None) is not None:
        cudnn_version = torch.backends.cudnn.version()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": cudnn_version,
        "nvidia_driver": driver_version,
        "gpus": gpus,
    }


def _evaluation_summary(log_history, output_dir, best_metric=None, best_checkpoint=None):
    evaluations = []
    for entry in log_history:
        if "eval_loss" not in entry:
            continue
        evaluations.append(
            {
                "step": entry.get("step"),
                "epoch": entry.get("epoch"),
                "eval_loss": float(entry["eval_loss"]),
            }
        )
    if not evaluations:
        return {
            "best_eval_loss": None,
            "best_checkpoint": None,
            "best_checkpoint_exists": None,
            "best_global_step": None,
            "best_epoch": None,
            "final_eval_loss": None,
            "history": [],
        }

    best = min(evaluations, key=lambda item: item["eval_loss"])
    resolved_checkpoint = best_checkpoint or str(Path(output_dir) / f"checkpoint-{best['step']}")
    resolved_best_metric = float(best_metric) if best_metric is not None else best["eval_loss"]
    return {
        "best_eval_loss": resolved_best_metric,
        "best_checkpoint": resolved_checkpoint,
        "best_checkpoint_exists": Path(resolved_checkpoint).exists(),
        "best_global_step": best["step"],
        "best_epoch": best["epoch"],
        "final_eval_loss": evaluations[-1]["eval_loss"],
        "history": evaluations,
    }


def _load_preprocessing_components(
    model_name,
    auto_config,
    auto_tokenizer,
    auto_processor,
    revision=None,
):
    """按模型配置选择 chat template 的持有者。

    Qwen3.5 是带视觉编码器的条件生成模型，官方模板由 processor 提供；本项目
    当前数据仅含文本和工具调用，因此 labels 仍用 processor.tokenizer 的 token id。
    其他纯文本因果模型保持原来的 tokenizer 路径。
    """
    load_kwargs = {"trust_remote_code": True}
    if revision:
        load_kwargs["revision"] = revision
    config = auto_config.from_pretrained(model_name, **load_kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    if is_multimodal:
        processor = auto_processor.from_pretrained(model_name, **load_kwargs)
        return processor.tokenizer, processor, True
    tokenizer = auto_tokenizer.from_pretrained(model_name, **load_kwargs)
    return tokenizer, tokenizer, False


def _torch_dataset(examples, torch):
    class TokenizedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            example = examples[index]
            return {
                "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(example["labels"], dtype=torch.long),
            }

    return TokenizedDataset()


def _collate(batch, pad_token_id, torch):
    """右侧 padding，labels 的 padding 永远不参与 loss。"""
    max_length = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].size(0)
        input_ids[row, :length] = item["input_ids"]
        attention_mask[row, :length] = item["attention_mask"]
        labels[row, :length] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    run_started_at = datetime.now(timezone.utc)
    _start_time = _time.time()
    args = parse_args()
    if args.max_length < 1 or args.epochs <= 0:
        raise SystemExit("--max-length 与 --epochs 必须为正数")
    _validate_optional_training_dependencies(args)
    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    ) = _training_dependencies()

    tokenizer, chat_template, is_multimodal = _load_preprocessing_components(
        args.model,
        auto_config=AutoConfig,
        auto_tokenizer=AutoTokenizer,
        auto_processor=AutoProcessor,
        revision=args.revision,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- Phase 1: 加载训练数据 ----
    print(f"\n{'='*60}")
    print(f"  Phase 1/3: 加载 & Tokenize 训练数据 (max_length={args.max_length})")
    print(f"{'='*60}")
    train_examples, train_stats = load_supervised_examples(
        args.train,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=args.max_length,
    )
    print("train_data=", train_stats)
    if not train_examples:
        raise SystemExit("训练集没有可用样本；请检查 data/sft/ 中的 JSONL 格式")
    validation_examples = []
    validation_stats = None
    if args.validation:
        validation_examples, validation_stats = load_supervised_examples(
            args.validation,
            tokenizer=tokenizer,
            chat_template=chat_template,
            max_length=args.max_length,
        )
        print("validation_data=", validation_stats)
        if not validation_examples:
            raise SystemExit("验证集没有可用样本；请调整划分或 --max-length")

    dtype_name, dtype = _resolve_dtype(args, torch)
    model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM

    # ---- Phase 2: 加载模型 + LoRA ----
    print(f"\n{'='*60}")
    print("  Phase 2/3: 加载模型与 LoRA")
    print(f"{'='*60}")
    print(f"  model={args.model}")
    print(f"  revision={args.revision or 'default/local'}")
    print(f"  dtype={dtype_name}")
    print(f"  attention_implementation={args.attention_implementation}")
    print(f"  qlora={args.qlora}")
    print(f"  lora_r={args.lora_r} lora_alpha={args.lora_alpha}")
    print(f"  lora_targets={','.join(args.target_modules)}")
    model = model_class.from_pretrained(
        args.model,
        **_model_load_kwargs(
            args,
            dtype=dtype,
            bits_and_bytes_config=BitsAndBytesConfig,
        ),
    )
    model = _prepare_model_for_training(
        model,
        args,
        prepare_model_for_kbit_training=prepare_model_for_kbit_training,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(args.target_modules),
        ),
    )
    model.print_trainable_parameters()

    args.output.mkdir(parents=True, exist_ok=True)
    report_to, run_name = _swanlab_config(args)
    if report_to == "swanlab":
        import swanlab
        swanlab.init(
            project=args.swanlab_project,
            name=run_name,
            mode=args.swanlab_mode,
            logdir=str(args.output / SWANLAB_DIRNAME),
        )
        print(
            f"[SwanLab] project={args.swanlab_project} run={run_name} "
            f"logdir={args.output / SWANLAB_DIRNAME}"
        )
    training_arg_values = dict(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=dtype_name == "bf16",
        fp16=dtype_name == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.liger_kernel,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if validation_examples else "no",
        report_to=report_to,
        run_name=run_name,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        seed=args.seed,
    )
    if validation_examples:
        training_arg_values.update(
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
        )
    training_args = TrainingArguments(**training_arg_values)

    ProgressCallback = _progress_callback_class(TrainerCallback, torch)
    progress_callback = ProgressCallback()
    MetricsJsonlCallback = _metrics_jsonl_callback_class(
        TrainerCallback,
        args.output / METRICS_FILENAME,
    )
    trainer_class = _loss_only_eval_trainer_class(
        Trainer,
        enable_skip_logits=args.liger_kernel and is_multimodal,
    )
    trainer_class = _metrics_logging_trainer_class(
        trainer_class,
        metrics_provider=progress_callback.metrics_for_log,
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=_torch_dataset(train_examples, torch),
        eval_dataset=_torch_dataset(validation_examples, torch) if validation_examples else None,
        data_collator=partial(_collate, pad_token_id=tokenizer.pad_token_id, torch=torch),
        callbacks=[progress_callback, MetricsJsonlCallback()],
    )
    if torch.cuda.is_available():
        # 排除模型加载阶段，只统计即将开始的训练过程峰值。
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output))
    chat_template.save_pretrained(str(args.output))

    # --- 训练完成摘要 ---
    gpu_peak_allocated = (
        torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else 0
    )
    gpu_peak_reserved = (
        torch.cuda.max_memory_reserved() / GIB if torch.cuda.is_available() else 0
    )
    evaluation = _evaluation_summary(
        trainer.state.log_history,
        output_dir=args.output,
        best_metric=trainer.state.best_metric,
        best_checkpoint=trainer.state.best_model_checkpoint,
    )
    repo_root = Path(__file__).resolve().parents[1]
    provenance = {
        "base_model": _model_fingerprint(args.model),
        "output_adapter": _adapter_fingerprint(args.output),
        "data": {
            "train": _file_fingerprint(args.train),
            "validation": _file_fingerprint(args.validation) if args.validation else None,
        },
        "git": _git_metadata(repo_root),
        "training_code": {
            "entrypoint": _file_fingerprint(Path(__file__)),
            "dependency_lock": _file_fingerprint(repo_root / "uv.lock"),
        },
        "environment": _environment_metadata(torch),
    }
    total_time = _time.time() - _start_time
    run_summary = {
        "schema_version": 1,
        "status": "completed",
        "started_at": run_started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "data_preprocessing": {
            "train": train_stats,
            "validation": validation_stats,
        },
        "train_loss": result.training_loss,
        **evaluation,
        "metrics": result.metrics,
        "peak_gpu_memory_allocated_gib": round(gpu_peak_allocated, 3),
        "peak_gpu_memory_reserved_gib": round(gpu_peak_reserved, 3),
        "total_time_minutes": round(total_time / 60, 1) if total_time else None,
        "monitoring": {
            "backend": report_to,
            "project": args.swanlab_project if args.swanlab else None,
            "run_name": run_name,
            "mode": args.swanlab_mode if args.swanlab else None,
            "log_directory": (
                str(args.output / SWANLAB_DIRNAME) if args.swanlab else None
            ),
            "metrics_jsonl": str(args.output / METRICS_FILENAME),
        },
        "acceleration": {
            "dtype": dtype_name,
            "liger_kernel": args.liger_kernel,
            "attention_implementation": args.attention_implementation,
            "qlora": args.qlora,
        },
        "provenance": provenance,
        "arguments": {
            "cli": vars(args),
            "transformers": training_args.to_dict(),
        },
    }

    print(f"\n{'='*60}")
    print("  训练完成")
    print(f"  train_loss={result.training_loss:.4f}")
    print(f"  best_eval_loss={evaluation['best_eval_loss']}")
    print(f"  best_checkpoint={evaluation['best_checkpoint']}")
    print(f"  peak_gpu_allocated={gpu_peak_allocated:.1f} GiB")
    print(f"  adapter → {args.output}")
    print(f"{'='*60}\n")

    if trainer.is_world_process_zero():
        summary_path = args.output / RUN_SUMMARY_FILENAME
        temporary_summary_path = summary_path.with_suffix(".json.tmp")
        temporary_summary_path.write_text(
            json.dumps(run_summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary_summary_path.replace(summary_path)
        print(f"运行摘要已保存到 {summary_path}")

    print(f"LoRA adapter 已保存到 {args.output}")


if __name__ == "__main__":
    main()
