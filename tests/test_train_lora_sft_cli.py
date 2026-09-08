"""验证 LoRA SFT 入口的关键默认值。"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.train_lora_sft import (
    DEFAULT_TARGET_MODULES,
    GIB,
    METRICS_FILENAME,
    RUN_SUMMARY_FILENAME,
    SWANLAB_DIRNAME,
    _adapter_fingerprint,
    _evaluation_summary,
    _load_preprocessing_components,
    _loss_only_eval_trainer_class,
    _metrics_jsonl_callback_class,
    _metrics_logging_trainer_class,
    _model_load_kwargs,
    _prepare_model_for_training,
    _progress_callback_class,
    _resolve_dtype,
    _sha256_file,
    _swanlab_config,
    parse_args,
)


class _FakeConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class _FakeAutoConfig:
    @staticmethod
    def from_pretrained(model_name, trust_remote_code):
        del model_name, trust_remote_code
        return _FakeConfig("qwen3_5")


class _FakeTokenizer:
    pass


class _FakeAutoTokenizer:
    called = False

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code):
        del model_name, trust_remote_code
        cls.called = True
        return _FakeTokenizer()


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeAutoProcessor:
    called = False

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code):
        del model_name, trust_remote_code
        cls.called = True
        return _FakeProcessor()


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeModel:
    def __init__(self):
        self.config = type("Config", (), {"use_cache": True})()
        self.input_grads_enabled = False

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


class _FakeTrainer:
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        return model, inputs, prediction_loss_only, ignore_keys


class _FakeLoggingTrainer:
    def __init__(self):
        self.logged = None

    def log(self, logs, start_time=None):
        self.logged = (dict(logs), start_time)
        return self.logged


class _FakeCallback:
    pass


class TrainLoraSftCliTest(unittest.TestCase):
    def test_defaults_are_suitable_for_small_qwen_lora_warmup(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "/models/Qwen3.5-0.8B",
                "--train",
                "outputs/batch/train.jsonl",
                "--output",
                "checkpoints/qwen-shopping-lora",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.model, "/models/Qwen3.5-0.8B")
        self.assertEqual(args.train, Path("outputs/batch/train.jsonl"))
        self.assertEqual(args.max_length, 24576)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.dtype, "auto")
        self.assertFalse(args.bf16)
        self.assertFalse(args.swanlab)
        self.assertEqual(args.swanlab_project, "shopping-grpo")

    def test_swanlab_flags_are_opt_in_and_keep_a_stable_run_name(self):
        """国内监控必须显式启用，且实验名可由调用方固定以便对比。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/adapter",
                "--swanlab",
                "--swanlab-project",
                "shopping-agent",
                "--swanlab-run-name",
                "qwen35-2b-lora-v1",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.swanlab)
        self.assertEqual(args.swanlab_project, "shopping-agent")
        self.assertEqual(args.swanlab_run_name, "qwen35-2b-lora-v1")

    def test_swanlab_config_returns_a_stable_default_run_name(self):
        """SwanLab 由 main 中的显式 init 配置；此处只验证纯配置函数。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/run/adapter",
                "--swanlab",
                "--swanlab-mode",
                "local",
            ],
        ):
            args = parse_args()

        with patch.dict(
            sys.modules,
            {"swanlab": object(), "swanboard": object()},
        ), patch.dict(os.environ, {}, clear=True):
            report_to, run_name = _swanlab_config(args)
            self.assertEqual(report_to, "swanlab")
            self.assertIn("lora-r16", run_name)

    def test_qwen35_uses_processor_template_and_underlying_tokenizer(self):
        """Qwen3.5 是多模态检查点，不能只加载 AutoTokenizer。"""
        tokenizer, chat_template, is_multimodal = _load_preprocessing_components(
            "Qwen/Qwen3.5-2B",
            auto_config=_FakeAutoConfig,
            auto_tokenizer=_FakeAutoTokenizer,
            auto_processor=_FakeAutoProcessor,
        )

        self.assertTrue(is_multimodal)
        self.assertIs(chat_template.tokenizer, tokenizer)
        self.assertTrue(_FakeAutoProcessor.called)
        self.assertFalse(_FakeAutoTokenizer.called)

    def test_default_lora_targets_cover_qwen35_linear_attention_layers(self):
        """Qwen3.5 的 3/4 层是 Gated DeltaNet，不能只训练少数全注意力层。"""
        self.assertIn("in_proj_qkv", DEFAULT_TARGET_MODULES)
        self.assertIn("out_proj", DEFAULT_TARGET_MODULES)

    def test_acceleration_flags_build_liger_sdpa_and_standard_qlora_configuration(self):
        """D 组必须在 C 的 SDPA 基础上显式添加 NF4 QLoRA，而非传递未验证的 dict。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model", "Qwen/Qwen3.5-2B",
                "--train", "outputs/train.jsonl",
                "--output", "outputs/adapter",
                "--liger-kernel",
                "--attention-implementation", "sdpa",
                "--qlora",
            ],
        ):
            args = parse_args()

        kwargs = _model_load_kwargs(args, dtype="bf16", bits_and_bytes_config=_FakeBitsAndBytesConfig)
        self.assertTrue(args.liger_kernel)
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertIsInstance(kwargs["quantization_config"], _FakeBitsAndBytesConfig)
        self.assertEqual(kwargs["quantization_config"].kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(kwargs["quantization_config"].kwargs["bnb_4bit_compute_dtype"], "bf16")

    def test_dtype_auto_prefers_bf16_then_fp16_and_cpu_fp32(self):
        class FakeCuda:
            available = True
            bf16_supported = True

            @classmethod
            def is_available(cls):
                return cls.available

            @classmethod
            def is_bf16_supported(cls):
                return cls.bf16_supported

        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": FakeCuda,
                "bfloat16": "bf16",
                "float16": "fp16",
                "float32": "fp32",
            },
        )
        args = type("Args", (), {"dtype": "auto", "bf16": False})()

        self.assertEqual(_resolve_dtype(args, fake_torch), ("bf16", "bf16"))
        FakeCuda.bf16_supported = False
        self.assertEqual(_resolve_dtype(args, fake_torch), ("fp16", "fp16"))
        FakeCuda.available = False
        self.assertEqual(_resolve_dtype(args, fake_torch), ("fp32", "fp32"))

    def test_model_revision_is_forwarded_to_loader(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "Qwen/Qwen3.5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/adapter",
                "--revision",
                "frozen-revision",
            ],
        ):
            args = parse_args()

        kwargs = _model_load_kwargs(
            args,
            dtype="bf16",
            bits_and_bytes_config=_FakeBitsAndBytesConfig,
        )
        self.assertEqual(kwargs["revision"], "frozen-revision")

    def test_qlora_prepares_model_before_lora_and_keeps_gradient_checkpointing_compatible(self):
        """量化基座必须先做 PEFT 标准预处理，再由后续 LoRA 注入 adapter。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model", "Qwen/Qwen3.5-2B",
                "--train", "outputs/train.jsonl",
                "--output", "outputs/adapter",
                "--qlora",
                "--gradient-checkpointing",
            ],
        ):
            args = parse_args()
        model = _FakeModel()
        prepared = _FakeModel()
        prepare = unittest.mock.MagicMock(return_value=prepared)

        result = _prepare_model_for_training(model, args, prepare)

        self.assertIs(result, prepared)
        prepare.assert_called_once_with(model, use_gradient_checkpointing=True)
        self.assertFalse(result.config.use_cache)

    def test_liger_qwen_loss_only_eval_skips_full_vocabulary_logits(self):
        """纯 eval_loss 必须显式走 Liger fused loss，避免 20K×248K logits。"""
        trainer_class = _loss_only_eval_trainer_class(
            _FakeTrainer,
            enable_skip_logits=True,
        )
        original_inputs = {"input_ids": [1, 2], "labels": [1, 2]}

        _, forwarded_inputs, prediction_loss_only, ignore_keys = trainer_class().prediction_step(
            model="model",
            inputs=original_inputs,
            prediction_loss_only=True,
            ignore_keys=["past_key_values"],
        )

        self.assertTrue(forwarded_inputs["skip_logits"])
        self.assertNotIn("skip_logits", original_inputs)
        self.assertTrue(prediction_loss_only)
        self.assertEqual(ignore_keys, ["past_key_values"])

    def test_eval_that_needs_predictions_does_not_skip_logits(self):
        """若调用方需要 predictions/metrics，则仍必须返回真实 logits。"""
        trainer_class = _loss_only_eval_trainer_class(
            _FakeTrainer,
            enable_skip_logits=True,
        )

        _, forwarded_inputs, _, _ = trainer_class().prediction_step(
            model="model",
            inputs={"input_ids": [1, 2], "labels": [1, 2]},
            prediction_loss_only=False,
        )

        self.assertNotIn("skip_logits", forwarded_inputs)

    def test_non_liger_training_keeps_standard_eval_forward(self):
        """未启用兼容的 Liger Qwen forward 时不能传入专用参数。"""
        trainer_class = _loss_only_eval_trainer_class(
            _FakeTrainer,
            enable_skip_logits=False,
        )

        _, forwarded_inputs, _, _ = trainer_class().prediction_step(
            model="model",
            inputs={"input_ids": [1, 2], "labels": [1, 2]},
            prediction_loss_only=True,
        )

        self.assertNotIn("skip_logits", forwarded_inputs)

    def test_runtime_metrics_are_injected_before_trainer_persists_logs(self):
        calls = []

        def metrics_provider():
            calls.append(True)
            return {"step_time_s": 12.5, "gpu_memory_allocated_gib": 7.25}

        trainer_class = _metrics_logging_trainer_class(
            _FakeLoggingTrainer,
            metrics_provider=metrics_provider,
        )
        trainer = trainer_class()

        trainer.log({"loss": 0.4}, start_time=123.0)

        self.assertEqual(calls, [True])
        self.assertEqual(trainer.logged[0]["step_time_s"], 12.5)
        self.assertEqual(trainer.logged[0]["gpu_memory_allocated_gib"], 7.25)
        self.assertEqual(trainer.logged[1], 123.0)

        trainer.log({"eval_loss": 0.3})
        self.assertEqual(calls, [True])
        self.assertNotIn("step_time_s", trainer.logged[0])

    def test_progress_callback_records_step_time_and_current_and_peak_cuda_memory(self):
        class FakeCuda:
            synchronize_calls = 0

            @staticmethod
            def is_available():
                return True

            @classmethod
            def synchronize(cls):
                cls.synchronize_calls += 1

            @staticmethod
            def memory_allocated():
                return 1 * GIB

            @staticmethod
            def memory_reserved():
                return 2 * GIB

            @staticmethod
            def max_memory_allocated():
                return 3 * GIB

            @staticmethod
            def max_memory_reserved():
                return 4 * GIB

        fake_torch = type("FakeTorch", (), {"cuda": FakeCuda})
        times = iter((10.0, 14.0))
        callback_class = _progress_callback_class(
            _FakeCallback,
            fake_torch,
            clock=lambda: next(times),
        )
        callback = callback_class()
        state = type("State", (), {})()
        control = object()

        callback.on_step_begin(None, state, control)
        callback.on_step_end(None, state, control)
        metrics = callback.metrics_for_log()

        self.assertEqual(metrics["step_time_s"], 4.0)
        self.assertEqual(metrics["step_time_sample_count"], 1)
        self.assertEqual(metrics["gpu_memory_allocated_gib"], 1.0)
        self.assertEqual(metrics["gpu_memory_reserved_gib"], 2.0)
        self.assertEqual(metrics["gpu_peak_memory_allocated_gib"], 3.0)
        self.assertEqual(metrics["gpu_peak_memory_reserved_gib"], 4.0)
        self.assertEqual(FakeCuda.synchronize_calls, 2)

    def test_metrics_jsonl_is_step_grpo_prefixed_and_written_on_each_log(self):
        self.assertEqual(METRICS_FILENAME, "metrics.jsonl")
        self.assertEqual(RUN_SUMMARY_FILENAME, "run-summary.json")
        self.assertEqual(SWANLAB_DIRNAME, "swanlab")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / METRICS_FILENAME
            callback_class = _metrics_jsonl_callback_class(_FakeCallback, path)
            callback = callback_class()
            state = type(
                "State",
                (),
                {"is_world_process_zero": True, "global_step": 7},
            )()

            callback.on_log(None, state, object(), {"loss": 0.25, "step_time_s": 9.0})

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["step"], 7)
            self.assertEqual(record["loss"], 0.25)
            self.assertEqual(record["step_time_s"], 9.0)

    def test_evaluation_summary_selects_best_checkpoint_from_log_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            best_checkpoint = Path(tmpdir) / "checkpoint-20"
            best_checkpoint.mkdir()
            summary = _evaluation_summary(
                [
                    {"step": 10, "epoch": 1.0, "eval_loss": 0.4},
                    {"step": 20, "epoch": 2.0, "eval_loss": 0.3},
                    {"step": 30, "epoch": 3.0, "eval_loss": 0.35},
                ],
                output_dir=tmpdir,
                best_metric=0.3,
                best_checkpoint=str(best_checkpoint),
            )

        self.assertEqual(summary["best_eval_loss"], 0.3)
        self.assertEqual(summary["best_checkpoint"], str(best_checkpoint))
        self.assertTrue(summary["best_checkpoint_exists"])
        self.assertEqual(summary["final_eval_loss"], 0.35)

        empty = _evaluation_summary([], output_dir="outputs/adapter")
        self.assertIsNone(empty["best_eval_loss"])
        self.assertIsNone(empty["best_checkpoint"])

    def test_adapter_fingerprint_is_streamed_and_excludes_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = root / "adapter_model.safetensors"
            adapter.write_bytes(b"adapter")
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "metrics.jsonl").write_text("record", encoding="utf-8")
            (root / "checkpoint-1").mkdir()

            fingerprint = _adapter_fingerprint(root)
            adapter_digest = _sha256_file(adapter)

        self.assertEqual(adapter_digest, hashlib.sha256(b"adapter").hexdigest())
        self.assertIn("adapter_model.safetensors", fingerprint["files"])
        self.assertIn("adapter_config.json", fingerprint["files"])
        self.assertNotIn("metrics.jsonl", fingerprint["files"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
