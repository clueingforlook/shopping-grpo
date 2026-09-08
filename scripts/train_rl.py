#!/usr/bin/env python3
"""Reward v4 GRPO launcher with preflight and safe resume."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = ROOT.parent / "models/Qwen3.5-2B"
DEFAULT_SFT_ADAPTER = ROOT / "outputs/models/sft-own-data-v1-lora"
DEFAULT_MERGED_MODEL = ROOT / "outputs/models/sft-own-data-v1-merged"
DEFAULT_DATA_DIR = ROOT / "data/step-grpo"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/models"
DEFAULT_CONFIG = ROOT / "configs/step_grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/step_agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "data/step-environment.json"
RUN_MANIFEST = "run-manifest.json"
TRAIN_LOG = "train.log"
_CONTINUATION_OVERRIDE_PREFIXES = (
    "trainer.total_training_steps=",
    "trainer.max_actor_ckpt_to_keep=",
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_RAY_PREFIX = re.compile(r"^\([^)]*\bpid=\d+\)\s*")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _small_file_signature(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _model_signature(model: Path) -> dict[str, Any]:
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"模型目录无效：{model}")
    weights = sorted(
        path
        for path in model.iterdir()
        if path.is_file()
        and (path.name.endswith((".safetensors", ".bin")) or "index.json" in path.name)
    )
    if not weights:
        raise SystemExit(f"模型目录中没有权重：{model}")
    metadata = []
    for name in ("config.json", "merge_manifest.json", "tokenizer_config.json"):
        path = model / name
        if path.is_file():
            metadata.append(_small_file_signature(path))
    return {
        "path": str(model.resolve()),
        "metadata": metadata,
        "weights": [{"name": path.name, "bytes": path.stat().st_size} for path in weights],
    }


def _run_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise argparse.ArgumentTypeError("run-name 必须以字母或数字开头，只能包含字母、数字、点、下划线和连字符")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "train", "resume"))
    parser.add_argument(
        "--run-name", type=_run_name, default="rl-v4-main-24k-noentropy"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MERGED_MODEL)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--logger", choices=("console", "swanlab"), default="console")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--total-steps", type=_positive_int)
    parser.add_argument("--max-checkpoints", type=_positive_int)
    parser.add_argument("--no-auto-merge", action="store_true")
    args, hydra_overrides = parser.parse_known_args()
    args.hydra_overrides = hydra_overrides
    return args


def _prepare_data(data_dir: Path) -> tuple[Path, Path, Path]:
    train = data_dir / "train.parquet"
    validation = data_dir / "validation.parquet"
    metadata = data_dir / "metadata.json"
    present = [path.is_file() for path in (train, validation, metadata)]
    if any(present) and not all(present):
        raise SystemExit(f"任务集不完整，请检查：{data_dir}")
    if not all(present):
        command = [sys.executable, str(ROOT / "scripts/prepare_rl_data.py"), "--output-dir", str(data_dir)]
        subprocess.run(command, cwd=ROOT, check=True)
    recorded = json.loads(metadata.read_text(encoding="utf-8"))
    if recorded.get("schema_version") != "wlx-grpo-data-v1":
        raise SystemExit(f"任务元数据版本不正确：{metadata}")
    if recorded["train"]["sha256"] != _sha256(train) or recorded["validation"]["sha256"] != _sha256(validation):
        raise SystemExit("任务文件与冻结元数据哈希不一致")
    return train.resolve(), validation.resolve(), metadata.resolve()


def _merge_model(args: argparse.Namespace) -> Path:
    model = args.model.expanduser().resolve()
    if (model / "config.json").is_file():
        _model_signature(model)
        return model
    if args.mode == "preflight":
        base = args.base_model.expanduser().resolve()
        adapter = args.sft_adapter.expanduser().resolve()
        if not (base / "config.json").is_file() or not (adapter / "adapter_config.json").is_file():
            raise SystemExit("找不到 merged 模型，且 SFT 合并所需的 base/adapter 不完整")
        print(f"预检查：merged 模型尚未生成；首次 train/smoke 时将自动合并到 {model}")
        return base
    if args.no_auto_merge:
        raise SystemExit(f"merged 模型不存在：{model}")
    command = [
        sys.executable,
        str(ROOT / "scripts/merge_lora_adapter.py"),
        "--base-model",
        str(args.base_model.expanduser().resolve()),
        "--adapter",
        str(args.sft_adapter.expanduser().resolve()),
        "--output",
        str(model),
        "--bf16",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    _model_signature(model)
    return model


def _environment(args: argparse.Namespace, model: Path, train: Path, validation: Path, output: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "SHOPPING_GRPO_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(train),
            "GRPO_VAL_FILE": str(validation),
            "GRPO_OUTPUT_DIR": str(output),
            "GRPO_CONFIG_NAME": DEFAULT_CONFIG.stem,
            "SHOPSIM_BASE_URL": args.env_url,
        }
    )
    if args.logger == "swanlab":
        environment.update({"SWANLAB_MODE": "online", "SWANLAB_LOG_DIR": str(output / "swanlab")})
    return environment


def _overrides(args: argparse.Namespace, resume_path: Path | None) -> list[str]:
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    result = [
        "trainer.logger=[console,swanlab]" if args.logger == "swanlab" else "trainer.logger=[console]",
        f"trainer.experiment_name={args.run_name}",
    ]
    if args.mode == "smoke":
        result.extend(["trainer.total_training_steps=5", "trainer.save_freq=5", "trainer.test_freq=5"])
    elif args.total_steps is not None:
        result.append(f"trainer.total_training_steps={args.total_steps}")
    if args.max_checkpoints is not None:
        result.append(f"trainer.max_actor_ckpt_to_keep={args.max_checkpoints}")
    if resume_path is None:
        result.extend(["trainer.resume_mode=disable", "trainer.resume_from_path=null"])
    else:
        result.extend(["trainer.resume_mode=resume_path", f"trainer.resume_from_path={resume_path}"])
    return [*result, *extra]


def _complete_checkpoints(output: Path) -> list[Path]:
    candidates = []
    for path in output.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        actor = path / "actor"
        if match and actor.is_dir() and any(actor.iterdir()) and (path / "data.pt").is_file():
            candidates.append(path)
    return sorted(candidates, key=lambda path: int(path.name.removeprefix("global_step_")))


def _resume_path(args: argparse.Namespace, output: Path) -> Path | None:
    if args.mode != "resume":
        if args.resume_from is not None:
            raise SystemExit("--resume-from 只能和 resume 一起使用")
        return None
    if not output.is_dir():
        raise SystemExit(f"续训目录不存在：{output}")
    path = args.resume_from.expanduser().resolve() if args.resume_from else (_complete_checkpoints(output)[-1] if _complete_checkpoints(output) else None)
    if path is None or path not in _complete_checkpoints(output):
        raise SystemExit(f"没有找到完整 checkpoint：{output}")
    return path


def _fingerprint(model: Path, train: Path, validation: Path, metadata: Path, overrides: list[str]) -> dict[str, Any]:
    sources = [
        DEFAULT_CONFIG,
        DEFAULT_AGENT_CONFIG,
        DEFAULT_TOOL_CONFIG,
        DEFAULT_MANIFEST,
        ROOT / "src/shopping_grpo/training/grpo/reward_v4.py",
        ROOT / "src/shopping_grpo/training/grpo/step_grpo.py",
        ROOT / "src/shopping_grpo/training/grpo/dynamic_sampling.py",
        ROOT / "src/shopping_grpo/training/grpo/adapter/agent_loop.py",
        ROOT / "src/shopping_grpo/training/grpo/adapter/runtime.py",
        ROOT / "src/shopping_grpo/training/grpo/adapter/tools.py",
    ]
    stable_overrides = [
        value
        for value in overrides
        if not value.startswith(("trainer.resume_mode=", "trainer.resume_from_path="))
    ]
    return {
        "schema_version": "wlx-rl-run-v1",
        "model": _model_signature(model),
        "train": _small_file_signature(train),
        "validation": _small_file_signature(validation),
        "data_metadata": _small_file_signature(metadata),
        "sources": [_small_file_signature(path) for path in sources],
        "hydra_overrides": stable_overrides,
    }


def _check_or_write_manifest(output: Path, fingerprint: dict[str, Any], resume: bool) -> None:
    manifest = output / RUN_MANIFEST
    if resume:
        if not manifest.is_file():
            raise SystemExit(f"续训缺少运行指纹：{manifest}")
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        # Extending the stopping point and tightening checkpoint retention do
        # not change the model, data, reward, optimizer, or sampling setup.
        # Normalize only these two continuation controls during resume.
        expected = copy.deepcopy(existing)
        current = copy.deepcopy(fingerprint)
        for value in (expected, current):
            value["hydra_overrides"] = [
                override
                for override in value.get("hydra_overrides", [])
                if not override.startswith(_CONTINUATION_OVERRIDE_PREFIXES)
            ]
        if expected != current:
            raise SystemExit("续训配置、模型或任务指纹发生变化，已拒绝启动")
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"新训练输出目录必须为空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_worker_line(line: str) -> str:
    return _RAY_PREFIX.sub("", _ANSI_ESCAPE.sub("", line)).strip()


def _metric_number(text: str, key: str) -> float | None:
    match = re.search(rf"(?:^| - ){re.escape(key)}:([^\s]+)", text)
    if match is None:
        return None
    raw = match.group(1)
    if "(" in raw:
        raw = raw.rsplit("(", 1)[-1].rstrip(")")
    try:
        return float(raw)
    except ValueError:
        return None


def _compact_step_line(line: str) -> str | None:
    text = _clean_worker_line(line)
    step = _metric_number(text, "training/global_step")
    if step is None:
        return None
    updated = _metric_number(text, "training/optimizer_updated")
    effective = _metric_number(text, "group/effective_ratio")
    if updated == 0:
        skipped = _metric_number(text, "shopping_dynamic_sampling/skipped_updates_total")
        parts = [f"[step {int(step)}] update skipped"]
        if effective is not None:
            parts.append(f"有效组 {effective:.0%}")
        if skipped is not None:
            parts.append(f"累计跳过 {int(skipped)}")
        return " | ".join(parts) + "\n"

    fields = [f"[step {int(step)}]"]
    for key, label, formatter in (
        ("wlx_reward/orm_mean", "ORM", lambda value: f"{value:.3f}"),
        ("wlx_reward/gold_rate", "Gold", lambda value: f"{value:.0%}"),
        ("wlx_reward/partial_purchase_rate", "部分购买", lambda value: f"{value:.0%}"),
        ("wlx_reward/no_purchase_rate", "未购买", lambda value: f"{value:.0%}"),
        ("group/effective_ratio", "有效组", lambda value: f"{value:.0%}"),
        ("actor/ppo_kl", "KL", lambda value: f"{value:.4f}"),
        ("actor/grad_norm", "梯度", lambda value: f"{value:.3f}"),
        ("response_length/mean", "长度", lambda value: f"{value:.0f}"),
        ("timing_s/step", "耗时", lambda value: f"{value:.0f}s"),
        (
            "actor/perf/max_memory_allocated_gb",
            "显存",
            lambda value: f"{value:.1f}GiB",
        ),
    ):
        value = _metric_number(text, key)
        if value is not None:
            fields.append(f"{label} {formatter(value)}")
    return " | ".join(fields) + "\n"


def _compact_sampling_line(line: str) -> str | None:
    marker = "SHOPPING_GRPO_DYNAMIC_SAMPLING_READY "
    text = _clean_worker_line(line)
    if marker not in text:
        return None
    try:
        payload = json.loads(text.split(marker, 1)[1])
    except json.JSONDecodeError:
        return None
    return (
        "[采样] "
        f"批次 {int(payload.get('generation_batches', 0))} | "
        f"训练组 {int(payload.get('trained_groups', 0))}/"
        f"{int(payload.get('generated_groups', 0))} | "
        f"轨迹 {int(payload.get('generated_trajectories', 0))} | "
        f"过滤组 {int(payload.get('filtered_groups', 0))}\n"
    )


class _TerminalOutput:
    def __init__(self) -> None:
        self.error_mode = False

    def render(self, line: str) -> str | None:
        text = _clean_worker_line(line)
        if self.error_mode:
            return line
        if any(
            marker in text
            for marker in (
                "ray.exceptions.",
                "Error executing job",
                "RuntimeError:",
                "OutOfMemoryError:",
            )
        ):
            self.error_mode = True
            return line

        compact_step = _compact_step_line(line)
        if compact_step is not None:
            return compact_step
        compact_sampling = _compact_sampling_line(line)
        if compact_sampling is not None:
            return compact_sampling
        if "SHOPPING_GRPO_DYNAMIC_SAMPLING_BATCH " in text:
            return None
        if "validation generation end" in text or "test_gen_batch meta info" in text:
            return None
        if text.startswith("ray init kwargs:"):
            return None
        if "local_global_step_folder:" in text:
            path = text.split("local_global_step_folder:", 1)[1].strip()
            return f"[checkpoint] 正在保存 {Path(path).name}\n"
        if _RAY_PREFIX.match(_ANSI_ESCAPE.sub("", line)):
            if "WARNING" in text or "ERROR" in text:
                return line
            return None
        return line


def _run_logged(command: list[str], environment: dict[str, str], log: Path) -> int:
    terminal = _TerminalOutput()
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            rendered = terminal.render(line)
            if rendered is not None:
                sys.stdout.write(rendered)
                sys.stdout.flush()
        return process.wait()


def _ensure_verl_patch(check_only: bool) -> None:
    command = [sys.executable, str(ROOT / "scripts/apply_verl_dynamic_sampling_patch.py")]
    if check_only:
        command.append("--check")
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"veRL 动态采样补丁检查失败（exit={exc.returncode}）") from None


def main() -> None:
    args = parse_args()
    _ensure_verl_patch(check_only=args.mode == "preflight")
    train, validation, metadata = _prepare_data(args.data_dir.expanduser().resolve())
    model = _merge_model(args)
    output = (args.output_root / args.run_name).expanduser().resolve()
    resume_path = _resume_path(args, output)
    if resume_path is not None and args.total_steps is not None:
        resume_step = int(resume_path.name.removeprefix("global_step_"))
        if args.total_steps <= resume_step:
            raise SystemExit(
                f"--total-steps 必须大于续训 checkpoint 的步数 {resume_step}"
            )
    overrides = _overrides(args, resume_path)
    environment = _environment(args, model, train, validation, output)
    preflight = [sys.executable, str(ROOT / "scripts/check_grpo_runtime.py"), *overrides]
    try:
        subprocess.run(preflight, cwd=ROOT, env=environment, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"RL 预检查失败（exit={exc.returncode}）") from None
    if args.mode == "preflight":
        print("RL 预检查通过；没有启动训练。")
        return
    fingerprint = _fingerprint(model, train, validation, metadata, overrides)
    _check_or_write_manifest(output, fingerprint, resume=args.mode == "resume")
    command = [sys.executable, "-m", "verl.trainer.main_ppo", f"--config-path={DEFAULT_CONFIG.parent}", f"--config-name={DEFAULT_CONFIG.stem}", *overrides]
    print(f"RL mode={args.mode} | output={output}")
    if resume_path is not None:
        print(f"从 checkpoint 续训：{resume_path.name}")
    print(f"完整原始输出：{output / TRAIN_LOG}")
    raise SystemExit(_run_logged(command, environment, output / TRAIN_LOG))


if __name__ == "__main__":
    main()
