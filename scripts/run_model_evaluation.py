#!/usr/bin/env python3
"""用一个已启动的 Base、SFT 或 GRPO 模型执行 逐题评测批次。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from tqdm.auto import tqdm


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.contracts import ENVIRONMENT_VERSION  # noqa: E402
from shopping_grpo.harness.evaluation_batch import (  # noqa: E402
    EvaluationBatchSafetyPause,
    run_evaluation_batch,
)
from shopping_grpo.harness.purchase_verifier import (  # noqa: E402
    PURCHASE_VERIFIER_VERSION,
)
from shopping_grpo.harness.runner import EpisodeRunner  # noqa: E402
from shopping_grpo.harness.sft_pipeline import (  # noqa: E402
    default_sft_harness_config,
    sft_harness_contract,
    sft_harness_contract_fingerprint,
)
from shopping_grpo.harness.sft_policy import (  # noqa: E402
    OpenAICompatibleTeacherPolicy,
)
from shopping_grpo.harness.sft_prompt import (  # noqa: E402
    SFT_SYSTEM_PROMPT,
    SFT_SYSTEM_PROMPT_SHA256,
    SFT_SYSTEM_PROMPT_VERSION,
)
from shopping_grpo.harness.sft_storage import file_sha256, read_jsonl  # noqa: E402
from shopping_grpo.harness.sft_tokenizer import (  # noqa: E402
    TransformersRuntimeTokenCounter,
    TRANSFORMERS_RUNTIME_COUNTER_VERSION,
)
from shopping_grpo.harness.tools import SFT_TOOL_REGISTRY  # noqa: E402
from shopping_grpo.harness.trajectory_evaluator import (  # noqa: E402
    TRAJECTORY_EVALUATION_VERSION,
)


DEFAULT_TASKS = REPOSITORY / "data/evaluation/tasks.jsonl"
DEFAULT_MODEL_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_SHOPSIM_BASE_URL = "http://127.0.0.1:5700"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-role", choices=("base", "sft", "grpo"), required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument(
        "--model-artifact",
        type=Path,
        required=True,
        help="Base/SFT/GRPO 模型或 adapter 目录，仅用于冻结 Run 来源",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="与模型一致的 tokenizer 本地目录或 Hugging Face 名称",
    )
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-base-url", default=DEFAULT_MODEL_BASE_URL)
    parser.add_argument("--shopsim-base-url", default=DEFAULT_SHOPSIM_BASE_URL)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        help="仅用于冒烟；省略时严格运行 tasks 文件中的全部 200 题",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭终端实时进度条",
    )
    return parser


async def run_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.concurrency < 1:
        raise ValueError("--concurrency 必须至少为 1")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit 必须至少为 1")
    if args.timeout <= 0 or args.api_retries < 0:
        raise ValueError("--timeout 必须大于 0，--api-retries 不能小于 0")
    if not args.model_artifact.exists():
        raise FileNotFoundError(f"模型产物不存在：{args.model_artifact}")

    all_task_ids = _load_task_ids(args.tasks)
    if args.limit is None and len(all_task_ids) != 200:
        raise ValueError("正式评测必须使用恰好 200 个任务；冒烟请显式传 --limit")
    selected_task_ids = (
        all_task_ids if args.limit is None else all_task_ids[: args.limit]
    )
    _configure_local_proxy_bypass(
        args.model_base_url,
        args.shopsim_base_url,
    )
    api_key, api_key_source = _load_api_key(
        args.api_key_file,
        model_base_url=args.model_base_url,
    )
    await asyncio.to_thread(
        _preflight_model,
        base_url=args.model_base_url,
        api_key=api_key,
        expected_model=args.served_model,
        timeout=args.timeout,
    )
    counter = await asyncio.to_thread(
        TransformersRuntimeTokenCounter.from_pretrained,
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    harness_config = default_sft_harness_config(
        environment_base_url=args.shopsim_base_url
    )
    runner = EpisodeRunner(
        tool_registry=SFT_TOOL_REGISTRY,
        system_prompt=SFT_SYSTEM_PROMPT,
        include_initial_observation=True,
    )

    def policy_factory(task_id: int) -> OpenAICompatibleTeacherPolicy:
        del task_id
        return OpenAICompatibleTeacherPolicy(
            model=args.served_model,
            base_url=args.model_base_url,
            api_key=api_key,
            context_policy=harness_config.context,
            observation_policy=harness_config.observation,
            count_chat_tokens=counter.count_chat,
            count_text_tokens=counter.count_text,
            temperature=0.0,
            top_p=1.0,
            timeout_s=args.timeout,
            max_retries=args.api_retries,
            thinking_mode=None,
            extra_body={
                "seed": int(args.seed),
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

    harness_contract = sft_harness_contract(harness_config, runner)
    run_contract = {
        "model": {
            "role": args.model_role,
            "served_model": args.served_model,
            "artifact": _artifact_identity(args.model_artifact),
            "served_model_preflight_verified": True,
        },
        "model_endpoint": str(args.model_base_url).rstrip("/"),
        "api_key_source": api_key_source,
        "tokenizer": {
            "path_or_name": _portable_path(args.tokenizer),
            "revision": args.tokenizer_revision,
            "runtime_class": counter.__class__.__name__,
            "runtime_contract_version": TRANSFORMERS_RUNTIME_COUNTER_VERSION,
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": int(args.seed),
            "enable_thinking": False,
        },
        "tasks": {
            "source": _portable_path(args.tasks),
            "source_sha256": file_sha256(args.tasks),
            "source_count": len(all_task_ids),
            "limit": args.limit,
        },
        "shopsimulator_base_url": str(args.shopsim_base_url).rstrip("/"),
        "environment_version": ENVIRONMENT_VERSION,
        "harness_config": asdict(harness_config),
        "harness_contract": harness_contract,
        "harness_contract_sha256": sft_harness_contract_fingerprint(
            harness_config, runner
        ),
        "system_prompt_version": SFT_SYSTEM_PROMPT_VERSION,
        "system_prompt_sha256": SFT_SYSTEM_PROMPT_SHA256,
        "tool_schema_version": SFT_TOOL_REGISTRY.version,
        "tool_schema_fingerprint": SFT_TOOL_REGISTRY.fingerprint,
        "trajectory_evaluator_version": TRAJECTORY_EVALUATION_VERSION,
        "purchase_verifier_version": PURCHASE_VERIFIER_VERSION,
        "concurrency": int(args.concurrency),
    }
    progress_bar = None

    def update_progress(snapshot: Mapping[str, Any]) -> None:
        nonlocal progress_bar
        if progress_bar is None:
            progress_bar = tqdm(
                total=int(snapshot["total"]),
                initial=int(snapshot["completed"]),
                desc=f"{args.model_role.upper()}",
                unit="task",
                dynamic_ncols=True,
            )
        else:
            progress_bar.update(
                max(0, int(snapshot["completed"]) - int(progress_bar.n))
            )
        postfix = {
            "format_ok": int(snapshot["format_correct"]),
            "purchase_ok": int(snapshot["purchase_correct"]),
            "invalid": int(snapshot["trajectory_invalid"]),
        }
        if snapshot.get("task_id") is not None:
            postfix["task_id"] = int(snapshot["task_id"])
        progress_bar.set_postfix(postfix, refresh=True)

    try:
        return await run_evaluation_batch(
            task_ids=selected_task_ids,
            output_dir=args.output_dir,
            runner=runner,
            harness_config=harness_config,
            policy_factory=policy_factory,
            run_contract=run_contract,
            concurrency=args.concurrency,
            progress_callback=(None if args.no_progress else update_progress),
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()


def _load_task_ids(path: Path) -> list[int]:
    task_ids = []
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError(f"任务文件含无效 task_id：{path}")
        task_ids.append(task_id)
    if not task_ids:
        raise ValueError(f"任务文件为空：{path}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"任务文件含重复 task_id：{path}")
    return task_ids


def _load_api_key(
    path: Path | None,
    *,
    model_base_url: str,
) -> tuple[str, str]:
    if path is not None:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise ValueError("--api-key-file 权限必须为 600 或更严格")
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise ValueError("--api-key-file 为空")
        return value, "api_key_file"
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value, "OPENAI_API_KEY"
    if _is_local_endpoint(model_base_url):
        return "local-vllm", "local_vllm_placeholder"
    raise ValueError("远程模型服务必须设置 OPENAI_API_KEY 或 --api-key-file")


def _preflight_model(
    *,
    base_url: str,
    api_key: str,
    expected_model: str,
    timeout: float,
) -> list[str]:
    request = Request(
        f"{str(base_url).rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "wlx-evaluation-batch/1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("模型 /models 响应必须是 JSON 对象")
    model_ids = sorted(
        str(item["id"])
        for item in payload.get("data") or []
        if isinstance(item, Mapping) and item.get("id")
    )
    if expected_model not in model_ids:
        raise ValueError(
            f"服务未暴露模型 {expected_model!r}；当前可用：{model_ids}"
        )
    return model_ids


def _artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": _portable_path(resolved),
            "type": "file",
            "bytes": resolved.stat().st_size,
            "sha256": file_sha256(resolved),
        }
    files = sorted(
        child
        for child in resolved.iterdir()
        if child.is_file()
        and (
            child.suffix == ".safetensors"
            or child.name
            in {
                "adapter_config.json",
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            }
        )
    )
    if not files:
        raise ValueError(f"模型目录根层没有可识别的权重或配置：{path}")
    records = [
        {
            "name": child.name,
            "bytes": child.stat().st_size,
            "sha256": file_sha256(child),
        }
        for child in files
    ]
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": _portable_path(resolved),
        "type": "directory_root_model_files",
        "sha256": digest,
        "files": records,
    }


def _portable_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except (OSError, ValueError):
        return str(value)


def _is_local_endpoint(value: str) -> bool:
    return _local_hostname(value) is not None


def _local_hostname(value: str) -> str | None:
    hostname = (urlsplit(str(value)).hostname or "").casefold()
    return hostname if hostname in {"127.0.0.1", "localhost", "::1"} else None


def _configure_local_proxy_bypass(*urls: str) -> None:
    local_hosts = [hostname for value in urls if (hostname := _local_hostname(value))]
    if not local_hosts:
        return
    existing = []
    for variable in ("NO_PROXY", "no_proxy"):
        existing.extend(
            item.strip()
            for item in os.environ.get(variable, "").split(",")
            if item.strip()
        )
    combined = list(dict.fromkeys([*existing, *local_hosts]))
    value = ",".join(combined)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = asyncio.run(run_from_args(args))
    except EvaluationBatchSafetyPause as exc:
        raise SystemExit(f"批次因环境安全状态暂停：{exc}") from exc
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
