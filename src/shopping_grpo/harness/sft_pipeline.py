"""把任务检查、Core 采样和 SFT 数据构建串成一条清楚的阶段流水线。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from shopping_grpo.harness.config import ContextPolicy, HarnessConfig, ObservationPolicy
from shopping_grpo.harness.runner import EpisodeRunner
from shopping_grpo.harness.sft_contracts import SamplingConfig, SftTask
from shopping_grpo.harness.sft_dataset import DatasetBuildConfig, build_dataset_artifacts
from shopping_grpo.harness.sft_metrics import TrainingTokenCounter
from shopping_grpo.harness.sft_prompt import (
    SFT_INITIAL_MESSAGE_VERSION,
    SFT_SYSTEM_PROMPT,
    SFT_SYSTEM_PROMPT_SHA256,
    SFT_SYSTEM_PROMPT_VERSION,
)
from shopping_grpo.harness.sft_sampler import CoreAttemptExecutor, SftSamplingScheduler
from shopping_grpo.harness.sft_storage import (
    RawTrajectoryStore,
    assert_no_held_out_tasks,
    load_held_out_task_ids,
    load_sft_tasks,
)
from shopping_grpo.harness.tools import SFT_TOOL_REGISTRY


@dataclass(frozen=True)
class SftBatchPaths:
    """集中保存一个采样批次的原始文件和最终冻结目录位置。"""

    batch_dir: Path
    raw_path: Path
    dataset_dir: Path

    @classmethod
    def under(cls, batch_dir: str | Path) -> "SftBatchPaths":
        """根据一个批次目录生成统一路径，不创建任何文件。"""

        root = Path(batch_dir)
        return cls(
            batch_dir=root,
            raw_path=root / "raw.jsonl",
            dataset_dir=root / "dataset-v1",
        )


def default_sft_harness_config(
    *,
    environment_base_url: str = "http://127.0.0.1:5700",
) -> HarnessConfig:
    """生成第一版安全默认值：24K 总窗口、不开上下文删除、开启结构化页面压缩。"""

    return HarnessConfig(
        max_steps=35,
        max_assistant_turns=45,
        environment_base_url=str(environment_base_url).rstrip("/"),
        max_guard_rejections=3,
        parallel_tool_call_policy="reject",
        context=ContextPolicy(
            window_tokens=24_576,
            generation_reserve_tokens=1_536,
            safety_margin_tokens=512,
            compaction_enabled=False,
        ),
        observation=ObservationPolicy(
            token_budget=1_536,
            detail_token_budget=4_096,
            generic_token_budget=768,
            search_top_k=20,
        ),
        validate_terminal_reward=True,
    )


class SftDataPipeline:
    """SFT 阶段总入口：Core 只跑轨迹，本类负责计划、落盘、清洗和冻结。"""

    def __init__(
        self,
        *,
        harness_config: HarnessConfig | None = None,
        runner: EpisodeRunner | None = None,
    ) -> None:
        """保存 Core 配置，并默认使用移除了 think 工具的 SFT Runner。"""

        self.harness_config = harness_config or default_sft_harness_config()
        candidate_runner = runner or EpisodeRunner(
            tool_registry=SFT_TOOL_REGISTRY,
            system_prompt=SFT_SYSTEM_PROMPT,
            include_initial_observation=True,
        )
        _assert_sft_runner_contract(candidate_runner)
        self.runner = candidate_runner

    def prepare_tasks(
        self,
        *,
        tasks_path: str | Path,
        held_out_tasks_path: str | Path,
    ) -> list[SftTask]:
        """读取官方 Train 任务并在花模型费用前排除 Evaluation task_id。"""

        tasks = load_sft_tasks(tasks_path, expected_split="train")
        held_out = load_held_out_task_ids(held_out_tasks_path)
        assert_no_held_out_tasks(tasks, held_out)
        return tasks

    async def collect(
        self,
        *,
        tasks: Sequence[SftTask],
        raw_path: str | Path,
        sampling_config: SamplingConfig,
        policy_factory: Callable[..., object],
    ) -> dict[str, Any]:
        """用 Core 完成在线采样；模型与环境配置正确后可随时断点续采。"""

        executor = CoreAttemptExecutor(
            runner=self.runner,
            harness_config=self.harness_config,
            policy_factory=policy_factory,
        )
        scheduler = SftSamplingScheduler(sampling_config)
        return await scheduler.collect(
            tasks,
            executor=executor,
            store=RawTrajectoryStore(raw_path),
        )

    def build_dataset(
        self,
        *,
        raw_path: str | Path,
        output_dir: str | Path,
        held_out_tasks_path: str | Path,
        training_token_counter: TrainingTokenCounter,
        build_config: DatasetBuildConfig | None = None,
        collection_config: Mapping[str, Any] | None = None,
        artifact_prefix: str = "",
    ) -> dict[str, Any]:
        """从 raw.jsonl 重建并冻结主 SFT 数据，严格只接收完整 Gold 购买。"""

        return build_dataset_artifacts(
            raw_path=raw_path,
            output_dir=output_dir,
            held_out_tasks_path=held_out_tasks_path,
            training_token_counter=training_token_counter,
            config=build_config,
            collection_config=collection_config,
            tool_registry=SFT_TOOL_REGISTRY,
            artifact_prefix=artifact_prefix,
        )


def sft_harness_contract(
    harness_config: HarnessConfig,
    runner: EpisodeRunner,
) -> dict[str, Any]:
    """生成会影响单条轨迹语义的稳定配置，供采样目录冻结和比对。"""

    _assert_sft_runner_contract(runner)
    return {
        "harness_config": asdict(harness_config),
        "runner": {
            "system_prompt_version": SFT_SYSTEM_PROMPT_VERSION,
            "system_prompt_sha256": SFT_SYSTEM_PROMPT_SHA256,
            "initial_message_version": SFT_INITIAL_MESSAGE_VERSION,
            "include_initial_observation": runner.include_initial_observation,
            "tool_schema_version": runner.tool_registry.version,
            "tool_schema_fingerprint": runner.tool_registry.fingerprint,
        },
    }


def sft_harness_contract_fingerprint(
    harness_config: HarnessConfig,
    runner: EpisodeRunner,
) -> str:
    """为完整 SFT Harness/Input 契约生成确定性的 SHA-256。"""

    payload = json.dumps(
        sft_harness_contract(harness_config, runner),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_sft_runner_contract(runner: EpisodeRunner) -> None:
    """拒绝用自定义 Runner 绕过专用 Prompt、工具表或首轮页面校验。"""

    if not isinstance(runner, EpisodeRunner):
        raise TypeError("runner 必须是 EpisodeRunner")
    SFT_TOOL_REGISTRY.assert_compatible(runner.tool_registry)
    if runner.system_prompt != SFT_SYSTEM_PROMPT:
        raise ValueError("SFT runner 必须使用冻结的 SFT_SYSTEM_PROMPT")
    if runner.include_initial_observation is not True:
        raise ValueError("SFT runner 必须包含经过验证的初始 Observation v2")


__all__ = [
    "SftBatchPaths",
    "SftDataPipeline",
    "default_sft_harness_config",
    "sft_harness_contract",
    "sft_harness_contract_fingerprint",
]
