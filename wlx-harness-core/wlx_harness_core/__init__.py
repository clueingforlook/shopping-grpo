"""WLX Harness Core 对外提供的统一入口。

这里只登记外部可以使用的名字，不会一上来就加载 ShopSimulator、模型客户端或
veRL。代码真正用到某个名字时，才会加载它所在的模块，避免无关依赖影响其他阶段。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final


_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "AssistantTurn": ("wlx_harness_core.wlx_contracts", "AssistantTurn"),
    "BridgeDependencyError": (
        "wlx_harness_core.wlx_stage_grpo",
        "BridgeDependencyError",
    ),
    "ConfigValidationError": (
        "wlx_harness_core.wlx_config",
        "ConfigValidationError",
    ),
    "ContextPolicy": ("wlx_harness_core.wlx_config", "ContextPolicy"),
    "DatasetBuildConfig": (
        "wlx_harness_core.wlx_sft_dataset",
        "DatasetBuildConfig",
    ),
    "DeepSeekRubricClient": (
        "wlx_harness_core.wlx_eval_rubric_generator",
        "DeepSeekRubricClient",
    ),
    "EnvironmentResult": ("wlx_harness_core.wlx_environment", "EnvironmentResult"),
    "EpisodeRequest": ("wlx_harness_core.wlx_contracts", "EpisodeRequest"),
    "EpisodeRunner": ("wlx_harness_core.wlx_runner", "EpisodeRunner"),
    "ErrorCategory": ("wlx_harness_core.wlx_contracts", "ErrorCategory"),
    "EvaluationStageAdapter": (
        "wlx_harness_core.wlx_stage_evaluation",
        "EvaluationStageAdapter",
    ),
    "build_process_judge_payload": (
        "wlx_harness_core.wlx_eval_judge",
        "build_process_judge_payload",
    ),
    "build_full_judge_payload": (
        "wlx_harness_core.wlx_eval_judge",
        "build_full_judge_payload",
    ),
    "build_judge_gold_draft": (
        "wlx_harness_core.wlx_eval_judge_gold",
        "build_judge_gold_draft",
    ),
    "calibrate_judge": (
        "wlx_harness_core.wlx_eval_judge_pipeline",
        "calibrate_judge",
    ),
    "compare_evaluation_runs": (
        "wlx_harness_core.wlx_eval_report",
        "compare_evaluation_runs",
    ),
    "evaluate_trajectory": (
        "wlx_harness_core.wlx_trajectory_evaluator",
        "evaluate_trajectory",
    ),
    "evaluate_frozen_requirements": (
        "wlx_harness_core.wlx_eval_requirements",
        "evaluate_frozen_requirements",
    ),
    "freeze_rubric_bundle": (
        "wlx_harness_core.wlx_eval_rubric",
        "freeze_rubric_bundle",
    ),
    "finalize_rubric_set": (
        "wlx_harness_core.wlx_eval_rubric_pipeline",
        "finalize_rubric_set",
    ),
    "ExtraFieldsContractError": (
        "wlx_harness_core.wlx_stage_grpo",
        "ExtraFieldsContractError",
    ),
    "GrpoStageConfig": ("wlx_harness_core.wlx_stage_grpo", "GrpoStageConfig"),
    "HarnessConfig": ("wlx_harness_core.wlx_config", "HarnessConfig"),
    "HarnessError": ("wlx_harness_core.wlx_contracts", "HarnessError"),
    "LegacyChatPolicyAdapter": (
        "wlx_harness_core.wlx_policy",
        "LegacyChatPolicyAdapter",
    ),
    "ModelRequest": ("wlx_harness_core.wlx_contracts", "ModelRequest"),
    "ObservationPolicy": ("wlx_harness_core.wlx_config", "ObservationPolicy"),
    "OpenAICompatibleTeacherPolicy": (
        "wlx_harness_core.wlx_sft_policy",
        "OpenAICompatibleTeacherPolicy",
    ),
    "RewardContractError": ("wlx_harness_core.wlx_reward", "RewardContractError"),
    "run_offline_evaluation": (
        "wlx_harness_core.wlx_eval_pipeline",
        "run_offline_evaluation",
    ),
    "run_rubric_generation": (
        "wlx_harness_core.wlx_eval_rubric_pipeline",
        "run_rubric_generation",
    ),
    "run_judge_batch": (
        "wlx_harness_core.wlx_eval_judge_pipeline",
        "run_judge_batch",
    ),
    "SFTStageAdapter": ("wlx_harness_core.wlx_stage_sft", "SFTStageAdapter"),
    "SamplingConfig": (
        "wlx_harness_core.wlx_sft_contracts",
        "SamplingConfig",
    ),
    "SamplingMode": (
        "wlx_harness_core.wlx_sft_contracts",
        "SamplingMode",
    ),
    "SHOPPING_TOOL_REGISTRY": (
        "wlx_harness_core.wlx_tools",
        "SHOPPING_TOOL_REGISTRY",
    ),
    "SftTask": ("wlx_harness_core.wlx_sft_contracts", "SftTask"),
    "ShopSimulatorEnvironmentAdapter": (
        "wlx_harness_core.wlx_environment",
        "ShopSimulatorEnvironmentAdapter",
    ),
    "TerminationCategory": (
        "wlx_harness_core.wlx_contracts",
        "TerminationCategory",
    ),
    "ToolAction": ("wlx_harness_core.wlx_contracts", "ToolAction"),
    "ToolCall": ("wlx_harness_core.wlx_contracts", "ToolCall"),
    "ToolConfigContractError": (
        "wlx_harness_core.wlx_stage_grpo",
        "ToolConfigContractError",
    ),
    "ToolRegistry": ("wlx_harness_core.wlx_tools", "ToolRegistry"),
    "ToolRegistryError": ("wlx_harness_core.wlx_tools", "ToolRegistryError"),
    "Trajectory": ("wlx_harness_core.wlx_contracts", "Trajectory"),
    "TrajectoryStep": ("wlx_harness_core.wlx_contracts", "TrajectoryStep"),
    "TransformersRuntimeTokenCounter": (
        "wlx_harness_core.wlx_sft_tokenizer",
        "TransformersRuntimeTokenCounter",
    ),
    "TransformersTrainingTokenCounter": (
        "wlx_harness_core.wlx_sft_tokenizer",
        "TransformersTrainingTokenCounter",
    ),
    "VerlHarnessBridge": (
        "wlx_harness_core.wlx_stage_grpo",
        "VerlHarnessBridge",
    ),
    "WLX_SFT_TOOL_REGISTRY": (
        "wlx_harness_core.wlx_tools",
        "WLX_SFT_TOOL_REGISTRY",
    ),
    "WLX_SFT_SYSTEM_PROMPT": (
        "wlx_harness_core.wlx_sft_prompt",
        "WLX_SFT_SYSTEM_PROMPT",
    ),
    "WLX_SFT_INITIAL_MESSAGE_VERSION": (
        "wlx_harness_core.wlx_sft_prompt",
        "WLX_SFT_INITIAL_MESSAGE_VERSION",
    ),
    "WLX_SFT_SYSTEM_PROMPT_SHA256": (
        "wlx_harness_core.wlx_sft_prompt",
        "WLX_SFT_SYSTEM_PROMPT_SHA256",
    ),
    "WLX_SFT_SYSTEM_PROMPT_VERSION": (
        "wlx_harness_core.wlx_sft_prompt",
        "WLX_SFT_SYSTEM_PROMPT_VERSION",
    ),
    "WlxSftDataPipeline": (
        "wlx_harness_core.wlx_sft_pipeline",
        "WlxSftDataPipeline",
    ),
    "validate_process_judgment": (
        "wlx_harness_core.wlx_eval_judge",
        "validate_process_judgment",
    ),
    "validate_full_judgment": (
        "wlx_harness_core.wlx_eval_judge",
        "validate_full_judgment",
    ),
    "default_sft_harness_config": (
        "wlx_harness_core.wlx_sft_pipeline",
        "default_sft_harness_config",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """按需加载公开对象；输入对象名，返回对应的类、函数或常量。"""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """返回这个包能看到的全部名字，方便自动补全和调试工具展示。"""

    return sorted({*globals(), *__all__})
