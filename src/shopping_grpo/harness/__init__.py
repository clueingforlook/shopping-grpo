"""Harness Core 对外提供的统一入口。

这里只登记外部可以使用的名字，不会一上来就加载 ShopSimulator、模型客户端或
veRL。代码真正用到某个名字时，才会加载它所在的模块，避免无关依赖影响其他阶段。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final


_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "AssistantTurn": ("shopping_grpo.harness.contracts", "AssistantTurn"),
    "BridgeDependencyError": (
        "shopping_grpo.harness.stage_grpo",
        "BridgeDependencyError",
    ),
    "ConfigValidationError": (
        "shopping_grpo.harness.config",
        "ConfigValidationError",
    ),
    "ContextPolicy": ("shopping_grpo.harness.config", "ContextPolicy"),
    "DatasetBuildConfig": (
        "shopping_grpo.harness.sft_dataset",
        "DatasetBuildConfig",
    ),
    "DeepSeekRubricClient": (
        "shopping_grpo.harness.eval_rubric_generator",
        "DeepSeekRubricClient",
    ),
    "EnvironmentResult": ("shopping_grpo.harness.environment", "EnvironmentResult"),
    "EpisodeRequest": ("shopping_grpo.harness.contracts", "EpisodeRequest"),
    "EpisodeRunner": ("shopping_grpo.harness.runner", "EpisodeRunner"),
    "ErrorCategory": ("shopping_grpo.harness.contracts", "ErrorCategory"),
    "EvaluationStageAdapter": (
        "shopping_grpo.harness.stage_evaluation",
        "EvaluationStageAdapter",
    ),
    "build_process_judge_payload": (
        "shopping_grpo.harness.eval_judge",
        "build_process_judge_payload",
    ),
    "build_full_judge_payload": (
        "shopping_grpo.harness.eval_judge",
        "build_full_judge_payload",
    ),
    "build_judge_gold_draft": (
        "shopping_grpo.harness.eval_judge_gold",
        "build_judge_gold_draft",
    ),
    "calibrate_judge": (
        "shopping_grpo.harness.eval_judge_pipeline",
        "calibrate_judge",
    ),
    "compare_evaluation_runs": (
        "shopping_grpo.harness.eval_report",
        "compare_evaluation_runs",
    ),
    "evaluate_trajectory": (
        "shopping_grpo.harness.trajectory_evaluator",
        "evaluate_trajectory",
    ),
    "evaluate_frozen_requirements": (
        "shopping_grpo.harness.eval_requirements",
        "evaluate_frozen_requirements",
    ),
    "freeze_rubric_bundle": (
        "shopping_grpo.harness.eval_rubric",
        "freeze_rubric_bundle",
    ),
    "finalize_rubric_set": (
        "shopping_grpo.harness.eval_rubric_pipeline",
        "finalize_rubric_set",
    ),
    "ExtraFieldsContractError": (
        "shopping_grpo.harness.stage_grpo",
        "ExtraFieldsContractError",
    ),
    "GrpoStageConfig": ("shopping_grpo.harness.stage_grpo", "GrpoStageConfig"),
    "HarnessConfig": ("shopping_grpo.harness.config", "HarnessConfig"),
    "HarnessError": ("shopping_grpo.harness.contracts", "HarnessError"),
    "LegacyChatPolicyAdapter": (
        "shopping_grpo.harness.policy",
        "LegacyChatPolicyAdapter",
    ),
    "ModelRequest": ("shopping_grpo.harness.contracts", "ModelRequest"),
    "ObservationPolicy": ("shopping_grpo.harness.config", "ObservationPolicy"),
    "OpenAICompatibleTeacherPolicy": (
        "shopping_grpo.harness.sft_policy",
        "OpenAICompatibleTeacherPolicy",
    ),
    "RewardContractError": ("shopping_grpo.harness.reward", "RewardContractError"),
    "run_offline_evaluation": (
        "shopping_grpo.harness.eval_pipeline",
        "run_offline_evaluation",
    ),
    "run_rubric_generation": (
        "shopping_grpo.harness.eval_rubric_pipeline",
        "run_rubric_generation",
    ),
    "run_judge_batch": (
        "shopping_grpo.harness.eval_judge_pipeline",
        "run_judge_batch",
    ),
    "SFTStageAdapter": ("shopping_grpo.harness.stage_sft", "SFTStageAdapter"),
    "SamplingConfig": (
        "shopping_grpo.harness.sft_contracts",
        "SamplingConfig",
    ),
    "SamplingMode": (
        "shopping_grpo.harness.sft_contracts",
        "SamplingMode",
    ),
    "SHOPPING_TOOL_REGISTRY": (
        "shopping_grpo.harness.tools",
        "SHOPPING_TOOL_REGISTRY",
    ),
    "SftTask": ("shopping_grpo.harness.sft_contracts", "SftTask"),
    "ShopSimulatorEnvironmentAdapter": (
        "shopping_grpo.harness.environment",
        "ShopSimulatorEnvironmentAdapter",
    ),
    "TerminationCategory": (
        "shopping_grpo.harness.contracts",
        "TerminationCategory",
    ),
    "ToolAction": ("shopping_grpo.harness.contracts", "ToolAction"),
    "ToolCall": ("shopping_grpo.harness.contracts", "ToolCall"),
    "ToolConfigContractError": (
        "shopping_grpo.harness.stage_grpo",
        "ToolConfigContractError",
    ),
    "ToolRegistry": ("shopping_grpo.harness.tools", "ToolRegistry"),
    "ToolRegistryError": ("shopping_grpo.harness.tools", "ToolRegistryError"),
    "Trajectory": ("shopping_grpo.harness.contracts", "Trajectory"),
    "TrajectoryStep": ("shopping_grpo.harness.contracts", "TrajectoryStep"),
    "TransformersRuntimeTokenCounter": (
        "shopping_grpo.harness.sft_tokenizer",
        "TransformersRuntimeTokenCounter",
    ),
    "TransformersTrainingTokenCounter": (
        "shopping_grpo.harness.sft_tokenizer",
        "TransformersTrainingTokenCounter",
    ),
    "VerlHarnessBridge": (
        "shopping_grpo.harness.stage_grpo",
        "VerlHarnessBridge",
    ),
    "SFT_TOOL_REGISTRY": (
        "shopping_grpo.harness.tools",
        "SFT_TOOL_REGISTRY",
    ),
    "SFT_SYSTEM_PROMPT": (
        "shopping_grpo.harness.sft_prompt",
        "SFT_SYSTEM_PROMPT",
    ),
    "SFT_INITIAL_MESSAGE_VERSION": (
        "shopping_grpo.harness.sft_prompt",
        "SFT_INITIAL_MESSAGE_VERSION",
    ),
    "SFT_SYSTEM_PROMPT_SHA256": (
        "shopping_grpo.harness.sft_prompt",
        "SFT_SYSTEM_PROMPT_SHA256",
    ),
    "SFT_SYSTEM_PROMPT_VERSION": (
        "shopping_grpo.harness.sft_prompt",
        "SFT_SYSTEM_PROMPT_VERSION",
    ),
    "SftDataPipeline": (
        "shopping_grpo.harness.sft_pipeline",
        "SftDataPipeline",
    ),
    "validate_process_judgment": (
        "shopping_grpo.harness.eval_judge",
        "validate_process_judgment",
    ),
    "validate_full_judgment": (
        "shopping_grpo.harness.eval_judge",
        "validate_full_judgment",
    ),
    "default_sft_harness_config": (
        "shopping_grpo.harness.sft_pipeline",
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
