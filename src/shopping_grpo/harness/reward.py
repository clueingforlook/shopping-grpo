"""统一检查各阶段返回的 Reward-v3 奖励是否完整、可信且互相不矛盾。"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

from shopping_grpo.harness.contracts import REWARD_VERSION


REWARD_V3_TYPES = frozenset(
    {
        "gold_purchase",
        "valid_alternative_purchase",
        "partial_alternative_purchase",
        "graceful_stop",
        "early_abstain",
        "wrong_purchase",
        "repeat_loop",
        "max_steps",
        "reward_unverifiable",
    }
)
PURCHASE_SUCCESS_TYPES = frozenset(
    {"gold_purchase", "valid_alternative_purchase"}
)
DIMENSION_NAMES = ("brand", "model", "core_functions", "key_options")
UNSCORED_REWARD_TYPES = frozenset(
    {"graceful_stop", "early_abstain", "repeat_loop", "max_steps"}
)


class RewardContractError(ValueError):
    """表示环境给出的奖励不符合 Reward-v3 约定，不能安全用于训练或评测。"""

    pass


def validate_reward_v3(
    raw_detail: object,
    *,
    terminal_reward: float | None = None,
) -> dict[str, Any]:
    """检查 Reward-v3 的字段和值是否合理，并只返回允许公开使用的字段。

    这一步会核对奖励类型、是否购买成功、各项得分和最终 reward 是否互相一致。
    这样可以防止格式错误或前后矛盾的奖励悄悄进入 SFT、GRPO 或评测结果。
    """

    if not isinstance(raw_detail, Mapping):
        raise RewardContractError("reward_detail must be an object")
    detail = deepcopy(dict(raw_detail))
    if detail.get("reward_version") != REWARD_VERSION:
        raise RewardContractError("unsupported reward_version")
    reward_type = detail.get("reward_type")
    if reward_type not in REWARD_V3_TYPES:
        raise RewardContractError(f"unknown Reward-v3 reward_type: {reward_type!r}")
    if detail.get("termination_reason") != reward_type:
        raise RewardContractError("termination_reason must equal reward_type")

    reward_valid = detail.get("reward_valid")
    if not isinstance(reward_valid, bool):
        raise RewardContractError("reward_valid must be boolean")
    if (reward_type == "reward_unverifiable") != (not reward_valid):
        raise RewardContractError("only reward_unverifiable may set reward_valid=false")
    sampling_invalid = detail.get("sampling_invalid")
    if not isinstance(sampling_invalid, bool) or sampling_invalid != (not reward_valid):
        raise RewardContractError("sampling_invalid must equal not reward_valid")

    purchase_success = detail.get("purchase_success")
    if not isinstance(purchase_success, bool):
        raise RewardContractError("purchase_success must be boolean")
    if purchase_success != (reward_type in PURCHASE_SUCCESS_TYPES):
        raise RewardContractError("purchase_success is inconsistent with reward_type")
    target_asin_match = detail.get("target_asin_match")
    if not isinstance(target_asin_match, bool):
        raise RewardContractError("target_asin_match must be boolean")

    utility = _finite_number("terminal_utility", detail.get("terminal_utility"))
    if terminal_reward is not None and utility != float(terminal_reward):
        raise RewardContractError("terminal_utility differs from terminal reward")
    _require_fields(detail, "weighted_score", "evidence_coverage", "dimension_scores")
    weighted = _finite_number("weighted_score", detail["weighted_score"])
    evidence = _finite_number("evidence_coverage", detail["evidence_coverage"])
    if not 0.0 <= weighted <= 1.0:
        raise RewardContractError("weighted_score must be in [0, 1]")
    if not 0.0 <= evidence <= 1.0:
        raise RewardContractError("evidence_coverage must be in [0, 1]")

    gates = detail.get("hard_gates")
    if not isinstance(gates, Mapping):
        raise RewardContractError("hard_gates must be an object")
    if reward_type == "gold_purchase":
        missing = {"category", "budget"}.difference(gates)
        if missing:
            raise RewardContractError(
                "gold_purchase is missing hard gates: " + ", ".join(sorted(missing))
            )
    public_gates: dict[str, dict[str, Any]] = {}
    for name, gate in gates.items():
        if not isinstance(gate, Mapping):
            raise RewardContractError(f"hard gate {name!r} must be an object")
        status = gate.get("status")
        if status not in {"pass", "fail", "unverifiable"}:
            raise RewardContractError(f"hard gate {name!r} has invalid status")
        if gate.get("passed") != (status == "pass"):
            raise RewardContractError(f"hard gate {name!r} has inconsistent passed")
        if gate.get("verifiable") != (status != "unverifiable"):
            raise RewardContractError(f"hard gate {name!r} has inconsistent verifiable")
        public_gates[str(name)] = {
            "status": status,
            "passed": gate["passed"],
            "verifiable": gate["verifiable"],
            "comparator": str(gate.get("comparator") or ""),
            "source_field": str(gate.get("source_field") or ""),
        }

    dimensions = detail["dimension_scores"]
    if not isinstance(dimensions, Mapping):
        raise RewardContractError("dimension_scores must be an object")
    public_dimensions: dict[str, float] = {}
    if not dimensions and reward_type in UNSCORED_REWARD_TYPES:
        # ShopSimulator 不会给未评价具体商品的终局生成偏好维度；公开结果仍补成
        # 稳定的四维零值，方便下游统一读取，同时保留环境原始 payload 不变。
        public_dimensions = {name: 0.0 for name in DIMENSION_NAMES}
    else:
        for name in DIMENSION_NAMES:
            if name not in dimensions:
                raise RewardContractError(f"dimension_scores is missing {name!r}")
            score = _finite_number(f"dimension_scores.{name}", dimensions[name])
            if not 0.0 <= score <= 1.0:
                raise RewardContractError(
                    f"dimension_scores.{name} must be in [0, 1]"
                )
            public_dimensions[name] = score

    return {
        "reward_version": REWARD_VERSION,
        "reward_type": reward_type,
        "termination_reason": reward_type,
        "reward_valid": reward_valid,
        "sampling_invalid": sampling_invalid,
        "purchase_success": purchase_success,
        "target_asin_match": target_asin_match,
        "terminal_utility": utility,
        "weighted_score": weighted,
        "evidence_coverage": evidence,
        "hard_gates": public_gates,
        "dimension_scores": public_dimensions,
    }


def validate_terminal_result(
    result: object,
    *,
    required_reward_version: str | None = REWARD_VERSION,
) -> dict[str, Any]:
    """检查一局结束时的环境结果，并返回经过验证的奖励明细。

    只有 ``done`` 和 ``over`` 都为真才算真正结束；如果指定了奖励版本，还会继续走
    对应的严格检查。这样运行器不会把半途状态或无法核实的奖励当成最终成绩。
    """

    if not isinstance(result, Mapping):
        raise RewardContractError("terminal result must be an object")
    if result.get("done") is not True or result.get("over") is not True:
        raise RewardContractError("terminal result must set done=true and over=true")
    reward = _finite_number("reward", result.get("reward", 0.0))
    detail = result.get("reward_detail")
    if required_reward_version is None:
        return deepcopy(dict(detail)) if isinstance(detail, Mapping) else {}
    if required_reward_version != REWARD_VERSION:
        raise RewardContractError(
            f"unsupported required reward version: {required_reward_version!r}"
        )
    return validate_reward_v3(detail, terminal_reward=reward)


def _require_fields(value: Mapping[str, Any], *names: str) -> None:
    """确认奖励明细包含所有必填字段，缺少任何一项都直接说明问题。"""

    missing = [name for name in names if name not in value]
    if missing:
        raise RewardContractError(
            "reward_detail is missing fields: " + ", ".join(missing)
        )


def _finite_number(name: str, value: object) -> float:
    """把奖励字段转成普通浮点数，并拒绝无穷大、非数字等不可训练的值。"""

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RewardContractError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise RewardContractError(f"{name} must be finite")
    return result


__all__ = [
    "REWARD_V3_TYPES",
    "RewardContractError",
    "validate_reward_v3",
    "validate_terminal_result",
]
