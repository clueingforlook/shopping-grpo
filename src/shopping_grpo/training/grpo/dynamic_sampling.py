"""Pure reward-group selection used by the bounded veRL sampling patch."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    if not shopping_infos:
        return {}

    utilities = []
    gold = []
    partial = []
    no_purchase = []
    wrong_category = []
    attribute_ratios = []
    option_ratios = []
    price_pass = []
    steps = []
    done = []
    max_steps = []
    infrastructure_invalid = []
    reward_unverifiable = []
    sampling_invalid = []
    prm_wrong_option = []
    prm_bad_buy = []
    prm_no_progress = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        if reward.get("version") != "wlx-reward-v4":
            raise ValueError(f"shopping reward at index {index} is not Reward v4")
        try:
            utility = float(reward["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"shopping reward at index {index} has no utility") from exc
        if not math.isfinite(utility):
            raise ValueError(f"shopping reward at index {index} has non-finite utility")
        utilities.append(utility)
        gold.append(float(reward.get("gold", 0.0)))
        partial.append(float(reward.get("partial_purchase", 0.0)))
        no_purchase.append(float(reward.get("no_purchase", 0.0)))
        wrong_category.append(float(reward.get("wrong_category", 0.0)))
        attribute_ratios.append(float(reward.get("attribute_ratio", 0.0)))
        option_ratios.append(float(reward.get("option_ratio", 0.0)))
        price_pass.append(float(reward.get("price_pass", 0.0)))
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))
        sampling_invalid.append(float(bool(reward.get("sampling_invalid"))))
        prm = info.get("prm") or {}
        kinds = list(prm.get("step_kinds") or ()) if isinstance(prm, Mapping) else []
        prm_wrong_option.append(float(kinds.count("wrong_option")))
        prm_bad_buy.append(float(kinds.count("bad_buy")))
        prm_no_progress.append(float(kinds.count("no_progress")))

    def mean(values):
        return sum(values) / len(values)

    return {
        "wlx_reward/orm_min": min(utilities),
        "wlx_reward/orm_mean": mean(utilities),
        "wlx_reward/orm_max": max(utilities),
        "wlx_reward/gold_rate": mean(gold),
        "wlx_reward/partial_purchase_rate": mean(partial),
        "wlx_reward/no_purchase_rate": mean(no_purchase),
        "wlx_reward/wrong_category_rate": mean(wrong_category),
        "wlx_reward/attribute_ratio_mean": mean(attribute_ratios),
        "wlx_reward/option_ratio_mean": mean(option_ratios),
        "wlx_reward/price_pass_rate": mean(price_pass),
        "wlx_prm/wrong_option_per_trajectory": mean(prm_wrong_option),
        "wlx_prm/bad_buy_per_trajectory": mean(prm_bad_buy),
        "wlx_prm/no_progress_per_trajectory": mean(prm_no_progress),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
        "trajectory/sampling_invalid_rate": mean(sampling_invalid),
    }


def extract_shopping_group_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[float], list[bool], list[bool], list[tuple[str, ...]]]:
    """Return terminal utility, success metrics, and explicit invalid reasons."""
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    invalid_reasons = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            terminal_utility = float(info["reward"]["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping extra field at index {index} is missing terminal_utility"
            ) from exc
        if not math.isfinite(terminal_utility):
            raise ValueError(
                f"shopping terminal_utility at index {index} is not finite"
            )
        raw_purchase_success = info["reward"].get(
            "gold", info["reward"].get("purchase_success")
        )
        if not isinstance(raw_purchase_success, (bool, int, float)):
            raise ValueError(
                f"shopping extra field at index {index} is missing purchase_success"
            )
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        reasons = []
        if bool(info["infrastructure_invalid"]):
            reasons.append("infrastructure_invalid")
        reward_sampling_invalid = bool(
            info["reward"].get("sampling_invalid", False)
        )
        if reward_sampling_invalid and not reasons:
            reasons.append(str(info["reward"].get("outcome") or "reward_sampling_invalid"))
        terminal_utilities.append(terminal_utility)
        purchase_success.append(bool(raw_purchase_success))
        sampling_invalid.append(bool(reasons))
        invalid_reasons.append(tuple(reasons))
    return (
        terminal_utilities,
        purchase_success,
        sampling_invalid,
        invalid_reasons,
    )


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    terminal_utilities: Sequence[float] | None = None,
    purchase_success: Sequence[bool] | None = None,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with non-constant reward.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    optional_sequences = {
        "terminal_utilities": terminal_utilities,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
        "sampling_invalid_reasons": sampling_invalid_reasons,
    }
    for name, values in optional_sequences.items():
        if values is not None and len(values) != len(uids):
            raise ValueError(f"{name} must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    utility_values = (
        terminal_utilities if terminal_utilities is not None else seq_rewards
    )
    success_values = (
        purchase_success if purchase_success is not None else [False] * len(uids)
    )
    invalid_values = (
        sampling_invalid if sampling_invalid is not None else [False] * len(uids)
    )
    reason_values = (
        sampling_invalid_reasons
        if sampling_invalid_reasons is not None
        else [()] * len(uids)
    )
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (
        uid,
        raw_reward,
        raw_utility,
        raw_success,
        raw_invalid,
        raw_reasons,
    ) in enumerate(
        zip(
            uids,
            seq_rewards,
            utility_values,
            success_values,
            invalid_values,
            reason_values,
            strict=True,
        )
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        utility = float(raw_utility)
        if not math.isfinite(utility):
            raise ValueError(
                f"terminal_utility at index {index} is not finite: {raw_utility!r}"
            )

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "terminal_utilities": [],
                "purchase_success": [],
                "sampling_invalid": [],
                "sampling_invalid_reasons": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["terminal_utilities"].append(utility)
        group["purchase_success"].append(bool(raw_success))
        group["sampling_invalid"].append(bool(raw_invalid))
        group["sampling_invalid_reasons"].extend(str(reason) for reason in raw_reasons)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    for uid, group in grouped.items():
        utilities = group["terminal_utilities"]
        valid_positions = [
            position
            for position, invalid in enumerate(group["sampling_invalid"])
            if not invalid
        ]
        valid_utilities = [utilities[position] for position in valid_positions]
        utility_min = min(valid_utilities) if valid_utilities else 0.0
        utility_max = max(valid_utilities) if valid_utilities else 0.0
        utility_varying = utility_max - utility_min > tolerance
        has_sampling_invalid = any(group["sampling_invalid"])
        has_step_prm_signal = any(
            abs(group["rewards"][position] - utilities[position]) > tolerance
            for position in valid_positions
        )
        reasons = tuple(sorted(set(group["sampling_invalid_reasons"])))
        if len(valid_positions) < 2:
            drop_reason = "insufficient_valid_trajectories"
        elif not utility_varying and not has_step_prm_signal:
            drop_reason = "constant_reward"
        else:
            drop_reason = None
        keep = drop_reason is None
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(group["rewards"]),
                "terminal_utilities": tuple(utilities),
                "purchase_success": tuple(group["purchase_success"]),
                "utility_min": utility_min,
                "utility_max": utility_max,
                "reward_varying": utility_varying or has_step_prm_signal,
                "orm_varying": utility_varying,
                "step_prm_signal": has_step_prm_signal,
                "valid_trajectory_count": len(valid_positions),
                "sampling_invalid": has_sampling_invalid,
                "sampling_invalid_reasons": reasons,
                "drop_reason": drop_reason,
                "kept": keep,
            }
        )

    kept_uid_set = set(kept_uids)
    trajectory_indices = [index for index, uid in enumerate(uids) if uid in kept_uid_set]
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "all_zero_utility_group_count": sum(
            max(abs(value) for value in group["terminal_utilities"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            all(group["purchase_success"])
            for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(group["purchase_success"]) for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "sampling_invalid_reason_counts": {
            reason: sum(
                reason in group["sampling_invalid_reasons"] for group in groups
            )
            for reason in sorted(
                {
                    reason
                    for group in groups
                    for reason in group["sampling_invalid_reasons"]
                }
            )
        },
        # Compatibility aliases for existing monitoring code.
        "infrastructure_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats
