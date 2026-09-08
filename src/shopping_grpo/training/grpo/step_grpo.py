"""Token-level Step-GRPO advantage estimator and veRL runtime hook."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


ADV_ESTIMATOR = "step_grpo"


def _estimator_name(value: object) -> str:
    return str(getattr(value, "value", value))


def _require_sequence(batch: Mapping[str, Any], key: str, size: int) -> Sequence[Any]:
    value = batch.get(key)
    if value is None or len(value) != size:
        raise ValueError(f"{key} must contain one value per trajectory")
    return value


def _as_list(value: Any) -> list[Any]:
    return [] if value is None else list(value)


@register_adv_est(ADV_ESTIMATOR)
def compute_step_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    *,
    non_tensor_batch: Mapping[str, Any],
    batch: Mapping[str, Any] | None = None,
    config: object = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine group-normalized ORM with PRM on exact assistant token spans."""
    del batch, config
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token rewards and response mask must have identical shapes")
    batch_size, response_length = response_mask.shape
    if len(index) != batch_size:
        raise ValueError("uid count must equal trajectory batch size")
    shopping = _require_sequence(non_tensor_batch, "shopping", batch_size)
    spans = _require_sequence(non_tensor_batch, "wlx_step_spans", batch_size)
    step_prm = _require_sequence(non_tensor_batch, "wlx_step_prm", batch_size)
    step_kinds = _require_sequence(non_tensor_batch, "wlx_step_kinds", batch_size)

    scores = torch.zeros(batch_size, dtype=torch.float32, device=response_mask.device)
    valid = torch.zeros(batch_size, dtype=torch.bool, device=response_mask.device)
    for row, info in enumerate(shopping):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping[{row}] is missing reward diagnostics")
        reward = info["reward"]
        if reward.get("version") != "wlx-reward-v4":
            raise ValueError(f"shopping[{row}] does not use Reward v4")
        scores[row] = float(reward["terminal_utility"])
        valid[row] = not bool(reward.get("sampling_invalid"))

    group_rows: dict[object, list[int]] = defaultdict(list)
    for row, uid in enumerate(index.tolist() if hasattr(index, "tolist") else index):
        group_rows[uid].append(row)

    orm_advantage = torch.zeros_like(scores)
    std_floor_groups = 0
    with torch.no_grad():
        for uid, rows in group_rows.items():
            valid_rows = [row for row in rows if bool(valid[row])]
            if len(valid_rows) < 2:
                raise ValueError(
                    f"Step-GRPO group {uid!r} has fewer than two valid trajectories"
                )
            values = scores[valid_rows]
            mean = values.mean()
            # Match veRL's default GRPO convention: sample standard deviation.
            std = values.std(unbiased=True)
            if float(std) < 0.1:
                std_floor_groups += 1
            denominator = torch.clamp(std, min=0.1)
            orm_advantage[valid_rows] = torch.clamp(
                (values - mean) / denominator,
                min=-2.0,
                max=2.0,
            )

        advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
        wrong_option_steps = 0
        bad_buy_steps = 0
        no_progress_steps = 0
        for row in range(batch_size):
            if not bool(valid[row]):
                continue
            mask = response_mask[row].to(dtype=torch.bool)
            advantages[row, mask] = orm_advantage[row]
            row_spans = _as_list(spans[row])
            row_prm = _as_list(step_prm[row])
            row_kinds = _as_list(step_kinds[row])
            if not (len(row_spans) == len(row_prm) == len(row_kinds)):
                raise ValueError(f"trajectory {row} has misaligned step fields")
            for raw_span, raw_prm, kind in zip(
                row_spans, row_prm, row_kinds, strict=True
            ):
                if isinstance(raw_span, (str, bytes)):
                    raise ValueError(f"trajectory {row} has invalid step span")
                try:
                    span_size = len(raw_span)
                except TypeError as exc:
                    raise ValueError(f"trajectory {row} has invalid step span") from exc
                if span_size != 2:
                    raise ValueError(f"trajectory {row} has invalid step span")
                start, end = int(raw_span[0]), int(raw_span[1])
                if not 0 <= start <= end <= response_length:
                    raise ValueError(f"trajectory {row} step span is out of bounds")
                if start == end:
                    continue
                prm = float(raw_prm)
                value = orm_advantage[row] + prm
                if kind == "wrong_option":
                    value = torch.minimum(value, value.new_tensor(-0.25))
                    wrong_option_steps += 1
                elif kind == "bad_buy":
                    value = torch.minimum(value, value.new_tensor(-0.50))
                    bad_buy_steps += 1
                elif kind == "no_progress":
                    no_progress_steps += 1
                elif kind != "normal":
                    raise ValueError(f"trajectory {row} has unknown step kind {kind!r}")
                span_mask = mask[start:end]
                advantages[row, start:end][span_mask] = torch.clamp(
                    value, min=-2.0, max=2.0
                )

        advantages *= response_mask.to(dtype=advantages.dtype)
        active = response_mask.to(dtype=torch.bool) & valid.unsqueeze(-1)
        values = advantages[active]
        if values.numel():
            positive = float((values > 0).float().mean())
            negative = float((values < 0).float().mean())
            zero = float((values == 0).float().mean())
            clipped = float((values.abs() >= 2.0 - 1.0e-6).float().mean())
            value_mean = float(values.mean())
            value_std = float(values.std(unbiased=False))
            value_min = float(values.min())
            value_max = float(values.max())
        else:
            positive = negative = zero = clipped = 0.0
            value_mean = value_std = value_min = value_max = 0.0
        summary = {
            "wlx_advantage/mean": value_mean,
            "wlx_advantage/std": value_std,
            "wlx_advantage/min": value_min,
            "wlx_advantage/max": value_max,
            "wlx_advantage/positive_token_ratio": positive,
            "wlx_advantage/negative_token_ratio": negative,
            "wlx_advantage/zero_token_ratio": zero,
            "wlx_advantage/clip_ratio": clipped,
            "wlx_advantage/invalid_trajectory_ratio": float((~valid).float().mean()),
            "wlx_advantage/std_floor_group_ratio": (
                std_floor_groups / max(len(group_rows), 1)
            ),
            "wlx_prm/wrong_option_steps": float(wrong_option_steps),
            "wlx_prm/bad_buy_steps": float(bad_buy_steps),
            "wlx_prm/no_progress_steps": float(no_progress_steps),
        }
        if isinstance(non_tensor_batch, dict):
            non_tensor_batch["wlx_advantage_summary"] = np.array(
                [summary] * batch_size, dtype=object
            )
    return advantages, advantages.clone()


def install_step_grpo() -> None:
    """Patch veRL's narrow dispatch/metrics boundary inside each Ray worker."""
    from verl.trainer.ppo import ray_trainer

    if getattr(ray_trainer.compute_advantage, "_step_grpo", False):
        return
    original_compute_advantage = ray_trainer.compute_advantage
    original_compute_data_metrics = ray_trainer.compute_data_metrics

    def compute_advantage_with_wlx(
        data,
        adv_estimator,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=None,
    ):
        if _estimator_name(adv_estimator) != ADV_ESTIMATOR:
            return original_compute_advantage(
                data,
                adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=config,
            )
        if "response_mask" not in data.batch:
            data.batch["response_mask"] = ray_trainer.compute_response_mask(data)
        advantages, returns = compute_step_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            non_tensor_batch=data.non_tensor_batch,
            batch=data.batch,
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    compute_advantage_with_wlx._step_grpo = True
    ray_trainer.compute_advantage = compute_advantage_with_wlx

    def compute_data_metrics_with_wlx(batch, use_critic=True):
        metrics = original_compute_data_metrics(batch, use_critic=use_critic)
        summaries = batch.non_tensor_batch.get("wlx_advantage_summary")
        if summaries is not None and len(summaries):
            summary = summaries[0]
            if isinstance(summary, Mapping):
                metrics.update({key: float(value) for key, value in summary.items()})
        return metrics

    ray_trainer.compute_data_metrics = compute_data_metrics_with_wlx
