"""Token-level credit assignment tests for Step-GRPO."""

import numpy as np
import torch

from shopping_grpo.training.grpo.step_grpo import compute_step_advantage


def _shopping(score, invalid=False):
    return {
        "reward": {
            "version": "wlx-reward-v4",
            "terminal_utility": score,
            "sampling_invalid": invalid,
        }
    }


def test_group_orm_is_broadcast_only_to_model_tokens():
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]])
    non_tensor = {
        "shopping": np.array([_shopping(1.0), _shopping(0.0)], dtype=object),
        "wlx_step_spans": np.array([[[0, 2]], [[0, 1]]], dtype=object),
        "wlx_step_prm": np.array([[0.0], [0.0]], dtype=object),
        "wlx_step_kinds": np.array([["normal"], ["normal"]], dtype=object),
    }
    advantage, returns = compute_step_advantage(
        torch.zeros_like(mask, dtype=torch.float32),
        mask,
        np.array(["task", "task"], dtype=object),
        non_tensor_batch=non_tensor,
    )
    scale = 2**-0.5
    expected = torch.tensor(
        [[scale, scale, 0.0, scale], [-scale, 0.0, -scale, -scale]]
    )
    assert torch.allclose(advantage, expected)
    assert torch.equal(returns, advantage)


def test_wrong_option_floor_applies_only_to_its_step_span():
    mask = torch.ones((2, 4), dtype=torch.long)
    non_tensor = {
        "shopping": np.array([_shopping(1.0), _shopping(0.0)], dtype=object),
        "wlx_step_spans": np.array([[[0, 2], [2, 4]], [[0, 4]]], dtype=object),
        "wlx_step_prm": np.array([[-0.5, 0.0], [0.0]], dtype=object),
        "wlx_step_kinds": np.array([["wrong_option", "normal"], ["normal"]], dtype=object),
    }
    advantage, _ = compute_step_advantage(
        torch.zeros((2, 4)), mask, np.array([7, 7]), non_tensor_batch=non_tensor
    )
    assert torch.allclose(
        advantage[0], torch.tensor([-0.25, -0.25, 2**-0.5, 2**-0.5])
    )
    assert torch.allclose(advantage[1], torch.full((4,), -(2**-0.5)))


def test_invalid_trajectory_is_zero_and_excluded_from_group_statistics():
    mask = torch.ones((3, 2), dtype=torch.long)
    non_tensor = {
        "shopping": np.array([_shopping(9.0, True), _shopping(0.0), _shopping(1.0)], dtype=object),
        "wlx_step_spans": np.array([[], [], []], dtype=object),
        "wlx_step_prm": np.array([[], [], []], dtype=object),
        "wlx_step_kinds": np.array([[], [], []], dtype=object),
    }
    advantage, _ = compute_step_advantage(
        torch.zeros((3, 2)), mask, np.array([1, 1, 1]), non_tensor_batch=non_tensor
    )
    assert torch.equal(advantage[0], torch.zeros(2))
    assert torch.allclose(advantage[1], torch.full((2,), -(2**-0.5)))
    assert torch.allclose(advantage[2], torch.full((2,), 2**-0.5))
