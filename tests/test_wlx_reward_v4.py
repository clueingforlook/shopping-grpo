"""Deterministic tests for WLX ORM and conservative PRM events."""

from shopping_grpo.training.grpo.wlx_reward_v4 import (
    compute_wlx_orm,
    compute_wlx_prm,
    reward_score_for_sampling,
)


def _purchase_state(*, reward_type="valid_alternative_purchase", category="pass", budget="pass"):
    return {
        "done": True,
        "terminal_result": {"done": True, "over": True},
        "reward_type": reward_type,
        "infrastructure_invalid": False,
        "reward_detail": {
            "hard_gates": {
                "category": {"status": category},
                "budget": {"status": budget},
            },
            "dimension_details": {
                "brand": {"required_count": 1, "passed_count": 1},
                "model": {"required_count": 0, "passed_count": 0},
                "core_functions": {"required_count": 3, "passed_count": 2},
                "key_options": {"required_count": 2, "passed_count": 1},
            },
            "option_failures": {"wrong_values": [], "missing_axes": []},
        },
        "model_steps": [],
    }


def test_orm_gold_is_one():
    state = _purchase_state(reward_type="gold_purchase")
    assert compute_wlx_orm(state)["terminal_utility"] == 1.0


def test_orm_partial_uses_category_gate_and_requirement_ratios():
    reward = compute_wlx_orm(_purchase_state())
    assert reward["attribute_ratio"] == 0.75
    assert reward["option_ratio"] == 0.5
    assert reward["terminal_utility"] == 0.1875


def test_orm_wrong_category_and_no_purchase_are_ordered():
    wrong_category = compute_wlx_orm(_purchase_state(category="fail"))
    no_purchase = compute_wlx_orm({"done": False, "reward_type": None})
    assert wrong_category["terminal_utility"] == -0.5
    assert no_purchase["terminal_utility"] == -0.25


def test_infrastructure_failure_is_masked_not_punished():
    reward = compute_wlx_orm({"infrastructure_invalid": True})
    assert reward["terminal_utility"] == 0.0
    assert reward["sampling_invalid"] is True


def test_prm_localizes_wrong_option_and_bad_buy():
    state = _purchase_state(category="fail")
    state["reward_detail"]["option_failures"] = {
        "wrong_values": [{"axis": "color", "selected": "red"}],
        "missing_axes": [],
    }
    state["model_steps"] = [
        {"tool": "select_option", "parameters": {"value": "red"}},
        {"tool": "buy_now", "parameters": {}},
    ]
    prm = compute_wlx_prm(state)
    assert prm["step_prm"] == [-0.5, -1.0]
    assert prm["step_kinds"] == ["wrong_option", "bad_buy"]
    assert reward_score_for_sampling(compute_wlx_orm(state), prm) != compute_wlx_orm(state)["terminal_utility"]


def test_prm_penalizes_only_sustained_no_progress():
    state = {"done": False, "model_steps": []}
    for no_progress in (1, 2, 3):
        state["model_steps"].append(
            {
                "tool": "search_products",
                "parameters": {"query": "same"},
                "progress": {"no_progress_steps": no_progress, "runtime_progress_added": []},
            }
        )
    prm = compute_wlx_prm(state)
    assert prm["step_prm"] == [0.0, 0.0, -0.25]
