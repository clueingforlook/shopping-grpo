"""购买结果验证器：只比较用户需求与实际购买，不读取 Reward v3。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence


PURCHASE_VERIFIER_VERSION = "wlx-purchase-verifier-v2"
GOLD_PURCHASE = "gold_purchase"
VALID_ALTERNATIVE_PURCHASE = "valid_alternative_purchase"
UNSUCCESSFUL_PURCHASE = "unsuccessful_purchase"

# 只收录经过人工确认且双向等价的稳定品牌别名。
# 新增条目必须升 Verifier 版本，
# 不能让 LLM 在确定性验证阶段临时猜测别名。
TEXT_EQUIVALENCE_GROUPS = (
    frozenset({"philips", "飞利浦"}),
)


@dataclass(frozen=True)
class PurchaseRequirementContract:
    """Verifier 实际使用的需求集合。

    ``reference_asin`` 只用于区分 Gold 与替代商品，不参与需求匹配。
    其他字段是从 ShopSimulator goal 中白名单提取的用户需求；目标
    商品的标题、完整属性和其他未声明特征不会进入本契约。
    """

    reference_asin: str | None
    category: str | None
    required_attributes: tuple[str, ...]
    required_options: tuple[str, ...]
    price_upper: float | None
    expected_brands: tuple[str, ...] = ()
    expected_models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_asin": self.reference_asin,
            "category": self.category,
            "required_attributes": list(self.required_attributes),
            "required_options": list(self.required_options),
            "price_upper": self.price_upper,
            "expected_brands": list(self.expected_brands),
            "expected_models": list(self.expected_models),
        }


@dataclass(frozen=True)
class PurchaseVerification:
    """一次购买的确定性验证结果。"""

    verifier_valid: bool
    purchase_correct: bool
    purchase_type: str
    reason: str | None
    checks: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier_version": PURCHASE_VERIFIER_VERSION,
            "verifier_valid": self.verifier_valid,
            "purchase_correct": self.purchase_correct,
            "purchase_type": self.purchase_type,
            "reason": self.reason,
            "checks": [deepcopy(dict(item)) for item in self.checks],
        }


def requirements_from_environment_goal(
    goal: Mapping[str, Any],
) -> PurchaseRequirementContract:
    """ShopSimulator goal 转成 需求契约。

    这里只接受 goal 中的需求字段。``name``、``query`` 和 Gold 商品
    的其他特征都会被忽略，避免把 abcde 中用户没要求的 d/e 加入
    验证条件。
    """

    if not isinstance(goal, Mapping):
        raise TypeError("environment goal must be an object")
    attributes = goal.get("expected_core_functions")
    if attributes is None:
        attributes = goal.get("attributes")
    return PurchaseRequirementContract(
        reference_asin=_optional_text(goal.get("asin")),
        category=_optional_text(goal.get("category")),
        required_attributes=_text_tuple(attributes),
        required_options=_goal_option_values(goal),
        price_upper=_optional_positive_float(goal.get("price_upper")),
        expected_brands=_text_tuple(goal.get("expected_brand")),
        expected_models=_text_tuple(goal.get("expected_model")),
    )


def verify_purchase(
    requirements: PurchaseRequirementContract,
    purchase: Mapping[str, Any] | None,
) -> PurchaseVerification:
    """判断实际购买是 Gold、有效替代品，还是未成功购买。"""

    if not isinstance(requirements, PurchaseRequirementContract):
        raise TypeError("requirements must be a PurchaseRequirementContract")
    if not purchase or not _optional_text(purchase.get("asin")):
        return PurchaseVerification(
            verifier_valid=True,
            purchase_correct=False,
            purchase_type=UNSUCCESSFUL_PURCHASE,
            reason="no_purchase",
        )

    checks: list[dict[str, Any]] = []
    if requirements.category:
        actual_category = _optional_text(purchase.get("category"))
        checks.append(
            _check(
                "category",
                _category_matches(requirements.category, actual_category),
                actual_category is not None,
                requirements.category,
                actual_category,
            )
        )

    searchable_text = _purchase_searchable_text(purchase)
    for value in requirements.expected_brands:
        checks.append(
            _check(
                f"brand:{value}",
                _contains(searchable_text, value),
                bool(searchable_text),
                value,
                searchable_text,
            )
        )
    for value in requirements.expected_models:
        checks.append(
            _check(
                f"model:{value}",
                _contains(searchable_text, value),
                bool(searchable_text),
                value,
                searchable_text,
            )
        )
    for value in requirements.required_attributes:
        checks.append(
            _check(
                f"attribute:{value}",
                _contains(searchable_text, value),
                bool(searchable_text),
                value,
                searchable_text,
            )
        )

    selected_options = purchase.get("options")
    option_values = (
        tuple(selected_options.values())
        if isinstance(selected_options, Mapping)
        else _sequence(selected_options)
    )
    normalized_options = tuple(_normalize(value) for value in option_values)
    for value in requirements.required_options:
        expected = _normalize(value)
        checks.append(
            _check(
                f"option:{value}",
                any(
                    expected == actual
                    or expected in actual
                    or actual in expected
                    for actual in normalized_options
                    if actual
                ),
                bool(normalized_options),
                value,
                list(option_values),
            )
        )

    if requirements.price_upper is not None:
        actual_price = _finite_float(purchase.get("price"))
        checks.append(
            _check(
                "price_upper",
                actual_price is not None
                and actual_price <= requirements.price_upper,
                actual_price is not None,
                requirements.price_upper,
                actual_price,
            )
        )

    if not checks:
        return PurchaseVerification(
            verifier_valid=False,
            purchase_correct=False,
            purchase_type=UNSUCCESSFUL_PURCHASE,
            reason="requirements_empty",
        )
    if any(not item["verifiable"] for item in checks):
        return PurchaseVerification(
            verifier_valid=False,
            purchase_correct=False,
            purchase_type=UNSUCCESSFUL_PURCHASE,
            reason="purchase_evidence_missing",
            checks=tuple(checks),
        )
    if not all(item["passed"] for item in checks):
        return PurchaseVerification(
            verifier_valid=True,
            purchase_correct=False,
            purchase_type=UNSUCCESSFUL_PURCHASE,
            reason="requirements_not_satisfied",
            checks=tuple(checks),
        )

    actual_asin = _optional_text(purchase.get("asin"))
    purchase_type = (
        GOLD_PURCHASE
        if requirements.reference_asin is not None
        and actual_asin == requirements.reference_asin
        else VALID_ALTERNATIVE_PURCHASE
    )
    return PurchaseVerification(
        verifier_valid=True,
        purchase_correct=True,
        purchase_type=purchase_type,
        reason=None,
        checks=tuple(checks),
    )


def verify_environment_terminal(
    terminal_result: Mapping[str, Any],
) -> PurchaseVerification:
    """ShopSimulator 终局字典中取 goal/purchase 并执行 验证。"""

    if not isinstance(terminal_result, Mapping):
        raise TypeError("terminal_result must be an object")
    purchase = terminal_result.get("purchase")
    if not purchase:
        return verify_purchase(
            PurchaseRequirementContract(None, None, (), (), None),
            None,
        )
    goal = terminal_result.get("goal")
    if not isinstance(goal, Mapping) or not goal:
        return PurchaseVerification(
            verifier_valid=False,
            purchase_correct=False,
            purchase_type=UNSUCCESSFUL_PURCHASE,
            reason="goal_missing",
        )
    if not isinstance(purchase, Mapping):
        return PurchaseVerification(
            verifier_valid=False,
            purchase_correct=False,
            purchase_type=UNSUCCESSFUL_PURCHASE,
            reason="purchase_invalid",
        )
    return verify_purchase(requirements_from_environment_goal(goal), purchase)


def _goal_option_values(goal: Mapping[str, Any]) -> tuple[str, ...]:
    resolved = goal.get("required_options_by_key")
    values: list[object] = []
    if isinstance(resolved, Mapping):
        for requirement in resolved.values():
            if isinstance(requirement, Mapping):
                values.append(requirement.get("value"))
            else:
                values.append(requirement)
    if not values:
        raw = goal.get("goal_options")
        if isinstance(raw, Mapping):
            values.extend(raw.values())
        else:
            values.extend(_sequence(raw))
    return _text_tuple(values)


def _purchase_searchable_text(purchase: Mapping[str, Any]) -> str:
    values: list[object] = []
    for key in ("brand", "model", "name", "title", "attributes"):
        value = purchase.get(key)
        if isinstance(value, Mapping):
            values.extend(value.keys())
            values.extend(value.values())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return " ".join(str(value) for value in values if value is not None)


def _category_matches(required: str, actual: str | None) -> bool:
    if actual is None:
        return False
    required_parts = [_normalize(item) for item in required.split("›") if _normalize(item)]
    actual_parts = [_normalize(item) for item in actual.split("›") if _normalize(item)]
    return bool(required_parts and actual_parts and required_parts[-1] == actual_parts[-1])


def _contains(actual: str, expected: str) -> bool:
    normalized_expected = _normalize(expected)
    normalized_actual = _normalize(actual)
    if normalized_expected and normalized_expected in normalized_actual:
        return True
    for group in TEXT_EQUIVALENCE_GROUPS:
        if normalized_expected in group:
            return any(alias in normalized_actual for alias in group)
    return False


def _check(
    name: str,
    passed: bool,
    verifiable: bool,
    required: object,
    actual: object,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed) if verifiable else False,
        "verifiable": bool(verifiable),
        "required": deepcopy(required),
        "actual": deepcopy(actual),
    }


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    # 商品规格常用 /、|、空格和各种括号分隔同一组 token；这些符号不表达
    # 不同商品语义，统一移除可以避免展示格式造成的假阴性。
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _text_tuple(value: object) -> tuple[str, ...]:
    result: list[str] = []
    for item in _sequence(value):
        text = _optional_text(item)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _sequence(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return (value,)


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_positive_float(value: object) -> float | None:
    result = _finite_float(value)
    return result if result is not None and result > 0 else None


__all__ = [
    "GOLD_PURCHASE",
    "PurchaseRequirementContract",
    "PurchaseVerification",
    "UNSUCCESSFUL_PURCHASE",
    "TEXT_EQUIVALENCE_GROUPS",
    "VALID_ALTERNATIVE_PURCHASE",
    "PURCHASE_VERIFIER_VERSION",
    "requirements_from_environment_goal",
    "verify_environment_terminal",
    "verify_purchase",
]
