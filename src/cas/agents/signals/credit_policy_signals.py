"""Deterministic credit-policy signals for Stage 2 LLM grounding."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["risk_increasing", "risk_mitigating", "neutral"]
Severity = Literal["low", "moderate", "high", "critical"]
Operator = Literal[">", ">=", "<", "<=", "==", "truthy"]

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_POLICY_PATH = ROOT / "configs" / "agent" / "credit_signal_policy.json"


class CreditPolicyCriterion(BaseModel):
    """One deterministic credit-signal rule loaded from config."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    feature: str
    operator: Operator
    threshold: float | str | None = None
    industry_percentile_min: float | None = None
    industry_percentile_max: float | None = None
    requires_truthy_features: list[str] = Field(default_factory=list)
    direction: Direction
    severity: Severity
    score_delta: float = Field(ge=-1.0, le=1.0)
    reason_kr: str
    basis: list[str] = Field(default_factory=list)


class CreditPolicyConfig(BaseModel):
    """Versioned deterministic credit policy."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str
    description: str = ""
    label_override_allowed: bool = False
    criteria: list[CreditPolicyCriterion]


class CreditPolicySignal(BaseModel):
    """One triggered deterministic signal for Stage 2 agent prompts."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    feature: str
    value: float | str | bool | None
    industry_percentile: float | None = None
    direction: Direction
    severity: Severity
    score_delta: float
    reason_kr: str
    basis: list[str]


class CreditPolicySnapshot(BaseModel):
    """Compact policy snapshot passed to Stage 2 agents."""

    model_config = ConfigDict(extra="forbid")

    policy_version: str
    label_override_allowed: bool = False
    signals: list[CreditPolicySignal] = Field(default_factory=list)
    net_policy_delta: float = 0.0
    risk_signal_count: int = 0
    mitigating_signal_count: int = 0
    critical_signal_count: int = 0


def load_credit_policy(path: Path = DEFAULT_POLICY_PATH) -> CreditPolicyConfig:
    """Load the versioned deterministic credit-signal policy."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CreditPolicyConfig.model_validate(payload)


def evaluate_credit_policy(
    *,
    source_feature_row: dict[str, Any],
    peer_comparison_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    policy: CreditPolicyConfig | None = None,
) -> CreditPolicySnapshot:
    """Evaluate deterministic credit signals without changing Stage 1 outputs."""
    active_policy = policy or load_credit_policy()
    peer_by_feature = _peer_rows_by_feature(peer_comparison_rows)

    signals: list[CreditPolicySignal] = []
    for criterion in active_policy.criteria:
        value = source_feature_row.get(criterion.feature)
        percentile = _industry_percentile(peer_by_feature.get(criterion.feature))

        if not _criterion_matches(
            criterion=criterion,
            value=value,
            source_feature_row=source_feature_row,
            industry_percentile=percentile,
        ):
            continue

        signals.append(
            CreditPolicySignal(
                criterion_id=criterion.criterion_id,
                feature=criterion.feature,
                value=_plain_value(value),
                industry_percentile=percentile,
                direction=criterion.direction,
                severity=criterion.severity,
                score_delta=criterion.score_delta,
                reason_kr=criterion.reason_kr,
                basis=criterion.basis,
            )
        )

    return CreditPolicySnapshot(
        policy_version=active_policy.policy_version,
        label_override_allowed=active_policy.label_override_allowed,
        signals=signals,
        net_policy_delta=round(sum(signal.score_delta for signal in signals), 4),
        risk_signal_count=sum(1 for signal in signals if signal.direction == "risk_increasing"),
        mitigating_signal_count=sum(
            1 for signal in signals if signal.direction == "risk_mitigating"
        ),
        critical_signal_count=sum(1 for signal in signals if signal.severity == "critical"),
    )


def _criterion_matches(
    *,
    criterion: CreditPolicyCriterion,
    value: object,
    source_feature_row: dict[str, Any],
    industry_percentile: float | None,
) -> bool:
    if not _passes_required_flags(criterion, source_feature_row):
        return False
    if not _passes_percentile_gate(criterion, industry_percentile):
        return False
    return _operator_matches(
        value=value, operator=criterion.operator, threshold=criterion.threshold
    )


def _passes_required_flags(
    criterion: CreditPolicyCriterion,
    source_feature_row: dict[str, Any],
) -> bool:
    return all(
        _truthy(source_feature_row.get(feature)) for feature in criterion.requires_truthy_features
    )


def _passes_percentile_gate(
    criterion: CreditPolicyCriterion,
    industry_percentile: float | None,
) -> bool:
    if criterion.industry_percentile_min is not None and (
        industry_percentile is None or industry_percentile < criterion.industry_percentile_min
    ):
        return False

    return not (
        criterion.industry_percentile_max is not None
        and (industry_percentile is None or industry_percentile > criterion.industry_percentile_max)
    )


def _operator_matches(*, value: object, operator: Operator, threshold: object) -> bool:
    if operator == "truthy":
        return _truthy(value)

    value_number = _optional_float(value)
    threshold_number = _optional_float(threshold)
    if value_number is None or threshold_number is None:
        return False

    if operator == ">":
        return value_number > threshold_number
    if operator == ">=":
        return value_number >= threshold_number
    if operator == "<":
        return value_number < threshold_number
    if operator == "<=":
        return value_number <= threshold_number
    if operator == "==":
        return value_number == threshold_number
    return False


def _peer_rows_by_feature(
    peer_comparison_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("feature")): dict(row)
        for row in peer_comparison_rows
        if isinstance(row.get("feature"), str)
    }


def _industry_percentile(peer_row: dict[str, Any] | None) -> float | None:
    if not peer_row:
        return None
    return _optional_float(peer_row.get("industry_percentile"))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        number = _optional_float(value)
        return number is not None and bool(number)
    text = str(value or "").strip().lower()
    return text in {"1", "1.0", "true", "yes", "y", "on"}


def _plain_value(value: object) -> float | str | bool | None:
    if value is None:
        return None
    if isinstance(value, bool | str):
        return value
    number = _optional_float(value)
    if number is not None:
        return number
    return str(value)


__all__ = [
    "CreditPolicyConfig",
    "CreditPolicyCriterion",
    "CreditPolicySignal",
    "CreditPolicySnapshot",
    "evaluate_credit_policy",
    "load_credit_policy",
]
