"""Apply deterministic policy rules on top of realtime model output."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState, AuditEntry, Recommendation, RiskBand, RuleResult
from cas.utils.io import read_yaml


def run(state: AgentState) -> dict[str, Any]:
    """Convert model output plus cached context into a service risk band."""
    cfg = read_yaml("configs/runtime/analysis.yaml")
    rule_cfg = dict(cfg.get("rule_engine", {}))
    xgb = dict(state.get("xgboost_result") or {})
    features = dict(state.get("normalized_features") or {})
    profile = dict(state.get("company_profile") or {})
    qualitative = dict(profile.get("qualitative") or {})

    if not xgb:
        result = RuleResult(
            risk_band="insufficient_data",
            label="insufficient_data",
            recommendation="review",
            confidence=0.0,
            reasons=["Realtime model output is unavailable."],
            blocking_flags=["missing_model_result"],
        )
        return {
            "rule_result": result.model_dump(),
            "final_recommendation": result.recommendation,
            "final_confidence": result.confidence,
            "audit": [
                AuditEntry(
                    node="rule_engine",
                    timestamp=_now(),
                    summary="Rule engine deferred because model output is unavailable.",
                )
            ],
        }

    probability = float(xgb.get("probability_speculative", 0.5))
    thresholds = dict(rule_cfg.get("risk_band_thresholds", {}))
    watch_threshold = float(thresholds.get("watch", 0.4))
    high_risk_threshold = float(thresholds.get("high_risk", 0.65))
    band = _band_from_probability(
        probability,
        watch_threshold=watch_threshold,
        high_risk_threshold=high_risk_threshold,
    )

    reasons = [f"model_probability_speculative={probability:.3f}"]
    blocking_flags: list[str] = []
    controversy_level = str(qualitative.get("controversy_level", "low")).lower()

    override_cfg = dict(rule_cfg.get("overrides", {}))
    if controversy_level == "high":
        band = "high_risk"
        blocking_flags.append("high_controversy")
        reasons.append("high controversy level overrides the model band")

    weak_liquidity = float(override_cfg.get("weak_liquidity_threshold", 0.25))
    weak_cash = float(override_cfg.get("weak_cash_generation_threshold", 0.25))
    if float(features.get("liquidity_score", 0.5)) < weak_liquidity:
        band = _max_band(band, "watch")
        reasons.append("liquidity score is below policy threshold")
    if float(features.get("cash_generation_score", 0.5)) < weak_cash:
        band = _max_band(band, "watch")
        reasons.append("cash generation score is below policy threshold")

    recommendation = _recommendation_for_band(band)
    confidence = _confidence_from_probability(
        probability,
        watch_threshold=watch_threshold,
        high_risk_threshold=high_risk_threshold,
    )
    result = RuleResult(
        risk_band=band,
        label=f"{xgb.get('prediction_label', 'unknown')}:{band}",
        recommendation=recommendation,
        confidence=confidence,
        reasons=reasons,
        blocking_flags=blocking_flags,
    )
    audit = AuditEntry(
        node="rule_engine",
        timestamp=_now(),
        summary=(
            f"Rule engine assigned risk_band={band}, "
            f"recommendation={recommendation}, confidence={confidence:.3f}"
        ),
        metrics={"rule_confidence": confidence, "probability_speculative": probability},
    )
    return {
        "rule_result": result.model_dump(),
        "final_recommendation": recommendation,
        "final_confidence": confidence,
        "audit": [audit],
    }


def _band_from_probability(
    probability: float,
    *,
    watch_threshold: float,
    high_risk_threshold: float,
) -> RiskBand:
    if probability >= high_risk_threshold:
        return "high_risk"
    if probability >= watch_threshold:
        return "watch"
    return "stable"


def _max_band(current: RiskBand, candidate: RiskBand) -> RiskBand:
    order = {
        "stable": 0,
        "watch": 1,
        "high_risk": 2,
        "insufficient_data": 3,
    }
    return candidate if order[candidate] > order[current] else current


def _recommendation_for_band(band: RiskBand) -> Recommendation:
    if band == "stable":
        return "priority"
    if band == "watch":
        return "watch"
    if band == "insufficient_data":
        return "review"
    return "defer"


def _confidence_from_probability(
    probability: float,
    *,
    watch_threshold: float,
    high_risk_threshold: float,
) -> float:
    anchors = [0.0, watch_threshold, high_risk_threshold, 1.0]
    nearest_boundary = min(abs(probability - anchor) for anchor in anchors)
    return round(max(0.35, min(0.95, 0.45 + nearest_boundary)), 4)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
