"""Normalize raw profile inputs into reusable feature scores."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState, AuditEntry
from cas.utils.io import read_yaml


def run(state: AgentState) -> dict[str, Any]:
    """Compute normalized feature scores for downstream lenses."""
    cfg = read_yaml("configs/runtime/analysis.yaml")
    financials = state.get("raw_financials") or {}
    profile = state.get("company_profile") or {}
    qualitative = profile.get("qualitative") or {}
    market_context = profile.get("market_context") or {}

    if not financials:
        audit = AuditEntry(
            node="feature",
            timestamp=_now(),
            summary="No financial inputs in state; skipping feature computation.",
        )
        return {"audit": [audit]}

    ranges = cfg["feature_ranges"]
    features = {
        "revenue_growth_score": _score(
            financials.get("revenue_growth_pct"), ranges["revenue_growth_pct"]
        ),
        "profitability_score": _score(
            financials.get("operating_margin_pct"), ranges["operating_margin_pct"]
        ),
        "leverage_health_score": _score(financials.get("debt_to_equity"), ranges["debt_to_equity"]),
        "liquidity_score": _score(financials.get("current_ratio"), ranges["current_ratio"]),
        "cash_generation_score": _score(
            financials.get("free_cash_flow_margin_pct"), ranges["free_cash_flow_margin_pct"]
        ),
        "interest_coverage_score": _score(financials.get("interest_coverage"), ranges["interest_coverage"]),
        "governance_score": _score(qualitative.get("governance_score"), ranges["governance_score"]),
        "product_momentum_score": _score(
            qualitative.get("product_momentum_score"), ranges["product_momentum_score"]
        ),
        "concentration_health_score": _score(
            qualitative.get("customer_concentration_pct", 0.0),
            ranges["customer_concentration_pct"],
        ),
        "industry_position_score": _score(
            market_context.get("industry_growth_score", 0.5),
            ranges["industry_growth_score"],
        ),
        "controversy_penalty": _controversy_penalty(str(qualitative.get("controversy_level", "low"))),
    }

    audit = AuditEntry(
        node="feature",
        timestamp=_now(),
        summary=f"Computed {len(features)} normalized features",
        metrics={"n_features": float(len(features))},
    )
    return {"normalized_features": features, "audit": [audit]}


def _score(value: Any, spec: dict[str, Any]) -> float:
    if value is None:
        return 0.5
    raw = float(value)
    lower = float(spec["min"])
    upper = float(spec["max"])
    if upper <= lower:
        return 0.5
    clipped = min(max(raw, lower), upper)
    ratio = (clipped - lower) / (upper - lower)
    if not bool(spec.get("higher_is_better", True)):
        ratio = 1.0 - ratio
    return round(ratio, 4)


def _controversy_penalty(level: str) -> float:
    return {"low": 0.9, "moderate": 0.45, "high": 0.1}.get(level.lower(), 0.45)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
