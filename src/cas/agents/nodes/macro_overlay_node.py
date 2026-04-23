"""Apply market-context adjustments to the current score."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from cas.agents.state import AgentState, AuditEntry, OverlayAssessment
from cas.utils.io import read_yaml


def run(state: AgentState) -> dict[str, Any]:
    """Apply bounded score adjustments from market context."""
    cfg = read_yaml("configs/runtime/analysis.yaml")
    profile = state.get("company_profile") or {}
    market_context = profile.get("market_context") or {}
    mapping = cfg["overlays"]["market"]
    cap = float(cfg["score_caps"]["market_adjustment"])

    adjustment = 0.0
    rationale: list[str] = []
    for key in ("demand_outlook", "cost_pressure", "regulatory_pressure"):
        level = str(market_context.get(key, "neutral")).lower()
        delta = float(mapping.get(key, {}).get(level, 0.0))
        adjustment += delta
        if delta != 0:
            rationale.append(f"{key}={level} ({delta:+.2f})")

    adjustment = max(-cap, min(cap, adjustment))
    current = float(state.get("overall_score", 0.0))
    overlay = OverlayAssessment(
        label="market",
        adjustment=adjustment,
        rationale="; ".join(rationale) or "No material market adjustment.",
        signals=market_context,
    )

    audit = AuditEntry(
        node="market_overlay",
        timestamp=_now(),
        summary=f"Market adjustment={adjustment:+.3f} -> score {current + adjustment:.3f}",
        metrics={"market_adjustment": adjustment},
    )
    return {
        "market_overlay": overlay.model_dump(),
        "overall_score": round(max(0.0, min(1.0, current + adjustment)), 4),
        "audit": [audit],
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
