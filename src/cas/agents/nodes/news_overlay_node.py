"""Apply qualitative and event-driven adjustments to the current score."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState, AuditEntry, OverlayAssessment
from cas.utils.io import read_yaml


def run(state: AgentState) -> dict[str, Any]:
    """Apply bounded adjustments from qualitative context."""
    cfg = read_yaml("configs/runtime/analysis.yaml")
    profile = state.get("company_profile") or {}
    qualitative = profile.get("qualitative") or {}
    news_context = profile.get("news_context") or {}
    mapping = cfg["overlays"]["qualitative"]
    cap = float(cfg["score_caps"]["qualitative_adjustment"])

    controversy_level = str(qualitative.get("controversy_level", "low")).lower()
    execution_risk = str(news_context.get("execution_risk", "moderate")).lower()
    sentiment_score = float(news_context.get("sentiment_score", 0.0) or 0.0)

    adjustment = 0.0
    adjustment += float(mapping["controversy_level"].get(controversy_level, 0.0))
    adjustment += float(mapping["execution_risk"].get(execution_risk, 0.0))
    adjustment += sentiment_score * 0.03

    rationale: list[str] = []
    if adjustment != 0:
        rationale.append(
            f"controversy={controversy_level}, execution_risk={execution_risk}, sentiment={sentiment_score:+.2f}"
        )
    notable_events = news_context.get("notable_events") or []
    if notable_events:
        rationale.append(f"events={len(notable_events)} tracked")

    adjustment = max(-cap, min(cap, adjustment))
    current = float(state.get("overall_score", 0.0))
    overlay = OverlayAssessment(
        label="qualitative",
        adjustment=adjustment,
        rationale=" / ".join(rationale) or "No material qualitative adjustment.",
        signals={"qualitative": qualitative, "news_context": news_context},
    )

    audit = AuditEntry(
        node="news_overlay",
        timestamp=_now(),
        summary=f"Qualitative adjustment={adjustment:+.3f} -> score {current + adjustment:.3f}",
        metrics={"qualitative_adjustment": adjustment},
    )
    return {
        "news_overlay": overlay.model_dump(),
        "overall_score": round(max(0.0, min(1.0, current + adjustment)), 4),
        "audit": [audit],
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
