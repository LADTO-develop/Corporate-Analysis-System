"""Run role-fixed Agno-style agents and synthesize the service response."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from cas.agents.state import (
    AgentOutput,
    AgentState,
    AuditEntry,
    CommitteeReview,
    Recommendation,
)


def run(state: AgentState) -> dict[str, Any]:
    """Run deterministic role agents over model, news, and rule-engine outputs."""
    xgb = dict(state.get("xgboost_result") or {})
    rule = dict(state.get("rule_result") or {})

    recommendation = cast(
        Recommendation,
        rule.get("recommendation") or state.get("final_recommendation") or "review",
    )
    confidence = round(float(rule.get("confidence", state.get("final_confidence", 0.0)) or 0.0), 4)
    agents = [
        _news_summary_agent(),
        _model_interpretation_agent(xgb),
        _risk_review_agent(rule),
        _synthesis_format_agent(rule, recommendation, confidence),
    ]
    reviews = [
        CommitteeReview(
            perspective=agent.role,
            recommendation=recommendation,
            confidence=agent.confidence,
            rationale=agent.summary,
        )
        for agent in agents
    ]
    agent_summary = {
        "final_recommendation": recommendation,
        "final_confidence": confidence,
        "synthesis": agents[-1].summary,
        "agents": {
            agent.role: {
                "summary": agent.summary,
                "findings": agent.findings,
                "confidence": agent.confidence,
            }
            for agent in agents
        },
    }

    audit = AuditEntry(
        node="agno_agents",
        timestamp=_now(),
        summary=(
            "Agno role-fixed agents completed: "
            f"{', '.join(agent.role for agent in agents)}"
        ),
        metrics={"n_agents": float(len(agents)), "final_confidence": confidence},
    )
    return {
        "agent_outputs": agents,
        "committee_reviews": reviews,
        "agent_summary": agent_summary,
        "final_recommendation": recommendation,
        "final_confidence": confidence,
        "audit": [audit],
    }


def _news_summary_agent() -> AgentOutput:
    summary = "News/crawling analysis is reserved for a future integration."
    return AgentOutput(
        role="news_summary",
        summary=summary,
        findings=["No crawling implementation is active in this node."],
        confidence=0.0,
    )


def _model_interpretation_agent(xgb: dict[str, Any]) -> AgentOutput:
    probability = float(xgb.get("probability_speculative", 0.0) or 0.0)
    drivers = [f"{name}={value:.3f}" for name, value in _driver_pairs(xgb)]
    summary = (
        f"Model registry result is {xgb.get('prediction_label', 'unknown')} "
        f"with speculative probability={probability:.3f}."
    )
    return AgentOutput(
        role="model_interpretation",
        summary=summary,
        findings=drivers or ["No model drivers were available."],
        confidence=0.8 if xgb else 0.4,
    )


def _risk_review_agent(rule: dict[str, Any]) -> AgentOutput:
    reasons = [str(reason) for reason in rule.get("reasons", [])]
    flags = [str(flag) for flag in rule.get("blocking_flags", [])]
    risk_band = str(rule.get("risk_band", "insufficient_data"))
    summary = f"Rule engine assigned risk_band={risk_band} with {len(flags)} blocking flag(s)."
    return AgentOutput(
        role="risk_review",
        summary=summary,
        findings=[*reasons[:3], *(f"flag:{flag}" for flag in flags[:2])]
        or ["No rule-engine reasons were available."],
        confidence=float(rule.get("confidence", 0.5) or 0.5),
    )


def _synthesis_format_agent(
    rule: dict[str, Any],
    recommendation: Recommendation,
    confidence: float,
) -> AgentOutput:
    label = str(rule.get("label", "unclassified"))
    summary = (
        f"Final service response is recommendation={recommendation}, "
        f"confidence={confidence:.3f}, label={label}."
    )
    return AgentOutput(
        role="synthesis_format",
        summary=summary,
        findings=[
            "Response will be emitted through the strict dashboard JSON schema.",
            "Model result and explanatory agent text remain separated.",
        ],
        confidence=max(0.5, confidence),
    )


def _driver_pairs(xgb: dict[str, Any]) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for item in xgb.get("top_drivers", []) or []:
        if isinstance(item, dict):
            name = str(item.get("name", item.get("feature", "")))
            value = float(item.get("value", item.get("score", 0.0)) or 0.0)
        else:
            name = str(item[0])
            value = float(item[1])
        if name:
            pairs.append((name, value))
    return pairs


def _recommendation_from_score(score: float, thresholds: dict[str, float]) -> Recommendation:
    """Map a numeric suitability score to the legacy recommendation buckets."""
    if score >= float(thresholds["priority"]):
        return "priority"
    if score >= float(thresholds["watch"]):
        return "watch"
    if score >= float(thresholds["review"]):
        return "review"
    return "defer"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
