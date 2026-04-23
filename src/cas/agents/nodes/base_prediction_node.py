"""Build lens-level assessments and an initial overall score."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState, AuditEntry, BaseAssessment
from cas.utils.io import read_yaml


def run(state: AgentState) -> dict[str, Any]:
    """Create deterministic lens scores from normalized features."""
    cfg = read_yaml("configs/runtime/analysis.yaml")
    features = state.get("normalized_features") or {}
    if not features:
        audit = AuditEntry(
            node="base_prediction",
            timestamp=_now(),
            summary="No normalized features available; skipping base assessment.",
        )
        return {"audit": [audit]}

    lens_scores: dict[str, BaseAssessment] = {}
    for lens_name, weights in cfg["lenses"].items():
        score = _weighted_score(features, weights)
        drivers = sorted(
            ((metric, float(features.get(metric, 0.5))) for metric in weights),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        lens_scores[lens_name] = BaseAssessment(
            lens_name=lens_name,
            score=score,
            summary=_lens_summary(lens_name, score),
            drivers=drivers,
        )

    overall_score = _weighted_score(
        {name: assessment.score for name, assessment in lens_scores.items()},
        cfg["overall_weights"],
    )
    audit = AuditEntry(
        node="base_prediction",
        timestamp=_now(),
        summary=(
            "Built base assessments: "
            + ", ".join(f"{k}={v.score:.3f}" for k, v in lens_scores.items())
            + f" | overall={overall_score:.3f}"
        ),
        metrics={f"score_{k}": v.score for k, v in lens_scores.items()} | {"overall_score": overall_score},
    )
    return {
        "base_assessments": lens_scores,
        "overall_score": overall_score,
        "audit": [audit],
    }


def _weighted_score(values: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(float(weight) for weight in weights.values())
    if total_weight <= 0:
        return 0.0
    total = 0.0
    for key, weight in weights.items():
        total += float(values.get(key, 0.5)) * float(weight)
    return round(total / total_weight, 4)


def _lens_summary(lens_name: str, score: float) -> str:
    if score >= 0.75:
        return f"{lens_name} is a clear strength."
    if score >= 0.55:
        return f"{lens_name} is acceptable with room to improve."
    return f"{lens_name} needs closer review."


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
