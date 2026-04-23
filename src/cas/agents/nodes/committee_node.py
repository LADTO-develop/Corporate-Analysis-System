"""Run deterministic multi-perspective review and produce a recommendation."""

from __future__ import annotations

from datetime import UTC, datetime

from cas.agents.state import AgentState, AuditEntry, CommitteeReview, Recommendation
from cas.utils.io import read_yaml


def run(state: AgentState) -> dict[str, object]:
    """Run the committee, aggregate reviews, and set the final recommendation."""
    cfg = read_yaml("configs/agent/committee.yaml")
    features = dict(state.get("normalized_features") or {})
    features["qualitative_adjustment"] = float((state.get("news_overlay") or {}).get("adjustment", 0.0))
    overall_score = float(state.get("overall_score", 0.0))

    reviews: list[CommitteeReview] = []
    for spec in cfg["perspectives"]:
        perspective = str(spec["kind"])
        focus_metrics = list(spec.get("focus_metrics", []))
        focus_score = _mean([float(features.get(metric, 0.5)) for metric in focus_metrics])
        blended_score = round((overall_score * 0.6) + (focus_score * 0.4), 4)
        recommendation = _recommendation_from_score(
            blended_score,
            cfg["aggregation"]["recommendation_thresholds"],
        )
        reviews.append(
            CommitteeReview(
                perspective=perspective,
                recommendation=recommendation,
                confidence=round(abs(blended_score - 0.5) * 2, 4),
                rationale=_build_rationale(perspective, focus_metrics, features, blended_score),
            )
        )

    recommendation, confidence = _aggregate(reviews, cfg)
    audit = AuditEntry(
        node="committee",
        timestamp=_now(),
        summary=(
            f"Committee recommendation={recommendation} "
            f"(confidence={confidence:.3f}) from {len(reviews)} reviews"
        ),
        metrics={"final_confidence": confidence, "n_reviews": float(len(reviews))},
    )
    return {
        "committee_reviews": reviews,
        "final_recommendation": recommendation,
        "final_confidence": confidence,
        "audit": [audit],
    }


def _aggregate(
    reviews: list[CommitteeReview],
    cfg: dict[str, object],
) -> tuple[Recommendation, float]:
    if not reviews:
        return "review", 0.0

    perspectives = cfg["perspectives"]  # type: ignore[index]
    aggregation = cfg["aggregation"]  # type: ignore[index]
    weights = {str(spec["kind"]): float(spec.get("weight", 0.25)) for spec in perspectives}
    scores: dict[Recommendation, float] = {"priority": 0.0, "watch": 0.0, "review": 0.0, "defer": 0.0}
    total_weight = 0.0
    for review in reviews:
        weight = weights.get(review.perspective, 0.25) * review.confidence
        scores[review.recommendation] += weight
        total_weight += weight

    if total_weight > 0:
        for key in scores:
            scores[key] /= total_weight

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best, best_score = ordered[0]
    runner_up_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence_gap = float(aggregation["confidence_gap"])
    confidence = max(0.0, min(1.0, best_score - runner_up_score + confidence_gap))
    return best, round(confidence, 4)


def _recommendation_from_score(score: float, thresholds: dict[str, float]) -> Recommendation:
    if score >= float(thresholds["priority"]):
        return "priority"
    if score >= float(thresholds["watch"]):
        return "watch"
    if score >= float(thresholds["review"]):
        return "review"
    return "defer"


def _build_rationale(
    perspective: str,
    focus_metrics: list[str],
    features: dict[str, float],
    blended_score: float,
) -> str:
    preview = ", ".join(
        f"{metric}={features.get(metric, 0.5):.2f}" for metric in focus_metrics[:3]
    )
    return f"{perspective} lens reviewed {preview}; blended score={blended_score:.2f}."


def _mean(values: list[float]) -> float:
    if not values:
        return 0.5
    return round(sum(values) / len(values), 4)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
