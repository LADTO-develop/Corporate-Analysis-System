"""Run the realtime XGBoost inference facade."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState, AuditEntry, BaseAssessment, ModelResult, RiskBand
from cas.utils.io import read_yaml


def run(state: AgentState) -> dict[str, Any]:
    """Create deterministic model output shaped like a realtime XGBoost result."""
    cfg = read_yaml("configs/runtime/analysis.yaml")
    features = state.get("normalized_features") or {}
    if not features:
        audit = AuditEntry(
            node="xgboost_inference",
            timestamp=_now(),
            summary="No feature-store snapshot available; skipping realtime inference.",
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
    model_registry = dict(cfg.get("model_registry", {}))
    threshold = float(model_registry.get("threshold", 0.5))
    watch_threshold = float(model_registry.get("watch_threshold", 0.4))
    high_risk_threshold = float(model_registry.get("high_risk_threshold", 0.65))
    probability_speculative = round(1.0 - overall_score, 4)
    xgboost_result = ModelResult(
        model_name=str(model_registry.get("active_model", "xgboost_realtime")),
        model_version=str(model_registry.get("model_version", "local-deterministic")),
        probability_speculative=probability_speculative,
        prediction_label=(
            "speculative_grade" if probability_speculative >= threshold else "investment_grade"
        ),
        risk_band=_risk_band(
            probability_speculative,
            watch_threshold=watch_threshold,
            high_risk_threshold=high_risk_threshold,
        ),
        threshold=threshold,
        top_drivers=_top_risk_drivers(features),
    )
    audit = AuditEntry(
        node="xgboost_inference",
        timestamp=_now(),
        summary=(
            "Realtime XGBoost inference completed: "
            f"probability_speculative={probability_speculative:.3f}, "
            f"risk_band={xgboost_result.risk_band}; lenses="
            + ", ".join(f"{k}={v.score:.3f}" for k, v in lens_scores.items())
            + f" | overall={overall_score:.3f}"
        ),
        metrics={f"score_{k}": v.score for k, v in lens_scores.items()}
        | {
            "overall_score": overall_score,
            "probability_speculative": probability_speculative,
        },
    )
    return {
        "base_assessments": lens_scores,
        "overall_score": overall_score,
        "xgboost_result": xgboost_result.model_dump(),
        "model_registry_ref": {
            "registry_name": model_registry.get("registry_name", "local_model_registry"),
            "active_model": xgboost_result.model_name,
            "model_version": xgboost_result.model_version,
            "threshold": threshold,
            "watch_threshold": watch_threshold,
            "high_risk_threshold": high_risk_threshold,
        },
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


def _risk_band(
    probability_speculative: float,
    *,
    watch_threshold: float,
    high_risk_threshold: float,
) -> RiskBand:
    if probability_speculative >= high_risk_threshold:
        return "high_risk"
    if probability_speculative >= watch_threshold:
        return "watch"
    return "stable"


def _top_risk_drivers(features: dict[str, float]) -> list[tuple[str, float]]:
    driver_scores = [
        (name, round(1.0 - float(value), 4))
        for name, value in features.items()
        if name not in {"controversy_penalty"}
    ]
    driver_scores.append(
        ("controversy_penalty", round(1.0 - float(features.get("controversy_penalty", 0.5)), 4))
    )
    return sorted(driver_scores, key=lambda item: item[1], reverse=True)[:5]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
