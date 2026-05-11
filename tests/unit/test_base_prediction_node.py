"""Unit tests for Stage 1 base prediction behavior."""

from __future__ import annotations

from pathlib import Path


def test_falls_back_when_model_artifact_is_missing(monkeypatch) -> None:
    from cas.agents.nodes import base_prediction_node

    base_prediction_node._load_model_bundle.cache_clear()
    monkeypatch.setattr(
        base_prediction_node,
        "_MODEL_ARTIFACT_PATH",
        Path("data/outputs/modeling/feature_43_xgboost/missing_model.pkl"),
    )

    state = {
        "company_id": "250",
        "market": "KOSDAQ",
        "analysis_year": 2024,
        "model_features": {
            "cash_ratio": 0.21,
            "interest_coverage_ratio": 3.4,
            "debt_to_assets": 0.42,
        },
        "normalized_features": {
            "profitability_score": 0.62,
            "leverage_score": 0.44,
            "liquidity_score": 0.57,
            "cashflow_score": 0.59,
            "market_signal_score": 0.52,
            "governance_score": 0.55,
        },
    }

    result = base_prediction_node.run(state)

    assert result["xgboost_result"]["prediction_label"] in {"투자적격", "부적격"}
    assert result["model_view"]["risk_band"] in {"stable", "watch", "high_risk"}
    assert "Saved XGBoost artifact was not found" in result["audit"][0].summary
