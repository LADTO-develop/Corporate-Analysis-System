"""Unit tests for Stage 1 base prediction behavior."""

from __future__ import annotations

from pathlib import Path


def test_applies_platt_probability_calibration() -> None:
    from cas.agents.nodes import base_prediction_node

    calibrated = base_prediction_node._apply_probability_calibration(
        0.8,
        {
            "method": "platt_sigmoid",
            "coef": 0.75,
            "intercept": -0.5,
            "clip_epsilon": 1e-6,
        },
    )

    assert round(calibrated, 4) == 0.6317


def test_native_missing_model_frame_keeps_nan() -> None:
    from cas.agents.nodes import base_prediction_node

    frame = base_prediction_node._build_model_frame(
        {"cash_ratio": 0.2, "market_to_book": float("nan")},
        ["cash_ratio", "market_to_book", "current_ratio"],
        {"market_to_book": 1.0, "current_ratio": 2.0},
        missing_value_strategy="xgboost_native_missing",
    )

    assert frame.loc[0, "cash_ratio"] == 0.2
    assert frame.loc[0, "market_to_book"] != frame.loc[0, "market_to_book"]
    assert frame.loc[0, "current_ratio"] != frame.loc[0, "current_ratio"]


def test_median_imputation_model_frame_uses_fill_values() -> None:
    from cas.agents.nodes import base_prediction_node

    frame = base_prediction_node._build_model_frame(
        {"cash_ratio": 0.2, "market_to_book": float("nan")},
        ["cash_ratio", "market_to_book", "current_ratio"],
        {"market_to_book": 1.0, "current_ratio": 2.0},
        missing_value_strategy="median_imputation",
    )

    assert frame.loc[0, "cash_ratio"] == 0.2
    assert frame.loc[0, "market_to_book"] == 1.0
    assert frame.loc[0, "current_ratio"] == 2.0


def test_falls_back_when_model_artifact_is_missing(monkeypatch) -> None:
    from cas.agents.nodes import base_prediction_node

    base_prediction_node._load_model_bundle.cache_clear()
    monkeypatch.setattr(
        base_prediction_node,
        "_MODEL_ARTIFACT_PATH",
        Path("data/outputs/modeling/feature_43_xgboost/missing_model.json"),
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
