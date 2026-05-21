"""Tests for Stage 2 input bundle normalization."""

from __future__ import annotations

from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.state import AgentState


def test_stage2_input_bundle_normalizes_state_for_agents() -> None:
    state: AgentState = {
        "company_id": "KOSPI-005930-2024",
        "company_name": "삼성전자",
        "market": "KOSPI",
        "analysis_year": 2025,
        "model_view": {"y_proba": 0.21},
        "xgboost_result": {"prediction_label": "투자적격"},
        "source_feature_row": {"market": "KOSPI", "current_ratio": 2.1},
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BBB-",
        },
        "peer_comparison_rows": [
            {"feature": "current_ratio", "industry_median": 1.5},
            {"feature": None, "industry_median": 0.0},
        ],
        "news_cache_snapshot": {"status": "not_implemented"},
    }

    bundle = build_stage2_input_bundle(state)

    assert bundle.company_name == "삼성전자"
    assert bundle.prediction_label == "투자적격"
    assert bundle.probability_speculative == 0.21
    assert bundle.news_status == "not_implemented"
    assert bundle.prior_rating_reference["prior_credit_rating"] == "BBB-"
    assert set(bundle.peer_rows_by_feature) == {"current_ratio"}


def test_stage2_input_bundle_exports_prompt_payload() -> None:
    state: AgentState = {
        "company_id": "KOSDAQ-000250-2023",
        "source_feature_row": {"company_name": "삼천당제약(주)", "market": "KOSDAQ"},
    }

    payload = build_stage2_input_bundle(state).to_prompt_payload()

    assert payload["company"]["company_id"] == "KOSDAQ-000250-2023"
    assert payload["company"]["company_name"] == "삼천당제약(주)"
    assert payload["company"]["market"] == "KOSDAQ"
    assert "model_view" in payload
    assert "prior_rating_reference" in payload
