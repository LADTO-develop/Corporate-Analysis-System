"""Unit tests for Stage 2 committee helpers."""

from __future__ import annotations

from cas.agents.nodes.committee_node import (
    _financial_model_agent,
    _recommendation_from_score,
)
from cas.agents.state import AgentState


def test_recommendation_thresholds() -> None:
    thresholds = {"priority": 0.75, "watch": 0.60, "review": 0.45}
    assert _recommendation_from_score(0.82, thresholds) == "priority"
    assert _recommendation_from_score(0.65, thresholds) == "watch"
    assert _recommendation_from_score(0.50, thresholds) == "review"
    assert _recommendation_from_score(0.30, thresholds) == "defer"


def test_financial_model_agent_generates_quant_summary() -> None:
    state: AgentState = {
        "company_id": "250",
        "company_name": "삼천당제약(주)",
        "market": "KOSDAQ",
        "source_feature_row": {
            "market": "KOSDAQ",
            "industry_macro_category": "manufacturing",
            "firm_size_group": "mid_sized",
            "current_ratio": 2.82,
            "cash_ratio": 0.69,
            "capital_impairment_ratio": -0.12,
            "gross_profit": 96966293.0,
            "dividend_payer": 1,
        },
        "peer_comparison_rows": [
            {
                "feature": "gross_profit",
                "industry_median": 55000000.0,
                "market_median": 30000000.0,
                "industry_percentile": 83.4,
            }
        ],
    }
    xgb_result = {
        "probability_speculative": 0.0474,
        "prediction_label": "투자적격",
        "top_drivers": [
            ("capital_impairment_ratio", -0.5535),
            ("gross_profit", -0.4322),
            ("dividend_payer", -0.4259),
        ],
    }

    agent = _financial_model_agent(state, xgb_result)

    assert agent.role == "financial_model"
    assert "투자적격" in agent.summary
    assert "위험확률" in agent.summary
    assert len(agent.findings) == 3
    assert any("핵심 위험 요인" in item for item in agent.findings)
    assert any("완화 요인" in item for item in agent.findings)
    assert "산업 중앙값" in " ".join(agent.findings)
