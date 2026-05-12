"""Unit tests for Stage 2 committee helpers."""

from __future__ import annotations

from cas.agents.nodes.committee_node import (
    _debt_liquidity_agent,
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


def test_debt_liquidity_agent_flags_liquidity_mismatch() -> None:
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.11,
            "debt_ratio": 1.40,
            "short_term_borrowings_share": 0.71,
            "cashflow_coverage_ratio": 0.80,
            "interest_coverage_ratio": 2.10,
            "ocf_to_total_liabilities": 0.04,
            "ocf_to_total_borrowings": 0.09,
            "is_2y_consecutive_ocf_deficit": 0,
            "icr_under_1": 0,
        },
        "xgboost_result": {"prediction_label": "투자적격"},
    }

    agent = _debt_liquidity_agent(state)

    assert agent.role == "debt_liquidity"
    assert "투자적격" in agent.summary
    assert "추가 경계" in agent.summary
    assert any("유동비율이 1.0 미만" in item for item in agent.findings)
    assert any("단기차입금 비중이 높아 차환 리스크" in item for item in agent.findings)


def test_debt_liquidity_agent_preserves_downside_but_notes_support() -> None:
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 2.10,
            "cash_ratio": 0.62,
            "debt_ratio": 2.80,
            "short_term_borrowings_share": 0.22,
            "cashflow_coverage_ratio": 5.40,
            "interest_coverage_ratio": 4.10,
            "ocf_to_total_liabilities": 0.14,
            "ocf_to_total_borrowings": 0.27,
            "is_2y_consecutive_ocf_deficit": 0,
            "icr_under_1": 0,
        },
        "xgboost_result": {"prediction_label": "부적격"},
    }

    agent = _debt_liquidity_agent(state)

    assert agent.role == "debt_liquidity"
    assert "부적격" in agent.summary
    assert "완화 신호" in agent.summary
    assert any("현금흐름 커버리지가 5배 이상" in item for item in agent.findings)
    assert any("영업현금흐름이 총부채 대비 0.1 이상" in item for item in agent.findings)
