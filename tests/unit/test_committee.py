"""Unit tests for Stage 2 committee helpers."""

from __future__ import annotations

from cas.agents.nodes.committee_node import (
    _evidence_audit_agent,
    _quant_credit_agent,
    _recommendation_from_score,
    run,
)
from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.state import AgentState


def test_recommendation_thresholds() -> None:
    thresholds = {"priority": 0.75, "watch": 0.60, "review": 0.45}
    assert _recommendation_from_score(0.82, thresholds) == "priority"
    assert _recommendation_from_score(0.65, thresholds) == "watch"
    assert _recommendation_from_score(0.50, thresholds) == "review"
    assert _recommendation_from_score(0.30, thresholds) == "defer"


def test_quant_credit_agent_generates_quant_summary() -> None:
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

    state["xgboost_result"] = xgb_result
    structured_output = _quant_credit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.role == "quant_credit"
    assert structured_output.key_risk_factors or structured_output.mitigating_factors
    assert agent.role == "quant_credit"
    assert "투자적격" in agent.summary
    assert "위험확률" in agent.summary
    assert len(agent.findings) == 3
    assert any("핵심 위험 요인" in item for item in agent.findings)
    assert any("완화 요인" in item for item in agent.findings)
    assert "산업 중앙값" in " ".join(agent.findings)


def test_evidence_audit_agent_flags_liquidity_mismatch() -> None:
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

    structured_output = _evidence_audit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.role == "evidence_audit"
    assert structured_output.debt_liquidity_cross_check
    assert agent.role == "evidence_audit"
    assert "투자적격" in agent.summary
    assert "추가 경계" in agent.summary
    assert any("유동비율이 1.0 미만" in item for item in agent.findings)
    assert any("단기차입금 비중이 높아 차환 리스크" in item for item in agent.findings)


def test_evidence_audit_agent_preserves_downside_but_notes_support() -> None:
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

    structured_output = _evidence_audit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.role == "evidence_audit"
    assert structured_output.debt_liquidity_cross_check
    assert agent.role == "evidence_audit"
    assert "부적격" in agent.summary
    assert "완화 신호" in agent.summary
    assert any("현금흐름 커버리지가 5배 이상" in item for item in agent.findings)
    assert any("영업현금흐름이 총부채 대비 0.1 이상" in item for item in agent.findings)


def test_evidence_audit_agent_scores_direct_external_risk_evidence() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {
            "stock_code": "000250",
            "current_ratio": 1.8,
            "cash_ratio": 0.4,
            "short_term_borrowings_share": 0.25,
            "cashflow_coverage_ratio": 4.2,
            "interest_coverage_ratio": 3.4,
        },
        "xgboost_result": {"prediction_label": "투자적격", "probability_speculative": 0.31},
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "삼천당제약(주) 횡령 혐의 발생",
                    "summary": "삼천당제약(주) 공시: 횡령 혐의 발생",
                    "reliability": "high",
                    "company_match": True,
                    "critical_terms": ["횡령"],
                    "critical_context_confirmed": True,
                    "veto_candidate": True,
                    "evidence_score": 0.91,
                    "evidence_quality": "high",
                }
            ],
            "direct_match_count": 1,
            "verified_item_count": 1,
            "veto_candidate_count": 1,
            "high_confidence_critical_count": 1,
            "critical_terms": ["횡령"],
            "has_critical_risk": True,
        },
    }

    structured_output = _evidence_audit_agent(build_stage2_input_bundle(state))
    agent = structured_output.to_agent_output()

    assert structured_output.evidence_strength == "strong"
    assert "보수 검토" in structured_output.model_challenge
    assert "보류 또는 부적격 검토" in structured_output.audit_conclusion
    assert any("외부근거 위험" in item for item in agent.findings)


def test_committee_view_exposes_final_decision_fields() -> None:
    state: AgentState = {
        "company_id": "250",
        "company_name": "삼천당제약(주)",
        "market": "KOSDAQ",
        "source_feature_row": {
            "market": "KOSDAQ",
            "industry_macro_category": "manufacturing",
            "firm_size_group": "mid_sized",
            "current_ratio": 0.82,
            "cash_ratio": 0.11,
            "short_term_borrowings_share": 0.71,
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.31,
            "top_drivers": [("current_ratio", 0.22)],
        },
        "rule_result": {
            "recommendation": "review",
            "confidence": 0.62,
            "blocking_flags": [],
        },
        "news_cache_snapshot": {"status": "not_implemented"},
    }

    result = run(state)
    committee_view = result["committee_view"]

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["veto_triggered"] is False
    assert "conflict_resolution" in committee_view
    assert committee_view["key_risk_factors"]
    assert committee_view["evidence_summary"][0]["source"] == "model_view"
    assert "deterministic runner" in result["audit"][0].summary
