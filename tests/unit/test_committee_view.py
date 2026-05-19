"""Tests for committee_view payload construction."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cas.agents.committee_schema import CommitteeViewPayload
from cas.agents.committee_view import build_committee_view, build_committee_view_model
from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.state import AgentOutput, AgentState


def test_committee_view_maps_review_to_hold_label() -> None:
    state: AgentState = {
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {"status": "not_implemented"},
    }
    agents = [
        AgentOutput(
            role="quant_credit",
            summary="모델은 투자적격으로 판단했습니다.",
            findings=["핵심 위험 요인: 유동비율이 낮습니다.", "완화 요인: 배당 이력이 있습니다."],
            confidence=0.8,
        ),
        AgentOutput(
            role="evidence_audit",
            summary="외부 근거는 아직 제한적입니다.",
            findings=["부채·유동성 검증 의견: 단기 유동성 추가 점검이 필요합니다."],
            confidence=0.6,
        ),
        AgentOutput(
            role="chair_report",
            summary="보류 의견으로 종합했습니다.",
            findings=[],
            confidence=0.7,
        ),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="review",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["veto_triggered"] is False
    assert committee_view["key_risk_factors"] == [
        "유동비율이 낮습니다.",
        "단기 유동성 추가 점검이 필요합니다.",
    ]
    assert committee_view["mitigating_factors"] == ["배당 이력이 있습니다."]


def test_committee_view_moves_debt_liquidity_support_to_mitigation() -> None:
    state: AgentState = {
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {"status": "disabled"},
    }
    agents = [
        AgentOutput(
            role="quant_credit",
            summary="모델은 투자적격으로 판단했습니다.",
            findings=["핵심 위험 요인: 상위 변수 기준 뚜렷한 위험 가중 요인은 제한적입니다."],
            confidence=0.8,
        ),
        AgentOutput(
            role="evidence_audit",
            summary="부채 유동성은 안정적입니다.",
            findings=[
                "부채·유동성 검증 의견: 부채 및 유동성 지표는 현재 투자적격 판단에 일부 완충 근거를 제공합니다."
            ],
            confidence=0.6,
        ),
        AgentOutput(role="chair_report", summary="적격 의견입니다.", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["key_risk_factors"] == [
        "현재 scaffold 기준 추가 위험 요인은 제한적입니다."
    ]
    assert committee_view["mitigating_factors"] == [
        "부채 및 유동성 지표는 현재 투자적격 판단에 일부 완충 근거를 제공합니다."
    ]


def test_committee_view_escalates_priority_to_hold_on_evidence_audit_risk() -> None:
    state: AgentState = {
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {"status": "ready"},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(
            role="evidence_audit",
            summary="외부근거 검토",
            findings=[
                "외부근거 위험: 직접 관련 외부 위험 근거가 있어 위원회 보수 검토가 필요합니다.",
                "EvidenceAudit 검토 결론: 외부 근거가 강하므로 모델 원판단보다 보수적인 보류 또는 부적격 검토가 필요합니다.",
            ],
            confidence=0.7,
        ),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["key_risk_factors"] == [
        "직접 관련 외부 위험 근거가 있어 위원회 보수 검토가 필요합니다.",
        "외부 근거가 강하므로 모델 원판단보다 보수적인 보류 또는 부적격 검토가 필요합니다.",
    ]
    assert "주요 위험은 직접 관련 외부 위험 근거" in committee_view["final_review_memo"]


def test_committee_view_does_not_treat_neutral_audit_conclusion_as_mitigation() -> None:
    state: AgentState = {
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {"status": "ready"},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(
            role="evidence_audit",
            summary="외부근거 검토",
            findings=[
                "EvidenceAudit 검토 결론: 현재 확인된 외부 근거는 모델 원판단을 뒤집기보다 설명과 점검 포인트를 보완합니다."
            ],
            confidence=0.7,
        ),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "적격"
    assert committee_view["mitigating_factors"] == [
        "현재 scaffold 기준 명시적 완화 요인은 제한적입니다."
    ]


def test_committee_view_does_not_escalate_on_unconfirmed_external_risk() -> None:
    state: AgentState = {
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {"status": "not_requested"},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(
            role="evidence_audit",
            summary="외부근거 검토",
            findings=[
                "외부근거 강도: none",
                "EvidenceAudit 검토 결론: 외부 치명 리스크는 확정되지 않았지만 부채·유동성 측면에서 보류 의견을 강화합니다.",
                "외부근거 점검: 수집 상태가 `not_requested`라서 확인 가능한 외부 근거가 제한적입니다.",
            ],
            confidence=0.6,
        ),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "적격"
    assert committee_view["key_risk_factors"] == [
        "현재 scaffold 기준 추가 위험 요인은 제한적입니다."
    ]


def test_committee_view_flags_hidden_tail_risk_from_direct_external_adverse_evidence() -> None:
    state: AgentState = {
        "company_id": "096770",
        "company_name": "에스케이이노베이션(주)",
        "source_feature_row": {"stock_code": "096770"},
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.015,
            "threshold": 0.315,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "에스케이이노베이션(주) 자본잠식 관련 주요사항보고",
                    "summary": "에스케이이노베이션(주) 직접 관련 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "risk",
                    "critical_terms": ["자본잠식"],
                    "evidence_quality": "high",
                    "evidence_score": 0.91,
                }
            ],
        },
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["veto_triggered"] is False
    assert committee_view["hidden_tail_risk_flag"] is True
    assert "숨은 꼬리위험 보완 플래그" in committee_view["hidden_tail_risk_reason"]
    assert "숨은 꼬리위험 보완 플래그" in committee_view["key_risk_factors"][0]
    assert "숨은 꼬리위험" in committee_view["conflict_resolution"]


def test_committee_view_does_not_flag_hidden_tail_risk_for_routine_external_context() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {"stock_code": "000250"},
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.04,
            "threshold": 0.315,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "삼천당제약(주) 사업보고서",
                    "summary": "정기 사업보고서 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "routine",
                    "critical_terms": [],
                    "evidence_quality": "high",
                    "evidence_score": 0.86,
                }
            ],
        },
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "적격"
    assert committee_view["hidden_tail_risk_flag"] is False
    assert committee_view["hidden_tail_risk_reason"] == ""


def test_committee_view_softens_near_threshold_overwarning_to_hold() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {"stock_code": "000250"},
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.34,
            "threshold": 0.315,
            "risk_band": "watch",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.34,
            "threshold": 0.315,
            "risk_band": "watch",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "삼천당제약(주) 사업보고서",
                    "summary": "정기 사업보고서 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "context",
                    "critical_terms": [],
                    "evidence_quality": "high",
                    "evidence_score": 0.86,
                }
            ],
        },
    }
    agents = [
        AgentOutput(
            role="quant_credit",
            summary="정량 결과",
            findings=["완화 요인: 현금비율이 산업 대비 양호합니다."],
            confidence=0.8,
        ),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["veto_triggered"] is False
    assert "과민 경고" in committee_view["mitigating_factors"][0]


def test_committee_view_keeps_high_probability_risk_as_reject() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {"stock_code": "000250"},
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.91,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.91,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    agents = [
        AgentOutput(
            role="quant_credit",
            summary="정량 결과",
            findings=["완화 요인: 일부 현금성 자산이 확인됩니다."],
            confidence=0.8,
        ),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "부적격"


def test_committee_view_softens_high_probability_risk_with_financial_resilience() -> None:
    state: AgentState = {
        "company_id": "196700",
        "company_name": "(주)웹스",
        "source_feature_row": {
            "stock_code": "196700",
            "current_ratio": 1.52,
            "cash_ratio": 0.29,
            "equity_ratio": 0.51,
            "debt_ratio": 0.98,
            "total_borrowings_ratio": 0.44,
            "capital_impairment_ratio": -3.75,
            "interest_coverage_ratio": 1.79,
            "net_margin": 0.03,
            "ocf_to_sales": -0.15,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
            "icr_under_1": 0,
            "short_term_borrowings_share": 0.79,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.81,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.81,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert "고확률 과민 경고 방어 신호" in committee_view["mitigating_factors"][0]


def test_committee_view_keeps_high_probability_risk_when_blockers_exist() -> None:
    state: AgentState = {
        "company_id": "317120",
        "company_name": "(주)라닉스",
        "source_feature_row": {
            "stock_code": "317120",
            "current_ratio": 1.90,
            "cash_ratio": 0.25,
            "equity_ratio": 0.36,
            "debt_ratio": 1.75,
            "total_borrowings_ratio": 0.62,
            "capital_impairment_ratio": -1.17,
            "interest_coverage_ratio": -1.92,
            "net_margin": -0.47,
            "ocf_to_sales": -0.11,
            "is_2y_consecutive_operating_loss": 1,
            "is_2y_consecutive_ocf_deficit": 1,
            "icr_under_1": 1,
            "short_term_borrowings_share": 0.28,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.95,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.95,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    agents = [
        AgentOutput(
            role="quant_credit",
            summary="정량 결과",
            findings=["완화 요인: 유동비율은 단기 방어력을 일부 제공합니다."],
            confidence=0.8,
        ),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "부적격"


def test_committee_view_limits_investment_model_rule_defer_to_hold_without_veto() -> None:
    state: AgentState = {
        "company_id": "014470",
        "company_name": "(주)부방",
        "source_feature_row": {
            "stock_code": "014470",
            "interest_coverage_ratio": -0.10,
            "icr_under_1": 1,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2923,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2923,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["veto_triggered"] is False


def test_committee_view_model_validates_strict_payload() -> None:
    state: AgentState = {
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {"status": "not_implemented"},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view_model(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view.final_committee_label == "적격"
    assert committee_view.model_dump(mode="json")["evidence_summary"][0]["source"] == "model_view"


def test_committee_view_schema_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        CommitteeViewPayload(
            final_committee_label="검토중",
            veto_triggered=False,
            conflict_resolution="invalid label should be rejected",
            key_risk_factors=[],
            mitigating_factors=[],
            evidence_summary=[],
            final_review_memo="invalid label should be rejected",
        )


def test_committee_view_veto_overrides_to_reject() -> None:
    state: AgentState = {
        "rule_result": {"blocking_flags": ["fraud_risk"]},
        "news_cache_snapshot": {"status": "collected"},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(
            {**state, "xgboost_result": {"prediction_label": "투자적격"}}
        ),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "부적격"
    assert committee_view["veto_triggered"] is True
    assert "보수 조정" in committee_view["conflict_resolution"]


def test_committee_view_veto_uses_configured_korean_marker() -> None:
    state: AgentState = {
        "rule_result": {"blocking_flags": ["횡령_공시"]},
        "news_cache_snapshot": {"status": "collected"},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(
            {**state, "xgboost_result": {"prediction_label": "투자적격"}}
        ),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "부적격"
    assert committee_view["veto_triggered"] is True


def test_committee_view_does_not_veto_on_external_keyword_only() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {"stock_code": "000250"},
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {
            "status": "ready",
            "has_critical_risk": True,
            "critical_terms": ["횡령", "배임"],
            "items": [
                {
                    "source": "tavily",
                    "title": "횡령 배임 공시 안내",
                    "summary": "회사명 직접 매칭이 없는 일반 안내 페이지입니다.",
                    "reliability": "medium",
                    "company_match": False,
                    "critical_terms": ["횡령", "배임"],
                }
            ],
        },
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "적격"
    assert committee_view["veto_triggered"] is False
    assert any(
        item["reliability"] == "low_relevance"
        for item in committee_view["evidence_summary"]
        if item["source"] == "tavily"
    )


def test_committee_view_vetoes_on_direct_high_reliability_external_evidence() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {"stock_code": "000250"},
        "xgboost_result": {"prediction_label": "투자적격"},
        "news_cache_snapshot": {
            "status": "ready",
            "has_critical_risk": True,
            "critical_terms": ["횡령"],
            "items": [
                {
                    "source": "opendart",
                    "title": "삼천당제약(주) 횡령 혐의 발생",
                    "summary": "삼천당제약(주) 공시: 횡령 혐의 발생",
                    "reliability": "high",
                    "company_match": True,
                    "critical_terms": ["횡령"],
                },
                {
                    "source": "naver_news",
                    "title": "삼천당제약 횡령 관련 보도",
                    "summary": "삼천당제약 관련 후속 보도입니다.",
                    "reliability": "medium",
                    "company_match": True,
                    "critical_terms": ["횡령"],
                },
            ],
        },
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "부적격"
    assert committee_view["veto_triggered"] is True
