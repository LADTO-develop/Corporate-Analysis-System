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
    assert committee_view["committee_decision_type"] == "review_hold"
    assert committee_view["committee_decision_type_label"] == "확인필요 보류"
    assert committee_view["committee_risk_signal"] is False
    assert committee_view["veto_triggered"] is False
    assert committee_view["key_risk_factors"] == [
        "유동비율이 낮습니다.",
        "단기 유동성 추가 점검이 필요합니다.",
    ]
    assert committee_view["mitigating_factors"] == ["배당 이력이 있습니다."]
    trace = committee_view["decision_trace"]
    assert trace[0]["gate"] == "stage1_model_view"
    assert trace[-1]["gate"] == "final_committee_decision"
    assert trace[-1]["summary"] == "최종 위원회 판단은 보류이며, 세부 유형은 확인필요 보류입니다."


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


def test_committee_view_surfaces_evidence_limitations_in_summary() -> None:
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
                "근거 한계: 과거 기준일 이후 또는 날짜 미확인 근거 2건을 제외했습니다.",
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

    limitation_items = [
        item
        for item in committee_view["evidence_summary"]
        if item["source"] == "evidence_limitations"
    ]
    assert limitation_items
    assert "날짜 미확인 근거 2건" in limitation_items[0]["summary"]


def test_committee_view_decision_trace_marks_gate_statuses() -> None:
    state: AgentState = {
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.95,
            "threshold": 0.31,
        },
        "source_feature_row": {
            "icr_under_1": 1,
            "capital_impairment_ratio": 0.6,
            "interest_coverage_ratio": 0.4,
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
        recommendation="reject",
        agents=agents,
    )

    trace_by_gate = {item["gate"]: item for item in committee_view["decision_trace"]}
    assert trace_by_gate["stage1_model_view"]["triggered"] is True
    assert trace_by_gate["stage1_model_view"]["severity"] == "risk"
    assert trace_by_gate["reject_confirmation"]["triggered"] is True
    assert trace_by_gate["final_committee_decision"]["severity"] == "risk"


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
    assert committee_view["committee_decision_type"] == "risk_hold"
    assert committee_view["committee_decision_type_label"] == "위험 보류"
    assert committee_view["committee_risk_signal"] is True
    assert committee_view["veto_triggered"] is False
    assert committee_view["hidden_tail_risk_flag"] is True
    assert "숨은 꼬리위험 보완 플래그" in committee_view["hidden_tail_risk_reason"]
    assert "숨은 꼬리위험 보완 플래그" in committee_view["key_risk_factors"][0]
    assert "숨은 꼬리위험" in committee_view["conflict_resolution"]


def test_committee_view_does_not_flag_hidden_tail_risk_for_routine_external_context() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {
            "stock_code": "000250",
            "interest_coverage_ratio": -2.0,
            "icr_under_1": 1,
            "is_2y_consecutive_operating_loss": 1,
            "is_2y_consecutive_ocf_deficit": 1,
        },
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


def test_committee_view_holds_investment_model_with_secondary_review_trigger() -> None:
    state: AgentState = {
        "company_id": "311390",
        "company_name": "(주)네오크레마",
        "source_feature_row": {
            "stock_code": "311390",
            "interest_coverage_ratio": 4.4,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2836,
            "threshold": 0.315,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "medium",
            "trigger_reason": "43개 모델은 투자적격이나 45개 변수셋이 위험 기준선을 넘었습니다.",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2836,
            "threshold": 0.315,
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "감사보고서제출",
                    "summary": "(주)네오크레마 직접 관련 정기 공시입니다.",
                    "company_match": True,
                    "provider_relevance": "routine",
                    "disclosure_severity": "routine",
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

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "boundary_hold"
    assert committee_view["committee_decision_type_label"] == "경계등급 보류"
    assert committee_view["committee_risk_signal"] is False
    assert committee_view["hidden_tail_risk_flag"] is False
    assert "경계등급 보류 플래그" in committee_view["key_risk_factors"][0]
    assert "2차 보조 레이더 플래그" in committee_view["key_risk_factors"][1]
    assert "경계등급 보류" in committee_view["conflict_resolution"]
    assert "경계" in committee_view["final_review_memo"]


def test_committee_view_appends_informative_chair_report_memo() -> None:
    state: AgentState = {
        "company_id": "311390",
        "company_name": "(주)네오크레마",
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2836,
            "threshold": 0.315,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "medium",
            "trigger_reason": "45개 보조 변수셋이 추가 검토 대상으로 올렸습니다.",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2836,
            "threshold": 0.315,
        },
        "news_cache_snapshot": {"status": "disabled", "items": []},
    }
    chair_memo = (
        "위원회는 추가 검토 필요성은 인정하지만, 현재 공개 근거만으로 부적격을 확정하기보다 "
        "재무 방어력과 다음 공시를 함께 확인하는 보류 의견이 적절하다고 판단했습니다."
    )
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(
            role="chair_report",
            summary="종합",
            findings=["모델 보존", "위원회 범위", chair_memo],
            confidence=0.7,
        ),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="priority",
        agents=agents,
    )

    assert "위원회 보강 의견" in committee_view["final_review_memo"]
    assert chair_memo in committee_view["final_review_memo"]


def test_committee_view_keeps_low_probability_secondary_liquidity_watch_eligible() -> None:
    state: AgentState = {
        "company_id": "086670",
        "company_name": "(주)비엠티",
        "source_feature_row": {
            "stock_code": "086670",
            "current_ratio": 1.24,
            "cash_ratio": 0.09,
            "interest_coverage_ratio": 6.2,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2316,
            "threshold": 0.315,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "medium",
            "trigger_reason": "45개 보조 변수셋이 기준선 근처로 재점검을 요구했습니다.",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2316,
            "threshold": 0.315,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "medium",
        },
        "rule_result": {
            "risk_band": "watch",
            "recommendation": "watch",
            "reasons": [
                "model_probability_speculative=0.232",
                "cash_ratio=0.09 indicates weak cash liquidity",
            ],
            "blocking_flags": [],
        },
        "news_cache_snapshot": {"status": "disabled", "items": []},
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
    assert committee_view["key_risk_factors"] == [
        "현재 scaffold 기준 추가 위험 요인은 제한적입니다."
    ]


def test_committee_view_holds_confident_low_probability_liquidity_watch() -> None:
    state: AgentState = {
        "company_id": "086670",
        "company_name": "(주)비엠티",
        "source_feature_row": {
            "stock_code": "086670",
            "current_ratio": 1.40,
            "cash_ratio": 0.098,
            "interest_coverage_ratio": 4.09,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2316,
            "threshold": 0.315,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "medium",
            "trigger_reason": "기준선 0.10 이내의 보수 검토 대상입니다.",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2316,
            "threshold": 0.315,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "medium",
        },
        "rule_result": {
            "risk_band": "watch",
            "recommendation": "watch",
            "confidence": 0.62,
            "reasons": [
                "model_probability_speculative=0.232",
                "cash_ratio=0.10 indicates weak cash liquidity",
            ],
            "blocking_flags": [],
        },
        "news_cache_snapshot": {"status": "disabled", "items": []},
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
    assert committee_view["committee_decision_type"] == "review_hold"
    assert committee_view["committee_decision_type_label"] == "확인필요 보류"
    assert committee_view["committee_risk_signal"] is False
    assert "유동성 watch 신호" in committee_view["key_risk_factors"][0]
    assert "확인필요 보류" in committee_view["conflict_resolution"]


def test_committee_view_keeps_low_fold_threshold_liquidity_watch_eligible() -> None:
    state: AgentState = {
        "company_id": "140290",
        "company_name": "청광건설(주)",
        "source_feature_row": {
            "stock_code": "140290",
            "current_ratio": 0.0,
            "cash_ratio": 0.0,
            "interest_coverage_ratio": 1_000_000.0,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2198,
            "threshold": 0.225,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "high",
            "trigger_reason": "낮은 fold threshold 근처의 보수 검토 대상입니다.",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.2198,
            "threshold": 0.225,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "high",
        },
        "rule_result": {
            "risk_band": "watch",
            "recommendation": "watch",
            "confidence": 0.62,
            "reasons": [
                "model_probability_speculative=0.220",
                "current_ratio=0.00 is below the watch floor",
                "cash_ratio=0.00 indicates weak cash liquidity",
            ],
            "blocking_flags": [],
        },
        "news_cache_snapshot": {"status": "disabled", "items": []},
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


def test_committee_view_keeps_investment_model_eligible_with_noncritical_evidence() -> None:
    state: AgentState = {
        "company_id": "086710",
        "company_name": "선진뷰티사이언스(주)",
        "source_feature_row": {
            "stock_code": "086710",
            "interest_coverage_ratio": 5.1,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.3134,
            "threshold": 0.315,
            "risk_band": "watch",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.3134,
            "threshold": 0.315,
            "risk_band": "watch",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "감사보고서제출",
                    "summary": "선진뷰티사이언스(주) 직접 관련 정기 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "critical_terms": [],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "medium",
                    "evidence_score": 0.66,
                }
            ],
        },
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(
            role="evidence_audit",
            summary="근거 검토",
            findings=[
                "EvidenceAudit 검토 결론: 현재 확인된 외부근거만으로는 Stage 1 모델 판단을 실질적으로 뒤집기 어렵습니다. 모델 라벨은 투자적격으로 보존합니다.",
            ],
            confidence=0.6,
        ),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="review",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "적격"
    assert committee_view["key_risk_factors"] == [
        "현재 scaffold 기준 추가 위험 요인은 제한적입니다."
    ]


def test_committee_view_keeps_investment_model_on_hold_with_adverse_evidence() -> None:
    state: AgentState = {
        "company_id": "086710",
        "company_name": "선진뷰티사이언스(주)",
        "source_feature_row": {
            "stock_code": "086710",
            "interest_coverage_ratio": 5.1,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.21,
            "threshold": 0.315,
            "risk_band": "stable",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.21,
            "threshold": 0.315,
            "risk_band": "stable",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "단일판매ㆍ공급계약해지",
                    "summary": "선진뷰티사이언스(주) 직접 관련 계약해지 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "critical_terms": [],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "high",
                    "evidence_score": 0.82,
                }
            ],
        },
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(
            role="evidence_audit",
            summary="근거 검토",
            findings=[
                "외부근거 위험: 직접 관련 외부 위험 근거가 있어 위원회 보수 검토가 필요합니다."
            ],
            confidence=0.6,
        ),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="review",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"


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
    assert committee_view["committee_decision_type"] == "mitigation_hold"
    assert committee_view["committee_decision_type_label"] == "과민경고 완화 보류"
    assert committee_view["committee_risk_signal"] is False
    assert committee_view["veto_triggered"] is False
    assert "과민 경고 가능성" in committee_view["conflict_resolution"]
    assert "과민 경고" in committee_view["mitigating_factors"][0]


def test_committee_view_keeps_high_probability_risk_as_reject() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {
            "stock_code": "000250",
            "interest_coverage_ratio": -2.0,
            "icr_under_1": 1,
            "is_2y_consecutive_operating_loss": 1,
            "is_2y_consecutive_ocf_deficit": 1,
        },
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


def test_committee_view_softens_cash_rich_loss_stage_warning_to_mitigation_hold() -> None:
    state: AgentState = {
        "company_id": "389140",
        "company_name": "(주)포바이포",
        "source_feature_row": {
            "stock_code": "389140",
            "current_ratio": 6.73,
            "cash_ratio": 4.37,
            "equity_ratio": 0.88,
            "debt_ratio": 0.14,
            "total_borrowings_ratio": 0.0,
            "interest_coverage_ratio": -127.3,
            "net_margin": -0.54,
            "cashflow_coverage_ratio": 1.47,
            "ocf_to_total_liabilities": 0.02,
            "ocf_to_sales": 0.01,
            "icr_under_1": 1,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.904,
            "threshold": 0.250,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.904,
            "threshold": 0.250,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(자기주식처분결정)",
                    "summary": "(주)포바이포 직접 관련 routine 공시입니다.",
                    "company_match": True,
                    "provider_relevance": "routine",
                    "disclosure_severity": "routine",
                    "evidence_quality": "low",
                    "evidence_score": 0.54,
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
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "mitigation_hold"
    assert committee_view["committee_decision_type_label"] == "과민경고 완화 보류"
    assert committee_view["committee_risk_signal"] is False
    assert "현금·자본 버퍼" in committee_view["mitigating_factors"][0]


def test_committee_view_uses_prior_rating_boundary_reference_for_hold() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {"stock_code": "000250"},
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BBB-",
            "prior_credit_rating_rank": 10,
            "prior_rating_boundary_group": "exact_bbb_minus_bb_plus_boundary",
            "prior_rating_date": "2022-12-31",
            "prior_rating_age_days": 365,
            "prior_rating_agency": "NICE평가정보주식회사",
            "as_of_date": "2023-12-31",
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.48,
            "threshold": 0.315,
            "risk_band": "watch",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.48,
            "threshold": 0.315,
            "risk_band": "watch",
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
    assert committee_view["committee_decision_type"] == "boundary_hold"
    assert committee_view["committee_decision_type_label"] == "경계등급 보류"
    assert "이전 공개등급" in committee_view["key_risk_factors"][0]
    assert "prior_rating_reference" in {
        item["source"] for item in committee_view["evidence_summary"]
    }


def test_committee_view_holds_prior_boundary_when_model_is_close_to_threshold() -> None:
    state: AgentState = {
        "company_id": "000250",
        "company_name": "삼천당제약(주)",
        "source_feature_row": {"stock_code": "000250"},
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BB+",
            "prior_credit_rating_rank": 11,
            "prior_rating_boundary_group": "exact_bbb_minus_bb_plus_boundary",
            "prior_rating_date": "2022-12-31",
            "prior_rating_agency": "한국신용평가",
            "as_of_date": "2023-12-31",
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.24,
            "threshold": 0.315,
            "risk_band": "watch",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.24,
            "threshold": 0.315,
            "risk_band": "watch",
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
        recommendation="priority",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "boundary_hold"
    assert "BB+" in committee_view["key_risk_factors"][0]


def test_committee_view_softens_high_probability_prior_boundary_without_decisive_blockers() -> None:
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
            "cashflow_coverage_ratio": -0.85,
            "is_2y_consecutive_operating_loss": 1,
            "is_2y_consecutive_ocf_deficit": 1,
            "icr_under_1": 1,
            "short_term_borrowings_share": 0.28,
        },
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BBB-",
            "prior_rating_boundary_group": "exact_bbb_minus_bb_plus_boundary",
            "prior_rating_date": "2023-12-13",
            "prior_rating_agency": "한국평가데이터",
            "as_of_date": "2023-12-31",
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.956,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.956,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "전환사채(해외전환사채포함)발행후만기전사채취득",
                    "summary": "(주)라닉스 직접 관련 자금조달성 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "critical_terms": [],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "medium",
                    "evidence_score": 0.66,
                },
            ],
        },
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

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "mitigation_hold"
    assert committee_view["committee_decision_type_label"] == "과민경고 완화 보류"
    assert committee_view["committee_risk_signal"] is False
    assert "경계등급 과민경고 완화" in committee_view["mitigating_factors"][0]


def test_committee_view_keeps_prior_boundary_reject_with_extreme_distress() -> None:
    state: AgentState = {
        "company_id": "211460",
        "company_name": "(주)에스디생명공학",
        "source_feature_row": {
            "stock_code": "211460",
            "current_ratio": 0.57,
            "cash_ratio": 0.10,
            "equity_ratio": 0.05,
            "debt_ratio": 18.98,
            "total_borrowings_ratio": 0.32,
            "capital_impairment_ratio": 0.77,
            "interest_coverage_ratio": -9.05,
            "net_margin": -0.81,
            "ocf_to_sales": -0.17,
            "cashflow_coverage_ratio": -3.39,
            "is_2y_consecutive_operating_loss": 1,
            "is_2y_consecutive_ocf_deficit": 1,
            "icr_under_1": 1,
            "short_term_borrowings_share": 0.58,
        },
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BBB-",
            "prior_rating_boundary_group": "exact_bbb_minus_bb_plus_boundary",
            "prior_rating_date": "2022-03-18",
            "prior_rating_agency": "이크레더블",
            "as_of_date": "2022-12-31",
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.977,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.977,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {"status": "ready", "items": []},
    }
    agents = [
        AgentOutput(
            role="quant_credit",
            summary="정량 결과",
            findings=["완화 요인: 일부 사업 규모는 유지됩니다."],
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
    assert committee_view["committee_decision_type"] == "reject"


def test_committee_view_marks_unconfirmed_reject_as_review_hold() -> None:
    state: AgentState = {
        "company_id": "900001",
        "company_name": "테스트기업(주)",
        "source_feature_row": {
            "stock_code": "900001",
            "equity_ratio": 0.35,
            "debt_ratio": 1.8,
            "capital_impairment_ratio": 0.0,
            "interest_coverage_ratio": 2.0,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.86,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.86,
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
    assert committee_view["committee_decision_type"] == "review_hold"
    assert committee_view["committee_decision_type_label"] == "확인필요 보류"
    assert committee_view["committee_risk_signal"] is False
    assert "부적격 확정 게이트 미충족" in committee_view["key_risk_factors"][0]
    assert "확인필요 보류" in committee_view["conflict_resolution"]


def test_committee_view_marks_unconfirmed_reject_with_repeated_financing_as_risk_hold() -> None:
    state: AgentState = {
        "company_id": "900002",
        "company_name": "테스트자금조달(주)",
        "source_feature_row": {
            "stock_code": "900002",
            "equity_ratio": 0.35,
            "debt_ratio": 1.8,
            "capital_impairment_ratio": 0.0,
            "interest_coverage_ratio": 2.0,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.889,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.889,
            "threshold": 0.315,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주요사항보고서(전환사채권발행결정)",
                    "summary": "테스트자금조달(주) 직접 관련 자금조달성 공시입니다.",
                    "company_match": True,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "evidence_quality": "medium",
                    "evidence_score": 0.50,
                },
                {
                    "source": "opendart",
                    "title": "주요사항보고서(유상증자결정)",
                    "summary": "테스트자금조달(주) 직접 관련 자금조달성 공시입니다.",
                    "company_match": True,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "evidence_quality": "medium",
                    "evidence_score": 0.50,
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
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "risk_hold"
    assert committee_view["committee_decision_type_label"] == "위험 보류"
    assert committee_view["committee_risk_signal"] is True
    assert "자금조달성 공시" in committee_view["key_risk_factors"][0]
    assert "위험 보류" in committee_view["conflict_resolution"]


def test_committee_view_softens_near_threshold_warning_with_noisy_aggregated_news() -> None:
    state: AgentState = {
        "company_id": "900003",
        "company_name": "테스트경계(주)",
        "source_feature_row": {
            "stock_code": "900003",
            "current_ratio": 1.31,
            "cash_ratio": 0.15,
            "equity_ratio": 0.50,
            "debt_ratio": 1.00,
            "total_borrowings_ratio": 0.11,
            "short_term_borrowings_share": 0.87,
            "capital_impairment_ratio": 0.0,
            "interest_coverage_ratio": 10.0,
            "net_margin": 0.03,
            "ocf_to_sales": 0.06,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.262,
            "threshold": 0.250,
            "risk_band": "stable",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.262,
            "threshold": 0.250,
            "risk_band": "stable",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "naver_news",
                    "title": "[전일 주요 공시] 한미약품ㆍ테스트경계 등",
                    "summary": (
                        "임직원 횡령ㆍ배임 유죄 △다른기업, 무상증자 권리락 "
                        "△테스트경계, 현저한 시황변동 조회공시"
                    ),
                    "reliability": "medium",
                    "company_match": True,
                    "provider_relevance": "unknown",
                    "disclosure_severity": "veto",
                    "critical_terms": ["배임", "횡령"],
                    "critical_context_confirmed": True,
                    "veto_candidate": True,
                    "evidence_quality": "high",
                    "evidence_score": 0.98,
                },
                {
                    "source": "opendart",
                    "title": "감사보고서제출",
                    "summary": "테스트경계 OpenDART 외부감사관련 공시: 감사보고서제출",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "critical_terms": [],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "medium",
                    "evidence_score": 0.68,
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
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "mitigation_hold"
    assert committee_view["committee_decision_type_label"] == "과민경고 완화 보류"
    assert committee_view["committee_risk_signal"] is False
    assert "기준선 근처" in committee_view["mitigating_factors"][0]


def test_committee_view_softens_resolved_reverse_listing_halt_boundary_warning() -> None:
    state: AgentState = {
        "company_id": "018700",
        "company_name": "(주)바른손",
        "source_feature_row": {
            "stock_code": "018700",
            "current_ratio": 7.38,
            "cash_ratio": 0.50,
            "equity_ratio": 0.86,
            "debt_ratio": 0.17,
            "total_borrowings_ratio": 0.0,
            "short_term_borrowings_share": 0.0,
            "capital_impairment_ratio": 0.0,
            "interest_coverage_ratio": -6.21,
            "net_margin": 0.44,
            "ocf_to_sales": -0.19,
            "cashflow_coverage_ratio": -8.11,
            "icr_under_1": 1,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 1,
        },
        "prior_rating_reference": {
            "has_prior_rating": True,
            "prior_credit_rating": "BB+",
            "prior_credit_rating_rank": 11,
            "prior_rating_boundary_group": "exact_bbb_minus_bb_plus_boundary",
            "prior_rating_date": "2021-07-01",
            "prior_rating_agency": "SCI평가정보",
            "as_of_date": "2021-12-31",
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.983,
            "threshold": 0.310,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.983,
            "threshold": 0.310,
            "risk_band": "high_risk",
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주권매매거래정지(우회상장여부 및 요건충족확인)",
                    "summary": "(주)바른손 OpenDART 거래소공시 공시",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "critical_terms": ["거래정지"],
                    "critical_context_confirmed": True,
                    "veto_candidate": True,
                    "evidence_quality": "high",
                    "evidence_score": 1.0,
                },
                {
                    "source": "opendart",
                    "title": "주권매매거래정지해제(우회상장 미해당)",
                    "summary": "(주)바른손 OpenDART 거래소공시 공시",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "critical_terms": [],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "medium",
                    "evidence_score": 0.68,
                },
            ],
        },
    }
    agents = [
        AgentOutput(
            role="quant_credit",
            summary="정량 결과",
            findings=["완화 요인: 유동비율과 자기자본비율은 단기 방어력을 제공합니다."],
            confidence=0.8,
        ),
        AgentOutput(role="evidence_audit", summary="근거 검토", findings=[], confidence=0.6),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="reject",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "mitigation_hold"
    assert committee_view["committee_decision_type_label"] == "과민경고 완화 보류"
    assert committee_view["committee_risk_signal"] is False
    assert "경계등급 과민경고 완화" in committee_view["mitigating_factors"][0]


def test_committee_view_treats_bonus_issue_trading_halt_as_procedural_context() -> None:
    state: AgentState = {
        "company_id": "059120",
        "company_name": "(주)아진엑스텍",
        "source_feature_row": {
            "stock_code": "059120",
            "assets_total": 0,
            "gross_profit": 0,
            "current_ratio": 0,
            "cash_ratio": 0,
            "interest_coverage_ratio": 1_000_000.0,
            "cashflow_coverage_ratio": 1_000_000.0,
            "icr_under_1": 0,
            "is_2y_consecutive_operating_loss": 0,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.315,
            "threshold": 0.325,
            "stage2_secondary_trigger": True,
            "stage2_review_priority": "high",
            "trigger_reason": "기준선 근처 또는 취약 세그먼트 추가 검토 대상입니다.",
        },
        "xgboost_result": {
            "prediction_label": "투자적격",
            "probability_speculative": 0.315,
            "threshold": 0.325,
        },
        "rule_result": {
            "risk_band": "watch",
            "recommendation": "watch",
            "confidence": 0.62,
            "reasons": ["current_ratio=0.00", "cash_ratio=0.00"],
        },
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "주권매매거래정지(무상증자)",
                    "summary": "(주)아진엑스텍 OpenDART 거래소공시 공시",
                    "company_match": True,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "critical_terms": ["거래정지"],
                    "critical_context_confirmed": True,
                    "veto_candidate": True,
                    "evidence_quality": "high",
                    "evidence_score": 1.0,
                },
                {
                    "source": "opendart",
                    "title": "현금ㆍ현물배당결정",
                    "summary": "(주)아진엑스텍 직접 관련 routine 공시입니다.",
                    "company_match": True,
                    "provider_relevance": "routine",
                    "disclosure_severity": "routine",
                    "evidence_quality": "low",
                    "evidence_score": 0.54,
                },
                {
                    "source": "naver_news",
                    "title": "직접금융 통한 자금조달 크게 늘었다",
                    "summary": "시장 전반 기사에 (주)아진엑스텍이 언급됐지만 직접 자금조달 공시는 아닙니다.",
                    "company_match": True,
                    "provider_relevance": "unknown",
                    "disclosure_severity": "unknown",
                    "evidence_quality": "medium",
                    "evidence_score": 0.74,
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

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "boundary_hold"
    assert committee_view["committee_risk_signal"] is False
    assert committee_view["veto_triggered"] is False
    assert "경계등급 보류" in committee_view["conflict_resolution"]


def test_committee_view_softens_model_only_high_probability_warning_to_hold() -> None:
    state: AgentState = {
        "company_id": "033540",
        "company_name": "(주)파라텍",
        "source_feature_row": {
            "stock_code": "033540",
            "current_ratio": 1.107,
            "cash_ratio": 0.019,
            "interest_coverage_ratio": -29.0,
            "capital_impairment_ratio": -5.253,
            "equity_ratio": 0.430,
            "debt_ratio": 1.328,
            "total_borrowings_ratio": 0.238,
            "short_term_borrowings_share": 0.651,
            "cashflow_coverage_ratio": 7.353,
            "ocf_to_total_liabilities": 0.133,
            "ocf_to_sales": 0.074,
            "icr_under_1": 1,
            "is_2y_consecutive_operating_loss": 1,
            "is_2y_consecutive_ocf_deficit": 0,
        },
        "model_view": {
            "prediction_label": "부적격",
            "probability_speculative": 0.851,
            "threshold": 0.225,
            "risk_band": "high_risk",
        },
        "xgboost_result": {
            "prediction_label": "부적격",
            "probability_speculative": 0.851,
            "threshold": 0.225,
            "risk_band": "high_risk",
        },
        "rule_result": {
            "risk_band": "high_risk",
            "recommendation": "defer",
            "blocking_flags": ["interest_coverage_under_1"],
        },
        "news_cache_snapshot": {"status": "disabled", "items": []},
    }
    agents = [
        AgentOutput(role="quant_credit", summary="정량 결과", findings=[], confidence=0.8),
        AgentOutput(
            role="evidence_audit",
            summary="근거 검토",
            findings=["완화 요인: 배당금 지급과 장기 상장 이력이 일부 완화 근거입니다."],
            confidence=0.6,
        ),
        AgentOutput(role="chair_report", summary="종합", findings=[], confidence=0.7),
    ]

    committee_view = build_committee_view(
        bundle=build_stage2_input_bundle(state),
        recommendation="defer",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["committee_decision_type"] == "mitigation_hold"
    assert committee_view["committee_decision_type_label"] == "과민경고 완화 보류"
    assert committee_view["committee_risk_signal"] is False
    assert "고확률 모델 단독 경고 완화" in committee_view["mitigating_factors"][0]
    assert "OCF와 자본/부채 구조" in committee_view["mitigating_factors"][0]


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


def test_committee_view_keeps_high_probability_risk_with_noncritical_evidence_only() -> None:
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
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "전환사채(해외전환사채포함)발행후만기전사채취득",
                    "summary": "(주)라닉스 직접 관련 자금조달성 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "caution",
                    "disclosure_severity": "caution",
                    "critical_terms": [],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "medium",
                    "evidence_score": 0.66,
                },
                {
                    "source": "naver_news",
                    "title": "[전일 주요 공시] 여러 기업 횡령 배임 소식",
                    "summary": "(주)라닉스가 언급된 종합 기사이나 위험 키워드 문맥은 다른 기업에 해당합니다.",
                    "reliability": "medium",
                    "company_match": True,
                    "provider_relevance": "unknown",
                    "disclosure_severity": "veto",
                    "critical_terms": ["횡령", "배임"],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "low",
                    "evidence_score": 0.54,
                },
            ],
        },
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
    assert committee_view["committee_decision_type"] == "reject"
    assert committee_view["committee_risk_signal"] is True
    assert "과민 경고" not in committee_view["conflict_resolution"]


def test_committee_view_keeps_reject_when_external_evidence_is_adverse() -> None:
    state: AgentState = {
        "company_id": "123456",
        "company_name": "테스트기업",
        "source_feature_row": {"stock_code": "123456"},
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
        "news_cache_snapshot": {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "단일판매ㆍ공급계약해지",
                    "summary": "테스트기업 직접 관련 계약해지 공시입니다.",
                    "reliability": "high",
                    "company_match": True,
                    "provider_relevance": "risk",
                    "disclosure_severity": "adverse",
                    "critical_terms": [],
                    "critical_context_confirmed": False,
                    "veto_candidate": False,
                    "evidence_quality": "high",
                    "evidence_score": 0.82,
                }
            ],
        },
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
    assert committee_view.committee_decision_type == "eligible"
    assert committee_view.committee_decision_type_label == "적격"
    assert committee_view.committee_risk_signal is False
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
