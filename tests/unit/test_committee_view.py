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


def test_committee_view_collects_evidence_audit_risk_conclusion() -> None:
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
        recommendation="watch",
        agents=agents,
    )

    assert committee_view["final_committee_label"] == "보류"
    assert committee_view["key_risk_factors"] == [
        "직접 관련 외부 위험 근거가 있어 위원회 보수 검토가 필요합니다.",
        "외부 근거가 강하므로 모델 원판단보다 보수적인 보류 또는 부적격 검토가 필요합니다.",
    ]
    assert "주요 위험은 직접 관련 외부 위험 근거" in committee_view["final_review_memo"]


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
