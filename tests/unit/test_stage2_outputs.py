"""Tests for Stage 2 agent-specific output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    ReviewQAOutput,
    RiskRecallQAOutput,
)


def test_quant_credit_output_flattens_to_common_agent_output() -> None:
    output = QuantCreditOutput(
        quant_summary="정량 모델 해석 요약",
        model_rationale="상위 SHAP 변수 2개를 검토했습니다.",
        key_risk_factors=["유동비율이 낮습니다."],
        mitigating_factors=["현금비율이 높습니다."],
        confidence=0.8,
    )

    agent = output.to_agent_output()

    assert agent.role == "quant_credit"
    assert agent.summary == "정량 모델 해석 요약"
    assert agent.findings[1] == "핵심 위험 요인: 유동비율이 낮습니다."
    assert agent.findings[2] == "완화 요인: 현금비율이 높습니다."


def test_evidence_audit_output_flattens_to_common_agent_output() -> None:
    output = EvidenceAuditOutput(
        evidence_summary="외부 근거와 유동성 신호를 검토했습니다.",
        evidence_status="collected",
        evidence_reliability="출처 신뢰도를 구분했습니다.",
        evidence_strength="moderate",
        model_challenge="정량 모델 판단과 외부 근거 사이의 중대한 충돌은 제한적입니다.",
        audit_conclusion="현재 확인된 외부 근거는 설명과 점검 포인트를 보완합니다.",
        debt_liquidity_cross_check=["부채·유동성 검증 의견: 추가 점검 필요"],
        macro_industry_sensitivity=["거시·시장 점검: 스프레드 확인"],
        external_evidence_findings=["외부 근거(naver, 신뢰도 medium): 기사"],
        evidence_limitations=["기준일 이후 근거는 제외했습니다."],
        confidence=0.7,
    )

    agent = output.to_agent_output()

    assert agent.role == "evidence_audit"
    assert "collected" in agent.findings[0]
    assert "recommended_evidence_treatment=context_only" in " ".join(agent.findings)
    assert "추가 점검 필요" in " ".join(agent.findings)
    assert "근거 한계: 기준일 이후 근거는 제외했습니다." in agent.findings


def test_chair_report_output_flattens_to_common_agent_output() -> None:
    output = ChairReportOutput(
        report_summary="최종 보고서 요약",
        model_preservation_note="model_view는 보존합니다.",
        committee_scope_note="committee_view만 보완합니다.",
        final_review_memo_seed="심사 메모 초안",
        confidence=0.75,
    )

    agent = output.to_agent_output()

    assert agent.role == "chair_report"
    assert agent.summary == "최종 보고서 요약"
    assert agent.findings == [
        "model_view는 보존합니다.",
        "committee_view만 보완합니다.",
        "심사 메모 초안",
    ]


def test_review_qa_output_flattens_to_common_agent_output() -> None:
    output = ReviewQAOutput(
        qa_summary="최종 라벨과 메모를 검수했습니다.",
        trigger_reasons=["investment_model_hold", "ambiguous_external_evidence"],
        label_memo_consistency="최종 라벨과 메모가 충돌하지 않습니다.",
        risk_hold_assessment="overstated",
        evidence_cutoff_check="기준일 이후 근거는 사용하지 않았습니다.",
        overhold_guardrail_assessment="정상기업 과잉 보류 guardrail 검토가 필요합니다.",
        recommended_action="downgrade_risk_hold_to_boundary_hold",
        manual_review_tasks=["보류 사유를 수동 확인합니다."],
        missing_evidence=["기준일 이전 직접 공시"],
        monitoring_triggers=["신규 공시 발생"],
        confidence=0.72,
    )

    agent = output.to_agent_output()

    assert agent.role == "review_qa"
    assert agent.summary == "최종 라벨과 메모를 검수했습니다."
    assert "investment_model_hold" in agent.findings[0]
    assert "QA 권고: downgrade_risk_hold_to_boundary_hold" in agent.findings
    assert "수동 검토 과제: 보류 사유를 수동 확인합니다." in agent.findings
    assert "누락 근거: 기준일 이전 직접 공시" in agent.findings
    assert "모니터링 트리거: 신규 공시 발생" in agent.findings


def test_risk_recall_qa_output_flattens_to_common_agent_output() -> None:
    output = RiskRecallQAOutput(
        qa_summary="적격 판단의 위험 누락 가능성을 재검수했습니다.",
        trigger_reasons=["eligible_near_threshold", "eligible_with_recall_watch_evidence"],
        eligible_safety_assessment="needs_boundary_review",
        financial_resilience_check="유동성은 약하지만 현금흐름 방어축은 일부 확인됩니다.",
        evidence_recall_check="외부 공시는 watch_context 수준입니다.",
        rating_boundary_check="모델 확률이 기준선 근처입니다.",
        recommended_action="escalate_eligible_to_boundary_hold",
        manual_review_tasks=["재무 방어축을 확인합니다."],
        missing_evidence=["최신 신용등급 reference"],
        monitoring_triggers=["기준선 재상회"],
        confidence=0.66,
    )

    agent = output.to_agent_output()

    assert agent.role == "risk_recall_qa"
    assert agent.summary == "적격 판단의 위험 누락 가능성을 재검수했습니다."
    assert "eligible_near_threshold" in agent.findings[0]
    assert "Recall QA 권고: escalate_eligible_to_boundary_hold" in agent.findings
    assert "수동 검토 과제: 재무 방어축을 확인합니다." in agent.findings
    assert "누락 근거: 최신 신용등급 reference" in agent.findings
    assert "모니터링 트리거: 기준선 재상회" in agent.findings


def test_stage2_output_schema_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        QuantCreditOutput(
            quant_summary="invalid",
            model_rationale="invalid",
            key_risk_factors=[],
            mitigating_factors=[],
            confidence=1.2,
        )


def test_evidence_audit_output_requires_llm_guardrail_fields() -> None:
    with pytest.raises(ValidationError):
        EvidenceAuditOutput(
            evidence_summary="외부 근거 검토",
            evidence_status="ready",
            evidence_reliability="신뢰도 점검",
            debt_liquidity_cross_check=[],
            macro_industry_sensitivity=[],
            external_evidence_findings=[],
            confidence=0.7,
        )
