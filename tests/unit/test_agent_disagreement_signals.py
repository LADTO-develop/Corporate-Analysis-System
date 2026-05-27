"""Unit tests for Stage 2 agent disagreement diagnostics."""

from __future__ import annotations

from cas.agents.signals import evaluate_agent_disagreement
from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.state import AgentState


def test_agent_disagreement_flags_quant_risk_without_critical_evidence() -> None:
    bundle = build_stage2_input_bundle(
        AgentState(
            xgboost_result={
                "prediction_label": "부적격",
                "probability_speculative": 0.62,
                "threshold": 0.31,
            },
            news_cache_snapshot={"status": "ready", "items": []},
        )
    )
    signals = evaluate_agent_disagreement(
        bundle=bundle,
        committee_view={
            "final_committee_label": "보류",
            "committee_decision_type": "risk_hold",
        },
        quant_credit=_quant_credit(confidence=0.86),
        evidence_audit=_evidence_audit(
            recommended_evidence_treatment="watch_context",
            critical_evidence_count=0,
            watch_context_count=1,
            confidence=0.45,
        ),
        chair_report=_chair_report(confidence=0.82),
    )

    assert signals.level == "high"
    assert signals.score >= 0.55
    assert signals.reasons == [
        "quant_risk_evidence_watch_context",
        "chair_risk_without_critical_evidence",
        "agent_confidence_gap",
    ]
    assert "판단 충돌 점수" in signals.summary


def test_agent_disagreement_flags_eligible_with_substantive_evidence() -> None:
    bundle = build_stage2_input_bundle(
        AgentState(
            xgboost_result={
                "prediction_label": "투자적격",
                "probability_speculative": 0.12,
                "threshold": 0.31,
            },
            news_cache_snapshot={"status": "ready", "items": []},
        )
    )
    signals = evaluate_agent_disagreement(
        bundle=bundle,
        committee_view={
            "final_committee_label": "적격",
            "committee_decision_type": "eligible",
        },
        quant_credit=_quant_credit(confidence=0.78),
        evidence_audit=_evidence_audit(
            recommended_evidence_treatment="substantive_review",
            critical_evidence_count=1,
            watch_context_count=0,
            confidence=0.72,
        ),
        chair_report=_chair_report(confidence=0.80),
    )

    assert signals.level == "high"
    assert signals.reasons == [
        "quant_investment_evidence_substantive",
        "chair_eligible_with_substantive_evidence",
    ]


def test_agent_disagreement_flags_label_memo_conflict() -> None:
    bundle = build_stage2_input_bundle(
        AgentState(
            xgboost_result={
                "prediction_label": "투자적격",
                "probability_speculative": 0.29,
                "threshold": 0.31,
            },
            news_cache_snapshot={"status": "ready", "items": []},
        )
    )
    signals = evaluate_agent_disagreement(
        bundle=bundle,
        committee_view={
            "final_committee_label": "보류",
            "committee_decision_type": "boundary_hold",
            "final_review_memo": "위원회는 정량 해석을 검토해 최종 의견을 적격으로 정리했습니다.",
        },
        quant_credit=_quant_credit(confidence=0.76),
        evidence_audit=_evidence_audit(
            recommended_evidence_treatment="context_only",
            critical_evidence_count=0,
            watch_context_count=0,
            confidence=0.74,
        ),
        chair_report=_chair_report(confidence=0.78),
    )

    assert signals.level == "medium"
    assert signals.reasons == ["committee_label_memo_conflict"]
    assert "메모 문구" in signals.summary


def test_agent_disagreement_ignores_negated_eligible_memo_phrase() -> None:
    bundle = build_stage2_input_bundle(
        AgentState(
            xgboost_result={
                "prediction_label": "투자적격",
                "probability_speculative": 0.29,
                "threshold": 0.31,
            },
            news_cache_snapshot={"status": "ready", "items": []},
        )
    )
    signals = evaluate_agent_disagreement(
        bundle=bundle,
        committee_view={
            "final_committee_label": "보류",
            "committee_decision_type": "boundary_hold",
            "final_review_memo": "이를 최종 적격으로 확정하지 않고 보류로 재점검합니다.",
        },
        quant_credit=_quant_credit(confidence=0.76),
        evidence_audit=_evidence_audit(
            recommended_evidence_treatment="context_only",
            critical_evidence_count=0,
            watch_context_count=0,
            confidence=0.74,
        ),
        chair_report=_chair_report(confidence=0.78),
    )

    assert signals.reasons == []


def _quant_credit(*, confidence: float) -> QuantCreditOutput:
    return QuantCreditOutput(
        quant_summary="정량 모델 요약",
        model_rationale="위험확률과 기준선을 비교했습니다.",
        key_risk_factors=["위험 요인"],
        mitigating_factors=["완화 요인"],
        confidence=confidence,
    )


def _evidence_audit(
    *,
    recommended_evidence_treatment: str,
    critical_evidence_count: int,
    watch_context_count: int,
    confidence: float,
) -> EvidenceAuditOutput:
    return EvidenceAuditOutput(
        evidence_summary="외부근거 요약",
        evidence_status="ready",
        evidence_reliability="직접 관련 근거만 반영했습니다.",
        evidence_strength="moderate",
        model_challenge="모델과 외부근거의 방향성을 비교했습니다.",
        audit_conclusion="추가 검토가 필요합니다.",
        debt_liquidity_cross_check=[],
        macro_industry_sensitivity=[],
        external_evidence_findings=[],
        critical_evidence_count=critical_evidence_count,
        watch_context_count=watch_context_count,
        hard_distress_detected=False,
        recommended_evidence_treatment=recommended_evidence_treatment,
        confidence=confidence,
    )


def _chair_report(*, confidence: float) -> ChairReportOutput:
    return ChairReportOutput(
        report_summary="위원회 요약",
        model_preservation_note="모델 원판단은 보존합니다.",
        committee_scope_note="위원회 판단은 보조 검토입니다.",
        final_review_memo_seed="최종 메모 초안",
        confidence=confidence,
    )
