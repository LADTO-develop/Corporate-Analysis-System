"""Tests for Stage 2 runner adapters."""

from __future__ import annotations

from typing import Any

from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
)
from cas.agents.stage2_runner import (
    AgnoStage2AgentRunner,
    DeterministicStage2AgentRunner,
    Stage2LLMResponse,
)
from cas.agents.state import AgentState


class _FakeStage2LLMClient:
    def run_structured(
        self,
        *,
        prompt_payload: dict[str, Any],
        output_schema: type[Stage2LLMResponse],
    ) -> object:
        assert prompt_payload["recommendation"] == "review"
        return output_schema(
            quant_credit=QuantCreditOutput(
                quant_summary="LLM 정량 요약",
                model_rationale="LLM 모델 판단 근거",
                key_risk_factors=["LLM 위험"],
                mitigating_factors=["LLM 완화"],
                confidence=0.75,
            ),
            evidence_audit=EvidenceAuditOutput(
                evidence_summary="LLM 근거 검토",
                evidence_status="ready",
                evidence_reliability="LLM 신뢰도 점검",
                debt_liquidity_cross_check=["LLM 부채 점검"],
                macro_industry_sensitivity=["LLM 거시 점검"],
                external_evidence_findings=["LLM 외부 근거"],
                confidence=0.65,
            ),
            chair_report=ChairReportOutput(
                report_summary="LLM 종합 보고",
                model_preservation_note="model_view 보존",
                committee_scope_note="committee_view 보완",
                final_review_memo_seed="메모 초안",
                confidence=0.7,
            ),
        )


class _FailingStage2LLMClient:
    def run_structured(
        self,
        *,
        prompt_payload: dict[str, Any],
        output_schema: type[Stage2LLMResponse],
    ) -> object:
        raise RuntimeError("temporary LLM outage")


def test_deterministic_stage2_runner_returns_fixed_role_order() -> None:
    bundle = build_stage2_input_bundle(_minimal_state())
    runner = DeterministicStage2AgentRunner(
        quant_credit_agent=lambda _: QuantCreditOutput(
            quant_summary="정량 요약",
            model_rationale="모델 판단 근거",
            key_risk_factors=["위험"],
            mitigating_factors=["완화"],
            confidence=0.8,
        ),
        evidence_audit_agent=lambda _: EvidenceAuditOutput(
            evidence_summary="근거 검토",
            evidence_status="disabled",
            evidence_reliability="신뢰도 점검",
            debt_liquidity_cross_check=["부채 점검"],
            macro_industry_sensitivity=["거시 점검"],
            external_evidence_findings=["외부 근거"],
            confidence=0.6,
        ),
        chair_report_agent=lambda _bundle, _recommendation, confidence: ChairReportOutput(
            report_summary="종합 보고",
            model_preservation_note="model_view 보존",
            committee_scope_note="committee_view 보완",
            final_review_memo_seed="메모 초안",
            confidence=confidence,
        ),
    )

    outputs = runner.run(bundle=bundle, recommendation="review", confidence=0.7)
    agents = [output.to_agent_output() for output in outputs]

    assert runner.backend_name == "deterministic"
    assert tuple(output.role for output in outputs) == (
        "quant_credit",
        "evidence_audit",
        "chair_report",
    )
    assert [agent.role for agent in agents] == [
        "quant_credit",
        "evidence_audit",
        "chair_report",
    ]


def test_agno_stage2_runner_accepts_structured_llm_client() -> None:
    runner = AgnoStage2AgentRunner(
        deterministic_runner=_deterministic_runner(),
        llm_client=_FakeStage2LLMClient(),
        model_name="claude-sonnet",
    )

    outputs = runner.run(
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
    )

    assert runner.backend_name == "agno"
    assert tuple(output.role for output in outputs) == (
        "quant_credit",
        "evidence_audit",
        "chair_report",
    )
    assert outputs[0].quant_summary == "LLM 정량 요약"


def test_agno_stage2_runner_falls_back_to_deterministic_runner_on_error() -> None:
    runner = AgnoStage2AgentRunner(
        deterministic_runner=_deterministic_runner(),
        llm_client=_FailingStage2LLMClient(),
        model_name="claude-sonnet",
    )

    outputs = runner.run(
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
    )

    assert runner.last_run_backend_name == "agno_fallback_deterministic"
    assert "temporary LLM outage" in runner.last_error_message
    assert outputs[0].quant_summary == "정량 요약"


def _deterministic_runner() -> DeterministicStage2AgentRunner:
    return DeterministicStage2AgentRunner(
        quant_credit_agent=lambda _: QuantCreditOutput(
            quant_summary="정량 요약",
            model_rationale="모델 판단 근거",
            key_risk_factors=["위험"],
            mitigating_factors=["완화"],
            confidence=0.8,
        ),
        evidence_audit_agent=lambda _: EvidenceAuditOutput(
            evidence_summary="근거 검토",
            evidence_status="disabled",
            evidence_reliability="신뢰도 점검",
            debt_liquidity_cross_check=["부채 점검"],
            macro_industry_sensitivity=["거시 점검"],
            external_evidence_findings=["외부 근거"],
            confidence=0.6,
        ),
        chair_report_agent=lambda _bundle, _recommendation, confidence: ChairReportOutput(
            report_summary="종합 보고",
            model_preservation_note="model_view 보존",
            committee_scope_note="committee_view 보완",
            final_review_memo_seed="메모 초안",
            confidence=confidence,
        ),
    )


def _minimal_state() -> AgentState:
    return {
        "company_id": "KOSDAQ-000250-2023",
        "company_name": "삼천당제약(주)",
        "market": "KOSDAQ",
        "xgboost_result": {"prediction_label": "투자적격", "probability_speculative": 0.2},
        "rule_result": {"recommendation": "review", "confidence": 0.7},
    }
