"""Tests for Stage 2 runner adapters."""

from __future__ import annotations

from typing import Any

import pytest

from cas.agents import stage2_runner as stage2_runner_module
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
                evidence_strength="moderate",
                model_challenge="LLM 모델-근거 충돌 점검",
                audit_conclusion="LLM EvidenceAudit 결론",
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
            evidence_strength="none",
            model_challenge="중대한 충돌은 제한적입니다.",
            audit_conclusion="모델 원판단을 설명하는 보완 의견입니다.",
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


def test_agno_stage2_runner_uses_triplet_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_triplet_agents(**kwargs: Any) -> Stage2LLMResponse:
        assert kwargs["model_name"] == "claude-sonnet"
        return Stage2LLMResponse(
            quant_credit=QuantCreditOutput(
                quant_summary="Triplet quant summary",
                model_rationale="Triplet model rationale",
                key_risk_factors=["Triplet risk"],
                mitigating_factors=["Triplet mitigation"],
                confidence=0.77,
            ),
            evidence_audit=EvidenceAuditOutput(
                evidence_summary="Triplet evidence summary",
                evidence_status="ready",
                evidence_reliability="Triplet reliability",
                evidence_strength="moderate",
                model_challenge="Triplet challenge",
                audit_conclusion="Triplet conclusion",
                debt_liquidity_cross_check=["Triplet debt check"],
                macro_industry_sensitivity=["Triplet macro check"],
                external_evidence_findings=["Triplet evidence"],
                confidence=0.72,
            ),
            chair_report=ChairReportOutput(
                report_summary="Triplet chair summary",
                model_preservation_note="Triplet model preservation",
                committee_scope_note="Triplet scope",
                final_review_memo_seed="Triplet memo",
                confidence=0.74,
            ),
        )

    monkeypatch.setattr(
        stage2_runner_module,
        "_run_triplet_agents_with_agno",
        fake_triplet_agents,
    )
    runner = AgnoStage2AgentRunner(
        deterministic_runner=_deterministic_runner(),
        model_name="claude-sonnet",
    )

    outputs = runner.run(
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
    )

    assert runner.last_run_backend_name == "agno"
    assert outputs[0].quant_summary == "Triplet quant summary"
    assert outputs[2].report_summary == "Triplet chair summary"


def test_agno_stage2_runner_routes_multi_llm_committee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_triplet_agents(**kwargs: Any) -> Stage2LLMResponse:
        captured.update(kwargs)
        return Stage2LLMResponse(
            quant_credit=QuantCreditOutput(
                quant_summary="Claude quant summary",
                model_rationale="Claude model rationale",
                key_risk_factors=["Claude risk"],
                mitigating_factors=["Claude mitigation"],
                confidence=0.77,
            ),
            evidence_audit=EvidenceAuditOutput(
                evidence_summary="GPT evidence summary",
                evidence_status="ready",
                evidence_reliability="GPT reliability",
                evidence_strength="moderate",
                model_challenge="GPT challenge",
                audit_conclusion="GPT conclusion",
                debt_liquidity_cross_check=["GPT debt check"],
                macro_industry_sensitivity=["GPT macro check"],
                external_evidence_findings=["GPT evidence"],
                confidence=0.72,
            ),
            chair_report=ChairReportOutput(
                report_summary="Gemini chair summary",
                model_preservation_note="Gemini model preservation",
                committee_scope_note="Gemini scope",
                final_review_memo_seed="Gemini memo",
                confidence=0.74,
            ),
        )

    monkeypatch.setattr(
        stage2_runner_module,
        "_run_triplet_agents_with_agno",
        fake_triplet_agents,
    )
    runner = AgnoStage2AgentRunner(
        deterministic_runner=_deterministic_runner(),
        routing_mode="multi_llm_committee",
        model_name="claude-sonnet",
    )

    outputs = runner.run(
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
    )

    assert captured["quant_model_provider"] == "anthropic"
    assert captured["quant_model_name"] == "claude-sonnet"
    assert captured["evidence_model_provider"] == "openai"
    assert captured["evidence_model_name"] == "gpt-5.4-mini"
    assert captured["chair_model_provider"] == "google"
    assert captured["chair_model_name"] == "gemini-flash-latest"
    assert outputs[2].report_summary == "Gemini chair summary"


def test_agno_stage2_runner_falls_back_when_triplet_agents_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_triplet_agents(**_kwargs: Any) -> Stage2LLMResponse:
        raise RuntimeError("missing agno runtime")

    monkeypatch.setattr(
        stage2_runner_module,
        "_run_triplet_agents_with_agno",
        fail_triplet_agents,
    )
    runner = AgnoStage2AgentRunner(
        deterministic_runner=_deterministic_runner(),
        model_name="claude-sonnet",
    )

    outputs = runner.run(
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
    )

    assert runner.last_run_backend_name == "agno_fallback_deterministic"
    assert "missing agno runtime" in runner.last_error_message
    assert outputs[0].role == "quant_credit"


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


def test_agno_stage2_runner_raises_when_fallback_is_disabled() -> None:
    runner = AgnoStage2AgentRunner(
        deterministic_runner=_deterministic_runner(),
        llm_client=_FailingStage2LLMClient(),
        model_name="claude-sonnet",
        fallback_on_error=False,
    )

    with pytest.raises(RuntimeError, match="temporary LLM outage"):
        runner.run(
            bundle=build_stage2_input_bundle(_minimal_state()),
            recommendation="review",
            confidence=0.7,
        )


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
            evidence_strength="none",
            model_challenge="중대한 충돌은 제한적입니다.",
            audit_conclusion="모델 원판단을 설명하는 보완 의견입니다.",
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
