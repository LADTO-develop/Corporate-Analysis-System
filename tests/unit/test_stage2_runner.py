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
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
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


def test_agno_stage2_runner_defaults_to_openai_single_model() -> None:
    runner = AgnoStage2AgentRunner(deterministic_runner=_deterministic_runner())

    assert runner.routing_mode == "single"
    assert runner.model_provider == "openai"
    assert runner.model_name == "gpt-4.1-mini"


def test_agno_stage2_runner_uses_triplet_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAS_STAGE2_LLM_CACHE_ENABLED", "0")

    def fake_triplet_agents(**kwargs: Any) -> Stage2LLMResponse:
        assert kwargs["model_name"] == "claude-sonnet"
        diagnostics = kwargs.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["agent_elapsed_seconds"] = {
                "quant_credit": 1.2,
                "evidence_audit": 2.3,
                "chair_report": 0.8,
            }
            diagnostics["parallel_independent_agents"] = True
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
    assert runner.last_run_diagnostics["cache_hit"] is False
    assert runner.last_run_diagnostics["backend_name"] == "agno"
    assert runner.last_run_diagnostics["agent_elapsed_seconds"]["evidence_audit"] == 2.3
    assert runner.last_run_diagnostics["agent_elapsed_seconds_sum"] == 4.3
    assert outputs[0].quant_summary == "Triplet quant summary"
    assert outputs[2].report_summary == "Triplet chair summary"


def test_agno_stage2_runner_reuses_cached_triplet_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    calls = 0

    def fake_triplet_agents(**_kwargs: Any) -> Stage2LLMResponse:
        nonlocal calls
        calls += 1
        return Stage2LLMResponse(
            quant_credit=QuantCreditOutput(
                quant_summary="Cached quant summary",
                model_rationale="Cached model rationale",
                key_risk_factors=["Cached risk"],
                mitigating_factors=["Cached mitigation"],
                confidence=0.77,
            ),
            evidence_audit=EvidenceAuditOutput(
                evidence_summary="Cached evidence summary",
                evidence_status="ready",
                evidence_reliability="Cached reliability",
                evidence_strength="moderate",
                model_challenge="Cached challenge",
                audit_conclusion="Cached conclusion",
                debt_liquidity_cross_check=["Cached debt check"],
                macro_industry_sensitivity=["Cached macro check"],
                external_evidence_findings=["Cached evidence"],
                confidence=0.72,
            ),
            chair_report=ChairReportOutput(
                report_summary="Cached chair summary",
                model_preservation_note="Cached model preservation",
                committee_scope_note="Cached scope",
                final_review_memo_seed="Cached memo",
                confidence=0.74,
            ),
        )

    monkeypatch.setenv("CAS_STAGE2_LLM_CACHE_ENABLED", "1")
    monkeypatch.setenv("CAS_STAGE2_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        stage2_runner_module,
        "_run_triplet_agents_with_agno",
        fake_triplet_agents,
    )
    runner = AgnoStage2AgentRunner(
        deterministic_runner=_deterministic_runner(),
        model_name="claude-sonnet",
    )
    bundle = build_stage2_input_bundle(_minimal_state())

    first_outputs = runner.run(bundle=bundle, recommendation="review", confidence=0.7)
    second_outputs = runner.run(bundle=bundle, recommendation="review", confidence=0.7)

    assert calls == 1
    assert first_outputs[0].quant_summary == "Cached quant summary"
    assert second_outputs[0].quant_summary == "Cached quant summary"
    assert runner.last_run_backend_name == "agno_cache"
    assert runner.last_run_diagnostics["cache_hit"] is True
    assert runner.last_run_diagnostics["agent_elapsed_seconds"] == {}


def test_stage2_cache_payload_includes_policy_version() -> None:
    runner = AgnoStage2AgentRunner(deterministic_runner=_deterministic_runner())
    payload = stage2_runner_module._stage2_cache_payload(
        runner=runner,
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
    )

    assert payload["stage2_policy_version"] == "stage2_policy_v1"
    assert payload["prompt_contract_versions"] == {
        "quant_credit": "stage2_role_prompt_contract_v2:quant_credit",
        "evidence_audit": "stage2_role_prompt_contract_v2:evidence_audit",
        "chair_report": "stage2_role_prompt_contract_v2:chair_report",
    }


def test_agno_stage2_runner_routes_multi_llm_committee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAS_STAGE2_LLM_CACHE_ENABLED", "0")
    captured: dict[str, Any] = {}

    def fake_triplet_agents(**kwargs: Any) -> Stage2LLMResponse:
        captured.update(kwargs)
        return Stage2LLMResponse(
            quant_credit=QuantCreditOutput(
                quant_summary="Gemini quant summary",
                model_rationale="Gemini model rationale",
                key_risk_factors=["Gemini risk"],
                mitigating_factors=["Gemini mitigation"],
                confidence=0.77,
            ),
            evidence_audit=EvidenceAuditOutput(
                evidence_summary="Claude evidence summary",
                evidence_status="ready",
                evidence_reliability="Claude reliability",
                evidence_strength="moderate",
                model_challenge="Claude challenge",
                audit_conclusion="Claude conclusion",
                debt_liquidity_cross_check=["Claude debt check"],
                macro_industry_sensitivity=["Claude macro check"],
                external_evidence_findings=["Claude evidence"],
                confidence=0.72,
            ),
            chair_report=ChairReportOutput(
                report_summary="OpenAI chair summary",
                model_preservation_note="OpenAI model preservation",
                committee_scope_note="OpenAI scope",
                final_review_memo_seed="OpenAI memo",
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
    )

    outputs = runner.run(
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
    )

    assert captured["quant_model_provider"] == "google"
    assert captured["quant_model_name"] == "gemini-2.5-flash"
    assert captured["evidence_model_provider"] == "anthropic"
    assert captured["evidence_model_name"] == "claude-sonnet-4-6"
    assert captured["chair_model_provider"] == "openai"
    assert captured["chair_model_name"] == "gpt-4.1-mini"
    assert outputs[2].report_summary == "OpenAI chair summary"


def test_agno_stage2_runner_falls_back_when_triplet_agents_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CAS_STAGE2_LLM_CACHE_ENABLED", "0")

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


def test_triplet_agents_write_per_run_runtime_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cas.agents.nodes import tripletagents

    monkeypatch.setattr(
        tripletagents,
        "run_quant_credit_agent",
        lambda **_kwargs: QuantCreditOutput(
            quant_summary="정량 요약",
            model_rationale="모델 판단 근거",
            key_risk_factors=["위험"],
            mitigating_factors=["완화"],
            confidence=0.8,
        ),
    )
    monkeypatch.setattr(
        tripletagents,
        "run_evidence_audit_agent",
        lambda **_kwargs: EvidenceAuditOutput(
            evidence_summary="근거 검토",
            evidence_status="ready",
            evidence_reliability="신뢰도 점검",
            evidence_strength="moderate",
            model_challenge="중대한 충돌은 제한적입니다.",
            audit_conclusion="모델 원판단을 설명하는 보완 의견입니다.",
            debt_liquidity_cross_check=["부채 점검"],
            macro_industry_sensitivity=["거시 점검"],
            external_evidence_findings=["외부 근거"],
            confidence=0.6,
        ),
    )
    monkeypatch.setattr(
        tripletagents,
        "run_chair_report_agent",
        lambda **_kwargs: ChairReportOutput(
            report_summary="종합 보고",
            model_preservation_note="model_view 보존",
            committee_scope_note="committee_view 보완",
            final_review_memo_seed="메모 초안",
            confidence=0.7,
        ),
    )
    diagnostics: dict[str, object] = {}

    outputs = tripletagents.run_triplet_agents(
        bundle=build_stage2_input_bundle(_minimal_state()),
        recommendation="review",
        confidence=0.7,
        model_provider="openai",
        model_name="test-model",
        max_tokens=100,
        runtime_config=Stage2RuntimeConfig(parallel_independent_agents=False),
        diagnostics=diagnostics,
    )

    assert tuple(output.role for output in outputs) == (
        "quant_credit",
        "evidence_audit",
        "chair_report",
    )
    assert diagnostics["parallel_independent_agents"] is False
    assert set(diagnostics["agent_elapsed_seconds"]) == {
        "quant_credit",
        "evidence_audit",
        "chair_report",
    }


def test_agno_runtime_passes_openai_timeout_and_provider_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cas.agents.nodes.tripletagents import runtime

    captured: dict[str, Any] = {}

    class FakeOpenAIResponses:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class FakeOpenAIModule:
        OpenAIResponses = FakeOpenAIResponses

    def fake_import_module(name: str) -> object:
        if name == "agno.models.openai":
            return FakeOpenAIModule()
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CAS_STAGE2_AGENT_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CAS_STAGE2_PROVIDER_MAX_RETRIES", "1")
    monkeypatch.setattr(runtime, "import_module", fake_import_module)

    runtime._build_agno_model(
        provider="openai",
        model_name="gpt-test",
        max_tokens=200,
    )

    assert captured["id"] == "gpt-test"
    assert captured["timeout"] == 30.0
    assert captured["max_retries"] == 1
    assert captured["api_key"] == "test-key"


def test_agno_runtime_uses_explicit_runtime_config_without_stage2_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cas.agents.nodes.tripletagents import runtime

    captured: dict[str, Any] = {}

    class FakeOpenAIResponses:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class FakeOpenAIModule:
        OpenAIResponses = FakeOpenAIResponses

    def fake_import_module(name: str) -> object:
        if name == "agno.models.openai":
            return FakeOpenAIModule()
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("CAS_STAGE2_AGENT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CAS_STAGE2_PROVIDER_MAX_RETRIES", raising=False)
    monkeypatch.setattr(runtime, "import_module", fake_import_module)

    runtime._build_agno_model(
        provider="openai",
        model_name="gpt-test",
        max_tokens=200,
        runtime_config=Stage2RuntimeConfig(
            agent_timeout_seconds=45.0,
            provider_max_retries=2,
        ),
    )

    assert captured["timeout"] == 45.0
    assert captured["max_retries"] == 2


def test_agno_runtime_passes_gemini_timeout_and_provider_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cas.agents.nodes.tripletagents import runtime

    captured: dict[str, Any] = {}

    class FakeGemini:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class FakeGoogleModule:
        Gemini = FakeGemini

    def fake_import_module(name: str) -> object:
        if name == "agno.models.google":
            return FakeGoogleModule()
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setattr(runtime, "import_module", fake_import_module)

    runtime._build_agno_model(
        provider="google",
        model_name="gemini-test",
        max_tokens=200,
        runtime_config=Stage2RuntimeConfig(
            agent_timeout_seconds=40.0,
            agent_retry_delay_seconds=2.4,
            provider_max_retries=3,
        ),
    )

    assert captured["id"] == "gemini-test"
    assert captured["timeout"] == 40.0
    assert captured["retries"] == 3
    assert captured["delay_between_retries"] == 2
    assert captured["api_key"] == "test-google-key"


def test_agno_runtime_timeout_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cas.agents.nodes.tripletagents import runtime

    monkeypatch.setenv("CAS_STAGE2_AGENT_TIMEOUT_SECONDS", "off")

    assert runtime._stage2_agent_timeout_seconds() is None


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
