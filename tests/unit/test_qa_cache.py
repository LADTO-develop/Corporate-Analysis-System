"""Tests for shared post-committee QA cache runner."""

from __future__ import annotations

from pathlib import Path

from cas.agents.nodes.qa_cache import run_cached_optional_agent
from cas.agents.stage2_outputs import ReviewQAOutput


def test_run_cached_optional_agent_reuses_cached_response(tmp_path: Path) -> None:
    calls = 0

    def payload_builder(model_provider: str, model_name: str) -> dict[str, object]:
        return {
            "cache_version": "unit_cache_v1",
            "model_provider": model_provider,
            "model_name": model_name,
            "prompt": {"company_id": "KOSDAQ-000250-2023"},
        }

    def agent_callable(usage: dict[str, object]) -> ReviewQAOutput:
        nonlocal calls
        calls += 1
        usage.update({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})
        return _review_qa_output("fresh QA response")

    cache_env = {
        "CAS_STAGE2_LLM_CACHE_ENABLED": "1",
        "CAS_STAGE2_CACHE_DIR": str(tmp_path),
    }
    first, first_diagnostics = run_cached_optional_agent(
        role="review_qa",
        cache_namespace="unit_review_qa",
        cache_version="unit_cache_v1",
        cache_env=cache_env,
        payload_builder=payload_builder,
        agent_callable=agent_callable,
        schema=ReviewQAOutput,
        model_provider="openai",
        model_name="gpt-4.1-mini",
    )
    second, second_diagnostics = run_cached_optional_agent(
        role="review_qa",
        cache_namespace="unit_review_qa",
        cache_version="unit_cache_v1",
        cache_env=cache_env,
        payload_builder=payload_builder,
        agent_callable=agent_callable,
        schema=ReviewQAOutput,
        model_provider="openai",
        model_name="gpt-4.1-mini",
    )

    assert calls == 1
    assert first.qa_summary == "fresh QA response"
    assert second.qa_summary == "fresh QA response"
    assert first_diagnostics["review_qa_cache_hit"] is False
    assert second_diagnostics["review_qa_cache_hit"] is True
    assert first_diagnostics["review_qa_cache_key"] == second_diagnostics["review_qa_cache_key"]
    assert set(second_diagnostics["agent_elapsed_seconds"]) == {"review_qa"}
    assert first_diagnostics["role_token_usage"]["review_qa"]["billable_total_tokens"] == 120
    assert second_diagnostics["role_token_usage"]["review_qa"]["billable_total_tokens"] == 0


def _review_qa_output(summary: str) -> ReviewQAOutput:
    return ReviewQAOutput(
        qa_summary=summary,
        trigger_reasons=["unit_trigger"],
        label_memo_consistency="consistent",
        risk_hold_assessment="adequate",
        evidence_cutoff_check="ok",
        overhold_guardrail_assessment="ok",
        recommended_action="keep_committee_view",
        confidence=0.8,
    )
