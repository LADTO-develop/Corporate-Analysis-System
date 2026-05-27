"""Tests for shared Stage 2 prompt contracts."""

from __future__ import annotations

from cas.agents.stage2_prompt_contracts import (
    all_stage2_prompt_contract_versions,
    build_stage2_llm_client_prompt_payload,
    build_stage2_role_instructions,
    build_stage2_role_query,
    stage2_llm_client_prompt_contract_versions,
    stage2_prompt_contract_versions,
)


def test_stage2_prompt_contract_versions_are_role_scoped() -> None:
    versions = stage2_prompt_contract_versions(("quant_credit", "evidence_audit"))
    all_versions = all_stage2_prompt_contract_versions()

    assert versions["quant_credit"] == "stage2_role_prompt_contract_v2:quant_credit"
    assert versions["evidence_audit"] == "stage2_role_prompt_contract_v2:evidence_audit"
    assert all_versions["review_qa"] == "stage2_role_prompt_contract_v2:review_qa"
    assert all_versions["risk_recall_qa"] == "stage2_role_prompt_contract_v2:risk_recall_qa"


def test_stage2_role_instructions_and_query_include_contract_version() -> None:
    instructions = build_stage2_role_instructions(
        "chair_report",
        provider_label="Gemini",
    )
    query = build_stage2_role_query(
        "review_qa",
        prompt_payload={"company": {"name": "테스트기업"}},
    )

    assert (
        instructions[0] == "You are the CAS ChairReportAgent speaking from the Gemini perspective."
    )
    assert "stage2_role_prompt_contract_v2:chair_report" in instructions[1]
    assert "normalized_signal_summary" in instructions[4]
    assert "stage2_role_prompt_contract_v2:review_qa" in query
    assert "role_checks" in query


def test_llm_client_prompt_payload_records_all_role_contracts() -> None:
    payload = build_stage2_llm_client_prompt_payload(
        recommendation="review",
        confidence=0.7,
        stage2_input_bundle={"company": {"name": "테스트기업"}},
        deterministic_draft_outputs={},
    )
    versions = stage2_llm_client_prompt_contract_versions()

    assert payload["prompt_contract"]["prompt_contract_version"] == ("stage2_llm_client_prompt_v5")
    assert payload["prompt_contract"]["role_prompt_contract_versions"] == {
        key: value for key, value in versions.items() if key != "stage2_llm_client"
    }
    assert "chair_report" in payload["prompt_contract"]["role_prompt_contract_versions"]
