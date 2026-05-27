"""Tests for versioned Stage 2 policy thresholds."""

from __future__ import annotations

from cas.agents.nodes.qa_cache import _review_qa_cache_payload, _risk_recall_qa_cache_payload
from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.stage2_policy import load_stage2_policy, stage2_policy_version


def test_stage2_policy_loads_versioned_thresholds_from_yaml() -> None:
    policy = load_stage2_policy()

    assert policy.policy_version == "stage2_policy_v1"
    assert policy.float("risk_recall_qa", "trigger", "debt_ratio_floor") == 2.0
    assert policy.float("committee_guardrails", "secondary_review", "threshold_buffer") == 0.10
    assert (
        policy.float(
            "committee_guardrails",
            "cashflow_backed_fp_resilience",
            "debt_ratio_ceiling",
        )
        == 1.50
    )
    assert policy.float("review_qa", "extreme_distress", "cashflow_coverage_ratio_floor") == 0.0
    assert policy.int("review_qa", "boundary_defense", "min_defensive_axes") == 3


def test_post_committee_qa_cache_payloads_include_policy_version() -> None:
    bundle = build_stage2_input_bundle(
        {
            "company_id": "KOSDAQ-000250-2023",
            "company_name": "삼천당제약(주)",
            "market": "KOSDAQ",
            "xgboost_result": {
                "prediction_label": "투자적격",
                "probability_speculative": 0.2,
                "threshold": 0.3,
            },
        }
    )
    quant = QuantCreditOutput(
        quant_summary="정량 요약",
        model_rationale="모델 근거",
        key_risk_factors=["위험"],
        mitigating_factors=["완화"],
        confidence=0.8,
    )
    evidence = EvidenceAuditOutput(
        evidence_summary="근거 요약",
        evidence_status="disabled",
        evidence_reliability="근거 제한",
        evidence_strength="none",
        model_challenge="충돌 제한",
        audit_conclusion="추가 근거 제한",
        debt_liquidity_cross_check=[],
        macro_industry_sensitivity=[],
        external_evidence_findings=[],
        confidence=0.6,
    )
    chair = ChairReportOutput(
        report_summary="종합",
        model_preservation_note="model_view 보존",
        committee_scope_note="committee_view 보완",
        final_review_memo_seed="메모",
        confidence=0.7,
    )

    review_payload = _review_qa_cache_payload(
        bundle=bundle,
        committee_view={"final_committee_label": "보류"},
        quant_credit=quant,
        evidence_audit=evidence,
        chair_report=chair,
        trigger_reasons=["risk_hold_without_critical_evidence"],
        model_provider="openai",
        model_name="gpt-4.1-mini",
    )
    recall_payload = _risk_recall_qa_cache_payload(
        bundle=bundle,
        committee_view={"final_committee_label": "적격"},
        quant_credit=quant,
        evidence_audit=evidence,
        chair_report=chair,
        trigger_reasons=["eligible_near_threshold"],
        model_provider="openai",
        model_name="gpt-4.1-mini",
    )

    assert review_payload["stage2_policy_version"] == stage2_policy_version()
    assert recall_payload["stage2_policy_version"] == stage2_policy_version()
    assert review_payload["prompt_contract_version"] == "stage2_role_prompt_contract_v2:review_qa"
    assert recall_payload["prompt_contract_version"] == (
        "stage2_role_prompt_contract_v2:risk_recall_qa"
    )
