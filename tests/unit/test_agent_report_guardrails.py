"""Regression tests for Stage 2 report guardrails."""

from __future__ import annotations

from typing import NoReturn

import pytest

from cas.agents.nodes import data_node
from cas.agents.nodes.tripletagents import evidence_audit_agent
from cas.agents.nodes.tripletagents.evidence_audit_agent import (
    AgnoEvidenceAuditResponse,
    run_evidence_audit_agent,
)
from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.reporting.export import render_report


def test_dataset_backed_payload_preserves_zero_padded_stock_code() -> None:
    payload = data_node._dataset_backed_payload(
        {
            "__requested_company_id": "005930",
            "corp_name": "삼성전자(주)",
            "stock_code": "5930",
            "market": "KOSPI",
            "fiscal_year": 2023,
            "eval_year": 2024,
            "firm_size_group": "large",
            "industry_macro_category": "manufacturing",
        }
    )

    assert payload["company_id"] == "005930"
    assert payload["processed_company"]["stock_code"] == "005930"
    assert payload["source_feature_row"]["stock_code"] == "005930"


def test_evidence_audit_skips_llm_when_external_evidence_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("LLM agent should not be built when evidence is disabled.")

    monkeypatch.setattr(evidence_audit_agent, "build_agno_agent", fail_if_called)
    bundle = build_stage2_input_bundle(
        {
            "company_id": "005930",
            "company_name": "삼성전자(주)",
            "market": "KOSPI",
            "analysis_year": 2024,
            "xgboost_result": {"prediction_label": "투자적격"},
            "news_cache_snapshot": {"status": "disabled"},
        }
    )

    output = run_evidence_audit_agent(
        bundle=bundle,
        model_name="claude-sonnet",
        max_tokens=1024,
    )

    assert output.evidence_strength == "none"
    assert output.evidence_reliability == "status=disabled; 외부근거 미수집"
    assert "외부근거 미수집" in output.model_challenge


def test_evidence_audit_criticality_requires_structured_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_agno_agent(*args: object, **kwargs: object) -> object:
        return object()

    def fake_run_structured_agent(*args: object, **kwargs: object) -> AgnoEvidenceAuditResponse:
        return AgnoEvidenceAuditResponse(
            macro_environmental_impact="LLM은 외부 충격이 크다고 봅니다.",
            critical_off_balance_risk="LLM은 치명 외부위험이라고 주장합니다.",
            has_critical_risk=True,
            external_risk_level="critical",
            evidence_limitations=["구조화 근거는 watch context 수준입니다."],
        )

    monkeypatch.setattr(evidence_audit_agent, "build_agno_agent", fake_build_agno_agent)
    monkeypatch.setattr(
        evidence_audit_agent,
        "run_structured_agent",
        fake_run_structured_agent,
    )
    bundle = build_stage2_input_bundle(
        {
            "company_id": "005930",
            "company_name": "삼성전자(주)",
            "market": "KOSPI",
            "analysis_year": 2024,
            "xgboost_result": {"prediction_label": "투자적격"},
            "news_cache_snapshot": {
                "status": "ready",
                "items": [
                    {
                        "source": "opendart",
                        "title": "정기보고서 제출",
                        "summary": "정기 공시 제출",
                        "company_match": True,
                        "disclosure_event_class": "routine_context",
                        "disclosure_materiality": "watch_context",
                    }
                ],
            },
        }
    )

    output = run_evidence_audit_agent(
        bundle=bundle,
        model_name="gpt-4.1-mini",
        max_tokens=1024,
    )

    assert output.evidence_strength == "strong"
    assert output.critical_evidence_count == 0
    assert output.watch_context_count == 1
    assert output.hard_distress_detected is False
    assert output.recommended_evidence_treatment == "watch_context"
    assert "structured_critical_evidence=False" in output.evidence_reliability


def test_render_report_uses_korean_headings_and_softens_official_rating_language() -> None:
    rendered = render_report(
        {
            "company_id": "005930",
            "response_json": {
                "company_overview": {
                    "company_id": "005930",
                    "company_name": "삼성전자(주)",
                    "market": "KOSPI",
                    "analysis_year": 2024,
                    "summary": "",
                },
                "model_result": {
                    "model_name": "credit_46_features",
                    "model_version": "test",
                    "prediction_label": "투자적격",
                    "risk_band": "stable",
                    "probability_speculative": 0.001,
                    "top_drivers": [],
                    "rule_label": "투자적격:stable",
                },
                "news_analysis": {
                    "status": "disabled",
                    "summary": "External evidence collection is disabled.",
                },
                "agent_summary": {
                    "final_recommendation": "priority",
                    "final_confidence": 0.67,
                    "synthesis": "삼성전자는 투자적격 등급을 확정합니다.",
                    "agents": {},
                },
                "committee_view": {
                    "final_committee_label": "적격",
                    "veto_triggered": False,
                    "hidden_tail_risk_flag": False,
                    "hidden_tail_risk_reason": "",
                    "conflict_resolution": "최종 승인합니다.",
                    "key_risk_factors": ["현재 scaffold 기준 추가 위험 요인은 제한적입니다."],
                    "mitigating_factors": ["완화 요인은 충분합니다.."],
                    "evidence_summary": [],
                    "final_review_memo": "위원회는 적격로 정리했습니다..",
                },
            },
        }
    )

    markdown = str(rendered["markdown"])
    assert "# 신용위험 검토 보고서" in markdown
    assert "## 위원회 검토 의견" in markdown
    assert "위원회는 적격으로 정리했습니다." in markdown
    assert "투자적격 검토 의견을 제시합니다" in markdown
    assert "최종 승인" not in markdown


def test_render_report_includes_stage2_runtime_and_decision_trace() -> None:
    rendered = render_report(
        {
            "company_id": "005930",
            "response_json": {
                "company_overview": {
                    "company_id": "005930",
                    "company_name": "삼성전자(주)",
                    "market": "KOSPI",
                    "analysis_year": 2024,
                    "summary": "",
                },
                "model_result": {
                    "model_name": "credit_46_features",
                    "model_version": "test",
                    "prediction_label": "투자적격",
                    "risk_band": "stable",
                    "probability_speculative": 0.001,
                    "top_drivers": [],
                    "rule_label": "투자적격:stable",
                },
                "news_analysis": {
                    "status": "disabled",
                    "summary": "External evidence collection is disabled.",
                },
                "agent_summary": {
                    "final_recommendation": "review",
                    "final_confidence": 0.67,
                    "synthesis": "위원회 종합 의견",
                    "agents": {},
                    "runtime": {
                        "backend_name": "agno",
                        "cache_hit": False,
                        "fallback_used": False,
                        "stage2_total_elapsed_seconds": 12.345,
                        "review_qa_triggered": True,
                        "review_qa_advisory_applied": True,
                        "review_qa_advisory_apply_reason": "watch_context_only_risk_hold_override",
                        "risk_recall_qa_triggered": False,
                        "risk_recall_qa_advisory_applied": False,
                    },
                },
                "committee_view": {
                    "final_committee_label": "보류",
                    "veto_triggered": False,
                    "hidden_tail_risk_flag": False,
                    "hidden_tail_risk_reason": "",
                    "conflict_resolution": "경계 보류로 정리했습니다.",
                    "key_risk_factors": ["기준선 근처"],
                    "mitigating_factors": ["방어축 확인"],
                    "evidence_summary": [],
                    "decision_trace": [
                        {
                            "gate": "boundary_rating_review",
                            "label": "경계등급 점검",
                            "triggered": True,
                            "severity": "watch",
                            "summary": "기준선 근처라 보류했습니다.",
                        }
                    ],
                    "final_review_memo": "최종 보류 의견입니다.",
                },
            },
        }
    )

    markdown = str(rendered["markdown"])
    assert "## Stage 2 실행 진단" in markdown
    assert "**Backend**: `agno`" in markdown
    assert "ReviewQA" in markdown
    assert "### 결정 추적" in markdown
    assert "경계등급 점검" in markdown
