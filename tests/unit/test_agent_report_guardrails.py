"""Regression tests for Stage 2 report guardrails."""

from __future__ import annotations

from typing import NoReturn

import pytest

from cas.agents.nodes import data_node
from cas.agents.nodes.tripletagents import evidence_audit_agent
from cas.agents.nodes.tripletagents.evidence_audit_agent import run_evidence_audit_agent
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
                    "model_name": "credit_44_features",
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
