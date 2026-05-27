"""Tests for dashboard schema node helper wording."""

from __future__ import annotations

from cas.agents.nodes.schema_node import _news_summary
from cas.agents.nodes.schema_node import run as schema_node_run
from cas.agents.response_schema import DashboardResponse
from cas.agents.state import AgentState


def test_news_summary_does_not_overstate_weak_keyword_hits() -> None:
    summary = _news_summary(
        {
            "status": "ready",
            "items": [
                {
                    "source": "naver",
                    "title": "타사 횡령 보도",
                    "company_match": False,
                    "critical_terms": ["횡령"],
                    "veto_candidate": False,
                }
            ],
            "has_critical_risk": True,
            "critical_terms": ["횡령"],
            "direct_match_count": 0,
            "weak_evidence_count": 1,
        },
        insufficient=False,
        company_name="삼천당제약(주)",
        stock_code="000250",
    )

    assert "unconfirmed keyword hit" in summary
    assert "no high-confidence direct veto evidence" in summary
    assert "0 direct-match, 1 weak/indirect" in summary


def test_news_summary_reports_direct_confirmed_veto_evidence() -> None:
    summary = _news_summary(
        {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "삼천당제약 횡령 공시",
                    "summary": "삼천당제약 공시: 횡령",
                    "reliability": "high",
                    "company_match": True,
                    "critical_terms": ["횡령"],
                    "veto_candidate": True,
                },
                {
                    "source": "naver_news",
                    "title": "삼천당제약 횡령 관련 보도",
                    "summary": "삼천당제약 횡령 관련 보도",
                    "reliability": "medium",
                    "company_match": True,
                    "critical_terms": ["횡령"],
                    "veto_candidate": True,
                },
            ],
            "has_critical_risk": True,
            "critical_terms": ["횡령"],
            "direct_match_count": 2,
            "weak_evidence_count": 0,
        },
        insufficient=False,
        company_name="삼천당제약(주)",
        stock_code="000250",
    )

    assert "confirmed high-confidence critical evidence" in summary
    assert "2 direct-match, 0 weak/indirect" in summary


def test_schema_node_exposes_stage2_runtime_diagnostics() -> None:
    state: AgentState = {
        "company_id": "KOSDAQ-000250-2023",
        "company_name": "삼천당제약(주)",
        "market": "KOSDAQ",
        "analysis_year": 2024,
        "company_profile": {"company": {"name": "삼천당제약(주)", "market": "KOSDAQ"}},
        "xgboost_result": {
            "model_name": "credit_46_features",
            "model_version": "unit",
            "prediction_label": "투자적격",
            "risk_band": "stable",
            "probability_speculative": 0.2,
            "top_drivers": [],
        },
        "rule_result": {"label": "eligible", "risk_band": "stable"},
        "news_cache_snapshot": {"status": "disabled", "items": []},
        "agent_summary": {
            "final_recommendation": "review",
            "final_confidence": 0.7,
            "synthesis": "위원회 검토 요약",
            "agents": {
                "quant_credit": {
                    "summary": "정량 요약",
                    "findings": ["정량 finding"],
                    "confidence": 0.7,
                }
            },
            "runtime": {
                "backend_name": "agno",
                "degraded": True,
                "failed_role": "evidence_audit",
                "failed_roles": ["evidence_audit"],
                "retry_count": 2,
                "role_fallback_used": {"evidence_audit": True},
            },
        },
        "committee_view": {
            "final_committee_label": "보류",
            "committee_decision_type": "review_hold",
            "committee_decision_type_label": "확인필요 보류",
            "committee_risk_signal": False,
            "risk_hold_reason_tags": [],
            "risk_hold_reason_labels": [],
            "risk_hold_reason_summary": "",
            "agent_disagreement_score": 0.0,
            "agent_disagreement_level": "low",
            "agent_disagreement_reasons": [],
            "agent_disagreement_summary": "",
            "veto_triggered": False,
            "hidden_tail_risk_flag": False,
            "hidden_tail_risk_reason": "",
            "conflict_resolution": "검토 보류",
            "key_risk_factors": [],
            "mitigating_factors": [],
            "evidence_summary": [],
            "manual_review_tasks": ["직접 공시를 확인합니다."],
            "missing_evidence": ["기준일 이전 공시 원문"],
            "monitoring_triggers": ["신규 DART 공시"],
            "decision_trace": [],
            "final_review_memo": "검토 보류",
        },
        "insufficient_data": False,
    }

    result = schema_node_run(state)

    DashboardResponse.model_validate(result["response_json"])
    runtime = result["response_json"]["agent_summary"]["runtime"]
    assert runtime["degraded"] is True
    assert runtime["failed_role"] == "evidence_audit"
    assert runtime["retry_count"] == 2
    committee_view = result["response_json"]["committee_view"]
    assert committee_view["manual_review_tasks"] == ["직접 공시를 확인합니다."]
    assert committee_view["missing_evidence"] == ["기준일 이전 공시 원문"]
    assert committee_view["monitoring_triggers"] == ["신규 DART 공시"]
