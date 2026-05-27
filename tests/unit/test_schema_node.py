"""Tests for dashboard schema node helper wording."""

from __future__ import annotations

from cas.agents.nodes.schema_node import _news_summary, run


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


def test_schema_response_exposes_stage2_runtime_and_decision_trace() -> None:
    result = run(
        {
            "company_id": "005930",
            "company_name": "삼성전자(주)",
            "market": "KOSPI",
            "analysis_year": 2024,
            "company_profile": {},
            "xgboost_result": {
                "model_name": "feature_46_xgboost",
                "model_version": "test",
                "prediction_label": "투자적격",
                "risk_band": "stable",
                "probability_speculative": 0.12,
                "top_drivers": [],
            },
            "rule_result": {"label": "투자적격:stable", "risk_band": "stable"},
            "news_cache_snapshot": {"status": "disabled", "items": []},
            "agent_summary": {
                "final_recommendation": "priority",
                "final_confidence": 0.82,
                "synthesis": "위원회 종합 의견",
                "agents": {
                    "quant_credit": {
                        "summary": "정량 검토",
                        "findings": ["모델 원판단 보존"],
                        "confidence": 0.8,
                    }
                },
                "runtime": {
                    "backend_name": "agno_fallback_deterministic",
                    "cache_hit": False,
                    "error_message": "provider timeout",
                    "stage2_total_elapsed_seconds": 3.25,
                    "agent_elapsed_seconds": {"quant_credit": 1.2},
                    "review_qa_triggered": True,
                    "review_qa_trigger_reasons": ["risk_hold_without_critical_evidence"],
                    "review_qa_advisory_applied": True,
                    "review_qa_advisory_apply_reason": "review_qa_overstated_risk_hold",
                    "risk_recall_qa_triggered": False,
                },
            },
            "committee_view": {
                "final_committee_label": "보류",
                "committee_decision_type": "boundary_hold",
                "committee_decision_type_label": "경계등급 보류",
                "committee_risk_signal": False,
                "risk_hold_reason_tags": [],
                "risk_hold_reason_labels": [],
                "risk_hold_reason_summary": "",
                "veto_triggered": False,
                "hidden_tail_risk_flag": False,
                "hidden_tail_risk_reason": "",
                "conflict_resolution": "경계 보류로 조정했습니다.",
                "key_risk_factors": ["기준선 근처"],
                "mitigating_factors": ["현금흐름 방어"],
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
        }
    )

    response = result["response_json"]
    runtime = response["agent_summary"]["runtime"]
    assert runtime["backend_name"] == "agno_fallback_deterministic"
    assert runtime["fallback_used"] is True
    assert runtime["fallback_reason"] == "provider timeout"
    assert runtime["review_qa_triggered"] is True
    assert runtime["review_qa_advisory_applied"] is True
    assert runtime["review_qa_advisory_apply_reason"] == "review_qa_overstated_risk_hold"
    assert response["committee_view"]["decision_trace"][0]["gate"] == "boundary_rating_review"
    assert response["committee_view"]["decision_trace"][0]["severity"] == "watch"
    assert result["json_schema_errors"] == []
