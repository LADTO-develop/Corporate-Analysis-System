"""Tests for dashboard schema node helper wording."""

from __future__ import annotations

from cas.agents.nodes.schema_node import _news_summary


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
