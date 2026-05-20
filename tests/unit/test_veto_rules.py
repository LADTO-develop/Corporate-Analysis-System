"""Tests for config-backed committee veto rules."""

from __future__ import annotations

from cas.veto_rules import (
    critical_terms_in_text,
    external_evidence_veto_triggered,
    flag_contains_veto_marker,
    load_veto_rules,
)


def test_veto_rules_load_from_committee_config() -> None:
    rules = load_veto_rules()

    assert rules.enabled is True
    assert rules.triggered_label == "부적격"
    assert "fraud" in rules.blocking_flag_markers
    assert "횡령" in rules.external_evidence_terms


def test_flag_contains_veto_marker_uses_configured_terms() -> None:
    rules = load_veto_rules()

    assert flag_contains_veto_marker("fraud_risk", rules=rules)
    assert flag_contains_veto_marker("횡령_공시", rules=rules)
    assert not flag_contains_veto_marker("interest_coverage_under_1", rules=rules)


def test_critical_terms_in_text_uses_configured_terms() -> None:
    terms = critical_terms_in_text("회사 관련 횡령 의혹과 상장폐지 가능성이 보도되었습니다.")

    assert "횡령" in terms
    assert "상장폐지" in terms


def test_external_evidence_veto_rejects_keyword_only_false_positive() -> None:
    snapshot = {
        "status": "ready",
        "has_critical_risk": True,
        "critical_terms": ["횡령"],
        "items": [
            {
                "source": "tavily",
                "title": "횡령 배임 공시 안내",
                "summary": "일반 공시 검색 안내 페이지입니다.",
                "reliability": "medium",
                "company_match": False,
                "critical_terms": ["횡령", "배임"],
            }
        ],
    }

    assert not external_evidence_veto_triggered(
        snapshot,
        company_name="삼천당제약(주)",
        stock_code="000250",
    )


def test_external_evidence_veto_requires_direct_multi_source_high_reliability() -> None:
    snapshot = {
        "status": "ready",
        "has_critical_risk": True,
        "critical_terms": ["횡령"],
        "items": [
            {
                "source": "opendart",
                "title": "삼천당제약(주) 횡령 혐의 발생",
                "summary": "삼천당제약(주) 공시: 횡령 혐의 발생",
                "reliability": "high",
                "company_match": True,
                "critical_terms": ["횡령"],
            },
            {
                "source": "naver_news",
                "title": "삼천당제약 횡령 관련 보도",
                "summary": "삼천당제약 관련 후속 보도입니다.",
                "reliability": "medium",
                "company_match": True,
                "critical_terms": ["횡령"],
            },
        ],
    }

    assert external_evidence_veto_triggered(
        snapshot,
        company_name="삼천당제약(주)",
        stock_code="000250",
    )


def test_external_evidence_veto_ignores_routine_or_caution_disclosures() -> None:
    snapshot = {
        "status": "ready",
        "has_critical_risk": True,
        "critical_terms": ["감사의견"],
        "items": [
            {
                "source": "opendart",
                "title": "삼천당제약(주) 감사보고서 제출",
                "summary": "삼천당제약(주) 정기 외부감사 관련 공시입니다.",
                "reliability": "high",
                "company_match": True,
                "critical_terms": ["감사의견"],
                "disclosure_severity": "caution",
            },
            {
                "source": "naver_news",
                "title": "삼천당제약 감사보고서 관련 보도",
                "summary": "삼천당제약 관련 후속 보도입니다.",
                "reliability": "medium",
                "company_match": True,
                "critical_terms": ["감사의견"],
            },
        ],
    }

    assert not external_evidence_veto_triggered(
        snapshot,
        company_name="삼천당제약(주)",
        stock_code="000250",
    )


def test_external_evidence_veto_rejects_unconfirmed_keyword_context() -> None:
    snapshot = {
        "status": "ready",
        "has_critical_risk": True,
        "critical_terms": ["횡령"],
        "items": [
            {
                "source": "opendart",
                "title": "삼천당제약(주) 횡령 혐의 발생",
                "summary": "삼천당제약(주) 공시입니다.",
                "reliability": "high",
                "company_match": True,
                "critical_terms": ["횡령"],
                "critical_context_confirmed": False,
            },
            {
                "source": "naver_news",
                "title": "삼천당제약 관련 기사",
                "summary": "다른 회사 임원이 횡령 혐의로 기소되었습니다.",
                "reliability": "medium",
                "company_match": True,
                "critical_terms": ["횡령"],
                "critical_context_confirmed": False,
            },
        ],
    }

    assert not external_evidence_veto_triggered(
        snapshot,
        company_name="삼천당제약(주)",
        stock_code="000250",
    )
