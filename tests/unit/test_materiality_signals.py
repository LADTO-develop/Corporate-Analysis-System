from __future__ import annotations

from cas.agents.signals.materiality_signals import (
    confirmed_hard_distress_item,
    has_substantive_external_risk,
    is_uncorroborated_material_financing_or_guarantee_item,
    material_financing_or_guarantee_has_financial_corroboration,
    substantive_external_risk_item,
)


def _material_financing_item() -> dict[str, object]:
    return {
        "source": "opendart",
        "title": "주요사항보고서(유상증자결정)",
        "summary": "직접 관련 유상증자 공시입니다.",
        "company_match": True,
        "provider_relevance": "risk",
        "disclosure_severity": "adverse",
        "disclosure_event_class": "material_financing",
        "disclosure_materiality": "substantive_adverse",
        "materiality_ratio": 0.20,
        "materiality_basis": "희석률: 20.00%",
        "critical_context_confirmed": False,
        "veto_candidate": False,
    }


def test_material_financing_needs_financial_corroboration_for_substantive_risk() -> None:
    defensive_row = {
        "current_ratio": 2.1,
        "cash_ratio": 0.35,
        "cashflow_coverage_ratio": 1.8,
        "ocf_to_total_liabilities": 0.09,
        "ocf_to_sales": 0.04,
        "interest_coverage_ratio": 5.0,
        "equity_ratio": 0.55,
        "debt_ratio": 0.8,
        "total_borrowings_ratio": 0.22,
        "capital_impairment_ratio": 0.0,
        "net_margin": 0.05,
        "icr_under_1": 0,
        "is_2y_consecutive_operating_loss": 0,
        "is_2y_consecutive_ocf_deficit": 0,
    }
    item = _material_financing_item()

    assert material_financing_or_guarantee_has_financial_corroboration(defensive_row) is False
    assert (
        is_uncorroborated_material_financing_or_guarantee_item(
            item,
            source_feature_row=defensive_row,
        )
        is True
    )
    assert substantive_external_risk_item(item, source_feature_row=defensive_row) is False
    assert (
        has_substantive_external_risk(
            {"items": [item]},
            source_feature_row=defensive_row,
        )
        is False
    )


def test_material_financing_becomes_substantive_with_financial_stress() -> None:
    stressed_row = {
        "current_ratio": 0.62,
        "cash_ratio": 0.04,
        "cashflow_coverage_ratio": -1.2,
        "ocf_to_total_liabilities": -0.03,
        "ocf_to_sales": -0.04,
        "interest_coverage_ratio": -3.0,
        "equity_ratio": 0.20,
        "debt_ratio": 3.4,
        "total_borrowings_ratio": 0.72,
        "capital_impairment_ratio": 0.0,
        "net_margin": -0.22,
        "icr_under_1": 1,
        "is_2y_consecutive_operating_loss": 1,
        "is_2y_consecutive_ocf_deficit": 1,
    }
    item = _material_financing_item()

    assert material_financing_or_guarantee_has_financial_corroboration(stressed_row) is True
    assert (
        is_uncorroborated_material_financing_or_guarantee_item(
            item,
            source_feature_row=stressed_row,
        )
        is False
    )
    assert substantive_external_risk_item(item, source_feature_row=stressed_row) is True


def test_non_financing_material_ratio_stays_substantive() -> None:
    item = {
        "source": "opendart",
        "title": "영업정지(종속회사의주요경영사항)",
        "company_match": True,
        "provider_relevance": "risk",
        "disclosure_severity": "adverse",
        "disclosure_event_class": "substantive_adverse",
        "disclosure_materiality": "substantive_adverse",
        "materiality_ratio": 0.1137,
    }

    assert substantive_external_risk_item(item, source_feature_row={}) is True


def test_routine_audit_report_keyword_hit_is_not_confirmed_hard_distress() -> None:
    item = {
        "source": "opendart",
        "title": "[첨부정정]감사보고서제출",
        "summary": "정례 감사보고서 제출 검색요약에 자본잠식 키워드가 포함됨",
        "company_match": True,
        "critical_terms": ["자본잠식"],
        "provider_relevance": "routine",
        "disclosure_severity": "routine",
        "disclosure_event_class": "routine_context",
        "disclosure_materiality": "routine_context",
        "evidence_quality": "medium",
        "evidence_score": 0.91,
    }

    assert confirmed_hard_distress_item(item) is False
    assert substantive_external_risk_item(item, source_feature_row={}) is False


def test_confirmed_audit_failure_remains_hard_distress() -> None:
    item = {
        "source": "opendart",
        "title": "감사의견거절",
        "summary": "해당 회사 감사보고서에서 의견거절 확인",
        "company_match": True,
        "critical_terms": ["감사의견거절"],
        "critical_context_confirmed": True,
        "provider_relevance": "risk",
        "disclosure_severity": "adverse",
        "disclosure_event_class": "audit_failure",
        "disclosure_materiality": "substantive_adverse",
        "evidence_quality": "high",
        "evidence_score": 0.93,
    }

    assert confirmed_hard_distress_item(item) is True
    assert substantive_external_risk_item(item, source_feature_row={}) is True
