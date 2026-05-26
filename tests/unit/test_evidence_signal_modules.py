"""Tests for EvidenceAuditAgent signal modules."""

from __future__ import annotations

from cas.agents.signals.debt_liquidity_signals import evaluate_debt_liquidity
from cas.agents.signals.evidence_treatment_signals import evaluate_evidence_treatment
from cas.agents.signals.external_evidence_signals import evaluate_external_evidence
from cas.agents.signals.macro_signals import evaluate_macro_market
from cas.agents.stage2_bundle import build_stage2_input_bundle
from cas.agents.state import AgentState


def test_debt_liquidity_signals_flag_liquidity_mismatch() -> None:
    state: AgentState = {
        "source_feature_row": {
            "current_ratio": 0.82,
            "cash_ratio": 0.11,
            "short_term_borrowings_share": 0.71,
            "cashflow_coverage_ratio": 0.80,
            "interest_coverage_ratio": 2.10,
        },
        "xgboost_result": {"prediction_label": "투자적격"},
    }

    signals = evaluate_debt_liquidity(build_stage2_input_bundle(state))

    assert "추가 경계" in signals.summary
    assert any("유동비율이 1.0 미만" in item for item in signals.findings)
    assert any("단기차입금 비중이 높아 차환 리스크" in item for item in signals.findings)
    assert signals.confidence > 0.0


def test_macro_signals_extract_speculative_spread_context() -> None:
    state: AgentState = {
        "market": "KOSDAQ",
        "source_feature_row": {"market": "KOSDAQ", "spec_spread": 0.45},
    }

    signals = evaluate_macro_market(build_stage2_input_bundle(state))

    assert "KOSDAQ" in signals.findings[0]
    assert "0.45%p" in signals.findings[1]


def test_external_evidence_signals_include_critical_risk_terms() -> None:
    signals = evaluate_external_evidence(
        {
            "items": [
                {
                    "source": "naver",
                    "title": "감사의견 관련 우려 보도",
                    "reliability": "medium",
                    "company_match": False,
                    "critical_terms": ["횡령"],
                    "veto_candidate": False,
                }
            ],
            "has_critical_risk": True,
            "critical_terms": ["횡령", "상장폐지"],
        }
    )

    assert any("감사의견 관련 우려 보도" in item for item in signals.findings)
    assert any("직접 관련성 낮음" in item for item in signals.findings)
    assert any("미확인 위험 키워드 히트" in item for item in signals.findings)


def test_evidence_treatment_separates_watch_context_from_critical() -> None:
    signals = evaluate_evidence_treatment(
        {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "단일판매ㆍ공급계약해지",
                    "company_match": True,
                    "disclosure_event_class": "watch_context",
                    "disclosure_materiality": "watch_context",
                    "materiality_ratio": 0.0592,
                    "materiality_basis": "매출 대비 계약해지 비율: 5.92%",
                }
            ],
        }
    )

    assert signals.critical_evidence_count == 0
    assert signals.watch_context_count == 1
    assert signals.hard_distress_detected is False
    assert signals.recommended_evidence_treatment == "watch_context"
    assert signals.materiality_summary["top_materiality_basis"].endswith("5.92%")


def test_evidence_treatment_downgrades_routine_audit_keyword_hits() -> None:
    signals = evaluate_evidence_treatment(
        {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "감사보고서제출",
                    "summary": "정기 감사보고서 검색 요약에 자본잠식 단어가 함께 노출됨",
                    "company_match": True,
                    "critical_terms": ["자본잠식"],
                    "provider_relevance": "routine",
                    "disclosure_severity": "routine",
                    "disclosure_event_class": "routine_context",
                    "disclosure_materiality": "routine_context",
                    "evidence_quality": "medium",
                    "evidence_score": 0.86,
                }
            ],
        }
    )

    assert signals.critical_evidence_count == 0
    assert signals.watch_context_count == 1
    assert signals.hard_distress_detected is False
    assert signals.materiality_summary["hard_distress_item_count"] == 0
    assert signals.recommended_evidence_treatment == "watch_context"


def test_evidence_treatment_ignores_weakly_related_search_summary_keywords() -> None:
    signals = evaluate_evidence_treatment(
        {
            "status": "ready",
            "items": [
                {
                    "source": "tavily",
                    "title": "시장 주요 공시 검색 요약",
                    "summary": "다른 기업의 횡령 및 상장폐지 키워드가 함께 노출됨",
                    "company_match": None,
                    "critical_terms": ["횡령", "상장폐지"],
                    "provider_relevance": "risk",
                    "evidence_quality": "high",
                    "evidence_score": 0.82,
                }
            ],
        }
    )

    assert signals.critical_evidence_count == 0
    assert signals.watch_context_count == 0
    assert signals.hard_distress_detected is False
    assert signals.recommended_evidence_treatment == "context_only"


def test_evidence_treatment_keeps_confirmed_hard_distress_critical() -> None:
    signals = evaluate_evidence_treatment(
        {
            "status": "ready",
            "items": [
                {
                    "source": "opendart",
                    "title": "횡령ㆍ배임혐의발생",
                    "summary": "해당 회사 임원의 횡령 혐의 발생 공시",
                    "company_match": True,
                    "critical_terms": ["횡령", "배임"],
                    "critical_context_confirmed": True,
                    "veto_candidate": True,
                    "provider_relevance": "risk",
                    "disclosure_severity": "veto",
                    "disclosure_event_class": "veto_event",
                    "disclosure_materiality": "critical",
                    "evidence_quality": "high",
                    "evidence_score": 0.94,
                }
            ],
        }
    )

    assert signals.critical_evidence_count == 1
    assert signals.watch_context_count == 0
    assert signals.hard_distress_detected is True
    assert signals.materiality_summary["hard_distress_item_count"] == 1
    assert signals.recommended_evidence_treatment == "critical_veto_review"
