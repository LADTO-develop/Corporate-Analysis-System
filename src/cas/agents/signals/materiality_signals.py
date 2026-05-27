"""Shared materiality interpretation for Stage 2 external evidence."""

from __future__ import annotations

from typing import Any

from cas.agents.committee_assessments import (
    ADVERSE_PROVIDER_RELEVANCE,
    FINANCING_EVIDENCE_TERMS,
)
from cas.agents.committee_utils import (
    flag_is_true,
    metric_above,
    metric_below,
    safe_float,
)

_SUBSTANTIVE_EVENT_CLASSES = {
    "substantive_adverse",
    "distress",
    "insolvency",
    "audit_failure",
    "veto_event",
}
_SUBSTANTIVE_MATERIALITY_CLASSES = {"substantive_adverse", "critical", "veto"}
_CONTEXT_EVENT_CLASSES = {
    "routine_context",
    "watch_context",
    "procedural_or_one_off",
    "procedural_trading_halt",
    "low_materiality_litigation",
    "one_off_contract_cancellation",
    "low_materiality_contract_cancellation",
    "business_suspension_low_materiality",
    "subsidiary_business_suspension_low_materiality",
    "low_materiality_financing",
    "low_materiality_debt_guarantee",
    "contract_cancellation_watch",
    "business_suspension_watch",
    "subsidiary_business_suspension_watch",
    "financing_watch",
    "debt_guarantee_watch",
    "litigation_watch",
}
_CONTEXT_MATERIALITY_CLASSES = {
    "routine_context",
    "watch_context",
    "procedural_or_one_off",
}
_CONTEXT_PROVIDER_RELEVANCE = {"routine", "caution", "context"}
_CONTEXT_SEVERITY_CLASSES = {"routine", "caution"}
_HARD_DISTRESS_TERMS = {
    "자본잠식",
    "부도",
    "파산",
    "회생",
    "상장폐지",
    "관리종목",
    "감사의견거절",
    "의견거절",
    "횡령",
    "배임",
    "채무불이행",
}
_AUDIT_REPORT_ROUTINE_MARKERS = (
    "감사보고서제출",
    "감사보고서(",
    "감사보고서",
    "연결감사보고서",
)
_AUDIT_REPORT_FAILURE_MARKERS = (
    "감사의견거절",
    "의견거절",
    "한정의견",
    "부적정의견",
    "계속기업불확실성",
    "감사범위제한",
    "상장폐지사유발생",
)
_GUARANTEE_MARKERS = ("채무보증", "타인에대한채무보증")

__all__ = [
    "confirmed_external_veto_item",
    "confirmed_hard_distress_item",
    "financial_observation_count",
    "financing_evidence_items",
    "has_hard_distress_terms",
    "has_substantive_external_risk",
    "hidden_tail_evidence_requires_risk_signal",
    "high_risk_financing_evidence_count",
    "is_material_financing_or_guarantee_item",
    "is_uncorroborated_material_financing_or_guarantee_item",
    "material_financing_evidence_blocks_tn_hold",
    "material_financing_or_guarantee_has_extreme_distress",
    "material_financing_or_guarantee_has_financial_corroboration",
    "material_financing_or_guarantee_has_severe_financial_corroboration",
    "substantive_external_risk_item",
]


def has_substantive_external_risk(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    """Return whether any evidence item is material enough to drive risk escalation."""
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    return any(
        isinstance(item, dict)
        and substantive_external_risk_item(
            item,
            source_feature_row=source_feature_row,
        )
        for item in raw_items
    )


def substantive_external_risk_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    """Return whether an evidence item should be treated as substantive risk.

    Financing and debt-guarantee disclosures are contextual unless they have hard
    distress context or enough financial-stress corroboration in the source row.
    """
    if item.get("company_match") is not True:
        return False
    if item.get("as_of_date_violation") is True:
        return False
    if _is_uncorroborated_name_only_search_item(item):
        return False
    if is_uncorroborated_material_financing_or_guarantee_item(
        item,
        source_feature_row=source_feature_row,
    ):
        return False
    if _is_routine_or_procedural_item(item):
        return False
    if confirmed_external_veto_item(item):
        return True

    event_class = str(item.get("disclosure_event_class", "")).lower()
    materiality = str(item.get("disclosure_materiality", "")).lower()
    severity = str(item.get("disclosure_severity", "")).lower()
    provider_relevance = str(item.get("provider_relevance", "")).lower()
    if event_class in _SUBSTANTIVE_EVENT_CLASSES:
        return True
    if materiality in _SUBSTANTIVE_MATERIALITY_CLASSES:
        return True

    ratio = safe_float(item.get("materiality_ratio"))
    if ratio is not None and ratio >= 0.10:
        return True
    if severity in {"adverse", "veto"}:
        return True
    return provider_relevance in ADVERSE_PROVIDER_RELEVANCE and severity not in {
        "routine",
        "caution",
    }


def material_financing_evidence_blocks_tn_hold(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    """Block TN overhold relief only for repeated or explicitly high-risk financing."""
    return bool(
        len(financing_evidence_items(news_cache)) >= 2
        or high_risk_financing_evidence_count(
            news_cache,
            source_feature_row=source_feature_row,
        )
        >= 1
    )


def high_risk_financing_evidence_count(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> int:
    """Count financing/guarantee items that remain high-risk after corroboration checks."""
    return sum(
        1
        for item in financing_evidence_items(news_cache)
        if (
            not is_uncorroborated_material_financing_or_guarantee_item(
                item,
                source_feature_row=source_feature_row,
            )
            and (
                item.get("veto_candidate") is True
                or item.get("critical_context_confirmed") is True
                or str(item.get("provider_relevance", "")).lower() in ADVERSE_PROVIDER_RELEVANCE
                or str(item.get("disclosure_severity", "")).lower() in {"adverse", "veto"}
            )
        )
    )


def financing_evidence_items(news_cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Return direct OpenDART financing and debt-guarantee evidence items."""
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict) or item.get("company_match") is not True:
            continue
        if str(item.get("source", "")).lower() != "opendart":
            continue
        text = _item_text(item)
        if any(term in text for term in FINANCING_EVIDENCE_TERMS):
            items.append(item)
    return items


def is_uncorroborated_material_financing_or_guarantee_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None,
) -> bool:
    """Treat material financing/guarantee as contextual unless distress corroborates it."""
    if not is_material_financing_or_guarantee_item(item):
        return False
    if confirmed_external_veto_item(item) or confirmed_hard_distress_item(item):
        return False
    if not source_feature_row:
        return False
    return not material_financing_or_guarantee_has_financial_corroboration(source_feature_row)


def hidden_tail_evidence_requires_risk_signal(
    items: list[dict[str, Any]],
    *,
    source_feature_row: dict[str, Any],
) -> bool:
    """Return whether hidden-tail evidence should keep the stronger risk signal."""
    if not items:
        return False
    for item in items:
        if not is_material_financing_or_guarantee_item(item):
            return True
        if confirmed_external_veto_item(item) or confirmed_hard_distress_item(item):
            return True
    if material_financing_or_guarantee_has_extreme_distress(source_feature_row):
        return True
    return material_financing_or_guarantee_has_severe_financial_corroboration(source_feature_row)


def is_material_financing_or_guarantee_item(item: dict[str, Any]) -> bool:
    """Return whether an item is a material financing or debt-guarantee disclosure."""
    if str(item.get("source", "")).lower() != "opendart":
        return False
    if item.get("company_match") is not True:
        return False
    event_class = str(item.get("disclosure_event_class", "")).lower()
    if event_class in {"material_financing", "material_debt_guarantee"}:
        return True
    materiality = str(item.get("disclosure_materiality", "")).lower()
    ratio = safe_float(item.get("materiality_ratio"))
    if materiality != "substantive_adverse" and (ratio is None or ratio < 0.10):
        return False
    text = _item_text(item)
    return any(term in text for term in FINANCING_EVIDENCE_TERMS) or any(
        marker in text for marker in _GUARANTEE_MARKERS
    )


def has_hard_distress_terms(item: dict[str, Any]) -> bool:
    """Return whether an item text or critical terms contain hard distress markers."""
    terms = {str(term).strip() for term in _item_critical_terms(item)}
    if terms.intersection(_HARD_DISTRESS_TERMS):
        return True
    return any(term in _item_text(item) for term in _HARD_DISTRESS_TERMS)


def confirmed_external_veto_item(item: dict[str, Any]) -> bool:
    """Return whether a veto marker is direct enough to drive critical review."""
    if item.get("company_match") is not True:
        return False
    if item.get("as_of_date_violation") is True:
        return False
    if _is_uncorroborated_name_only_search_item(item):
        return False
    if _is_contextual_or_routine_item(item):
        return False
    if _is_routine_audit_report_item(item):
        return False
    if item.get("veto_candidate") is True or item.get("critical_context_confirmed") is True:
        return True

    source = str(item.get("source") or "").lower()
    event_class = str(item.get("disclosure_event_class") or "").lower()
    materiality = str(item.get("disclosure_materiality") or "").lower()
    severity = str(item.get("disclosure_severity") or "").lower()
    return bool(
        source == "opendart"
        and (
            event_class == "veto_event" or materiality in {"critical", "veto"} or severity == "veto"
        )
    )


def confirmed_hard_distress_item(item: dict[str, Any]) -> bool:
    """Return whether hard distress keywords are confirmed enough for critical treatment."""
    if item.get("company_match") is not True:
        return False
    if item.get("as_of_date_violation") is True:
        return False
    if _is_uncorroborated_name_only_search_item(item):
        return False
    if not has_hard_distress_terms(item):
        return False
    if _is_contextual_or_routine_item(item):
        return False
    if _is_routine_audit_report_item(item):
        return False
    if confirmed_external_veto_item(item):
        return True

    source = str(item.get("source") or "").lower()
    event_class = str(item.get("disclosure_event_class") or "").lower()
    materiality = str(item.get("disclosure_materiality") or "").lower()
    severity = str(item.get("disclosure_severity") or "").lower()
    quality = str(item.get("evidence_quality") or "").lower()
    score = safe_float(item.get("evidence_score"))
    if quality == "low":
        return False
    if source == "opendart" and (
        event_class in _SUBSTANTIVE_EVENT_CLASSES
        or materiality in _SUBSTANTIVE_MATERIALITY_CLASSES
        or severity in {"adverse", "veto"}
    ):
        return True
    return bool(
        quality in {"medium", "high"}
        and score is not None
        and score >= 0.75
        and str(item.get("provider_relevance") or "").lower() in ADVERSE_PROVIDER_RELEVANCE
    )


def material_financing_or_guarantee_has_financial_corroboration(
    row: dict[str, Any],
) -> bool:
    """Return whether financial stress corroborates a material financing disclosure."""
    if financial_observation_count(row) < 3:
        return True
    if material_financing_or_guarantee_has_extreme_distress(row):
        return True
    weak_axes = [
        metric_below(row, "cashflow_coverage_ratio", 0.0)
        or metric_below(row, "ocf_to_total_liabilities", 0.0)
        or metric_below(row, "ocf_to_sales", 0.0)
        or flag_is_true(row.get("is_2y_consecutive_ocf_deficit")),
        metric_below(row, "interest_coverage_ratio", 1.0) or flag_is_true(row.get("icr_under_1")),
        metric_below(row, "net_margin", -0.10)
        or flag_is_true(row.get("is_2y_consecutive_operating_loss")),
        metric_below(row, "equity_ratio", 0.25)
        or metric_above(row, "debt_ratio", 3.0)
        or metric_above(row, "total_borrowings_ratio", 0.65)
        or metric_above(row, "capital_impairment_ratio", 0.0),
        metric_below(row, "current_ratio", 1.0) and metric_below(row, "cash_ratio", 0.10),
    ]
    return sum(1 for passed in weak_axes if passed) >= 2


def material_financing_or_guarantee_has_severe_financial_corroboration(
    row: dict[str, Any],
) -> bool:
    """Return whether financial stress is severe enough to keep a risk-hold signal."""
    if financial_observation_count(row) < 3:
        return True
    cashflow_weak = (
        metric_below(row, "cashflow_coverage_ratio", 0.0)
        or metric_below(row, "ocf_to_total_liabilities", 0.0)
        or metric_below(row, "ocf_to_sales", 0.0)
        or flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
    )
    interest_weak = metric_below(row, "interest_coverage_ratio", 1.0) or flag_is_true(
        row.get("icr_under_1")
    )
    earnings_weak = metric_below(row, "net_margin", -0.10) or flag_is_true(
        row.get("is_2y_consecutive_operating_loss")
    )
    leverage_weak = (
        metric_below(row, "equity_ratio", 0.25)
        or metric_above(row, "debt_ratio", 3.0)
        or metric_above(row, "total_borrowings_ratio", 0.65)
        or metric_above(row, "capital_impairment_ratio", 0.0)
    )
    liquidity_weak = metric_below(row, "current_ratio", 1.0) and metric_below(
        row, "cash_ratio", 0.10
    )
    weak_axis_count = sum(
        1
        for passed in (
            cashflow_weak,
            interest_weak,
            earnings_weak,
            leverage_weak,
            liquidity_weak,
        )
        if passed
    )
    return bool(weak_axis_count >= 3 or (cashflow_weak and weak_axis_count >= 2))


def material_financing_or_guarantee_has_extreme_distress(row: dict[str, Any]) -> bool:
    """Return whether the row has extreme distress that can corroborate financing risk."""
    if metric_above(row, "capital_impairment_ratio", 0.50):
        return True
    if metric_below(row, "equity_ratio", 0.15):
        return True
    if metric_above(row, "debt_ratio", 5.0):
        return True

    short_term_share = safe_float(row.get("short_term_borrowings_share"))
    short_term_maturity_wall = short_term_share is not None and short_term_share >= 0.95
    weak_cashflow = (
        metric_below(row, "cashflow_coverage_ratio", 0.0)
        or metric_below(row, "ocf_to_total_liabilities", 0.0)
        or metric_below(row, "ocf_to_sales", 0.0)
    )
    recurring_loss_or_ocf_deficit = flag_is_true(
        row.get("is_2y_consecutive_operating_loss")
    ) or flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
    interest_blocked = flag_is_true(row.get("icr_under_1")) or metric_below(
        row,
        "interest_coverage_ratio",
        1.0,
    )
    return bool(
        short_term_maturity_wall
        and weak_cashflow
        and recurring_loss_or_ocf_deficit
        and interest_blocked
    )


def financial_observation_count(row: dict[str, Any]) -> int:
    """Count available financial observations used by materiality guardrails."""
    keys = (
        "cashflow_coverage_ratio",
        "ocf_to_total_liabilities",
        "ocf_to_sales",
        "interest_coverage_ratio",
        "equity_ratio",
        "debt_ratio",
        "total_borrowings_ratio",
        "current_ratio",
        "cash_ratio",
        "net_margin",
    )
    return sum(1 for key in keys if row.get(key) is not None)


def _is_contextual_or_routine_item(item: dict[str, Any]) -> bool:
    event_class = str(item.get("disclosure_event_class") or "").lower()
    materiality = str(item.get("disclosure_materiality") or "").lower()
    severity = str(item.get("disclosure_severity") or "").lower()
    provider_relevance = str(item.get("provider_relevance") or "").lower()
    return bool(
        event_class in _CONTEXT_EVENT_CLASSES
        or materiality in _CONTEXT_MATERIALITY_CLASSES
        or severity in _CONTEXT_SEVERITY_CLASSES
        or provider_relevance in _CONTEXT_PROVIDER_RELEVANCE
    )


def _is_routine_or_procedural_item(item: dict[str, Any]) -> bool:
    event_class = str(item.get("disclosure_event_class") or "").lower()
    materiality = str(item.get("disclosure_materiality") or "").lower()
    severity = str(item.get("disclosure_severity") or "").lower()
    provider_relevance = str(item.get("provider_relevance") or "").lower()
    return bool(
        event_class
        in {
            "routine_context",
            "procedural_or_one_off",
            "procedural_trading_halt",
            "low_materiality_litigation",
            "one_off_contract_cancellation",
            "low_materiality_contract_cancellation",
            "business_suspension_low_materiality",
            "subsidiary_business_suspension_low_materiality",
            "low_materiality_financing",
            "low_materiality_debt_guarantee",
        }
        or materiality in {"routine_context", "procedural_or_one_off"}
        or severity == "routine"
        or provider_relevance == "routine"
    )


def _is_routine_audit_report_item(item: dict[str, Any]) -> bool:
    text = _item_text(item)
    if any(marker in text for marker in _compact_terms(_AUDIT_REPORT_FAILURE_MARKERS)):
        return False
    return any(marker in text for marker in _compact_terms(_AUDIT_REPORT_ROUTINE_MARKERS))


def _is_uncorroborated_name_only_search_item(item: dict[str, Any]) -> bool:
    source = str(item.get("source", "")).lower()
    if source not in {"naver_news", "tavily"}:
        return False
    if str(item.get("company_disambiguation", "")).lower() != "name_only_search_result":
        return False
    duplicate_sources = item.get("duplicate_sources", [])
    if not isinstance(duplicate_sources, list | tuple):
        return True
    return len({str(source) for source in duplicate_sources if str(source).strip()}) < 2


def _item_critical_terms(item: dict[str, Any]) -> list[str]:
    raw_terms = item.get("critical_terms", [])
    if isinstance(raw_terms, list | tuple):
        return [str(term) for term in raw_terms if str(term).strip()]
    return []


def _item_text(item: dict[str, Any]) -> str:
    return _compact_text(" ".join(str(item.get(key, "")) for key in ("title", "summary")))


def _compact_text(text: str) -> str:
    return "".join(str(text).lower().split())


def _compact_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_compact_text(term) for term in terms)
