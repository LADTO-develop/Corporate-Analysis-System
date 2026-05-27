"""External-evidence helpers for Stage 2 committee decisions."""

from __future__ import annotations

from typing import Any, cast

from cas.agents.committee_assessments import (
    ADVERSE_EVIDENCE_QUALITY,
    ADVERSE_PROVIDER_RELEVANCE,
    NoncriticalEvidenceAssessment,
)
from cas.agents.committee_utils import safe_float as _safe_float
from cas.agents.signals.materiality_signals import (
    is_uncorroborated_material_financing_or_guarantee_item as _is_uncorroborated_material_financing_or_guarantee_item,
)
from cas.veto_rules import critical_terms_in_text


def noncritical_external_evidence_assessment(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> NoncriticalEvidenceAssessment:
    """Treat verified but non-critical external evidence as a reason to hold, not reject."""
    status = str(news_cache.get("status", "")).strip().lower()
    if status != "ready":
        return NoncriticalEvidenceAssessment(False, "", 0, 0)
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return NoncriticalEvidenceAssessment(False, "", 0, 0)
    direct_items = [
        item for item in raw_items if isinstance(item, dict) and item.get("company_match") is True
    ]
    if not direct_items:
        return NoncriticalEvidenceAssessment(False, "", 0, 0)
    blocking_items = [
        item
        for item in direct_items
        if is_blocking_external_adverse_item(
            item,
            source_feature_row=source_feature_row,
        )
    ]
    if blocking_items:
        return NoncriticalEvidenceAssessment(False, "", len(direct_items), len(blocking_items))
    contextual_items = [
        item
        for item in direct_items
        if str(item.get("disclosure_severity", "")).lower() in {"routine", "caution"}
        or str(item.get("provider_relevance", "")).lower() in {"routine", "context", "caution"}
        or str(item.get("source", "")).lower() in {"opendart", "naver_news", "tavily"}
    ]
    if not contextual_items:
        return NoncriticalEvidenceAssessment(False, "", len(direct_items), 0)
    reason = (
        f"외부근거 완화 신호: 직접 관련 근거 {len(direct_items)}건을 수집했지만 "
        "강제 경고·치명 키워드·실질 adverse 공시는 확인되지 않았습니다. "
        "따라서 2차 위원회는 모델의 부적격 경고를 확정하기보다 보류로 재점검합니다."
    )
    return NoncriticalEvidenceAssessment(True, reason, len(direct_items), 0)


def no_direct_external_items(news_cache: dict[str, Any]) -> bool:
    """Return whether the cache has no direct company-matched evidence items."""
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return True
    return not any(
        isinstance(item, dict) and item.get("company_match") is True for item in raw_items
    )


def has_nonblocking_external_context(news_cache: dict[str, Any]) -> bool:
    """Return whether direct external context exists but does not block mitigation."""
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    has_direct_item = any(
        isinstance(item, dict) and item.get("company_match") is True for item in raw_items
    )
    return has_direct_item and not overwarning_blocking_external_items(news_cache)


def adverse_external_items(news_cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Return direct external items classified as adverse enough for committee review."""
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return []
    adverse_items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if item.get("company_match") is not True:
            continue
        if is_adverse_external_item(item):
            adverse_items.append(item)
    return adverse_items


def verified_adverse_external_items(news_cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Return adverse evidence items that pass quality or score checks."""
    return [item for item in adverse_external_items(news_cache) if is_verified_adverse_item(item)]


def is_actionable_verified_adverse_external_item(
    item: dict[str, Any],
    news_cache: dict[str, Any],
) -> bool:
    """Return whether a verified adverse item is actionable for committee escalation."""
    return (
        is_verified_adverse_item(item)
        and not is_noisy_aggregated_news_item(item)
        and not is_resolved_procedural_trading_halt_item(item, news_cache)
    )


def overwarning_blocking_external_items(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return external items strong enough to block FP mitigation."""
    return [
        item
        for item in verified_adverse_external_items(news_cache)
        if is_actionable_verified_adverse_external_item(item, news_cache)
        and not _is_uncorroborated_material_financing_or_guarantee_item(
            item,
            source_feature_row=source_feature_row,
        )
    ]


def is_adverse_external_item(item: dict[str, Any]) -> bool:
    """Return whether an item's structured metadata marks adverse external context."""
    if item.get("as_of_date_violation") is True:
        return False
    if _is_uncorroborated_name_only_search_item(item):
        return False
    if item.get("veto_candidate") is True:
        return True
    severity = str(item.get("disclosure_severity", "")).lower()
    if severity in {"veto", "adverse"}:
        if str(item.get("source", "")).lower() == "opendart":
            return True
        return item.get("critical_context_confirmed") is True
    if severity in {"routine", "caution"}:
        return False
    if item.get("critical_context_confirmed") is True:
        return True
    if str(item.get("provider_relevance", "")).lower() in ADVERSE_PROVIDER_RELEVANCE:
        return True
    terms = item_critical_terms(item)
    if not terms:
        return False
    return str(item.get("evidence_quality", "")).lower() in ADVERSE_EVIDENCE_QUALITY


def is_blocking_external_adverse_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    """Return whether an evidence item should prevent FP mitigation."""
    if _is_uncorroborated_material_financing_or_guarantee_item(
        item,
        source_feature_row=source_feature_row,
    ):
        return False
    if item.get("veto_candidate") is True:
        return True
    source = str(item.get("source", "")).lower()
    if _is_uncorroborated_name_only_search_item(item):
        return False
    severity = str(item.get("disclosure_severity", "")).lower()
    if severity in {"veto", "adverse"}:
        return source == "opendart" or item.get("critical_context_confirmed") is True
    # Keyword hits from aggregated news snippets are noisy. They should block
    # over-warning mitigation only when the collector confirmed the risky context.
    return item.get("critical_context_confirmed") is True


def is_noisy_aggregated_news_item(item: dict[str, Any]) -> bool:
    """Detect multi-company market wrap snippets where risk terms can belong elsewhere."""
    source = str(item.get("source", "")).lower()
    if source not in {"naver_news", "tavily"}:
        return False
    if not item_critical_terms(item):
        return False
    title = _compact_text(str(item.get("title", "")))
    noisy_title_markers = (
        "주요공시",
        "기업공시",
        "주요종목뉴스",
        "장마감후주요종목뉴스",
        "전일주요공시",
    )
    return any(marker in title for marker in noisy_title_markers)


def is_resolved_procedural_trading_halt_item(
    item: dict[str, Any],
    news_cache: dict[str, Any],
) -> bool:
    """Do not treat resolved procedural trading-halt checks as hard adverse blockers."""
    if str(item.get("source", "")).lower() != "opendart":
        return False
    text = _compact_text(" ".join(str(item.get(key, "")) for key in ("title", "summary")))
    if "거래정지" not in text:
        return False
    hard_markers = ("관리종목", "상장폐지", "감사의견", "회생", "파산", "불성실공시")
    if any(marker in text for marker in hard_markers):
        return False
    procedural_capital_action_markers = (
        "무상증자",
        "주식분할",
        "액면분할",
        "권리락",
        "주식병합",
    )
    if any(marker in text for marker in procedural_capital_action_markers):
        return True
    if _is_resolved_spac_merger_halt_item(text, news_cache):
        return True
    if "우회상장" not in text:
        return False
    return _has_resolved_reverse_listing_halt(news_cache)


def is_verified_adverse_item(item: dict[str, Any]) -> bool:
    """Return whether an adverse item is verified by evidence quality or score."""
    if item.get("as_of_date_violation") is True:
        return False
    quality = str(item.get("evidence_quality", "")).lower()
    if quality in ADVERSE_EVIDENCE_QUALITY:
        return True
    score = _safe_float(item.get("evidence_score"))
    return score is not None and score >= 0.55


def item_critical_terms(item: dict[str, Any]) -> list[str]:
    """Return explicit critical terms or terms found in item title/summary."""
    raw_terms = item.get("critical_terms", [])
    if isinstance(raw_terms, list | tuple):
        return [str(term) for term in raw_terms if str(term).strip()]
    text = " ".join(str(item.get(key, "")) for key in ("title", "summary"))
    return cast(list[str], critical_terms_in_text(text))


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


def _is_resolved_spac_merger_halt_item(text: str, news_cache: dict[str, Any]) -> bool:
    if "합병" not in text:
        return False
    spac_markers = ("spac", "스팩", "기업인수목적")
    if not any(marker in text.lower() for marker in spac_markers):
        return False
    return _has_resolved_spac_merger_halt(news_cache)


def _has_resolved_spac_merger_halt(news_cache: dict[str, Any]) -> bool:
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    for raw_item in raw_items:
        if not isinstance(raw_item, dict) or raw_item.get("company_match") is not True:
            continue
        text = _compact_text(" ".join(str(raw_item.get(key, "")) for key in ("title", "summary")))
        if "거래정지해제" in text and "상장예비심사결과" in text and "승인" in text:
            return True
    return False


def _has_resolved_reverse_listing_halt(news_cache: dict[str, Any]) -> bool:
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    for raw_item in raw_items:
        if not isinstance(raw_item, dict) or raw_item.get("company_match") is not True:
            continue
        text = _compact_text(" ".join(str(raw_item.get(key, "")) for key in ("title", "summary")))
        if "거래정지해제" in text and "우회상장" in text and "미해당" in text:
            return True
    return False


def _compact_text(text: str) -> str:
    return "".join(str(text).lower().split())


__all__ = [
    "adverse_external_items",
    "has_nonblocking_external_context",
    "is_actionable_verified_adverse_external_item",
    "item_critical_terms",
    "no_direct_external_items",
    "noncritical_external_evidence_assessment",
    "overwarning_blocking_external_items",
]
