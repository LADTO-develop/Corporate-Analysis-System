"""Evidence item validation, scoring, dedupe, and prioritization."""

from __future__ import annotations

import re
from collections.abc import Mapping

from cas.evidence.opendart_materiality import _disclosure_like_severity
from cas.evidence.policy import (
    _DIRECT_EVIDENCE_SCORE_FLOOR,
    _MAX_COMBINED_ITEMS,
    _MAX_WEAK_WEB_ITEMS,
)
from cas.evidence.utils import (
    _canonical_url,
    _company_match_type,
    _entity_aliases,
    _evidence_relevance,
    _normalize_entity_text,
    _recency_bucket,
    _url_domain,
)
from cas.veto_rules import critical_terms_in_text


def _combined_items(
    providers: Mapping[str, object],
    *,
    company_name: str,
    stock_code: str | None,
) -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    for provider in providers.values():
        if not isinstance(provider, dict):
            continue
        raw_items = provider.get("items", [])
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            combined.append(
                _validated_item(
                    item,
                    company_name=company_name,
                    stock_code=stock_code,
                )
            )
    return _prioritized_combined_items(_dedupe_items(combined))


def _validated_item(
    item: dict[str, object],
    *,
    company_name: str,
    stock_code: str | None,
) -> dict[str, object]:
    source = str(item.get("source", ""))
    title = str(item.get("title", ""))
    summary = str(item.get("summary", ""))
    url = str(item.get("url", ""))
    published_at = str(item.get("published_at", ""))
    reliability = str(item.get("reliability", "unknown"))
    provider_relevance = str(item.get("provider_relevance", "unknown"))
    disclosure_severity = str(item.get("disclosure_severity", "unknown")).lower()
    if disclosure_severity in {"", "none", "unknown"}:
        disclosure_severity = _disclosure_like_severity(title, summary)
    text = "\n".join(part for part in (title, summary, url) if part)
    critical_terms = critical_terms_in_text(text)
    if disclosure_severity in {"routine", "caution"}:
        critical_terms = []
    company_match_type = _company_match_type(
        text,
        company_name=company_name,
        stock_code=stock_code,
    )
    company_match = company_match_type != "none"
    critical_context_confirmed = _critical_context_confirmed(
        text,
        critical_terms=critical_terms,
        company_name=company_name,
        stock_code=stock_code,
        source=source,
        company_match=company_match,
    )
    evidence_score = _evidence_score(
        source=source,
        reliability=reliability,
        company_match=company_match,
        critical_terms=critical_terms,
        critical_context_confirmed=critical_context_confirmed,
        published_at=published_at,
        provider_relevance=provider_relevance,
        disclosure_severity=disclosure_severity,
        company_match_type=company_match_type,
    )
    validated = {
        "source": source,
        "title": title,
        "summary": summary,
        "url": url,
        "domain": _url_domain(url),
        "canonical_url": _canonical_url(url),
        "published_at": published_at,
        "recency_bucket": _recency_bucket(published_at),
        "reliability": reliability,
        "provider_relevance": provider_relevance,
        "disclosure_severity": disclosure_severity,
        "disclosure_severity_reason": str(item.get("disclosure_severity_reason", "")),
        "company_match": company_match,
        "company_match_type": company_match_type,
        "evidence_relevance": _evidence_relevance(company_match_type, source=source),
        "critical_terms": critical_terms,
        "critical_context_confirmed": critical_context_confirmed,
        "veto_candidate": bool(critical_terms) and company_match and critical_context_confirmed,
        "evidence_score": round(evidence_score, 4),
        "evidence_quality": _evidence_quality(evidence_score),
        "verification_flags": _verification_flags(
            source=source,
            company_match=company_match,
            company_match_type=company_match_type,
            critical_terms=critical_terms,
            critical_context_confirmed=critical_context_confirmed,
            published_at=published_at,
        ),
        "duplicate_count": 1,
    }
    for key in (
        "corp_code",
        "rcept_no",
        "disclosure_type",
        "disclosure_type_label",
        "disclosure_severity",
        "disclosure_severity_reason",
        "disclosure_event_class",
        "disclosure_materiality",
        "materiality_ratio",
        "materiality_basis",
        "materiality_source",
        "materiality_confidence",
        "dilution_ratio",
        "dilution_basis",
        "business_suspension_scope",
    ):
        if item.get(key):
            validated[key] = str(item.get(key, ""))
    return validated


def _dedupe_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[str, dict[str, object]] = {}
    ordered_keys: list[str] = []
    for item in items:
        key = _dedupe_key(item)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = dict(item)
            ordered_keys.append(key)
            continue
        _merge_duplicate_item(existing, item)
    return [by_key[key] for key in ordered_keys]


def _prioritized_combined_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Prioritize verified/direct evidence while keeping a few weak web hits for auditability."""
    ordered = sorted(items, key=_combined_item_sort_key, reverse=True)
    selected: list[dict[str, object]] = []
    weak_web_count = 0
    for item in ordered:
        if _is_weak_web_item(item):
            if weak_web_count >= _MAX_WEAK_WEB_ITEMS:
                continue
            weak_web_count += 1
        selected.append(item)
        if len(selected) >= _MAX_COMBINED_ITEMS:
            break
    return selected


def _combined_item_sort_key(item: dict[str, object]) -> tuple[float, float, int, str]:
    source = str(item.get("source", "")).lower()
    source_rank = {
        "opendart": 4.0,
        "naver_news": 3.0,
        "tavily": 2.0,
    }.get(source, 1.0)
    match_type = str(item.get("company_match_type", "none"))
    direct_bonus = 0.7 if match_type == "name" else 0.2 if match_type == "stock_code" else 0.0
    context_bonus = 0.4 if item.get("critical_context_confirmed") is True else 0.0
    provider_bonus = 0.25 if item.get("provider_relevance") == "risk" else 0.0
    weak_penalty = -0.5 if _is_weak_web_item(item) else 0.0
    source_score = source_rank + direct_bonus + context_bonus + provider_bonus + weak_penalty
    return (
        source_score,
        _evidence_score_from_item(item),
        _recency_rank(str(item.get("recency_bucket", ""))),
        str(item.get("published_at", "")),
    )


def _is_weak_web_item(item: dict[str, object]) -> bool:
    source = str(item.get("source", "")).lower()
    if source not in {"naver_news", "tavily"}:
        return False
    if str(item.get("evidence_quality", "low")) != "low":
        return False
    match_type = str(item.get("company_match_type", "none"))
    return match_type in {"none", "stock_code"}


def _recency_rank(bucket: str) -> int:
    return {
        "within_90d": 4,
        "within_1y": 3,
        "within_2y": 2,
        "older_than_2y": 1,
        "unknown": 0,
    }.get(bucket, 0)


def _merge_duplicate_item(existing: dict[str, object], incoming: dict[str, object]) -> None:
    existing["duplicate_count"] = _int_from_object(existing.get("duplicate_count"), default=1) + 1
    existing_score = _evidence_score_from_item(existing)
    incoming_score = _evidence_score_from_item(incoming)
    if incoming_score > existing_score:
        for key in (
            "source",
            "title",
            "summary",
            "url",
            "domain",
            "canonical_url",
            "published_at",
            "recency_bucket",
            "reliability",
            "company_match",
            "company_match_type",
            "evidence_relevance",
            "critical_context_confirmed",
            "veto_candidate",
            "evidence_score",
            "evidence_quality",
            "provider_relevance",
            "disclosure_severity",
            "disclosure_severity_reason",
            "disclosure_event_class",
            "disclosure_materiality",
            "materiality_ratio",
            "materiality_basis",
            "materiality_source",
            "materiality_confidence",
            "dilution_ratio",
            "dilution_basis",
            "business_suspension_scope",
            "corp_code",
            "rcept_no",
            "disclosure_type",
            "disclosure_type_label",
        ):
            existing[key] = incoming.get(key)
    existing["critical_terms"] = sorted(
        {
            *_string_list(existing.get("critical_terms")),
            *_string_list(incoming.get("critical_terms")),
        }
    )
    existing["verification_flags"] = sorted(
        {
            *_string_list(existing.get("verification_flags")),
            *_string_list(incoming.get("verification_flags")),
            "duplicate_merged",
        }
    )


def _dedupe_key(item: dict[str, object]) -> str:
    rcept_no = str(item.get("rcept_no", ""))
    if rcept_no:
        return f"opendart:{rcept_no}"
    canonical_url = str(item.get("canonical_url", ""))
    if canonical_url:
        return f"url:{canonical_url}"
    title = _normalize_entity_text(str(item.get("title", "")))
    return f"title:{title}" if title else f"item:{id(item)}"


def _evidence_score_from_item(item: dict[str, object]) -> float:
    value = item.get("evidence_score", 0.0)
    try:
        return float(value) if isinstance(value, int | float | str) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int_from_object(value: object, *, default: int) -> int:
    try:
        return int(value) if isinstance(value, int | float | str) else default
    except (TypeError, ValueError):
        return default


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if str(item).strip()]


def _evidence_score(
    *,
    source: str,
    reliability: str,
    company_match: bool,
    company_match_type: str,
    critical_terms: list[str],
    critical_context_confirmed: bool,
    published_at: str,
    provider_relevance: str,
    disclosure_severity: str,
) -> float:
    source_lower = source.lower()
    reliability_lower = reliability.lower()
    if source_lower == "opendart":
        score = 0.82
    elif source_lower == "naver_news":
        score = 0.62
    else:
        score = 0.52

    if reliability_lower == "high":
        score += 0.10
    elif reliability_lower in {"low", "low_relevance"}:
        score -= 0.12

    if company_match_type == "name":
        score += 0.18
    elif company_match_type == "stock_code":
        score += 0.02 if source_lower in {"naver_news", "tavily"} else 0.18
    elif company_match:
        score += 0.08
    else:
        score -= 0.20
    if critical_terms and critical_context_confirmed:
        score += 0.10
    elif critical_terms:
        score -= 0.08

    if disclosure_severity == "veto":
        score += 0.14
    elif disclosure_severity == "adverse":
        score += 0.08
    elif disclosure_severity == "caution":
        score -= 0.03
    elif disclosure_severity == "routine":
        score -= 0.12
    elif provider_relevance == "risk":
        score += 0.06
    elif provider_relevance == "routine":
        score -= 0.04

    recency_bucket = _recency_bucket(published_at)
    if recency_bucket == "older_than_2y":
        score -= 0.06
    elif recency_bucket == "unknown":
        score -= 0.03
    if critical_terms and not critical_context_confirmed:
        score = min(score, 0.54)
    if disclosure_severity == "routine":
        score = min(score, 0.54)
    elif disclosure_severity == "caution":
        score = min(score, 0.68)
    return min(max(score, 0.0), 1.0)


def _evidence_quality(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= _DIRECT_EVIDENCE_SCORE_FLOOR:
        return "medium"
    return "low"


def _verification_flags(
    *,
    source: str,
    company_match: bool,
    company_match_type: str,
    critical_terms: list[str],
    critical_context_confirmed: bool,
    published_at: str,
) -> list[str]:
    flags = ["trusted_disclosure" if source == "opendart" else "search_result_snippet"]
    flags.append("company_direct_match" if company_match else "company_not_matched")
    if company_match_type == "stock_code":
        flags.append("stock_code_only_match")
    elif company_match_type == "name":
        flags.append("company_name_match")
    if critical_terms:
        flags.append("critical_terms_detected")
        flags.append(
            "critical_context_confirmed"
            if critical_context_confirmed
            else "critical_context_unconfirmed"
        )
    recency_bucket = _recency_bucket(published_at)
    if recency_bucket == "unknown":
        flags.append("published_date_unknown")
    elif recency_bucket == "older_than_2y":
        flags.append("published_date_stale")
    return flags


def _critical_context_confirmed(
    text: str,
    *,
    critical_terms: list[str],
    company_name: str,
    stock_code: str | None,
    source: str,
    company_match: bool,
) -> bool:
    if not critical_terms or not company_match:
        return False
    if source.lower() == "opendart":
        return True

    normalized_text = _normalize_entity_text(text)
    normalized_terms = [_normalize_entity_text(term) for term in critical_terms]
    if _critical_context_scope_excluded(normalized_text, normalized_terms):
        return False
    for segment in _critical_context_segments(text):
        if _company_match_type(segment, company_name=company_name, stock_code=stock_code) == "none":
            continue
        normalized_segment = _normalize_entity_text(segment)
        for alias in _entity_aliases(company_name=company_name, stock_code=stock_code):
            alias_index = normalized_segment.find(alias)
            if alias_index < 0:
                continue
            for term in normalized_terms:
                term_index = normalized_segment.find(term)
                if term_index >= 0 and abs(alias_index - term_index) <= 80:
                    return True
    return False


def _critical_context_segments(text: str) -> list[str]:
    """Split noisy snippets so unrelated company/news-list items do not share risk terms."""
    pattern = "[\\n\\r.!?\\u3002\\uff01\\uff1f;\\uff1b]|\\u2026+|\\s[-\\u2013\\u2014]\\s|\\s/[ ]*|\\s\\u25b3|\\s\\u25b6"
    segments = re.split(pattern, str(text))
    return [segment.strip() for segment in segments if segment.strip()]


def _critical_context_scope_excluded(normalized_text: str, normalized_terms: list[str]) -> bool:
    """Exclude snippets where a critical term is scoped away from the selected common stock."""
    if "상장폐지" not in normalized_terms:
        return False
    preferred_only_markers = (
        "우선주에만해당",
        "우선주만해당",
        "우선주에해당",
    )
    common_stock_unaffected_markers = (
        "보통주에는영향",
        "보통주에영향",
        "보통주에는해당없",
        "보통주해당없",
    )
    return any(marker in normalized_text for marker in preferred_only_markers) or any(
        marker in normalized_text for marker in common_stock_unaffected_markers
    )


def _verification_summary(items: list[dict[str, object]]) -> dict[str, object]:
    quality_counts = {"high": 0, "medium": 0, "low": 0}
    source_counts: dict[str, int] = {}
    weak_web_count = 0
    for item in items:
        quality = str(item.get("evidence_quality", "low"))
        if quality in quality_counts:
            quality_counts[quality] += 1
        source = str(item.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        if _is_weak_web_item(item):
            weak_web_count += 1
    return {
        "quality_counts": quality_counts,
        "source_counts": source_counts,
        "weak_web_item_count": weak_web_count,
        "deduplicated_item_count": len(items),
        "duplicate_merged_count": sum(
            max(_int_from_object(item.get("duplicate_count"), default=1) - 1, 0) for item in items
        ),
    }


def _critical_terms(items: list[dict[str, object]]) -> list[str]:
    terms: set[str] = set()
    for item in items:
        if item.get("company_match") is not True:
            continue
        if item.get("critical_context_confirmed") is not True:
            continue
        source = str(item.get("source", "")).lower()
        severity = str(item.get("disclosure_severity", "")).lower()
        if source == "opendart" and severity in {"routine", "caution"}:
            continue
        terms.update(_string_list(item.get("critical_terms")))
    return sorted(terms)
