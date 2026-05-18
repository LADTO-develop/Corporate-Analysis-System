"""Config-backed veto rules for committee review and evidence collection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from cas.utils.io import read_yaml

DEFAULT_COMMITTEE_CONFIG_PATH = Path("configs/agent/committee.yaml")


@dataclass(frozen=True)
class VetoRules:
    """Veto-rule configuration shared by Stage 2 and external evidence collection."""

    enabled: bool
    triggered_label: str
    honor_news_cache_critical_risk: bool
    external_veto_requires_company_match: bool
    external_veto_min_confirming_items: int
    external_veto_min_confirming_sources: int
    external_veto_min_high_reliability_items: int
    external_veto_high_reliability_sources: tuple[str, ...]
    external_veto_high_reliability_levels: tuple[str, ...]
    blocking_flag_markers: tuple[str, ...]
    external_evidence_terms: tuple[str, ...]


@lru_cache(maxsize=4)
def load_veto_rules(config_path: str | Path = DEFAULT_COMMITTEE_CONFIG_PATH) -> VetoRules:
    """Load committee veto rules from YAML config."""
    config = read_yaml(config_path)
    raw_rules = config.get("veto_rules", {})
    rules = raw_rules if isinstance(raw_rules, dict) else {}
    return VetoRules(
        enabled=bool(rules.get("enabled", True)),
        triggered_label=str(rules.get("triggered_label", "부적격")),
        honor_news_cache_critical_risk=bool(rules.get("honor_news_cache_critical_risk", True)),
        external_veto_requires_company_match=bool(
            rules.get("external_veto_requires_company_match", True)
        ),
        external_veto_min_confirming_items=_positive_int(
            rules.get("external_veto_min_confirming_items", 2)
        ),
        external_veto_min_confirming_sources=_positive_int(
            rules.get("external_veto_min_confirming_sources", 2)
        ),
        external_veto_min_high_reliability_items=_positive_int(
            rules.get("external_veto_min_high_reliability_items", 1)
        ),
        external_veto_high_reliability_sources=_string_tuple(
            rules.get("external_veto_high_reliability_sources", ["opendart"])
        ),
        external_veto_high_reliability_levels=_string_tuple(
            rules.get("external_veto_high_reliability_levels", ["high"])
        ),
        blocking_flag_markers=_string_tuple(rules.get("blocking_flag_markers", [])),
        external_evidence_terms=_string_tuple(rules.get("external_evidence_terms", [])),
    )


def flag_contains_veto_marker(
    flag: object,
    *,
    rules: VetoRules | None = None,
) -> bool:
    """Return whether a rule-engine blocking flag matches a configured veto marker."""
    active_rules = rules or load_veto_rules()
    if not active_rules.enabled:
        return False
    flag_text = str(flag).lower()
    return any(marker.lower() in flag_text for marker in active_rules.blocking_flag_markers)


def critical_terms_in_text(
    text: str,
    *,
    rules: VetoRules | None = None,
) -> list[str]:
    """Return configured external-evidence critical terms found in free text."""
    active_rules = rules or load_veto_rules()
    if not active_rules.enabled:
        return []
    lowered = text.lower()
    return sorted(
        {term for term in active_rules.external_evidence_terms if term.lower() in lowered}
    )


def external_evidence_veto_triggered(
    news_cache_snapshot: dict[str, Any],
    *,
    company_name: str,
    stock_code: str,
    rules: VetoRules | None = None,
) -> bool:
    """Return whether external evidence is strong enough to trigger a veto."""
    active_rules = rules or load_veto_rules()
    if not active_rules.enabled or not active_rules.honor_news_cache_critical_risk:
        return False

    candidates = _external_veto_candidates(
        news_cache_snapshot,
        company_name=company_name,
        stock_code=stock_code,
        rules=active_rules,
    )
    if len(candidates) < active_rules.external_veto_min_confirming_items:
        return False

    confirming_sources = {str(item.get("source", "")).lower() for item in candidates}
    if len(confirming_sources) < active_rules.external_veto_min_confirming_sources:
        return False

    high_reliability_count = sum(
        _is_high_reliability_external_item(item, rules=active_rules) for item in candidates
    )
    return high_reliability_count >= active_rules.external_veto_min_high_reliability_items


def _external_veto_candidates(
    news_cache_snapshot: dict[str, Any],
    *,
    company_name: str,
    stock_code: str,
    rules: VetoRules,
) -> list[dict[str, Any]]:
    raw_items = news_cache_snapshot.get("items", [])
    if not isinstance(raw_items, list):
        return []

    candidates: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = _external_item_text(item)
        terms = _critical_terms_from_item(item, text=text, rules=rules)
        if not terms:
            continue
        if rules.external_veto_requires_company_match and not _item_matches_company(
            item,
            text=text,
            company_name=company_name,
            stock_code=stock_code,
        ):
            continue
        candidates.append(item)
    return candidates


def _is_high_reliability_external_item(item: dict[str, Any], *, rules: VetoRules) -> bool:
    source = str(item.get("source", "")).lower()
    reliability = str(item.get("reliability", "")).lower()
    high_sources = {source.lower() for source in rules.external_veto_high_reliability_sources}
    high_levels = {level.lower() for level in rules.external_veto_high_reliability_levels}
    return source in high_sources or reliability in high_levels


def _critical_terms_from_item(
    item: dict[str, Any],
    *,
    text: str,
    rules: VetoRules,
) -> list[str]:
    raw_terms = item.get("critical_terms", [])
    if isinstance(raw_terms, list):
        terms = [str(term) for term in raw_terms if str(term).strip()]
        if terms:
            return terms
    if isinstance(raw_terms, str) and raw_terms.strip():
        return [term.strip() for term in raw_terms.split("|") if term.strip()]
    return critical_terms_in_text(text, rules=rules)


def _item_matches_company(
    item: dict[str, Any],
    *,
    text: str,
    company_name: str,
    stock_code: str,
) -> bool:
    raw_match = item.get("company_match")
    if isinstance(raw_match, bool):
        return raw_match
    if isinstance(raw_match, str) and raw_match.strip().lower() in {"true", "1", "yes"}:
        return True
    return _direct_company_match(text, company_name=company_name, stock_code=stock_code)


def _external_item_text(item: dict[str, Any]) -> str:
    return " ".join(str(item.get(key, "")) for key in ("title", "summary", "url"))


def _direct_company_match(text: str, *, company_name: str, stock_code: str) -> bool:
    normalized_text = _normalize_entity_text(text)
    normalized_name = _normalize_entity_text(company_name)
    if normalized_name and normalized_name in normalized_text:
        return True

    normalized_stock = "".join(ch for ch in str(stock_code) if ch.isdigit())
    if len(normalized_stock) >= 6 and normalized_stock in normalized_text:
        return True
    return 0 < len(normalized_stock) < 6 and normalized_stock.zfill(6) in normalized_text


def _normalize_entity_text(text: str) -> str:
    normalized = str(text).lower()
    for token in ("주식회사", "(주)", "㈜", "주식", "회사"):
        normalized = normalized.replace(token, "")
    return "".join(ch for ch in normalized if ch.isalnum() or "\uac00" <= ch <= "\ud7a3")


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    values: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            values.append(text)
    return tuple(values)


def _positive_int(value: object) -> int:
    try:
        numeric = int(value) if isinstance(value, int | float | str) else 1
    except (TypeError, ValueError):
        numeric = 1
    return max(1, numeric)


__all__ = [
    "DEFAULT_COMMITTEE_CONFIG_PATH",
    "VetoRules",
    "critical_terms_in_text",
    "external_evidence_veto_triggered",
    "flag_contains_veto_marker",
    "load_veto_rules",
]
