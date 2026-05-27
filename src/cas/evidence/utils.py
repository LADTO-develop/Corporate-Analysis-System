"""Shared normalization, date, and query helpers for evidence collection."""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urlunparse

from cas.evidence.policy import _NAVER_RISK_KEYWORDS


def _collection_end_date(as_of_date: date | str | None) -> date:
    if isinstance(as_of_date, date):
        return min(as_of_date, date.today())
    if isinstance(as_of_date, str) and as_of_date.strip():
        try:
            return min(date.fromisoformat(as_of_date.strip()), date.today())
        except ValueError:
            return date.today()
    return date.today()


def _filter_provider_items_by_as_of_date(
    items: list[dict[str, str]],
    *,
    as_of_date: date | str | None,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Remove evidence that would not have been visible at the evaluation cut-off."""
    end_date = _collection_end_date(as_of_date)
    historical_mode = end_date < date.today()
    kept: list[dict[str, str]] = []
    filtered_after_cutoff = 0
    filtered_undated = 0
    for item in items:
        parsed = _parse_published_at(str(item.get("published_at", "")))
        if parsed is None:
            if historical_mode:
                filtered_undated += 1
                continue
            kept.append(item)
            continue
        if parsed.date() <= end_date:
            kept.append(item)
        else:
            filtered_after_cutoff += 1
    return kept, {
        "end_date": end_date.isoformat(),
        "historical_mode": historical_mode,
        "filtered_after_cutoff_count": filtered_after_cutoff,
        "filtered_undated_count": filtered_undated,
    }


def _normalize_corp_code(value: object) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(8)


def _normalize_stock_code(value: object) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(6)


def _risk_query(*, company_name: str, stock_code: str | None) -> str:
    identifier = f" {stock_code}" if stock_code else ""
    normalized_name = _searchable_company_name(company_name)
    alias = f" {normalized_name}" if normalized_name and normalized_name != company_name else ""
    return (
        f"{company_name}{alias}{identifier} 신용위험 횡령 배임 소송 감사의견 "
        "상장폐지 유동성 차입금 회사채"
    )


def _naver_news_queries(*, company_name: str) -> list[str]:
    normalized_name = _searchable_company_name(company_name) or company_name
    return [f"{normalized_name} {keyword}" for keyword in _NAVER_RISK_KEYWORDS]


def _searchable_company_name(company_name: str) -> str:
    cleaned = str(company_name)
    for token in ("주식회사", "(주)", "㈜", "주식", "회사"):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def _mapping_value(payload: object, key: str) -> object:
    if not isinstance(payload, dict):
        return None
    return payload.get(key)


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).replace("&quot;", '"').replace("&amp;", "&").strip()


def _url_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower().removeprefix("www.")


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower().removeprefix("www."),
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )


def _recency_bucket(published_at: str) -> str:
    parsed = _parse_published_at(published_at)
    if parsed is None:
        return "unknown"
    age_days = (datetime.now(UTC) - parsed).days
    if age_days <= 90:
        return "within_90d"
    if age_days <= 365:
        return "within_1y"
    if age_days <= 730:
        return "within_2y"
    return "older_than_2y"


def _parse_published_at(published_at: str) -> datetime | None:
    text = published_at.strip()
    if not text:
        return None
    for parser in (_parse_yyyymmdd, _parse_iso_datetime, _parse_rfc_datetime):
        parsed = parser(text)
        if parsed is not None:
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _parse_yyyymmdd(text: str) -> datetime | None:
    if not re.fullmatch(r"\d{8}", text):
        return None
    try:
        return datetime.strptime(text, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_iso_datetime(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_rfc_datetime(text: str) -> datetime | None:
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


def _direct_company_match(text: str, *, company_name: str, stock_code: str | None) -> bool:
    return _company_match_type(text, company_name=company_name, stock_code=stock_code) != "none"


def _company_match_type(text: str, *, company_name: str, stock_code: str | None) -> str:
    normalized_text = _normalize_entity_text(text)
    if _contains_company_name_alias(text, company_name=company_name):
        return "name"
    normalized_stock = _stock_code_alias(stock_code)
    if normalized_stock and normalized_stock in normalized_text:
        return "stock_code"
    return "none"


def _evidence_relevance(company_match_type: str, *, source: str) -> str:
    if source.lower() == "opendart" and company_match_type != "none":
        return "direct"
    if company_match_type == "name":
        return "direct"
    if company_match_type == "stock_code":
        return "ticker_only"
    return "weak"


def _entity_aliases(company_name: str, stock_code: str | None) -> list[str]:
    aliases = _company_name_aliases(company_name)
    stock_alias = _stock_code_alias(stock_code)
    if stock_alias:
        aliases.append(stock_alias)
    return aliases


def _company_name_aliases(company_name: str) -> list[str]:
    normalized_name = _normalize_entity_text(company_name)
    return [normalized_name] if normalized_name else []


def _raw_company_name_aliases(company_name: str) -> list[str]:
    cleaned = str(company_name).lower()
    for token in ("주식회사", "(주)", "㈜", "주식", "회사"):
        cleaned = cleaned.replace(token, "")
    aliases = [cleaned.strip(), "".join(cleaned.split())]
    return [alias for alias in dict.fromkeys(aliases) if alias]


def _contains_company_name_alias(text: str, *, company_name: str) -> bool:
    raw_text = str(text).lower()
    particle_chars = "은는이가을를의와과도에로"
    for alias in _raw_company_name_aliases(company_name):
        pattern = (
            rf"(?<![0-9A-Za-z가-힣]){re.escape(alias)}"
            rf"(?=$|[^0-9A-Za-z가-힣]|[{particle_chars}])"
        )
        if re.search(pattern, raw_text):
            return True
    return False


def _stock_code_alias(stock_code: str | None) -> str:
    normalized_stock = "".join(ch for ch in str(stock_code or "") if ch.isdigit())
    if len(normalized_stock) >= 6:
        return normalized_stock
    if 0 < len(normalized_stock) < 6:
        return normalized_stock.zfill(6)
    return ""


def _normalize_entity_text(text: str) -> str:
    normalized = str(text).lower()
    for token in ("주식회사", "(주)", "㈜", "주식", "회사"):
        normalized = normalized.replace(token, "")
    return "".join(ch for ch in normalized if ch.isalnum() or "\uac00" <= ch <= "\ud7a3")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _running_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
