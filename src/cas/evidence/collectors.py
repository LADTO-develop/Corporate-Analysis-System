"""Optional external evidence collectors used by EvidenceAuditAgent."""

from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse, urlunparse
from xml.etree import ElementTree

import requests

from cas.veto_rules import critical_terms_in_text

_NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_OPENDART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
_OPENDART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
_OPENDART_DEFAULT_CORP_CODE_CACHE = Path("data/external/opendart/corp_codes.csv")
_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_MAX_ITEMS = 3
_WEB_SEARCH_MAX_ITEMS = 5
_NAVER_RESULTS_PER_QUERY = 3
_MAX_COMBINED_ITEMS = 12
_MAX_WEAK_WEB_ITEMS = 3
_OPENDART_LOOKBACK_DAYS = 730
_OPENDART_PAGE_COUNT = 10
_OPENDART_MAX_ITEMS = 6
_OPENDART_DISCLOSURE_TYPES = {
    "B": "주요사항보고",
    "F": "외부감사관련",
    "I": "거래소공시",
    "A": "정기공시",
}
_OPENDART_RISK_TERMS = (
    "횡령",
    "배임",
    "감사의견",
    "상장폐지",
    "거래정지",
    "회생",
    "부도",
    "소송",
    "관리종목",
    "불성실공시",
    "자본잠식",
)
_NAVER_RISK_KEYWORDS = (
    "소송",
    "횡령",
    "배임",
    "감사의견",
    "상장폐지",
    "유동성",
    "차입금",
    "회사채",
)
_DIRECT_EVIDENCE_SCORE_FLOOR = 0.55


class HttpResponse(Protocol):
    """Small response protocol used to keep collectors testable."""

    @property
    def content(self) -> bytes:
        """Return raw response bytes when an endpoint is not JSON."""

    def raise_for_status(self) -> None:
        """Raise when the HTTP response indicates failure."""

    def json(self) -> object:
        """Return the decoded JSON body."""


class HttpClient(Protocol):
    """Small HTTP client protocol compatible with requests.Session."""

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        """Send an HTTP GET request."""

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        """Send an HTTP POST request."""


def external_evidence_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether live external evidence calls are explicitly enabled."""
    if (
        env is None
        and _running_pytest()
        and not _truthy(os.environ.get("CAS_ALLOW_LIVE_EXTERNAL_EVIDENCE_IN_TESTS", ""))
    ):
        return False
    source = os.environ if env is None else env
    value = source.get("CAS_ENABLE_EXTERNAL_EVIDENCE", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def collect_external_evidence(
    *,
    company_name: str,
    stock_code: str | None = None,
    corp_code: str | None = None,
    as_of_date: date | str | None = None,
    env: Mapping[str, str] | None = None,
    session: HttpClient | None = None,
) -> dict[str, object]:
    """Collect optional Naver, Tavily, and OpenDART evidence into one snapshot."""
    source = os.environ if env is None else env
    if not external_evidence_enabled(source):
        return {
            "status": "disabled",
            "source": "external_evidence",
            "enabled": False,
            "items": [],
            "providers": {},
            "has_critical_risk": False,
            "critical_terms": [],
            "message": "Set CAS_ENABLE_EXTERNAL_EVIDENCE=1 to enable live evidence calls.",
        }

    http = session or cast(HttpClient, requests.Session())
    query = _risk_query(company_name=company_name, stock_code=stock_code)
    naver_queries = _naver_news_queries(company_name=company_name)
    providers = {
        "naver_news": _collect_naver_news(queries=naver_queries, env=source, session=http),
        "tavily": _collect_tavily(query=query, env=source, session=http),
        "opendart": _collect_opendart(
            company_name=company_name,
            stock_code=stock_code,
            corp_code=corp_code,
            as_of_date=as_of_date,
            env=source,
            session=http,
        ),
    }
    items = _combined_items(providers, company_name=company_name, stock_code=stock_code)
    critical_terms = _critical_terms(items)
    direct_match_count = sum(1 for item in items if item.get("company_match") is True)
    veto_candidate_count = sum(1 for item in items if item.get("veto_candidate") is True)
    verified_item_count = sum(
        1 for item in items if _evidence_score_from_item(item) >= _DIRECT_EVIDENCE_SCORE_FLOOR
    )
    high_confidence_critical_count = sum(
        1 for item in items if item.get("critical_context_confirmed") is True
    )
    provider_statuses = [str(provider.get("status", "unknown")) for provider in providers.values()]
    if items:
        status = "ready"
    elif all(status == "missing_key" for status in provider_statuses):
        status = "missing_credentials"
    elif any(status == "error" for status in provider_statuses):
        status = "partial_error"
    else:
        status = "no_results"

    return {
        "status": status,
        "source": "external_evidence",
        "enabled": True,
        "company_name": company_name,
        "stock_code": stock_code or "",
        "corp_code": str(dict(providers.get("opendart") or {}).get("corp_code") or corp_code or ""),
        "as_of_date": _collection_end_date(as_of_date).isoformat(),
        "query": query,
        "naver_queries": naver_queries,
        "fetched_at": _now(),
        "items": items,
        "providers": providers,
        "has_critical_risk": bool(critical_terms),
        "critical_terms": critical_terms,
        "direct_match_count": direct_match_count,
        "weak_evidence_count": max(len(items) - direct_match_count, 0),
        "verified_item_count": verified_item_count,
        "veto_candidate_count": veto_candidate_count,
        "high_confidence_critical_count": high_confidence_critical_count,
        "verification_summary": _verification_summary(items),
    }


def _collect_naver_news(
    *,
    queries: list[str],
    env: Mapping[str, str],
    session: HttpClient,
) -> dict[str, object]:
    client_id = env.get("NAVER_CLIENT_ID", "")
    client_secret = env.get("NAVER_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return _provider_result("missing_key", "NAVER_CLIENT_ID/NAVER_CLIENT_SECRET not set.")

    items: list[dict[str, str]] = []
    errors: list[str] = []
    for query in queries:
        try:
            response = session.get(
                _NAVER_NEWS_URL,
                params={"query": query, "display": _NAVER_RESULTS_PER_QUERY, "sort": "date"},
                headers={
                    "X-Naver-Client-Id": client_id,
                    "X-Naver-Client-Secret": client_secret,
                },
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            errors.append(f"{query}: {error}")
            continue

        raw_items = _mapping_value(payload, "items")
        if not isinstance(raw_items, list):
            continue
        for item in raw_items[:_NAVER_RESULTS_PER_QUERY]:
            if not isinstance(item, dict):
                continue
            title = _strip_html(str(item.get("title", "")))
            description = _strip_html(str(item.get("description", "")))
            items.append(
                {
                    "source": "naver_news",
                    "title": title,
                    "summary": description,
                    "url": str(item.get("originallink") or item.get("link") or ""),
                    "published_at": str(item.get("pubDate", "")),
                    "reliability": "medium",
                }
            )
    items = _dedupe_raw_provider_items(items)
    if items:
        return {"status": "ready", "items": items, "queries": queries, "errors": errors}
    if errors:
        return {"status": "error", "message": "; ".join(errors), "items": [], "queries": queries}
    return {"status": "no_results", "items": [], "queries": queries}


def _dedupe_raw_provider_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[str, dict[str, str]] = {}
    ordered: list[str] = []
    for item in items:
        key = _raw_provider_item_key(item)
        if key not in by_key:
            by_key[key] = item
            ordered.append(key)
    return [by_key[key] for key in ordered]


def _raw_provider_item_key(item: dict[str, str]) -> str:
    canonical_url = _canonical_url(item.get("url", ""))
    if canonical_url:
        return f"url:{canonical_url}"
    title = _normalize_entity_text(item.get("title", ""))
    return f"title:{title}" if title else f"item:{id(item)}"


def _collect_tavily(
    *,
    query: str,
    env: Mapping[str, str],
    session: HttpClient,
) -> dict[str, object]:
    api_key = env.get("TAVILY_API_KEY", "")
    if not api_key:
        return _provider_result("missing_key", "TAVILY_API_KEY not set.")

    try:
        response = session.post(
            _TAVILY_SEARCH_URL,
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": _WEB_SEARCH_MAX_ITEMS,
                "include_answer": False,
            },
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as error:
        return _provider_result("error", str(error))

    raw_results = _mapping_value(payload, "results")
    items: list[dict[str, str]] = []
    if isinstance(raw_results, list):
        for item in raw_results[:_WEB_SEARCH_MAX_ITEMS]:
            if not isinstance(item, dict):
                continue
            items.append(
                {
                    "source": "tavily",
                    "title": str(item.get("title", "")),
                    "summary": str(item.get("content", "")),
                    "url": str(item.get("url", "")),
                    "published_at": "",
                    "reliability": "medium",
                }
            )
    return {"status": "ready" if items else "no_results", "items": items}


def _collect_opendart(
    *,
    company_name: str,
    stock_code: str | None,
    corp_code: str | None,
    as_of_date: date | str | None,
    env: Mapping[str, str],
    session: HttpClient,
) -> dict[str, object]:
    api_key = env.get("OPENDART_API_KEY", "")
    if not api_key:
        return _provider_result("missing_key", "OPENDART_API_KEY not set.")
    effective_corp_code = _normalize_corp_code(corp_code) or _resolve_opendart_corp_code(
        stock_code=stock_code,
        env=env,
        session=session,
    )
    if not effective_corp_code:
        return _provider_result(
            "missing_corp_code",
            "OpenDART disclosure search needs corp_code or a stock_code that can be mapped.",
        )

    end_date = _collection_end_date(as_of_date)
    begin_date = end_date - timedelta(days=_OPENDART_LOOKBACK_DAYS)
    items: list[dict[str, str]] = []
    provider_errors: list[str] = []
    for disclosure_type, disclosure_label in _OPENDART_DISCLOSURE_TYPES.items():
        try:
            response = session.get(
                _OPENDART_LIST_URL,
                params={
                    "crtfc_key": api_key,
                    "corp_code": effective_corp_code,
                    "bgn_de": begin_date.strftime("%Y%m%d"),
                    "end_de": end_date.strftime("%Y%m%d"),
                    "pblntf_ty": disclosure_type,
                    "sort": "date",
                    "sort_mth": "desc",
                    "page_no": 1,
                    "page_count": _OPENDART_PAGE_COUNT,
                },
                timeout=_DEFAULT_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            provider_errors.append(f"{disclosure_type}:{error}")
            continue

        raw_reports = _mapping_value(payload, "list")
        if not isinstance(raw_reports, list):
            continue
        for item in raw_reports:
            if not isinstance(item, dict):
                continue
            report_name = str(item.get("report_nm", ""))
            receipt_no = str(item.get("rcept_no", ""))
            receipt_date = str(item.get("rcept_dt", ""))
            relevance = _opendart_relevance(report_name)
            items.append(
                {
                    "source": "opendart",
                    "title": report_name,
                    "summary": (f"{company_name} OpenDART {disclosure_label} 공시: {report_name}"),
                    "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}"
                    if receipt_no
                    else "",
                    "published_at": receipt_date,
                    "reliability": "high",
                    "corp_code": effective_corp_code,
                    "rcept_no": receipt_no,
                    "disclosure_type": disclosure_type,
                    "disclosure_type_label": disclosure_label,
                    "provider_relevance": relevance,
                }
            )
    items = _prioritized_opendart_items(items)
    if items:
        status = "ready"
    elif provider_errors:
        status = "error"
    else:
        status = "no_results"
    return {
        "status": status,
        "items": items,
        "corp_code": effective_corp_code,
        "query_window": {
            "begin_date": begin_date.isoformat(),
            "end_date": end_date.isoformat(),
            "lookback_days": _OPENDART_LOOKBACK_DAYS,
        },
        "disclosure_types": list(_OPENDART_DISCLOSURE_TYPES),
        "errors": provider_errors,
    }


def _provider_result(status: str, message: str) -> dict[str, object]:
    return {"status": status, "message": message, "items": []}


def _resolve_opendart_corp_code(
    *,
    stock_code: str | None,
    env: Mapping[str, str],
    session: HttpClient,
) -> str | None:
    normalized_stock = _normalize_stock_code(stock_code)
    if not normalized_stock:
        return None
    cached = _lookup_cached_corp_code(normalized_stock, env=env)
    if cached:
        return cached
    if not env.get("OPENDART_API_KEY", ""):
        return None
    try:
        _refresh_opendart_corp_code_cache(env=env, session=session)
    except Exception:
        return None
    return _lookup_cached_corp_code(normalized_stock, env=env)


def _lookup_cached_corp_code(stock_code: str, *, env: Mapping[str, str]) -> str | None:
    cache_path = _corp_code_cache_path(env)
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if _normalize_stock_code(row.get("stock_code")) == stock_code:
                return _normalize_corp_code(row.get("corp_code"))
    return None


def _refresh_opendart_corp_code_cache(
    *,
    env: Mapping[str, str],
    session: HttpClient,
) -> None:
    api_key = env.get("OPENDART_API_KEY", "")
    if not api_key:
        return
    response = session.get(
        _OPENDART_CORP_CODE_URL,
        params={"crtfc_key": api_key},
        timeout=_DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    rows = _parse_opendart_corp_code_zip(response.content)
    if not rows:
        return
    cache_path = _corp_code_cache_path(env)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["corp_code", "corp_name", "stock_code", "modify_date"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _parse_opendart_corp_code_zip(content: bytes) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        xml_name = next((name for name in archive.namelist() if name.lower().endswith(".xml")), "")
        if not xml_name:
            return []
        root = ElementTree.fromstring(archive.read(xml_name))
    for element in root.findall("list"):
        stock_code = _text_from_xml(element, "stock_code")
        if not stock_code:
            continue
        rows.append(
            {
                "corp_code": _normalize_corp_code(_text_from_xml(element, "corp_code")) or "",
                "corp_name": _text_from_xml(element, "corp_name"),
                "stock_code": _normalize_stock_code(stock_code) or "",
                "modify_date": _text_from_xml(element, "modify_date"),
            }
        )
    return rows


def _text_from_xml(element: ElementTree.Element, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def _corp_code_cache_path(env: Mapping[str, str]) -> Path:
    configured = env.get("CAS_OPENDART_CORP_CODE_CACHE_PATH", "")
    return Path(configured) if configured else _OPENDART_DEFAULT_CORP_CODE_CACHE


def _prioritized_opendart_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = _dedupe_raw_opendart_items(items)
    unique.sort(
        key=lambda item: (
            1 if item.get("provider_relevance") == "risk" else 0,
            item.get("published_at", ""),
        ),
        reverse=True,
    )
    return unique[:_OPENDART_MAX_ITEMS]


def _dedupe_raw_opendart_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    by_receipt: dict[str, dict[str, str]] = {}
    ordered: list[str] = []
    for item in items:
        key = item.get("rcept_no") or f"{item.get('title', '')}:{item.get('published_at', '')}"
        if key not in by_receipt:
            by_receipt[key] = item
            ordered.append(key)
    return [by_receipt[key] for key in ordered]


def _opendart_relevance(report_name: str) -> str:
    if any(term in report_name for term in _OPENDART_RISK_TERMS):
        return "risk"
    if any(
        term in report_name for term in ("사업보고서", "반기보고서", "분기보고서", "감사보고서")
    ):
        return "context"
    return "routine"


def _collection_end_date(as_of_date: date | str | None) -> date:
    if isinstance(as_of_date, date):
        return min(as_of_date, date.today())
    if isinstance(as_of_date, str) and as_of_date.strip():
        try:
            return min(date.fromisoformat(as_of_date.strip()), date.today())
        except ValueError:
            return date.today()
    return date.today()


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
    text = f"{title} {summary} {url}"
    critical_terms = critical_terms_in_text(text)
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

    if provider_relevance == "risk":
        score += 0.06
    elif provider_relevance == "routine":
        score -= 0.04

    recency_bucket = _recency_bucket(published_at)
    if recency_bucket == "older_than_2y":
        score -= 0.06
    elif recency_bucket == "unknown":
        score -= 0.03
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
    aliases = _entity_aliases(company_name=company_name, stock_code=stock_code)
    normalized_terms = [_normalize_entity_text(term) for term in critical_terms]
    if _critical_context_scope_excluded(normalized_text, normalized_terms):
        return False
    for alias in aliases:
        alias_index = normalized_text.find(alias)
        if alias_index < 0:
            continue
        for term in normalized_terms:
            term_index = normalized_text.find(term)
            if term_index >= 0 and abs(alias_index - term_index) <= 80:
                return True
    return False


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
    text = " ".join(f"{item.get('title', '')} {item.get('summary', '')}".lower() for item in items)
    return cast(list[str], critical_terms_in_text(text))


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
    if any(alias in normalized_text for alias in _company_name_aliases(company_name)):
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
