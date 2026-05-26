"""Optional external evidence collectors used by EvidenceAuditAgent."""

from __future__ import annotations

import csv
import html
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

from cas.utils.live_cache import read_json_cache, stable_cache_key, write_json_cache
from cas.veto_rules import critical_terms_in_text

_NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_OPENDART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
_OPENDART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
_OPENDART_DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"
_OPENDART_BUSINESS_SUSPENSION_URL = "https://opendart.fss.or.kr/api/bsnSp.json"
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
_OPENDART_MATERIALITY_LOW_RATIO = 0.03
_OPENDART_MATERIALITY_HIGH_RATIO = 0.10
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
_OPENDART_VETO_TERMS = (
    "횡령",
    "배임",
    "감사의견거절",
    "감사의견 거절",
    "상장폐지",
    "부도",
    "회생절차",
    "파산",
)
_OPENDART_ADVERSE_TERMS = (
    "거래정지",
    "관리종목",
    "불성실공시",
    "자본잠식",
    "영업정지",
    "채권은행등의관리절차개시",
    "소송등의제기",
    "소송등의판결",
    "단일판매ㆍ공급계약해지",
    "단일판매·공급계약해지",
)
_OPENDART_CAUTION_TERMS = (
    "타인에대한채무보증",
    "유상증자결정",
    "전환사채권발행",
    "신주인수권부사채",
    "만기전사채취득",
    "최대주주변경",
    "감사보고서제출",
    "감사보고서 제출",
)
_OPENDART_ROUTINE_TERMS = (
    "사업보고서",
    "분기보고서",
    "반기보고서",
    "주주명부폐쇄기간",
    "기준일설정",
    "주주총회소집결의",
    "임시주주총회결과",
    "정기주주총회결과",
    "증권발행결과",
    "자기주식취득신탁계약해지",
)
_BENIGN_TRADING_HALT_TERMS = (
    "무상증자",
    "주식분할",
    "액면분할",
    "주식병합",
    "액면병합",
)
_HARD_DISTRESS_TERMS = (
    "횡령",
    "배임",
    "감사의견",
    "상장폐지",
    "관리종목",
    "불성실공시",
    "자본잠식",
    "영업정지",
    "회생",
    "파산",
    "부도",
)
_LITIGATION_DISCLOSURE_TERMS = (
    "소송등의제기",
    "소송등의판결",
    "소송등의결정",
    "소송등의 판결",
    "소송 등의 제기",
    "소송 등의 판결",
)
_LOW_MATERIALITY_DISCLOSURE_MARKERS = (
    "자율공시",
    "일정금액미만",
    "일정금액 미만",
)
_SUPPLY_CONTRACT_CANCELLATION_TERMS = (
    "단일판매ㆍ공급계약해지",
    "단일판매·공급계약해지",
    "단일판매공급계약해지",
)
_FINANCING_DISCLOSURE_TERMS = (
    "유상증자결정",
    "전환사채권발행",
    "신주인수권부사채",
    "교환사채권발행",
    "사채권발행",
)
_DEBT_GUARANTEE_DISCLOSURE_TERMS = (
    "타인에대한채무보증",
    "채무보증결정",
    "채무보증",
)
_PROCEDURAL_MERGER_HALT_MARKERS = (
    "spac",
    "스팩",
    "기업인수목적",
    "상장예비심사",
    "예비심사청구대상",
    "합병예비심사",
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
    cache_key = _external_evidence_cache_key(
        company_name=company_name,
        stock_code=stock_code,
        corp_code=corp_code,
        as_of_date=as_of_date,
        env=source,
    )
    if session is None or _cache_custom_session_enabled(source):
        cached_snapshot = _read_external_evidence_cache(cache_key, env=source)
        if cached_snapshot is not None:
            return cached_snapshot

    providers = {
        "naver_news": _collect_naver_news(
            queries=naver_queries,
            as_of_date=as_of_date,
            env=source,
            session=http,
        ),
        "tavily": _collect_tavily(
            query=query,
            as_of_date=as_of_date,
            env=source,
            session=http,
        ),
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

    snapshot = {
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
        "cache_hit": False,
        "cache_key": cache_key,
    }
    if session is None or _cache_custom_session_enabled(source):
        cache_path = write_json_cache(
            "external_evidence",
            cache_key,
            snapshot,
            env_var="CAS_EXTERNAL_EVIDENCE_CACHE_ENABLED",
            default=True,
            env=source,
        )
        if cache_path is not None:
            snapshot["cache_path"] = str(cache_path)
    return snapshot


def _external_evidence_cache_key(
    *,
    company_name: str,
    stock_code: str | None,
    corp_code: str | None,
    as_of_date: date | str | None,
    env: Mapping[str, str],
) -> str:
    return str(
        stable_cache_key(
            {
                "cache_version": "external_evidence_v7",
                "company_name": company_name,
                "stock_code": stock_code or "",
                "corp_code": corp_code or "",
                "as_of_date": _collection_end_date(as_of_date).isoformat(),
                "opendart_detail_materiality": _opendart_detail_materiality_enabled(env),
                "providers": {
                    "naver_news": bool(
                        env.get("NAVER_CLIENT_ID") and env.get("NAVER_CLIENT_SECRET")
                    ),
                    "tavily": bool(env.get("TAVILY_API_KEY")),
                    "opendart": bool(env.get("OPENDART_API_KEY")),
                },
            }
        )
    )


def _read_external_evidence_cache(
    cache_key: str,
    *,
    env: Mapping[str, str],
) -> dict[str, object] | None:
    cached_snapshot = read_json_cache(
        "external_evidence",
        cache_key,
        env_var="CAS_EXTERNAL_EVIDENCE_CACHE_ENABLED",
        default=True,
        env=env,
    )
    if cached_snapshot is None:
        return None
    snapshot = dict(cached_snapshot)
    snapshot["cache_hit"] = True
    snapshot["cache_key"] = cache_key
    return snapshot


def _cache_custom_session_enabled(env: Mapping[str, str]) -> bool:
    value = env.get("CAS_EXTERNAL_EVIDENCE_CACHE_SESSION", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _collect_naver_news(
    *,
    queries: list[str],
    as_of_date: date | str | None,
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
    items, date_filter = _filter_provider_items_by_as_of_date(items, as_of_date=as_of_date)
    if items:
        return {
            "status": "ready",
            "items": items,
            "queries": queries,
            "errors": errors,
            "as_of_date_filter": date_filter,
        }
    if errors:
        return {
            "status": "error",
            "message": "; ".join(errors),
            "items": [],
            "queries": queries,
            "as_of_date_filter": date_filter,
        }
    return {
        "status": "no_results",
        "items": [],
        "queries": queries,
        "as_of_date_filter": date_filter,
    }


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
    as_of_date: date | str | None,
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
                    "published_at": str(
                        item.get("published_date")
                        or item.get("published_at")
                        or item.get("date")
                        or ""
                    ),
                    "reliability": "medium",
                }
            )
    items, date_filter = _filter_provider_items_by_as_of_date(items, as_of_date=as_of_date)
    return {
        "status": "ready" if items else "no_results",
        "items": items,
        "as_of_date_filter": date_filter,
    }


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
    detail_materiality_enabled = _opendart_detail_materiality_enabled(env)
    business_suspension_rows: list[dict[str, object]] | None = None
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
            severity = _opendart_disclosure_severity(report_name)
            relevance = _opendart_relevance(report_name)
            event_class = _opendart_disclosure_event_class(report_name, severity=severity)
            evidence_item = {
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
                "disclosure_severity": severity,
                "disclosure_severity_reason": _opendart_severity_reason(
                    severity,
                    report_name=report_name,
                ),
                "disclosure_event_class": event_class,
                "disclosure_materiality": _opendart_disclosure_materiality(
                    severity,
                    event_class=event_class,
                ),
            }
            if detail_materiality_enabled:
                if _is_business_suspension_candidate(report_name):
                    if business_suspension_rows is None:
                        try:
                            business_suspension_rows = _fetch_opendart_business_suspensions(
                                api_key=api_key,
                                corp_code=effective_corp_code,
                                begin_date=begin_date,
                                end_date=end_date,
                                session=session,
                            )
                        except Exception as error:
                            provider_errors.append(f"bsnSp:{error}")
                            business_suspension_rows = []
                    evidence_item = _enrich_business_suspension_materiality(
                        evidence_item,
                        rows=business_suspension_rows,
                    )
                    if not evidence_item.get("materiality_ratio"):
                        try:
                            document_text = _fetch_opendart_document_text(
                                api_key=api_key,
                                receipt_no=receipt_no,
                                session=session,
                            )
                        except Exception as error:
                            provider_errors.append(f"document:{receipt_no}:{error}")
                            document_text = ""
                        evidence_item = _enrich_business_suspension_document_materiality(
                            evidence_item,
                            document_text=document_text,
                        )
                elif _requires_opendart_document_materiality(report_name):
                    try:
                        document_text = _fetch_opendart_document_text(
                            api_key=api_key,
                            receipt_no=receipt_no,
                            session=session,
                        )
                    except Exception as error:
                        provider_errors.append(f"document:{receipt_no}:{error}")
                        document_text = ""
                    evidence_item = _enrich_opendart_document_materiality(
                        evidence_item,
                        report_name=report_name,
                        document_text=document_text,
                    )
            items.append(evidence_item)
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


def _opendart_detail_materiality_enabled(env: Mapping[str, str]) -> bool:
    configured = env.get("CAS_OPENDART_DETAIL_MATERIALITY_ENABLED", "")
    return True if not configured else _truthy(configured)


def _is_business_suspension_candidate(report_name: str) -> bool:
    return "영업정지" in _compact_text(report_name)


def _is_contract_cancellation_candidate(report_name: str) -> bool:
    compact_text = _compact_text(report_name)
    return any(term in compact_text for term in _compact_terms(_SUPPLY_CONTRACT_CANCELLATION_TERMS))


def _is_financing_candidate(report_name: str) -> bool:
    compact_text = _compact_text(report_name)
    return any(term in compact_text for term in _compact_terms(_FINANCING_DISCLOSURE_TERMS))


def _is_debt_guarantee_candidate(report_name: str) -> bool:
    compact_text = _compact_text(report_name)
    return any(term in compact_text for term in _compact_terms(_DEBT_GUARANTEE_DISCLOSURE_TERMS))


def _is_litigation_candidate(report_name: str) -> bool:
    compact_text = _compact_text(report_name)
    return any(term in compact_text for term in _compact_terms(_LITIGATION_DISCLOSURE_TERMS))


def _requires_opendart_document_materiality(report_name: str) -> bool:
    return (
        _is_contract_cancellation_candidate(report_name)
        or _is_financing_candidate(report_name)
        or _is_debt_guarantee_candidate(report_name)
        or _is_litigation_candidate(report_name)
    )


def _fetch_opendart_business_suspensions(
    *,
    api_key: str,
    corp_code: str,
    begin_date: date,
    end_date: date,
    session: HttpClient,
) -> list[dict[str, object]]:
    response = session.get(
        _OPENDART_BUSINESS_SUSPENSION_URL,
        params={
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bgn_de": begin_date.strftime("%Y%m%d"),
            "end_de": end_date.strftime("%Y%m%d"),
        },
        timeout=_DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    rows = _mapping_value(payload, "list")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _fetch_opendart_document_text(
    *,
    api_key: str,
    receipt_no: str,
    session: HttpClient,
) -> str:
    if not receipt_no:
        return ""
    response = session.get(
        _OPENDART_DOCUMENT_URL,
        params={"crtfc_key": api_key, "rcept_no": receipt_no},
        timeout=_DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _opendart_document_bytes_to_text(response.content)


def _opendart_document_bytes_to_text(content: bytes) -> str:
    if not content:
        return ""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return "\n".join(
                _decode_text_bytes(archive.read(name))
                for name in archive.namelist()
                if not name.endswith("/")
            )
    except zipfile.BadZipFile:
        return _decode_text_bytes(content)


def _decode_text_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def _enrich_business_suspension_materiality(
    item: dict[str, str],
    *,
    rows: list[dict[str, object]],
) -> dict[str, str]:
    row = _matching_opendart_detail_row(rows, receipt_no=str(item.get("rcept_no", "")))
    if row is None:
        return item
    ratio, basis = _business_suspension_materiality_ratio(row)
    if ratio is None:
        return item
    scope = _business_suspension_scope(item=item, row=row)
    low_event_class = (
        "subsidiary_business_suspension_low_materiality"
        if scope == "subsidiary"
        else "business_suspension_low_materiality"
    )
    watch_event_class = (
        "subsidiary_business_suspension_watch"
        if scope == "subsidiary"
        else "business_suspension_watch"
    )
    enriched = _apply_opendart_detail_materiality(
        item,
        ratio=ratio,
        basis=basis,
        detail_source="opendart_bsnSp",
        confidence="high",
        low_event_class=low_event_class,
        watch_event_class=watch_event_class,
        high_event_class="substantive_adverse",
    )
    enriched["business_suspension_scope"] = scope
    return enriched


def _enrich_business_suspension_document_materiality(
    item: dict[str, str],
    *,
    document_text: str,
) -> dict[str, str]:
    plain_text = _plain_opendart_document_text(document_text)
    if not plain_text:
        return item
    ratio, basis = _business_suspension_document_materiality_ratio(plain_text)
    if ratio is None:
        return item
    scope = _business_suspension_scope_from_text(item=item, text=plain_text)
    low_event_class = (
        "subsidiary_business_suspension_low_materiality"
        if scope == "subsidiary"
        else "business_suspension_low_materiality"
    )
    watch_event_class = (
        "subsidiary_business_suspension_watch"
        if scope == "subsidiary"
        else "business_suspension_watch"
    )
    enriched = _apply_opendart_detail_materiality(
        item,
        ratio=ratio,
        basis=basis,
        detail_source="opendart_document_xml",
        confidence="medium",
        low_event_class=low_event_class,
        watch_event_class=watch_event_class,
        high_event_class="substantive_adverse",
    )
    enriched["business_suspension_scope"] = scope
    return enriched


def _enrich_contract_cancellation_materiality(
    item: dict[str, str],
    *,
    document_text: str,
) -> dict[str, str]:
    plain_text = _plain_opendart_document_text(document_text)
    if not plain_text:
        return item
    ratio, basis = _contract_cancellation_materiality_ratio(plain_text)
    if ratio is None:
        return item
    return _apply_opendart_detail_materiality(
        item,
        ratio=ratio,
        basis=basis,
        detail_source="opendart_document_xml",
        confidence="medium",
        low_event_class="low_materiality_contract_cancellation",
        watch_event_class="contract_cancellation_watch",
        high_event_class="material_contract_cancellation",
    )


def _enrich_opendart_document_materiality(
    item: dict[str, str],
    *,
    report_name: str,
    document_text: str,
) -> dict[str, str]:
    if _is_contract_cancellation_candidate(report_name):
        return _enrich_contract_cancellation_materiality(
            item,
            document_text=document_text,
        )
    if _is_financing_candidate(report_name):
        return _enrich_financing_materiality(item, document_text=document_text)
    if _is_debt_guarantee_candidate(report_name):
        return _enrich_debt_guarantee_materiality(item, document_text=document_text)
    if _is_litigation_candidate(report_name):
        return _enrich_litigation_materiality(item, document_text=document_text)
    return item


def _enrich_financing_materiality(
    item: dict[str, str],
    *,
    document_text: str,
) -> dict[str, str]:
    plain_text = _plain_opendart_document_text(document_text)
    if not plain_text:
        return item
    ratio, basis = _financing_materiality_ratio(plain_text)
    if ratio is None:
        return item
    enriched = _apply_opendart_detail_materiality(
        item,
        ratio=ratio,
        basis=basis,
        detail_source="opendart_document_xml",
        confidence="medium",
        low_event_class="low_materiality_financing",
        watch_event_class="financing_watch",
        high_event_class="material_financing",
    )
    dilution_ratio = _financing_dilution_ratio(plain_text)
    if dilution_ratio is not None:
        enriched["dilution_ratio"] = f"{dilution_ratio:.4f}"
        enriched["dilution_basis"] = f"희석률: {_format_materiality_ratio(dilution_ratio)}"
    return enriched


def _enrich_debt_guarantee_materiality(
    item: dict[str, str],
    *,
    document_text: str,
) -> dict[str, str]:
    plain_text = _plain_opendart_document_text(document_text)
    if not plain_text:
        return item
    ratio, basis = _debt_guarantee_materiality_ratio(plain_text)
    if ratio is None:
        return item
    return _apply_opendart_detail_materiality(
        item,
        ratio=ratio,
        basis=basis,
        detail_source="opendart_document_xml",
        confidence="medium",
        low_event_class="low_materiality_debt_guarantee",
        watch_event_class="debt_guarantee_watch",
        high_event_class="material_debt_guarantee",
    )


def _enrich_litigation_materiality(
    item: dict[str, str],
    *,
    document_text: str,
) -> dict[str, str]:
    plain_text = _plain_opendart_document_text(document_text)
    if not plain_text:
        return item
    ratio, basis = _litigation_materiality_ratio(plain_text)
    if ratio is None:
        return item
    return _apply_opendart_detail_materiality(
        item,
        ratio=ratio,
        basis=basis,
        detail_source="opendart_document_xml",
        confidence="medium",
        low_event_class="low_materiality_litigation",
        watch_event_class="litigation_watch",
        high_event_class="material_litigation",
    )


def _matching_opendart_detail_row(
    rows: list[dict[str, object]],
    *,
    receipt_no: str,
) -> dict[str, object] | None:
    for row in rows:
        if str(row.get("rcept_no", "")).strip() == receipt_no:
            return row
    return rows[0] if len(rows) == 1 else None


def _business_suspension_materiality_ratio(
    row: dict[str, object],
) -> tuple[float | None, str]:
    ratio = _parse_percent_ratio_value(row.get("sl_vs"))
    if ratio is not None:
        return ratio, "영업정지금액 매출액 대비"
    suspension_amount = _parse_amount_value(row.get("bsnsp_amt"))
    recent_sales = _parse_amount_value(row.get("rsl"))
    if suspension_amount is not None and recent_sales and recent_sales > 0:
        return suspension_amount / recent_sales, "영업정지금액/최근매출총액"
    return None, ""


def _business_suspension_scope(
    *,
    item: dict[str, str],
    row: dict[str, object],
) -> str:
    text = " ".join(
        str(part)
        for part in (
            item.get("title", ""),
            item.get("summary", ""),
            row.get("bsnsp_rm", ""),
            row.get("bsnsp_cn", ""),
            row.get("bsnsp_rs", ""),
            row.get("bsnsp_af", ""),
        )
        if part
    )
    compact_text = _compact_text(text)
    if "종속회사" in compact_text or "자회사" in compact_text:
        return "subsidiary"
    return "parent_or_direct"


def _business_suspension_scope_from_text(*, item: dict[str, str], text: str) -> str:
    compact_text = _compact_text(
        " ".join(
            part
            for part in (
                item.get("title", ""),
                item.get("summary", ""),
                text,
            )
            if part
        )
    )
    if "종속회사" in compact_text or "자회사" in compact_text:
        return "subsidiary"
    return "parent_or_direct"


def _business_suspension_document_materiality_ratio(text: str) -> tuple[float | None, str]:
    ratio = _extract_percent_ratio_near_labels(
        text,
        labels=(
            "최근매출액 대비",
            "최근매출액대비",
            "최근 사업연도 매출액 대비",
            "최근사업연도매출액대비",
            "매출액 대비",
            "매출액대비",
            "영업정지금액 비율",
            "영업정지 금액 비율",
        ),
    )
    if ratio is not None:
        return ratio, "영업정지금액 매출액 대비"
    suspension_amount = _extract_amount_near_labels(
        text,
        labels=(
            "영업정지금액",
            "영업정지 금액",
            "영업정지 분야의 매출액",
            "영업정지 분야 매출액",
            "영업정지 사업부문 매출액",
        ),
    )
    recent_sales = _extract_amount_near_labels(
        text,
        labels=(
            "최근매출액",
            "최근 매출액",
            "최근 사업연도 매출액",
            "최근사업연도매출액",
        ),
    )
    if suspension_amount is not None and recent_sales and recent_sales > 0:
        return suspension_amount / recent_sales, "영업정지금액/최근매출액"
    return None, ""


def _contract_cancellation_materiality_ratio(text: str) -> tuple[float | None, str]:
    ratio = _extract_percent_ratio_near_labels(
        text,
        labels=(
            "최근매출액 대비",
            "최근매출액대비",
            "매출액 대비",
            "매출액대비",
            "계약금액 비율",
            "해지금액 비율",
        ),
    )
    if ratio is not None:
        return ratio, "계약해지 금액 매출액 대비"
    cancellation_amount = _extract_amount_near_labels(
        text,
        labels=("계약해지금액", "해지금액", "계약금액"),
    )
    recent_sales = _extract_amount_near_labels(
        text,
        labels=("최근매출액", "최근 매출액", "매출액"),
    )
    if cancellation_amount is not None and recent_sales and recent_sales > 0:
        return cancellation_amount / recent_sales, "계약해지금액/최근매출액"
    return None, ""


def _financing_materiality_ratio(text: str) -> tuple[float | None, str]:
    equity_ratio = _extract_percent_ratio_near_labels(
        text,
        labels=(
            "자기자본 대비",
            "자기자본대비",
            "자기자본에 대한 비율",
            "자기자본에대한비율",
            "발행금액 자기자본 대비",
            "발행금액의 자기자본 대비",
            "조달금액 자기자본 대비",
        ),
    )
    dilution_ratio = _financing_dilution_ratio(text)
    candidates: list[tuple[float, str]] = []
    if equity_ratio is not None:
        candidates.append((equity_ratio, "발행금액/자기자본"))
    financing_amount = _extract_amount_near_labels(
        text,
        labels=(
            "발행금액",
            "자금조달금액",
            "조달금액",
            "사채의 권면총액",
            "권면총액",
            "전환사채 발행금액",
            "신주인수권부사채 발행금액",
            "모집총액",
        ),
    )
    equity = _extract_amount_near_labels(
        text,
        labels=("자기자본", "자본총계", "연결자기자본"),
    )
    if financing_amount is not None and equity and equity > 0:
        candidates.append((financing_amount / equity, "발행금액/자기자본"))
    if dilution_ratio is not None:
        candidates.append((dilution_ratio, "희석률"))
    if not candidates:
        return None, ""
    return max(candidates, key=lambda candidate: candidate[0])


def _financing_dilution_ratio(text: str) -> float | None:
    ratio = _extract_percent_ratio_near_labels(
        text,
        labels=(
            "증자비율",
            "희석률",
            "신주 발행비율",
            "신주발행비율",
            "발행주식총수 대비",
            "발행주식총수대비",
        ),
    )
    if ratio is not None:
        return ratio
    new_shares = _extract_amount_near_labels(
        text,
        labels=("신주의 수", "신주수", "발행할 주식수", "발행주식수"),
    )
    outstanding_shares = _extract_amount_near_labels(
        text,
        labels=("발행주식총수", "기발행주식총수", "현재 발행주식총수"),
    )
    if new_shares is not None and outstanding_shares and outstanding_shares > 0:
        return new_shares / outstanding_shares
    return None


def _debt_guarantee_materiality_ratio(text: str) -> tuple[float | None, str]:
    ratio = _extract_percent_ratio_near_labels(
        text,
        labels=(
            "자기자본 대비",
            "자기자본대비",
            "자기자본에 대한 비율",
            "자기자본에대한비율",
            "보증금액 자기자본 대비",
            "채무보증금액 자기자본 대비",
        ),
    )
    if ratio is not None:
        return ratio, "채무보증금액/자기자본"
    guarantee_amount = _extract_amount_near_labels(
        text,
        labels=("채무보증금액", "보증금액", "담보제공금액"),
    )
    equity = _extract_amount_near_labels(
        text,
        labels=("자기자본", "자본총계", "연결자기자본"),
    )
    if guarantee_amount is not None and equity and equity > 0:
        return guarantee_amount / equity, "채무보증금액/자기자본"
    return None, ""


def _litigation_materiality_ratio(text: str) -> tuple[float | None, str]:
    equity_ratio = _extract_percent_ratio_near_labels(
        text,
        labels=(
            "자기자본 대비",
            "자기자본대비",
            "자기자본에 대한 비율",
            "자기자본에대한비율",
            "청구금액 자기자본 대비",
        ),
    )
    sales_ratio = _extract_percent_ratio_near_labels(
        text,
        labels=(
            "매출액 대비",
            "매출액대비",
            "최근매출액 대비",
            "최근매출액대비",
            "청구금액 매출액 대비",
        ),
    )
    candidates: list[tuple[float, str]] = []
    if equity_ratio is not None:
        candidates.append((equity_ratio, "청구금액/자기자본"))
    if sales_ratio is not None:
        candidates.append((sales_ratio, "청구금액/매출액"))
    claim_amount = _extract_amount_near_labels(
        text,
        labels=("청구금액", "소송가액", "소가", "청구취지 금액"),
    )
    equity = _extract_amount_near_labels(
        text,
        labels=("자기자본", "자본총계", "연결자기자본"),
    )
    recent_sales = _extract_amount_near_labels(
        text,
        labels=("최근매출액", "최근 매출액", "매출액"),
    )
    if claim_amount is not None and equity and equity > 0:
        candidates.append((claim_amount / equity, "청구금액/자기자본"))
    if claim_amount is not None and recent_sales and recent_sales > 0:
        candidates.append((claim_amount / recent_sales, "청구금액/매출액"))
    if not candidates:
        return None, ""
    return max(candidates, key=lambda candidate: candidate[0])


def _apply_opendart_detail_materiality(
    item: dict[str, str],
    *,
    ratio: float,
    basis: str,
    detail_source: str,
    confidence: str,
    low_event_class: str,
    watch_event_class: str,
    high_event_class: str,
) -> dict[str, str]:
    enriched = dict(item)
    ratio_text = _format_materiality_ratio(ratio)
    enriched["materiality_ratio"] = f"{ratio:.4f}"
    enriched["materiality_basis"] = f"{basis}: {ratio_text}"
    enriched["materiality_source"] = detail_source
    enriched["materiality_confidence"] = confidence
    if ratio < _OPENDART_MATERIALITY_LOW_RATIO:
        enriched["provider_relevance"] = "caution"
        enriched["disclosure_severity"] = "caution"
        enriched["disclosure_event_class"] = low_event_class
        enriched["disclosure_materiality"] = "procedural_or_one_off"
        enriched["disclosure_severity_reason"] = (
            f"상세 공시에서 {basis} {ratio_text}로 낮은 중요도 확인"
        )
    elif ratio < _OPENDART_MATERIALITY_HIGH_RATIO:
        enriched["provider_relevance"] = "caution"
        enriched["disclosure_severity"] = "caution"
        enriched["disclosure_event_class"] = watch_event_class
        enriched["disclosure_materiality"] = "watch_context"
        enriched["disclosure_severity_reason"] = (
            f"상세 공시에서 {basis} {ratio_text}로 관찰 수준 중요도 확인"
        )
    else:
        enriched["provider_relevance"] = "risk"
        enriched["disclosure_severity"] = "adverse"
        enriched["disclosure_event_class"] = high_event_class
        enriched["disclosure_materiality"] = "substantive_adverse"
        enriched["disclosure_severity_reason"] = f"상세 공시에서 {basis} {ratio_text}로 중대성 확인"
    return enriched


def _plain_opendart_document_text(text: str) -> str:
    unescaped = html.unescape(str(text))
    unescaped = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", unescaped, flags=re.DOTALL)
    unescaped = re.sub(r"<br\s*/?>", " ", unescaped, flags=re.IGNORECASE)
    unescaped = re.sub(
        r"</(?:td|th|tr|p|div|span|section|article|table)>",
        " ",
        unescaped,
        flags=re.IGNORECASE,
    )
    unescaped = re.sub(r"<[^>]+>", " ", unescaped)
    return re.sub(r"\s+", " ", unescaped).strip()


def _extract_percent_ratio_near_labels(text: str, *, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        pattern = rf"{re.escape(label)}[^0-9\-]{{0,80}}(-?\d[\d,]*(?:\.\d+)?)\s*%?"
        match = re.search(pattern, text)
        if not match:
            continue
        ratio = _parse_percent_ratio_value(match.group(1))
        if ratio is not None:
            return ratio
    return None


def _extract_amount_near_labels(text: str, *, labels: tuple[str, ...]) -> float | None:
    for label in labels:
        pattern = rf"{re.escape(label)}[^0-9\-]{{0,80}}(-?\d[\d,]*(?:\.\d+)?)"
        match = re.search(pattern, text)
        if not match:
            continue
        amount = _parse_amount_value(match.group(1))
        if amount is not None:
            return amount
    return None


def _parse_percent_ratio_value(value: object) -> float | None:
    number = _parse_amount_value(value)
    if number is None or number < 0:
        return None
    return number / 100


def _parse_amount_value(value: object) -> float | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _format_materiality_ratio(ratio: float) -> str:
    return f"{ratio * 100:.2f}%"


def _prioritized_opendart_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = _dedupe_raw_opendart_items(items)
    severity_order = {"veto": 4, "adverse": 3, "caution": 2, "routine": 1}
    unique.sort(
        key=lambda item: (
            severity_order.get(str(item.get("disclosure_severity", "")), 0),
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
    severity = _opendart_disclosure_severity(report_name)
    if severity in {"veto", "adverse"}:
        return "risk"
    if severity == "caution":
        return "caution"
    if severity == "routine":
        return "routine"
    if any(
        term in report_name for term in ("사업보고서", "반기보고서", "분기보고서", "감사보고서")
    ):
        return "context"
    return "routine"


def _opendart_disclosure_severity(report_name: str) -> str:
    return _disclosure_term_severity(report_name) or "routine"


def _disclosure_like_severity(title: str, summary: str) -> str:
    """Classify DART-like search snippets without treating every news item as a filing."""
    return _disclosure_term_severity(f"{title} {summary}") or "unknown"


def _disclosure_term_severity(text: str) -> str:
    if _contains_any_term(text, _OPENDART_VETO_TERMS):
        return "veto"
    benign_trading_halt_severity = _benign_trading_halt_severity(text)
    if benign_trading_halt_severity:
        return benign_trading_halt_severity
    nonmaterial_disclosure_severity = _nonmaterial_or_procedural_disclosure_severity(text)
    if nonmaterial_disclosure_severity:
        return nonmaterial_disclosure_severity
    if _contains_any_term(text, _OPENDART_ADVERSE_TERMS):
        return "adverse"
    if _contains_any_term(text, _OPENDART_CAUTION_TERMS):
        return "caution"
    if _contains_any_term(text, _OPENDART_ROUTINE_TERMS):
        return "routine"
    if _contains_any_term(text, _OPENDART_RISK_TERMS):
        return "adverse"
    return ""


def _nonmaterial_or_procedural_disclosure_severity(text: str) -> str:
    """Downgrade one-off or procedural disclosures before broad adverse terms match."""
    compact_text = _compact_text(text)
    if any(term in compact_text for term in _compact_terms(_HARD_DISTRESS_TERMS)):
        return ""
    if _is_low_materiality_litigation_disclosure(compact_text):
        return "caution"
    if _is_voluntary_supply_contract_cancellation(compact_text):
        return "caution"
    if _is_procedural_merger_trading_halt(compact_text):
        return "caution"
    return ""


def _is_low_materiality_litigation_disclosure(compact_text: str) -> bool:
    has_litigation = any(
        term in compact_text for term in _compact_terms(_LITIGATION_DISCLOSURE_TERMS)
    )
    if not has_litigation:
        return False
    return any(
        marker in compact_text for marker in _compact_terms(_LOW_MATERIALITY_DISCLOSURE_MARKERS)
    )


def _is_voluntary_supply_contract_cancellation(compact_text: str) -> bool:
    has_contract_cancellation = any(
        term in compact_text for term in _compact_terms(_SUPPLY_CONTRACT_CANCELLATION_TERMS)
    )
    return has_contract_cancellation and "자율공시" in compact_text


def _is_procedural_merger_trading_halt(compact_text: str) -> bool:
    if "거래정지" not in compact_text or "합병" not in compact_text:
        return False
    return any(marker in compact_text for marker in _compact_terms(_PROCEDURAL_MERGER_HALT_MARKERS))


def _benign_trading_halt_severity(text: str) -> str:
    """Downgrade procedural trading halts that are tied to benign corporate actions."""
    lowered = str(text).lower()
    compact_text = _compact_text(lowered)
    has_trading_halt = "거래정지" in lowered or "거래정지" in compact_text
    if not has_trading_halt:
        return ""

    adverse_terms_without_trading_halt = tuple(
        term for term in _OPENDART_ADVERSE_TERMS if term != "거래정지"
    )
    if _contains_any_term(text, adverse_terms_without_trading_halt):
        return ""
    if _contains_any_term(text, _BENIGN_TRADING_HALT_TERMS):
        return "routine"
    if "거래정지해제" in lowered or "거래정지해제" in compact_text:
        return "caution"
    return ""


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = str(text).lower()
    compact_text = _compact_text(lowered)
    return any(
        term.lower() in lowered or "".join(term.lower().split()) in compact_text for term in terms
    )


def _compact_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_compact_text(term) for term in terms)


def _compact_text(text: str) -> str:
    return "".join(str(text).lower().split())


def _opendart_disclosure_event_class(report_name: str, *, severity: str) -> str:
    compact_text = _compact_text(report_name)
    if severity == "veto":
        return "veto_event"
    if _is_procedural_merger_trading_halt(compact_text) or _benign_trading_halt_severity(
        report_name
    ):
        return "procedural_trading_halt"
    if _is_low_materiality_litigation_disclosure(compact_text):
        return "low_materiality_litigation"
    if _is_voluntary_supply_contract_cancellation(compact_text):
        return "one_off_contract_cancellation"
    if any(term in compact_text for term in _compact_terms(_SUPPLY_CONTRACT_CANCELLATION_TERMS)):
        return "material_contract_cancellation"
    if any(term in compact_text for term in _compact_terms(_FINANCING_DISCLOSURE_TERMS)):
        return "financing_watch"
    if any(term in compact_text for term in _compact_terms(_DEBT_GUARANTEE_DISCLOSURE_TERMS)):
        return "debt_guarantee_watch"
    if any(term in compact_text for term in _compact_terms(_LITIGATION_DISCLOSURE_TERMS)):
        return "material_litigation"
    if any(term in compact_text for term in _compact_terms(_OPENDART_CAUTION_TERMS)):
        return "financing_or_governance_watch"
    if severity == "routine":
        return "routine_context"
    if severity == "adverse":
        return "substantive_adverse"
    return "unclassified"


def _opendart_disclosure_materiality(
    severity: str,
    *,
    event_class: str,
) -> str:
    if severity == "veto":
        return "critical"
    if severity == "adverse":
        return "substantive_adverse"
    if event_class in {
        "procedural_trading_halt",
        "low_materiality_litigation",
        "one_off_contract_cancellation",
        "low_materiality_contract_cancellation",
        "business_suspension_low_materiality",
        "subsidiary_business_suspension_low_materiality",
        "low_materiality_financing",
        "low_materiality_debt_guarantee",
    }:
        return "procedural_or_one_off"
    if event_class in {
        "contract_cancellation_watch",
        "business_suspension_watch",
        "subsidiary_business_suspension_watch",
        "financing_watch",
        "debt_guarantee_watch",
        "litigation_watch",
    }:
        return "watch_context"
    if severity == "caution":
        return "watch_context"
    return "routine_context"


def _opendart_severity_reason(severity: str, *, report_name: str = "") -> str:
    event_class = (
        _opendart_disclosure_event_class(report_name, severity=severity) if report_name else ""
    )
    if event_class == "procedural_trading_halt":
        return "절차성 거래정지/해제 공시"
    if event_class == "low_materiality_litigation":
        return "일정금액 미만 또는 자율공시 소송 공시"
    if event_class == "one_off_contract_cancellation":
        return "자율공시 단일 계약해지 공시"
    return {
        "veto": "강제 경고 후보 공시",
        "adverse": "실질 위험 점검 공시",
        "caution": "주의 관찰 공시",
        "routine": "정기/일상 공시",
    }.get(severity, "공시 성격 미분류")


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
