"""Provider clients for Naver, Tavily, and OpenDART evidence collection."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree

from cas.evidence.http_client import HttpClient
from cas.evidence.opendart_materiality import (
    _enrich_business_suspension_document_materiality,
    _enrich_business_suspension_materiality,
    _enrich_opendart_document_materiality,
    _fetch_opendart_business_suspensions,
    _fetch_opendart_document_text,
    _is_business_suspension_candidate,
    _opendart_detail_materiality_enabled,
    _opendart_disclosure_event_class,
    _opendart_disclosure_materiality,
    _opendart_disclosure_severity,
    _opendart_relevance,
    _opendart_severity_reason,
    _prioritized_opendart_items,
    _requires_opendart_document_materiality,
)
from cas.evidence.policy import (
    _DEFAULT_TIMEOUT_SECONDS,
    _NAVER_NEWS_URL,
    _NAVER_RESULTS_PER_QUERY,
    _OPENDART_CORP_CODE_URL,
    _OPENDART_DEFAULT_CORP_CODE_CACHE,
    _OPENDART_DISCLOSURE_TYPES,
    _OPENDART_LIST_URL,
    _OPENDART_LOOKBACK_DAYS,
    _OPENDART_PAGE_COUNT,
    _TAVILY_SEARCH_URL,
    _WEB_SEARCH_MAX_ITEMS,
)
from cas.evidence.utils import (
    _canonical_url,
    _collection_end_date,
    _filter_provider_items_by_as_of_date,
    _mapping_value,
    _normalize_corp_code,
    _normalize_entity_text,
    _normalize_stock_code,
    _strip_html,
)


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
                corp_code = _normalize_corp_code(row.get("corp_code"))
                return str(corp_code) if corp_code is not None else None
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
