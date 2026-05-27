"""Optional external evidence collectors used by EvidenceAuditAgent."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import date
from typing import cast

import requests

from cas.evidence.http_client import HttpClient, HttpResponse
from cas.evidence.opendart_materiality import _opendart_detail_materiality_enabled
from cas.evidence.policy import _DIRECT_EVIDENCE_SCORE_FLOOR
from cas.evidence.providers import _collect_naver_news, _collect_opendart, _collect_tavily
from cas.evidence.scoring import (
    _combined_items,
    _critical_terms,
    _evidence_score_from_item,
    _verification_summary,
)
from cas.evidence.utils import (
    _collection_end_date,
    _naver_news_queries,
    _now,
    _risk_query,
    _running_pytest,
    _truthy,
)
from cas.utils.live_cache import read_json_cache, stable_cache_key, write_json_cache


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
                "cache_version": "external_evidence_v8",
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


__all__ = ["HttpClient", "HttpResponse", "collect_external_evidence", "external_evidence_enabled"]
