"""Config-backed external evidence policy constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from cas.utils.io import read_yaml

_POLICY_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "evidence" / "external_evidence_policy.yaml"
)
_POLICY = read_yaml(_POLICY_PATH) if _POLICY_PATH.exists() else {}


def _section(name: str) -> dict[str, Any]:
    raw = _POLICY.get(name, {})
    return cast(dict[str, Any], raw if isinstance(raw, dict) else {})


def _tuple(section: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = section.get(key, [])
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(str(item) for item in raw)


def _float(section: dict[str, Any], key: str, default: float) -> float:
    raw = section.get(key, default)
    try:
        return float(raw) if isinstance(raw, int | float | str) else default
    except ValueError:
        return default


def _int(section: dict[str, Any], key: str, default: int) -> int:
    raw = section.get(key, default)
    try:
        return int(raw) if isinstance(raw, int | float | str) else default
    except ValueError:
        return default


_provider_limits = _section("provider_limits")
_naver = _section("naver")
_opendart = _section("opendart")

_NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_OPENDART_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
_OPENDART_CORP_CODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"
_OPENDART_DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"
_OPENDART_BUSINESS_SUSPENSION_URL = "https://opendart.fss.or.kr/api/bsnSp.json"
_OPENDART_DEFAULT_CORP_CODE_CACHE = Path(
    str(_opendart.get("default_corp_code_cache") or "data/external/opendart/corp_codes.csv")
)
_DEFAULT_TIMEOUT_SECONDS = _float(_provider_limits, "default_timeout_seconds", 8.0)
_DEFAULT_MAX_ITEMS = _int(_provider_limits, "default_max_items", 3)
_WEB_SEARCH_MAX_ITEMS = _int(_provider_limits, "web_search_max_items", 5)
_NAVER_RESULTS_PER_QUERY = _int(_provider_limits, "naver_results_per_query", 3)
_MAX_COMBINED_ITEMS = _int(_provider_limits, "max_combined_items", 12)
_MAX_WEAK_WEB_ITEMS = _int(_provider_limits, "max_weak_web_items", 3)
_DIRECT_EVIDENCE_SCORE_FLOOR = _float(_provider_limits, "direct_evidence_score_floor", 0.55)
_OPENDART_LOOKBACK_DAYS = _int(_opendart, "lookback_days", 730)
_OPENDART_PAGE_COUNT = _int(_opendart, "page_count", 10)
_OPENDART_MAX_ITEMS = _int(_opendart, "max_items", 6)
_OPENDART_MATERIALITY_LOW_RATIO = _float(_opendart, "materiality_low_ratio", 0.03)
_OPENDART_MATERIALITY_HIGH_RATIO = _float(_opendart, "materiality_high_ratio", 0.10)
_OPENDART_DISCLOSURE_TYPES = {
    str(key): str(value)
    for key, value in cast(dict[str, object], _opendart.get("disclosure_types", {})).items()
}
_OPENDART_RISK_TERMS = _tuple(_opendart, "risk_terms")
_OPENDART_VETO_TERMS = _tuple(_opendart, "veto_terms")
_OPENDART_ADVERSE_TERMS = _tuple(_opendart, "adverse_terms")
_OPENDART_CAUTION_TERMS = _tuple(_opendart, "caution_terms")
_OPENDART_ROUTINE_TERMS = _tuple(_opendart, "routine_terms")
_BENIGN_TRADING_HALT_TERMS = _tuple(_opendart, "benign_trading_halt_terms")
_HARD_DISTRESS_TERMS = _tuple(_opendart, "hard_distress_terms")
_LITIGATION_DISCLOSURE_TERMS = _tuple(_opendart, "litigation_disclosure_terms")
_LOW_MATERIALITY_DISCLOSURE_MARKERS = _tuple(_opendart, "low_materiality_disclosure_markers")
_SUPPLY_CONTRACT_CANCELLATION_TERMS = _tuple(_opendart, "supply_contract_cancellation_terms")
_FINANCING_DISCLOSURE_TERMS = _tuple(_opendart, "financing_disclosure_terms")
_DEBT_GUARANTEE_DISCLOSURE_TERMS = _tuple(_opendart, "debt_guarantee_disclosure_terms")
_PROCEDURAL_MERGER_HALT_MARKERS = _tuple(_opendart, "procedural_merger_halt_markers")
_NAVER_RISK_KEYWORDS = _tuple(_naver, "risk_keywords")
