"""OpenDART disclosure classification and materiality parsing."""

from __future__ import annotations

import html
import io
import re
import zipfile
from collections.abc import Mapping
from datetime import date

from cas.evidence.http_client import HttpClient
from cas.evidence.policy import (
    _BENIGN_TRADING_HALT_TERMS,
    _DEBT_GUARANTEE_DISCLOSURE_TERMS,
    _DEFAULT_TIMEOUT_SECONDS,
    _FINANCING_DISCLOSURE_TERMS,
    _HARD_DISTRESS_TERMS,
    _LITIGATION_DISCLOSURE_TERMS,
    _LOW_MATERIALITY_DISCLOSURE_MARKERS,
    _OPENDART_ADVERSE_TERMS,
    _OPENDART_BUSINESS_SUSPENSION_URL,
    _OPENDART_CAUTION_TERMS,
    _OPENDART_DOCUMENT_URL,
    _OPENDART_MATERIALITY_HIGH_RATIO,
    _OPENDART_MATERIALITY_LOW_RATIO,
    _OPENDART_MAX_ITEMS,
    _OPENDART_RISK_TERMS,
    _OPENDART_ROUTINE_TERMS,
    _OPENDART_VETO_TERMS,
    _PROCEDURAL_MERGER_HALT_MARKERS,
    _SUPPLY_CONTRACT_CANCELLATION_TERMS,
)
from cas.evidence.utils import _mapping_value, _truthy


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
