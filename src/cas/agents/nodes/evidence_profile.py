"""External evidence profile helpers for Stage 2 committee review."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from cas.agents.signals.materiality_signals import (
    has_substantive_external_risk as _shared_has_substantive_external_risk,
)
from cas.agents.signals.materiality_signals import (
    substantive_external_risk_item as _shared_substantive_external_risk_item,
)
from cas.agents.stage2_bundle import Stage2InputBundle

_EvidenceStrength = Literal["none", "weak", "moderate", "strong", "critical"]


class _EvidenceProfile(TypedDict):
    status: str
    strength: _EvidenceStrength
    finding: str
    item_count: int
    direct_count: int
    verified_count: int
    weak_count: int
    adverse_count: int
    verified_adverse_count: int
    veto_candidate_count: int
    high_confidence_critical_count: int
    critical_terms: list[str]
    score: float


def _external_evidence_quality(
    news_cache: dict[str, Any],
    *,
    veto_triggered: bool,
) -> float:
    status = str(news_cache.get("status", "not_implemented"))
    if status in {"disabled", "not_implemented", "placeholder", "missing_credentials"}:
        return 0.35
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list) or not raw_items:
        return 0.4

    verified_count = _safe_int(news_cache.get("verified_item_count"))
    direct_count = sum(
        1 for item in raw_items if isinstance(item, dict) and item.get("company_match") is True
    )
    weak_count = sum(
        1 for item in raw_items if isinstance(item, dict) and item.get("company_match") is False
    )
    high_reliability_count = sum(
        1
        for item in raw_items
        if isinstance(item, dict)
        and (
            str(item.get("reliability", "")).lower() == "high"
            or str(item.get("source", "")).lower() == "opendart"
        )
    )
    average_item_score = _average_evidence_item_score(raw_items)
    score = 0.38 + 0.07 * min(verified_count, 3) + 0.04 * min(direct_count, 3)
    score += 0.08 * min(high_reliability_count, 2) + 0.15 * average_item_score
    score -= 0.05 * min(weak_count, 3)
    if veto_triggered:
        score += 0.15
    elif news_cache.get("has_critical_risk"):
        score -= 0.08
    return _clamp(score, minimum=0.2, maximum=0.85)


def _average_evidence_item_score(raw_items: list[object]) -> float:
    scores: list[float] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        score = item.get("evidence_score")
        if isinstance(score, int | float | str):
            try:
                scores.append(_clamp(float(score)))
            except ValueError:
                continue
    if not scores:
        return 0.35
    return sum(scores) / len(scores)


def _safe_int(value: object) -> int:
    try:
        return int(value) if isinstance(value, int | float | str) else 0
    except (TypeError, ValueError):
        return 0


def _clamp(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(max(value, minimum), maximum)


def _external_evidence_profile(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> _EvidenceProfile:
    status = str(news_cache.get("status", "not_implemented"))
    raw_items = news_cache.get("items", [])
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    item_count = len(items)
    direct_count = _safe_int(news_cache.get("direct_match_count"))
    if direct_count == 0:
        direct_count = sum(1 for item in items if item.get("company_match") is True)
    weak_count = _safe_int(news_cache.get("weak_evidence_count"))
    if weak_count == 0:
        weak_count = sum(1 for item in items if item.get("company_match") is not True)
    verified_count = _safe_int(news_cache.get("verified_item_count"))
    if verified_count == 0:
        verified_count = sum(1 for item in items if _is_verified_evidence_item(item))
    adverse_items = [
        item
        for item in items
        if _is_adverse_evidence_item(item, source_feature_row=source_feature_row)
    ]
    adverse_count = len(adverse_items)
    verified_adverse_count = sum(1 for item in adverse_items if _is_verified_evidence_item(item))
    veto_candidate_count = _safe_int(news_cache.get("veto_candidate_count"))
    if veto_candidate_count == 0:
        veto_candidate_count = sum(1 for item in items if item.get("veto_candidate") is True)
    high_confidence_critical_count = _safe_int(news_cache.get("high_confidence_critical_count"))
    if high_confidence_critical_count == 0:
        high_confidence_critical_count = sum(
            1 for item in items if _is_high_confidence_external_critical_item(item)
        )
    critical_terms = [str(term) for term in news_cache.get("critical_terms", []) or []]
    strength = _evidence_strength(
        status=status,
        item_count=item_count,
        direct_count=direct_count,
        verified_count=verified_count,
        adverse_count=adverse_count,
        verified_adverse_count=verified_adverse_count,
        veto_candidate_count=veto_candidate_count,
        high_confidence_critical_count=high_confidence_critical_count,
    )
    score = _evidence_strength_score(strength)
    return {
        "status": status,
        "strength": strength,
        "finding": _evidence_profile_finding(
            status=status,
            strength=strength,
            item_count=item_count,
            direct_count=direct_count,
            verified_count=verified_count,
            weak_count=weak_count,
            adverse_count=adverse_count,
            verified_adverse_count=verified_adverse_count,
            veto_candidate_count=veto_candidate_count,
            critical_terms=critical_terms,
        ),
        "item_count": item_count,
        "direct_count": direct_count,
        "verified_count": verified_count,
        "weak_count": weak_count,
        "adverse_count": adverse_count,
        "verified_adverse_count": verified_adverse_count,
        "veto_candidate_count": veto_candidate_count,
        "high_confidence_critical_count": high_confidence_critical_count,
        "critical_terms": critical_terms,
        "score": score,
    }


def _is_verified_evidence_item(item: dict[str, Any]) -> bool:
    score = _safe_float(item.get("evidence_score"))
    return score is not None and score >= 0.55


def _is_high_confidence_external_critical_item(item: dict[str, Any]) -> bool:
    if item.get("critical_context_confirmed") is not True:
        return False
    if item.get("as_of_date_violation") is True:
        return False
    source = str(item.get("source") or "").lower()
    if source == "opendart":
        return True
    return str(item.get("company_disambiguation") or "").lower() in {
        "resolved_by_name_and_stock_code",
        "resolved_by_disclosure_corp_code",
    }


def _is_adverse_evidence_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    return bool(
        _shared_substantive_external_risk_item(
            item,
            source_feature_row=source_feature_row,
        )
    )


def _evidence_strength(
    *,
    status: str,
    item_count: int,
    direct_count: int,
    verified_count: int,
    adverse_count: int,
    verified_adverse_count: int,
    veto_candidate_count: int,
    high_confidence_critical_count: int,
) -> _EvidenceStrength:
    if status in {"disabled", "not_implemented", "placeholder", "missing_credentials"}:
        return "none"
    if item_count <= 0:
        return "none"
    if veto_candidate_count >= 2 and high_confidence_critical_count >= 1:
        return "critical"
    if veto_candidate_count >= 1 or high_confidence_critical_count >= 1:
        return "strong"
    if verified_adverse_count >= 1:
        return "strong"
    if adverse_count >= 1:
        return "moderate"
    if direct_count >= 1 and verified_count >= 1:
        return "weak"
    return "weak"


def _evidence_strength_score(strength: _EvidenceStrength) -> float:
    return {
        "none": 0.0,
        "weak": 0.18,
        "moderate": 0.38,
        "strong": 0.62,
        "critical": 0.85,
    }[strength]


def _evidence_profile_finding(
    *,
    status: str,
    strength: _EvidenceStrength,
    item_count: int,
    direct_count: int,
    verified_count: int,
    weak_count: int,
    adverse_count: int,
    verified_adverse_count: int,
    veto_candidate_count: int,
    critical_terms: list[str],
) -> str:
    if strength == "none":
        if status == "disabled":
            return "외부근거 점검: 외부 뉴스/공시 수집이 꺼져 있어 정성 근거는 판단 보류입니다."
        return f"외부근거 점검: 수집 상태가 `{status}`라서 확인 가능한 외부 근거가 제한적입니다."

    terms = ", ".join(critical_terms[:4]) if critical_terms else "configured critical terms"
    counts = (
        f"총 {item_count}건 중 직접 관련 {direct_count}건, 검증 가능 {verified_count}건, "
        f"위험 후보 {adverse_count}건, 검증된 위험 후보 {verified_adverse_count}건, "
        f"약한/간접 근거 {weak_count}건"
    )
    if strength in {"critical", "strong"}:
        return (
            f"외부근거 위험: {counts}이며, 위험 키워드 후보 {veto_candidate_count}건이 "
            f"감지되었습니다({terms}). 다중 출처·고신뢰 조건 충족 여부를 보수적으로 확인해야 합니다."
        )
    if strength == "moderate":
        return (
            f"외부근거 점검: {counts}입니다. 강한 위험 신호로 확인된 항목은 없으며 "
            "모델 판단을 보완할 참고 근거로 활용합니다."
        )
    return (
        f"외부근거 점검: {counts}입니다. 현재 확인된 항목은 routine/context 성격이거나 "
        "약한 근거이므로 모델 판단을 뒤집는 근거로 쓰지 않습니다."
    )


def _evidence_reliability_text(evidence_profile: _EvidenceProfile) -> str:
    return (
        "출처 신뢰도, 기업 직접 관련성, 최신성, 중복 여부, 위험 키워드의 문맥 확인 여부를 "
        "나눠 검증합니다. "
        f"현재 외부근거 강도는 `{evidence_profile['strength']}`이며, "
        f"직접 관련 {evidence_profile['direct_count']}건, "
        f"검증 가능 {evidence_profile['verified_count']}건으로 요약됩니다."
    )


def _evidence_limitations(
    news_cache: dict[str, Any],
    *,
    evidence_profile: _EvidenceProfile,
) -> list[str]:
    """Explain evidence coverage limits so Stage 2 does not overstate weak signals."""
    limitations: list[str] = []
    status = evidence_profile["status"]
    if status in {"disabled", "not_implemented", "placeholder", "missing_credentials"}:
        limitations.append(
            f"외부근거 수집 상태가 `{status}`라서 뉴스·웹·공시 기반 검증은 제한적입니다."
        )
    elif evidence_profile["item_count"] > 0 and evidence_profile["direct_count"] == 0:
        limitations.append(
            "수집 항목은 있지만 기업명 또는 종목코드 직접 관련성이 확인된 근거가 없습니다."
        )
    elif evidence_profile["weak_count"] > evidence_profile["direct_count"]:
        limitations.append(
            "간접/약한 근거가 직접 관련 근거보다 많아, 위험 신호를 확정 사실로 보지 않습니다."
        )

    date_filter_note = _historical_evidence_filter_note(news_cache)
    if date_filter_note:
        limitations.append(date_filter_note)

    provider_note = _provider_coverage_limitation_note(news_cache.get("providers"))
    if provider_note:
        limitations.append(provider_note)

    if not limitations and evidence_profile["strength"] in {"none", "weak"}:
        limitations.append(
            "현재 외부근거 강도는 낮아 모델 판단을 뒤집기보다 설명 보완용으로만 사용합니다."
        )
    return limitations[:3]


def _historical_evidence_filter_note(news_cache: dict[str, Any]) -> str:
    providers = news_cache.get("providers")
    if not isinstance(providers, dict):
        return ""
    end_dates: set[str] = set()
    filtered_after_cutoff = 0
    filtered_undated = 0
    historical_mode = False
    for provider in providers.values():
        if not isinstance(provider, dict):
            continue
        date_filter = provider.get("as_of_date_filter")
        if isinstance(date_filter, dict):
            historical_mode = historical_mode or bool(date_filter.get("historical_mode", False))
            end_date = str(date_filter.get("end_date") or "")
            if end_date:
                end_dates.add(end_date)
            filtered_after_cutoff += _safe_int(date_filter.get("filtered_after_cutoff_count"))
            filtered_undated += _safe_int(date_filter.get("filtered_undated_count"))
        query_window = provider.get("query_window")
        if isinstance(query_window, dict):
            end_date = str(query_window.get("end_date") or "")
            if end_date:
                end_dates.add(end_date)
    if not historical_mode:
        return ""
    cutoff = sorted(end_dates)[-1] if end_dates else str(news_cache.get("as_of_date") or "")
    filtered_count = filtered_after_cutoff + filtered_undated
    if filtered_count <= 0:
        return f"과거 기준일 {cutoff} 이전 공개 근거만 사용하도록 날짜 필터를 적용했습니다."
    return (
        f"과거 기준일 {cutoff} 이후 또는 날짜 미확인 근거 {filtered_count}건을 제외해 "
        "look-ahead bias를 줄였습니다."
    )


def _provider_coverage_limitation_note(providers: object) -> str:
    if not isinstance(providers, dict) or not providers:
        return ""
    limited: list[str] = []
    for provider_name, raw_provider in providers.items():
        if not isinstance(raw_provider, dict):
            continue
        status = str(raw_provider.get("status") or "")
        if status in {"missing_key", "error", "partial_error", "missing_corp_code"}:
            limited.append(f"{provider_name}:{status}")
    if not limited:
        return ""
    return "일부 수집 경로에 제한이 있습니다(" + ", ".join(limited[:3]) + ")."


def _model_evidence_challenge(
    *,
    bundle: Stage2InputBundle,
    debt_findings: list[str],
    evidence_profile: _EvidenceProfile,
) -> str:
    prediction_label = bundle.prediction_label
    strength = evidence_profile["strength"]
    has_debt_risk = _contains_any(
        debt_findings,
        ("추가 경계", "부족", "취약", "제한적", "어렵습니다", "약합니다", "차환 리스크"),
    )
    has_debt_support = _contains_any(
        debt_findings,
        ("완충 근거", "완화 신호", "방어력", "양호", "확보", "여력"),
    )
    if prediction_label == "투자적격" and strength in {"strong", "critical"}:
        return (
            "정량상 투자적격이지만 직접 관련 외부 위험 근거가 있어 위원회 보수 검토가 필요합니다."
        )
    has_offsetting_support = (
        has_debt_support and strength in {"none", "weak"} and bundle.probability_speculative < 0.10
    )
    if prediction_label == "투자적격" and has_debt_risk and not has_offsetting_support:
        return "정량상 투자적격이지만 유동성·상환여력 신호가 일부 충돌해 추가 점검이 필요합니다."
    if prediction_label == "투자적격" and has_debt_risk and has_offsetting_support:
        return (
            "일부 부채·유동성 경고 신호가 있으나 현금흐름과 상환여력 완화 요인이 더 커 "
            "현재 모델 원판단을 뒤집을 수준은 아닙니다."
        )
    if prediction_label == "부적격" and has_debt_support and strength in {"none", "weak"}:
        return "정량상 부적격 판단은 유지하되, 부채·현금흐름 일부 지표는 완화 근거로 재검토할 수 있습니다."
    return "정량 모델 판단과 외부/유동성 검증 사이의 중대한 충돌은 제한적입니다."


def _evidence_audit_conclusion(
    *,
    bundle: Stage2InputBundle,
    debt_findings: list[str],
    evidence_profile: _EvidenceProfile,
) -> str:
    strength = evidence_profile["strength"]
    if strength == "critical":
        return "외부 근거가 치명 리스크 후보에 가까워 veto 규칙 충족 여부를 최우선으로 확인해야 합니다."
    if strength == "strong":
        return "외부 근거가 강하므로 모델 원판단보다 보수적인 보류 또는 부적격 검토가 필요합니다."
    if _contains_any(debt_findings, ("추가 경계", "차환 리스크", "상환 재원", "1배 미만")):
        if strength in {"none", "weak"} and _contains_any(
            debt_findings,
            ("완충 근거", "완화 신호", "현금 여력", "상환 방어력", "양호", "확보", "여력"),
        ):
            return (
                "부채·유동성 일부 경고는 있으나 현금흐름과 상환여력 완화 요인이 함께 확인되어 "
                "모델 원판단을 뒤집기보다 참고 점검 포인트로 처리합니다."
            )
        return "외부 치명 리스크는 확정되지 않았지만 부채·유동성 측면에서 보류 의견을 강화합니다."
    if bundle.prediction_label == "부적격" and _contains_any(
        debt_findings,
        ("완화 신호", "현금 여력", "상환 방어력", "양호"),
    ):
        return "부적격 원판단은 보존하되, 현금흐름과 상환여력 완화 요인을 함께 표시해야 합니다."
    return "현재 확인된 외부 근거는 모델 원판단을 뒤집기보다 설명과 점검 포인트를 보완합니다."


def _evidence_audit_confidence(
    *,
    status: str,
    debt_confidence: float,
    evidence_profile: _EvidenceProfile,
) -> float:
    if status in {"not_implemented", "disabled", "placeholder"}:
        return round(_clamp(max(0.25, debt_confidence - 0.08), maximum=0.62), 4)
    score = 0.28 + 0.35 * _clamp(debt_confidence) + 0.32 * float(evidence_profile["score"])
    if evidence_profile["direct_count"] > 0:
        score += 0.05
    if evidence_profile["weak_count"] > evidence_profile["direct_count"]:
        score -= 0.04
    return round(_clamp(score, minimum=0.28, maximum=0.88), 4)


def _contains_any(values: list[str], markers: tuple[str, ...]) -> bool:
    text = " ".join(values)
    return any(marker in text for marker in markers)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if not isinstance(value, int | float | str):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_below_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value < threshold


def _metric_at_least_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value >= threshold


def _metric_above_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value > threshold


def _metric_at_most_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value <= threshold


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _has_substantive_external_risk(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    return bool(
        _shared_has_substantive_external_risk(
            news_cache,
            source_feature_row=source_feature_row,
        )
    )


def _is_substantive_external_risk_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    return bool(
        _shared_substantive_external_risk_item(
            item,
            source_feature_row=source_feature_row,
        )
    )
