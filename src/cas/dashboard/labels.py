"""Small label conversion helpers for the credit dashboard."""

from __future__ import annotations

MARKET_LABELS = {
    "KOSPI": "코스피",
    "KOSDAQ": "코스닥",
}

SIZE_LABELS = {
    "large": "대기업",
    "mid_sized": "중견기업",
    "small_and_medium": "중소기업",
    "other": "기타",
}

INDUSTRY_LABELS = {
    "construction": "건설업",
    "it_services": "IT·서비스업",
    "manufacturing": "제조업",
    "other": "기타",
    "transport_storage": "운수·창고업",
    "wholesale_retail": "도소매업",
}

PREDICTION_LABELS = {
    0: "투자적격",
    1: "투기등급",
}

STAGE2_AGENT_ROLE_LABELS = {
    "quant_credit": "QuantCreditAgent",
    "evidence_audit": "EvidenceAuditAgent",
    "chair_report": "ChairReportAgent",
}

STAGE2_RISK_BAND_LABELS = {
    "stable": "안정",
    "watch": "관찰",
    "high_risk": "고위험",
    "insufficient_data": "데이터 부족",
}


def to_market_label(value: object) -> str:
    """Convert a market code into a Korean label."""
    return MARKET_LABELS.get(str(value), str(value))


def to_market_display_label(value: object) -> str:
    """Convert a market code into a readable label for the market selector."""
    labels = {"KOSPI": "코스피", "KOSDAQ": "코스닥"}
    return labels.get(str(value), to_market_label(value))


def to_size_label(value: object) -> str:
    """Convert a firm size code into a Korean label."""
    return SIZE_LABELS.get(str(value), str(value))


def to_industry_label(value: object) -> str:
    """Convert an industry code into a Korean label."""
    return INDUSTRY_LABELS.get(str(value), str(value))


def to_industry_display_label(value: object) -> str:
    """Convert an industry code into a readable label for the market selector."""
    labels = {
        "construction": "건설",
        "it_services": "IT/서비스",
        "manufacturing": "제조",
        "other": "기타",
        "transport_storage": "운수/창고",
        "wholesale_retail": "도소매",
    }
    return labels.get(str(value), to_industry_label(value))


def to_prediction_label(value: object) -> str:
    """Convert a numeric prediction label into a Korean label."""
    try:
        return PREDICTION_LABELS.get(int(float(str(value))), str(value))
    except (TypeError, ValueError):
        return str(value)


def to_stage2_model_label(value: object) -> str:
    """Convert dashboard prediction labels into the Stage 2 model_view label space."""
    label = to_prediction_label(value)
    if label in {"투기등급", "부적격"}:
        return "부적격"
    if label in {"투자적격", "적격"}:
        return "투자적격"
    return label


def to_committee_base_label(model_label: object) -> str:
    """Map a binary Stage 1 model label onto the committee label space."""
    label = str(model_label)
    if label == "투자적격":
        return "적격"
    if label in {"투기등급", "부적격"}:
        return "부적격"
    return "보류"


def to_stage2_risk_band(value: object) -> str:
    """Normalize dashboard risk bands into the Stage 2 risk band vocabulary."""
    band = str(value or "insufficient_data").strip()
    if band in {"stable", "watch", "high_risk", "insufficient_data"}:
        return band
    if band == "데이터 부족":
        return "insufficient_data"
    if band in {"고위험", "위험"}:
        return "high_risk"
    if band in {"관찰", "주의"}:
        return "watch"
    if band in {"안정", "낮음"}:
        return "stable"
    return "insufficient_data"


def format_stage2_risk_band(value: object) -> str:
    """Format Stage 2 risk bands for the dashboard."""
    return STAGE2_RISK_BAND_LABELS.get(to_stage2_risk_band(value), "데이터 부족")
