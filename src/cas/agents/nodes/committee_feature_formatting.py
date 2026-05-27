"""Feature driver formatting helpers for Stage 2 committee agents."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from cas.utils.io import read_json

_FEATURE_METADATA_PATH = Path("data/input/credit_46_features/feature_46_dictionary_metadata.json")

_INDUSTRY_LABELS = {
    "manufacturing": "제조업",
    "construction": "건설업",
    "retail_wholesale": "도소매업",
    "it_services": "IT·서비스업",
    "transport_storage": "운수·창고업",
    "other": "기타",
}

_SIZE_LABELS = {
    "large": "대기업",
    "mid_sized": "중견기업",
    "small_medium": "중소기업",
    "other": "기타",
}

_POLARITY: dict[str, Literal["higher_better", "lower_better", "contextual", "flag_positive"]] = {
    "current_ratio": "higher_better",
    "cash_ratio": "higher_better",
    "equity_ratio": "higher_better",
    "debt_ratio": "lower_better",
    "total_borrowings_ratio": "lower_better",
    "capital_impairment_ratio": "lower_better",
    "net_margin": "higher_better",
    "gross_profit": "higher_better",
    "interest_coverage_ratio": "higher_better",
    "pretax_roa": "higher_better",
    "operating_roa": "higher_better",
    "pretax_roe": "higher_better",
    "ocf_to_total_liabilities": "higher_better",
    "ocf_to_total_borrowings": "higher_better",
    "ocf_to_sales": "higher_better",
    "cashflow_coverage_ratio": "higher_better",
    "accruals_ratio": "lower_better",
    "intangible_assets_ratio": "lower_better",
    "total_debt_turnover": "higher_better",
    "dividend_payer": "flag_positive",
    "market_to_book": "contextual",
    "spec_spread": "lower_better",
    "short_term_borrowings_share": "lower_better",
    "total_assets_growth": "contextual",
    "net_margin_diff": "higher_better",
    "is_2y_consecutive_ocf_deficit": "lower_better",
    "icr_under_1": "lower_better",
    "is_2y_consecutive_operating_loss": "lower_better",
}


def describe_top_drivers(
    xgb: dict[str, Any],
    source_row: dict[str, Any],
    peer_by_feature: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn top SHAP drivers into user-facing finding details."""
    details: list[dict[str, Any]] = []
    for name, shap_value in _driver_pairs(xgb):
        metadata = _feature_metadata().get(name, {})
        feature_name = str(metadata.get("korean_name") or _prettify_feature_name(name))
        unit = str(metadata.get("unit") or "")
        raw_value = source_row.get(name)
        direction = _driver_direction(name, shap_value)
        details.append(
            {
                "name": name,
                "feature": feature_name,
                "shap_value": shap_value,
                "direction": direction,
                "detail": _feature_point_text(
                    feature_name=feature_name,
                    feature_key=name,
                    raw_value=raw_value,
                    unit=unit,
                    shap_value=shap_value,
                    peer_row=peer_by_feature.get(name),
                ),
            }
        )
    return details


def humanize_category(
    value: object,
    *,
    mapping: dict[str, str] | None = None,
    fallback: str = "unknown",
) -> str:
    """Return a display label for categorical feature values."""
    if value is None:
        return fallback
    raw = str(value)
    if not raw:
        return fallback
    if mapping and raw in mapping:
        return mapping[raw]
    return raw


def humanize_industry(value: object, *, fallback: str = "업종 정보 미확인") -> str:
    """Return a Korean display label for the model industry group."""
    return humanize_category(value, mapping=_INDUSTRY_LABELS, fallback=fallback)


def humanize_size_group(value: object, *, fallback: str = "규모 정보 미확인") -> str:
    """Return a Korean display label for the model firm-size group."""
    return humanize_category(value, mapping=_SIZE_LABELS, fallback=fallback)


def _feature_point_text(
    *,
    feature_name: str,
    feature_key: str,
    raw_value: object,
    unit: str,
    shap_value: float,
    peer_row: dict[str, Any] | None,
) -> str:
    direction = _driver_direction(feature_key, shap_value)
    value_text = _format_driver_value(feature_key=feature_key, raw_value=raw_value, unit=unit)
    comparison_text = _peer_comparison_text(peer_row=peer_row, unit=unit)
    if direction == "risk":
        return (
            f"{feature_name}({value_text})이(가) 현재 모델에서 위험을 높이는 방향으로 작용했습니다."
            f"{comparison_text}"
        )
    return (
        f"{feature_name}({value_text})이(가) 현재 모델에서 위험을 낮추는 방향으로 작용했습니다."
        f"{comparison_text}"
    )


def _format_driver_value(*, feature_key: str, raw_value: object, unit: str) -> str:
    if unit == "category":
        if feature_key == "firm_size_group":
            return humanize_size_group(raw_value)
        if feature_key == "industry_macro_category":
            return humanize_industry(raw_value)
        return humanize_category(raw_value, fallback="범주 정보 미확인")
    return _format_feature_value(raw_value, unit)


def _driver_direction(feature_key: str, shap_value: float) -> Literal["risk", "support"]:
    polarity = _POLARITY.get(feature_key, "contextual")
    if polarity in {"higher_better", "flag_positive"}:
        return "risk" if shap_value > 0 else "support"
    if polarity == "lower_better":
        return "risk" if shap_value > 0 else "support"
    return "risk" if shap_value > 0 else "support"


def _driver_pairs(xgb: dict[str, Any]) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for item in xgb.get("top_drivers", []) or []:
        if isinstance(item, dict):
            name = str(item.get("name", item.get("feature", "")))
            value = float(item.get("value", item.get("score", 0.0)) or 0.0)
        else:
            name = str(item[0])
            value = float(item[1])
        if name:
            pairs.append((name, value))
    return pairs


@lru_cache(maxsize=1)
def _feature_metadata() -> dict[str, dict[str, Any]]:
    metadata = read_json(_FEATURE_METADATA_PATH)
    columns = metadata.get("columns", [])
    return {
        str(column.get("variable_name")): dict(column)
        for column in columns
        if isinstance(column, dict) and column.get("variable_name")
    }


def _prettify_feature_name(name: str) -> str:
    return name.replace("_", " ")


def _format_feature_value(value: object, unit: str) -> str:
    if value is None:
        return "값 없음"
    if unit == "0/1":
        numeric = _safe_float(value)
        if numeric is None:
            return "값 없음"
        return "예" if numeric >= 1.0 else "아니오"
    numeric = _safe_float(value)
    return _format_number(numeric, unit)


def _format_number(value: float | None, unit: str) -> str:
    if value is None:
        return "값 없음"
    if unit == "KRW thousand":
        return f"{value:,.0f}"
    if unit == "%p":
        return f"{value:.2f}%p"
    if unit == "ratio":
        return f"{value:.3f}"
    return f"{value:.3f}"


def _peer_comparison_text(*, peer_row: dict[str, Any] | None, unit: str) -> str:
    if not peer_row:
        return ""

    industry_median = _safe_float(peer_row.get("industry_median"))
    market_median = _safe_float(peer_row.get("market_median"))
    industry_percentile = _safe_float(peer_row.get("industry_percentile"))

    parts: list[str] = []
    if industry_median is not None:
        parts.append(f"산업 중앙값은 {_format_number(industry_median, unit)}입니다")
    if market_median is not None:
        parts.append(f"시장 중앙값은 {_format_number(market_median, unit)}입니다")
    if industry_percentile is not None:
        parts.append(f"산업 내 위치는 {industry_percentile:.1f}백분위입니다")

    if not parts:
        return ""
    return " " + " ".join(parts) + "."


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if not isinstance(value, int | float | str):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
