"""Shared helper utilities for Stage 2 committee rules."""

from __future__ import annotations

from typing import Any


def safe_float(value: object) -> float | None:
    """Return a float when the value is numeric-like and finite."""
    try:
        if value is None or not isinstance(value, int | float | str):
            return None
        numeric = float(value)
        if numeric != numeric:
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def safe_int(value: object) -> int | None:
    """Return an int converted through safe_float."""
    numeric = safe_float(value)
    if numeric is None:
        return None
    return int(numeric)


def metric_at_least(row: dict[str, Any], key: str, threshold: float) -> bool:
    """Return whether a numeric row metric is at least the threshold."""
    value = safe_float(row.get(key))
    return value is not None and value >= threshold


def metric_at_most(row: dict[str, Any], key: str, threshold: float) -> bool:
    """Return whether a numeric row metric is at most the threshold."""
    value = safe_float(row.get(key))
    return value is not None and value <= threshold


def metric_above(row: dict[str, Any], key: str, threshold: float) -> bool:
    """Return whether a numeric row metric is strictly above the threshold."""
    value = safe_float(row.get(key))
    return value is not None and value > threshold


def metric_below(row: dict[str, Any], key: str, threshold: float) -> bool:
    """Return whether a numeric row metric is strictly below the threshold."""
    value = safe_float(row.get(key))
    return value is not None and value < threshold


def flag_is_true(value: object) -> bool:
    """Interpret bool-like values used by model feature rows."""
    if isinstance(value, bool):
        return value
    numeric = safe_float(value)
    if numeric is not None:
        return numeric >= 0.5
    return str(value).strip().lower() in {"true", "yes", "y", "on"}


def flag_is_false(value: object) -> bool:
    """Interpret bool-like false values used by model feature rows."""
    if isinstance(value, bool):
        return not value
    numeric = safe_float(value)
    if numeric is not None:
        return numeric < 0.5
    return str(value).strip().lower() in {"false", "no", "n", "off"}


def clean_text_items(items: list[str]) -> list[str]:
    """Clean a list of Korean committee prose items."""
    return [clean_korean_review_text(item) for item in items]


def clean_evidence_summary_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """Clean committee evidence summaries while preserving other fields."""
    return [
        {
            **item,
            "summary": clean_korean_review_text(str(item.get("summary", ""))),
        }
        for item in items
    ]


def clean_korean_review_text(text: str) -> str:
    """Clean committee prose for Korean report output."""
    cleaned = str(text).strip()
    replacements = {
        "적격로": "적격으로",
        "부적격로": "부적격으로",
        "투자적격 등급을 확정합니다": "투자적격 검토 의견을 제시합니다",
        "부적격 등급을 확정합니다": "부적격 검토 의견을 제시합니다",
        "신용등급을 확정합니다": "신용위험 검토 의견을 제시합니다",
        "등급을 확정합니다": "검토 의견을 제시합니다",
        "최종 승인합니다": "검토 의견으로 정리합니다",
        "최종 승인": "검토 의견",
        "확정합니다": "검토 의견을 제시합니다",
        "승인합니다": "의견을 제시합니다",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    return cleaned
