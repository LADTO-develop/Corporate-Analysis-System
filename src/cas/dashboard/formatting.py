"""Dependency-light dashboard formatting helpers."""

from __future__ import annotations

COVERAGE_MULTIPLE_FEATURES = {
    "cashflow_coverage_ratio",
    "interest_coverage_ratio",
}
VALUATION_MULTIPLE_FEATURES = {"market_to_book"}
TURNOVER_MULTIPLE_FEATURES = {"total_debt_turnover"}
MULTIPLE_RATIO_FEATURES = (
    COVERAGE_MULTIPLE_FEATURES | VALUATION_MULTIPLE_FEATURES | TURNOVER_MULTIPLE_FEATURES
)
COVERAGE_CAP_VALUE = 1_000_000.0
COVERAGE_CAP_LABEL = "\uc0c1\ud55c\uac12(\uc774\uc790\ube44\uc6a9 0/\uadf9\uc18c)"


def format_ratio_value(number: float, feature: str | None = None, *, signed: bool = False) -> str:
    """Format ratio-like features as percentages or financial multiples."""
    if feature in MULTIPLE_RATIO_FEATURES:
        return format_ratio_multiple(number, feature, signed=signed)

    sign = "+" if signed and number > 0 else ""
    suffix = "%p" if signed else "%"
    return f"{sign}{number * 100:.2f}{suffix}"


def format_ratio_multiple(number: float, feature: str | None, *, signed: bool = False) -> str:
    """Format ratio features that are financial multiples instead of percentages."""
    if feature in COVERAGE_MULTIPLE_FEATURES and abs(number) >= COVERAGE_CAP_VALUE:
        return COVERAGE_CAP_LABEL

    sign = "+" if signed and number > 0 else ""
    suffix = "\ud68c" if feature in TURNOVER_MULTIPLE_FEATURES else "\ubc30"
    return f"{sign}{number:,.2f}{suffix}"
