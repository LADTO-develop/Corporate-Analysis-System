"""Tests for dashboard value formatting helpers."""

from __future__ import annotations

from cas.dashboard.formatting import COVERAGE_CAP_LABEL, format_ratio_value


def test_coverage_ratios_are_displayed_as_multiples() -> None:
    assert format_ratio_value(15.29068, "interest_coverage_ratio") == "15.29배"
    assert format_ratio_value(35.59144, "cashflow_coverage_ratio") == "35.59배"
    assert format_ratio_value(2.5, "interest_coverage_ratio", signed=True) == "+2.50배"


def test_ordinary_ratios_still_display_as_percentages() -> None:
    assert format_ratio_value(0.057773, "net_margin") == "5.78%"
    assert format_ratio_value(0.025, "net_margin", signed=True) == "+2.50%p"


def test_capped_coverage_ratios_are_not_rendered_as_huge_percentages() -> None:
    assert format_ratio_value(1_000_000, "interest_coverage_ratio") == COVERAGE_CAP_LABEL
