"""Tests for dashboard value formatting helpers."""

from __future__ import annotations

import math

import pandas as pd

from cas.dashboard.chart_data import finite_chart_frame, finite_float_or_none
from cas.dashboard.committee_copy import committee_decision_type_info
from cas.dashboard.formatting import COVERAGE_CAP_LABEL, format_ratio_value
from cas.dashboard.labels import (
    format_stage2_risk_band,
    to_committee_base_label,
    to_industry_label,
    to_market_display_label,
    to_stage2_risk_band,
)


def test_coverage_ratios_are_displayed_as_multiples() -> None:
    assert format_ratio_value(15.29068, "interest_coverage_ratio") == "15.29배"
    assert format_ratio_value(35.59144, "cashflow_coverage_ratio") == "35.59배"
    assert format_ratio_value(2.5, "interest_coverage_ratio", signed=True) == "+2.50배"


def test_ordinary_ratios_still_display_as_percentages() -> None:
    assert format_ratio_value(0.057773, "net_margin") == "5.78%"
    assert format_ratio_value(0.025, "net_margin", signed=True) == "+2.50%p"


def test_capped_coverage_ratios_are_not_rendered_as_huge_percentages() -> None:
    assert format_ratio_value(1_000_000, "interest_coverage_ratio") == COVERAGE_CAP_LABEL


def test_negative_capital_impairment_keeps_its_signed_percentage() -> None:
    assert format_ratio_value(-42.108489, "capital_impairment_ratio") == "-4210.85%"
    assert format_ratio_value(0.452343, "capital_impairment_ratio") == "45.23%"
    assert format_ratio_value(-0.2, "capital_impairment_ratio", signed=True) == "-20.00%p"


def test_committee_decision_copy_is_user_friendly() -> None:
    boundary = committee_decision_type_info("경계등급 보류", risk_signal=False)
    mitigation = committee_decision_type_info("과민경고 완화 보류", risk_signal=False)

    assert "딱 잘라 말하기 어려운" in boundary["body"]
    assert "BBB-/BB+" in boundary["detail"]
    assert "바로 부적격으로 단정하긴 이릅니다" in mitigation["body"]
    assert "SHAP" in mitigation["action"]


def test_dashboard_label_helpers_match_user_facing_copy() -> None:
    assert to_market_display_label("KOSPI") == "코스피"
    assert to_industry_label("it_services") == "IT·서비스업"
    assert to_committee_base_label("투자적격") == "적격"
    assert to_committee_base_label("unknown") == "보류"
    assert to_stage2_risk_band("고위험") == "high_risk"
    assert format_stage2_risk_band("watch") == "관찰"


def test_chart_numeric_helpers_drop_non_finite_values() -> None:
    assert finite_float_or_none("12.5") == 12.5
    assert finite_float_or_none(math.inf) is None

    frame = finite_chart_frame(
        [
            {"label": "valid", "값": "1.25"},
            {"label": "missing", "값": pd.NA},
            {"label": "infinite", "값": math.inf},
        ],
        ["값"],
    )

    assert frame["label"].tolist() == ["valid"]
    assert frame["값"].tolist() == [1.25]
