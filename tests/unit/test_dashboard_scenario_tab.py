"""Tests for dashboard scenario tab calculations."""

from __future__ import annotations

import pandas as pd

from cas.dashboard.scenario_tab import (
    ScenarioTabFormatters,
    approximate_percentile,
    build_scenario_frame,
)


def _formatters() -> ScenarioTabFormatters:
    def display_name(feature: str, feature_map: pd.DataFrame) -> str:
        matched = feature_map.loc[feature_map["feature"] == feature, "korean_name"]
        return feature if matched.empty else str(matched.iloc[0])

    def feature_unit(feature: str, feature_map: pd.DataFrame) -> str:
        matched = feature_map.loc[feature_map["feature"] == feature, "unit"]
        return "" if matched.empty else str(matched.iloc[0])

    return ScenarioTabFormatters(
        display_name=display_name,
        feature_unit=feature_unit,
        value_with_unit=lambda value, unit, _feature: "-" if value is None else f"{value}{unit}",
        delta_with_unit=lambda value, unit, _feature: f"{float(value):+.2f}{unit}",
        percentile_label=lambda value: "-" if pd.isna(value) else f"{float(value):.1f}백분위",
        scalar=lambda value: str(value),
        feature_direction_label=lambda _feature: "높을수록 긍정",
        unit_description=lambda unit: f"단위:{unit}",
    )


def test_approximate_percentile_adds_scenario_value_to_distribution() -> None:
    percentile = approximate_percentile(pd.Series([1.0, 2.0, 3.0]), 4.0)

    assert percentile == 100.0


def test_build_scenario_frame_formats_values_and_percentile() -> None:
    selected_row = pd.Series({"cash_ratio": 0.3, "debt_ratio": 1.2})
    company_universe = pd.DataFrame(
        {
            "cash_ratio": [0.1, 0.2, 0.3, 0.4],
            "debt_ratio": [0.8, 1.0, 1.2, 1.4],
        }
    )
    feature_map = pd.DataFrame(
        [
            {"feature": "cash_ratio", "korean_name": "현금비율", "unit": "ratio"},
            {"feature": "debt_ratio", "korean_name": "부채비율", "unit": "ratio"},
        ]
    )

    frame = build_scenario_frame(
        selected_row=selected_row,
        company_universe=company_universe,
        feature_map=feature_map,
        deltas={"cash_ratio": 0.1, "debt_ratio": -0.2},
        formatters=_formatters(),
        scenario_features=("cash_ratio", "debt_ratio"),
    )

    assert frame.loc[0, "변수"] == "현금비율"
    assert frame.loc[0, "시나리오 조정값"] == 0.4
    assert frame.loc[0, "현재값_표시"] == "0.3ratio"
    assert frame.loc[0, "시나리오 적용 후 위치"] == "90.0백분위"
    assert frame.loc[1, "시나리오 조정값"] == 1.0
    assert frame.loc[1, "일반 해석 방향"] == "높을수록 긍정"


def test_build_scenario_frame_handles_missing_baseline() -> None:
    frame = build_scenario_frame(
        selected_row=pd.Series({"cash_ratio": None}),
        company_universe=pd.DataFrame({"cash_ratio": [0.1, 0.2]}),
        feature_map=pd.DataFrame(
            [{"feature": "cash_ratio", "korean_name": "현금비율", "unit": "ratio"}]
        ),
        deltas={"cash_ratio": 0.1},
        formatters=_formatters(),
        scenario_features=("cash_ratio",),
    )

    assert frame.loc[0, "시나리오 조정값"] is None
    assert frame.loc[0, "시나리오 적용 후 위치"] == "-"
