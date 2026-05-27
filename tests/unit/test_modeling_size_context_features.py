from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cas.modeling.size_context_features import (
    add_binary_group_context_features,
    add_group_percentile_features,
    add_group_zscore_features,
    add_signed_log_features,
    signed_log1p,
)


def test_signed_log1p_preserves_sign_and_missing_values() -> None:
    values = pd.Series([-9.0, 0.0, 99.0, np.nan])

    transformed = signed_log1p(values)

    assert transformed.iloc[0] == pytest.approx(-np.log1p(9.0))
    assert transformed.iloc[1] == 0.0
    assert transformed.iloc[2] == pytest.approx(np.log1p(99.0))
    assert pd.isna(transformed.iloc[3])


def test_add_signed_log_features_adds_only_available_columns() -> None:
    frame = pd.DataFrame({"assets_total": [0.0, 99.0]})

    output, added = add_signed_log_features(frame, ["assets_total", "gross_profit"])

    assert added == ["log_assets_total"]
    assert output["log_assets_total"].tolist() == pytest.approx([0.0, np.log1p(99.0)])


def test_add_group_percentile_features_uses_group_ranks() -> None:
    frame = pd.DataFrame(
        {
            "industry_macro_category": ["manufacturing", "manufacturing", "it_services"],
            "fiscal_year": [2023, 2023, 2023],
            "assets_total": [10.0, 30.0, 20.0],
        }
    )

    output, added = add_group_percentile_features(
        frame,
        group_columns=["fiscal_year", "industry_macro_category"],
        value_columns=["assets_total"],
        suffix="industry_year",
    )

    assert added == ["assets_total_industry_year_pct"]
    assert output["assets_total_industry_year_pct"].tolist() == pytest.approx([0.5, 1.0, 1.0])


def test_add_group_zscore_features_uses_zero_for_single_row_group() -> None:
    frame = pd.DataFrame(
        {
            "market": ["KOSPI", "KOSPI", "KOSDAQ"],
            "firm_size_group": ["large", "large", "small"],
            "assets_total": [10.0, 30.0, 20.0],
        }
    )

    output, added = add_group_zscore_features(
        frame,
        group_columns=["market", "firm_size_group"],
        value_columns=["assets_total"],
        suffix="market_size",
    )

    assert added == ["assets_total_market_size_zscore"]
    assert output["assets_total_market_size_zscore"].tolist() == pytest.approx(
        [-0.70710678, 0.70710678, 0.0]
    )


def test_add_binary_group_context_features_adds_rate_and_deviation() -> None:
    frame = pd.DataFrame(
        {
            "market": ["KOSPI", "KOSPI", "KOSPI"],
            "firm_size_group": ["large", "large", "large"],
            "dividend_payer": [1, 0, 1],
        }
    )

    output, added = add_binary_group_context_features(
        frame,
        group_columns=["market", "firm_size_group"],
        value_columns=["dividend_payer"],
        suffix="market_size",
    )

    assert added == ["dividend_payer_market_size_rate", "dividend_payer_market_size_deviation"]
    assert output["dividend_payer_market_size_rate"].tolist() == pytest.approx([2 / 3] * 3)
    assert output["dividend_payer_market_size_deviation"].tolist() == pytest.approx(
        [1 / 3, -2 / 3, 1 / 3]
    )
