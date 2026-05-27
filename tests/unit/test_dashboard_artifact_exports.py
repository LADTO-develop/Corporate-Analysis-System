from __future__ import annotations

import pandas as pd

from cas.artifacts.dashboard_exports import (
    build_company_latest,
    build_feature_dictionary,
    build_industry_latest_summary,
    build_peer_percentiles,
    risk_band,
)


def test_risk_band_uses_dashboard_thresholds() -> None:
    assert risk_band(0.34) == "안정"
    assert risk_band(0.35) == "관찰"
    assert risk_band(0.65) == "고위험"


def test_build_company_latest_picks_latest_company_year() -> None:
    master = pd.DataFrame(
        [
            {
                "market": "KOSPI",
                "stock_code": "000001",
                "corp_name": "테스트",
                "fiscal_year": 2023,
                "eval_year": 2024,
                "listed_year": 2000,
                "firm_size_group": "large",
                "industry_macro_category": "IT",
                "current_ratio": 1.2,
            },
            {
                "market": "KOSPI",
                "stock_code": "000001",
                "corp_name": "테스트",
                "fiscal_year": 2024,
                "eval_year": 2025,
                "listed_year": 2000,
                "firm_size_group": "large",
                "industry_macro_category": "IT",
                "current_ratio": 1.5,
            },
        ]
    )

    latest = build_company_latest(master, ["current_ratio"])

    assert len(latest) == 1
    assert latest.iloc[0]["fiscal_year"] == 2024
    assert latest.iloc[0]["current_ratio"] == 1.5


def test_build_peer_percentiles_uses_market_and_industry_groups() -> None:
    master = pd.DataFrame(
        [
            {
                "stock_code": "000001",
                "corp_name": "A",
                "fiscal_year": 2024,
                "eval_year": 2025,
                "market": "KOSPI",
                "industry_macro_category": "IT",
                "current_ratio": 1.0,
            },
            {
                "stock_code": "000002",
                "corp_name": "B",
                "fiscal_year": 2024,
                "eval_year": 2025,
                "market": "KOSPI",
                "industry_macro_category": "IT",
                "current_ratio": 2.0,
            },
        ]
    )

    percentiles = build_peer_percentiles(master, ["current_ratio"])

    assert list(percentiles["overall_percentile"]) == [50.0, 100.0]
    assert list(percentiles["industry_median"]) == [1.5, 1.5]


def test_build_feature_dictionary_joins_metadata_and_feature_groups() -> None:
    metadata_columns = [
        {
            "variable_name": "current_ratio",
            "korean_name": "유동비율",
            "description": "유동성 지표",
            "unit": "ratio",
        }
    ]
    feature_json = {
        "selected_source_features": ["current_ratio"],
        "feature_metadata": [{"source_feature": "current_ratio", "feature_group": "liquidity"}],
    }

    dictionary = build_feature_dictionary(metadata_columns, feature_json)

    assert dictionary.iloc[0]["feature"] == "current_ratio"
    assert dictionary.iloc[0]["feature_group"] == "liquidity"
    assert dictionary.iloc[0]["korean_name"] == "유동비율"


def test_build_industry_latest_summary_uses_latest_company_rows() -> None:
    prediction_scores = pd.DataFrame(
        [
            {
                "market": "KOSPI",
                "stock_code": "000001",
                "corp_name": "A",
                "industry_macro_category": "IT",
                "fiscal_year": 2023,
                "eval_year": 2024,
                "is_speculative": 0,
                "prob_speculative": 0.2,
                "pred_label_0_5": 0,
                "pred_label_tuned": 0,
            },
            {
                "market": "KOSPI",
                "stock_code": "000001",
                "corp_name": "A",
                "industry_macro_category": "IT",
                "fiscal_year": 2024,
                "eval_year": 2025,
                "is_speculative": 1,
                "prob_speculative": 0.7,
                "pred_label_0_5": 1,
                "pred_label_tuned": 1,
            },
        ]
    )

    summary = build_industry_latest_summary(prediction_scores)

    assert summary.iloc[0]["companies"] == 1
    assert summary.iloc[0]["positive_companies"] == 1
    assert summary.iloc[0]["mean_prob_speculative"] == 0.7
