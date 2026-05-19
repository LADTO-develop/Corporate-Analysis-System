from __future__ import annotations

import pandas as pd
from scripts.build_feature_44_inference_2026 import (
    add_derived_features as add_inference_derived_features,
)
from scripts.rebuild_feature_44_dataset import add_derived_features


def test_add_derived_features_builds_industry_current_ratio_percentile() -> None:
    frame = pd.DataFrame(
        {
            "fiscal_year": [2021, 2021, 2021, 2022],
            "industry_macro_category": [
                "manufacturing",
                "manufacturing",
                "it_services",
                "manufacturing",
            ],
            "current_ratio": [1.0, 3.0, 2.0, 4.0],
        }
    )

    derived = add_derived_features(frame)

    assert derived["industry_current_ratio_percentile"].tolist() == [0.5, 1.0, 1.0, 1.0]


def test_inference_and_training_derived_feature_logic_match() -> None:
    frame = pd.DataFrame(
        {
            "fiscal_year": [2025, 2025, 2025],
            "industry_macro_category": ["manufacturing", "manufacturing", "manufacturing"],
            "current_ratio": [0.5, 1.5, 2.5],
        }
    )

    training = add_derived_features(frame)
    inference = add_inference_derived_features(frame)

    pd.testing.assert_series_equal(
        training["industry_current_ratio_percentile"],
        inference["industry_current_ratio_percentile"],
        check_names=False,
    )
