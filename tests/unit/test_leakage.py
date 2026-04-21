"""Guard the t → t+1 invariant with explicit tests.

THESE TESTS PROTECT THE PROJECT'S CORE CORRECTNESS. Do not xfail or skip.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bfd.data.splitters import (
    WalkForwardFold,
    map_financials_to_next_year_rating,
    walk_forward_folds,
)
from bfd.utils.time import target_rating_year
from bfd.validation.leakage import (
    LeakageError,
    assert_next_year_mapping,
    assert_no_future_features,
    assert_unique_firm_year,
)


class TestTargetRatingYear:
    def test_basic(self) -> None:
        assert target_rating_year(2022) == 2023
        assert target_rating_year(2015) == 2016


class TestMapFinancialsToNextYearRating:
    def test_happy_path(self) -> None:
        fin = pd.DataFrame(
            {
                "corp_code": ["005930", "005930", "000660"],
                "fiscal_year": [2020, 2021, 2020],
                "total_assets": [1e12, 1.1e12, 5e11],
            }
        )
        ratings = pd.DataFrame(
            {
                "corp_code": ["005930", "005930", "000660"],
                "rating_year": [2021, 2022, 2021],
                "rating_normalized": ["AA+", "AA", "BBB"],
            }
        )

        out = map_financials_to_next_year_rating(fin, ratings)
        assert len(out) == 3
        # Post-condition enforced inside the function
        assert (out["target_rating_year"] == out["fiscal_year"] + 1).all()

    def test_drops_rows_without_next_year_rating(self) -> None:
        fin = pd.DataFrame(
            {
                "corp_code": ["005930", "000660"],
                "fiscal_year": [2022, 2022],
                "total_assets": [1e12, 5e11],
            }
        )
        # Only one firm has a 2023 rating
        ratings = pd.DataFrame(
            {"corp_code": ["005930"], "rating_year": [2023], "rating_normalized": ["AA"]}
        )
        out = map_financials_to_next_year_rating(fin, ratings)
        assert len(out) == 1
        assert out.iloc[0]["corp_code"] == "005930"

    def test_raises_on_same_year_mapping_attempt(self) -> None:
        # Constructed to *fail* if someone ever weakens the assertion.
        fin = pd.DataFrame({"corp_code": ["005930"], "fiscal_year": [2022], "x": [1]})
        # Rating dated in the same fiscal year — must not match
        ratings = pd.DataFrame(
            {"corp_code": ["005930"], "rating_year": [2022], "rating_normalized": ["AA"]}
        )
        out = map_financials_to_next_year_rating(fin, ratings)
        assert out.empty  # should not leak same-year rating


class TestLeakageAssertions:
    def test_assert_next_year_mapping_ok(self) -> None:
        df = pd.DataFrame({"fiscal_year": [2020, 2021], "target_rating_year": [2021, 2022]})
        assert_next_year_mapping(df)  # no raise

    def test_assert_next_year_mapping_violation(self) -> None:
        df = pd.DataFrame({"fiscal_year": [2020, 2021], "target_rating_year": [2020, 2023]})
        with pytest.raises(LeakageError):
            assert_next_year_mapping(df)

    def test_assert_unique_firm_year(self) -> None:
        df = pd.DataFrame(
            {
                "corp_code": ["005930", "005930", "000660"],
                "fiscal_year": [2020, 2020, 2020],
            }
        )
        with pytest.raises(LeakageError):
            assert_unique_firm_year(df)

    def test_assert_no_future_features(self) -> None:
        df = pd.DataFrame(
            {
                "fiscal_year": [2020],
                "target_rating_year": [2021],
                "some_event_year": [2022],  # future
            }
        )
        with pytest.raises(LeakageError):
            assert_no_future_features(df)


class TestWalkForwardFolds:
    def test_basic(self) -> None:
        years = list(range(2015, 2021))
        folds = list(walk_forward_folds(years, train_window=3, val_window=1, step=1))
        assert folds == [
            WalkForwardFold(
                fold_index=0, train_years=(2015, 2016, 2017), val_years=(2018,)
            ),
            WalkForwardFold(
                fold_index=1, train_years=(2016, 2017, 2018), val_years=(2019,)
            ),
            WalkForwardFold(
                fold_index=2, train_years=(2017, 2018, 2019), val_years=(2020,)
            ),
        ]

    def test_min_train_year_filters(self) -> None:
        years = list(range(2010, 2021))
        folds = list(
            walk_forward_folds(
                years, train_window=3, val_window=1, step=1, min_train_year=2015
            )
        )
        # Should start with train=[2015..2017], val=2018
        assert folds[0].train_years == (2015, 2016, 2017)
        assert folds[0].val_years == (2018,)
