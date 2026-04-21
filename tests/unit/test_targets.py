"""Tests for the investment-grade / speculative binarisation."""

from __future__ import annotations

import pandas as pd
import pytest

from bfd.ratings.targets import (
    INVESTMENT_GRADE_MIN_NOTCH,
    add_binary_target,
    to_binary_target,
)


class TestBinaryTarget:
    def test_investment_grade_boundary(self) -> None:
        # BBB- is the lowest investment grade
        assert to_binary_target("BBB-") == 0
        # BB+ is the highest speculative grade
        assert to_binary_target("BB+") == 1

    @pytest.mark.parametrize("rating", ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"])
    def test_investment_grades(self, rating: str) -> None:
        assert to_binary_target(rating) == 0

    @pytest.mark.parametrize("rating", ["BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"])
    def test_speculative_grades(self, rating: str) -> None:
        assert to_binary_target(rating) == 1

    def test_min_notch_constant(self) -> None:
        # Sanity check so nobody accidentally edits the boundary
        assert INVESTMENT_GRADE_MIN_NOTCH == 13


class TestAddBinaryTarget:
    def test_adds_column_and_drops_nr(self) -> None:
        df = pd.DataFrame(
            {
                "rating_normalized": ["AA", "BBB-", "BB+", "NR", "D"],
            }
        )
        out = add_binary_target(df)
        assert "target" in out.columns
        assert len(out) == 4  # NR dropped
        assert list(out["target"]) == [0, 0, 1, 1]

    def test_keeps_nr_when_drop_false(self) -> None:
        df = pd.DataFrame({"rating_normalized": ["AA", "NR"]})
        with pytest.raises(ValueError):
            # to_binary_target raises on NR
            add_binary_target(df, drop_nr=False)
