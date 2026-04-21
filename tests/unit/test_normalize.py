"""Tests for rating normalisation and notch encoding."""

from __future__ import annotations

import pytest

from bfd.ratings.normalize import (
    CANONICAL_SCALE,
    NOTCH_TO_RATING,
    RATING_TO_NOTCH,
    normalize_rating,
    notch_to_rating,
    rating_to_notch,
)


class TestCanonicalScale:
    def test_scale_has_22_notches(self) -> None:
        assert len(CANONICAL_SCALE) == 22

    def test_aaa_is_top(self) -> None:
        assert CANONICAL_SCALE[0] == "AAA"
        assert RATING_TO_NOTCH["AAA"] == 22

    def test_d_is_bottom(self) -> None:
        assert CANONICAL_SCALE[-1] == "D"
        assert RATING_TO_NOTCH["D"] == 1

    def test_mapping_is_bijective(self) -> None:
        for rating, notch in RATING_TO_NOTCH.items():
            assert NOTCH_TO_RATING[notch] == rating


class TestNormalizeRating:
    @pytest.mark.parametrize("agency", ["한국기업평가", "한국신용평가", "NICE신용평가"])
    def test_identity_on_canonical_backbone(self, agency: str) -> None:
        assert normalize_rating("AAA", agency) == "AAA"
        assert normalize_rating("BBB-", agency) == "BBB-"
        assert normalize_rating("BB+", agency) == "BB+"

    def test_nr_variants(self) -> None:
        assert normalize_rating("NR", "한국기업평가") == "NR"
        assert normalize_rating("", "한국기업평가") == "NR"
        assert normalize_rating("-", "한국기업평가") == "NR"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_rating("XYZ", "한국기업평가")


class TestNotchConversions:
    def test_roundtrip(self) -> None:
        for rating in CANONICAL_SCALE:
            assert notch_to_rating(rating_to_notch(rating)) == rating

    def test_nr_raises(self) -> None:
        with pytest.raises(ValueError):
            rating_to_notch("NR")
