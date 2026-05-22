"""Tests for non-leaky prior rating reference lookup."""

from __future__ import annotations

import pandas as pd

from cas.ratings.prior_reference import lookup_prior_rating_reference


def test_lookup_prior_rating_reference_returns_exact_row(tmp_path) -> None:
    reference_path = tmp_path / "prior_rating_reference.csv"
    pd.DataFrame(
        [
            {
                "universe": "model_v1",
                "market": "KOSDAQ",
                "stock_code": "000250",
                "corp_name": "삼천당제약(주)",
                "fiscal_year": 2023,
                "eval_year": 2024,
                "as_of_date": "2023-12-31",
                "has_prior_rating": True,
                "prior_credit_rating": "BBB-",
                "prior_credit_rating_rank": 10,
                "prior_is_speculative": 0,
                "prior_rating_boundary_group": "exact_bbb_minus_bb_plus_boundary",
                "prior_rating_date": "2022-12-31",
                "prior_rating_age_days": 365,
                "prior_rating_agency": "NICE평가정보주식회사",
            }
        ]
    ).to_csv(reference_path, index=False, encoding="utf-8-sig")

    payload = lookup_prior_rating_reference(
        stock_code="250",
        fiscal_year=2023,
        eval_year=2024,
        universe="model_v1",
        path=reference_path,
    )

    assert payload["has_prior_rating"] is True
    assert payload["prior_credit_rating"] == "BBB-"
    assert payload["prior_rating_boundary_group"] == "exact_bbb_minus_bb_plus_boundary"


def test_lookup_prior_rating_reference_returns_empty_for_unmatched_year(tmp_path) -> None:
    reference_path = tmp_path / "prior_rating_reference.csv"
    pd.DataFrame(
        [
            {
                "universe": "model_v1",
                "stock_code": "000250",
                "fiscal_year": 2022,
                "eval_year": 2023,
                "as_of_date": "2022-12-31",
                "has_prior_rating": True,
                "prior_credit_rating": "A-",
            }
        ]
    ).to_csv(reference_path, index=False, encoding="utf-8-sig")

    payload = lookup_prior_rating_reference(
        stock_code="000250",
        fiscal_year=2023,
        eval_year=2024,
        universe="model_v1",
        path=reference_path,
    )

    assert payload == {}
