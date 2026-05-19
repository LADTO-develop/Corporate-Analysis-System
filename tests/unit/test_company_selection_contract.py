"""Tests for the web listing company-selection contract."""

from __future__ import annotations

import pytest

from cas.agents.contracts import (
    CompanySelectionError,
    build_agent_state_seed,
    build_company_id,
    build_company_selection_from_row,
    normalize_company_selection,
)


def test_normalize_company_selection_pads_stock_code() -> None:
    selection = normalize_company_selection(
        {
            "request_id": "req-test",
            "source": "web_listing",
            "selected_at": "2026-05-11T04:30:00Z",
            "as_of_date": "2026-05-11",
            "company": {
                "market": "kosdaq",
                "stock_code": "250",
                "corp_name": " 삼천당제약(주)\u200b ",
                "corp_code": "123",
            },
            "analysis": {"fiscal_year": 2023, "eval_year": 2024},
        }
    )

    assert selection.company.market == "KOSDAQ"
    assert selection.company.stock_code == "000250"
    assert selection.company.corp_name == "삼천당제약(주)"
    assert selection.company.corp_code == "00000123"
    assert build_company_id(selection) == "KOSDAQ-000250-2023"


def test_build_agent_state_seed_uses_company_selection_contract() -> None:
    seed = build_agent_state_seed(
        {
            "request_id": "req-seed",
            "source": "web_listing",
            "selected_at": "2026-05-11T04:30:00Z",
            "as_of_date": "2026-05-11",
            "company": {
                "market": "KOSDAQ",
                "stock_code": "000250",
                "corp_name": "삼천당제약(주)",
            },
            "analysis": {"fiscal_year": 2023, "eval_year": 2024},
        }
    )

    assert seed["company_id"] == "KOSDAQ-000250-2023"
    assert seed["company_name"] == "삼천당제약(주)"
    assert seed["analysis_year"] == 2024
    assert seed["company_selection"]["request_id"] == "req-seed"


def test_build_company_selection_from_row_matches_dashboard_shape() -> None:
    payload = build_company_selection_from_row(
        {
            "market": "KOSDAQ",
            "stock_code": 250,
            "corp_name": "삼천당제약(주)",
            "fiscal_year": 2023,
            "eval_year": 2024,
        },
        request_id="req-row",
        selected_at="2026-05-11T04:30:00Z",
        as_of_date="2026-05-11",
    )

    assert payload["company"]["stock_code"] == "000250"
    assert payload["analysis"] == {"fiscal_year": 2023, "eval_year": 2024}


def test_historical_replay_allows_next_year_label_with_fiscal_year_cutoff() -> None:
    selection = normalize_company_selection(
        {
            "request_id": "req-historical-replay",
            "source": "web_listing",
            "selected_at": "2026-05-11T04:30:00Z",
            "as_of_date": "2023-12-31",
            "company": {
                "market": "KOSDAQ",
                "stock_code": "250",
                "corp_name": "삼천당제약(주)",
            },
            "analysis": {"fiscal_year": 2023, "eval_year": 2024},
        }
    )

    assert selection.analysis.fiscal_year == 2023
    assert selection.analysis.eval_year == 2024


def test_invalid_year_pair_raises_stable_error_code() -> None:
    with pytest.raises(CompanySelectionError) as error:
        normalize_company_selection(
            {
                "request_id": "req-invalid",
                "source": "web_listing",
                "selected_at": "2026-05-11T04:30:00Z",
                "as_of_date": "2026-05-11",
                "company": {
                    "market": "KOSDAQ",
                    "stock_code": "250",
                    "corp_name": "삼천당제약(주)",
                },
                "analysis": {"fiscal_year": 2023, "eval_year": 2025},
            }
        )

    assert error.value.code == "invalid_analysis_year"
