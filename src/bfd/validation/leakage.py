"""Target-leakage assertions — *the* defence against the most dangerous failure
mode of this project.

If any of these assertions ever fails in CI, the offending commit must be
reverted before merge. Do NOT silence them with ``# type: ignore`` or
``pytest.mark.xfail``.
"""

from __future__ import annotations

import pandas as pd

from bfd.utils.logging import get_logger
from bfd.utils.time import target_rating_year

logger = get_logger(__name__)


class LeakageError(AssertionError):
    """Raised when the fiscal-year / rating-year mapping is violated."""


def assert_next_year_mapping(
    df: pd.DataFrame,
    *,
    fy_col: str = "fiscal_year",
    target_year_col: str = "target_rating_year",
) -> None:
    """Assert every row satisfies ``target_rating_year == fiscal_year + 1``.

    Raises:
        LeakageError: if any row violates the rule.
    """
    if fy_col not in df.columns or target_year_col not in df.columns:
        raise KeyError(f"Missing columns: need {fy_col!r} and {target_year_col!r}")

    expected = df[fy_col].apply(target_rating_year)
    mismatches = df.loc[df[target_year_col] != expected]
    if not mismatches.empty:
        raise LeakageError(
            f"{len(mismatches)} rows violate t → t+1 mapping. "
            f"First offending row: {mismatches.iloc[0].to_dict()}"
        )


def assert_no_future_features(
    df: pd.DataFrame,
    *,
    fy_col: str = "fiscal_year",
    feature_year_cols: list[str] | None = None,
) -> None:
    """Assert that no feature's observation year exceeds ``fiscal_year``.

    For any column whose name ends in ``_year`` (heuristic for a timestamp),
    verify the value is ``<= fiscal_year``.
    """
    candidate_cols = feature_year_cols or [
        c for c in df.columns if c.endswith("_year") and c != fy_col and c != "target_rating_year"
    ]
    for col in candidate_cols:
        if col not in df.columns:
            continue
        violators = df.loc[df[col] > df[fy_col]]
        if not violators.empty:
            raise LeakageError(
                f"Column {col!r} has values > fiscal_year in {len(violators)} rows."
            )


def assert_unique_firm_year(
    df: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("corp_code", "fiscal_year"),
) -> None:
    """Assert one row per (corp_code, fiscal_year). Duplicates indicate an
    unaggregated rating table on the right side of the join."""
    dupes = df.duplicated(subset=list(keys), keep=False)
    if dupes.any():
        n = int(dupes.sum())
        sample = df.loc[dupes].head(5)
        raise LeakageError(
            f"{n} duplicate rows on {keys}. Sample:\n{sample}"
        )


def run_all_checks(df: pd.DataFrame) -> None:
    """Run every leakage check. Called by the pipeline before training."""
    assert_unique_firm_year(df)
    assert_next_year_mapping(df)
    assert_no_future_features(df)
    logger.info("leakage_all_checks_passed", n_rows=len(df))
