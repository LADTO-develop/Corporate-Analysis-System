"""Dataset splitters — the single source of truth for ``t → t+1`` rating mapping.

THIS MODULE IS LOAD-BEARING. If you change ``map_financials_to_next_year_rating``,
you must also update ``bfd.validation.leakage`` which asserts the invariant in CI.

The core rule:
    financials from fiscal year  t
    are paired with the rating observed at year  t + 1.

Never pair FY ``t`` financials with a rating dated inside FY ``t`` — that
rating was published after the financials, but it uses data the financials
themselves disclose, so the model would be cheating.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from bfd.utils.logging import get_logger
from bfd.utils.time import target_rating_year

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Year mapping
# ---------------------------------------------------------------------------
def map_financials_to_next_year_rating(
    financials: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    fy_col: str = "fiscal_year",
    rating_year_col: str = "rating_year",
    key_cols: tuple[str, ...] = ("corp_code",),
) -> pd.DataFrame:
    """Join financials (year ``t``) with ratings (year ``t + 1``).

    Args:
        financials: TS2000 wide table, one row per (corp_code, fiscal_year).
        ratings: rating records, one row per (corp_code, rating_year, agency).
            If multiple agencies have rated the same firm-year, the caller
            should aggregate upstream (see ``bfd.ratings.agencies``).
        fy_col: fiscal year column name in ``financials``.
        rating_year_col: rating year column name in ``ratings``.
        key_cols: other join keys (default just ``corp_code``).

    Returns:
        A DataFrame with one row per (corp_code, fiscal_year), carrying both
        the financial fields and the ``rating_normalized``/``target`` from
        ``fiscal_year + 1``. Rows without a matching next-year rating are
        dropped and logged.
    """
    if fy_col not in financials.columns:
        raise KeyError(f"{fy_col!r} not in financials")
    if rating_year_col not in ratings.columns:
        raise KeyError(f"{rating_year_col!r} not in ratings")

    fin = financials.copy()
    fin["target_rating_year"] = fin[fy_col].apply(target_rating_year)

    merged = fin.merge(
        ratings,
        left_on=[*key_cols, "target_rating_year"],
        right_on=[*key_cols, rating_year_col],
        how="inner",
        validate="many_to_one",  # ratings should be pre-aggregated per firm-year
    )

    dropped = len(fin) - len(merged)
    if dropped:
        logger.info(
            "splitter_unmatched_financials_dropped",
            before=len(fin),
            after=len(merged),
            dropped=dropped,
        )
    # Post-condition — enforced again by the schema in validation/leakage.py
    assert (merged["target_rating_year"] == merged[fy_col] + 1).all(), (
        "Post-condition violated: target_rating_year != fiscal_year + 1"
    )
    return merged


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForwardFold:
    """One fold of a walk-forward split (by fiscal year)."""

    fold_index: int
    train_years: tuple[int, ...]
    val_years: tuple[int, ...]


def walk_forward_folds(
    years: list[int],
    *,
    train_window: int,
    val_window: int = 1,
    step: int = 1,
    min_train_year: int | None = None,
) -> Iterator[WalkForwardFold]:
    """Yield ``WalkForwardFold`` objects covering the given year range.

    Example (train_window=3, val_window=1, step=1, years=[2015..2020]):
        fold 0: train=(2015,2016,2017) val=(2018,)
        fold 1: train=(2016,2017,2018) val=(2019,)
        fold 2: train=(2017,2018,2019) val=(2020,)
    """
    years_sorted = sorted(set(int(y) for y in years))
    if min_train_year is not None:
        years_sorted = [y for y in years_sorted if y >= min_train_year]

    idx = 0
    start = 0
    while True:
        train_end = start + train_window
        val_end = train_end + val_window
        if val_end > len(years_sorted):
            break
        yield WalkForwardFold(
            fold_index=idx,
            train_years=tuple(years_sorted[start:train_end]),
            val_years=tuple(years_sorted[train_end:val_end]),
        )
        idx += 1
        start += step


def slice_by_years(
    df: pd.DataFrame,
    years: tuple[int, ...],
    *,
    year_col: str = "fiscal_year",
) -> pd.DataFrame:
    """Return rows where ``year_col`` is in the given tuple."""
    return df.loc[df[year_col].isin(years)].reset_index(drop=True)
