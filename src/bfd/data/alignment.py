"""Helpers for aligning (corp_code, time) observations across data sources.

Three sources have three different time granularities:
    * TS2000 financials  — fiscal year (+ quarter)
    * ECOS macro         — daily / monthly / quarterly / annual
    * Ratings            — any date (event-driven)
    * News               — article-level timestamps

``as_of_alignment`` collapses ECOS series to a single "as-of" value at
``fiscal_year_end(t)`` so they can be joined onto TS2000 observations.
"""

from __future__ import annotations

import pandas as pd

from bfd.utils.logging import get_logger
from bfd.utils.time import fiscal_year_end

logger = get_logger(__name__)


def as_of_ecos_snapshot(
    macro_df: pd.DataFrame,
    fiscal_year: int,
    *,
    time_col: str = "time_parsed",
    value_col: str = "value",
) -> float | None:
    """Return the most recent macro value as of ``fiscal_year_end(fiscal_year)``.

    Args:
        macro_df: output of ``ECOSClient.fetch_series`` (must contain ``time_parsed``).
        fiscal_year: the FY whose year-end we use as the snapshot anchor.

    Returns:
        Scalar value or ``None`` if no observation exists on/before the anchor.
    """
    if macro_df.empty:
        return None
    anchor = fiscal_year_end(fiscal_year)
    eligible = macro_df[macro_df[time_col] <= anchor].sort_values(time_col)
    if eligible.empty:
        return None
    return float(eligible[value_col].iloc[-1])


def join_financials_with_macro(
    financials: pd.DataFrame,
    macro_snapshots: dict[str, dict[int, float | None]],
) -> pd.DataFrame:
    """Attach macro snapshots to each financial observation.

    Args:
        financials: TS2000 wide table (must have ``fiscal_year`` column).
        macro_snapshots: ``{series_name: {fiscal_year: value}}``.
    """
    out = financials.copy()
    for series_name, year_to_value in macro_snapshots.items():
        out[f"macro_{series_name}"] = out["fiscal_year"].map(year_to_value)
    return out


def filter_to_annual_consolidated(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only annual (Q4) consolidated statements — the canonical supervised unit."""
    mask = (df["fiscal_quarter"] == 4) & (df["report_type"] == "consolidated")
    filtered = df.loc[mask].reset_index(drop=True)
    logger.info(
        "filter_annual_consolidated",
        before=len(df),
        after=len(filtered),
        dropped=len(df) - len(filtered),
    )
    return filtered
