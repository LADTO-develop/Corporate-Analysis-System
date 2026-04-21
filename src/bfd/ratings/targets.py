"""Target variable construction — investment-grade vs speculative binarisation.

Project decision (see TEMPMODELPLAN §2): instead of multi-class rating
prediction, we collapse to a binary target at the standard investment-grade
cutoff ``BBB-`` / ``BB+``. This boundary is where Korean banks and insurers
typically draw their risk-weighted capital thresholds, so it's also the
business-relevant line for "borderline firm".

    healthy     (target = 0)  : AAA .. BBB-       (investment grade)
    borderline  (target = 1)  : BB+ .. D          (speculative)
    NR / excluded               -> dropped upstream
"""

from __future__ import annotations

import pandas as pd

from bfd.ratings.agencies import DEFAULT_AGENCY_WEIGHTS
from bfd.ratings.normalize import RATING_TO_NOTCH, rating_to_notch
from bfd.utils.logging import get_logger

logger = get_logger(__name__)

# BBB- is notch 13 (with AAA = 22). Anything strictly below BBB- is speculative.
INVESTMENT_GRADE_MIN_NOTCH = RATING_TO_NOTCH["BBB-"]


def to_binary_target(rating: str) -> int:
    """Return 1 if the canonical rating is speculative grade, else 0.

    Raises if rating is ``"NR"`` — callers must filter those out.
    """
    notch = rating_to_notch(rating)
    return 0 if notch >= INVESTMENT_GRADE_MIN_NOTCH else 1


def add_binary_target(
    df: pd.DataFrame,
    *,
    rating_col: str = "rating_normalized",
    target_col: str = "target",
    drop_nr: bool = True,
) -> pd.DataFrame:
    """Append a binary ``target`` column to ``df``.

    Args:
        df: must contain the normalized rating column.
        rating_col: name of the normalized rating column.
        target_col: name of the new binary target column.
        drop_nr: if True, rows with ``"NR"`` are dropped.
    """
    out = df.copy()
    if drop_nr:
        before = len(out)
        out = out.loc[out[rating_col] != "NR"].reset_index(drop=True)
        logger.info("drop_nr", before=before, after=len(out), dropped=before - len(out))
    out[target_col] = out[rating_col].apply(to_binary_target).astype(int)
    return out


# ---------------------------------------------------------------------------
# Per-firm-year aggregation across multiple agencies
# ---------------------------------------------------------------------------
def aggregate_ratings_per_firm_year(
    ratings: pd.DataFrame,
    *,
    market_weights: dict[str, dict[str, float]] | None = None,
    rating_col: str = "rating_normalized",
) -> pd.DataFrame:
    """Collapse multi-agency ratings to one row per ``(corp_code, rating_year)``.

    The aggregation uses a weighted average of notches (per-market weights),
    then rounds to the nearest integer notch to produce the consensus rating.
    """
    if market_weights is None:
        market_weights = DEFAULT_AGENCY_WEIGHTS

    df = ratings.copy()
    df = df.loc[df[rating_col] != "NR"].reset_index(drop=True)
    df["notch"] = df[rating_col].apply(rating_to_notch)

    # Attach per-agency weight by market
    def _weight(row: pd.Series) -> float:
        return market_weights.get(row["market"], {}).get(row["agency"], 0.1)

    df["w"] = df.apply(_weight, axis=1)

    grouped = (
        df.groupby(["corp_code", "rating_year", "market"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "consensus_notch": (g["notch"] * g["w"]).sum() / g["w"].sum(),
                    "n_agencies": len(g),
                    "agencies": ",".join(sorted(set(g["agency"]))),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    from bfd.ratings.normalize import NOTCH_TO_RATING

    grouped["consensus_notch_int"] = grouped["consensus_notch"].round().astype(int)
    grouped["rating_normalized"] = grouped["consensus_notch_int"].map(NOTCH_TO_RATING)
    return grouped
