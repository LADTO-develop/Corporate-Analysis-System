"""Normalize heterogeneous agency rating codes to a common scale.

All five Korean agencies use the same S&P-style letter backbone (AAA, AA+, …,
D) for long-term credit ratings, but small variations exist in how they
denote modifiers and how short-term ratings map into the long-term scale.

The mapping table is stored at ``data/external/rating_scale_mapping.csv`` so
non-engineers can audit it; this module only loads and applies it. If that
file is missing we fall back to an in-code identity mapping (the five
agencies largely agree on the backbone).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from bfd.utils.logging import get_logger

logger = get_logger(__name__)

_MAPPING_PATH = Path("data/external/rating_scale_mapping.csv")


# The canonical 22-notch long-term scale used throughout the project.
CANONICAL_SCALE: list[str] = [
    "AAA",
    "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "CCC+", "CCC", "CCC-",
    "CC", "C", "D",
]
CANONICAL_SCALE_SET = set(CANONICAL_SCALE)


# Integer encoding — higher = better credit quality
RATING_TO_NOTCH: dict[str, int] = {r: len(CANONICAL_SCALE) - i for i, r in enumerate(CANONICAL_SCALE)}
NOTCH_TO_RATING: dict[int, str] = {v: k for k, v in RATING_TO_NOTCH.items()}


@lru_cache(maxsize=1)
def _load_mapping_table() -> pd.DataFrame:
    """Load the (agency, raw, normalized) mapping table if present."""
    if not _MAPPING_PATH.exists():
        logger.warning(
            "rating_mapping_file_missing",
            path=str(_MAPPING_PATH),
            detail="Falling back to identity normalization on canonical backbone.",
        )
        return pd.DataFrame(columns=["agency", "rating_raw", "rating_normalized"])
    df = pd.read_csv(_MAPPING_PATH, dtype=str)
    required = {"agency", "rating_raw", "rating_normalized"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{_MAPPING_PATH} must contain columns {required}; got {set(df.columns)}"
        )
    return df


def normalize_rating(raw: str | float, agency: str) -> str:
    """Normalize one raw rating code to the canonical scale.

    Raises ``ValueError`` if the code is unknown and not already canonical.
    Returns the uppercased canonical notch (e.g. ``"BBB-"``).
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "NR"
    raw_s = str(raw).strip().upper()
    if raw_s in {"", "NR", "N/A", "NA", "-"}:
        return "NR"

    table = _load_mapping_table()
    if not table.empty:
        hit = table[(table["agency"] == agency) & (table["rating_raw"].str.upper() == raw_s)]
        if not hit.empty:
            return hit.iloc[0]["rating_normalized"].upper()

    # Fallback — if the raw code already looks canonical, accept it.
    if raw_s in CANONICAL_SCALE_SET:
        return raw_s
    raise ValueError(f"Unknown rating code {raw!r} from agency {agency!r}")


def rating_to_notch(rating: str) -> int:
    """Map a canonical rating to its integer notch (22 = AAA, 1 = D)."""
    if rating == "NR":
        raise ValueError("'NR' has no notch; filter before calling.")
    return RATING_TO_NOTCH[rating]


def notch_to_rating(notch: int) -> str:
    """Inverse of ``rating_to_notch``."""
    return NOTCH_TO_RATING[notch]
