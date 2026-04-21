"""Loader for credit-rating records from the five covered agencies.

Expected file layout under ``data/raw/ratings/``::

    ratings_{agency_slug}.csv     # raw per-agency exports

Agency slugs follow ``bfd.ratings.agencies.AGENCY_SLUG`` (한국기업평가 → kisr, etc.).
Each file must contain at minimum:
    corp_code, rating_date, rating_raw, outlook (optional), market

Normalization to a common scale happens in ``bfd.ratings.normalize``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bfd.data.schemas import rating_schema
from bfd.ratings.agencies import AGENCY_SLUG
from bfd.ratings.normalize import normalize_rating
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


class RatingsLoader:
    """Loads per-agency CSVs and concatenates them into a single normalized table."""

    def __init__(self, root: str | Path = "data/raw/ratings") -> None:
        self.root = Path(root)

    def load_agency(self, agency_name: str) -> pd.DataFrame:
        """Load one agency's raw file and normalize its rating codes."""
        slug = AGENCY_SLUG[agency_name]
        path = self.root / f"ratings_{slug}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Agency file missing: {path}")

        df = pd.read_csv(path, dtype={"corp_code": str})
        df["rating_date"] = pd.to_datetime(df["rating_date"], errors="raise")
        df["rating_year"] = df["rating_date"].dt.year
        df["agency"] = agency_name
        df["rating_normalized"] = df["rating_raw"].apply(
            lambda r: normalize_rating(r, agency=agency_name)
        )
        if "outlook" not in df.columns:
            df["outlook"] = "NR"

        return df

    def load_all(self) -> pd.DataFrame:
        """Load every agency file that exists under ``root`` and concatenate."""
        frames: list[pd.DataFrame] = []
        for agency in AGENCY_SLUG:
            try:
                frames.append(self.load_agency(agency))
            except FileNotFoundError as exc:
                logger.warning("ratings_agency_file_missing", agency=agency, error=str(exc))

        if not frames:
            raise RuntimeError(f"No agency rating files found under {self.root}")

        combined = pd.concat(frames, ignore_index=True)
        combined = rating_schema.validate(combined, lazy=True)
        return combined
