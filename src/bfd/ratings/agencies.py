"""Credit rating agency metadata and market coverage.

Five agencies issue long-term credit ratings for Korean listed firms:

    한국기업평가 (KIS Ratings / Korea Ratings / KR, stylized 한기평)
    한국신용평가 (Korea Investors Service / KIS)
    NICE신용평가 (NICE Investors Service)
    이크레더블      (E-Credible; KOSDAQ/SME focus)
    나이스디앤비   (NICE D&B; KOSDAQ/SME focus)

The first three dominate KOSPI coverage; the last two have disproportionate
reach into KOSDAQ and small private firms.
"""

from __future__ import annotations

from typing import Final

# Slug used in filenames — ASCII-only for portability
AGENCY_SLUG: Final[dict[str, str]] = {
    "한국기업평가": "kr",
    "한국신용평가": "kis",
    "NICE신용평가": "nice",
    "이크레더블": "ecredible",
    "나이스디앤비": "nicednb",
}

# Reverse lookup
SLUG_TO_AGENCY: Final[dict[str, str]] = {v: k for k, v in AGENCY_SLUG.items()}

# Market-level dominance weights — used by ``aggregate_ratings`` to resolve
# conflicting ratings on the same firm-year. Values were derived from coverage
# share in the TS2000 five-year sample and are meant to be overridden by
# ``data/external/agency_weights.csv`` when that file exists.
DEFAULT_AGENCY_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    "KOSPI": {
        "한국기업평가": 0.35,
        "한국신용평가": 0.30,
        "NICE신용평가": 0.25,
        "이크레더블": 0.05,
        "나이스디앤비": 0.05,
    },
    "KOSDAQ": {
        "이크레더블": 0.30,
        "나이스디앤비": 0.25,
        "NICE신용평가": 0.20,
        "한국기업평가": 0.15,
        "한국신용평가": 0.10,
    },
}
