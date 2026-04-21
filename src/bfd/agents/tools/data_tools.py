"""Data-layer tools exposed to LangChain / LangGraph agents.

Each function wraps a loader in ``bfd.data.loaders`` with a LangChain
``@tool`` decorator so it can be attached to a tool-using agent.
"""

from __future__ import annotations

from langchain_core.tools import tool

from bfd.data.loaders.ratings import RatingsLoader
from bfd.data.loaders.ts2000 import TS2000Loader


@tool
def load_financial_statements(corp_code: str, fiscal_year: int) -> dict[str, object]:
    """Load the wide TS2000 annual-consolidated row for a firm-year.

    Args:
        corp_code: KRX 6-digit 종목코드.
        fiscal_year: 회계연도 (YYYY).

    Returns:
        A dict of field → value for the single matching row, or
        ``{'found': False}`` if no record exists.
    """
    loader = TS2000Loader()
    wide = loader.load_wide(year=fiscal_year)
    firm = wide[
        (wide["corp_code"] == corp_code)
        & (wide["fiscal_quarter"] == 4)
        & (wide["report_type"] == "consolidated")
    ]
    if firm.empty:
        return {"found": False}
    return {"found": True, **firm.iloc[0].to_dict()}


@tool
def load_ratings(corp_code: str) -> list[dict[str, object]]:
    """Return every rating observation on record for the firm.

    Args:
        corp_code: KRX 6-digit 종목코드.
    """
    loader = RatingsLoader()
    df = loader.load_all()
    hits = df[df["corp_code"] == corp_code]
    return hits.to_dict(orient="records")


@tool
def list_available_years() -> list[int]:
    """Return every fiscal year for which TS2000 files exist on disk."""
    loader = TS2000Loader()
    years_cfg = loader.config.get("years", {})
    start = int(years_cfg.get("start", 2010))
    end = int(years_cfg.get("end", 2024))

    # Probe: only return years where the balance-sheet file is present.
    available: list[int] = []
    for year in range(start, end + 1):
        path = loader.root / loader.files["balance_sheet"].format(year=year)
        if path.exists():
            available.append(year)
    return available
