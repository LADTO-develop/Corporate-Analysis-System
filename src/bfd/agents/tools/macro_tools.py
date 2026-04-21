"""Macro-layer tools exposed to agents — thin wrappers over ECOSClient."""

from __future__ import annotations

from bfd.data.loaders.ecos import Cycle, ECOSClient
from langchain_core.tools import tool


@tool
def fetch_ecos_series(
    stat_code: str,
    cycle: str,
    start: str,
    end: str,
    item_code: str = "",
) -> list[dict[str, object]]:
    """Fetch a raw ECOS series between two dates.

    Args:
        stat_code: ECOS statistic code (e.g. ``"722Y001"``).
        cycle: one of ``"D"``, ``"M"``, ``"Q"``, ``"A"``.
        start: ECOS-formatted start date (D=YYYYMMDD, M=YYYYMM, Q=YYYYQn, A=YYYY).
        end: same format as ``start``.
        item_code: optional sub-series identifier.
    """
    client = ECOSClient()
    df = client.fetch_series(
        stat_code=stat_code,
        cycle=cycle,  # type: ignore[arg-type]
        start=start,
        end=end,
        item_code=item_code,
    )
    return df.to_dict(orient="records")


@tool
def compute_credit_spread(fiscal_year: int) -> dict[str, float | None]:
    """Compute the BBB-/AA- corporate bond credit spread at fiscal year end.

    Args:
        fiscal_year: the FY whose Dec 31 we snapshot to.
    """
    from bfd.data.alignment import as_of_ecos_snapshot

    client = ECOSClient()
    series = client.fetch_all_configured(
        start=f"{fiscal_year - 1}0101", end=f"{fiscal_year}1231"
    )
    aa = as_of_ecos_snapshot(series["corp_bond_3y_aa_minus"], fiscal_year)
    bbb = as_of_ecos_snapshot(series["corp_bond_3y_bbb_minus"], fiscal_year)
    spread = None if aa is None or bbb is None else bbb - aa
    return {"aa_minus": aa, "bbb_minus": bbb, "credit_spread": spread}


@tool
def compute_fx_volatility(fiscal_year: int, window: int = 30) -> dict[str, float | None]:
    """Compute rolling stdev of USD/KRW log returns at FY end."""
    from bfd.features.macro import rolling_log_return_std

    client = ECOSClient()
    series = client.fetch_all_configured(
        start=f"{fiscal_year - 1}0101", end=f"{fiscal_year}1231"
    )
    usd_krw = series["usd_krw"].sort_values("time_parsed")
    vol = rolling_log_return_std(usd_krw["value"], window=window).iloc[-1]
    return {"fx_volatility_30d": float(vol) if vol == vol else None}  # NaN-safe
