"""Macro-economic derived features from ECOS snapshots.

These features are per-fiscal-year scalars (snapshot at 12/31) rather than
per-firm quantities, so they're broadcast onto every firm in that year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bfd.features.registry import feature


@feature(
    source="macro",
    kind="numeric",
    description="AA-/BBB- 회사채 신용 스프레드 (3년물). 경기-신용 사이클 대용지표.",
)
def credit_spread(df: pd.DataFrame) -> pd.Series:
    if "macro_corp_bond_3y_bbb_minus" in df.columns and "macro_corp_bond_3y_aa_minus" in df.columns:
        return df["macro_corp_bond_3y_bbb_minus"] - df["macro_corp_bond_3y_aa_minus"]
    return pd.Series(np.nan, index=df.index)


@feature(
    source="macro",
    kind="numeric",
    description="실질금리 = 기준금리 - CPI 전년동월비.",
)
def real_rate(df: pd.DataFrame) -> pd.Series:
    if "macro_base_rate" in df.columns and "macro_cpi_yoy" in df.columns:
        return df["macro_base_rate"] - df["macro_cpi_yoy"]
    return pd.Series(np.nan, index=df.index)


@feature(
    source="macro",
    kind="numeric",
    description="30일 USD/KRW 로그수익률 표준편차 — FX 변동성 스냅샷.",
)
def fx_volatility_30d(df: pd.DataFrame) -> pd.Series:
    # Precomputed upstream by ECOS processing since it needs the full daily series.
    return df.get("macro_fx_volatility_30d", pd.Series(np.nan, index=df.index))


def rolling_log_return_std(series: pd.Series, window: int = 30) -> pd.Series:
    """Utility: rolling stdev of log returns, used by ECOS preprocessing."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window=window, min_periods=window // 2).std()
