"""Statement-of-Changes-in-Equity derived features — 자본변동표 기반 파생변수.

Mapping (신용평가과정 ↔ 자본변동표):
    자본변동표 → '내부정보 분석' 단계의 자본 관리 및 채권자 권리 희석 여부 분석.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bfd.features.registry import feature


@feature(
    source="equity_changes",
    kind="numeric",
    description="자기자본 증가율 = (기말자본 - 기초자본) / 기초자본.",
)
def equity_growth_rate(df: pd.DataFrame) -> pd.Series:
    beg = df["equity_beginning"].replace(0, np.nan)
    return (df["equity_ending"] - df["equity_beginning"]) / beg


@feature(
    source="equity_changes",
    kind="boolean",
    description="유상증자 여부 — 자본금 증가 이벤트 발생.",
)
def had_capital_increase(df: pd.DataFrame) -> pd.Series:
    return (df["capital_increase"].fillna(0) > 0).astype(int)


@feature(
    source="equity_changes",
    kind="boolean",
    description="감자 여부 — 자본금 감소 이벤트. 부실 징후로 간주.",
)
def had_capital_decrease(df: pd.DataFrame) -> pd.Series:
    return (df["capital_decrease"].fillna(0) > 0).astype(int)


@feature(
    source="equity_changes",
    kind="boolean",
    description="배당 정지 여부 — 전년도 배당 대비 당해 연도 0원.",
)
def dividend_suspension(df: pd.DataFrame) -> pd.Series:
    # Requires a lagged dividend column; upstream pipeline provides it if available.
    if "dividends_paid_prev" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return ((df["dividends_paid_prev"].fillna(0) > 0) & (df["dividends_paid"].fillna(0) == 0)).astype(int)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every equity-changes feature and return a new DataFrame."""
    from bfd.features.registry import REGISTRY

    out = df[["corp_code", "fiscal_year"]].copy()
    for spec in REGISTRY.list_subset("kospi_v1"):
        if spec.source != "equity_changes":
            continue
        out[spec.name] = spec.fn(df)
    return out
