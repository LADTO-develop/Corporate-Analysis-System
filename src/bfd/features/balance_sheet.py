"""Balance-sheet derived features — 재무상태표 기반 파생변수.

Following credit-analysis convention: liquidity, leverage, capital-impairment
flags. Each function receives a wide DataFrame and returns a Series aligned
to its index.

Reference mapping (신용평가과정 ↔ 재무상태표):
    재무상태표 → '내부정보 분석' 단계의 자산 건전성 평가
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bfd.features.registry import feature


# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------
@feature(
    source="balance_sheet",
    kind="numeric",
    description="유동비율 = 유동자산 / 유동부채. 단기 지급능력.",
)
def current_ratio(df: pd.DataFrame) -> pd.Series:
    return df["current_assets"] / df["current_liabilities"].replace(0, np.nan)


@feature(
    source="balance_sheet",
    kind="numeric",
    description="당좌비율 근사 = (유동자산 - 재고) / 유동부채. 재고 항목이 없으면 유동비율로 대체.",
)
def quick_ratio(df: pd.DataFrame) -> pd.Series:
    inventory = df.get("inventory", pd.Series(0, index=df.index))
    return (df["current_assets"] - inventory) / df["current_liabilities"].replace(0, np.nan)


# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------
@feature(
    source="balance_sheet",
    kind="numeric",
    description="부채비율 = 총부채 / 자기자본. 대표적인 재무건전성 지표.",
)
def debt_to_equity(df: pd.DataFrame) -> pd.Series:
    return df["total_liabilities"] / df["total_equity"].replace(0, np.nan)


@feature(
    source="balance_sheet",
    kind="numeric",
    description="부채자산비율 = 총부채 / 총자산. 레버리지 강도.",
)
def debt_to_assets(df: pd.DataFrame) -> pd.Series:
    return df["total_liabilities"] / df["total_assets"].replace(0, np.nan)


@feature(
    source="balance_sheet",
    kind="numeric",
    description="자기자본비율 = 자기자본 / 총자산. 완충력.",
)
def equity_ratio(df: pd.DataFrame) -> pd.Series:
    return df["total_equity"] / df["total_assets"].replace(0, np.nan)


# ---------------------------------------------------------------------------
# Capital impairment flags
# ---------------------------------------------------------------------------
@feature(
    source="balance_sheet",
    kind="boolean",
    description="자본잠식 여부 = 자기자본 < 0. 한계기업 핵심 신호.",
)
def capital_impairment(df: pd.DataFrame) -> pd.Series:
    return (df["total_equity"] < 0).astype(int)


@feature(
    source="balance_sheet",
    kind="boolean",
    description="부분 자본잠식 = 자기자본 < 자본금. 이익잉여금의 음수 전환.",
    dependencies=("total_equity", "paid_in_capital"),
)
def partial_capital_impairment(df: pd.DataFrame) -> pd.Series:
    if "paid_in_capital" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return ((df["total_equity"] > 0) & (df["total_equity"] < df["paid_in_capital"])).astype(int)


@feature(
    source="balance_sheet",
    kind="numeric",
    description="자본잠식률 = (자본금 - 자기자본) / 자본금. 양수면 잠식 진행 중.",
)
def capital_impairment_ratio(df: pd.DataFrame) -> pd.Series:
    if "paid_in_capital" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return (df["paid_in_capital"] - df["total_equity"]) / df["paid_in_capital"].replace(0, np.nan)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every balance-sheet feature and return a new DataFrame."""
    from bfd.features.registry import REGISTRY

    out = df[["corp_code", "fiscal_year"]].copy()
    for spec in REGISTRY.list_subset("kospi_v1"):
        if spec.source != "balance_sheet":
            continue
        out[spec.name] = spec.fn(df)
    return out
