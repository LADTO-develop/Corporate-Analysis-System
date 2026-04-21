"""Cash-flow derived features — 현금흐름표 기반 파생변수.

Mapping (신용평가과정 ↔ 현금흐름표):
    현금흐름표 → 'C등급 인터뷰' 단계의 재무변제 계획과 부실 여부 판단.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bfd.features.registry import feature


@feature(
    source="cash_flow",
    kind="numeric",
    description="영업현금흐름 / 총자산 — 자산 대비 현금 창출력.",
)
def cfo_to_assets(df: pd.DataFrame) -> pd.Series:
    return df["cfo"] / df["total_assets"].replace(0, np.nan)


@feature(
    source="cash_flow",
    kind="numeric",
    description="영업현금흐름 / 총부채 — 현금흐름 기반 부채 상환능력.",
)
def cfo_to_total_debt(df: pd.DataFrame) -> pd.Series:
    return df["cfo"] / df["total_liabilities"].replace(0, np.nan)


@feature(
    source="cash_flow",
    kind="numeric",
    description="자유현금흐름 = 영업현금흐름 - 자본적지출.",
)
def free_cash_flow(df: pd.DataFrame) -> pd.Series:
    capex = df["capex"].abs() if "capex" in df.columns else 0
    return df["cfo"] - capex


@feature(
    source="cash_flow",
    kind="boolean",
    description=(
        "흑자도산 의심 플래그: 영업이익 > 0 이면서 영업현금흐름 < 0."
        "이익은 나는데 현금은 마르는 상태."
    ),
)
def profit_but_negative_cfo(df: pd.DataFrame) -> pd.Series:
    return ((df["operating_income"] > 0) & (df["cfo"] < 0)).astype(int)


@feature(
    source="cash_flow",
    kind="boolean",
    description="영업·투자·재무 모두 음수 — 전방위 유동성 압박.",
)
def triple_negative_cashflow(df: pd.DataFrame) -> pd.Series:
    return ((df["cfo"] < 0) & (df["cfi"] < 0) & (df["cff"] < 0)).astype(int)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every cash-flow feature and return a new DataFrame."""
    from bfd.features.registry import REGISTRY

    out = df[["corp_code", "fiscal_year"]].copy()
    for spec in REGISTRY.list_subset("kospi_v1"):
        if spec.source != "cash_flow":
            continue
        out[spec.name] = spec.fn(df)
    return out
