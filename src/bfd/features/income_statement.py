"""Income-statement derived features — 손익계산서 기반 파생변수.

Mapping (신용평가과정 ↔ 손익계산서):
    손익계산서 → 'C등급 인터뷰' 단계의 수익성 및 이자보상 능력 분석.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bfd.features.registry import feature


@feature(
    source="income_statement",
    kind="numeric",
    description="매출총이익률 = 매출총이익 / 매출액.",
)
def gross_margin(df: pd.DataFrame) -> pd.Series:
    return df["gross_profit"] / df["revenue"].replace(0, np.nan)


@feature(
    source="income_statement",
    kind="numeric",
    description="영업이익률 = 영업이익 / 매출액.",
)
def operating_margin(df: pd.DataFrame) -> pd.Series:
    return df["operating_income"] / df["revenue"].replace(0, np.nan)


@feature(
    source="income_statement",
    kind="numeric",
    description="순이익률 = 순이익 / 매출액.",
)
def net_margin(df: pd.DataFrame) -> pd.Series:
    return df["net_income"] / df["revenue"].replace(0, np.nan)


@feature(
    source="income_statement",
    kind="numeric",
    description=(
        "이자보상배율 = 영업이익 / 이자비용. 3년 연속 1 미만이면 한계기업 정의 일부."
    ),
)
def interest_coverage(df: pd.DataFrame) -> pd.Series:
    return df["operating_income"] / df["interest_expense"].replace(0, np.nan)


@feature(
    source="income_statement",
    kind="boolean",
    description="이자보상배율 1 미만 플래그. (단년 스냅샷. 연속성은 별도 롤링 피처가 처리.)",
    dependencies=("interest_coverage",),
)
def interest_coverage_below_one(df: pd.DataFrame) -> pd.Series:
    ic = df["operating_income"] / df["interest_expense"].replace(0, np.nan)
    return (ic < 1).fillna(False).astype(int)


@feature(
    source="income_statement",
    kind="boolean",
    description="영업손실 발생 여부.",
)
def operating_loss_flag(df: pd.DataFrame) -> pd.Series:
    return (df["operating_income"] < 0).astype(int)


@feature(
    source="income_statement",
    kind="boolean",
    description="당기순손실 발생 여부.",
)
def net_loss_flag(df: pd.DataFrame) -> pd.Series:
    return (df["net_income"] < 0).astype(int)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every income-statement feature and return a new DataFrame."""
    from bfd.features.registry import REGISTRY

    out = df[["corp_code", "fiscal_year"]].copy()
    for spec in REGISTRY.list_subset("kospi_v1"):
        if spec.source != "income_statement":
            continue
        out[spec.name] = spec.fn(df)
    return out
