"""Pandera schemas for TS2000 five-file structure and derived tables.

These schemas are the authoritative contract for every DataFrame that crosses
a module boundary in this project. If a loader's output does not match its
schema, the pipeline aborts — this is the project's defence against silent
data corruption.

Reference: https://pandera.readthedocs.io/en/stable/
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series

# ---------------------------------------------------------------------------
# Common join keys — the five TS2000 files share these columns
# ---------------------------------------------------------------------------
_CORP_CODE = pa.Column(
    str,
    checks=pa.Check.str_matches(r"^\d{6}$"),
    description="KRX 6-digit 종목코드",
    nullable=False,
)
_FISCAL_YEAR = pa.Column(
    int,
    checks=pa.Check.in_range(2000, 2100),
    description="회계연도 (YYYY)",
    nullable=False,
)
_FISCAL_QUARTER = pa.Column(
    int,
    checks=pa.Check.isin([1, 2, 3, 4]),
    description="회계 분기. 연간 결산은 4.",
    nullable=False,
)
_REPORT_TYPE = pa.Column(
    str,
    checks=pa.Check.isin(["consolidated", "separate"]),
    description="연결/별도 재무제표 구분",
    nullable=False,
)
_MARKET = pa.Column(
    str,
    checks=pa.Check.isin(["KOSPI", "KOSDAQ", "KONEX"]),
    description="시장 구분",
    nullable=False,
)


# ---------------------------------------------------------------------------
# 1. 재무상태표 Balance Sheet
# ---------------------------------------------------------------------------
balance_sheet_schema = pa.DataFrameSchema(
    {
        "corp_code": _CORP_CODE,
        "fiscal_year": _FISCAL_YEAR,
        "fiscal_quarter": _FISCAL_QUARTER,
        "report_type": _REPORT_TYPE,
        "total_assets": pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
        "current_assets": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "non_current_assets": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "total_liabilities": pa.Column(float, nullable=False, checks=pa.Check.ge(0)),
        "current_liabilities": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "non_current_liabilities": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "total_equity": pa.Column(float, nullable=False),
        "paid_in_capital": pa.Column(float, nullable=True),
        "retained_earnings": pa.Column(float, nullable=True),
    },
    strict=False,  # extra columns allowed — TS2000 has many niche items
    coerce=True,
    checks=[
        # Balance sheet identity: A = L + E (with 1% tolerance for rounding).
        pa.Check(
            lambda df: (
                (df["total_assets"] - (df["total_liabilities"] + df["total_equity"])).abs()
                <= 0.01 * df["total_assets"].abs()
            ),
            error="Balance sheet identity violation: |A - (L + E)| > 1% of A",
        ),
    ],
)


# ---------------------------------------------------------------------------
# 2. 손익계산서 Income Statement
# ---------------------------------------------------------------------------
income_statement_schema = pa.DataFrameSchema(
    {
        "corp_code": _CORP_CODE,
        "fiscal_year": _FISCAL_YEAR,
        "fiscal_quarter": _FISCAL_QUARTER,
        "report_type": _REPORT_TYPE,
        "revenue": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "cost_of_sales": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "gross_profit": pa.Column(float, nullable=True),
        "operating_income": pa.Column(float, nullable=True),
        "interest_expense": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
        "pretax_income": pa.Column(float, nullable=True),
        "net_income": pa.Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# 3. 현금흐름표 Cash Flow Statement
# ---------------------------------------------------------------------------
cash_flow_schema = pa.DataFrameSchema(
    {
        "corp_code": _CORP_CODE,
        "fiscal_year": _FISCAL_YEAR,
        "fiscal_quarter": _FISCAL_QUARTER,
        "report_type": _REPORT_TYPE,
        "cfo": pa.Column(float, nullable=True, description="영업활동 현금흐름"),
        "cfi": pa.Column(float, nullable=True, description="투자활동 현금흐름"),
        "cff": pa.Column(float, nullable=True, description="재무활동 현금흐름"),
        "capex": pa.Column(float, nullable=True, description="유형자산 취득(-)"),
        "cash_end_of_period": pa.Column(float, nullable=True, checks=pa.Check.ge(0)),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# 4. 자본변동표 Statement of Changes in Equity
# ---------------------------------------------------------------------------
equity_changes_schema = pa.DataFrameSchema(
    {
        "corp_code": _CORP_CODE,
        "fiscal_year": _FISCAL_YEAR,
        "fiscal_quarter": _FISCAL_QUARTER,
        "report_type": _REPORT_TYPE,
        "equity_beginning": pa.Column(float, nullable=True),
        "equity_ending": pa.Column(float, nullable=True),
        "capital_increase": pa.Column(float, nullable=True),
        "capital_decrease": pa.Column(float, nullable=True),
        "dividends_paid": pa.Column(float, nullable=True),
        "treasury_stock_change": pa.Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# 5. 주석 Footnotes (often free-text risk disclosures)
# ---------------------------------------------------------------------------
footnotes_schema = pa.DataFrameSchema(
    {
        "corp_code": _CORP_CODE,
        "fiscal_year": _FISCAL_YEAR,
        "fiscal_quarter": _FISCAL_QUARTER,
        "report_type": _REPORT_TYPE,
        "contingent_liabilities_text": pa.Column(str, nullable=True),
        "litigation_text": pa.Column(str, nullable=True),
        "going_concern_text": pa.Column(str, nullable=True),
        "related_party_text": pa.Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# Credit rating records
# ---------------------------------------------------------------------------
rating_schema = pa.DataFrameSchema(
    {
        "corp_code": _CORP_CODE,
        "rating_date": pa.Column(pa.DateTime, nullable=False),
        "rating_year": pa.Column(int, checks=pa.Check.in_range(2000, 2100)),
        "agency": pa.Column(
            str,
            checks=pa.Check.isin(
                ["한국기업평가", "한국신용평가", "NICE신용평가", "이크레더블", "나이스디앤비"]
            ),
            description="평가사명. 타 평가사 추가 시 이 리스트를 확장.",
        ),
        "rating_raw": pa.Column(str, nullable=False, description="평가사 원본 등급 기호"),
        "rating_normalized": pa.Column(
            str,
            nullable=False,
            description="공통 S&P 스타일 등급으로 정규화된 값 (AAA...D)",
        ),
        "outlook": pa.Column(
            str,
            nullable=True,
            checks=pa.Check.isin(["Positive", "Stable", "Negative", "Developing", "NR"]),
        ),
        "market": _MARKET,
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# Joined supervised dataset (features aligned to t+1 rating target)
# ---------------------------------------------------------------------------
supervised_dataset_schema = pa.DataFrameSchema(
    {
        "corp_code": _CORP_CODE,
        "fiscal_year": _FISCAL_YEAR,
        "target_rating_year": pa.Column(
            int,
            checks=pa.Check.in_range(2000, 2100),
            description="반드시 fiscal_year + 1이어야 함. splitters.map_financials_to_next_year_rating 참조.",
        ),
        "market": _MARKET,
        "target": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            description="1 = borderline (투기등급), 0 = healthy (투자적격)",
        ),
    },
    strict=False,
    coerce=True,
    checks=[
        pa.Check(
            lambda df: (df["target_rating_year"] == df["fiscal_year"] + 1).all(),
            error="Target leakage: target_rating_year must equal fiscal_year + 1",
        ),
    ],
)


__all__ = [
    "balance_sheet_schema",
    "income_statement_schema",
    "cash_flow_schema",
    "equity_changes_schema",
    "footnotes_schema",
    "rating_schema",
    "supervised_dataset_schema",
]
