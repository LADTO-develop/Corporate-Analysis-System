"""Fiscal-calendar helpers for the Korean markets.

Most Korean listed firms use a calendar-year fiscal year (12월 결산).
Sec filings deadlines:
  - annual report 사업보고서: within 90 days of fiscal year end
  - quarterly reports:         within 45 days of quarter end
This means for FY ``t`` data, the earliest legitimate observation date is
roughly ``t+1`` March/April, which is why ``t → t+1`` rating mapping is safe.
"""

from __future__ import annotations

from datetime import date, timedelta


def fiscal_year_end(year: int) -> date:
    """Return the calendar-year fiscal year end (Dec 31)."""
    return date(year, 12, 31)


def annual_report_deadline(fiscal_year: int) -> date:
    """Annual report filing deadline (사업보고서 제출 기한).

    Korean Financial Investment Services Act §159: 90 days after fiscal year end.
    """
    return fiscal_year_end(fiscal_year) + timedelta(days=90)


def earliest_observable_date(fiscal_year: int) -> date:
    """Earliest date at which a firm's FY ``year`` financials are publicly
    observable — used as the anchor for ``t → t+1`` rating mapping.

    We use the annual report deadline as the conservative upper bound.
    """
    return annual_report_deadline(fiscal_year)


def target_rating_year(fiscal_year: int) -> int:
    """Given fiscal year ``t``, return the target rating year ``t + 1``.

    This is the single source of truth for the mapping rule; all splitters
    and leakage checks must reference this function.
    """
    return fiscal_year + 1
