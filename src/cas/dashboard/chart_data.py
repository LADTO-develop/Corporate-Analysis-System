"""Chart data helpers that do not depend on Streamlit or Altair."""

from __future__ import annotations

import math
from contextlib import suppress

import pandas as pd


def finite_float_or_none(value: object) -> float | None:
    """Return a float only when the input carries a finite numeric value."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, str):
        with suppress(AttributeError, TypeError, ValueError):
            value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_chart_frame(
    data: list[dict[str, object]] | pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """Return chart data with non-finite numeric values removed."""
    frame = pd.DataFrame(data).copy()
    if frame.empty:
        return frame
    existing_columns = [column for column in numeric_columns if column in frame.columns]
    for column in existing_columns:
        frame[column] = frame[column].map(finite_float_or_none)
    if existing_columns:
        frame = frame.dropna(subset=existing_columns)
    return frame
