"""Lookup helpers for non-leaky prior credit-rating context."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PRIOR_RATING_REFERENCE_PATH = Path("data/evaluation/prior_rating_reference.csv")


def normalize_stock_code(value: object) -> str:
    """Normalize listed stock codes while preserving non-numeric codes."""
    text = str(value or "").strip().upper()
    if text.endswith(".0"):
        text = text.removesuffix(".0")
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def lookup_prior_rating_reference(
    *,
    stock_code: object,
    fiscal_year: object | None = None,
    eval_year: object | None = None,
    universe: str | None = None,
    path: Path = PRIOR_RATING_REFERENCE_PATH,
) -> dict[str, Any]:
    """Return the row-level prior rating context, or an empty dict if unavailable."""
    frame = _load_prior_rating_reference(path)
    if frame.empty:
        return {}

    normalized_stock_code = normalize_stock_code(stock_code)
    if not normalized_stock_code:
        return {}

    matched = frame.loc[frame["stock_code"].map(normalize_stock_code) == normalized_stock_code]
    fiscal_year_int = _optional_int(fiscal_year)
    if fiscal_year_int is not None:
        matched = matched.loc[matched["fiscal_year"] == fiscal_year_int]
    eval_year_int = _optional_int(eval_year)
    if eval_year_int is not None:
        matched = matched.loc[matched["eval_year"] == eval_year_int]
    if universe:
        universe_matches = matched.loc[matched["universe"].astype(str).eq(universe)]
        if not universe_matches.empty:
            matched = universe_matches

    if matched.empty:
        return {}

    row = matched.sort_values(["has_prior_rating", "as_of_date"], ascending=[False, False]).iloc[0]
    return {
        key: value
        for key, value in _clean_record(row.to_dict()).items()
        if key.startswith("prior_")
        or key in {"universe", "as_of_date", "has_prior_rating", "stock_code"}
    }


@lru_cache(maxsize=4)
def _load_prior_rating_reference(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    frame = frame.copy()
    frame["stock_code"] = frame["stock_code"].map(normalize_stock_code)
    for column in ["fiscal_year", "eval_year"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    if "has_prior_rating" in frame.columns:
        frame["has_prior_rating"] = frame["has_prior_rating"].map(_bool_value)
    return frame


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def _optional_int(value: object | None) -> int | None:
    try:
        if value is None:
            return None
        if pd.isna(value):
            return None
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


__all__ = ["PRIOR_RATING_REFERENCE_PATH", "lookup_prior_rating_reference", "normalize_stock_code"]
