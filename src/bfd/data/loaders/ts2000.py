"""Loader for the TS2000 five-file financial statement export.

Each of the five files is a CSV keyed on ``(corp_code, fiscal_year,
fiscal_quarter, report_type)``. ``load_all`` returns a dictionary of
validated DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from bfd.data.schemas import (
    balance_sheet_schema,
    cash_flow_schema,
    equity_changes_schema,
    footnotes_schema,
    income_statement_schema,
)
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)

FileKey = Literal[
    "balance_sheet", "income_statement", "cash_flow", "equity_changes", "footnotes"
]

_SCHEMAS = {
    "balance_sheet": balance_sheet_schema,
    "income_statement": income_statement_schema,
    "cash_flow": cash_flow_schema,
    "equity_changes": equity_changes_schema,
    "footnotes": footnotes_schema,
}


class TS2000Loader:
    """Loads, validates, and aligns the five TS2000 files."""

    def __init__(self, config_path: str | Path = "configs/data/ts2000.yaml") -> None:
        self.config = read_yaml(config_path)
        self.root = Path(self.config["root"])
        self.files: dict[str, str] = self.config["files"]
        self.encoding: str = self.config.get("preprocessing", {}).get("encoding", "cp949")

    # ------------------------------------------------------------------
    # Single-file loading
    # ------------------------------------------------------------------
    def load_file(self, key: FileKey, year: int) -> pd.DataFrame:
        """Load a single file for a given year, then validate against its schema."""
        pattern = self.files[key]
        path = self.root / pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"TS2000 file missing: {path}")

        logger.info("loading_ts2000_file", key=key, year=year, path=str(path))
        df = pd.read_csv(path, encoding=self.encoding, low_memory=False)
        df = self._normalize_columns(df)
        df = self._coerce_numeric(df)

        schema = _SCHEMAS[key]
        validated: pd.DataFrame = schema.validate(df, lazy=True)
        return validated

    def load_all(
        self,
        year: int,
        markets: list[str] | None = None,
    ) -> dict[FileKey, pd.DataFrame]:
        """Load all five files for a given year."""
        out: dict[FileKey, pd.DataFrame] = {}
        for key in _SCHEMAS:
            df = self.load_file(key, year)  # type: ignore[arg-type]
            if markets is not None and "market" in df.columns:
                df = df[df["market"].isin(markets)].reset_index(drop=True)
            out[key] = df  # type: ignore[index]
        return out

    # ------------------------------------------------------------------
    # Wide table — all five files joined on common keys
    # ------------------------------------------------------------------
    def load_wide(
        self,
        year: int,
        markets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a single wide DataFrame joining the five files on common keys."""
        parts = self.load_all(year=year, markets=markets)
        keys = ["corp_code", "fiscal_year", "fiscal_quarter", "report_type"]

        wide = parts["balance_sheet"]
        for key in ["income_statement", "cash_flow", "equity_changes", "footnotes"]:
            wide = wide.merge(
                parts[key],  # type: ignore[index]
                on=keys,
                how="outer",
                suffixes=("", f"__{key}"),
                validate="one_to_one",
            )
        return wide

    # ------------------------------------------------------------------
    # Multi-year loading
    # ------------------------------------------------------------------
    def load_range(
        self,
        start_year: int | None = None,
        end_year: int | None = None,
        markets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load and concatenate wide tables for a range of fiscal years."""
        years_cfg = self.config.get("years", {})
        start = start_year or years_cfg.get("start")
        end = end_year or years_cfg.get("end")
        markets = markets or self.config.get("markets")

        frames: list[pd.DataFrame] = []
        for year in range(start, end + 1):
            try:
                frames.append(self.load_wide(year, markets=markets))
            except FileNotFoundError as exc:
                logger.warning("ts2000_year_missing", year=year, error=str(exc))
        if not frames:
            raise RuntimeError(f"No TS2000 files found in {start}..{end}")
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace and lowercase column names for schema alignment."""
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce columns that look numeric but came in as strings (with 콤마)."""
        # TS2000 exports often have numbers like "1,234,567" in strings.
        strict = self.config.get("preprocessing", {}).get("numeric_coercion") == "strict"
        for col in df.columns:
            if df[col].dtype == object and col not in {
                "corp_code",
                "report_type",
                "market",
                "contingent_liabilities_text",
                "litigation_text",
                "going_concern_text",
                "related_party_text",
            }:
                cleaned = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
                coerced = pd.to_numeric(cleaned, errors="coerce" if not strict else "raise")
                # Only replace when we actually gained numeric content
                if coerced.notna().any():
                    df[col] = coerced
        return df
