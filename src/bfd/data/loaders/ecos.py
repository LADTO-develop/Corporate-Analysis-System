"""Loader for Bank of Korea ECOS macro-economic statistics.

ECOS Open API reference: https://ecos.bok.or.kr/api/
The service ID used below is ``StatisticSearch``; it returns rows for a stat
series between two dates at a given cycle (D / M / Q / A).
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)

Cycle = Literal["D", "M", "Q", "A"]

_ECOS_DATE_FORMAT = {
    "D": "%Y%m%d",
    "M": "%Y%m",
    "Q": "%Y%q",  # ECOS actually uses YYYYQQ e.g. 2024Q1 — handled specially
    "A": "%Y",
}


class ECOSClient:
    """Thin client for the ECOS StatisticSearch endpoint."""

    def __init__(self, config_path: str | Path = "configs/data/ecos.yaml") -> None:
        self.config = read_yaml(config_path)
        api = self.config["api"]
        self.base_url: str = api["base_url"]
        self.language: str = api.get("language", "kr")
        self.format: str = api.get("format", "json")
        self.timeout: int = api.get("request_timeout_seconds", 30)

        api_key_env = api.get("api_key_env", "ECOS_API_KEY")
        key = os.getenv(api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set; cannot call ECOS."
            )
        self.api_key = key

    # ------------------------------------------------------------------
    # Low-level request
    # ------------------------------------------------------------------
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}/{path}"
        logger.debug("ecos_request", url=url)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Public: fetch a single series
    # ------------------------------------------------------------------
    def fetch_series(
        self,
        stat_code: str,
        cycle: Cycle,
        start: str,
        end: str,
        item_code: str = "",
        per_page: int = 10000,
    ) -> pd.DataFrame:
        """Fetch a single ECOS statistic between ``start`` and ``end``.

        Args:
            stat_code: e.g. ``"722Y001"`` for the BOK base rate.
            cycle: one of ``"D"``, ``"M"``, ``"Q"``, ``"A"``.
            start, end: ECOS-formatted dates (D=YYYYMMDD, M=YYYYMM, Q=YYYYQ1, A=YYYY).
            item_code: sub-series identifier; leave empty for series without items.
            per_page: rows per page. ECOS caps at 100000 per request.
        """
        path = (
            f"StatisticSearch/{self.api_key}/{self.format}/{self.language}/"
            f"1/{per_page}/{stat_code}/{cycle}/{start}/{end}/{item_code}"
        )
        payload = self._get(path)

        rows = payload.get("StatisticSearch", {}).get("row", [])
        if not rows:
            logger.warning("ecos_empty_result", stat_code=stat_code, cycle=cycle)
            return pd.DataFrame(columns=["time", "value", "unit", "stat_code", "cycle"])

        df = pd.DataFrame(rows)
        df = df.rename(columns={"TIME": "time", "DATA_VALUE": "value", "UNIT_NAME": "unit"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["stat_code"] = stat_code
        df["cycle"] = cycle
        df["time_parsed"] = df["time"].apply(lambda t: _parse_ecos_time(t, cycle))
        return df[["time", "time_parsed", "value", "unit", "stat_code", "cycle"]]

    # ------------------------------------------------------------------
    # Public: fetch all configured series
    # ------------------------------------------------------------------
    def fetch_all_configured(
        self,
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch every series defined in the config between the given dates."""
        out: dict[str, pd.DataFrame] = {}
        for name, spec in self.config["series"].items():
            out[name] = self.fetch_series(
                stat_code=spec["stat_code"],
                cycle=spec["cycle"],
                start=start,
                end=end,
                item_code=spec.get("item_code", ""),
            )
        return out


# ---------------------------------------------------------------------------
# Time parsing — ECOS uses different formats per cycle
# ---------------------------------------------------------------------------
def _parse_ecos_time(raw: str, cycle: Cycle) -> date | None:
    """Parse an ECOS ``TIME`` field into a ``date``.

    Formats observed in practice:
        D: YYYYMMDD
        M: YYYYMM
        Q: YYYYQn (n = 1..4)
        A: YYYY
    """
    raw = raw.strip()
    try:
        if cycle == "D":
            return date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
        if cycle == "M":
            return date(int(raw[0:4]), int(raw[4:6]), 1)
        if cycle == "Q":
            year = int(raw[0:4])
            q = int(raw[-1])
            month = 3 * q
            return date(year, month, 1)
        if cycle == "A":
            return date(int(raw), 12, 31)
    except (ValueError, IndexError):
        logger.warning("ecos_time_parse_failure", raw=raw, cycle=cycle)
    return None
