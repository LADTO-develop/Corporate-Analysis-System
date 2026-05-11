"""Fetch latest macro-market indicators from the Bank of Korea ECOS API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast
from urllib import error, parse, request

from cas.macro.schemas import EcosIndicatorSpec, EcosObservation, EcosRegistry, MacroCycle

ECOS_STATISTIC_SEARCH_URL = "https://ecos.bok.or.kr/api/StatisticSearch"


class EcosApiError(RuntimeError):
    """Raised when ECOS live data cannot be fetched or parsed."""


def load_ecos_registry(path: str | Path) -> EcosRegistry:
    """Load and validate the ECOS indicator registry."""
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    return EcosRegistry.model_validate(payload)


def fetch_latest_observations(
    registry: EcosRegistry,
    *,
    api_key: str,
    as_of_date: date,
    timeout_seconds: int = 20,
) -> tuple[list[EcosObservation], list[str]]:
    """Fetch latest available observations for all registry indicators."""
    observations: list[EcosObservation] = []
    missing: list[str] = []
    for spec in registry.indicators:
        try:
            observation = fetch_latest_observation(
                spec,
                api_key=api_key,
                as_of_date=as_of_date,
                timeout_seconds=timeout_seconds,
            )
        except EcosApiError:
            missing.append(spec.code)
            continue
        if observation is None:
            missing.append(spec.code)
            continue
        observations.append(observation)
    return observations, missing


def fetch_latest_observation(
    spec: EcosIndicatorSpec,
    *,
    api_key: str,
    as_of_date: date,
    timeout_seconds: int = 20,
) -> EcosObservation | None:
    """Fetch the latest available ECOS observation for one indicator."""
    start_time, end_time = _period_range(spec, as_of_date)
    encoded_item = parse.quote(spec.item_code, safe="")
    url = (
        f"{ECOS_STATISTIC_SEARCH_URL}/{api_key}/json/kr/1/1000/"
        f"{spec.stat_code}/{spec.cycle}/{start_time}/{end_time}/{encoded_item}"
    )
    payload = _fetch_json(url, timeout_seconds=timeout_seconds)
    rows = _extract_rows(payload)
    if not rows:
        return None

    latest_row = max(rows, key=lambda row: _row_text(row, "TIME"))
    value = _parse_float(_row_text(latest_row, "DATA_VALUE"))
    if value is None:
        return None

    time_value = _row_text(latest_row, "TIME")
    observed_at = _parse_observed_date(time_value, spec.cycle)
    return EcosObservation(
        code=spec.code,
        name_kr=spec.name_kr,
        name_en=spec.name_en,
        value=value,
        unit=spec.unit,
        time=time_value,
        observed_at=observed_at.isoformat(),
        cycle=spec.cycle,
        stat_code=spec.stat_code,
        item_code=spec.item_code,
        lag_days=(as_of_date - observed_at).days,
        description_kr=spec.description_kr,
    )


def _fetch_json(url: str, *, timeout_seconds: int) -> object:
    """Return parsed JSON from a URL using only the standard library."""
    try:
        response = request.urlopen(url, timeout=timeout_seconds)
    except error.URLError as exc:
        raise EcosApiError(f"ECOS request failed: {exc}") from exc

    try:
        body = response.read().decode("utf-8")
    finally:
        response.close()

    try:
        return cast(object, json.loads(body))
    except json.JSONDecodeError as exc:
        raise EcosApiError("ECOS response was not valid JSON.") from exc


def _extract_rows(payload: object) -> list[Mapping[str, object]]:
    """Extract ECOS row objects from the StatisticSearch response."""
    if not isinstance(payload, dict):
        return []
    statistic_search = payload.get("StatisticSearch")
    if not isinstance(statistic_search, dict):
        return []
    rows = statistic_search.get("row")
    if not isinstance(rows, list):
        return []
    return [cast(Mapping[str, object], row) for row in rows if isinstance(row, dict)]


def _row_text(row: Mapping[str, object], key: str) -> str:
    """Return a row field as stripped text."""
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _parse_float(raw_value: str) -> float | None:
    """Parse an ECOS numeric value, returning ``None`` for empty values."""
    if not raw_value:
        return None
    try:
        return float(raw_value.replace(",", ""))
    except ValueError:
        return None


def _period_range(spec: EcosIndicatorSpec, as_of_date: date) -> tuple[str, str]:
    """Create ECOS start and end period strings for an indicator."""
    start_date = as_of_date - timedelta(days=spec.lookback_days)
    return _format_period(start_date, spec.cycle), _format_period(as_of_date, spec.cycle)


def _format_period(value: date, cycle: MacroCycle) -> str:
    """Format a date as an ECOS period code for the given cycle."""
    if cycle == "D":
        return value.strftime("%Y%m%d")
    if cycle == "M":
        return value.strftime("%Y%m")
    if cycle == "Q":
        quarter = ((value.month - 1) // 3) + 1
        return f"{value.year}Q{quarter}"
    return value.strftime("%Y")


def _parse_observed_date(time_value: str, cycle: MacroCycle) -> date:
    """Convert an ECOS period code to a conservative observation date."""
    if cycle == "D":
        return datetime.strptime(time_value, "%Y%m%d").date()
    if cycle == "M":
        return datetime.strptime(time_value, "%Y%m").date()
    if cycle == "Q":
        year, quarter = time_value.split("Q", maxsplit=1)
        month = (int(quarter) - 1) * 3 + 1
        return date(int(year), month, 1)
    return date(int(time_value), 1, 1)
