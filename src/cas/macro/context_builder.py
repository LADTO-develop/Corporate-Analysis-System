"""Build MacroMarketAgent context bundles from live ECOS observations."""

from __future__ import annotations

import csv
import json
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from cas.macro.ecos_fetcher import fetch_latest_observations, load_ecos_registry
from cas.macro.schemas import (
    DerivedMacroMetric,
    DerivedMetricSpec,
    EcosObservation,
    EcosRegistry,
    MacroMarketContext,
)

KOREA_TIMEZONE = ZoneInfo("Asia/Seoul")


def build_macro_market_context(
    *,
    registry_path: str | Path,
    api_key: str,
    as_of_date: date | None = None,
) -> MacroMarketContext:
    """Collect live ECOS observations and return a macro context bundle."""
    effective_date = as_of_date or datetime.now(KOREA_TIMEZONE).date()
    registry = load_ecos_registry(registry_path)
    observations, missing_indicators = fetch_latest_observations(
        registry,
        api_key=api_key,
        as_of_date=effective_date,
    )
    derived_metrics = _build_derived_metrics(registry, observations)
    stale_indicators = _find_stale_indicators(registry, observations)
    notes = [
        "ECOS 지표는 공표 주기가 서로 달라 최신 관측일이 지표별로 다를 수 있습니다.",
        "월별 지표는 해당 월의 1일을 관측 기준일로 저장합니다.",
        "MacroMarketAgent는 XGBoost 예측확률을 직접 수정하지 않고 위원회 검토용 해석만 생성합니다.",
    ]
    if missing_indicators:
        notes.append(f"수집 실패 또는 미공표 지표: {', '.join(missing_indicators)}")
    if stale_indicators:
        notes.append(f"공표 지연 점검 필요 지표: {', '.join(stale_indicators)}")

    return MacroMarketContext(
        produced_at=_now(),
        as_of_date=effective_date.isoformat(),
        source_name=registry.source_name_kr,
        observations=observations,
        derived_metrics=derived_metrics,
        missing_indicators=missing_indicators,
        stale_indicators=stale_indicators,
        notes_kr=notes,
    )


def write_macro_market_context(
    context: MacroMarketContext,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write context JSON and flat CSV artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    context_path = output_path / "macro_market_context.json"
    raw_path = output_path / "ecos_macro_raw_latest.csv"
    _write_json(context_path, context.model_dump(mode="json"))
    write_macro_context_csv(context, raw_path)
    return {
        "macro_market_context": context_path,
        "ecos_macro_raw_latest": raw_path,
    }


def macro_context_rows(context: MacroMarketContext) -> list[dict[str, object]]:
    """Flatten the macro context into CSV-friendly rows."""
    rows: list[dict[str, object]] = []
    for observation in context.observations:
        rows.append(
            {
                "kind": "observation",
                "code": observation.code,
                "name_kr": observation.name_kr,
                "value": observation.value,
                "unit": observation.unit,
                "observed_at": observation.observed_at,
                "source": observation.source,
                "formula": "",
                "input_codes": "",
                "lag_days": observation.lag_days,
            }
        )
    for metric in context.derived_metrics:
        rows.append(
            {
                "kind": "derived",
                "code": metric.code,
                "name_kr": metric.name_kr,
                "value": metric.value,
                "unit": metric.unit,
                "observed_at": metric.observed_at,
                "source": metric.source,
                "formula": metric.formula,
                "input_codes": ",".join(metric.input_codes),
                "lag_days": "",
            }
        )
    return rows


def write_macro_context_csv(context: MacroMarketContext, path: str | Path) -> None:
    """Write the macro context rows to a CSV file."""
    rows = macro_context_rows(context)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "code",
        "name_kr",
        "value",
        "unit",
        "observed_at",
        "source",
        "formula",
        "input_codes",
        "lag_days",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_derived_metrics(
    registry: EcosRegistry,
    observations: list[EcosObservation],
) -> list[DerivedMacroMetric]:
    """Compute derived metrics defined in the registry."""
    by_code = {observation.code: observation for observation in observations}
    derived: list[DerivedMacroMetric] = []
    for spec in registry.derived_metrics:
        metric = _build_subtraction_metric(spec, by_code)
        if metric is not None:
            derived.append(metric)
    return derived


def _build_subtraction_metric(
    spec: DerivedMetricSpec,
    by_code: dict[str, EcosObservation],
) -> DerivedMacroMetric | None:
    """Build a derived metric when the registry formula is ``left - right``."""
    parts = [part.strip() for part in spec.formula.split(" - ", maxsplit=1)]
    if len(parts) != 2:
        return None
    left = by_code.get(parts[0])
    right = by_code.get(parts[1])
    if left is None or right is None:
        return None

    observed_dates = [
        datetime.fromisoformat(left.observed_at).date(),
        datetime.fromisoformat(right.observed_at).date(),
    ]
    return DerivedMacroMetric(
        code=spec.code,
        name_kr=spec.name_kr,
        name_en=spec.name_en,
        value=round(left.value - right.value, 6),
        unit=spec.unit,
        formula=spec.formula,
        input_codes=spec.input_codes,
        observed_at=min(observed_dates).isoformat(),
        description_kr=spec.description_kr,
    )


def _find_stale_indicators(
    registry: EcosRegistry,
    observations: list[EcosObservation],
) -> list[str]:
    """Return indicators whose latest ECOS observation is older than expected."""
    specs = {spec.code: spec for spec in registry.indicators}
    stale: list[str] = []
    for observation in observations:
        threshold = specs[observation.code].stale_after_days
        if observation.lag_days > threshold:
            stale.append(observation.code)
    return stale


def _now() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: str | Path, payload: object) -> None:
    """Write JSON without importing the broader project I/O module."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
