"""Validation-layer tools exposed to agents."""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool


@tool
def check_leakage(financials_year: int, rating_year: int) -> dict[str, object]:
    """Assert the ``t → t+1`` rating-mapping rule for a single pair.

    Returns ``{'ok': True}`` iff ``rating_year == financials_year + 1``.
    """
    from bfd.utils.time import target_rating_year

    expected = target_rating_year(financials_year)
    ok = expected == rating_year
    return {
        "ok": ok,
        "financials_year": financials_year,
        "rating_year": rating_year,
        "expected_rating_year": expected,
        "detail": "" if ok else "Leakage: rating_year must be financials_year + 1",
    }


@tool
def check_schema(rows: list[dict[str, object]], schema_name: str) -> dict[str, object]:
    """Validate a list of records against one of the project's pandera schemas.

    Args:
        rows: list of record dicts.
        schema_name: one of 'balance_sheet', 'income_statement', 'cash_flow',
            'equity_changes', 'footnotes', 'rating', 'supervised_dataset'.
    """
    import bfd.data.schemas as schemas_mod

    schema = getattr(schemas_mod, f"{schema_name}_schema", None)
    if schema is None:
        return {"ok": False, "detail": f"Unknown schema: {schema_name}"}
    try:
        schema.validate(pd.DataFrame(rows), lazy=True)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc)}


@tool
def check_drift(
    reference_rows: list[dict[str, float]],
    current_rows: list[dict[str, float]],
    feature: str,
    threshold: float = 0.2,
) -> dict[str, object]:
    """Simple Population-Stability-Index drift check for one feature."""
    from bfd.validation.drift import population_stability_index

    ref = pd.Series([r[feature] for r in reference_rows if feature in r])
    cur = pd.Series([r[feature] for r in current_rows if feature in r])
    if ref.empty or cur.empty:
        return {"ok": False, "detail": f"Not enough data for feature {feature!r}"}
    psi = float(population_stability_index(ref, cur))
    return {"psi": psi, "drift_detected": psi > threshold, "threshold": threshold}
