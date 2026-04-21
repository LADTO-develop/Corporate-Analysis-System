"""Feature-layer tools exposed to agents."""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from bfd.features import balance_sheet, cash_flow, equity_changes, income_statement
from bfd.features.registry import REGISTRY


@tool
def compute_balance_sheet_features(row: dict[str, float]) -> dict[str, float]:
    """Run every balance-sheet feature on a single firm-year row.

    Args:
        row: dict of balance-sheet fields (total_assets, total_liabilities, …).

    Returns:
        Dict of feature name → numeric value.
    """
    df = pd.DataFrame([row])
    out = balance_sheet.compute_all(df)
    return {k: float(v) for k, v in out.iloc[0].items() if isinstance(v, (int, float))}


@tool
def compute_cash_flow_features(row: dict[str, float]) -> dict[str, float]:
    """Run every cash-flow feature on a single firm-year row."""
    df = pd.DataFrame([row])
    out = cash_flow.compute_all(df)
    return {k: float(v) for k, v in out.iloc[0].items() if isinstance(v, (int, float))}


@tool
def compute_income_statement_features(row: dict[str, float]) -> dict[str, float]:
    """Run every income-statement feature on a single firm-year row."""
    df = pd.DataFrame([row])
    out = income_statement.compute_all(df)
    return {k: float(v) for k, v in out.iloc[0].items() if isinstance(v, (int, float))}


@tool
def compute_equity_changes_features(row: dict[str, float]) -> dict[str, float]:
    """Run every equity-changes feature on a single firm-year row."""
    df = pd.DataFrame([row])
    out = equity_changes.compute_all(df)
    return {k: float(v) for k, v in out.iloc[0].items() if isinstance(v, (int, float))}


@tool
def lookup_feature_definition(name: str) -> dict[str, str]:
    """Return the registered spec (source, kind, description) of a feature."""
    if name not in REGISTRY:
        return {"found": "false", "detail": f"Unknown feature: {name}"}
    spec = REGISTRY.get(name)
    return {
        "found": "true",
        "name": spec.name,
        "source": spec.source,
        "kind": spec.kind,
        "description": spec.description,
        "subsets": ",".join(spec.subsets),
    }
