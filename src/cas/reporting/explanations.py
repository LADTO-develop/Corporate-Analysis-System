"""Helpers for exporting generic feature-attribution payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cas.utils.io import ensure_dir, write_json


def global_explanation(importances: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize a global importance payload into a DataFrame."""
    if isinstance(importances, pd.DataFrame):
        return importances.copy()
    if not importances:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame(importances)


def local_explanation(
    attributions: list[dict[str, Any]],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Sort and trim local attributions for downstream report rendering."""
    if not attributions:
        return []

    normalized: list[dict[str, Any]] = [
        {
            "feature": str(item.get("feature", "")),
            "score": float(item.get("score", 0.0)),
            "contribution_direction": str(
                item.get(
                    "contribution_direction",
                    "+" if float(item.get("score", 0.0)) > 0 else "-",
                )
            ),
        }
        for item in attributions
    ]
    ranked = sorted(normalized, key=lambda item: abs(item["score"]), reverse=True)
    return ranked[:top_k]


def export_global(
    importances: pd.DataFrame | list[dict[str, Any]],
    output_dir: str | Path,
    *,
    basename: str = "global_explanations",
) -> dict[str, str]:
    """Export global feature importance to JSON and CSV."""
    out_dir = ensure_dir(output_dir)
    df = global_explanation(importances)
    csv_path = out_dir / f"{basename}.csv"
    json_path = out_dir / f"{basename}.json"
    df.to_csv(csv_path, index=False)
    write_json(df.to_dict(orient="records"), json_path)
    return {"csv": str(csv_path), "json": str(json_path)}
