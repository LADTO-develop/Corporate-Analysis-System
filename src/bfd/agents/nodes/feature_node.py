"""Feature-engineering node.

Runs every feature function registered for the current market's subset
(defined in ``configs/market/{market}.yaml``) and stashes the resulting
single-row feature vector in ``state['features']``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bfd.agents.state import AgentState, AuditEntry
from bfd.features.registry import REGISTRY
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


def run(state: AgentState) -> dict[str, Any]:
    """Compute all features for the single firm/year in state."""
    market = state["market"].lower()
    market_cfg = read_yaml(Path(f"configs/market/{market}.yaml"))
    subset = market_cfg["feature_subset"]

    raw = state.get("raw_financials") or {}
    if not raw:
        audit = AuditEntry(
            node="feature",
            timestamp=_now(),
            summary="No raw_financials in state; skipping feature computation.",
        )
        return {"audit": [audit]}

    # Feature functions expect a DataFrame; wrap the single row.
    df = pd.DataFrame([raw])
    features: dict[str, float | int | bool] = {}

    for spec in REGISTRY.list_subset(subset):
        if spec.source == "macro":
            # Macro features are handled in macro_overlay_node
            continue
        try:
            values = spec.fn(df)
            val = values.iloc[0]
            if pd.isna(val):
                continue
            features[spec.name] = _coerce(val, spec.kind)
        except Exception as exc:  # noqa: BLE001 — logged and skipped individually
            logger.warning("feature_compute_failed", feature=spec.name, error=str(exc))

    audit = AuditEntry(
        node="feature",
        timestamp=_now(),
        summary=f"Computed {len(features)} features for subset={subset}",
        metrics={"n_features": float(len(features))},
    )
    return {"features": features, "audit": [audit]}


def _coerce(value: Any, kind: str) -> float | int | bool:
    if kind == "boolean":
        return bool(int(value))
    if kind == "numeric":
        return float(value)
    return value  # categorical — pass through


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
