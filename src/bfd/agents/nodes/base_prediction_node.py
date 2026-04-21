"""Base-prediction node — runs TabPFN, LightGBM, EBM on the feature vector.

Loads the latest trained ensemble artifact for the current market (via
``ModelRegistry``) and emits one ``BasePrediction`` per base model plus the
ensemble's final probability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from bfd.agents.state import AgentState, AuditEntry, BasePrediction
from bfd.models.ebm_model import EBMModel
from bfd.models.ensemble import StackingEnsemble
from bfd.models.registry import ModelRegistry
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


def run(state: AgentState) -> dict[str, Any]:
    """Predict with the latest artifact for the current market."""
    market = state["market"]
    features = state.get("features") or {}
    if not features:
        audit = AuditEntry(
            node="base_prediction",
            timestamp=_now(),
            summary="No features — skipping prediction.",
        )
        return {"audit": [audit]}

    registry = ModelRegistry()
    latest = registry.latest(market)
    if latest is None:
        audit = AuditEntry(
            node="base_prediction",
            timestamp=_now(),
            summary=f"No trained artifact for market={market}. Run `bfd-train --market {market.lower()}` first.",
        )
        return {"audit": [audit]}

    ensemble = StackingEnsemble.load(latest.path)

    # Feature vector in a single-row DataFrame
    feature_row = pd.DataFrame([features])
    base_outputs = ensemble.predict_proba_bases(feature_row)

    predictions: dict[str, BasePrediction] = {}
    for name, proba_matrix in base_outputs.items():
        p1 = float(proba_matrix[0, 1])
        predictions[name] = BasePrediction(
            model_name=name,
            proba_borderline=p1,
            feature_top_k=_top_k_from_base(ensemble, name, feature_row),
        )

    ensemble_proba = float(ensemble.predict_proba(feature_row)[0, 1])

    audit = AuditEntry(
        node="base_prediction",
        timestamp=_now(),
        summary=(
            f"Base predictions: "
            + ", ".join(f"{k}={v.proba_borderline:.3f}" for k, v in predictions.items())
            + f" | ensemble={ensemble_proba:.3f}"
        ),
        metrics={f"p_{k}": v.proba_borderline for k, v in predictions.items()}
        | {"p_ensemble": ensemble_proba},
    )
    return {
        "base_predictions": predictions,
        "ensemble_proba": ensemble_proba,
        "audit": [audit],
    }


def _top_k_from_base(
    ensemble: StackingEnsemble,
    base_name: str,
    feature_row: pd.DataFrame,
    k: int = 5,
) -> list[tuple[str, float]]:
    """Pull top-k contributing features from whichever base model supports it."""
    for entry in ensemble._bases:  # noqa: SLF001 — internal access by design
        if entry.name != base_name:
            continue
        model = entry.model
        if isinstance(model, EBMModel):
            explanation = model.explain_local(feature_row)
            # InterpretML local explanation — dict with 'specific' list
            try:
                data = explanation.data(0)
                names = data["names"]
                scores = data["scores"]
                pairs = sorted(zip(names, scores, strict=False), key=lambda p: abs(p[1]), reverse=True)
                return [(str(n), float(s)) for n, s in pairs[:k]]
            except Exception as exc:  # noqa: BLE001
                logger.debug("ebm_local_explain_failed", error=str(exc))
                return []
        if hasattr(model, "feature_importance"):
            importances = model.feature_importance()
            return [(str(n), float(v)) for n, v in importances.head(k).items()]
    return []


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
