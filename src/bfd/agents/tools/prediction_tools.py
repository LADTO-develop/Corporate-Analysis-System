"""Prediction-layer tools exposed to agents."""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

from bfd.models.ensemble import StackingEnsemble
from bfd.models.registry import ModelRegistry


def _load_latest(market: str) -> StackingEnsemble:
    record = ModelRegistry().latest(market)
    if record is None:
        raise RuntimeError(f"No trained artifact for market={market}")
    return StackingEnsemble.load(record.path)


@tool
def predict_with_tabpfn(features: dict[str, float], market: str) -> dict[str, float]:
    """Run ONLY the TabPFN base model on a feature dict.

    Args:
        features: dict of feature name → value.
        market: 'KOSPI' or 'KOSDAQ'.
    """
    ensemble = _load_latest(market)
    row = pd.DataFrame([features])
    probs = ensemble.predict_proba_bases(row)
    if "tabpfn" not in probs:
        return {"error": 1.0, "detail": -1.0}
    return {"proba_borderline": float(probs["tabpfn"][0, 1])}


@tool
def predict_with_lightgbm(features: dict[str, float], market: str) -> dict[str, float]:
    """Run ONLY the LightGBM base model on a feature dict."""
    ensemble = _load_latest(market)
    row = pd.DataFrame([features])
    probs = ensemble.predict_proba_bases(row)
    if "lightgbm" not in probs:
        return {"error": 1.0, "detail": -1.0}
    return {"proba_borderline": float(probs["lightgbm"][0, 1])}


@tool
def predict_with_ebm(features: dict[str, float], market: str) -> dict[str, float]:
    """Run ONLY the EBM base model on a feature dict."""
    ensemble = _load_latest(market)
    row = pd.DataFrame([features])
    probs = ensemble.predict_proba_bases(row)
    if "ebm" not in probs:
        return {"error": 1.0, "detail": -1.0}
    return {"proba_borderline": float(probs["ebm"][0, 1])}


@tool
def ensemble_predictions(features: dict[str, float], market: str) -> dict[str, float]:
    """Run all base models + meta-learner and return the ensemble P(borderline)."""
    ensemble = _load_latest(market)
    row = pd.DataFrame([features])
    p1 = float(ensemble.predict_proba(row)[0, 1])
    return {"proba_borderline": p1}


@tool
def explain_with_ebm(features: dict[str, float], market: str, top_k: int = 5) -> list[dict[str, object]]:
    """Return top-k EBM term contributions for a single prediction.

    Args:
        features: dict of feature name → value.
        market: 'KOSPI' or 'KOSDAQ'.
        top_k: number of top contributing terms to return.
    """
    from bfd.models.ebm_model import EBMModel

    ensemble = _load_latest(market)
    row = pd.DataFrame([features])
    for entry in ensemble._bases:  # noqa: SLF001
        if not isinstance(entry.model, EBMModel):
            continue
        explanation = entry.model.explain_local(row)
        try:
            data = explanation.data(0)
            pairs = sorted(
                zip(data["names"], data["scores"], strict=False),
                key=lambda p: abs(p[1]),
                reverse=True,
            )
            return [{"feature": str(n), "score": float(s)} for n, s in pairs[:top_k]]
        except Exception:  # noqa: BLE001
            return []
    return []
