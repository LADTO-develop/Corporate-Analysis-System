"""Feature-attribution reports — pulls EBM shape functions + top-k terms.

The EBM is kept in every ensemble specifically so this module can always
produce per-firm and global explanations without relying on post-hoc SHAP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bfd.models.ebm_model import EBMModel
from bfd.models.ensemble import StackingEnsemble
from bfd.utils.io import ensure_dir, write_json


def _find_ebm(ensemble: StackingEnsemble) -> EBMModel | None:
    for entry in ensemble._bases:  # noqa: SLF001
        if isinstance(entry.model, EBMModel):
            return entry.model
    return None


def global_explanation(ensemble: StackingEnsemble) -> pd.DataFrame:
    """Return a DataFrame of ``(term, mean_abs_score)`` for the fitted EBM."""
    ebm = _find_ebm(ensemble)
    if ebm is None:
        return pd.DataFrame(columns=["term", "mean_abs_score"])
    importances = ebm.term_importances()
    return importances.reset_index().rename(
        columns={"index": "term", 0: "mean_abs_score"}
    )


def local_explanation(
    ensemble: StackingEnsemble,
    features: dict[str, float],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return the top-k contributing terms for a single prediction."""
    ebm = _find_ebm(ensemble)
    if ebm is None:
        return []

    row = pd.DataFrame([features])
    explanation = ebm.explain_local(row)
    try:
        data = explanation.data(0)
    except Exception:  # noqa: BLE001
        return []

    pairs = sorted(
        zip(data.get("names", []), data.get("scores", []), strict=False),
        key=lambda p: abs(p[1]),
        reverse=True,
    )
    return [
        {"term": str(n), "score": float(s), "contribution_direction": "+" if s > 0 else "-"}
        for n, s in pairs[:top_k]
    ]


def export_global(
    ensemble: StackingEnsemble,
    output_dir: str | Path,
    *,
    basename: str = "ebm_global",
) -> dict[str, str]:
    """Export global EBM importance to JSON + CSV."""
    out_dir = ensure_dir(output_dir)
    df = global_explanation(ensemble)
    csv_path = out_dir / f"{basename}.csv"
    json_path = out_dir / f"{basename}.json"
    df.to_csv(csv_path, index=False)
    write_json(df.to_dict(orient="records"), json_path)
    return {"csv": str(csv_path), "json": str(json_path)}
