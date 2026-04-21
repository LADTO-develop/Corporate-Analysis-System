"""Distribution-drift checks.

Two tools commonly used in credit-risk monitoring:
    * Population Stability Index (PSI) — bucket-based distributional shift.
    * Two-sample KS test — works for continuous scores.

Both operate on 1D arrays; call per-feature and aggregate upstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI between a reference and current distribution.

    Rule of thumb:
        PSI < 0.10 — no significant change
        0.10 ≤ PSI < 0.25 — modest change, investigate
        PSI ≥ 0.25 — material change, re-train or re-calibrate

    Args:
        reference: reference (training) distribution.
        current: current (production) distribution.
        n_bins: number of quantile bins.
        epsilon: small constant to avoid log(0).
    """
    ref = reference.dropna().astype(float).values
    cur = current.dropna().astype(float).values
    if ref.size == 0 or cur.size == 0:
        return float("nan")

    # Quantile bin edges from the reference distribution
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    # Deduplicate edges (piled-up bins for low-cardinality features)
    edges = np.unique(edges)
    if edges.size < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
    cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_two_sample(reference: pd.Series, current: pd.Series) -> dict[str, float]:
    """Two-sample KS test between reference and current."""
    ref = reference.dropna().astype(float).values
    cur = current.dropna().astype(float).values
    if ref.size == 0 or cur.size == 0:
        return {"statistic": float("nan"), "pvalue": float("nan")}
    result = stats.ks_2samp(ref, cur)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    psi_threshold: float = 0.25,
) -> pd.DataFrame:
    """Per-feature PSI + KS report between two sample frames."""
    cols = feature_cols or [c for c in reference.columns if reference[c].dtype != object]
    rows: list[dict[str, float | str | bool]] = []
    for col in cols:
        if col not in current.columns:
            continue
        psi = population_stability_index(reference[col], current[col])
        ks = ks_two_sample(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "psi": psi,
                "ks_statistic": ks["statistic"],
                "ks_pvalue": ks["pvalue"],
                "drift_detected": psi > psi_threshold,
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
