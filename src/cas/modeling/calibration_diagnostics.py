"""Diagnostics for Stage 1 probability calibration experiments."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

PROBABILITY_EPSILON = 1e-6
DEFAULT_CALIBRATION_BINS = 10

FloatArray = npt.NDArray[np.float64]


def as_float_array(values: npt.ArrayLike) -> FloatArray:
    """Convert array-like values to a float64 numpy array."""
    return cast(FloatArray, np.asarray(values, dtype=np.float64))


def clip_probabilities(
    probabilities: npt.ArrayLike,
    *,
    epsilon: float = PROBABILITY_EPSILON,
) -> FloatArray:
    """Clip probabilities away from exact zero and one."""
    return cast(FloatArray, np.clip(as_float_array(probabilities), epsilon, 1.0 - epsilon))


def probability_to_logit(
    probabilities: npt.ArrayLike,
    *,
    epsilon: float = PROBABILITY_EPSILON,
) -> FloatArray:
    """Convert probabilities to logits after clipping."""
    clipped = clip_probabilities(probabilities, epsilon=epsilon)
    return cast(FloatArray, np.log(clipped / (1.0 - clipped)))


def beta_calibration_features(probabilities: npt.ArrayLike) -> FloatArray:
    """Build beta-calibration logistic features from raw probabilities."""
    clipped = clip_probabilities(probabilities)
    return cast(FloatArray, np.column_stack([np.log(clipped), np.log1p(-clipped)]))


def probability_quality_metrics(
    y_true: pd.Series,
    probabilities: npt.ArrayLike,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, float | int | None]:
    """Compute calibration-focused probability quality metrics."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    labels = y_true.astype(int).reset_index(drop=True)
    probs = pd.Series(as_float_array(probabilities), index=labels.index)
    valid = probs.notna()
    labels = labels.loc[valid]
    probs = probs.loc[valid]
    if labels.empty:
        return _empty_probability_metrics()

    has_two_classes = labels.nunique(dropna=True) == 2
    clipped = clip_probabilities(probs.to_numpy(dtype=np.float64))
    calibration = calibration_error(labels, probs, n_bins=n_bins)
    return {
        "rows": len(labels),
        "positive_rows": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "brier": float(brier_score_loss(labels, probs)),
        "logloss": float(log_loss(labels, clipped)) if has_two_classes else None,
        "pr_auc": float(average_precision_score(labels, probs)) if has_two_classes else None,
        "roc_auc": float(roc_auc_score(labels, probs)) if has_two_classes else None,
        "ece": calibration["ece"],
        "mce": calibration["mce"],
        "mean_probability": float(probs.mean()),
        "calibration_bias": float(probs.mean() - labels.mean()),
        "calibration_slope": calibration_slope_intercept(labels, probs)["slope"],
        "calibration_intercept": calibration_slope_intercept(labels, probs)["intercept"],
    }


def calibration_error(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, float]:
    """Return expected and maximum calibration error over equal-width bins."""
    bins = calibration_bin_table(y_true, probabilities, n_bins=n_bins)
    if bins.empty:
        return {"ece": 0.0, "mce": 0.0}
    weighted_gap = bins["rows"] * bins["calibration_gap"].abs()
    return {
        "ece": float(weighted_gap.sum() / bins["rows"].sum()),
        "mce": float(bins["calibration_gap"].abs().max()),
    }


def calibration_bin_table(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> pd.DataFrame:
    """Build equal-width calibration bins."""
    labels = y_true.astype(int).reset_index(drop=True)
    probs = pd.Series(as_float_array(probabilities), index=labels.index)
    valid = probs.notna()
    labels = labels.loc[valid]
    probs = probs.loc[valid]
    if labels.empty:
        return pd.DataFrame(
            columns=[
                "probability_bin",
                "rows",
                "mean_probability",
                "actual_positive_rate",
                "calibration_gap",
            ]
        )

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    binned = pd.DataFrame({"y_true": labels, "probability": probs})
    binned["probability_bin"] = pd.cut(
        binned["probability"],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    grouped = (
        binned.groupby("probability_bin", observed=False)
        .agg(
            rows=("y_true", "size"),
            mean_probability=("probability", "mean"),
            actual_positive_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["rows"].gt(0)].copy()
    grouped["probability_bin"] = grouped["probability_bin"].astype(str)
    grouped["calibration_gap"] = grouped["mean_probability"] - grouped["actual_positive_rate"]
    return grouped


def calibration_slope_intercept(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float | None]:
    """Fit observed labels on predicted logits to estimate calibration slope."""
    if y_true.nunique(dropna=True) < 2:
        return {"slope": None, "intercept": None}

    from sklearn.linear_model import LogisticRegression

    logits = probability_to_logit(probabilities.to_numpy(dtype=np.float64)).reshape(-1, 1)
    model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
    model.fit(logits, y_true.astype(int))
    return {
        "slope": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }


def classification_metrics_at_threshold(
    y_true: pd.Series,
    probabilities: npt.ArrayLike,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Compute classification metrics for calibrated probabilities."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    labels = y_true.astype(int).reset_index(drop=True)
    probs = pd.Series(as_float_array(probabilities), index=labels.index)
    predictions = probs.ge(threshold).astype(int)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _empty_probability_metrics() -> dict[str, float | int | None]:
    return {
        "rows": 0,
        "positive_rows": 0,
        "positive_rate": None,
        "brier": None,
        "logloss": None,
        "pr_auc": None,
        "roc_auc": None,
        "ece": None,
        "mce": None,
        "mean_probability": None,
        "calibration_bias": None,
        "calibration_slope": None,
        "calibration_intercept": None,
    }
