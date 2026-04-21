"""Classification metrics for the borderline-firm binary task.

Thin wrappers around sklearn with a single entry point,
``compute_all_metrics``, that returns a dict suitable for logging to MLflow
or the model registry.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def ks_statistic(y_true: np.ndarray, p_score: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic: max separation between CDFs of scores for
    the two classes. Widely used in Korean credit scoring."""
    y_true = np.asarray(y_true)
    p_score = np.asarray(p_score)
    pos = np.sort(p_score[y_true == 1])
    neg = np.sort(p_score[y_true == 0])
    if pos.size == 0 or neg.size == 0:
        return 0.0
    # Empirical CDFs evaluated on the merged sorted axis
    all_sorted = np.sort(np.unique(np.concatenate([pos, neg])))
    cdf_pos = np.searchsorted(pos, all_sorted, side="right") / pos.size
    cdf_neg = np.searchsorted(neg, all_sorted, side="right") / neg.size
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def compute_all_metrics(
    y_true: np.ndarray,
    p_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the standard metric bundle for one fold / one artifact.

    Args:
        y_true: 0/1 ground-truth labels.
        p_score: P(y=1) model outputs.
        threshold: threshold used to derive hard predictions for F1 / confusion.
    """
    y_true = np.asarray(y_true)
    p_score = np.asarray(p_score)
    y_hat = (p_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    return {
        "auc": float(roc_auc_score(y_true, p_score)),
        "pr_auc": float(average_precision_score(y_true, p_score)),
        "log_loss": float(log_loss(y_true, p_score, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p_score)),
        "ks": ks_statistic(y_true, p_score),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }
