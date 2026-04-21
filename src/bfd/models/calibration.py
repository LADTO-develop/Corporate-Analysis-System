"""Probability calibration for the final ensemble output.

Uses sklearn's ``CalibratedClassifierCV`` wrapper pattern but in
``prefit`` mode so it can be applied *after* the main training loop,
reusing OOF predictions rather than re-training the base model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from bfd.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibratedProbability:
    """Post-hoc calibrator trained on OOF predictions."""

    method: str = "isotonic"
    _isotonic: IsotonicRegression | None = None
    _sigmoid: LogisticRegression | None = None

    def fit(self, p_uncal: np.ndarray, y_true: np.ndarray) -> "CalibratedProbability":
        """Fit the calibrator on uncalibrated probability estimates."""
        p_uncal = np.asarray(p_uncal).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1)

        if self.method == "isotonic":
            self._isotonic = IsotonicRegression(out_of_bounds="clip")
            self._isotonic.fit(p_uncal, y_true)
        elif self.method == "sigmoid":
            self._sigmoid = LogisticRegression(max_iter=1000)
            self._sigmoid.fit(p_uncal.reshape(-1, 1), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        logger.info("calibrator_fit", method=self.method, n=len(p_uncal))
        return self

    def transform(self, p_uncal: np.ndarray) -> np.ndarray:
        """Return calibrated P(y=1)."""
        p_uncal = np.asarray(p_uncal).reshape(-1)
        if self.method == "isotonic":
            assert self._isotonic is not None
            return np.asarray(self._isotonic.predict(p_uncal))
        if self.method == "sigmoid":
            assert self._sigmoid is not None
            return np.asarray(self._sigmoid.predict_proba(p_uncal.reshape(-1, 1))[:, 1])
        raise RuntimeError("Calibrator not fit.")


def optimize_decision_threshold(
    p: np.ndarray,
    y_true: np.ndarray,
    *,
    metric: str = "expected_cost",
    cost_fn: float = 1.0,
    cost_fp: float = 0.2,
) -> float:
    """Sweep thresholds in [0, 1] and return the one minimising expected cost
    (or maximising F1 / Youden's J depending on ``metric``)."""
    p = np.asarray(p).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_score = np.inf if metric == "expected_cost" else -np.inf

    for t in thresholds:
        y_hat = (p >= t).astype(int)
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())

        if metric == "expected_cost":
            score = cost_fn * fn + cost_fp * fp
            if score < best_score:
                best_score = score
                best_t = t
        elif metric == "f1":
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_score:
                best_score = f1
                best_t = t
        elif metric == "youden_j":
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            j = tpr - fpr
            if j > best_score:
                best_score = j
                best_t = t
        else:
            raise ValueError(f"Unknown metric: {metric}")

    logger.info("threshold_optimized", metric=metric, threshold=best_t, score=best_score)
    return float(best_t)
