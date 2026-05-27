"""Probability calibration and threshold selection helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

PROBABILITY_CLIP_EPSILON = 1e-6
DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR = 0.85
DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 6)

FloatArray = npt.NDArray[np.float64]


def _as_float_array(values: npt.ArrayLike) -> FloatArray:
    return cast(FloatArray, np.asarray(values, dtype=np.float64))


def _probability_to_logit(
    probabilities: npt.ArrayLike,
    *,
    clip_epsilon: float = PROBABILITY_CLIP_EPSILON,
) -> FloatArray:
    clipped = np.clip(_as_float_array(probabilities), clip_epsilon, 1.0 - clip_epsilon)
    return cast(FloatArray, np.log(clipped / (1.0 - clipped)))


def _sigmoid(values: npt.ArrayLike) -> FloatArray:
    return cast(FloatArray, 1.0 / (1.0 + np.exp(-_as_float_array(values))))


def fit_platt_calibration(
    y_valid: pd.Series,
    valid_probabilities: npt.ArrayLike,
) -> dict[str, object]:
    """Fit a sigmoid calibration layer on validation predictions."""
    from sklearn.linear_model import LogisticRegression

    valid_logits = _probability_to_logit(valid_probabilities).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
    calibrator.fit(valid_logits, y_valid.astype(int))
    return {
        "method": "platt_sigmoid",
        "fit_split": "valid",
        "input": "xgboost_probability_logit",
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "clip_epsilon": PROBABILITY_CLIP_EPSILON,
    }


def apply_probability_calibration(
    probabilities: npt.ArrayLike,
    calibration: Mapping[str, object],
) -> FloatArray:
    """Apply saved sigmoid calibration parameters to raw probabilities."""
    if calibration.get("method") != "platt_sigmoid":
        return _as_float_array(probabilities)
    coef = float(cast(float, calibration["coef"]))
    intercept = float(cast(float, calibration["intercept"]))
    logits = _probability_to_logit(probabilities)
    return _sigmoid(intercept + coef * logits)


def build_calibration_summary(
    *,
    calibration: Mapping[str, object],
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_raw_probabilities: npt.ArrayLike,
    test_raw_probabilities: npt.ArrayLike,
    valid_calibrated_probabilities: npt.ArrayLike,
    test_calibrated_probabilities: npt.ArrayLike,
) -> dict[str, object]:
    """Build validation/test diagnostics for raw and calibrated probabilities."""
    from sklearn.metrics import brier_score_loss, log_loss

    def calibration_metrics(
        y_true: pd.Series,
        raw: npt.ArrayLike,
        calibrated: npt.ArrayLike,
    ) -> dict[str, float]:
        raw_array = _as_float_array(raw)
        calibrated_array = _as_float_array(calibrated)
        return {
            "brier_raw": float(brier_score_loss(y_true, raw_array)),
            "brier_calibrated": float(brier_score_loss(y_true, calibrated_array)),
            "logloss_raw": float(
                log_loss(
                    y_true,
                    np.clip(
                        raw_array,
                        PROBABILITY_CLIP_EPSILON,
                        1 - PROBABILITY_CLIP_EPSILON,
                    ),
                )
            ),
            "logloss_calibrated": float(
                log_loss(
                    y_true,
                    np.clip(
                        calibrated_array,
                        PROBABILITY_CLIP_EPSILON,
                        1 - PROBABILITY_CLIP_EPSILON,
                    ),
                )
            ),
        }

    return {
        **dict(calibration),
        "probability_output": "calibrated_probability",
        "valid": calibration_metrics(
            y_valid,
            valid_raw_probabilities,
            valid_calibrated_probabilities,
        ),
        "test": calibration_metrics(
            y_test,
            test_raw_probabilities,
            test_calibrated_probabilities,
        ),
        "note": (
            "XGBoost raw probabilities are transformed with a validation-fitted Platt "
            "sigmoid before being shown as prob_speculative."
        ),
    }


def choose_tuned_threshold(
    y_valid: pd.Series,
    valid_probabilities: npt.ArrayLike,
    *,
    threshold_grid: npt.ArrayLike = DEFAULT_THRESHOLD_GRID,
    recall_floor: float = DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR,
) -> float:
    """Select the highest-precision threshold that satisfies the recall floor."""
    from sklearn.metrics import precision_recall_fscore_support

    probabilities = _as_float_array(valid_probabilities)
    candidates: list[tuple[float, float, float, float]] = []
    for threshold in _as_float_array(threshold_grid):
        predictions = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_valid,
            predictions,
            average="binary",
            zero_division=0,
        )
        candidates.append((float(threshold), float(precision), float(recall), float(f1)))

    recall_candidates = [candidate for candidate in candidates if candidate[2] >= recall_floor]
    if recall_candidates:
        threshold, _, _, _ = max(
            recall_candidates,
            key=lambda candidate: (candidate[1], candidate[3], candidate[0]),
        )
        return float(threshold)

    threshold, _, _, _ = max(candidates, key=lambda candidate: (candidate[3], candidate[2]))
    return float(threshold)


def choose_max_precision_threshold_at_recall(
    y_valid: pd.Series,
    valid_probabilities: npt.ArrayLike,
    recall_floor: float,
    *,
    threshold_grid: npt.ArrayLike = DEFAULT_THRESHOLD_GRID,
) -> float:
    """Select the max-precision threshold among candidates above a recall floor."""
    return choose_tuned_threshold(
        y_valid,
        valid_probabilities,
        threshold_grid=threshold_grid,
        recall_floor=recall_floor,
    )
