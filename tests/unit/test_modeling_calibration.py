from __future__ import annotations

import numpy as np
import pandas as pd

from cas.modeling.calibration import (
    apply_probability_calibration,
    build_calibration_summary,
    choose_tuned_threshold,
)


def test_choose_tuned_threshold_prefers_precision_inside_recall_floor() -> None:
    y_valid = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = choose_tuned_threshold(
        y_valid,
        probabilities,
        threshold_grid=np.array([0.2, 0.5, 0.8]),
        recall_floor=1.0,
    )

    assert threshold == 0.5


def test_probability_calibration_applies_saved_platt_parameters() -> None:
    probabilities = np.array([0.2, 0.5, 0.8])
    calibration = {
        "method": "platt_sigmoid",
        "coef": 1.0,
        "intercept": 0.0,
    }

    calibrated = apply_probability_calibration(probabilities, calibration)

    assert np.allclose(calibrated, probabilities)


def test_build_calibration_summary_compares_raw_and_calibrated_scores() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    raw = np.array([0.1, 0.3, 0.7, 0.9])
    calibrated = np.array([0.05, 0.25, 0.75, 0.95])

    summary = build_calibration_summary(
        calibration={"method": "platt_sigmoid", "coef": 1.0, "intercept": 0.0},
        y_valid=y_true,
        y_test=y_true,
        valid_raw_probabilities=raw,
        test_raw_probabilities=raw,
        valid_calibrated_probabilities=calibrated,
        test_calibrated_probabilities=calibrated,
    )

    assert summary["probability_output"] == "calibrated_probability"
    assert summary["valid"]["brier_calibrated"] < summary["valid"]["brier_raw"]
