from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cas.modeling.calibration_diagnostics import (
    beta_calibration_features,
    calibration_bin_table,
    calibration_error,
    classification_metrics_at_threshold,
    probability_quality_metrics,
)


def test_beta_calibration_features_are_finite_for_boundary_probabilities() -> None:
    probabilities = np.array([0.0, 0.2, 0.8, 1.0])

    features = beta_calibration_features(probabilities)

    assert features.shape == (4, 2)
    assert np.isfinite(features).all()


def test_calibration_bin_table_and_error_use_weighted_absolute_gap() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.2, 0.8, 0.9])

    bins = calibration_bin_table(y_true, probabilities, n_bins=2)
    error = calibration_error(y_true, probabilities, n_bins=2)

    assert bins["rows"].tolist() == [2, 2]
    assert bins["mean_probability"].tolist() == pytest.approx([0.15, 0.85])
    assert bins["actual_positive_rate"].tolist() == pytest.approx([0.0, 1.0])
    assert error["ece"] == pytest.approx(0.15)
    assert error["mce"] == pytest.approx(0.15)


def test_probability_quality_metrics_reward_better_probability_ranking() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    good_probabilities = np.array([0.05, 0.15, 0.85, 0.95])
    weak_probabilities = np.array([0.4, 0.45, 0.55, 0.6])

    good = probability_quality_metrics(y_true, good_probabilities, n_bins=2)
    weak = probability_quality_metrics(y_true, weak_probabilities, n_bins=2)

    assert good["rows"] == 4
    assert good["positive_rate"] == 0.5
    assert float(good["brier"]) < float(weak["brier"])
    assert float(good["ece"]) < float(weak["ece"])
    assert good["pr_auc"] == pytest.approx(1.0)


def test_classification_metrics_at_threshold_counts_confusion_matrix() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.7, 0.6, 0.9])

    metrics = classification_metrics_at_threshold(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics["true_positive"] == 2
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
