from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cas.modeling.stage1_xgboost import (
    build_monotonic_constraints,
    build_time_decay_weights,
    choose_policy_threshold,
    classification_metrics,
    read_stage1_feature_columns,
    read_stage1_master,
)


def test_choose_policy_threshold_prefers_precision_inside_recall_floor() -> None:
    y_policy = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold, metrics = choose_policy_threshold(
        y_policy,
        probabilities,
        threshold_grid=np.array([0.2, 0.5, 0.8]),
        recall_floor=1.0,
    )

    assert threshold == 0.5
    assert metrics["threshold_selection_rule"] == "policy_max_precision_with_recall_ge_1.00"
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_classification_metrics_returns_confusion_counts() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    predictions = np.array([0, 1, 0, 1])

    metrics = classification_metrics(y_true, predictions)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1


def test_build_time_decay_weights_normalizes_recent_years_higher() -> None:
    frame = pd.DataFrame({"fiscal_year": [2019, 2020, 2021]})

    weights = build_time_decay_weights(frame, reference_year=2021, half_life_years=1.0)

    assert np.isclose(weights.mean(), 1.0)
    assert weights[0] < weights[1] < weights[2]


def test_build_monotonic_constraints_aligns_to_feature_columns() -> None:
    constraints = build_monotonic_constraints(
        ["debt_ratio", "cash_ratio", "market_KOSPI"],
        {"debt_ratio": 1, "cash_ratio": -1},
    )

    assert constraints == (1, -1, 0)


def test_read_stage1_inputs_validate_feature_spec(tmp_path) -> None:
    master_path = tmp_path / "master.csv"
    spec_path = tmp_path / "features.json"
    pd.DataFrame(
        {
            "market": ["KOSPI"],
            "stock_code": ["5930"],
            "corp_name": ["삼성전자"],
            "fiscal_year": [2023],
            "feature_a": [1.0],
            "is_speculative": [0],
        }
    ).to_csv(master_path, index=False, encoding="utf-8-sig")
    spec_path.write_text(json.dumps({"model_features": ["feature_a"]}), encoding="utf-8")

    master = read_stage1_master(
        master_path,
        duplicate_keys=["market", "stock_code", "corp_name", "fiscal_year"],
    )
    columns = read_stage1_feature_columns(spec_path, master)

    assert master["stock_code"].iloc[0] == "005930"
    assert columns == ["feature_a"]
