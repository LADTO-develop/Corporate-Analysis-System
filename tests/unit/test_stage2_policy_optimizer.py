"""Tests for Stage 2 policy threshold optimizer scoring."""

from __future__ import annotations

import pandas as pd

from cas.agents.stage2_policy_optimizer import (
    objective_score,
    policy_metrics_from_predictions,
)


def test_policy_metrics_count_fn_rescue_and_tn_overhold_release() -> None:
    frame = pd.DataFrame(
        {
            "is_speculative": [1, 1, 0, 0],
            "pred_label_tuned": [1, 0, 1, 0],
        }
    )
    predicted = pd.Series([True, True, False, False])

    metrics = policy_metrics_from_predictions(frame, predicted_risk=predicted)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 2
    assert metrics["fn_rescued_count"] == 1
    assert metrics["fn_rescue_rate"] == 1.0
    assert metrics["fp_softened_count"] == 1
    assert metrics["tn_overhold_rate"] == 0.0


def test_fn_rescue_objective_rewards_rescued_false_negative() -> None:
    frame = pd.DataFrame(
        {
            "is_speculative": [1, 1, 0, 0],
            "pred_label_tuned": [1, 0, 1, 0],
        }
    )
    baseline = policy_metrics_from_predictions(
        frame,
        predicted_risk=pd.Series([True, False, True, False]),
    )
    rescued = policy_metrics_from_predictions(
        frame,
        predicted_risk=pd.Series([True, True, True, False]),
    )

    assert objective_score(rescued, "fn_rescue") > objective_score(baseline, "fn_rescue")
    assert objective_score(rescued, "strict") > objective_score(baseline, "strict")
