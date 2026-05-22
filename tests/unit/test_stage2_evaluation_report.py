"""Tests for consolidated Stage 2 evaluation report exports."""

from __future__ import annotations

import pandas as pd
from scripts import export_stage2_evaluation_report as report


def test_normalize_metrics_and_run_summary_compare_stage1_to_stage2() -> None:
    metrics = report.normalize_metrics(
        pd.DataFrame(
            [
                {
                    "run": "pilot",
                    "target": "1차 모델",
                    "n": 2,
                    "tp": 1,
                    "fp": 1,
                    "tn": 0,
                    "fn": 0,
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1": 0.6667,
                    "accuracy": 0.5,
                },
                {
                    "run": "pilot",
                    "target": "2차 위험신호(risk_signal)",
                    "n": 2,
                    "tp": 1,
                    "fp": 0,
                    "tn": 1,
                    "fn": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "accuracy": 1.0,
                },
            ]
        )
    )

    summary = report.summarize_runs(metrics)

    assert list(metrics["TP"]) == [1, 1]
    assert summary.loc[0, "run"] == "pilot"
    assert summary.loc[0, "stage1_f1"] == 0.6667
    assert summary.loc[0, "risk_f1"] == 1.0
    assert summary.loc[0, "risk_f1_delta_vs_stage1"] == 0.3333


def test_latest_batch_metrics_use_explicit_risk_signal_when_available() -> None:
    frame = pd.DataFrame(
        [
            {
                "actual_label_name": "투기등급",
                "model_predicted_label_name": "투자적격",
                "final_committee_label": "보류",
                "committee_risk_signal": True,
            },
            {
                "actual_label_name": "투자적격",
                "model_predicted_label_name": "투기등급",
                "final_committee_label": "보류",
                "committee_risk_signal": False,
            },
        ]
    )

    metrics = report.latest_batch_metrics(frame)
    risk = metrics.loc[metrics["target"].eq("2차 위험신호(risk_signal)")].iloc[0]

    assert risk["TP"] == 1
    assert risk["FP"] == 0
    assert risk["TN"] == 1
    assert risk["FN"] == 0
    assert risk["F1"] == 1.0


def test_latest_batch_metrics_marks_fallback_when_risk_signal_is_missing() -> None:
    frame = pd.DataFrame(
        [
            {
                "actual_label_name": "투기등급",
                "model_predicted_label_name": "투자적격",
                "final_committee_label": "보류",
            }
        ]
    )

    metrics = report.latest_batch_metrics(frame)

    assert "risk_signal 미제공" in metrics.loc[2, "target"]
