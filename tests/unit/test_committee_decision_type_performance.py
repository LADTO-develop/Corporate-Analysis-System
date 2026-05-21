"""Tests for Stage 2 committee decision-type performance exports."""

from __future__ import annotations

import pandas as pd
from scripts import export_stage2_committee_decision_type_performance as perf


def test_overall_signal_metrics_compare_stage1_and_committee_risk_signal() -> None:
    frame = perf.enrich_results(
        pd.DataFrame(
            [
                {
                    "actual_label_name": "투기등급",
                    "model_predicted_label_name": "투자적격",
                    "final_committee_label": "보류",
                    "committee_decision_type_label": "위험 보류",
                    "committee_risk_signal": True,
                },
                {
                    "actual_label_name": "투자적격",
                    "model_predicted_label_name": "투기등급",
                    "final_committee_label": "보류",
                    "committee_decision_type_label": "경계등급 보류",
                    "committee_risk_signal": False,
                },
            ]
        )
    )

    metrics = perf.overall_signal_metrics(frame)
    stage1 = metrics.loc[metrics["metric_scope"].eq("1차 모델")].iloc[0]
    risk_signal = metrics.loc[metrics["metric_scope"].eq("2차 위험신호(risk_signal)")].iloc[0]

    assert stage1["FN"] == 1
    assert stage1["FP"] == 1
    assert risk_signal["TP"] == 1
    assert risk_signal["TN"] == 1
    assert risk_signal["F1"] == 1.0


def test_decision_type_performance_keeps_mitigation_semantics() -> None:
    frame = perf.enrich_results(
        pd.DataFrame(
            [
                {
                    "actual_label_name": "투자적격",
                    "model_predicted_label_name": "투기등급",
                    "final_committee_label": "보류",
                    "committee_decision_type_label": "과민경고 완화 보류",
                    "committee_risk_signal": False,
                    "sample_category": "fp_needing_committee_mitigation",
                },
                {
                    "actual_label_name": "투기등급",
                    "model_predicted_label_name": "투자적격",
                    "final_committee_label": "보류",
                    "committee_decision_type_label": "위험 보류",
                    "committee_risk_signal": True,
                    "sample_category": "fn_caught_by_stage2_review",
                },
            ]
        )
    )

    by_type = perf.decision_type_performance(frame)
    mitigation = by_type.loc[
        by_type["committee_decision_type_label"].eq("과민경고 완화 보류")
    ].iloc[0]
    risk_hold = by_type.loc[by_type["committee_decision_type_label"].eq("위험 보류")].iloc[0]

    assert mitigation["expected_alignment_label"] == "투자적격"
    assert mitigation["expected_alignment_rate"] == 1.0
    assert risk_hold["expected_alignment_label"] == "투기등급"
    assert risk_hold["expected_alignment_rate"] == 1.0
