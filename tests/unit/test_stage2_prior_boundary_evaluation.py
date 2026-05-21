"""Tests for prior BBB-/BB+ Stage 2 boundary diagnostics."""

from __future__ import annotations

import pandas as pd
from scripts import export_stage2_prior_boundary_evaluation as boundary


def test_overall_metrics_show_stage2_catches_prior_boundary_fn() -> None:
    frame = boundary.enrich_boundary_frame(
        pd.DataFrame(
            [
                {
                    "is_speculative": 1,
                    "predicted_label": 0,
                    "stage2_review_trigger": True,
                    "stage2_secondary_trigger": True,
                    "stage2_overwarning_filter_candidate": False,
                },
                {
                    "is_speculative": 0,
                    "predicted_label": 1,
                    "stage2_review_trigger": True,
                    "stage2_secondary_trigger": False,
                    "stage2_overwarning_filter_candidate": True,
                },
            ]
        )
    )

    metrics = boundary.overall_metrics(frame)
    stage1 = metrics.loc[metrics["metric_scope"].eq("1차 모델 위험 판단")].iloc[0]
    review = metrics.loc[metrics["metric_scope"].eq("2차 조심검토 게이트")].iloc[0]
    risk_proxy = metrics.loc[metrics["metric_scope"].eq("2차 위험신호 근사")].iloc[0]

    assert stage1["FN"] == 1
    assert stage1["FP"] == 1
    assert review["Recall"] == 1.0
    assert risk_proxy["TP"] == 1
    assert risk_proxy["FP"] == 0


def test_boundary_group_summary_tracks_fn_catch_and_fp_soften() -> None:
    frame = boundary.enrich_boundary_frame(
        pd.DataFrame(
            [
                {
                    "split": "test",
                    "prior_credit_rating": "BB+",
                    "is_speculative": 1,
                    "predicted_label": 0,
                    "stage2_review_trigger": True,
                    "stage2_secondary_trigger": True,
                    "stage2_overwarning_filter_candidate": False,
                },
                {
                    "split": "test",
                    "prior_credit_rating": "BBB-",
                    "is_speculative": 0,
                    "predicted_label": 1,
                    "stage2_review_trigger": True,
                    "stage2_secondary_trigger": False,
                    "stage2_overwarning_filter_candidate": True,
                },
            ]
        )
    )

    summary = boundary.boundary_group_summary(frame)
    bb_plus = summary.loc[summary["prior_credit_rating"].eq("BB+")].iloc[0]
    bbb_minus = summary.loc[summary["prior_credit_rating"].eq("BBB-")].iloc[0]

    assert bb_plus["stage1_FN"] == 1
    assert bb_plus["stage2_FN_caught_by_review"] == 1
    assert bbb_minus["stage1_FP"] == 1
    assert bbb_minus["stage2_FP_soften_candidate"] == 1
