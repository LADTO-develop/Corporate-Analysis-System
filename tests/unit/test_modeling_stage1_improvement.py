from __future__ import annotations

import pandas as pd

from cas.modeling.stage1_improvement import (
    Stage1ImprovementTables,
    build_candidate_feature_set,
    build_stage1_improvement_report,
    build_stage1_improvement_summary,
    select_promotion_candidates,
)


def test_select_promotion_candidates_applies_strict_error_gate() -> None:
    table = pd.DataFrame(
        [
            _selection_row("baseline_43_native", "", 0.0, 0.0, 0.0, 0, 0),
            _selection_row(
                "candidate_good",
                "market_spread_diff, accounts_receivable_ratio, ocf_to_sales_diff",
                0.01,
                0.002,
                0.003,
                0,
                -2,
            ),
            _selection_row(
                "candidate_more_fn", "base_rate_diff, usd_krw", 0.02, 0.003, 0.004, 1, -4
            ),
            _selection_row(
                "candidate_more_fp", "market_spread, usd_krw", 0.02, 0.003, 0.004, -2, 1
            ),
        ]
    )

    candidates = select_promotion_candidates(table)

    assert candidates["variant"].tolist() == ["candidate_good"]
    feature_set = build_candidate_feature_set(candidates)
    assert feature_set["name"] == "feature_46_robust_candidate"
    assert feature_set["added_features"] == [
        "market_spread_diff",
        "accounts_receivable_ratio",
        "ocf_to_sales_diff",
    ]


def test_build_stage1_improvement_report_renders_candidate_decision() -> None:
    tables = Stage1ImprovementTables(
        candidate_pack_metrics=pd.DataFrame(
            [
                _pack_row("baseline_43_native", "", 0.78, 0.76, 74, 30),
                _pack_row("profitability_quality_add_native", "roe", 0.79, 0.75, 80, 32),
            ]
        ),
        rolling_validation_summary=pd.DataFrame(
            [
                _rolling_row("baseline_43_native", "", 0.75, 0.83, 250, 99),
                _rolling_row("candidate_pack", "base_rate", 0.76, 0.84, 240, 95),
            ]
        ),
        rolling_selection_comparison=pd.DataFrame(
            [
                _selection_row("baseline_43_native", "", 0.0, 0.0, 0.0, 0, 0),
                _selection_row(
                    "candidate_good",
                    "market_spread_diff, accounts_receivable_ratio, ocf_to_sales_diff",
                    0.01,
                    0.002,
                    0.003,
                    0,
                    -2,
                ),
            ]
        ),
    )

    summary = build_stage1_improvement_summary(tables)
    report = build_stage1_improvement_report(tables, summary)

    assert summary["candidate_feature_set"]["name"] == "feature_46_robust_candidate"
    assert report.startswith("# Stage 1 XGBoost Improvement Report")
    assert "feature_46_robust_candidate" in report
    assert "market_spread_diff" in report


def _selection_row(
    variant: str,
    features: str,
    rolling_delta: float,
    rolling_pr_delta: float,
    test_delta: float,
    fn_delta: int,
    fp_delta: int,
) -> dict[str, object]:
    return {
        "variant": variant,
        "selection_stage": "baseline" if variant == "baseline_43_native" else "pair",
        "added_features": features,
        "folds": 4,
        "eval_f1_mean": 0.75 + rolling_delta,
        "eval_pr_auc_mean": 0.83 + rolling_pr_delta,
        "eval_f1_min": 0.72,
        "rolling_f1_delta_vs_baseline": rolling_delta,
        "rolling_pr_auc_delta_vs_baseline": rolling_pr_delta,
        "test_f1_at_threshold": 0.76 + test_delta,
        "test_pr_auc": 0.82,
        "test_f1_delta_vs_baseline": test_delta,
        "test_fn_delta_vs_baseline": fn_delta,
        "test_fp_delta_vs_baseline": fp_delta,
    }


def _pack_row(
    variant: str,
    features: str,
    valid_f1: float,
    test_f1: float,
    fp: int,
    fn: int,
) -> dict[str, object]:
    return {
        "variant": variant,
        "added_features": features,
        "added_feature_count": 0 if variant == "baseline_43_native" else 1,
        "valid_f1_at_threshold": valid_f1,
        "valid_pr_auc": valid_f1 + 0.05,
        "test_f1_at_threshold": test_f1,
        "test_pr_auc": test_f1 + 0.05,
        "test_false_positive_at_threshold": fp,
        "test_false_negative_at_threshold": fn,
    }


def _rolling_row(
    variant: str,
    features: str,
    f1: float,
    pr_auc: float,
    fp: int,
    fn: int,
) -> dict[str, object]:
    return {
        "variant": variant,
        "added_features": features,
        "folds": 4,
        "eval_f1_mean": f1,
        "eval_pr_auc_mean": pr_auc,
        "eval_f1_min": f1 - 0.02,
        "total_false_positive": fp,
        "total_false_negative": fn,
    }
