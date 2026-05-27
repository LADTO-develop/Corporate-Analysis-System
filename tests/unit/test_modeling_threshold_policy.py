from __future__ import annotations

import pandas as pd

from cas.modeling.threshold_policy import (
    build_threshold_policy_experiments,
    build_threshold_policy_report,
)


def test_build_threshold_policy_experiments_returns_tables_and_summary() -> None:
    scores = pd.DataFrame(
        [
            {
                "split": "valid",
                "market": "KOSDAQ",
                "corp_name": "A",
                "fiscal_year": 2023,
                "industry_macro_category": "it_services",
                "is_speculative": 1,
                "prob_speculative": 0.90,
                "threshold": 0.50,
            },
            {
                "split": "valid",
                "market": "KOSPI",
                "corp_name": "B",
                "fiscal_year": 2023,
                "industry_macro_category": "manufacturing",
                "is_speculative": 0,
                "prob_speculative": 0.30,
                "threshold": 0.50,
            },
            {
                "split": "valid",
                "market": "KOSDAQ",
                "corp_name": "C",
                "fiscal_year": 2023,
                "industry_macro_category": "it_services",
                "is_speculative": 1,
                "prob_speculative": 0.70,
                "threshold": 0.50,
            },
            {
                "split": "valid",
                "market": "KOSPI",
                "corp_name": "D",
                "fiscal_year": 2023,
                "industry_macro_category": "manufacturing",
                "is_speculative": 0,
                "prob_speculative": 0.20,
                "threshold": 0.50,
            },
            {
                "split": "test",
                "market": "KOSDAQ",
                "corp_name": "E",
                "fiscal_year": 2024,
                "industry_macro_category": "it_services",
                "is_speculative": 1,
                "prob_speculative": 0.85,
                "threshold": 0.50,
            },
            {
                "split": "test",
                "market": "KOSPI",
                "corp_name": "F",
                "fiscal_year": 2024,
                "industry_macro_category": "manufacturing",
                "is_speculative": 0,
                "prob_speculative": 0.25,
                "threshold": 0.50,
            },
            {
                "split": "test",
                "market": "KOSDAQ",
                "corp_name": "G",
                "fiscal_year": 2024,
                "industry_macro_category": "it_services",
                "is_speculative": 0,
                "prob_speculative": 0.65,
                "threshold": 0.50,
            },
            {
                "split": "test",
                "market": "KOSPI",
                "corp_name": "H",
                "fiscal_year": 2024,
                "industry_macro_category": "manufacturing",
                "is_speculative": 1,
                "prob_speculative": 0.75,
                "threshold": 0.50,
            },
        ]
    )

    metrics, segment_thresholds, focus_segment_metrics, summary = (
        build_threshold_policy_experiments(scores)
    )

    assert "current_artifact_threshold" in set(metrics["policy_name"])
    assert not segment_thresholds.empty
    assert not focus_segment_metrics.empty
    assert summary["current_test"]["threshold_detail"] == "0.500000"


def test_build_threshold_policy_report_renders_markdown() -> None:
    scores = pd.DataFrame(
        [
            {
                "split": split,
                "market": market,
                "corp_name": corp_name,
                "fiscal_year": 2024,
                "industry_macro_category": industry,
                "is_speculative": label,
                "prob_speculative": probability,
                "threshold": 0.50,
            }
            for split, market, corp_name, industry, label, probability in [
                ("valid", "KOSDAQ", "A", "it_services", 1, 0.90),
                ("valid", "KOSPI", "B", "manufacturing", 0, 0.20),
                ("test", "KOSDAQ", "C", "it_services", 1, 0.80),
                ("test", "KOSPI", "D", "manufacturing", 0, 0.30),
            ]
        ]
    )
    metrics, segment_thresholds, focus_segment_metrics, summary = (
        build_threshold_policy_experiments(scores)
    )

    report = build_threshold_policy_report(
        metrics,
        segment_thresholds,
        focus_segment_metrics,
        summary,
    )

    assert report.startswith("# Feature 43 Threshold Policy Experiments")
    assert "threshold_policy_experiment_metrics.csv" in report
