from __future__ import annotations

import pandas as pd

from cas.modeling.candidate_evaluation import (
    SegmentSpec,
    add_rating_boundary_columns,
    build_target_segment_metrics,
    compare_segment_metrics,
    feature_readiness_summary,
    normalize_stock_code,
)


def test_add_rating_boundary_columns_flags_bbb_minus_bb_plus() -> None:
    frame = pd.DataFrame(
        {
            "market": ["KOSDAQ", "KOSPI", "KOSDAQ"],
            "credit_rating": ["BBB-", "BB+", "A-"],
            "credit_rating_rank": [10, 11, 7],
        }
    )

    output = add_rating_boundary_columns(frame)

    assert output["is_exact_boundary_bbb_minus_bb_plus"].tolist() == [True, True, False]
    assert output["is_near_boundary_bbb_bb"].tolist() == [True, True, False]
    assert output["rating_boundary_group"].tolist() == [
        "near_investment_BBB_plus_to_BBB_minus",
        "near_speculative_BB_plus_to_BB_minus",
        "upper_investment_A_or_above",
    ]


def test_build_target_segment_metrics_uses_requested_segments() -> None:
    frame = pd.DataFrame(
        {
            "market": ["KOSDAQ", "KOSDAQ", "KOSPI", "KOSPI"],
            "is_speculative": [1, 0, 1, 0],
            "prob_speculative": [0.9, 0.8, 0.2, 0.1],
            "pred_label_tuned": [1, 1, 0, 0],
        }
    )

    metrics = build_target_segment_metrics(
        frame,
        model_name="candidate",
        evaluation_scope="test",
        segments=[
            SegmentSpec("overall", "all"),
            SegmentSpec("market", "KOSDAQ", "market", "KOSDAQ"),
        ],
    )

    overall = metrics.loc[metrics["dimension"].eq("overall")].iloc[0]
    kosdaq = metrics.loc[metrics["segment"].eq("KOSDAQ")].iloc[0]
    assert overall["rows"] == 4
    assert overall["false_positive"] == 1
    assert overall["false_negative"] == 1
    assert kosdaq["rows"] == 2
    assert kosdaq["precision"] == 0.5
    assert kosdaq["recall"] == 1.0


def test_compare_segment_metrics_adds_candidate_deltas() -> None:
    baseline = pd.DataFrame(
        {
            "evaluation_scope": ["test"],
            "dimension": ["overall"],
            "segment": ["all"],
            "rows": [10],
            "positive_rows": [4],
            "negative_rows": [6],
            "pr_auc": [0.7],
            "roc_auc": [0.8],
            "precision": [0.6],
            "recall": [0.75],
            "f1": [0.6667],
            "true_positive": [3],
            "false_positive": [2],
            "false_negative": [1],
            "true_negative": [4],
        }
    )
    candidate = baseline.assign(f1=[0.75], false_positive=[1], false_negative=[1])

    comparison = compare_segment_metrics(
        baseline,
        candidate,
        baseline_name="baseline",
        candidate_name="candidate",
    )

    row = comparison.iloc[0]
    assert row["delta_f1"] == 0.08330000000000004
    assert row["delta_false_positive"] == -1
    assert row["delta_false_negative"] == 0


def test_feature_readiness_summary_handles_missing_columns() -> None:
    frame = pd.DataFrame({"a": [1.0, None, 3.0]})

    readiness = feature_readiness_summary(frame, candidate_columns=["a", "b"])

    assert readiness == [
        {
            "feature": "a",
            "rows": 3,
            "available_rows": 2,
            "missing_rows": 1,
            "available_rate": 2 / 3,
        },
        {
            "feature": "b",
            "rows": 3,
            "available_rows": 0,
            "missing_rows": 3,
            "available_rate": 0.0,
        },
    ]


def test_normalize_stock_code_handles_numeric_and_alpha_codes() -> None:
    assert normalize_stock_code("250") == "000250"
    assert normalize_stock_code("250.0") == "000250"
    assert normalize_stock_code("0007C0") == "0007C0"
