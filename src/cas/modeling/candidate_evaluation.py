"""Evaluation helpers for promoted Stage 1 feature candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

KEY_COLUMNS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]

GRADE_RANK = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22,
}


@dataclass(frozen=True)
class SegmentSpec:
    """A named segment to evaluate for candidate promotion checks."""

    dimension: str
    segment: str
    column: str | None = None
    value: object | None = None


TARGET_SEGMENTS = [
    SegmentSpec("overall", "all"),
    SegmentSpec("market", "KOSDAQ", "market", "KOSDAQ"),
    SegmentSpec(
        "industry_macro_category",
        "manufacturing",
        "industry_macro_category",
        "manufacturing",
    ),
    SegmentSpec(
        "rating_boundary",
        "BBB-/BB+",
        "is_exact_boundary_bbb_minus_bb_plus",
        True,
    ),
    SegmentSpec("credit_rating", "BBB-", "credit_rating", "BBB-"),
    SegmentSpec("credit_rating", "BB+", "credit_rating", "BB+"),
]

MetricValue = float | int | str | None


def normalize_stock_code(value: object) -> str:
    """Normalize Korean stock codes to zero-padded six-character strings."""
    text = str(value or "").replace("\ufeff", "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text.upper()


def normalize_stock_code_column(series: pd.Series) -> pd.Series:
    """Normalize a pandas stock-code column."""
    return series.map(normalize_stock_code)


def normalize_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize stock code and year columns used for evaluation joins."""
    output = frame.copy()
    if "stock_code" in output.columns:
        output["stock_code"] = normalize_stock_code_column(output["stock_code"])
    for column in ["fiscal_year", "eval_year", "label_eval_year", "credit_rating_rank"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def read_rating_labels(path: str | pd.io.common.FilePath) -> pd.DataFrame:
    """Read target labels and keep only columns used by diagnostics."""
    labels = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    labels = normalize_key_columns(labels)
    if "credit_rating" in labels.columns:
        labels["credit_rating"] = labels["credit_rating"].astype("string").str.strip()
    if "credit_rating_rank" not in labels.columns and "credit_rating" in labels.columns:
        labels["credit_rating_rank"] = labels["credit_rating"].map(GRADE_RANK)
    keep_columns = [
        column
        for column in [
            *KEY_COLUMNS,
            "is_speculative",
            "credit_rating",
            "credit_rating_rank",
            "rating_agency",
            "rating_agency_group",
            "rating_agency_code",
            "rating_target",
            "rating_date",
            "current_outlook",
            "selection_scope",
            "selection_rule",
        ]
        if column in labels.columns
    ]
    return labels.loc[:, keep_columns].copy()


def attach_rating_labels(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    include_target: bool = False,
) -> pd.DataFrame:
    """Attach rating label metadata to scored rows."""
    scores_normalized = normalize_key_columns(scores)
    labels_normalized = normalize_key_columns(labels)
    label_columns = [
        column
        for column in labels_normalized.columns
        if column in KEY_COLUMNS or include_target or column != "is_speculative"
    ]
    joined = scores_normalized.merge(
        labels_normalized.loc[:, label_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    return add_rating_boundary_columns(joined)


def add_rating_boundary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add BBB-/BB+ boundary flags used by promotion diagnostics."""
    output = frame.copy()
    if "credit_rating" not in output.columns:
        return output

    rating = output["credit_rating"].astype("string").str.strip()
    rank = (
        pd.to_numeric(output.get("credit_rating_rank"), errors="coerce")
        if "credit_rating_rank" in output.columns
        else rating.map(GRADE_RANK)
    )
    output["credit_rating_rank"] = rank
    output["is_exact_boundary_bbb_minus_bb_plus"] = rating.isin(["BBB-", "BB+"])
    output["is_near_boundary_bbb_bb"] = rank.between(8, 13, inclusive="both")
    output["rating_boundary_group"] = np.select(
        [
            rank.le(7),
            rank.between(8, 10, inclusive="both"),
            rank.between(11, 13, inclusive="both"),
            rank.ge(14),
        ],
        [
            "upper_investment_A_or_above",
            "near_investment_BBB_plus_to_BBB_minus",
            "near_speculative_BB_plus_to_BB_minus",
            "deep_speculative_B_plus_or_lower",
        ],
        default="missing_rating",
    )
    output["market_x_rating_boundary"] = (
        output["market"].astype(str) + " / " + output["rating_boundary_group"].astype(str)
    )
    return output


def classification_summary(
    frame: pd.DataFrame,
    *,
    label_column: str = "is_speculative",
    probability_column: str = "prob_speculative",
    prediction_column: str = "pred_label_tuned",
) -> dict[str, MetricValue]:
    """Build probability and classification metrics for one scored frame."""
    if frame.empty:
        return _empty_metric_summary()

    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = pd.to_numeric(frame[label_column], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(frame[prediction_column], errors="coerce").fillna(0).astype(int)
    probabilities = pd.to_numeric(frame[probability_column], errors="coerce")

    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())

    probability_metrics = _safe_probability_metrics(y_true, probabilities)
    return {
        "rows": len(frame),
        "positive_rows": positives,
        "negative_rows": negatives,
        "positive_rate": float(y_true.mean()) if len(frame) else None,
        "pr_auc": probability_metrics["pr_auc"],
        "roc_auc": probability_metrics["roc_auc"],
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "fp_rate_among_negatives": false_positive / negatives if negatives else None,
        "fn_rate_among_positives": false_negative / positives if positives else None,
    }


def build_target_segment_metrics(
    frame: pd.DataFrame,
    *,
    model_name: str,
    evaluation_scope: str,
    label_column: str = "is_speculative",
    probability_column: str = "prob_speculative",
    prediction_column: str = "pred_label_tuned",
    segments: list[SegmentSpec] | None = None,
) -> pd.DataFrame:
    """Evaluate the fixed promotion-check target segments."""
    specs = segments or TARGET_SEGMENTS
    rows: list[dict[str, MetricValue]] = []
    for spec in specs:
        segment_frame = _select_segment(frame, spec)
        rows.append(
            {
                "model_name": model_name,
                "evaluation_scope": evaluation_scope,
                "dimension": spec.dimension,
                "segment": spec.segment,
                **classification_summary(
                    segment_frame,
                    label_column=label_column,
                    probability_column=probability_column,
                    prediction_column=prediction_column,
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_segment_metrics(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    baseline_name: str,
    candidate_name: str,
) -> pd.DataFrame:
    """Join baseline and candidate segment metrics and add candidate deltas."""
    join_columns = ["evaluation_scope", "dimension", "segment"]
    value_columns = [
        "rows",
        "positive_rows",
        "negative_rows",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]
    left = baseline.loc[:, [*join_columns, *value_columns]].rename(
        columns={column: f"{baseline_name}_{column}" for column in value_columns}
    )
    right = candidate.loc[:, [*join_columns, *value_columns]].rename(
        columns={column: f"{candidate_name}_{column}" for column in value_columns}
    )
    merged = left.merge(right, on=join_columns, how="outer")
    for metric in ["pr_auc", "roc_auc", "precision", "recall", "f1"]:
        merged[f"delta_{metric}"] = pd.to_numeric(
            merged[f"{candidate_name}_{metric}"], errors="coerce"
        ) - pd.to_numeric(merged[f"{baseline_name}_{metric}"], errors="coerce")
    for metric in ["true_positive", "false_positive", "false_negative", "true_negative"]:
        merged[f"delta_{metric}"] = pd.to_numeric(
            merged[f"{candidate_name}_{metric}"], errors="coerce"
        ) - pd.to_numeric(merged[f"{baseline_name}_{metric}"], errors="coerce")
    return merged


def feature_readiness_summary(
    frame: pd.DataFrame,
    *,
    candidate_columns: list[str],
) -> list[dict[str, MetricValue]]:
    """Summarize candidate-feature availability in an inference frame."""
    rows: list[dict[str, MetricValue]] = []
    total = len(frame)
    for column in candidate_columns:
        if column in frame.columns:
            present = pd.to_numeric(frame[column], errors="coerce").notna()
            available = int(present.sum())
        else:
            available = 0
        rows.append(
            {
                "feature": column,
                "rows": total,
                "available_rows": available,
                "missing_rows": total - available,
                "available_rate": available / total if total else None,
            }
        )
    return rows


def _select_segment(frame: pd.DataFrame, spec: SegmentSpec) -> pd.DataFrame:
    if spec.column is None:
        return frame
    if spec.column not in frame.columns:
        return frame.iloc[0:0]
    series = frame[spec.column]
    if isinstance(spec.value, bool):
        mask = series.fillna(False).astype(bool).eq(spec.value)
    else:
        mask = series.astype(str).eq(str(spec.value))
    return frame.loc[mask].copy()


def _safe_probability_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
) -> dict[str, float | None]:
    if y_true.nunique(dropna=True) < 2 or probabilities.notna().sum() < 2:
        return {"pr_auc": None, "roc_auc": None}

    from sklearn.metrics import average_precision_score, roc_auc_score

    valid = probabilities.notna()
    if y_true.loc[valid].nunique(dropna=True) < 2:
        return {"pr_auc": None, "roc_auc": None}
    return {
        "pr_auc": float(average_precision_score(y_true.loc[valid], probabilities.loc[valid])),
        "roc_auc": float(roc_auc_score(y_true.loc[valid], probabilities.loc[valid])),
    }


def _empty_metric_summary() -> dict[str, MetricValue]:
    return cast(
        dict[str, MetricValue],
        {
            "rows": 0,
            "positive_rows": 0,
            "negative_rows": 0,
            "positive_rate": None,
            "pr_auc": None,
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "fp_rate_among_negatives": None,
            "fn_rate_among_positives": None,
        },
    )
