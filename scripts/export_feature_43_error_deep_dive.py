from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_SCORES_PATH = (
    ROOT / "data" / "outputs" / "dashboard" / "feature_43_mvp" / "prediction_scores.csv"
)
FEATURE_MASTER_PATH = ROOT / "data" / "input" / "credit_43_features" / "feature_43_master.csv"
FEATURE_SPEC_PATH = ROOT / "data" / "input" / "credit_43_features" / "feature_43_list.json"
TARGET_PROCESSED_PATH = ROOT / "data" / "evaluation" / "target_label_reference.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_43_xgboost" / "diagnostics"

KEY_COLUMNS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
BASE_SEGMENT_DIMENSIONS = [
    "market",
    "industry_macro_category",
    "firm_size_group",
    "fiscal_year",
    "market_x_industry",
    "market_x_firm_size",
    "industry_x_firm_size",
]
RATING_SEGMENT_DIMENSIONS = [
    "credit_rating",
    "rating_boundary_group",
    "is_exact_boundary_bbb_minus_bb_plus",
    "is_near_boundary_bbb_bb",
    "rating_agency_group",
    "selection_scope",
    "market_x_rating_boundary",
]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deeper official 43-feature error diagnosis focused on FP/FN concentration, "
            "weak segments, and feature profiles."
        )
    )
    parser.add_argument("--prediction-scores", type=Path, default=PREDICTION_SCORES_PATH)
    parser.add_argument("--feature-master", type=Path, default=FEATURE_MASTER_PATH)
    parser.add_argument("--feature-spec", type=Path, default=FEATURE_SPEC_PATH)
    parser.add_argument(
        "--target-processed",
        type=Path,
        default=TARGET_PROCESSED_PATH,
        help=(
            "Optional diagnostic target-label reference CSV with credit_rating. "
            "Used only for diagnostics, not for model training."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )


def read_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    required = {
        "split",
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "firm_size_group",
        "industry_macro_category",
        "is_speculative",
        "prob_speculative",
        "threshold",
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise KeyError(f"prediction_scores.csv is missing columns: {missing}")
    scores = scores.copy()
    scores["stock_code"] = normalize_stock_code(scores["stock_code"])
    for column in ["fiscal_year", "eval_year", "is_speculative", "prob_speculative", "threshold"]:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    scores["threshold"] = scores["threshold"].fillna(scores["threshold"].dropna().iloc[0])
    scores["predicted"] = (scores["prob_speculative"] >= scores["threshold"]).astype(int)
    scores["prediction_result"] = np.select(
        [
            (scores["is_speculative"] == 1) & (scores["predicted"] == 1),
            (scores["is_speculative"] == 0) & (scores["predicted"] == 0),
            (scores["is_speculative"] == 0) & (scores["predicted"] == 1),
            (scores["is_speculative"] == 1) & (scores["predicted"] == 0),
        ],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown",
    )
    scores["error_type"] = scores["prediction_result"].where(
        scores["prediction_result"].isin(["false_positive", "false_negative"]),
        "correct",
    )
    scores["probability_margin"] = scores["prob_speculative"] - scores["threshold"]
    scores["distance_from_threshold"] = scores["probability_margin"].abs()
    scores["market_x_industry"] = (
        scores["market"].astype(str) + " / " + scores["industry_macro_category"].astype(str)
    )
    scores["market_x_firm_size"] = (
        scores["market"].astype(str) + " / " + scores["firm_size_group"].astype(str)
    )
    scores["industry_x_firm_size"] = (
        scores["industry_macro_category"].astype(str) + " / " + scores["firm_size_group"].astype(str)
    )
    return scores


def read_master(path: Path) -> pd.DataFrame:
    master = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    master["stock_code"] = normalize_stock_code(master["stock_code"])
    for column in ["fiscal_year", "eval_year"]:
        master[column] = pd.to_numeric(master[column], errors="coerce")
    return master


def read_target_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    required = {*KEY_COLUMNS, "credit_rating"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise KeyError(f"Target labels file is missing columns: {missing}")

    labels = labels.copy()
    labels["stock_code"] = normalize_stock_code(labels["stock_code"])
    for column in ["fiscal_year", "eval_year"]:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")
    labels["credit_rating"] = labels["credit_rating"].astype("string").str.strip()

    if "rating_rank" in labels.columns and "credit_rating_rank" not in labels.columns:
        labels = labels.rename(columns={"rating_rank": "credit_rating_rank"})
    if "credit_rating_rank" not in labels.columns:
        labels["credit_rating_rank"] = labels["credit_rating"].map(GRADE_RANK)
    labels["credit_rating_rank"] = pd.to_numeric(labels["credit_rating_rank"], errors="coerce")

    optional_columns = [
        "credit_rating_rank",
        "rating_agency",
        "rating_agency_group",
        "selection_scope",
        "selection_rule",
        "evaluation_date",
        "candidate_count_in_year",
        "big3_candidate_count_in_year",
        "other_domestic_candidate_count_in_year",
        "foreign_candidate_count_in_year",
    ]
    keep_columns = [*KEY_COLUMNS, "credit_rating", *[c for c in optional_columns if c in labels.columns]]
    output = labels.loc[:, keep_columns].copy()
    duplicates = output.duplicated(KEY_COLUMNS).sum()
    if duplicates:
        raise ValueError(f"Target labels file has duplicate key rows: {duplicates}")
    return output


def add_rating_boundary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "credit_rating" not in output.columns:
        return output

    rank = pd.to_numeric(output.get("credit_rating_rank"), errors="coerce")
    rating = output["credit_rating"].astype("string").str.strip()
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


def attach_target_labels(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return scores
    merged = scores.merge(labels, on=KEY_COLUMNS, how="left", validate="many_to_one")
    return add_rating_boundary_columns(merged)


def read_feature_columns(feature_spec_path: Path, master: pd.DataFrame) -> list[str]:
    spec = json.loads(feature_spec_path.read_text(encoding="utf-8"))
    categorical = set(spec.get("categorical_one_hot_columns", []))
    source_features = [str(value) for value in spec.get("selected_source_features", [])]
    columns = [
        column
        for column in source_features
        if column not in categorical and column in master.columns and column not in KEY_COLUMNS
    ]
    return columns


def safe_probability_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame["is_speculative"].nunique() < 2:
        return {"pr_auc": None, "roc_auc": None}
    return {
        "pr_auc": float(average_precision_score(frame["is_speculative"], frame["prob_speculative"])),
        "roc_auc": float(roc_auc_score(frame["is_speculative"], frame["prob_speculative"])),
    }


def classification_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    y_true = frame["is_speculative"].astype(int)
    y_pred = frame["predicted"].astype(int)
    true_positive = int(((y_true == 1) & (y_pred == 1)).sum())
    true_negative = int(((y_true == 0) & (y_pred == 0)).sum())
    false_positive = int(((y_true == 0) & (y_pred == 1)).sum())
    false_negative = int(((y_true == 1) & (y_pred == 0)).sum())
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    probability_metrics = safe_probability_metrics(frame)
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


def build_segment_metrics(test: pd.DataFrame) -> pd.DataFrame:
    rows = [{"dimension": "overall", "segment": "test_all", **classification_summary(test)}]
    segment_dimensions = [
        dimension
        for dimension in [*BASE_SEGMENT_DIMENSIONS, *RATING_SEGMENT_DIMENSIONS]
        if dimension in test.columns
    ]
    for dimension in segment_dimensions:
        for segment, segment_frame in test.groupby(dimension, dropna=False):
            rows.append(
                {
                    "dimension": dimension,
                    "segment": segment,
                    **classification_summary(segment_frame),
                }
            )
    return pd.DataFrame(rows)


def build_error_concentration(segment_metrics: pd.DataFrame) -> pd.DataFrame:
    overall = segment_metrics.loc[segment_metrics["dimension"].eq("overall")].iloc[0]
    total_fp = int(overall["false_positive"])
    total_fn = int(overall["false_negative"])
    concentration = segment_metrics.loc[~segment_metrics["dimension"].eq("overall")].copy()
    concentration["fp_share_of_total_fp"] = (
        concentration["false_positive"] / total_fp if total_fp else 0.0
    )
    concentration["fn_share_of_total_fn"] = (
        concentration["false_negative"] / total_fn if total_fn else 0.0
    )
    concentration["error_count"] = concentration["false_positive"] + concentration["false_negative"]
    concentration["error_share"] = concentration["error_count"] / (
        total_fp + total_fn if total_fp + total_fn else 1
    )
    return concentration.sort_values(
        ["error_count", "false_negative", "false_positive"],
        ascending=False,
    )


def build_error_cases(test: pd.DataFrame) -> pd.DataFrame:
    error_cases = test.loc[test["prediction_result"].isin(["false_positive", "false_negative"])].copy()
    error_cases["confidence_error_score"] = np.where(
        error_cases["prediction_result"].eq("false_positive"),
        error_cases["prob_speculative"],
        1.0 - error_cases["prob_speculative"],
    )
    error_cases["actual_label_name"] = np.where(
        error_cases["is_speculative"].eq(1),
        "투기등급",
        "투자적격",
    )
    error_cases["predicted_label_name"] = np.where(
        error_cases["predicted"].eq(1),
        "투기등급",
        "투자적격",
    )
    columns = [
        "prediction_result",
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "credit_rating",
        "credit_rating_rank",
        "rating_boundary_group",
        "rating_agency_group",
        "selection_scope",
        "industry_macro_category",
        "firm_size_group",
        "prob_speculative",
        "threshold",
        "distance_from_threshold",
        "actual_label_name",
        "predicted_label_name",
        "confidence_error_score",
    ]
    return error_cases.sort_values(
        ["prediction_result", "confidence_error_score"],
        ascending=[True, False],
    ).loc[:, [column for column in columns if column in error_cases.columns]]


def feature_profile(
    merged: pd.DataFrame,
    feature_columns: list[str],
    *,
    error_result: str,
    reference_result: str,
) -> pd.DataFrame:
    error_frame = merged.loc[merged["prediction_result"].eq(error_result)]
    reference_frame = merged.loc[merged["prediction_result"].eq(reference_result)]
    rows = []
    for feature in feature_columns:
        error_values = pd.to_numeric(error_frame[feature], errors="coerce")
        reference_values = pd.to_numeric(reference_frame[feature], errors="coerce")
        all_values = pd.to_numeric(merged[feature], errors="coerce")
        iqr = all_values.quantile(0.75) - all_values.quantile(0.25)
        std = all_values.std()
        denominator = iqr if pd.notna(iqr) and iqr > 0 else std
        median_error = error_values.median()
        median_reference = reference_values.median()
        raw_delta = median_error - median_reference
        standardized_delta = raw_delta / denominator if pd.notna(denominator) and denominator else np.nan
        rows.append(
            {
                "comparison": f"{error_result}_vs_{reference_result}",
                "feature": feature,
                "error_median": median_error,
                "reference_median": median_reference,
                "median_delta": raw_delta,
                "standardized_median_delta": standardized_delta,
                "abs_standardized_median_delta": abs(standardized_delta)
                if pd.notna(standardized_delta)
                else np.nan,
                "error_mean": error_values.mean(),
                "reference_mean": reference_values.mean(),
                "error_missing_rate": float(error_values.isna().mean()),
                "reference_missing_rate": float(reference_values.isna().mean()),
                "error_rows": len(error_frame),
                "reference_rows": len(reference_frame),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_standardized_median_delta", ascending=False)


def build_feature_profile(merged: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    profiles = [
        feature_profile(
            merged,
            feature_columns,
            error_result="false_negative",
            reference_result="true_positive",
        ),
        feature_profile(
            merged,
            feature_columns,
            error_result="false_positive",
            reference_result="true_negative",
        ),
    ]
    return pd.concat(profiles, ignore_index=True)


def build_threshold_distance_summary(test: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.02, 0.05, 0.10, 0.20, np.inf]
    labels = ["<=0.02", "0.02-0.05", "0.05-0.10", "0.10-0.20", ">0.20"]
    frame = test.copy()
    frame["distance_bucket"] = pd.cut(
        frame["distance_from_threshold"],
        bins=bins,
        labels=labels,
        right=True,
    )
    rows = []
    for bucket, bucket_frame in frame.groupby("distance_bucket", observed=False):
        rows.append(
            {
                "distance_bucket": str(bucket),
                **classification_summary(bucket_frame),
            }
        )
    return pd.DataFrame(rows)


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}%"


def format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]], max_rows: int = 20) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    separator = "| " + " | ".join("---" if kind == "text" else "---:" for _, _, kind in columns) + " |"
    rows = []
    for row in frame.head(max_rows).to_dict(orient="records"):
        values = []
        for _, column, kind in columns:
            value = row.get(column)
            if kind == "metric":
                values.append(format_metric(value))
            elif kind == "pct":
                values.append(format_pct(value))
            elif kind == "int":
                values.append(format_int(value))
            else:
                values.append(str(value) if value is not None else "")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def segment_top(
    concentration: pd.DataFrame,
    *,
    dimension: str,
    sort_column: str,
    min_denominator_column: str,
    min_denominator: int = 5,
) -> pd.DataFrame:
    frame = concentration.loc[
        (concentration["dimension"].eq(dimension))
        & (concentration[min_denominator_column] >= min_denominator)
    ].copy()
    return frame.sort_values(sort_column, ascending=False)


def build_report(
    *,
    segment_metrics: pd.DataFrame,
    concentration: pd.DataFrame,
    feature_profiles: pd.DataFrame,
    threshold_distance: pd.DataFrame,
    error_cases: pd.DataFrame,
    grade_columns_available: bool,
    target_processed_path: Path,
    matched_rating_rows: int,
) -> str:
    overall = segment_metrics.loc[segment_metrics["dimension"].eq("overall")].iloc[0]
    market = segment_metrics.loc[segment_metrics["dimension"].eq("market")].copy()
    industry_recall = segment_top(
        concentration,
        dimension="industry_macro_category",
        sort_column="fn_rate_among_positives",
        min_denominator_column="positive_rows",
    )
    industry_fp = segment_top(
        concentration,
        dimension="industry_macro_category",
        sort_column="fp_share_of_total_fp",
        min_denominator_column="negative_rows",
    )
    firm_size_recall = segment_top(
        concentration,
        dimension="firm_size_group",
        sort_column="fn_rate_among_positives",
        min_denominator_column="positive_rows",
    )
    year_recall = segment_top(
        concentration,
        dimension="fiscal_year",
        sort_column="fn_rate_among_positives",
        min_denominator_column="positive_rows",
    )
    cross_errors = concentration.loc[
        concentration["dimension"].isin(["market_x_industry", "market_x_firm_size"])
    ].sort_values(["error_count", "false_negative"], ascending=False)
    fn_profile = feature_profiles.loc[
        feature_profiles["comparison"].eq("false_negative_vs_true_positive")
    ]
    fp_profile = feature_profiles.loc[
        feature_profiles["comparison"].eq("false_positive_vs_true_negative")
    ]
    high_confidence_fn = error_cases.loc[error_cases["prediction_result"].eq("false_negative")]
    high_confidence_fp = error_cases.loc[error_cases["prediction_result"].eq("false_positive")]
    rating_boundary = segment_metrics.loc[
        segment_metrics["dimension"].eq("rating_boundary_group")
    ].copy()
    credit_rating = segment_metrics.loc[segment_metrics["dimension"].eq("credit_rating")].copy()
    exact_boundary = segment_metrics.loc[
        segment_metrics["dimension"].eq("is_exact_boundary_bbb_minus_bb_plus")
    ].copy()
    rating_agency_group = segment_metrics.loc[
        segment_metrics["dimension"].eq("rating_agency_group")
    ].copy()

    grade_note = (
        f"`{target_processed_path}`에서 대표 신용등급을 붙여 경계등급 분석을 수행했습니다. "
        f"test rows 중 등급이 매칭된 행은 `{matched_rating_rows}`개입니다. 이 등급 정보는 "
        "모델 학습에는 쓰지 않고 diagnostics 전용으로만 사용합니다."
        if grade_columns_available
        else "대표 신용등급 컬럼을 찾지 못해 BBB-/BB+ 경계 성능은 직접 계산하지 못했습니다."
    )

    return "\n".join(
        [
            "# Official 43-Feature Error Deep Dive",
            "",
            "공식 43개 XGBoost 모델의 test 구간 오답을 중심으로 시장/산업/기업규모/연도별 취약 구간을 진단했습니다.",
            "이 리포트는 새 변수를 바로 추가하기보다, 어떤 구간에서 어떤 방식의 보완이 필요한지 찾기 위한 자료입니다.",
            "",
            "## 1. Overall Test Performance",
            "",
            f"- Rows/positive rate: `{format_int(overall['rows'])}` / `{format_pct(overall['positive_rate'])}`",
            f"- PR-AUC/ROC-AUC: `{format_metric(overall['pr_auc'])}` / `{format_metric(overall['roc_auc'])}`",
            f"- Precision/Recall/F1: `{format_metric(overall['precision'])}` / "
            f"`{format_metric(overall['recall'])}` / `{format_metric(overall['f1'])}`",
            f"- FP/FN: `{format_int(overall['false_positive'])}` / `{format_int(overall['false_negative'])}`",
            "",
            "## 2. Market Split",
            "",
            markdown_table(
                market.sort_values("segment"),
                [
                    ("Market", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos rate", "positive_rate", "pct"),
                    ("PR-AUC", "pr_auc", "metric"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 3. Rating Boundary Split",
            "",
            grade_note,
            "",
            "BBB-/BB+ 주변은 투자적격과 투기등급이 갈리는 경계라, 모델의 객관적 평가 근거로 따로 보는 것이 좋습니다.",
            "",
            markdown_table(
                rating_boundary.sort_values("segment"),
                [
                    ("Boundary group", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos rate", "positive_rate", "pct"),
                    ("PR-AUC", "pr_auc", "metric"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
                max_rows=10,
            ),
            "",
            "### Exact BBB-/BB+ Boundary",
            "",
            markdown_table(
                exact_boundary.sort_values("segment"),
                [
                    ("BBB-/BB+", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos rate", "positive_rate", "pct"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
                max_rows=5,
            ),
            "",
            "### Individual Credit Ratings",
            "",
            markdown_table(
                credit_rating.sort_values("segment"),
                [
                    ("Rating", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos rate", "positive_rate", "pct"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
                max_rows=25,
            ),
            "",
            "### Rating Agency Group",
            "",
            markdown_table(
                rating_agency_group.sort_values("segment"),
                [
                    ("Agency group", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos rate", "positive_rate", "pct"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
                max_rows=10,
            ),
            "",
            "## 4. Weak Recall Segments",
            "",
            "실제 투기등급 중 놓친 비율이 높은 구간입니다. positive 표본이 너무 작은 구간은 제외했습니다.",
            "",
            markdown_table(
                industry_recall,
                [
                    ("Industry", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos", "positive_rows", "int"),
                    ("FN", "false_negative", "int"),
                    ("FN rate", "fn_rate_among_positives", "pct"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                ],
            ),
            "",
            markdown_table(
                firm_size_recall,
                [
                    ("Firm size", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos", "positive_rows", "int"),
                    ("FN", "false_negative", "int"),
                    ("FN rate", "fn_rate_among_positives", "pct"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                ],
            ),
            "",
            markdown_table(
                year_recall,
                [
                    ("Fiscal year", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Pos", "positive_rows", "int"),
                    ("FN", "false_negative", "int"),
                    ("FN rate", "fn_rate_among_positives", "pct"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                ],
            ),
            "",
            "## 5. False Positive Concentration",
            "",
            "전체 FP 중 비중이 큰 산업 구간입니다. FP가 몰리는 곳은 threshold/Stage 2 과민경고 필터를 우선 검토합니다.",
            "",
            markdown_table(
                industry_fp,
                [
                    ("Industry", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Neg", "negative_rows", "int"),
                    ("FP", "false_positive", "int"),
                    ("FP share", "fp_share_of_total_fp", "pct"),
                    ("FP rate", "fp_rate_among_negatives", "pct"),
                    ("Precision", "precision", "metric"),
                ],
            ),
            "",
            "## 6. Cross-Segment Error Concentration",
            "",
            markdown_table(
                cross_errors,
                [
                    ("Dimension", "dimension", "text"),
                    ("Segment", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                    ("Error count", "error_count", "int"),
                    ("FP share", "fp_share_of_total_fp", "pct"),
                    ("FN share", "fn_share_of_total_fn", "pct"),
                ],
                max_rows=15,
            ),
            "",
            "## 7. Feature Profile: FN vs TP",
            "",
            "FN은 실제 투기등급인데 모델이 안정적으로 본 사례입니다. 아래 변수 차이는 모델이 위험을 낮게 본 이유를 찾는 데 사용합니다.",
            "",
            markdown_table(
                fn_profile,
                [
                    ("Feature", "feature", "text"),
                    ("FN median", "error_median", "metric"),
                    ("TP median", "reference_median", "metric"),
                    ("Std delta", "standardized_median_delta", "metric"),
                    ("FN miss", "error_missing_rate", "pct"),
                    ("TP miss", "reference_missing_rate", "pct"),
                ],
                max_rows=12,
            ),
            "",
            "## 8. Feature Profile: FP vs TN",
            "",
            "FP는 실제 투자적격인데 모델이 위험하다고 본 사례입니다. 아래 변수 차이는 과민경고 원인을 찾는 데 사용합니다.",
            "",
            markdown_table(
                fp_profile,
                [
                    ("Feature", "feature", "text"),
                    ("FP median", "error_median", "metric"),
                    ("TN median", "reference_median", "metric"),
                    ("Std delta", "standardized_median_delta", "metric"),
                    ("FP miss", "error_missing_rate", "pct"),
                    ("TN miss", "reference_missing_rate", "pct"),
                ],
                max_rows=12,
            ),
            "",
            "## 9. Threshold Distance",
            "",
            "threshold 바로 근처 오류는 threshold 조정으로 개선 여지가 있고, 멀리 떨어진 고확신 오류는 변수/외부근거 보완이 더 필요합니다.",
            "",
            markdown_table(
                threshold_distance,
                [
                    ("Distance", "distance_bucket", "text"),
                    ("Rows", "rows", "int"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                ],
            ),
            "",
            "## 10. High-Confidence Error Examples",
            "",
            "확률상 모델이 자신 있게 틀린 사례입니다. 이 기업들은 뉴스/공시/등급전망 확인 우선순위가 높습니다.",
            "",
            "### False Negative",
            "",
            markdown_table(
                high_confidence_fn,
                [
                    ("Company", "corp_name", "text"),
                    ("Market", "market", "text"),
                    ("FY", "fiscal_year", "int"),
                    ("Industry", "industry_macro_category", "text"),
                    ("Size", "firm_size_group", "text"),
                    ("Prob", "prob_speculative", "metric"),
                    ("Threshold", "threshold", "metric"),
                ],
                max_rows=10,
            ),
            "",
            "### False Positive",
            "",
            markdown_table(
                high_confidence_fp,
                [
                    ("Company", "corp_name", "text"),
                    ("Market", "market", "text"),
                    ("FY", "fiscal_year", "int"),
                    ("Industry", "industry_macro_category", "text"),
                    ("Size", "firm_size_group", "text"),
                    ("Prob", "prob_speculative", "metric"),
                    ("Threshold", "threshold", "metric"),
                ],
                max_rows=10,
            ),
            "",
            "## 11. Rating Boundary Availability",
            "",
            grade_note,
            "",
            "## 12. What To Do Next",
            "",
            "1. FN이 몰린 산업/규모/연도 조합에서 실제 외부 이벤트가 있었는지 뉴스·공시·등급전망을 먼저 확인합니다.",
            "2. FP가 많은 KOSDAQ/제조업/소형 구간은 전체 threshold를 바꾸기보다 Stage 2 과민경고 필터 또는 구간별 보조 판단으로 완화합니다.",
            "3. FN과 TP의 차이가 큰 변수는 '위험을 숨기는 안정 신호'인지 확인합니다. 특히 규모성/배당/상장연도/절대금액 신호가 실제 위험을 가리는지 봅니다.",
            "4. FP와 TN의 차이가 큰 변수는 단기 악화와 지속 악화를 구분하는 추가 변수 후보로 바꿔봅니다. 예: 2~3년 지속 손실, 현금흐름 악화 지속기간, 운전자본 변화 지속성.",
            "5. BBB-/BB+ 경계 성능은 별도 객관 평가표로 발표 자료에 포함합니다.",
            "",
        ]
    )


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    scores = read_scores(args.prediction_scores)
    target_labels = read_target_labels(args.target_processed)
    scores = attach_target_labels(scores, target_labels)
    test = scores.loc[scores["split"].eq("test")].copy()
    master = read_master(args.feature_master)
    feature_columns = read_feature_columns(args.feature_spec, master)
    merged = test.merge(
        master.loc[:, [*KEY_COLUMNS, *feature_columns]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    grade_columns_available = "credit_rating" in test.columns and bool(test["credit_rating"].notna().any())
    matched_rating_rows = int(test["credit_rating"].notna().sum()) if "credit_rating" in test.columns else 0

    segment_metrics = build_segment_metrics(test)
    concentration = build_error_concentration(segment_metrics)
    error_cases = build_error_cases(test)
    feature_profiles = build_feature_profile(merged, feature_columns)
    threshold_distance = build_threshold_distance_summary(test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segment_path = args.output_dir / "official_43_error_segment_metrics.csv"
    concentration_path = args.output_dir / "official_43_error_concentration.csv"
    cases_path = args.output_dir / "official_43_error_cases_ranked.csv"
    profile_path = args.output_dir / "official_43_error_feature_profile.csv"
    threshold_path = args.output_dir / "official_43_threshold_distance_errors.csv"
    rating_boundary_path = args.output_dir / "official_43_rating_boundary_metrics.csv"
    report_path = args.output_dir / "official_43_error_deep_dive_report.md"
    summary_path = args.output_dir / "official_43_error_deep_dive_summary.json"
    rating_boundary_metrics = segment_metrics.loc[
        segment_metrics["dimension"].isin(
            [
                "credit_rating",
                "rating_boundary_group",
                "is_exact_boundary_bbb_minus_bb_plus",
                "is_near_boundary_bbb_bb",
                "rating_agency_group",
                "selection_scope",
                "market_x_rating_boundary",
            ]
        )
    ].copy()

    segment_metrics.to_csv(segment_path, index=False, encoding="utf-8-sig")
    concentration.to_csv(concentration_path, index=False, encoding="utf-8-sig")
    error_cases.to_csv(cases_path, index=False, encoding="utf-8-sig")
    feature_profiles.to_csv(profile_path, index=False, encoding="utf-8-sig")
    threshold_distance.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    rating_boundary_metrics.to_csv(rating_boundary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            segment_metrics=segment_metrics,
            concentration=concentration,
            feature_profiles=feature_profiles,
            threshold_distance=threshold_distance,
            error_cases=error_cases,
            grade_columns_available=grade_columns_available,
            target_processed_path=args.target_processed,
            matched_rating_rows=matched_rating_rows,
        ),
        encoding="utf-8",
    )

    overall = segment_metrics.loc[segment_metrics["dimension"].eq("overall")].iloc[0]
    weak_recall = segment_top(
        concentration,
        dimension="industry_macro_category",
        sort_column="fn_rate_among_positives",
        min_denominator_column="positive_rows",
    ).head(5)
    fp_concentration = segment_top(
        concentration,
        dimension="industry_macro_category",
        sort_column="fp_share_of_total_fp",
        min_denominator_column="negative_rows",
    ).head(5)
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiment_type": "official_43_error_deep_dive",
        "overall": overall.to_dict(),
        "grade_boundary_available": grade_columns_available,
        "target_processed_path": str(args.target_processed),
        "matched_rating_rows": matched_rating_rows,
        "top_weak_recall_industries": weak_recall.to_dict(orient="records"),
        "top_fp_concentration_industries": fp_concentration.to_dict(orient="records"),
        "paths": {
            "segment_metrics": str(segment_path.relative_to(ROOT)),
            "error_concentration": str(concentration_path.relative_to(ROOT)),
            "error_cases_ranked": str(cases_path.relative_to(ROOT)),
            "feature_profile": str(profile_path.relative_to(ROOT)),
            "threshold_distance_errors": str(threshold_path.relative_to(ROOT)),
            "rating_boundary_metrics": str(rating_boundary_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[Saved] {segment_path}")
    print(f"[Saved] {concentration_path}")
    print(f"[Saved] {cases_path}")
    print(f"[Saved] {profile_path}")
    print(f"[Saved] {threshold_path}")
    print(f"[Saved] {rating_boundary_path}")
    print(f"[Saved] {report_path}")
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
