from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_43_xgboost" / "diagnostics"

RANDOM_STATE = 42
PROBABILITY_CLIP_EPSILON = 1e-6
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 6)
RECALL_FLOOR = 0.85

AMOUNT_COLUMNS = ["assets_total", "gross_profit", "depreciation"]
FIRM_SIZE_DUMMIES = [
    "firm_size_group_large",
    "firm_size_group_mid_sized",
    "firm_size_group_other",
    "firm_size_group_small_and_medium",
]
RATIO_PERCENTILE_COLUMNS = [
    "interest_coverage_ratio",
    "cashflow_coverage_ratio",
    "net_margin",
    "equity_ratio",
    "capital_impairment_ratio",
    "total_debt_turnover",
    "short_term_borrowings_share",
    "ocf_to_total_borrowings",
    "market_to_book",
]
LAG_DELTA_COLUMNS = [
    "current_ratio",
    "cash_ratio",
    "debt_ratio",
    "equity_ratio",
    "net_margin",
    "operating_roa",
    "interest_coverage_ratio",
    "cashflow_coverage_ratio",
    "total_borrowings_ratio",
]
FOCUS_SEGMENTS = [
    ("overall", "all", None, None),
    ("market", "KOSDAQ", "market", "KOSDAQ"),
    ("market", "KOSPI", "market", "KOSPI"),
    ("industry", "manufacturing", "industry_macro_category", "manufacturing"),
    ("industry", "it_services", "industry_macro_category", "it_services"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SHAP-motivated feature experiments for the 43-feature XGBoost model. "
            "Experiments use XGBoost native missing, Platt scaling, and a validation "
            "threshold chosen for max precision with recall >= 0.85."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_split_frames(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(input_dir / "xgb_train.csv", encoding="utf-8-sig"),
        pd.read_csv(input_dir / "xgb_valid.csv", encoding="utf-8-sig"),
        pd.read_csv(input_dir / "xgb_test.csv", encoding="utf-8-sig"),
    )


def read_id_frames(input_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(input_dir / f"xgb_id_{split}.csv", encoding="utf-8-sig")
        for split in ["train", "valid", "test"]
    }


def signed_log1p(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.sign(values) * np.log1p(np.abs(values))


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce")
    denominator_values = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return numerator_values / denominator_values


def add_industry_percentiles(
    frame: pd.DataFrame,
    id_frame: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    keys = id_frame.loc[:, ["industry_macro_category", "fiscal_year"]].reset_index(drop=True)
    added_columns = []

    for column in columns:
        if column not in output.columns:
            continue
        values = pd.to_numeric(output[column], errors="coerce")
        ranking_frame = pd.DataFrame(
            {
                "industry": keys["industry_macro_category"],
                "fiscal_year": keys["fiscal_year"],
                "value": values,
            }
        )
        percentile_column = f"{column}_industry_pct"
        output[percentile_column] = ranking_frame.groupby(
            ["fiscal_year", "industry"],
        )["value"].rank(pct=True, method="average")
        added_columns.append(percentile_column)

    return output, added_columns


def add_lag_delta_features(
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    columns: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    pieces = []
    for split in ["train", "valid", "test"]:
        frame = frames[split].reset_index(drop=True).copy()
        ids = id_frames[split].reset_index(drop=True).copy()
        frame["_split"] = split
        frame["_row_order"] = np.arange(len(frame))
        frame["_stock_code"] = ids["stock_code"].astype(str)
        frame["_fiscal_year"] = pd.to_numeric(ids["fiscal_year"], errors="coerce")
        pieces.append(frame)

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.sort_values(["_stock_code", "_fiscal_year", "_split", "_row_order"])
    added_columns = []
    for column in columns:
        if column not in combined.columns:
            continue
        delta_column = f"{column}_yoy_delta"
        values = pd.to_numeric(combined[column], errors="coerce")
        previous_values = values.groupby(combined["_stock_code"]).shift(1)
        combined[delta_column] = values - previous_values
        added_columns.append(delta_column)

    output_frames = {}
    combined = combined.sort_values(["_split", "_row_order"])
    for split in ["train", "valid", "test"]:
        split_frame = combined.loc[combined["_split"] == split].copy()
        split_frame = split_frame.sort_values("_row_order")
        output_frames[split] = split_frame.drop(
            columns=["_split", "_row_order", "_stock_code", "_fiscal_year"]
        )

    return output_frames, added_columns


def add_log_amount_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    added_columns = []
    for column in AMOUNT_COLUMNS:
        if column not in output.columns:
            continue
        log_column = f"log_{column}"
        output[log_column] = signed_log1p(output[column])
        added_columns.append(log_column)
    return output, added_columns


def add_scale_adjusted_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    added_columns = []
    if {"gross_profit", "assets_total"}.issubset(output.columns):
        output["gross_profit_to_assets"] = safe_ratio(
            output["gross_profit"], output["assets_total"]
        )
        added_columns.append("gross_profit_to_assets")
    if {"depreciation", "assets_total"}.issubset(output.columns):
        output["depreciation_to_assets"] = safe_ratio(
            output["depreciation"], output["assets_total"]
        )
        added_columns.append("depreciation_to_assets")
    if "assets_total" in output.columns:
        output["log_assets_total"] = signed_log1p(output["assets_total"])
        added_columns.append("log_assets_total")
    return output, added_columns


def fit_platt_calibration(
    y_valid: pd.Series,
    valid_probabilities: np.ndarray,
) -> tuple[float, float]:
    clipped = np.clip(
        valid_probabilities,
        PROBABILITY_CLIP_EPSILON,
        1.0 - PROBABILITY_CLIP_EPSILON,
    )
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, y_valid.astype(int))
    return float(calibrator.coef_[0][0]), float(calibrator.intercept_[0])


def apply_platt_calibration(
    probabilities: np.ndarray,
    coef: float,
    intercept: float,
) -> np.ndarray:
    clipped = np.clip(probabilities, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(intercept + coef * logits)))


def classification_counts(y_true: pd.Series, predictions: np.ndarray) -> dict[str, int]:
    y_true_array = y_true.to_numpy(dtype=int)
    pred_array = predictions.astype(int)
    return {
        "true_negative": int(((y_true_array == 0) & (pred_array == 0)).sum()),
        "false_positive": int(((y_true_array == 0) & (pred_array == 1)).sum()),
        "false_negative": int(((y_true_array == 1) & (pred_array == 0)).sum()),
        "true_positive": int(((y_true_array == 1) & (pred_array == 1)).sum()),
    }


def classification_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float | int]:
    counts = classification_counts(y_true, predictions)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        **counts,
    }


def choose_threshold(
    y_valid: pd.Series,
    valid_probabilities: np.ndarray,
    recall_floor: float = RECALL_FLOOR,
) -> tuple[float, dict[str, float | int | str]]:
    rows = []
    for threshold in THRESHOLD_GRID:
        predictions = valid_probabilities >= threshold
        rows.append({"threshold": float(threshold), **classification_metrics(y_valid, predictions)})

    sweep = pd.DataFrame(rows)
    candidates = sweep.loc[sweep["recall"] >= recall_floor]
    selection_rule = f"valid_max_precision_with_recall_ge_{recall_floor:.2f}"
    if candidates.empty:
        candidates = sweep
        selection_rule = "valid_best_f1_fallback"
        row = candidates.sort_values(
            ["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    else:
        row = candidates.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]

    metrics = row.drop(labels=["threshold"]).to_dict()
    metrics["threshold_selection_rule"] = selection_rule
    return float(row["threshold"]), metrics


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> XGBClassifier:
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=float(negative / positive) if positive else 1.0,
        early_stopping_rounds=50,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def probability_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "logloss": float(log_loss(y_true, clipped)),
    }


def build_variant(
    variant: str,
    base_frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[str, dict[str, pd.DataFrame], list[str]]:
    frames = {split: frame.copy() for split, frame in base_frames.items()}
    features = list(feature_columns)

    if variant == "baseline_43_native":
        return "현재 43개 변수, XGBoost native missing 기준", frames, features

    if variant in {
        "amount_log_add_native",
        "amount_log_replace_native",
        "log_amounts_drop_firm_size_native",
        "full_shap_context_add_native",
    }:
        added_columns: list[str] = []
        for split in ["train", "valid", "test"]:
            frames[split], split_added_columns = add_log_amount_features(frames[split])
            added_columns = split_added_columns
        if variant == "amount_log_add_native":
            return "절대금액 raw 유지 + signed log1p 변수 추가", frames, [*features, *added_columns]
        if variant == "amount_log_replace_native":
            features = [column for column in features if column not in AMOUNT_COLUMNS]
            return "절대금액 3개를 signed log1p 변수로 대체", frames, [*features, *added_columns]
        if variant == "log_amounts_drop_firm_size_native":
            features = [
                column
                for column in features
                if column not in AMOUNT_COLUMNS and column not in FIRM_SIZE_DUMMIES
            ]
            return (
                "절대금액은 log로 대체하고 기업규모 one-hot 제거",
                frames,
                [*features, *added_columns],
            )

    if variant in {
        "industry_amount_pct_add_native",
        "industry_amount_pct_replace_native",
        "full_shap_context_add_native",
    }:
        added_columns = []
        for split in ["train", "valid", "test"]:
            frames[split], split_added_columns = add_industry_percentiles(
                frames[split],
                id_frames[split],
                AMOUNT_COLUMNS,
            )
            added_columns = split_added_columns
        if variant == "industry_amount_pct_add_native":
            return (
                "절대금액 raw 유지 + 산업-연도 내부 백분위 추가",
                frames,
                [*features, *added_columns],
            )
        if variant == "industry_amount_pct_replace_native":
            features = [column for column in features if column not in AMOUNT_COLUMNS]
            return (
                "절대금액 3개를 산업-연도 내부 백분위로 대체",
                frames,
                [*features, *added_columns],
            )

    if variant in {"scale_adjusted_amounts_add_native", "full_shap_context_add_native"}:
        added_columns = []
        for split in ["train", "valid", "test"]:
            frames[split], split_added_columns = add_scale_adjusted_features(frames[split])
            added_columns = split_added_columns
        if variant == "scale_adjusted_amounts_add_native":
            return (
                "총자산 대비 매출총이익/감가상각비 + log 총자산 추가",
                frames,
                [*features, *added_columns],
            )

    if variant in {"ratio_industry_pct_add_native", "full_shap_context_add_native"}:
        added_columns = []
        for split in ["train", "valid", "test"]:
            frames[split], split_added_columns = add_industry_percentiles(
                frames[split],
                id_frames[split],
                RATIO_PERCENTILE_COLUMNS,
            )
            added_columns = split_added_columns
        if variant == "ratio_industry_pct_add_native":
            return (
                "SHAP 상위 재무비율의 산업-연도 내부 백분위 추가",
                frames,
                [*features, *added_columns],
            )

    if variant in {"lag_delta_key_ratios_add_native", "full_shap_context_add_native"}:
        frames, added_columns = add_lag_delta_features(frames, id_frames, LAG_DELTA_COLUMNS)
        if variant == "lag_delta_key_ratios_add_native":
            return "주요 재무비율의 전년 대비 변화량 추가", frames, [*features, *added_columns]

    if variant == "full_shap_context_add_native":
        new_columns = [
            column
            for column in frames["train"].columns
            if column not in base_frames["train"].columns and column != "is_speculative"
        ]
        return (
            "log 절대금액 + 산업 백분위 + 규모보정 + 전년 변화량 통합",
            frames,
            [*features, *new_columns],
        )

    if variant == "drop_firm_size_dummies_native":
        features = [column for column in features if column not in FIRM_SIZE_DUMMIES]
        return "기업규모 one-hot 제거", frames, features

    raise ValueError(f"Unknown variant: {variant}")


def evaluate_variant(
    *,
    variant: str,
    note: str,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_frame = frames["train"]
    valid_frame = frames["valid"]
    test_frame = frames["test"]
    y_train = train_frame["is_speculative"].astype(int)
    y_valid = valid_frame["is_speculative"].astype(int)
    y_test = test_frame["is_speculative"].astype(int)

    model = train_xgboost(
        train_frame.loc[:, feature_columns],
        y_train,
        valid_frame.loc[:, feature_columns],
        y_valid,
    )
    valid_raw_probabilities = model.predict_proba(valid_frame.loc[:, feature_columns])[:, 1]
    test_raw_probabilities = model.predict_proba(test_frame.loc[:, feature_columns])[:, 1]
    coef, intercept = fit_platt_calibration(y_valid, valid_raw_probabilities)
    valid_probabilities = apply_platt_calibration(valid_raw_probabilities, coef, intercept)
    test_probabilities = apply_platt_calibration(test_raw_probabilities, coef, intercept)
    threshold, valid_threshold_metrics = choose_threshold(y_valid, valid_probabilities)
    predictions = test_probabilities >= threshold

    calibrated_probability_metrics = probability_metrics(y_test, test_probabilities)
    raw_probability_metrics = probability_metrics(y_test, test_raw_probabilities)
    test_classification_metrics = classification_metrics(y_test, predictions)
    metric_row = {
        "variant": variant,
        "note": note,
        "feature_count": len(feature_columns),
        "best_iteration": getattr(model, "best_iteration", None),
        "threshold_tuned": threshold,
        "threshold_selection_rule": valid_threshold_metrics["threshold_selection_rule"],
        "valid_precision_at_threshold": valid_threshold_metrics["precision"],
        "valid_recall_at_threshold": valid_threshold_metrics["recall"],
        "valid_f1_at_threshold": valid_threshold_metrics["f1"],
        "test_pr_auc": calibrated_probability_metrics["pr_auc"],
        "test_roc_auc": calibrated_probability_metrics["roc_auc"],
        "test_brier": calibrated_probability_metrics["brier"],
        "test_logloss": calibrated_probability_metrics["logloss"],
        "raw_test_pr_auc": raw_probability_metrics["pr_auc"],
        "raw_test_roc_auc": raw_probability_metrics["roc_auc"],
        "raw_test_brier": raw_probability_metrics["brier"],
        "raw_test_logloss": raw_probability_metrics["logloss"],
        "test_precision": test_classification_metrics["precision"],
        "test_recall": test_classification_metrics["recall"],
        "test_f1": test_classification_metrics["f1"],
        "test_true_negative": test_classification_metrics["true_negative"],
        "test_false_positive": test_classification_metrics["false_positive"],
        "test_false_negative": test_classification_metrics["false_negative"],
        "test_true_positive": test_classification_metrics["true_positive"],
    }

    test_ids = id_frames["test"].reset_index(drop=True)
    segment_base = test_ids.assign(
        is_speculative=y_test.reset_index(drop=True),
        prediction=predictions.astype(int),
        prob_speculative=test_probabilities,
    )
    segment_rows = []
    for segment_type, segment_name, column, value in FOCUS_SEGMENTS:
        segment = (
            segment_base if column is None else segment_base.loc[segment_base[column] == value]
        )
        if segment.empty:
            continue
        segment_y = segment["is_speculative"].astype(int)
        segment_predictions = segment["prediction"].to_numpy(dtype=int)
        segment_metrics = classification_metrics(segment_y, segment_predictions)
        negatives = int((segment_y == 0).sum())
        positives = int((segment_y == 1).sum())
        segment_rows.append(
            {
                "variant": variant,
                "segment_type": segment_type,
                "segment": segment_name,
                "rows": len(segment),
                "positives": positives,
                "negatives": negatives,
                "fp_rate_among_negatives": (
                    segment_metrics["false_positive"] / negatives if negatives else None
                ),
                "fn_rate_among_positives": (
                    segment_metrics["false_negative"] / positives if positives else None
                ),
                **segment_metrics,
            }
        )

    return metric_row, segment_rows


def run_experiments(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in train.columns if column != "is_speculative"]
    base_frames = {"train": train, "valid": valid, "test": test}
    variants = [
        "baseline_43_native",
        "amount_log_add_native",
        "amount_log_replace_native",
        "industry_amount_pct_add_native",
        "industry_amount_pct_replace_native",
        "scale_adjusted_amounts_add_native",
        "ratio_industry_pct_add_native",
        "lag_delta_key_ratios_add_native",
        "drop_firm_size_dummies_native",
        "log_amounts_drop_firm_size_native",
        "full_shap_context_add_native",
    ]

    metric_rows = []
    segment_rows = []
    for variant in variants:
        note, frames, variant_features = build_variant(
            variant,
            base_frames,
            id_frames,
            feature_columns,
        )
        metric_row, variant_segment_rows = evaluate_variant(
            variant=variant,
            note=note,
            frames=frames,
            id_frames=id_frames,
            feature_columns=variant_features,
        )
        metric_rows.append(metric_row)
        segment_rows.extend(variant_segment_rows)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["test_f1", "test_precision", "test_pr_auc"],
        ascending=False,
    )
    segments = pd.DataFrame(segment_rows)
    return metrics, segments


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame.to_dict(orient="records"):
        values = []
        for _, column, kind in columns:
            value = row.get(column)
            if kind == "metric":
                values.append(format_metric(value))
            elif kind == "int":
                values.append(format_int(value))
            else:
                values.append(str(value) if value is not None else "")
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def build_report(metrics: pd.DataFrame, segments: pd.DataFrame) -> str:
    baseline = metrics.loc[metrics["variant"] == "baseline_43_native"].iloc[0]
    best = metrics.iloc[0]
    baseline_fp = int(baseline["test_false_positive"])
    baseline_fn = int(baseline["test_false_negative"])
    best_fp_delta = int(best["test_false_positive"]) - baseline_fp
    best_fn_delta = int(best["test_false_negative"]) - baseline_fn
    best_f1_delta = float(best["test_f1"]) - float(baseline["test_f1"])

    kosdaq_segments = segments.loc[segments["segment"] == "KOSDAQ"].copy()
    kosdaq_baseline = kosdaq_segments.loc[kosdaq_segments["variant"] == "baseline_43_native"].iloc[
        0
    ]
    kosdaq_best = kosdaq_segments.sort_values(
        ["false_positive", "fn_rate_among_positives", "f1"],
        ascending=[True, True, False],
    ).iloc[0]

    if best["variant"] == "baseline_43_native":
        recommendation = (
            "- 현재 기준에서는 새 변수 추가보다 기존 43개 변수셋이 가장 안정적입니다. "
            "성능 개선은 변수 추가보다 오류 사례 기반 라벨/원천 변수 보강 쪽이 더 유망합니다."
        )
    elif best_f1_delta >= 0.005:
        recommendation = (
            f"- `{best['variant']}`가 baseline 대비 F1을 `{best_f1_delta:+.4f}` 개선했습니다. "
            "다음 후보 모델로 별도 재학습/대시보드 검증을 진행할 가치가 있습니다."
        )
    else:
        recommendation = (
            f"- `{best['variant']}`가 F1 기준 최상위지만 개선 폭이 `{best_f1_delta:+.4f}`로 작습니다. "
            "바로 운영 반영하기보다 추가 검증 후보로 보관하는 편이 안전합니다."
        )

    return "\n".join(
        [
            "# SHAP-Driven Feature Improvement Experiments",
            "",
            "오류 사례 SHAP 분석에서 반복적으로 나타난 절대금액, 기업규모, 산업 내 위치, "
            "전년 대비 악화 신호를 변수 후보로 만들어 비교한 실험입니다.",
            "모든 실험은 XGBoost native missing, Platt scaling, validation 기준 "
            f"`recall >= {RECALL_FLOOR:.2f}` 조건에서 precision 최대 threshold를 사용했습니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline F1: `{format_metric(baseline['test_f1'])}`, Precision: "
            f"`{format_metric(baseline['test_precision'])}`, Recall: "
            f"`{format_metric(baseline['test_recall'])}`, FP/FN: "
            f"`{baseline_fp}/{baseline_fn}`",
            f"- Best F1 variant: `{best['variant']}` "
            f"(F1 `{format_metric(best['test_f1'])}`, Precision "
            f"`{format_metric(best['test_precision'])}`, Recall "
            f"`{format_metric(best['test_recall'])}`, FP/FN "
            f"`{format_int(best['test_false_positive'])}/"
            f"{format_int(best['test_false_negative'])}`)",
            f"- Best vs baseline: F1 `{best_f1_delta:+.4f}`, FP `{best_fp_delta:+d}`, "
            f"FN `{best_fn_delta:+d}`",
            recommendation,
            "",
            "## 2. 전체 성능 비교",
            "",
            markdown_table(
                metrics,
                [
                    ("Variant", "variant", "text"),
                    ("Features", "feature_count", "int"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("PR-AUC", "test_pr_auc", "metric"),
                    ("ROC-AUC", "test_roc_auc", "metric"),
                    ("Precision", "test_precision", "metric"),
                    ("Recall", "test_recall", "metric"),
                    ("F1", "test_f1", "metric"),
                    ("FP", "test_false_positive", "int"),
                    ("FN", "test_false_negative", "int"),
                ],
            ),
            "",
            "## 3. KOSDAQ FP 관점",
            "",
            f"- Baseline KOSDAQ FP: `{format_int(kosdaq_baseline['false_positive'])}`, "
            f"FN: `{format_int(kosdaq_baseline['false_negative'])}`",
            f"- KOSDAQ FP 최소 variant: `{kosdaq_best['variant']}` "
            f"(FP `{format_int(kosdaq_best['false_positive'])}`, "
            f"FN `{format_int(kosdaq_best['false_negative'])}`, "
            f"F1 `{format_metric(kosdaq_best['f1'])}`)",
            "",
            markdown_table(
                kosdaq_segments.sort_values(["false_positive", "false_negative"]).head(8),
                [
                    ("Variant", "variant", "text"),
                    ("Rows", "rows", "int"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 4. 해석",
            "",
            "- 절대금액을 log나 산업 백분위로 바꾸는 실험은 FP를 줄일 수 있는지 확인하기 위한 실험입니다.",
            "- 전년 대비 변화량은 SK이노베이션, KG모빌리티처럼 규모가 큰 기업의 위험을 "
            "너무 안정적으로 보는 FN 문제를 줄일 수 있는지 확인하기 위한 실험입니다.",
            "- 개선 폭이 작거나 특정 구간만 좋아지는 경우에는 운영 모델을 즉시 교체하지 않고, "
            "추가 feature 후보 또는 세그먼트별 보정 후보로만 관리합니다.",
        ]
    )


def write_outputs(metrics: pd.DataFrame, segments: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "shap_feature_experiment_metrics.csv"
    segments_path = output_dir / "shap_feature_experiment_segment_metrics.csv"
    report_path = output_dir / "shap_feature_experiment_report.md"
    summary_path = output_dir / "shap_feature_experiment_summary.json"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")
    report = build_report(metrics, segments)
    report_path.write_text(report, encoding="utf-8")

    baseline = metrics.loc[metrics["variant"] == "baseline_43_native"].iloc[0]
    best = metrics.iloc[0]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "threshold_policy": f"max precision with validation recall >= {RECALL_FLOOR:.2f}",
        "baseline": baseline.to_dict(),
        "best_by_test_f1": best.to_dict(),
        "output_files": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "segment_metrics": str(segments_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    train, valid, test = read_split_frames(args.input_dir)
    id_frames = read_id_frames(args.input_dir)
    metrics, segments = run_experiments(train, valid, test, id_frames)
    write_outputs(metrics, segments, args.output_dir)
    print(
        json.dumps(
            {
                "best_variant": metrics.iloc[0]["variant"],
                "best_f1": float(metrics.iloc[0]["test_f1"]),
                "baseline_f1": float(
                    metrics.loc[metrics["variant"] == "baseline_43_native"].iloc[0]["test_f1"]
                ),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
