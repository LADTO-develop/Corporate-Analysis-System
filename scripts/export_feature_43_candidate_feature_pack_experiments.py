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
INPUT_DIR = ROOT / "data" / "input" / "credit_46_features"
RAW_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"

RANDOM_STATE = 42
PROBABILITY_CLIP_EPSILON = 1e-6
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 6)
RECALL_FLOOR = 0.85
JOIN_KEYS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
FOCUS_SEGMENTS = [
    ("overall", "all", None, None),
    ("market", "KOSDAQ", "market", "KOSDAQ"),
    ("market", "KOSPI", "market", "KOSPI"),
    ("industry", "manufacturing", "industry_macro_category", "manufacturing"),
    ("industry", "it_services", "industry_macro_category", "it_services"),
]

FEATURE_PACKS: dict[str, dict[str, Any]] = {
    "profitability_quality_add_native": {
        "note": "ROE, EBITDA margin, 이자부담률 등 수익성/상환능력 후보 추가",
        "columns": [
            "roe",
            "operating_roe",
            "ebitda_margin",
            "interest_burden_ratio",
            "gross_margin",
            "operating_margin_diff",
            "ebitda_margin_diff",
        ],
    },
    "cashflow_quality_add_native": {
        "note": "영업현금흐름의 질과 변화량 후보 추가",
        "columns": [
            "ocf_to_total_assets",
            "ocf_deficit_flag",
            "delta_accruals_ratio",
            "ocf_to_total_liabilities_diff",
            "ocf_to_total_borrowings_diff",
            "rolling_3y_cv_ocf_to_total_borrowings",
        ],
    },
    "distress_flags_add_native": {
        "note": "좀비/연속 적자/현금흐름 악화 등 해석 가능한 부실 징후 플래그 추가",
        "columns": [
            "is_zombie_3y",
            "is_3y_consecutive_operating_loss",
            "is_3y_consecutive_ocf_deficit",
            "is_operating_income_turn_negative",
            "is_ocf_turn_negative",
            "negative_equity_flag",
            "is_negative_equity_entry",
            "is_current_ratio_below_1",
        ],
    },
    "working_capital_quality_add_native": {
        "note": "매출채권/재고/매입채무 회전일수와 운전자본 비율 후보 추가",
        "columns": [
            "ar_days",
            "inventory_days",
            "ap_days",
            "ar_days_diff",
            "inventory_days_diff",
            "ap_days_diff",
            "accounts_receivable_ratio",
            "inventory_ratio",
            "contract_assets_ratio",
            "advances_from_customers_ratio",
        ],
    },
    "macro_delta_add_native": {
        "note": "금리/환율/시장 스프레드 변화량 후보 추가",
        "columns": [
            "base_rate_diff",
            "treasury_3y_diff",
            "usd_krw_diff",
            "market_spread_diff",
            "spec_spread_diff",
        ],
    },
    "macro_context_add_native": {
        "note": "거시 레벨 지표와 변화량 후보 추가",
        "columns": [
            "base_rate",
            "treasury_3y",
            "corp_aa_3y",
            "corp_bbb_3y",
            "market_spread",
            "usd_krw",
            "ppi",
            "base_rate_diff",
            "treasury_3y_diff",
            "usd_krw_diff",
            "market_spread_diff",
            "spec_spread_diff",
        ],
    },
    "audit_flag_add_native": {
        "note": "감사의견 관련 플래그 후보 추가",
        "columns": ["audit_qualified_flag"],
    },
    "top_univariate_add_native": {
        "note": "단변량 선별에서 상대적으로 강했던 후보 묶음 추가",
        "columns": [
            "roe",
            "ebitda_margin",
            "interest_burden_ratio",
            "operating_roe",
            "ocf_to_total_assets",
            "is_zombie_3y",
            "rolling_3y_cv_operating_margin",
            "ar_days",
            "capital_impairment_diff",
            "equity_growth",
            "non_paid_in_equity_ratio",
        ],
    },
    "combined_interpretable_add_native": {
        "note": "수익성, 현금흐름, 부실 플래그, 운전자본 후보 통합",
        "packs": [
            "profitability_quality_add_native",
            "cashflow_quality_add_native",
            "distress_flags_add_native",
            "working_capital_quality_add_native",
        ],
    },
    "combined_all_candidate_add_native": {
        "note": "전체 후보 변수팩 통합",
        "packs": [
            "profitability_quality_add_native",
            "cashflow_quality_add_native",
            "distress_flags_add_native",
            "working_capital_quality_add_native",
            "macro_context_add_native",
            "audit_flag_add_native",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run candidate feature-pack experiments for the 46-feature XGBoost credit model. "
            "Candidate variables are joined from the Model V1 raw dataset and compared with "
            "the current 46-feature baseline under the same OOT validation policy."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
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
        split: pd.read_csv(
            input_dir / f"xgb_id_{split}.csv",
            encoding="utf-8-sig",
            dtype={"stock_code": str},
        )
        for split in ["train", "valid", "test"]
    }


def read_raw_features(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"stock_code": str})
    duplicates = raw.duplicated(JOIN_KEYS).sum()
    if duplicates:
        raise ValueError(f"Raw Model V1 has duplicate rows for join keys: {duplicates}")
    return raw


def all_candidate_columns() -> list[str]:
    columns: list[str] = []
    for pack_name in FEATURE_PACKS:
        columns.extend(feature_pack_columns(pack_name))
    return unique_preserve_order(columns)


def feature_pack_columns(pack_name: str) -> list[str]:
    spec = FEATURE_PACKS[pack_name]
    columns = list(spec.get("columns", []))
    for child_pack in spec.get("packs", []):
        columns.extend(feature_pack_columns(child_pack))
    return unique_preserve_order(columns)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def attach_candidate_columns(
    *,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    raw: pd.DataFrame,
    candidate_columns: list[str],
) -> dict[str, pd.DataFrame]:
    missing_columns = [column for column in candidate_columns if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Candidate columns are missing from raw Model V1: {missing_columns}")

    raw_subset = raw.loc[:, [*JOIN_KEYS, *candidate_columns]].copy()
    for column in candidate_columns:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="coerce")

    output = {}
    for split, frame in frames.items():
        ids = id_frames[split].reset_index(drop=True).copy()
        joined = ids.loc[:, JOIN_KEYS].merge(raw_subset, on=JOIN_KEYS, how="left", indicator=True)
        unmatched = int(joined["_merge"].ne("both").sum())
        if unmatched:
            raise ValueError(f"{split} split has unmatched raw rows: {unmatched}")
        joined = joined.drop(columns=["_merge"])
        split_frame = frame.reset_index(drop=True).copy()
        for column in candidate_columns:
            split_frame[column] = joined[column]
        output[split] = split_frame
    return output


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


def probability_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "logloss": float(log_loss(y_true, clipped)),
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


def evaluate_variant(
    *,
    variant: str,
    note: str,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    added_columns: list[str],
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
    valid_predictions = valid_probabilities >= threshold
    test_predictions = test_probabilities >= threshold
    valid_probability_metrics = probability_metrics(y_valid, valid_probabilities)
    test_probability_metrics = probability_metrics(y_test, test_probabilities)
    valid_classification_metrics = classification_metrics(y_valid, valid_predictions)
    test_classification_metrics = classification_metrics(y_test, test_predictions)

    metric_row: dict[str, Any] = {
        "variant": variant,
        "note": note,
        "feature_count": len(feature_columns),
        "added_feature_count": len(added_columns),
        "added_features": ", ".join(added_columns),
        "best_iteration": getattr(model, "best_iteration", None),
        "threshold_tuned": threshold,
        "threshold_selection_rule": valid_threshold_metrics["threshold_selection_rule"],
        "valid_precision_at_policy": valid_threshold_metrics["precision"],
        "valid_recall_at_policy": valid_threshold_metrics["recall"],
        "valid_f1_at_policy": valid_threshold_metrics["f1"],
    }
    metric_row.update({f"valid_{key}": value for key, value in valid_probability_metrics.items()})
    metric_row.update(
        {f"valid_{key}_at_threshold": value for key, value in valid_classification_metrics.items()}
    )
    metric_row.update({f"test_{key}": value for key, value in test_probability_metrics.items()})
    metric_row.update(
        {f"test_{key}_at_threshold": value for key, value in test_classification_metrics.items()}
    )

    test_ids = id_frames["test"].reset_index(drop=True)
    segment_base = test_ids.assign(
        is_speculative=y_test.reset_index(drop=True),
        prediction=test_predictions.astype(int),
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
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_feature_columns = [column for column in train.columns if column != "is_speculative"]
    base_frames = {"train": train, "valid": valid, "test": test}
    all_candidates = all_candidate_columns()
    candidate_frames = attach_candidate_columns(
        frames=base_frames,
        id_frames=id_frames,
        raw=raw,
        candidate_columns=all_candidates,
    )

    metric_rows = []
    segment_rows = []
    variants = ["baseline_43_native", *FEATURE_PACKS.keys()]
    for variant in variants:
        if variant == "baseline_43_native":
            note = "현재 46개 변수, XGBoost native missing 기준"
            frames = base_frames
            added_columns: list[str] = []
        else:
            note = FEATURE_PACKS[variant]["note"]
            frames = candidate_frames
            added_columns = feature_pack_columns(variant)
        feature_columns = unique_preserve_order([*base_feature_columns, *added_columns])
        metric_row, variant_segment_rows = evaluate_variant(
            variant=variant,
            note=note,
            frames=frames,
            id_frames=id_frames,
            feature_columns=feature_columns,
            added_columns=added_columns,
        )
        metric_rows.append(metric_row)
        segment_rows.extend(variant_segment_rows)

    metrics = pd.DataFrame(metric_rows)
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


def rank_by_valid(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(
        [
            "valid_f1_at_threshold",
            "valid_pr_auc",
            "valid_precision_at_threshold",
            "test_f1_at_threshold",
        ],
        ascending=False,
    )


def rank_by_test(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(
        [
            "test_f1_at_threshold",
            "test_pr_auc",
            "test_precision_at_threshold",
            "valid_f1_at_threshold",
        ],
        ascending=False,
    )


def segment_row(segments: pd.DataFrame, variant: str, segment: str) -> dict[str, Any]:
    frame = segments.loc[segments["variant"].eq(variant) & segments["segment"].eq(segment)]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def recommendation_text(
    *,
    baseline: pd.Series,
    best_valid: pd.Series,
    valid_delta: float,
    test_delta: float,
) -> str:
    if str(best_valid["variant"]) == "baseline_43_native":
        return (
            "- Validation 기준으로도 현재 43개 baseline이 가장 안정적입니다. "
            "새 변수는 운영 반영보다 추가 후보로 보관하는 편이 안전합니다."
        )
    if valid_delta >= 0.005 and test_delta >= 0.005:
        return (
            f"- `{best_valid['variant']}`는 validation과 test에서 모두 baseline보다 좋아졌습니다. "
            "다음 모델 후보로 별도 재학습/대시보드 검증을 진행할 가치가 있습니다."
        )
    if valid_delta >= 0.005 and test_delta < 0:
        return (
            f"- `{best_valid['variant']}`는 validation에서는 좋아졌지만 test에서는 악화되었습니다. "
            "과적합 가능성이 있어 production 반영은 보류하는 편이 좋습니다."
        )
    return (
        f"- `{best_valid['variant']}`가 validation 기준 상위지만 개선 폭이 작습니다. "
        "운영 모델 교체보다 추가 OOT split 또는 오류 사례 검증이 필요합니다."
    )


def build_report(metrics: pd.DataFrame, segments: pd.DataFrame) -> str:
    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    best_valid = rank_by_valid(metrics).iloc[0]
    best_test = rank_by_test(metrics).iloc[0]
    valid_delta = float(best_valid["valid_f1_at_threshold"]) - float(
        baseline["valid_f1_at_threshold"]
    )
    test_delta = float(best_valid["test_f1_at_threshold"]) - float(baseline["test_f1_at_threshold"])
    test_only_delta = float(best_test["test_f1_at_threshold"]) - float(
        baseline["test_f1_at_threshold"]
    )
    baseline_kosdaq = segment_row(segments, "baseline_43_native", "KOSDAQ")
    best_valid_kosdaq = segment_row(segments, str(best_valid["variant"]), "KOSDAQ")

    return "\n".join(
        [
            "# Candidate Feature-Pack Experiments",
            "",
            "원본 Model V1에는 존재하지만 현재 43-feature 입력에는 빠져 있는 후보 변수를 묶음별로 추가해 비교한 실험입니다.",
            "모든 실험은 XGBoost native missing, Platt scaling, validation 기준 "
            f"`recall >= {RECALL_FLOOR:.2f}` 조건에서 precision 최대 threshold를 사용했습니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline valid/test F1: `{format_metric(baseline['valid_f1_at_threshold'])}` / "
            f"`{format_metric(baseline['test_f1_at_threshold'])}`",
            f"- Validation 기준 선택 후보: `{best_valid['variant']}` "
            f"(valid F1 `{format_metric(best_valid['valid_f1_at_threshold'])}`, "
            f"test F1 `{format_metric(best_valid['test_f1_at_threshold'])}`)",
            f"- Validation 선택 후보의 baseline 대비 변화: valid F1 `{valid_delta:+.4f}`, "
            f"test F1 `{test_delta:+.4f}`",
            f"- 참고용 test F1 최상위 후보: `{best_test['variant']}` "
            f"(test F1 `{format_metric(best_test['test_f1_at_threshold'])}`, "
            f"baseline 대비 `{test_only_delta:+.4f}`)",
            recommendation_text(
                baseline=baseline,
                best_valid=best_valid,
                valid_delta=valid_delta,
                test_delta=test_delta,
            ),
            "",
            "## 2. Validation 기준 성능 비교",
            "",
            markdown_table(
                rank_by_valid(metrics),
                [
                    ("Variant", "variant", "text"),
                    ("Added", "added_feature_count", "int"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("Valid PR", "valid_pr_auc", "metric"),
                    ("Valid P", "valid_precision_at_threshold", "metric"),
                    ("Valid R", "valid_recall_at_threshold", "metric"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 3. 참고용 Test 기준 상위 후보",
            "",
            "아래 표는 사후 점검용이며, 모델 선택 기준으로는 사용하지 않습니다.",
            "",
            markdown_table(
                rank_by_test(metrics).head(8),
                [
                    ("Variant", "variant", "text"),
                    ("Added", "added_feature_count", "int"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 4. KOSDAQ 오류 관점",
            "",
            f"- Baseline KOSDAQ FP/FN: `{format_int(baseline_kosdaq.get('false_positive'))}` / "
            f"`{format_int(baseline_kosdaq.get('false_negative'))}`",
            f"- Validation 선택 후보 KOSDAQ FP/FN: "
            f"`{format_int(best_valid_kosdaq.get('false_positive'))}` / "
            f"`{format_int(best_valid_kosdaq.get('false_negative'))}`",
            "",
            "## 5. 후보 변수 묶음",
            "",
            markdown_table(
                metrics.loc[~metrics["variant"].eq("baseline_43_native"), :]
                .sort_values("variant")
                .assign(
                    added_features=lambda frame: frame["added_features"].str.replace(", ", "<br>")
                ),
                [
                    ("Variant", "variant", "text"),
                    ("Note", "note", "text"),
                    ("Features", "added_features", "text"),
                ],
            ),
            "",
            "## 6. 해석 원칙",
            "",
            "- 실제 모델 선택은 validation 성능만 기준으로 합니다.",
            "- test 기준 최상위 후보는 사후 참고용이며, 운영 반영 전 추가 OOT 검증이 필요합니다.",
            "- 절대금액 원값은 기존 오류 분석에서 FP를 키울 수 있어 이번 실험에서는 해석 가능한 비율/플래그/변화량 후보를 우선했습니다.",
        ]
    )


def write_outputs(metrics: pd.DataFrame, segments: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "candidate_feature_pack_metrics.csv"
    segments_path = output_dir / "candidate_feature_pack_segment_metrics.csv"
    report_path = output_dir / "candidate_feature_pack_report.md"
    summary_path = output_dir / "candidate_feature_pack_summary.json"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(metrics, segments), encoding="utf-8")

    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    best_valid = rank_by_valid(metrics).iloc[0]
    best_test = rank_by_test(metrics).iloc[0]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_source": str(RAW_PATH.relative_to(ROOT)),
        "threshold_policy": f"max precision with validation recall >= {RECALL_FLOOR:.2f}",
        "baseline": baseline.to_dict(),
        "best_by_validation": best_valid.to_dict(),
        "best_by_test_reference_only": best_test.to_dict(),
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
    raw = read_raw_features(args.raw_path)
    metrics, segments = run_experiments(train, valid, test, id_frames, raw)
    write_outputs(metrics, segments, args.output_dir)
    best_valid = rank_by_valid(metrics).iloc[0]
    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    print(
        json.dumps(
            {
                "best_by_validation": best_valid["variant"],
                "best_valid_f1": float(best_valid["valid_f1_at_threshold"]),
                "best_test_f1": float(best_valid["test_f1_at_threshold"]),
                "baseline_valid_f1": float(baseline["valid_f1_at_threshold"]),
                "baseline_test_f1": float(baseline["test_f1_at_threshold"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
