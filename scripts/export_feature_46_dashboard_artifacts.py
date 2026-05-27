from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xgboost import XGBClassifier

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas.artifacts.dashboard_exports import (  # noqa: E402
    build_company_latest,
    build_company_universe,
    build_feature_dictionary,
    build_global_shap_reference,
    build_industry_latest_summary,
    build_industry_shap_summary,
    build_industry_year_summary,
    build_peer_percentiles,
    read_json,
    risk_band,
    sanitize_feature_name,
    write_json,
)
from cas.modeling.calibration import (  # noqa: E402
    DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR,
    apply_probability_calibration,
    build_calibration_summary,
    choose_max_precision_threshold_at_recall,
    choose_tuned_threshold,
    fit_platt_calibration,
)
from cas.modeling.fn_rescue import (  # noqa: E402
    FN_RESCUE_DEFAULT_MIN_GROUPS,
    FN_RESCUE_DEFAULT_PROB_CEILING,
    FN_RESCUE_DEFAULT_SCORE_THRESHOLD,
    FN_RESCUE_POLICY_NAME,
    FN_RESCUE_RAW_COLUMNS,
    FN_RESCUE_SCORE_COLUMNS,
    add_manufacturing_fn_rescue_scores,
    build_manufacturing_fn_rescue_gate,
)

INPUT_DIR = ROOT / "data" / "input" / "credit_46_features"
METADATA_PATH = INPUT_DIR / "feature_46_dictionary_metadata.json"
RAW_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "dashboard" / "feature_46_mvp"
MODEL_OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost"
TUNED_THRESHOLD_RECALL_FLOOR = DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR
TUNED_THRESHOLD_SELECTION_RULE = "valid_max_precision_with_recall_ge_0.85"
JOIN_KEYS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
STAGE2_REVIEW_AUX_BASE_FEATURES = [
    "delta_accruals_ratio",
    "is_3y_consecutive_operating_loss",
]
STAGE2_REVIEW_MACRO_REGIME_FEATURES = [
    "market_spread_diff",
    "spec_spread_diff",
    "base_rate_diff",
    "treasury_3y_diff",
    "usd_krw_diff",
]
STAGE2_REVIEW_RAW_FEATURES = [
    *STAGE2_REVIEW_AUX_BASE_FEATURES,
    *[column for column in FN_RESCUE_RAW_COLUMNS if column not in STAGE2_REVIEW_AUX_BASE_FEATURES],
    *STAGE2_REVIEW_MACRO_REGIME_FEATURES,
]
STAGE2_REVIEW_FEATURES = [
    *STAGE2_REVIEW_AUX_BASE_FEATURES,
    *FN_RESCUE_SCORE_COLUMNS,
    *[column for column in FN_RESCUE_RAW_COLUMNS if column not in STAGE2_REVIEW_AUX_BASE_FEATURES],
    *STAGE2_REVIEW_MACRO_REGIME_FEATURES,
]
STAGE2_REVIEW_AUX_FEATURE_SET = "full_review_trigger_73"
STAGE2_REVIEW_AUX_PROB_COLUMN = "prob_speculative_stage2_review_aux"
STAGE2_REVIEW_AUX_RAW_PROB_COLUMN = "prob_speculative_stage2_review_aux_raw"
STAGE2_REVIEW_AUX_CALIBRATION_COLUMN = "probability_stage2_review_aux_calibration_method"
STAGE2_REVIEW_AUX_THRESHOLD_COLUMN = "threshold_stage2_review_aux"
STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN = "threshold_stage2_review_aux_it_services_review"
STAGE2_REVIEW_AUX_LABEL_COLUMN = "pred_label_stage2_review_aux_tuned"
STAGE2_REVIEW_AUX_TRIGGER_POLICY = (
    "stage1_or_stage2_review_aux_or_it_services_or_manufacturing_fn_rescue"
)
STAGE2_IT_SERVICES_RECALL_FLOOR = 0.90
STAGE2_OVERWARNING_FILTER_FEATURES = [
    "cashflow_debt_stress_score",
    "working_capital_shock_score",
    "liquidity_buffer_gap_score",
    "profit_quality_deterioration_score",
    "industry_relative_stress_score",
]
STAGE2_OVERWARNING_COMPONENT_COLUMNS = [
    "interest_burden_ratio",
    "total_borrowings_ratio",
    "rolling_3y_cv_ocf_to_total_borrowings",
    "short_term_borrowings_share",
    "ocf_to_total_borrowings",
    "ocf_to_total_liabilities",
    "cashflow_coverage_ratio",
    "ar_days_diff",
    "inventory_days_diff",
    "ap_days_diff",
    "current_ratio",
    "cash_ratio",
    "current_ratio_diff",
    "cash_ratio_diff",
    "accruals_ratio",
    "delta_accruals_ratio",
    "ocf_to_sales",
    "net_margin_diff",
    "operating_margin_diff",
    "equity_ratio",
    "debt_ratio",
    "capital_impairment_ratio",
    "net_margin",
    "interest_coverage_ratio",
]

SCENARIO_PRESETS: dict[str, dict[str, float]] = {
    "base": {},
    "mild_stress": {
        "spec_spread": 0.50,
        "cash_ratio": -0.05,
        "net_margin": -0.01,
    },
    "severe_stress": {
        "spec_spread": 1.00,
        "cash_ratio": -0.10,
        "net_margin": -0.02,
        "short_term_borrowings_share": 0.05,
        "capital_impairment_ratio": 0.05,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official 46-feature model artifacts for the dashboard."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-output-dir", type=Path, default=MODEL_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-shap", type=int, default=10)
    return parser.parse_args()


def metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "logloss": float(log_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def _normalized_join_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.loc[:, JOIN_KEYS].copy()
    normalized["stock_code"] = normalized["stock_code"].astype(str)
    for column in ["fiscal_year", "eval_year"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(int)
    return normalized


def attach_stage2_review_features(
    *,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    raw_path: Path,
) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"stock_code": str})
    missing_columns = [column for column in STAGE2_REVIEW_RAW_FEATURES if column not in raw.columns]
    if missing_columns:
        raise KeyError(
            f"Stage 2 review feature columns are missing from raw data: {missing_columns}"
        )
    duplicates = raw.duplicated(JOIN_KEYS).sum()
    if duplicates:
        raise ValueError(f"Raw Model V1 has duplicate rows for join keys: {duplicates}")

    raw_with_scores = add_manufacturing_fn_rescue_scores(raw)
    raw_subset = raw_with_scores.loc[:, [*JOIN_KEYS, *STAGE2_REVIEW_FEATURES]].copy()
    raw_subset["stock_code"] = raw_subset["stock_code"].astype(str)
    for column in ["fiscal_year", "eval_year"]:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="raise").astype(int)
    for column in STAGE2_REVIEW_FEATURES:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="coerce")

    output: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        join_keys = _normalized_join_frame(id_frames[split].reset_index(drop=True))
        joined = join_keys.merge(raw_subset, on=JOIN_KEYS, how="left", indicator=True)
        unmatched = int(joined["_merge"].ne("both").sum())
        if unmatched:
            raise ValueError(
                f"{split} split has unmatched Stage 2 review feature rows: {unmatched}"
            )
        split_frame = frame.reset_index(drop=True).copy()
        for column in STAGE2_REVIEW_FEATURES:
            split_frame[column] = joined[column]
        output[split] = split_frame
    return output


def _stage2_risk_percentile(
    frame: pd.DataFrame,
    column: str,
    *,
    high_value_is_risk: bool = True,
    absolute_value: bool = False,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if absolute_value:
        values = values.abs()
    ranks = values.groupby([frame["fiscal_year"], frame["industry_macro_category"]]).rank(
        pct=True,
        method="average",
    )
    if high_value_is_risk:
        return ranks
    return 1.0 - ranks


def _stage2_mean_score(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.loc[:, columns].mean(axis=1, skipna=True)


def add_stage2_overwarning_filter_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Create composite scores used only as Stage 2 over-warning review signals."""
    missing_columns = [
        column for column in STAGE2_OVERWARNING_COMPONENT_COLUMNS if column not in raw.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Stage 2 over-warning component columns are missing from raw data: {missing_columns}"
        )

    output = raw.copy()
    output["_risk_interest_burden"] = _stage2_risk_percentile(output, "interest_burden_ratio")
    output["_risk_total_borrowings"] = _stage2_risk_percentile(output, "total_borrowings_ratio")
    output["_risk_ocf_borrowing_volatility"] = _stage2_risk_percentile(
        output,
        "rolling_3y_cv_ocf_to_total_borrowings",
    )
    output["_risk_short_term_borrowing_share"] = _stage2_risk_percentile(
        output,
        "short_term_borrowings_share",
    )
    output["_risk_low_ocf_to_borrowings"] = _stage2_risk_percentile(
        output,
        "ocf_to_total_borrowings",
        high_value_is_risk=False,
    )
    output["_risk_low_ocf_to_liabilities"] = _stage2_risk_percentile(
        output,
        "ocf_to_total_liabilities",
        high_value_is_risk=False,
    )
    output["_risk_low_cashflow_coverage"] = _stage2_risk_percentile(
        output,
        "cashflow_coverage_ratio",
        high_value_is_risk=False,
    )
    output["cashflow_debt_stress_score"] = _stage2_mean_score(
        output,
        [
            "_risk_interest_burden",
            "_risk_total_borrowings",
            "_risk_ocf_borrowing_volatility",
            "_risk_short_term_borrowing_share",
            "_risk_low_ocf_to_borrowings",
            "_risk_low_ocf_to_liabilities",
            "_risk_low_cashflow_coverage",
        ],
    )

    output["_risk_ar_days_shock"] = _stage2_risk_percentile(
        output,
        "ar_days_diff",
        absolute_value=True,
    )
    output["_risk_inventory_days_shock"] = _stage2_risk_percentile(
        output,
        "inventory_days_diff",
        absolute_value=True,
    )
    output["_risk_ap_days_shock"] = _stage2_risk_percentile(
        output,
        "ap_days_diff",
        absolute_value=True,
    )
    output["working_capital_shock_score"] = _stage2_mean_score(
        output,
        ["_risk_ar_days_shock", "_risk_inventory_days_shock", "_risk_ap_days_shock"],
    )

    output["_risk_low_current_ratio"] = _stage2_risk_percentile(
        output,
        "current_ratio",
        high_value_is_risk=False,
    )
    output["_risk_low_cash_ratio"] = _stage2_risk_percentile(
        output,
        "cash_ratio",
        high_value_is_risk=False,
    )
    output["_risk_current_ratio_drop"] = _stage2_risk_percentile(
        output,
        "current_ratio_diff",
        high_value_is_risk=False,
    )
    output["_risk_cash_ratio_drop"] = _stage2_risk_percentile(
        output,
        "cash_ratio_diff",
        high_value_is_risk=False,
    )
    output["liquidity_buffer_gap_score"] = _stage2_mean_score(
        output,
        [
            "_risk_low_current_ratio",
            "_risk_low_cash_ratio",
            "_risk_current_ratio_drop",
            "_risk_cash_ratio_drop",
            "_risk_short_term_borrowing_share",
        ],
    )

    output["_risk_accruals"] = _stage2_risk_percentile(output, "accruals_ratio")
    output["_risk_delta_accruals_abs"] = _stage2_risk_percentile(
        output,
        "delta_accruals_ratio",
        absolute_value=True,
    )
    output["_risk_low_ocf_to_sales"] = _stage2_risk_percentile(
        output,
        "ocf_to_sales",
        high_value_is_risk=False,
    )
    output["_risk_net_margin_drop"] = _stage2_risk_percentile(
        output,
        "net_margin_diff",
        high_value_is_risk=False,
    )
    output["_risk_operating_margin_drop"] = _stage2_risk_percentile(
        output,
        "operating_margin_diff",
        high_value_is_risk=False,
    )
    output["profit_quality_deterioration_score"] = _stage2_mean_score(
        output,
        [
            "_risk_accruals",
            "_risk_delta_accruals_abs",
            "_risk_low_ocf_to_sales",
            "_risk_net_margin_drop",
            "_risk_operating_margin_drop",
        ],
    )

    output["_risk_low_equity_ratio"] = _stage2_risk_percentile(
        output,
        "equity_ratio",
        high_value_is_risk=False,
    )
    output["_risk_debt_ratio"] = _stage2_risk_percentile(output, "debt_ratio")
    output["_risk_capital_impairment"] = _stage2_risk_percentile(
        output,
        "capital_impairment_ratio",
    )
    output["_risk_low_net_margin"] = _stage2_risk_percentile(
        output,
        "net_margin",
        high_value_is_risk=False,
    )
    output["_risk_low_interest_coverage"] = _stage2_risk_percentile(
        output,
        "interest_coverage_ratio",
        high_value_is_risk=False,
    )
    output["industry_relative_stress_score"] = _stage2_mean_score(
        output,
        [
            "_risk_low_equity_ratio",
            "_risk_debt_ratio",
            "_risk_capital_impairment",
            "_risk_low_net_margin",
            "_risk_low_interest_coverage",
            "_risk_low_current_ratio",
            "_risk_low_cash_ratio",
        ],
    )
    return output


def attach_stage2_overwarning_filter_features(
    *,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    raw_path: Path,
) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"stock_code": str})
    duplicates = raw.duplicated(JOIN_KEYS).sum()
    if duplicates:
        raise ValueError(f"Raw Model V1 has duplicate rows for join keys: {duplicates}")

    raw_with_scores = add_stage2_overwarning_filter_scores(raw)
    raw_subset = raw_with_scores.loc[
        :,
        [*JOIN_KEYS, *STAGE2_OVERWARNING_FILTER_FEATURES],
    ].copy()
    raw_subset["stock_code"] = raw_subset["stock_code"].astype(str)
    for column in ["fiscal_year", "eval_year"]:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="raise").astype(int)
    for column in STAGE2_OVERWARNING_FILTER_FEATURES:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="coerce")

    output: dict[str, pd.DataFrame] = {}
    for split, frame in frames.items():
        join_keys = _normalized_join_frame(id_frames[split].reset_index(drop=True))
        joined = join_keys.merge(raw_subset, on=JOIN_KEYS, how="left", indicator=True)
        unmatched = int(joined["_merge"].ne("both").sum())
        if unmatched:
            raise ValueError(
                f"{split} split has unmatched Stage 2 over-warning feature rows: {unmatched}"
            )
        split_frame = frame.reset_index(drop=True).copy()
        for column in STAGE2_OVERWARNING_FILTER_FEATURES:
            split_frame[column] = joined[column]
        output[split] = split_frame
    return output


def attach_manufacturing_fn_rescue_scores(
    prediction_scores: pd.DataFrame,
    *,
    raw_path: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, encoding="utf-8-sig", dtype={"stock_code": str})
    missing_columns = [column for column in FN_RESCUE_RAW_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise KeyError(f"FN rescue component columns are missing from raw data: {missing_columns}")
    duplicates = raw.duplicated(JOIN_KEYS).sum()
    if duplicates:
        raise ValueError(f"Raw Model V1 has duplicate rows for join keys: {duplicates}")

    raw_with_scores = add_manufacturing_fn_rescue_scores(raw)
    raw_subset = raw_with_scores.loc[:, [*JOIN_KEYS, *FN_RESCUE_SCORE_COLUMNS]].copy()
    raw_subset["stock_code"] = raw_subset["stock_code"].astype(str)
    for column in ["fiscal_year", "eval_year"]:
        raw_subset[column] = pd.to_numeric(raw_subset[column], errors="raise").astype(int)
    output = prediction_scores.copy()
    output_keys = _normalized_join_frame(output)
    joined = output_keys.merge(raw_subset, on=JOIN_KEYS, how="left", indicator=True)
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"Prediction scores have unmatched FN rescue score rows: {unmatched}")
    for column in FN_RESCUE_SCORE_COLUMNS:
        output[column] = joined[column].to_numpy()
    return output


def build_stage2_review_probabilities(
    *,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    raw_path: Path,
    base_model_features: list[str],
    seed: int,
) -> dict[str, object]:
    from xgboost import XGBClassifier

    review_frames = attach_stage2_review_features(
        frames=frames,
        id_frames=id_frames,
        raw_path=raw_path,
    )
    review_features = [*base_model_features, *STAGE2_REVIEW_FEATURES]
    y_train = review_frames["train"]["is_speculative"].astype(int)
    y_valid = review_frames["valid"]["is_speculative"].astype(int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / pos) if pos else 1.0
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
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
    )
    model.fit(
        review_frames["train"].loc[:, review_features],
        y_train,
        eval_set=[(review_frames["valid"].loc[:, review_features], y_valid)],
        verbose=False,
    )
    raw_probabilities = {
        split: model.predict_proba(review_frames[split].loc[:, review_features])[:, 1]
        for split in ["train", "valid", "test"]
    }
    calibration = fit_platt_calibration(y_valid, raw_probabilities["valid"])
    probabilities = {
        split: apply_probability_calibration(raw_probabilities[split], calibration)
        for split in ["train", "valid", "test"]
    }
    default_threshold = choose_tuned_threshold(y_valid, probabilities["valid"])
    valid_ids = id_frames["valid"].reset_index(drop=True)
    valid_it_mask = valid_ids["industry_macro_category"].astype(str).eq("it_services")
    if valid_it_mask.any():
        it_threshold = choose_max_precision_threshold_at_recall(
            y_valid.loc[valid_it_mask.to_numpy()],
            probabilities["valid"][valid_it_mask.to_numpy()],
            STAGE2_IT_SERVICES_RECALL_FLOOR,
        )
    else:
        it_threshold = default_threshold
    return {
        "probabilities": probabilities,
        "raw_probabilities": raw_probabilities,
        "default_threshold": default_threshold,
        "it_services_threshold": it_threshold,
        "feature_columns": review_features,
        "calibration_method": calibration["method"],
    }


def build_stage2_overwarning_filter_probabilities(
    *,
    frames: dict[str, pd.DataFrame],
    id_frames: dict[str, pd.DataFrame],
    raw_path: Path,
    base_model_features: list[str],
    seed: int,
) -> dict[str, object]:
    from xgboost import XGBClassifier

    filter_frames = attach_stage2_overwarning_filter_features(
        frames=frames,
        id_frames=id_frames,
        raw_path=raw_path,
    )
    filter_features = [*base_model_features, *STAGE2_OVERWARNING_FILTER_FEATURES]
    y_train = filter_frames["train"]["is_speculative"].astype(int)
    y_valid = filter_frames["valid"]["is_speculative"].astype(int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / pos) if pos else 1.0
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
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
    )
    model.fit(
        filter_frames["train"].loc[:, filter_features],
        y_train,
        eval_set=[(filter_frames["valid"].loc[:, filter_features], y_valid)],
        verbose=False,
    )
    raw_probabilities = {
        split: model.predict_proba(filter_frames[split].loc[:, filter_features])[:, 1]
        for split in ["train", "valid", "test"]
    }
    calibration = fit_platt_calibration(y_valid, raw_probabilities["valid"])
    probabilities = {
        split: apply_probability_calibration(raw_probabilities[split], calibration)
        for split in ["train", "valid", "test"]
    }
    threshold = choose_tuned_threshold(y_valid, probabilities["valid"])
    return {
        "probabilities": probabilities,
        "raw_probabilities": raw_probabilities,
        "threshold": threshold,
        "feature_columns": filter_features,
        "calibration_method": calibration["method"],
    }


def add_stage2_review_signals(
    prediction_scores: pd.DataFrame,
    *,
    review_probabilities: dict[str, np.ndarray],
    review_raw_probabilities: dict[str, np.ndarray],
    review_default_threshold: float,
    review_it_services_threshold: float,
    review_calibration_method: str,
) -> pd.DataFrame:
    output = prediction_scores.copy()
    for split, split_probabilities in review_probabilities.items():
        split_mask = output["split"].eq(split)
        output.loc[split_mask, STAGE2_REVIEW_AUX_PROB_COLUMN] = split_probabilities
        output.loc[split_mask, STAGE2_REVIEW_AUX_RAW_PROB_COLUMN] = review_raw_probabilities[split]

    stage1_risk = output["pred_label_tuned"].astype(int).eq(1)
    aux_risk = output[STAGE2_REVIEW_AUX_PROB_COLUMN].astype(float).ge(review_default_threshold)
    it_services_review = output["industry_macro_category"].astype(str).eq("it_services") & output[
        STAGE2_REVIEW_AUX_PROB_COLUMN
    ].astype(float).ge(review_it_services_threshold)
    secondary_trigger = (~stage1_risk) & (aux_risk | it_services_review)
    output[STAGE2_REVIEW_AUX_CALIBRATION_COLUMN] = review_calibration_method
    output[STAGE2_REVIEW_AUX_THRESHOLD_COLUMN] = review_default_threshold
    output[STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN] = review_it_services_threshold
    output[STAGE2_REVIEW_AUX_LABEL_COLUMN] = aux_risk.astype(int)
    output["stage2_review_trigger"] = stage1_risk | secondary_trigger
    output["stage2_secondary_trigger"] = secondary_trigger
    output["stage2_review_priority"] = np.select(
        [
            stage1_risk,
            (~stage1_risk) & aux_risk,
            (~stage1_risk) & it_services_review,
        ],
        ["high", "medium", "watch"],
        default="none",
    )
    output["trigger_reason_code"] = np.select(
        [
            stage1_risk & aux_risk,
            stage1_risk,
            (~stage1_risk) & aux_risk,
            (~stage1_risk) & it_services_review,
        ],
        [
            "stage1_and_stage2_review_aux_risk",
            "stage1_model_risk",
            "stage2_review_aux_only",
            "stage2_review_aux_it_services_low_threshold",
        ],
        default="none",
    )
    output["trigger_reason"] = np.select(
        [
            stage1_risk & aux_risk,
            stage1_risk,
            (~stage1_risk) & aux_risk,
            (~stage1_risk) & it_services_review,
        ],
        [
            "1차 공식 모델과 보조 변수셋이 모두 위험 기준선을 넘겨 위원회 검토 대상으로 분류했습니다.",
            "1차 모델이 위험 기준선을 넘겨 위원회 검토 대상으로 분류했습니다.",
            "1차 공식 모델은 투자적격이나 보조 변수셋이 위험 기준선을 넘어 추가 검토 대상으로 올렸습니다.",
            "IT서비스 업종 보조 기준선을 넘어 추가 검토 대상으로 올렸습니다.",
        ],
        default="추가 위원회 검토 트리거 없음",
    )
    output["trigger_policy"] = STAGE2_REVIEW_AUX_TRIGGER_POLICY
    return output


def add_manufacturing_fn_rescue_signals(
    prediction_scores: pd.DataFrame,
    *,
    raw_path: Path,
) -> pd.DataFrame:
    output = attach_manufacturing_fn_rescue_scores(prediction_scores, raw_path=raw_path)
    rescue_trigger = build_manufacturing_fn_rescue_gate(
        output,
        probability_column="prob_speculative",
        prediction_column="pred_label_tuned",
        probability_ceiling=FN_RESCUE_DEFAULT_PROB_CEILING,
        score_threshold=FN_RESCUE_DEFAULT_SCORE_THRESHOLD,
        min_group_count=FN_RESCUE_DEFAULT_MIN_GROUPS,
    )
    existing_trigger = output["stage2_review_trigger"].astype(bool)
    existing_secondary = output["stage2_secondary_trigger"].astype(bool)
    output["stage2_fn_rescue_trigger"] = rescue_trigger.astype(bool)
    output["stage2_review_trigger"] = existing_trigger | rescue_trigger
    output["stage2_secondary_trigger"] = existing_secondary | rescue_trigger
    output["fn_rescue_probability_ceiling"] = FN_RESCUE_DEFAULT_PROB_CEILING
    output["fn_rescue_score_threshold"] = FN_RESCUE_DEFAULT_SCORE_THRESHOLD
    output["fn_rescue_min_group_count"] = FN_RESCUE_DEFAULT_MIN_GROUPS
    output["fn_rescue_policy"] = FN_RESCUE_POLICY_NAME

    rescue_only = rescue_trigger & output["trigger_reason_code"].astype(str).eq("none")
    rescue_overlap = rescue_trigger & ~rescue_only
    output.loc[rescue_only, "stage2_review_priority"] = "watch"
    output.loc[rescue_only, "trigger_reason_code"] = "manufacturing_fn_rescue_gate"
    output.loc[rescue_only, "trigger_reason"] = (
        "1차 공식 모델은 정상으로 봤지만 KOSDAQ 제조업 FN rescue gate가 "
        "재무 스트레스 조합 신호를 감지해 Stage2 에이전트 검토 대상으로 올렸습니다."
    )
    output.loc[rescue_overlap, "trigger_reason_code"] = (
        "stage2_review_aux_and_manufacturing_fn_rescue"
    )
    output.loc[rescue_overlap, "trigger_reason"] = (
        "보조 변수셋과 KOSDAQ 제조업 FN rescue gate가 모두 추가 검토 신호를 냈습니다."
    )
    output["trigger_policy"] = STAGE2_REVIEW_AUX_TRIGGER_POLICY
    return output


def add_stage2_overwarning_filter_signals(
    prediction_scores: pd.DataFrame,
    *,
    filter_probabilities: dict[str, np.ndarray],
    filter_raw_probabilities: dict[str, np.ndarray],
    filter_threshold: float,
    filter_calibration_method: str,
) -> pd.DataFrame:
    output = prediction_scores.copy()
    for split, split_probabilities in filter_probabilities.items():
        split_mask = output["split"].eq(split)
        output.loc[split_mask, "prob_speculative_overwarning_filter"] = split_probabilities
        output.loc[split_mask, "prob_speculative_overwarning_filter_raw"] = (
            filter_raw_probabilities[split]
        )

    stage1_risk = output["pred_label_tuned"].astype(int).eq(1)
    filter_risk = output["prob_speculative_overwarning_filter"].astype(float).ge(filter_threshold)
    overwarning_candidate = stage1_risk & (~filter_risk)
    output["probability_overwarning_filter_calibration_method"] = filter_calibration_method
    output["threshold_overwarning_filter"] = filter_threshold
    output["pred_label_overwarning_filter_tuned"] = filter_risk.astype(int)
    output["stage2_overwarning_filter_candidate"] = overwarning_candidate
    output["overwarning_filter_reason_code"] = np.where(
        overwarning_candidate,
        "stage1_risk_but_composite_filter_normal",
        "none",
    )
    output["overwarning_filter_reason"] = np.where(
        overwarning_candidate,
        (
            "1차 모델은 위험으로 판단했지만 조합형 재무 스트레스 필터는 기준선 미만입니다. "
            "2차 위원회에서 과민 경고 가능성과 완화 요인을 함께 검토합니다."
        ),
        "과민 경고 보조필터 특이 신호 없음",
    )
    output["overwarning_filter_policy"] = (
        "stage1_risk_and_composite_filter_below_threshold_for_committee_mitigation_review"
    )
    return output


def build_stage2_review_signal_summary(prediction_scores: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split, split_frame in prediction_scores.groupby("split"):
        y_true = split_frame["is_speculative"].astype(int)
        trigger = split_frame["stage2_review_trigger"].astype(bool)
        secondary_trigger = split_frame["stage2_secondary_trigger"].astype(bool)
        fn_rescue_trigger = (
            split_frame["stage2_fn_rescue_trigger"].astype(bool)
            if "stage2_fn_rescue_trigger" in split_frame
            else pd.Series(False, index=split_frame.index)
        )
        true_positive = int((trigger & y_true.eq(1)).sum())
        false_positive = int((trigger & y_true.eq(0)).sum())
        false_negative = int((~trigger & y_true.eq(1)).sum())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        summary[str(split)] = {
            "rows": len(split_frame),
            "stage2_review_trigger_count": int(trigger.sum()),
            "stage2_secondary_trigger_count": int(secondary_trigger.sum()),
            "stage2_secondary_true_risk_count": int((secondary_trigger & y_true.eq(1)).sum()),
            "stage2_secondary_normal_count": int((secondary_trigger & y_true.eq(0)).sum()),
            "stage2_fn_rescue_trigger_count": int(fn_rescue_trigger.sum()),
            "stage2_fn_rescue_true_risk_count": int((fn_rescue_trigger & y_true.eq(1)).sum()),
            "stage2_fn_rescue_normal_count": int((fn_rescue_trigger & y_true.eq(0)).sum()),
            "trigger_precision": true_positive / precision_denominator
            if precision_denominator
            else 0.0,
            "trigger_recall": true_positive / recall_denominator if recall_denominator else 0.0,
            "trigger_reason_counts": split_frame["trigger_reason_code"].value_counts().to_dict(),
        }
    return summary


def build_stage2_overwarning_filter_summary(prediction_scores: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split, split_frame in prediction_scores.groupby("split"):
        y_true = split_frame["is_speculative"].astype(int)
        stage1_risk = split_frame["pred_label_tuned"].astype(int).eq(1)
        candidate = split_frame["stage2_overwarning_filter_candidate"].astype(bool)
        stage1_risk_count = int(stage1_risk.sum())
        candidate_count = int(candidate.sum())
        candidate_false_positive = int((candidate & y_true.eq(0)).sum())
        candidate_true_positive = int((candidate & y_true.eq(1)).sum())
        summary[str(split)] = {
            "rows": len(split_frame),
            "stage1_risk_count": stage1_risk_count,
            "overwarning_filter_candidate_count": candidate_count,
            "candidate_false_positive_count": candidate_false_positive,
            "candidate_true_positive_count": candidate_true_positive,
            "candidate_share_among_stage1_risk": candidate_count / stage1_risk_count
            if stage1_risk_count
            else 0.0,
            "candidate_precision_for_overwarning": candidate_false_positive / candidate_count
            if candidate_count
            else 0.0,
        }
    return summary


def build_prediction_scores(
    id_frames: dict[str, pd.DataFrame],
    probabilities: dict[str, np.ndarray],
    tuned_threshold: float,
    y_frames: dict[str, pd.Series],
    raw_probabilities: dict[str, np.ndarray] | None = None,
    calibration_method: str | None = None,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for split, id_frame in id_frames.items():
        scored = id_frame.copy()
        scored["split"] = split
        scored["is_speculative"] = y_frames[split].astype(int).to_numpy()
        if raw_probabilities is not None:
            scored["prob_speculative_raw"] = raw_probabilities[split]
        scored["probability_calibration_method"] = calibration_method or "none"
        scored["prob_speculative"] = probabilities[split]
        scored["pred_label_0_5"] = (scored["prob_speculative"] >= 0.5).astype(int)
        scored["pred_label_tuned"] = (scored["prob_speculative"] >= tuned_threshold).astype(int)
        scored["predicted_label"] = scored["pred_label_tuned"]
        scored["threshold_default"] = 0.5
        scored["threshold_tuned"] = tuned_threshold
        scored["threshold"] = tuned_threshold
        scored["risk_band"] = scored["prob_speculative"].map(risk_band)
        chunks.append(scored)
    return pd.concat(chunks, ignore_index=True)


def build_local_shap(
    scored_frame: pd.DataFrame,
    master: pd.DataFrame,
    shap_values_by_split: dict[str, np.ndarray],
    model_feature_names: list[str],
    source_feature_mapping: dict[str, str],
    source_features: list[str],
    *,
    top_k_shap: int,
) -> pd.DataFrame:
    master_keyed = master.set_index(
        ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
    )
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, feature_name in enumerate(model_feature_names):
        grouped_feature = sanitize_feature_name(feature_name, source_feature_mapping)
        grouped_indices[grouped_feature].append(index)

    rows: list[dict[str, Any]] = []
    for split, shap_values in shap_values_by_split.items():
        split_frame = scored_frame.loc[scored_frame["split"] == split].reset_index(drop=True)
        grouped_shap = np.zeros((shap_values.shape[0], len(source_features)), dtype=float)
        for feature_index, feature in enumerate(source_features):
            source_indices = grouped_indices.get(feature, [])
            if source_indices:
                grouped_shap[:, feature_index] = shap_values[:, source_indices].sum(axis=1)

        for row_index in range(grouped_shap.shape[0]):
            row_values = grouped_shap[row_index]
            top_indices = np.argsort(np.abs(row_values))[::-1][:top_k_shap]
            score_row = split_frame.iloc[row_index]
            key = (
                score_row["market"],
                score_row["stock_code"],
                score_row["corp_name"],
                score_row["fiscal_year"],
                score_row["eval_year"],
            )
            master_row = master_keyed.loc[key]
            for rank, feature_index in enumerate(top_indices, start=1):
                feature = source_features[feature_index]
                shap_value = float(row_values[feature_index])
                rows.append(
                    {
                        "market": score_row["market"],
                        "stock_code": score_row["stock_code"],
                        "corp_name": score_row["corp_name"],
                        "fiscal_year": score_row["fiscal_year"],
                        "eval_year": score_row["eval_year"],
                        "industry_macro_category": score_row["industry_macro_category"],
                        "firm_size_group": score_row["firm_size_group"],
                        "split": split,
                        "is_speculative": int(score_row["is_speculative"]),
                        "prob_speculative": float(score_row["prob_speculative"]),
                        "feature": feature,
                        "rank": rank,
                        "shap_value": shap_value,
                        "abs_shap": abs(shap_value),
                        "direction": "increase_risk" if shap_value > 0 else "decrease_risk",
                        "feature_value": master_row.get(feature),
                    }
                )
    return pd.DataFrame(rows)


def build_model_summary(
    train_y: pd.Series,
    valid_y: pd.Series,
    test_y: pd.Series,
    valid_prob: np.ndarray,
    test_prob: np.ndarray,
    tuned_threshold: float,
    calibration_summary: dict[str, object],
    valid_raw_prob: np.ndarray,
    test_raw_prob: np.ndarray,
) -> dict[str, object]:
    return {
        "selected_model": "feature_46_xgboost",
        "dataset_name": "credit_46_features",
        "test_overall_models": [
            {
                "model": "feature_46_xgboost",
                "rows": len(test_y),
                "positive_rows": int(test_y.sum()),
                "positive_rate": float(test_y.mean()),
                "pr_auc": metrics(test_y, test_prob, 0.5)["pr_auc"],
                "roc_auc": metrics(test_y, test_prob, 0.5)["roc_auc"],
                "precision_at_0_5": metrics(test_y, test_prob, 0.5)["precision"],
                "recall_at_0_5": metrics(test_y, test_prob, 0.5)["recall"],
            }
        ],
        "xgboost_thresholds": [
            {
                "threshold_type": "default",
                "threshold": 0.5,
                "selection_rule": "fixed_0_5",
                "test_precision": metrics(test_y, test_prob, 0.5)["precision"],
                "test_recall": metrics(test_y, test_prob, 0.5)["recall"],
                "test_f1": metrics(test_y, test_prob, 0.5)["f1"],
                "test_pr_auc": metrics(test_y, test_prob, 0.5)["pr_auc"],
                "test_roc_auc": metrics(test_y, test_prob, 0.5)["roc_auc"],
            },
            {
                "threshold_type": "tuned",
                "threshold": tuned_threshold,
                "selection_rule": TUNED_THRESHOLD_SELECTION_RULE,
                "test_precision": metrics(test_y, test_prob, tuned_threshold)["precision"],
                "test_recall": metrics(test_y, test_prob, tuned_threshold)["recall"],
                "test_f1": metrics(test_y, test_prob, tuned_threshold)["f1"],
                "test_pr_auc": metrics(test_y, test_prob, tuned_threshold)["pr_auc"],
                "test_roc_auc": metrics(test_y, test_prob, tuned_threshold)["roc_auc"],
            },
        ],
        "prediction_artifacts_ready": True,
        "prediction_artifacts_note": (
            "Per-company prediction probabilities, local SHAP, and industry summaries are "
            "exported from the credit_46_features split."
        ),
        "probability_calibration": calibration_summary,
        "split_summary": {
            "train": {"rows": len(train_y), "positive_rate": float(train_y.mean())},
            "valid": {"rows": len(valid_y), "positive_rate": float(valid_y.mean())},
            "test": {"rows": len(test_y), "positive_rate": float(test_y.mean())},
        },
        "valid_default_0_5": metrics(valid_y, valid_prob, 0.5),
        "test_default_0_5": metrics(test_y, test_prob, 0.5),
        "test_tuned": metrics(test_y, test_prob, tuned_threshold),
        "raw_valid_default_0_5": metrics(valid_y, valid_raw_prob, 0.5),
        "raw_test_default_0_5": metrics(test_y, test_raw_prob, 0.5),
    }


def save_model_artifacts(
    *,
    model_output_dir: Path,
    model: XGBClassifier,
    model_features: list[str],
    fill_values: pd.Series,
    tuned_threshold: float,
    source_features: list[str],
    calibration_summary: dict[str, object],
) -> None:
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model.get_booster().save_model(model_output_dir / "xgboost_model.json")

    write_json(
        model_output_dir / "model_artifact_metadata.json",
        {
            "dataset_name": "credit_46_features",
            "model_type": "xgboost_classifier",
            "feature_count": len(model_features),
            "feature_columns": model_features,
            "source_features": source_features,
            "missing_value_strategy": "xgboost_native_missing",
            "fill_values": {str(key): float(value) for key, value in fill_values.to_dict().items()},
            "threshold_default": 0.5,
            "threshold_tuned": tuned_threshold,
            "threshold_selection_rule": TUNED_THRESHOLD_SELECTION_RULE,
            "threshold_recall_floor": TUNED_THRESHOLD_RECALL_FLOOR,
            "probability_output": "calibrated_probability",
            "probability_calibration": calibration_summary,
            "best_iteration": getattr(model, "best_iteration", None),
            "best_score": getattr(model, "best_score", None),
            "saved_files": [
                "xgboost_model.json",
                "model_artifact_metadata.json",
            ],
        },
    )


def write_model_readme(model_output_dir: Path) -> None:
    content = """# 46-Feature XGBoost Model Artifacts

이 폴더는 `credit_46_features` 데이터를 기준으로 다시 학습한
XGBoost 모델링 산출물을 저장한 결과입니다. CAS 기준 원본은
`data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`이고,
전체 5,451개 라벨 기업-연도 중 train 3,851개 행으로 학습합니다.
TS2000 연결재무제표 값이 비어 있는 기업-연도는 OpenDART 사업보고서 기준
CFS를 먼저 사용하고, CFS가 없을 때만 OFS로 보강한 뒤 46-feature 입력을
재생성합니다.

구성:
- `xgboost_model.json`: XGBoost 원본 모델 파일
- `model_artifact_metadata.json`: 사용 변수, 결측 처리 전략, 기준선 등 메타데이터
- `diagnostics/`: 연도/시장/산업별 성능, threshold trade-off, calibration,
  대표 오류 사례, threshold 정책을 정리한 공식 46-feature 모델 진단 산출물

이 경로는 팀 공유용 모델링 산출물이자 Stage 1 런타임이 직접 참조하는 기준
모델 artifact 위치입니다.

`prob_speculative`는 검증셋 기준 Platt scaling을 적용한 보정 확률입니다.
결측값은 XGBoost native missing 방향 학습을 사용하며, metadata의
`fill_values`는 진단/후속 비교용 참고값으로만 보존합니다.
`threshold_tuned`는 validation 기준 Recall 0.85 이상을 유지하는 후보 중
Precision이 가장 높은 기준선을 사용합니다.

현재 test 성능은 다음과 같습니다.

| 기준선 | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| threshold=0.5 | 0.8321 | 0.9415 | 0.7656 | 0.7241 | 0.7443 |
| tuned threshold=0.30 | 0.8321 | 0.9415 | 0.6941 | 0.8719 | 0.7729 |

43-feature baseline tuned threshold 기준 test 성능은 PR-AUC 0.8329,
ROC-AUC 0.9415, Precision 0.7004, Recall 0.8522, F1 0.7689였습니다.
46-feature 승격 후 Precision은 소폭 낮아졌지만 Recall은 0.8719로 상승했고,
FN은 30건에서 26건으로 줄었습니다.

Rolling validation은 단일 1년 validation에 대한 과신을 줄이기 위해 사용합니다.
특정 경기/시장 국면에 우연히 잘 맞은 후보 변수를 바로 채택하지 않고, 여러
평가연도에서 반복적으로 안정적인지 확인한 뒤 final test는 마지막 확인용으로만
사용합니다.

데이터와 모델 artifact 전체 재생성 순서는 아래와 같습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind model-v1 --all-years --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_financial_supplements.py
/opt/anaconda3/envs/aura/bin/python scripts/rebuild_feature_46_dataset.py
/opt/anaconda3/envs/aura/bin/python scripts/import_feature_46_inference_2026_aux.py
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source-kind inference --target-fiscal-year 2025 --fallback-ofs
/opt/anaconda3/envs/aura/bin/python scripts/export_inference_2026_missing_2024_lag_targets.py
/opt/anaconda3/envs/aura/bin/python scripts/collect_opendart_financial_statements.py --source data/raw/opendart/inference_2026_missing_2024_lag_targets.csv --source-kind inference --target-fiscal-year 2025 --opendart-bsns-year 2024 --fallback-ofs --output-dir data/raw/opendart/lag_2024_tmp
/opt/anaconda3/envs/aura/bin/python scripts/apply_opendart_inference_financial_supplements.py --lag-raw-supplement data/raw/opendart/lag_2024_tmp/financial_statements_inference_2024_cfs_with_ofs_fallback_raw.csv
/opt/anaconda3/envs/aura/bin/python scripts/build_feature_46_inference_2026.py --check-only
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_dashboard_artifacts.py
```

진단 산출물은 모델을 다시 학습하지 않고 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_model_diagnostics.py
```

threshold 정책별 valid/test 성능 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_threshold_policy_experiments.py
```

46-feature 입력을 유지한 XGBoost regularization rolling OOT 실험은 아래 명령으로
재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_regularized_xgboost_experiments.py
```

46-feature 입력에 trend diff와 peer-relative percentile 후보를 추가하는 feature
pack 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_trend_peer_feature_experiments.py
```

46-feature 공식 모델 score를 유지하면서 calibration 후보와 dashboard 운영
threshold mode를 비교하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_calibration_operating_policy_experiments.py
```

KOSDAQ 제조업 FN rescue gate의 rolling OOT 실험은 아래 명령으로 재생성할 수
있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_46_manufacturing_fn_rescue_experiments.py
```

43-feature 기준 중간 실험 diagnostics와 Stage 2 반복 실행 산출물은 46-feature
공식 승격 후 제거했습니다. 승격 판단에 필요한 전후 성능과 근거는
`docs/stage1_46_feature_promotion_ko.md` 문서 본문에 보존합니다.
"""
    (model_output_dir / "README.md").write_text(content, encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    content = """# 46-Feature Dashboard Artifacts

이 폴더는 `credit_46_features` 입력 파일을
대시보드가 바로 읽을 수 있는 형식으로 변환한 결과입니다.

핵심 파일:
- `company_universe.csv`: 기업-연도 전체 기본값
- `company_latest.csv`: 기업별 최신 행
- `peer_percentiles.csv`: 산업/시장 비교용 백분위
- `feature_dictionary.csv`: 지표 설명 사전
- `prediction_scores.csv`: 기업별 예측확률/판정
- `stage2_review_signals.csv`: `full_review_trigger_73` 기반 2차 위원회 추가 검토 트리거
- `local_shap.csv`: 기업별 주요 영향 요인
- `industry_*`: 산업 집계 요약
- `model_summary.json`: 성능/기준선 요약

`stage2_review_trigger`는 1차 공식 모델 판단을 덮어쓰지 않습니다.
공식 모델이 위험으로 본 기업 또는 `full_review_trigger_73`/IT서비스 보조 기준선이 추가로
감지한 기업을 2차 위원회 검토 대상으로 표시하는 보조 신호입니다.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    try:
        import shap
        from xgboost import XGBClassifier
    except ModuleNotFoundError as error:  # pragma: no cover
        raise SystemExit(
            "Missing ML dependency. Run this script inside the aura environment with "
            "xgboost and shap installed."
        ) from error

    input_dir = args.input_dir
    master = pd.read_csv(input_dir / "feature_46_master.csv", encoding="utf-8-sig")
    feature_json = read_json(input_dir / "feature_46_list.json")
    metadata_json = read_json(args.metadata_path)
    metadata_columns = metadata_json.get("columns", [])

    train_ready = pd.read_csv(input_dir / "xgb_train.csv", encoding="utf-8-sig")
    valid_ready = pd.read_csv(input_dir / "xgb_valid.csv", encoding="utf-8-sig")
    test_ready = pd.read_csv(input_dir / "xgb_test.csv", encoding="utf-8-sig")
    id_frames = {
        "train": pd.read_csv(input_dir / "xgb_id_train.csv", encoding="utf-8-sig"),
        "valid": pd.read_csv(input_dir / "xgb_id_valid.csv", encoding="utf-8-sig"),
        "test": pd.read_csv(input_dir / "xgb_id_test.csv", encoding="utf-8-sig"),
    }

    source_features = list(feature_json["selected_source_features"])
    model_features = [column for column in train_ready.columns if column != "is_speculative"]
    source_feature_mapping = {
        model_feature: item["source_feature"]
        for item in feature_json.get("feature_metadata", [])
        for model_feature in item.get("model_features", [])
        if "source_feature" in item
    }
    categorical_source_features = list(feature_json.get("categorical_one_hot_columns", []))
    numeric_source_features = [
        feature for feature in source_features if feature not in categorical_source_features
    ]

    medians = train_ready[model_features].median(numeric_only=True)
    x_train = train_ready[model_features]
    y_train = train_ready["is_speculative"].astype(int)
    x_valid = valid_ready[model_features]
    y_valid = valid_ready["is_speculative"].astype(int)
    x_test = test_ready[model_features]
    y_test = test_ready["is_speculative"].astype(int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / pos) if pos else 1.0
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
        random_state=args.seed,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    valid_raw_prob = model.predict_proba(x_valid)[:, 1]
    test_raw_prob = model.predict_proba(x_test)[:, 1]
    train_raw_prob = model.predict_proba(x_train)[:, 1]
    calibration = fit_platt_calibration(y_valid, valid_raw_prob)
    valid_prob = apply_probability_calibration(valid_raw_prob, calibration)
    test_prob = apply_probability_calibration(test_raw_prob, calibration)
    train_prob = apply_probability_calibration(train_raw_prob, calibration)
    calibration_summary = build_calibration_summary(
        calibration=calibration,
        y_valid=y_valid,
        y_test=y_test,
        valid_raw_probabilities=valid_raw_prob,
        test_raw_probabilities=test_raw_prob,
        valid_calibrated_probabilities=valid_prob,
        test_calibrated_probabilities=test_prob,
    )
    tuned_threshold = choose_tuned_threshold(y_valid, valid_prob)

    probabilities = {
        "train": train_prob,
        "valid": valid_prob,
        "test": test_prob,
    }
    raw_probabilities = {
        "train": train_raw_prob,
        "valid": valid_raw_prob,
        "test": test_raw_prob,
    }
    y_frames = {"train": y_train, "valid": y_valid, "test": y_test}
    prediction_scores = build_prediction_scores(
        id_frames,
        probabilities,
        tuned_threshold,
        y_frames,
        raw_probabilities=raw_probabilities,
        calibration_method=str(calibration["method"]),
    )
    stage2_review_model = build_stage2_review_probabilities(
        frames={"train": train_ready, "valid": valid_ready, "test": test_ready},
        id_frames=id_frames,
        raw_path=args.raw_path,
        base_model_features=model_features,
        seed=args.seed,
    )
    prediction_scores = add_stage2_review_signals(
        prediction_scores,
        review_probabilities=stage2_review_model["probabilities"],
        review_raw_probabilities=stage2_review_model["raw_probabilities"],
        review_default_threshold=float(stage2_review_model["default_threshold"]),
        review_it_services_threshold=float(stage2_review_model["it_services_threshold"]),
        review_calibration_method=str(stage2_review_model["calibration_method"]),
    )
    prediction_scores = add_manufacturing_fn_rescue_signals(
        prediction_scores,
        raw_path=args.raw_path,
    )
    stage2_overwarning_filter_model = build_stage2_overwarning_filter_probabilities(
        frames={"train": train_ready, "valid": valid_ready, "test": test_ready},
        id_frames=id_frames,
        raw_path=args.raw_path,
        base_model_features=model_features,
        seed=args.seed,
    )
    prediction_scores = add_stage2_overwarning_filter_signals(
        prediction_scores,
        filter_probabilities=stage2_overwarning_filter_model["probabilities"],
        filter_raw_probabilities=stage2_overwarning_filter_model["raw_probabilities"],
        filter_threshold=float(stage2_overwarning_filter_model["threshold"]),
        filter_calibration_method=str(stage2_overwarning_filter_model["calibration_method"]),
    )

    explainer = shap.TreeExplainer(model)
    shap_values_by_split = {}
    for split, ready_frame in [("train", x_train), ("valid", x_valid), ("test", x_test)]:
        shap_values = explainer.shap_values(ready_frame.to_numpy())
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        if getattr(shap_values, "ndim", 2) == 3:
            shap_values = shap_values[:, :, -1]
        shap_values_by_split[split] = np.asarray(shap_values)

    feature_dictionary = build_feature_dictionary(metadata_columns, feature_json)
    company_universe = build_company_universe(master, source_features)
    company_latest = build_company_latest(master, source_features)
    peer_percentiles = build_peer_percentiles(master, numeric_source_features)
    local_shap = build_local_shap(
        prediction_scores,
        master,
        shap_values_by_split,
        model_features,
        source_feature_mapping,
        source_features,
        top_k_shap=args.top_k_shap,
    )
    global_shap_reference = build_global_shap_reference(local_shap, feature_dictionary)
    industry_year_summary = build_industry_year_summary(prediction_scores)
    industry_latest_summary = build_industry_latest_summary(prediction_scores)
    industry_shap_summary = build_industry_shap_summary(local_shap)
    model_summary = build_model_summary(
        y_train,
        y_valid,
        y_test,
        valid_prob,
        test_prob,
        tuned_threshold,
        calibration_summary,
        valid_raw_prob,
        test_raw_prob,
    )
    model_summary["stage2_review_trigger_policy"] = {
        "purpose": (
            "공식 Stage 1 모델 원판단은 유지하고, full_review_trigger_73은 "
            "2차 위원회 검토 대상을 넓히는 보조 레이더로 사용합니다."
        ),
        "base_model": "feature_46_xgboost",
        "secondary_feature_set": STAGE2_REVIEW_AUX_FEATURE_SET,
        "secondary_feature_count": len(stage2_review_model["feature_columns"]),
        "base_feature_count": len(model_features),
        "secondary_added_feature_count": len(STAGE2_REVIEW_FEATURES),
        "secondary_features": STAGE2_REVIEW_FEATURES,
        "default_stage2_review_aux_threshold": float(stage2_review_model["default_threshold"]),
        "it_services_review_threshold": float(stage2_review_model["it_services_threshold"]),
        "it_services_recall_floor": STAGE2_IT_SERVICES_RECALL_FLOOR,
        "trigger_columns": [
            "stage2_review_trigger",
            "stage2_secondary_trigger",
            "stage2_review_priority",
            "trigger_reason_code",
            "trigger_reason",
        ],
        "score_columns": [
            STAGE2_REVIEW_AUX_PROB_COLUMN,
            STAGE2_REVIEW_AUX_LABEL_COLUMN,
            STAGE2_REVIEW_AUX_THRESHOLD_COLUMN,
            STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN,
        ],
        "summary": build_stage2_review_signal_summary(prediction_scores),
    }
    model_summary["stage2_fn_rescue_gate_policy"] = {
        "purpose": (
            "공식 Stage 1 모델 원판단은 유지하고, 낮은 Stage1 확률로 놓치기 쉬운 "
            "KOSDAQ 제조업 재무 스트레스 케이스를 Stage2 에이전트 검토 큐에 추가합니다."
        ),
        "policy_name": FN_RESCUE_POLICY_NAME,
        "target_market": "KOSDAQ",
        "target_industry": "manufacturing",
        "probability_ceiling": FN_RESCUE_DEFAULT_PROB_CEILING,
        "score_threshold": FN_RESCUE_DEFAULT_SCORE_THRESHOLD,
        "min_group_count": FN_RESCUE_DEFAULT_MIN_GROUPS,
        "score_columns": FN_RESCUE_SCORE_COLUMNS,
        "trigger_columns": [
            "stage2_fn_rescue_trigger",
            "fn_rescue_score",
            "fn_rescue_group_count",
            "fn_rescue_probability_ceiling",
            "fn_rescue_score_threshold",
            "fn_rescue_policy",
        ],
    }
    model_summary["stage2_overwarning_filter_policy"] = {
        "purpose": (
            "공식 Stage 1 모델 원판단은 유지하고, 조합형 점수 모델은 1차 위험 경고가 "
            "과민할 수 있는지 2차 위원회가 확인하는 완화 검토 필터로 사용합니다."
        ),
        "base_model": "feature_46_xgboost",
        "filter_feature_set": "feature_46_plus_composite_scores",
        "filter_features": STAGE2_OVERWARNING_FILTER_FEATURES,
        "threshold": float(stage2_overwarning_filter_model["threshold"]),
        "trigger_columns": [
            "prob_speculative_overwarning_filter",
            "pred_label_overwarning_filter_tuned",
            "stage2_overwarning_filter_candidate",
            "overwarning_filter_reason_code",
            "overwarning_filter_reason",
        ],
        "summary": build_stage2_overwarning_filter_summary(prediction_scores),
    }

    scenario_presets = {
        name: {feature: value for feature, value in preset.items() if feature in source_features}
        for name, preset in SCENARIO_PRESETS.items()
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    company_universe.to_csv(output_dir / "company_universe.csv", index=False, encoding="utf-8-sig")
    company_latest.to_csv(output_dir / "company_latest.csv", index=False, encoding="utf-8-sig")
    peer_percentiles.to_csv(output_dir / "peer_percentiles.csv", index=False, encoding="utf-8-sig")
    feature_dictionary.to_csv(
        output_dir / "feature_dictionary.csv", index=False, encoding="utf-8-sig"
    )
    global_shap_reference.to_csv(
        output_dir / "global_shap_reference.csv", index=False, encoding="utf-8-sig"
    )
    prediction_scores.to_csv(
        output_dir / "prediction_scores.csv", index=False, encoding="utf-8-sig"
    )
    stage2_review_columns = [
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "split",
        "is_speculative",
        "prob_speculative",
        "pred_label_tuned",
        STAGE2_REVIEW_AUX_PROB_COLUMN,
        STAGE2_REVIEW_AUX_LABEL_COLUMN,
        "threshold",
        STAGE2_REVIEW_AUX_THRESHOLD_COLUMN,
        STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN,
        "stage2_review_trigger",
        "stage2_secondary_trigger",
        "stage2_review_priority",
        "trigger_reason_code",
        "trigger_reason",
        *FN_RESCUE_SCORE_COLUMNS,
        "stage2_fn_rescue_trigger",
        "fn_rescue_probability_ceiling",
        "fn_rescue_score_threshold",
        "fn_rescue_min_group_count",
        "fn_rescue_policy",
        "prob_speculative_overwarning_filter",
        "pred_label_overwarning_filter_tuned",
        "threshold_overwarning_filter",
        "stage2_overwarning_filter_candidate",
        "overwarning_filter_reason_code",
        "overwarning_filter_reason",
    ]
    prediction_scores.loc[:, stage2_review_columns].to_csv(
        output_dir / "stage2_review_signals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    local_shap.to_csv(output_dir / "local_shap.csv", index=False, encoding="utf-8-sig")
    industry_year_summary.to_csv(
        output_dir / "industry_year_summary.csv", index=False, encoding="utf-8-sig"
    )
    industry_latest_summary.to_csv(
        output_dir / "industry_latest_summary.csv", index=False, encoding="utf-8-sig"
    )
    industry_shap_summary.to_csv(
        output_dir / "industry_shap_summary.csv", index=False, encoding="utf-8-sig"
    )

    write_json(output_dir / "scenario_presets.json", scenario_presets)
    write_json(output_dir / "model_summary.json", model_summary)
    save_model_artifacts(
        model_output_dir=args.model_output_dir,
        model=model,
        model_features=model_features,
        fill_values=medians,
        tuned_threshold=tuned_threshold,
        source_features=source_features,
        calibration_summary=calibration_summary,
    )
    write_json(
        output_dir / "dashboard_export_manifest.json",
        {
            "dataset_name": "credit_46_features",
            "dataset_note": (
                "37개 원천/파생 변수 / 원핫 후 46개 공식 입력 변수셋을 "
                "대시보드용 형식으로 변환한 결과입니다."
            ),
            "generated_files": sorted(
                [path.name for path in output_dir.iterdir() if path.is_file()]
            ),
            "prediction_artifacts_ready": True,
            "prediction_artifacts_note": (
                "Per-company prediction probabilities, local SHAP, and industry summaries are "
                "generated from the credit_46_features split."
            ),
            "model_artifacts_ready": True,
            "model_artifacts_path": str(args.model_output_dir.relative_to(ROOT)),
        },
    )
    write_readme(output_dir)
    write_model_readme(args.model_output_dir)
    print(f"feature_46 dashboard artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
