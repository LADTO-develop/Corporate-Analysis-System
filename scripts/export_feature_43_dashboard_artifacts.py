from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xgboost import XGBClassifier

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
METADATA_PATH = INPUT_DIR / "feature_43_dictionary_metadata.json"
RAW_PATH = ROOT / "data" / "raw" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "dashboard" / "feature_43_mvp"
MODEL_OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_43_xgboost"
PROBABILITY_CLIP_EPSILON = 1e-6
TUNED_THRESHOLD_RECALL_FLOOR = 0.85
TUNED_THRESHOLD_SELECTION_RULE = "valid_max_precision_with_recall_ge_0.85"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 6)
JOIN_KEYS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]
STAGE2_REVIEW_FEATURES = ["delta_accruals_ratio", "is_3y_consecutive_operating_loss"]
STAGE2_IT_SERVICES_RECALL_FLOOR = 0.90

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
        description="Export 43-feature model artifacts for the dashboard."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-output-dir", type=Path, default=MODEL_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-shap", type=int, default=10)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_mode(series: pd.Series, default: object) -> object:
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return mode.iloc[0]


def build_company_universe(master: pd.DataFrame, source_features: list[str]) -> pd.DataFrame:
    keep_columns = list(
        dict.fromkeys(
            [
                "market",
                "stock_code",
                "corp_name",
                "fiscal_year",
                "eval_year",
                "listed_year",
                "firm_size_group",
                "industry_macro_category",
                "is_speculative",
                *source_features,
            ]
        )
    )
    available = [column for column in keep_columns if column in master.columns]
    return (
        master.loc[:, available]
        .sort_values(["market", "corp_name", "stock_code", "fiscal_year", "eval_year"])
        .reset_index(drop=True)
    )


def build_company_latest(master: pd.DataFrame, source_features: list[str]) -> pd.DataFrame:
    latest = (
        master.sort_values(["fiscal_year", "eval_year"])
        .groupby(["market", "stock_code", "corp_name"], as_index=False)
        .tail(1)
    )
    keep_columns = list(
        dict.fromkeys(
            [
                "market",
                "stock_code",
                "corp_name",
                "fiscal_year",
                "eval_year",
                "listed_year",
                "firm_size_group",
                "industry_macro_category",
                *source_features,
            ]
        )
    )
    available = [column for column in keep_columns if column in latest.columns]
    return (
        latest.loc[:, available]
        .sort_values(["market", "corp_name", "stock_code"])
        .reset_index(drop=True)
    )


def build_peer_percentiles(master: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    base_columns = [
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "market",
        "industry_macro_category",
    ]
    for feature in numeric_features:
        values = pd.to_numeric(master[feature], errors="coerce")
        chunk = master.loc[:, base_columns].copy()
        chunk["feature"] = feature
        chunk["value"] = values
        chunk["overall_percentile"] = values.rank(method="average", pct=True) * 100.0
        chunk["market_percentile"] = (
            master.groupby("market")[feature].rank(method="average", pct=True) * 100.0
        )
        chunk["industry_percentile"] = (
            master.groupby("industry_macro_category")[feature].rank(
                method="average",
                pct=True,
            )
            * 100.0
        )
        chunk["overall_median"] = values.median(skipna=True)
        chunk["market_median"] = master.groupby("market")[feature].transform("median")
        chunk["industry_median"] = master.groupby("industry_macro_category")[feature].transform(
            "median"
        )
        chunks.append(chunk)
    return (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["stock_code", "fiscal_year", "feature"])
        .reset_index(drop=True)
    )


def build_feature_dictionary(
    metadata_columns: list[dict[str, object]],
    feature_json: dict[str, object],
) -> pd.DataFrame:
    metadata_lookup = {
        str(column["variable_name"]): column
        for column in metadata_columns
        if "variable_name" in column
    }
    feature_group_lookup = {
        str(item["source_feature"]): str(item.get("feature_group", "unknown"))
        for item in feature_json.get("feature_metadata", [])
        if "source_feature" in item
    }
    rows: list[dict[str, object]] = []
    for feature in feature_json["selected_source_features"]:
        info = metadata_lookup.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "feature_group": feature_group_lookup.get(feature, "unknown"),
                "korean_name": info.get("korean_name", feature),
                "description": info.get("description", ""),
                "formula_or_logic": info.get("formula_or_logic", ""),
                "unit": info.get("unit", ""),
                "source": info.get("source", "credit_43_features"),
                "note": info.get("note", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_group", "feature"]).reset_index(drop=True)


def sanitize_feature_name(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(name, name)


def risk_band(probability: float) -> str:
    if probability < 0.35:
        return "안정"
    if probability < 0.65:
        return "관찰"
    return "고위험"


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


def _probability_to_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        probabilities.astype(float), PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON
    )
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def fit_platt_calibration(
    y_valid: pd.Series,
    valid_probabilities: np.ndarray,
) -> dict[str, object]:
    """Fit a sigmoid calibration layer on validation predictions."""
    from sklearn.linear_model import LogisticRegression

    valid_logits = _probability_to_logit(valid_probabilities).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
    calibrator.fit(valid_logits, y_valid.astype(int))
    return {
        "method": "platt_sigmoid",
        "fit_split": "valid",
        "input": "xgboost_probability_logit",
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "clip_epsilon": PROBABILITY_CLIP_EPSILON,
    }


def apply_probability_calibration(
    probabilities: np.ndarray,
    calibration: dict[str, object],
) -> np.ndarray:
    """Apply the saved sigmoid calibration parameters to raw XGBoost probabilities."""
    if calibration.get("method") != "platt_sigmoid":
        return probabilities.astype(float)
    coef = float(calibration["coef"])
    intercept = float(calibration["intercept"])
    logits = _probability_to_logit(probabilities)
    return _sigmoid(intercept + coef * logits)


def build_calibration_summary(
    *,
    calibration: dict[str, object],
    y_valid: pd.Series,
    y_test: pd.Series,
    valid_raw_probabilities: np.ndarray,
    test_raw_probabilities: np.ndarray,
    valid_calibrated_probabilities: np.ndarray,
    test_calibrated_probabilities: np.ndarray,
) -> dict[str, object]:
    from sklearn.metrics import brier_score_loss, log_loss

    def calibration_metrics(
        y_true: pd.Series, raw: np.ndarray, calibrated: np.ndarray
    ) -> dict[str, float]:
        return {
            "brier_raw": float(brier_score_loss(y_true, raw)),
            "brier_calibrated": float(brier_score_loss(y_true, calibrated)),
            "logloss_raw": float(
                log_loss(
                    y_true, np.clip(raw, PROBABILITY_CLIP_EPSILON, 1 - PROBABILITY_CLIP_EPSILON)
                )
            ),
            "logloss_calibrated": float(
                log_loss(
                    y_true,
                    np.clip(
                        calibrated,
                        PROBABILITY_CLIP_EPSILON,
                        1 - PROBABILITY_CLIP_EPSILON,
                    ),
                )
            ),
        }

    return {
        **calibration,
        "probability_output": "calibrated_probability",
        "valid": calibration_metrics(
            y_valid, valid_raw_probabilities, valid_calibrated_probabilities
        ),
        "test": calibration_metrics(y_test, test_raw_probabilities, test_calibrated_probabilities),
        "note": (
            "XGBoost raw probabilities are transformed with a validation-fitted Platt "
            "sigmoid before being shown as prob_speculative."
        ),
    }


def choose_tuned_threshold(y_valid: pd.Series, valid_probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_fscore_support

    candidates: list[tuple[float, float, float, float]] = []
    for threshold in THRESHOLD_GRID:
        predictions = (valid_probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_valid,
            predictions,
            average="binary",
            zero_division=0,
        )
        candidates.append((float(threshold), float(precision), float(recall), float(f1)))

    recall_candidates = [
        candidate for candidate in candidates if candidate[2] >= TUNED_THRESHOLD_RECALL_FLOOR
    ]
    if recall_candidates:
        threshold, _, _, _ = max(
            recall_candidates,
            key=lambda candidate: (candidate[1], candidate[3], candidate[0]),
        )
        return threshold

    threshold, _, _, _ = max(candidates, key=lambda candidate: (candidate[3], candidate[2]))
    return threshold


def choose_max_precision_threshold_at_recall(
    y_valid: pd.Series,
    valid_probabilities: np.ndarray,
    recall_floor: float,
) -> float:
    from sklearn.metrics import precision_recall_fscore_support

    candidates: list[tuple[float, float, float, float]] = []
    for threshold in THRESHOLD_GRID:
        predictions = (valid_probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_valid,
            predictions,
            average="binary",
            zero_division=0,
        )
        candidates.append((float(threshold), float(precision), float(recall), float(f1)))

    recall_candidates = [candidate for candidate in candidates if candidate[2] >= recall_floor]
    if not recall_candidates:
        threshold, _, _, _ = max(candidates, key=lambda candidate: (candidate[3], candidate[2]))
        return threshold
    threshold, _, _, _ = max(
        recall_candidates,
        key=lambda candidate: (candidate[1], candidate[3], candidate[0]),
    )
    return threshold


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
    missing_columns = [column for column in STAGE2_REVIEW_FEATURES if column not in raw.columns]
    if missing_columns:
        raise KeyError(
            f"Stage 2 review feature columns are missing from raw data: {missing_columns}"
        )
    duplicates = raw.duplicated(JOIN_KEYS).sum()
    if duplicates:
        raise ValueError(f"Raw Model V1 has duplicate rows for join keys: {duplicates}")

    raw_subset = raw.loc[:, [*JOIN_KEYS, *STAGE2_REVIEW_FEATURES]].copy()
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
        output.loc[split_mask, "prob_speculative_45"] = split_probabilities
        output.loc[split_mask, "prob_speculative_45_raw"] = review_raw_probabilities[split]

    stage1_risk = output["pred_label_tuned"].astype(int).eq(1)
    feature45_risk = output["prob_speculative_45"].astype(float).ge(review_default_threshold)
    it_services_review = output["industry_macro_category"].astype(str).eq("it_services") & output[
        "prob_speculative_45"
    ].astype(float).ge(review_it_services_threshold)
    secondary_trigger = (~stage1_risk) & (feature45_risk | it_services_review)
    output["probability_45_calibration_method"] = review_calibration_method
    output["threshold_45"] = review_default_threshold
    output["threshold_45_it_services_review"] = review_it_services_threshold
    output["pred_label_45_tuned"] = feature45_risk.astype(int)
    output["stage2_review_trigger"] = stage1_risk | secondary_trigger
    output["stage2_secondary_trigger"] = secondary_trigger
    output["stage2_review_priority"] = np.select(
        [
            stage1_risk,
            (~stage1_risk) & feature45_risk,
            (~stage1_risk) & it_services_review,
        ],
        ["high", "medium", "watch"],
        default="none",
    )
    output["trigger_reason_code"] = np.select(
        [
            stage1_risk & feature45_risk,
            stage1_risk,
            (~stage1_risk) & feature45_risk,
            (~stage1_risk) & it_services_review,
        ],
        [
            "stage1_and_45_risk",
            "stage1_model_risk",
            "45_feature_set_only",
            "it_services_low_threshold",
        ],
        default="none",
    )
    output["trigger_reason"] = np.select(
        [
            stage1_risk & feature45_risk,
            stage1_risk,
            (~stage1_risk) & feature45_risk,
            (~stage1_risk) & it_services_review,
        ],
        [
            "1차 모델과 45개 변수셋이 모두 위험 기준선을 넘겨 위원회 검토 대상으로 분류했습니다.",
            "1차 모델이 위험 기준선을 넘겨 위원회 검토 대상으로 분류했습니다.",
            "43개 모델은 투자적격이나 45개 변수셋이 위험 기준선을 넘어 추가 검토 대상으로 올렸습니다.",
            "IT서비스 업종 보조 기준선을 넘어 45개 변수셋 기반 추가 검토 대상으로 올렸습니다.",
        ],
        default="추가 위원회 검토 트리거 없음",
    )
    output["trigger_policy"] = "43_model_default_or_45_feature_set_or_it_services_review_threshold"
    return output


def build_stage2_review_signal_summary(prediction_scores: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for split, split_frame in prediction_scores.groupby("split"):
        y_true = split_frame["is_speculative"].astype(int)
        trigger = split_frame["stage2_review_trigger"].astype(bool)
        secondary_trigger = split_frame["stage2_secondary_trigger"].astype(bool)
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
            "trigger_precision": true_positive / precision_denominator
            if precision_denominator
            else 0.0,
            "trigger_recall": true_positive / recall_denominator if recall_denominator else 0.0,
            "trigger_reason_counts": split_frame["trigger_reason_code"].value_counts().to_dict(),
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


def build_global_shap_reference(
    local_shap: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        local_shap.groupby("feature", as_index=False)
        .agg(mean_abs_shap=("abs_shap", "mean"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = grouped.index + 1
    merged = grouped.merge(feature_dictionary, how="left", on="feature")
    return merged.loc[
        :,
        [
            "rank",
            "feature",
            "feature_group",
            "mean_abs_shap",
            "korean_name",
            "description",
            "unit",
            "note",
        ],
    ]


def build_industry_year_summary(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    return (
        prediction_scores.groupby(
            ["market", "industry_macro_category", "fiscal_year", "split"],
            dropna=False,
        )
        .agg(
            rows=("stock_code", "size"),
            companies=("stock_code", "nunique"),
            positive_rows=("is_speculative", "sum"),
            positive_rate=("is_speculative", "mean"),
            mean_prob_speculative=("prob_speculative", "mean"),
            median_prob_speculative=("prob_speculative", "median"),
            pred_share_0_5=("pred_label_0_5", "mean"),
            pred_share_tuned=("pred_label_tuned", "mean"),
        )
        .reset_index()
        .sort_values(["market", "industry_macro_category", "fiscal_year"])
    )


def build_industry_latest_summary(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    latest = (
        prediction_scores.sort_values(["fiscal_year", "eval_year"])
        .groupby(["market", "stock_code", "corp_name"], as_index=False)
        .tail(1)
    )
    return (
        latest.groupby(["market", "industry_macro_category"], dropna=False)
        .agg(
            companies=("stock_code", "nunique"),
            positive_companies=("is_speculative", "sum"),
            positive_rate=("is_speculative", "mean"),
            mean_prob_speculative=("prob_speculative", "mean"),
            median_prob_speculative=("prob_speculative", "median"),
            pred_share_0_5=("pred_label_0_5", "mean"),
            pred_share_tuned=("pred_label_tuned", "mean"),
        )
        .reset_index()
        .sort_values(["market", "industry_macro_category"])
    )


def build_industry_shap_summary(local_shap: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        local_shap.groupby(["market", "industry_macro_category", "split", "feature"], dropna=False)
        .agg(
            count=("feature", "size"),
            mean_abs_shap=("abs_shap", "mean"),
            mean_signed_shap=("shap_value", "mean"),
        )
        .reset_index()
    )
    grouped["rank_within_group"] = (
        grouped.groupby(["market", "industry_macro_category", "split"])["mean_abs_shap"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return grouped.sort_values(
        ["market", "industry_macro_category", "split", "rank_within_group", "feature"]
    ).reset_index(drop=True)


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
        "selected_model": "feature_43_xgboost",
        "dataset_name": "credit_43_features",
        "test_overall_models": [
            {
                "model": "feature_43_xgboost",
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
            "exported from the credit_43_features split."
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


def build_llm_payload_template(source_features: list[str]) -> dict[str, object]:
    key_metrics = [
        feature
        for feature in [
            "current_ratio",
            "cash_ratio",
            "interest_coverage_ratio",
            "capital_impairment_ratio",
            "net_margin",
            "spec_spread",
        ]
        if feature in source_features
    ]
    return {
        "company_profile": {
            "corp_name": "<기업명>",
            "market": "<시장>",
            "industry_macro_category": "<산업>",
            "firm_size_group": "<규모>",
        },
        "model_output": {
            "prob_speculative": "<확률>",
            "predicted_label": "<투자적격/투기등급>",
            "threshold": "<기준선>",
            "risk_band": "<안정/관찰/고위험>",
        },
        "key_metrics": key_metrics,
        "top_shap": "<local_shap.csv의 상위 요인 5~10개>",
        "peer_context": "<peer_percentiles.csv 기반 산업/시장 비교 맥락>",
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
            "dataset_name": "credit_43_features",
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
    content = """# 43-Feature XGBoost Model Artifacts

이 폴더는 `credit_43_features` 데이터를 기준으로 다시 학습한
XGBoost 모델링 산출물을 저장한 결과입니다. CAS 기준 원본은
`data/raw/ts2000/TS2000_Credit_Model_Dataset_Model_V1.csv`이고,
전체 5,199개 라벨 기업-연도 중 train 3,851개 행으로 학습합니다.

구성:
- `xgboost_model.json`: XGBoost 원본 모델 파일
- `model_artifact_metadata.json`: 사용 변수, 결측 처리 전략, 기준선 등 메타데이터
- `diagnostics/`: 연도/시장/산업별 성능, threshold trade-off, calibration,
  대표 오류 사례, threshold 정책, FP 집중 구간, SHAP 기반 변수 개선 후보
  실험을 정리한 모델 진단 산출물

이 경로는 팀 공유용 모델링 산출물이자 Stage 1 런타임이 직접 참조하는 기준
모델 artifact 위치입니다.

`prob_speculative`는 검증셋 기준 Platt scaling을 적용한 보정 확률입니다.
결측값은 XGBoost native missing 방향 학습을 사용하며, metadata의
`fill_values`는 진단/후속 비교용 참고값으로만 보존합니다.
`threshold_tuned`는 validation 기준 Recall 0.85 이상을 유지하는 후보 중
Precision이 가장 높은 기준선을 사용합니다.

Rolling validation은 단일 1년 validation에 대한 과신을 줄이기 위해 사용합니다.
특정 경기/시장 국면에 우연히 잘 맞은 후보 변수를 바로 채택하지 않고, 여러
평가연도에서 반복적으로 안정적인지 확인한 뒤 final test는 마지막 확인용으로만
사용합니다.

진단 산출물은 모델을 다시 학습하지 않고 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_model_diagnostics.py
```

threshold 정책별 valid/test 성능 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_threshold_policy_experiments.py
```

오류 사례별 SHAP 패턴 분석은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_error_shap_analysis.py
```

오류 사례별 리뷰 테이블은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_error_case_review.py
```

SHAP 오류 패턴 기반 변수 개선 후보 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_shap_feature_experiments.py
```

원본 Model V1의 미사용 후보 변수를 묶음별로 추가하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_candidate_feature_pack_experiments.py
```

단일 후보 변수와 2개 조합 기반 forward selection 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_forward_selection_experiments.py
```

여러 연도 walk-forward rolling OOT validation 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_rolling_validation_experiments.py
```

rolling validation으로 전체 후보를 선별한 뒤 final test 성능을 확인하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_rolling_selection_test_experiments.py
```

43개 기준 모델과 45개 변수셋(`delta_accruals_ratio`,
`is_3y_consecutive_operating_loss` 추가)을 직접 비교하는 실험은 아래 명령으로
재생성할 수 있습니다. 이 산출물은 운영 모델 교체가 아니라 Recall 우선 후보
검토용입니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_45_experiment.py
```

45개 변수셋 기준으로 하이퍼파라미터, threshold 정책, Stage 2 보조 트리거 가능성을
비교하는 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_45_improvement_experiments.py
```

XGBoost 하이퍼파라미터 튜닝 실험은 아래 명령으로 재생성할 수 있습니다.

```bash
/opt/anaconda3/envs/aura/bin/python scripts/export_feature_43_xgboost_tuning_experiments.py
```
"""
    (model_output_dir / "README.md").write_text(content, encoding="utf-8")


def write_readme(output_dir: Path) -> None:
    content = """# 43-Feature Dashboard Artifacts

이 폴더는 `credit_43_features` 입력 파일을
대시보드가 바로 읽을 수 있는 형식으로 변환한 결과입니다.

핵심 파일:
- `company_universe.csv`: 기업-연도 전체 기본값
- `company_latest.csv`: 기업별 최신 행
- `peer_percentiles.csv`: 산업/시장 비교용 백분위
- `feature_dictionary.csv`: 지표 설명 사전
- `prediction_scores.csv`: 기업별 예측확률/판정
- `stage2_review_signals.csv`: 45개 변수셋 기반 2차 위원회 추가 검토 트리거
- `local_shap.csv`: 기업별 주요 영향 요인
- `industry_*`: 산업 집계 요약
- `model_summary.json`: 성능/기준선 요약

`stage2_review_trigger`는 1차 43개 모델 판단을 덮어쓰지 않습니다.
43개 모델이 위험으로 본 기업 또는 45개 변수셋/IT서비스 보조 기준선이 추가로
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
    master = pd.read_csv(input_dir / "feature_43_master.csv", encoding="utf-8-sig")
    feature_json = read_json(input_dir / "feature_43_list.json")
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
            "43개 모델 원판단은 유지하고, 45개 변수셋은 2차 위원회 검토 대상을 "
            "넓히는 보조 레이더로 사용합니다."
        ),
        "base_model": "feature_43_xgboost",
        "secondary_feature_set": "feature_45",
        "secondary_features": STAGE2_REVIEW_FEATURES,
        "default_45_threshold": float(stage2_review_model["default_threshold"]),
        "it_services_review_threshold": float(stage2_review_model["it_services_threshold"]),
        "it_services_recall_floor": STAGE2_IT_SERVICES_RECALL_FLOOR,
        "trigger_columns": [
            "stage2_review_trigger",
            "stage2_secondary_trigger",
            "stage2_review_priority",
            "trigger_reason_code",
            "trigger_reason",
        ],
        "summary": build_stage2_review_signal_summary(prediction_scores),
    }

    scenario_presets = {
        name: {feature: value for feature, value in preset.items() if feature in source_features}
        for name, preset in SCENARIO_PRESETS.items()
    }
    llm_payload_template = build_llm_payload_template(source_features)

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
        "prob_speculative_45",
        "pred_label_45_tuned",
        "threshold",
        "threshold_45",
        "threshold_45_it_services_review",
        "stage2_review_trigger",
        "stage2_secondary_trigger",
        "stage2_review_priority",
        "trigger_reason_code",
        "trigger_reason",
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
    write_json(output_dir / "llm_payload_template.json", llm_payload_template)
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
            "dataset_name": "credit_43_features",
            "dataset_note": (
                "34개 원천 변수 / 원핫 후 43개 입력 변수셋을 대시보드용 형식으로 변환한 결과입니다."
            ),
            "generated_files": sorted(
                [path.name for path in output_dir.iterdir() if path.is_file()]
            ),
            "prediction_artifacts_ready": True,
            "prediction_artifacts_note": (
                "Per-company prediction probabilities, local SHAP, and industry summaries are "
                "generated from the credit_43_features split."
            ),
            "model_artifacts_ready": True,
            "model_artifacts_path": str(args.model_output_dir.relative_to(ROOT)),
        },
    )
    write_readme(output_dir)
    write_model_readme(args.model_output_dir)
    print(f"feature_43 dashboard artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
