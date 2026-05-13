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
OUTPUT_DIR = ROOT / "data" / "outputs" / "dashboard" / "feature_43_mvp"
MODEL_OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_43_xgboost"

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


def choose_tuned_threshold(y_valid: pd.Series, valid_probabilities: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    candidates = np.unique(np.round(valid_probabilities, 6))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        predictions = (valid_probabilities >= threshold).astype(int)
        score = float(f1_score(y_valid, predictions, zero_division=0))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
    return best_threshold


def build_prediction_scores(
    id_frames: dict[str, pd.DataFrame],
    probabilities: dict[str, np.ndarray],
    tuned_threshold: float,
    y_frames: dict[str, pd.Series],
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for split, id_frame in id_frames.items():
        scored = id_frame.copy()
        scored["split"] = split
        scored["is_speculative"] = y_frames[split].astype(int).to_numpy()
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
                "selection_rule": "best_valid_f1",
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
        "split_summary": {
            "train": {"rows": len(train_y), "positive_rate": float(train_y.mean())},
            "valid": {"rows": len(valid_y), "positive_rate": float(valid_y.mean())},
            "test": {"rows": len(test_y), "positive_rate": float(test_y.mean())},
        },
        "valid_default_0_5": metrics(valid_y, valid_prob, 0.5),
        "test_default_0_5": metrics(test_y, test_prob, 0.5),
        "test_tuned": metrics(test_y, test_prob, tuned_threshold),
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
            "fill_values": {str(key): float(value) for key, value in fill_values.to_dict().items()},
            "threshold_default": 0.5,
            "threshold_tuned": tuned_threshold,
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
- `model_artifact_metadata.json`: 사용 변수, 결측 대치값, 기준선 등 메타데이터

이 경로는 팀 공유용 모델링 산출물이자 Stage 1 런타임이 직접 참조하는 기준
모델 artifact 위치입니다.
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
- `local_shap.csv`: 기업별 주요 영향 요인
- `industry_*`: 산업 집계 요약
- `model_summary.json`: 성능/기준선 요약
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
    x_train = train_ready[model_features].fillna(medians)
    y_train = train_ready["is_speculative"].astype(int)
    x_valid = valid_ready[model_features].fillna(medians)
    y_valid = valid_ready["is_speculative"].astype(int)
    x_test = test_ready[model_features].fillna(medians)
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

    valid_prob = model.predict_proba(x_valid)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    train_prob = model.predict_proba(x_train)[:, 1]
    tuned_threshold = choose_tuned_threshold(y_valid, valid_prob)

    probabilities = {
        "train": train_prob,
        "valid": valid_prob,
        "test": test_prob,
    }
    y_frames = {"train": y_train, "valid": y_valid, "test": y_test}
    prediction_scores = build_prediction_scores(id_frames, probabilities, tuned_threshold, y_frames)

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
    )

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
