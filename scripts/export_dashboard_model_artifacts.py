"""Export per-company prediction and industry summary artifacts for the TS2000 dashboard."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TS2000_DIR = ROOT / "data" / "external" / "ts2000"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "outputs" / "dashboard" / "ts2000_core29_mvp"

DATASET_PATH = TS2000_DIR / "TS2000_Credit_Model_Dataset_Model_V1.csv"
MANIFEST_PATH = TS2000_DIR / "TS2000_Model_Core29_Manifest.json"
THRESHOLD_SUMMARY_PATH = (
    TS2000_DIR / "model_results" / "xgboost_threshold_shap" / "threshold_summary.csv"
)
EXPORT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "dashboard_export_manifest.json"
MODEL_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "model_summary_core29.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export company-level prediction and industry summary artifacts for the TS2000 dashboard."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--threshold-summary", type=Path, default=THRESHOLD_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-end-year", type=int, default=2020)
    parser.add_argument("--valid-end-year", type=int, default=2022)
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


def split_name(fiscal_year: int, train_end_year: int, valid_end_year: int) -> str:
    if fiscal_year <= train_end_year:
        return "train"
    if fiscal_year <= valid_end_year:
        return "valid"
    return "test"


def risk_band(probability: float) -> str:
    if probability < 0.35:
        return "안정"
    if probability < 0.65:
        return "관찰"
    return "고위험"


def load_tuned_threshold(path: Path) -> float:
    threshold_summary = pd.read_csv(path, encoding="utf-8-sig")
    tuned = threshold_summary.loc[threshold_summary["threshold_type"] == "tuned", "threshold"]
    if tuned.empty:
        return 0.5
    return float(tuned.iloc[0])


def sanitize_feature_name(name: str, categorical_columns: list[str]) -> str:
    transformer_name = ""
    stripped_name = name
    if "__" in name:
        transformer_name, stripped_name = name.split("__", 1)

    if transformer_name == "categorical":
        for column in sorted(categorical_columns, key=len, reverse=True):
            if stripped_name == column or stripped_name.startswith(f"{column}_"):
                return column

    return stripped_name


def fit_global_imputation_stats(
    train: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, Any]]:
    numeric_stats = {
        column: float(pd.to_numeric(train[column], errors="coerce").median())
        for column in numeric_columns
    }
    categorical_stats = {
        column: safe_mode(train[column], "__missing__") for column in categorical_columns
    }
    return {"numeric": numeric_stats, "categorical": categorical_stats}


def fit_marketwise_imputation_stats(
    train: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    global_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    per_market_numeric: dict[str, dict[str, Any]] = {}
    per_market_categorical: dict[str, dict[str, Any]] = {}

    for market, subset in train.groupby("market", dropna=False):
        market_key = str(market)
        per_market_numeric[market_key] = {}
        per_market_categorical[market_key] = {}

        for column in numeric_columns:
            value = pd.to_numeric(subset[column], errors="coerce").median()
            if pd.isna(value):
                value = global_stats["numeric"][column]
            per_market_numeric[market_key][column] = float(value)

        for column in categorical_columns:
            per_market_categorical[market_key][column] = safe_mode(
                subset[column],
                global_stats["categorical"][column],
            )

    return {"numeric": per_market_numeric, "categorical": per_market_categorical}


def apply_marketwise_imputation(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    global_stats: dict[str, dict[str, Any]],
    marketwise_stats: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    result = frame.copy()
    market_key = result["market"].astype("string").fillna("__missing__")

    for column in numeric_columns:
        numeric_series = pd.to_numeric(result[column], errors="coerce")
        fill_values = market_key.map(
            lambda key, current_column=column: marketwise_stats["numeric"]
            .get(
                str(key),
                {},
            )
            .get(current_column, global_stats["numeric"][current_column])
        )
        result[column] = numeric_series.fillna(fill_values).astype(float)

    for column in categorical_columns:
        fill_values = market_key.map(
            lambda key, current_column=column: marketwise_stats["categorical"]
            .get(
                str(key),
                {},
            )
            .get(current_column, global_stats["categorical"][current_column])
        )
        result[column] = result[column].fillna(fill_values)

    return result


def export_prediction_scores(
    dataset: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    tuned_threshold: float,
    train_end_year: int,
    valid_end_year: int,
    target_column: str,
) -> pd.DataFrame:
    scores = dataset.loc[
        :,
        [
            "market",
            "stock_code",
            "corp_name",
            "fiscal_year",
            "eval_year",
            "industry_macro_category",
            "firm_size_group",
            target_column,
        ],
    ].copy()
    scores["split"] = scores["fiscal_year"].apply(
        lambda year: split_name(int(year), train_end_year, valid_end_year)
    )
    scores["prob_speculative"] = probabilities
    scores["pred_label_0_5"] = (scores["prob_speculative"] >= 0.5).astype(int)
    scores["pred_label_tuned"] = (scores["prob_speculative"] >= tuned_threshold).astype(int)
    scores["predicted_label"] = scores["pred_label_tuned"]
    scores["threshold_default"] = 0.5
    scores["threshold_tuned"] = tuned_threshold
    scores["threshold"] = tuned_threshold
    scores["risk_band"] = scores["prob_speculative"].map(risk_band)
    scores = scores.rename(columns={target_column: "is_speculative"})
    return scores


def export_local_shap(
    scored_frame: pd.DataFrame,
    imputed_features: pd.DataFrame,
    shap_values: np.ndarray,
    *,
    transformed_feature_names: list[str],
    feature_columns: list[str],
    categorical_columns: list[str],
    top_k_shap: int,
) -> pd.DataFrame:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, transformed_name in enumerate(transformed_feature_names):
        grouped_feature = sanitize_feature_name(transformed_name, categorical_columns)
        grouped_indices[grouped_feature].append(index)

    grouped_shap = np.zeros((shap_values.shape[0], len(feature_columns)), dtype=float)
    for feature_index, feature in enumerate(feature_columns):
        source_indices = grouped_indices.get(feature, [])
        if not source_indices:
            continue
        grouped_shap[:, feature_index] = shap_values[:, source_indices].sum(axis=1)

    rows: list[dict[str, Any]] = []
    key_columns = [
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "industry_macro_category",
        "firm_size_group",
        "split",
        "is_speculative",
        "prob_speculative",
    ]

    for row_index in range(grouped_shap.shape[0]):
        row_values = grouped_shap[row_index]
        top_indices = np.argsort(np.abs(row_values))[::-1][:top_k_shap]
        for rank, feature_index in enumerate(top_indices, start=1):
            feature = feature_columns[feature_index]
            shap_value = float(row_values[feature_index])
            rows.append(
                {
                    **{column: scored_frame.iloc[row_index][column] for column in key_columns},
                    "feature": feature,
                    "rank": rank,
                    "shap_value": shap_value,
                    "abs_shap": abs(shap_value),
                    "direction": "increase_risk" if shap_value > 0 else "decrease_risk",
                    "feature_value": imputed_features.iloc[row_index][feature],
                    "feature_index": feature_index,
                }
            )

    return pd.DataFrame(rows)


def build_industry_year_summary(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = (
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
    return grouped


def build_industry_latest_summary(prediction_scores: pd.DataFrame) -> pd.DataFrame:
    latest = (
        prediction_scores.sort_values(["fiscal_year", "eval_year"])
        .groupby(["market", "stock_code", "corp_name"], as_index=False)
        .tail(1)
    )
    grouped = (
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
    return grouped


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


def update_export_manifest(
    output_dir: Path,
    *,
    generated_files: list[str],
) -> None:
    export_manifest_path = output_dir / "dashboard_export_manifest.json"
    export_manifest = read_json(export_manifest_path) if export_manifest_path.exists() else {}
    existing_files = export_manifest.get("generated_files", [])
    merged_files = sorted(set(existing_files) | set(generated_files))
    export_manifest["generated_files"] = merged_files
    export_manifest["prediction_artifacts_ready"] = True
    export_manifest["prediction_artifacts_note"] = (
        "Per-company prediction probabilities, local SHAP, and industry summaries are "
        "exported by retraining the official Core29 XGBoost recipe inside the repository."
    )
    write_json(export_manifest_path, export_manifest)


def update_model_summary(output_dir: Path) -> None:
    model_summary_path = output_dir / "model_summary_core29.json"
    if not model_summary_path.exists():
        return
    model_summary = read_json(model_summary_path)
    model_summary["prediction_artifacts_ready"] = True
    model_summary["prediction_artifacts_note"] = (
        "Per-company prediction probabilities, local SHAP, and industry summaries are "
        "available in this dashboard export directory."
    )
    write_json(model_summary_path, model_summary)


def main() -> None:
    args = parse_args()

    try:
        import shap
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier
    except ModuleNotFoundError as error:  # pragma: no cover
        raise SystemExit(
            "Missing ML dependency. Run this script inside the aura environment with "
            "xgboost, shap, and scikit-learn installed."
        ) from error

    dataset = pd.read_csv(args.dataset, encoding="utf-8-sig")
    manifest = read_json(args.manifest)

    feature_columns = manifest.get("core29_feature_columns") or manifest["feature_columns"]
    categorical_columns = [
        column
        for column in manifest.get("recommended_categorical_columns", [])
        if column in feature_columns
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    target_column = str(manifest["target_column"])
    train_mask = dataset["fiscal_year"] <= args.train_end_year
    valid_mask = (dataset["fiscal_year"] > args.train_end_year) & (
        dataset["fiscal_year"] <= args.valid_end_year
    )

    train = dataset.loc[train_mask].copy()
    valid = dataset.loc[valid_mask].copy()

    global_stats = fit_global_imputation_stats(
        train,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    marketwise_stats = fit_marketwise_imputation_stats(
        train,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        global_stats=global_stats,
    )

    train_imputed = apply_marketwise_imputation(
        train,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        global_stats=global_stats,
        marketwise_stats=marketwise_stats,
    )
    valid_imputed = apply_marketwise_imputation(
        valid,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        global_stats=global_stats,
        marketwise_stats=marketwise_stats,
    )
    all_imputed = apply_marketwise_imputation(
        dataset,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        global_stats=global_stats,
        marketwise_stats=marketwise_stats,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
    )
    x_train = preprocessor.fit_transform(train_imputed[feature_columns])
    x_valid = preprocessor.transform(valid_imputed[feature_columns])
    x_all = preprocessor.transform(all_imputed[feature_columns])

    y_train = train_imputed[target_column].astype(int)
    y_valid = valid_imputed[target_column].astype(int)
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count) if pos_count else 1.0

    model = XGBClassifier(
        random_state=args.seed,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )

    probabilities = model.predict_proba(x_all)[:, 1]
    tuned_threshold = load_tuned_threshold(args.threshold_summary)
    prediction_scores = export_prediction_scores(
        dataset,
        probabilities,
        tuned_threshold=tuned_threshold,
        train_end_year=args.train_end_year,
        valid_end_year=args.valid_end_year,
        target_column=target_column,
    )

    x_all_dense = x_all.toarray() if hasattr(x_all, "toarray") else x_all
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_all_dense)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, -1]

    local_shap = export_local_shap(
        prediction_scores,
        all_imputed[feature_columns],
        np.asarray(shap_values),
        transformed_feature_names=[str(name) for name in preprocessor.get_feature_names_out()],
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        top_k_shap=args.top_k_shap,
    )

    industry_year_summary = build_industry_year_summary(prediction_scores)
    industry_latest_summary = build_industry_latest_summary(prediction_scores)
    industry_shap_summary = build_industry_shap_summary(local_shap)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_scores_path = args.output_dir / "prediction_scores_core29.csv"
    local_shap_path = args.output_dir / "local_shap_core29.csv"
    industry_year_path = args.output_dir / "industry_year_summary_core29.csv"
    industry_latest_path = args.output_dir / "industry_latest_summary_core29.csv"
    industry_shap_path = args.output_dir / "industry_shap_summary_core29.csv"

    prediction_scores.to_csv(prediction_scores_path, index=False, encoding="utf-8-sig")
    local_shap.to_csv(local_shap_path, index=False, encoding="utf-8-sig")
    industry_year_summary.to_csv(industry_year_path, index=False, encoding="utf-8-sig")
    industry_latest_summary.to_csv(industry_latest_path, index=False, encoding="utf-8-sig")
    industry_shap_summary.to_csv(industry_shap_path, index=False, encoding="utf-8-sig")

    update_export_manifest(
        args.output_dir,
        generated_files=[
            prediction_scores_path.name,
            local_shap_path.name,
            industry_year_path.name,
            industry_latest_path.name,
            industry_shap_path.name,
        ],
    )
    update_model_summary(args.output_dir)

    print(f"prediction_scores: {prediction_scores_path}")
    print(f"local_shap: {local_shap_path}")
    print(f"industry_year_summary: {industry_year_path}")
    print(f"industry_latest_summary: {industry_latest_path}")
    print(f"industry_shap_summary: {industry_shap_path}")
    print(f"rows(prediction_scores): {len(prediction_scores):,}")
    print(f"rows(local_shap): {len(local_shap):,}")


if __name__ == "__main__":
    main()
