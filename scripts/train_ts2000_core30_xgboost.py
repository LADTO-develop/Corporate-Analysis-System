"""Train an OOT XGBoost baseline on a TS2000 feature set manifest."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
OUTPUTS_ROOT = PROJECT_ROOT.parent / "03_Outputs"

DATASET_PATH = PROJECT_ROOT / "data" / "external" / "ts2000" / "TS2000_Credit_Model_Dataset_Model_V1.csv"
CORE30_MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "ts2000" / "TS2000_Model_Core30_Manifest.json"
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "modeling" / "core30_xgboost"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to TS2000_Credit_Model_Dataset_Model_V1.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CORE30_MANIFEST_PATH,
        help="Path to TS2000_Model_Core30_Manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save metrics, predictions, and feature importance outputs",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2020,
        help="Last fiscal year included in the training split",
    )
    parser.add_argument(
        "--valid-end-year",
        type=int,
        default=2022,
        help="Last fiscal year included in the validation split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for XGBoost and preprocessing",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Maximum boosting rounds",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for XGBoost",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum tree depth",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.9,
        help="Row subsample ratio",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.9,
        help="Column subsample ratio per tree",
    )
    parser.add_argument(
        "--imputation-strategy",
        choices=["marketwise_train_stats", "global_train_stats"],
        default="marketwise_train_stats",
        help="Missing-value imputation strategy fitted on train only",
    )
    return parser.parse_args()


MetricFn = Callable[..., float]


def load_project_helpers() -> tuple[
    Callable[[str | Path], Path],
    Callable[[str | Path], Any],
    Callable[[Any, str | Path], None],
    Callable[..., None],
    Callable[..., Any],
    Callable[[pd.DataFrame | list[dict[str, Any]], str | Path], dict[str, str]],
    Callable[[int | None], int],
]:
    """Import project-local helpers after adding ``src`` to sys.path."""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from cas.reporting.explanations import export_global
    from cas.utils.io import ensure_dir, read_json, write_json
    from cas.utils.logging import configure_logging, get_logger
    from cas.utils.seeds import set_seeds

    return ensure_dir, read_json, write_json, configure_logging, get_logger, export_global, set_seeds


def load_ml_dependencies() -> tuple[Any, Any, Any, Any, MetricFn, MetricFn, MetricFn, MetricFn, Any]:
    """Import sklearn/xgboost lazily with a friendly error if missing."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.metrics import (
            average_precision_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment dependent
        raise SystemExit(
            "Missing ML dependency. Install with: "
            'pip install -e ".[dev,ml]"'
        ) from exc

    return (
        ColumnTransformer,
        Pipeline,
        OneHotEncoder,
        XGBClassifier,
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
        exc_safe_get_feature_names,
    )


def exc_safe_get_feature_names(preprocessor: object) -> list[str]:
    """Return transformed feature names from a fitted ColumnTransformer."""
    names = preprocessor.get_feature_names_out()  # type: ignore[attr-defined]
    return [str(name) for name in names]


def sanitize_feature_name(name: str, categorical_columns: list[str]) -> str:
    """Strip transformer prefixes and recover the original feature name."""
    transformer_name = ""
    stripped_name = name
    if "__" in name:
        transformer_name, stripped_name = name.split("__", 1)

    if transformer_name == "categorical":
        for column in sorted(categorical_columns, key=len, reverse=True):
            if stripped_name == column or stripped_name.startswith(f"{column}_"):
                return column

    return stripped_name


def split_oot(df: pd.DataFrame, train_end_year: int, valid_end_year: int) -> dict[str, pd.DataFrame]:
    """Create out-of-time train/valid/test splits based on fiscal_year."""
    train = df[df["fiscal_year"] <= train_end_year].copy()
    valid = df[(df["fiscal_year"] > train_end_year) & (df["fiscal_year"] <= valid_end_year)].copy()
    test = df[df["fiscal_year"] > valid_end_year].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            "OOT split produced an empty subset. "
            "Adjust train_end_year / valid_end_year for the current fiscal_year range."
        )
    return {"train": train, "valid": valid, "test": test}


def safe_mode(series: pd.Series, default: object) -> object:
    """Return the first mode, or a default when no non-null value exists."""
    mode = series.dropna().mode()
    if mode.empty:
        return default
    return mode.iloc[0]


def fit_global_imputation_stats(
    train: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Fit train-only global median/mode statistics."""
    numeric_stats = {
        column: float(pd.to_numeric(train[column], errors="coerce").median())
        for column in numeric_columns
    }
    categorical_stats = {
        column: safe_mode(train[column], "__missing__")
        for column in categorical_columns
    }
    return {"numeric": numeric_stats, "categorical": categorical_stats}


def fit_marketwise_imputation_stats(
    train: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    global_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fit train-only market-level stats with global fallback."""
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
                subset[column], global_stats["categorical"][column]
            )

    return {"numeric": per_market_numeric, "categorical": per_market_categorical}


def apply_global_imputation(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    global_stats: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Fill missing values using train-only global stats."""
    result = frame.copy()
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(global_stats["numeric"][column])
    for column in categorical_columns:
        result[column] = result[column].fillna(global_stats["categorical"][column])
    return result


def apply_marketwise_imputation(
    frame: pd.DataFrame,
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    global_stats: dict[str, dict[str, Any]],
    marketwise_stats: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    """Fill missing values using train-only market stats, then global fallback."""
    result = frame.copy()
    market_key = result["market"].astype("string").fillna("__missing__")

    for column in numeric_columns:
        series = pd.to_numeric(result[column], errors="coerce")
        fill_values = market_key.map(
            lambda key: marketwise_stats["numeric"].get(
                str(key), {}
            ).get(column, global_stats["numeric"][column])
        )
        result[column] = series.fillna(fill_values).astype(float)

    for column in categorical_columns:
        fill_values = market_key.map(
            lambda key: marketwise_stats["categorical"].get(
                str(key), {}
            ).get(column, global_stats["categorical"][column])
        )
        result[column] = result[column].fillna(fill_values)

    return result


def safe_metric(
    func: MetricFn,
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    threshold: float | None = None,
) -> float | None:
    """Compute a metric and return None instead of crashing on degenerate subsets."""
    try:
        if threshold is None:
            return float(func(y_true, y_score))
        y_pred = (y_score >= threshold).astype(int)
        return float(func(y_true, y_pred, zero_division=0))
    except ValueError:
        return None


def evaluate_split(
    *,
    frame: pd.DataFrame,
    scores: pd.Series,
    threshold: float = 0.5,
    average_precision_score: MetricFn,
    precision_score: MetricFn,
    recall_score: MetricFn,
    roc_auc_score: MetricFn,
) -> dict[str, Any]:
    """Summarize overall and market-level performance for one split."""
    y_true = frame["is_speculative"].astype(int)
    result: dict[str, Any] = {
        "rows": len(frame),
        "positive_rows": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "pr_auc": safe_metric(average_precision_score, y_true, scores),
        "roc_auc": safe_metric(roc_auc_score, y_true, scores),
        "precision_at_0_5": safe_metric(precision_score, y_true, scores, threshold=threshold),
        "recall_at_0_5": safe_metric(recall_score, y_true, scores, threshold=threshold),
        "by_market": {},
    }
    for market, sub in frame.groupby("market", sort=True):
        sub_scores = scores.loc[sub.index]
        sub_true = sub["is_speculative"].astype(int)
        result["by_market"][str(market)] = {
            "rows": len(sub),
            "positive_rows": int(sub_true.sum()),
            "positive_rate": float(sub_true.mean()),
            "pr_auc": safe_metric(average_precision_score, sub_true, sub_scores),
            "roc_auc": safe_metric(roc_auc_score, sub_true, sub_scores),
            "precision_at_0_5": safe_metric(precision_score, sub_true, sub_scores, threshold=threshold),
            "recall_at_0_5": safe_metric(recall_score, sub_true, sub_scores, threshold=threshold),
        }
    return result


def build_feature_importance(
    feature_names: list[str],
    importances: list[float],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create transformed-feature and original-feature importance tables."""
    transformed_df = pd.DataFrame(
        {"feature_transformed": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False, ignore_index=True)
    transformed_df["feature_name"] = transformed_df["feature_transformed"].map(
        lambda name: sanitize_feature_name(name, categorical_columns)
    )

    original_df = (
        transformed_df.groupby("feature_name", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False, ignore_index=True)
    )
    return transformed_df, original_df


def main() -> None:
    """Train Core 30 XGBoost baseline and export artifacts."""
    args = parse_args()
    ensure_dir, read_json, write_json, configure_logging, get_logger, export_global, set_seeds = (
        load_project_helpers()
    )
    configure_logging(level="INFO", json_output=False)
    logger = get_logger("train_core30_xgboost")
    seed = set_seeds(args.seed)

    (
        column_transformer_cls,
        pipeline_cls,
        one_hot_encoder_cls,
        xgb_classifier_cls,
        average_precision_score,
        precision_score,
        recall_score,
        roc_auc_score,
        get_feature_names,
    ) = load_ml_dependencies()

    manifest = read_json(args.manifest)
    feature_columns = list(
        manifest.get("feature_columns")
        or manifest.get("core30_feature_columns")
        or []
    )
    if not feature_columns:
        raise ValueError(
            "Manifest must contain either 'feature_columns' or 'core30_feature_columns'."
        )
    manifest_name = str(manifest.get("manifest_name", "ts2000_feature_set"))
    target_column = str(manifest["target_column"])
    id_columns = list(manifest["id_columns"])
    time_columns = list(manifest["time_columns"])
    categorical_columns = list(manifest["recommended_categorical_columns"])
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]

    df = pd.read_csv(args.dataset, encoding="utf-8-sig")
    missing = [column for column in [*feature_columns, target_column] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    splits = split_oot(df, args.train_end_year, args.valid_end_year)
    train = splits["train"]
    valid = splits["valid"]
    test = splits["test"]

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

    if args.imputation_strategy == "marketwise_train_stats":
        impute_fn = lambda frame: apply_marketwise_imputation(  # noqa: E731
            frame,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            global_stats=global_stats,
            marketwise_stats=marketwise_stats,
        )
    else:
        impute_fn = lambda frame: apply_global_imputation(  # noqa: E731
            frame,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            global_stats=global_stats,
        )

    train_imputed = impute_fn(train)
    valid_imputed = impute_fn(valid)
    test_imputed = impute_fn(test)

    x_train = train_imputed[feature_columns].copy()
    y_train = train[target_column].astype(int).copy()
    x_valid = valid_imputed[feature_columns].copy()
    y_valid = valid[target_column].astype(int).copy()
    x_test = test_imputed[feature_columns].copy()

    preprocessor = column_transformer_cls(
        transformers=[
            (
                "categorical",
                pipeline_cls(steps=[("onehot", one_hot_encoder_cls(handle_unknown="ignore"))]),
                categorical_columns,
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns,
            ),
        ]
    )

    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)
    x_test_t = preprocessor.transform(x_test)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / pos) if pos else 1.0

    model = xgb_classifier_cls(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=3,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
    )
    model.fit(
        x_train_t,
        y_train,
        eval_set=[(x_train_t, y_train), (x_valid_t, y_valid)],
        verbose=False,
    )

    valid_scores = pd.Series(model.predict_proba(x_valid_t)[:, 1], index=valid.index)
    test_scores = pd.Series(model.predict_proba(x_test_t)[:, 1], index=test.index)

    output_dir = ensure_dir(args.output_dir)

    metrics_payload = {
        "config": {
            "dataset": str(args.dataset.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_name": manifest_name,
            "seed": seed,
            "train_end_year": args.train_end_year,
            "valid_end_year": args.valid_end_year,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "imputation_strategy": args.imputation_strategy,
        },
        "splits": {
            "train_years": [int(train["fiscal_year"].min()), int(train["fiscal_year"].max())],
            "valid_years": [int(valid["fiscal_year"].min()), int(valid["fiscal_year"].max())],
            "test_years": [int(test["fiscal_year"].min()), int(test["fiscal_year"].max())],
            "train": {
                "rows": len(train),
                "positive_rows": int(y_train.sum()),
                "positive_rate": float(y_train.mean()),
            },
            "valid": evaluate_split(
                frame=valid,
                scores=valid_scores,
                threshold=0.5,
                average_precision_score=average_precision_score,
                precision_score=precision_score,
                recall_score=recall_score,
                roc_auc_score=roc_auc_score,
            ),
            "test": evaluate_split(
                frame=test,
                scores=test_scores,
                threshold=0.5,
                average_precision_score=average_precision_score,
                precision_score=precision_score,
                recall_score=recall_score,
                roc_auc_score=roc_auc_score,
            ),
        },
        "best_iteration": int(getattr(model, "best_iteration", args.n_estimators - 1)),
        "best_score": float(getattr(model, "best_score", math.nan)),
    }
    write_json(metrics_payload, output_dir / "metrics_summary.json")

    prediction_columns = id_columns + time_columns + ["market", target_column]
    valid_predictions = valid[prediction_columns].copy()
    valid_predictions["pred_score"] = valid_scores
    valid_predictions["pred_label_0_5"] = (valid_scores >= 0.5).astype(int)
    valid_predictions.to_csv(output_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")

    test_predictions = test[prediction_columns].copy()
    test_predictions["pred_score"] = test_scores
    test_predictions["pred_label_0_5"] = (test_scores >= 0.5).astype(int)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    feature_names = get_feature_names(preprocessor)
    transformed_df, original_df = build_feature_importance(
        feature_names=feature_names,
        importances=list(model.feature_importances_),
        categorical_columns=categorical_columns,
    )
    transformed_df.to_csv(output_dir / "feature_importance_transformed.csv", index=False, encoding="utf-8-sig")
    original_df.to_csv(output_dir / "feature_importance_original.csv", index=False, encoding="utf-8-sig")
    export_global(
        original_df.rename(columns={"feature_name": "feature"}),
        output_dir,
        basename="feature_importance_original",
    )

    run_manifest = {
        "run_name": manifest_name.lower().replace(" ", "_"),
        "dataset_file": str(args.dataset.resolve()),
        "core_manifest_file": str(args.manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "imputation_strategy": args.imputation_strategy,
        "artifacts": {
            "metrics_summary": str((output_dir / "metrics_summary.json").resolve()),
            "valid_predictions": str((output_dir / "valid_predictions.csv").resolve()),
            "test_predictions": str((output_dir / "test_predictions.csv").resolve()),
            "feature_importance_transformed": str(
                (output_dir / "feature_importance_transformed.csv").resolve()
            ),
            "feature_importance_original": str(
                (output_dir / "feature_importance_original.csv").resolve()
            ),
            "feature_importance_original_json": str(
                (output_dir / "feature_importance_original.json").resolve()
            ),
        },
    }
    write_json(run_manifest, output_dir / "run_manifest.json")

    logger.info(
        "ts2000_xgboost_training_complete",
        output_dir=str(output_dir),
        manifest_name=manifest_name,
        test_pr_auc=metrics_payload["splits"]["test"]["pr_auc"],
        test_recall_at_0_5=metrics_payload["splits"]["test"]["recall_at_0_5"],
        top_features=original_df.head(10).to_dict(orient="records"),
    )


if __name__ == "__main__":
    main()
