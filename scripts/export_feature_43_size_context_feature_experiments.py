from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas.modeling.calibration import (  # noqa: E402
    apply_probability_calibration,
    choose_tuned_threshold,
    fit_platt_calibration,
)
from cas.modeling.calibration_diagnostics import (  # noqa: E402
    classification_metrics_at_threshold,
    probability_quality_metrics,
)
from cas.modeling.size_context_features import (  # noqa: E402
    add_binary_group_context_features,
    add_group_percentile_features,
    add_group_zscore_features,
    add_signed_log_features,
)

INPUT_DIR = ROOT / "data" / "input" / "credit_46_features"
MASTER_PATH = INPUT_DIR / "feature_46_master.csv"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"

RANDOM_STATE = 42
RECALL_FLOOR = 0.85
MODEL_NAME = "feature_46_xgboost"
ROLLING_EVAL_YEARS = [2019, 2020, 2021, 2022]

ID_COLUMNS = [
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "firm_size_group",
    "industry_macro_category",
]
AMOUNT_COLUMNS = ["assets_total", "gross_profit", "depreciation"]
DIVIDEND_CONTEXT_COLUMNS = ["dividend_payer"]
INDUSTRY_YEAR_GROUP = ["fiscal_year", "industry_macro_category"]
MARKET_SIZE_GROUP = ["market", "firm_size_group"]
MARKET_SIZE_YEAR_GROUP = ["fiscal_year", "market", "firm_size_group"]
FOCUS_SEGMENTS = [
    ("overall", "all", None, None),
    ("market", "KOSDAQ", "market", "KOSDAQ"),
    ("market", "KOSPI", "market", "KOSPI"),
    ("industry", "manufacturing", "industry_macro_category", "manufacturing"),
    ("industry", "it_services", "industry_macro_category", "it_services"),
    ("firm_size_group", "large", "firm_size_group", "large"),
    ("firm_size_group", "mid_sized", "firm_size_group", "mid_sized"),
    ("firm_size_group", "small_and_medium", "firm_size_group", "small_and_medium"),
]


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    note: str
    transforms: tuple[str, ...] = ()
    drop_raw_amounts: bool = False


VARIANTS = [
    VariantSpec(
        name="baseline_43_native",
        note="현재 46개 변수, XGBoost native missing 기준",
    ),
    VariantSpec(
        name="amount_log_add_native",
        note="assets_total/gross_profit/depreciation 원값 유지 + signed log1p 추가",
        transforms=("log_amount",),
    ),
    VariantSpec(
        name="amount_log_replace_native",
        note="assets_total/gross_profit/depreciation 원값을 signed log1p로 대체",
        transforms=("log_amount",),
        drop_raw_amounts=True,
    ),
    VariantSpec(
        name="industry_year_amount_pct_add_native",
        note="원값 유지 + 산업-연도 percentile 추가",
        transforms=("industry_year_pct",),
    ),
    VariantSpec(
        name="industry_year_amount_pct_replace_native",
        note="원값을 산업-연도 percentile로 대체",
        transforms=("industry_year_pct",),
        drop_raw_amounts=True,
    ),
    VariantSpec(
        name="market_size_amount_zscore_add_native",
        note="원값 유지 + market-size bucket z-score 추가",
        transforms=("market_size_zscore",),
    ),
    VariantSpec(
        name="market_size_amount_zscore_replace_native",
        note="원값을 market-size bucket z-score로 대체",
        transforms=("market_size_zscore",),
        drop_raw_amounts=True,
    ),
    VariantSpec(
        name="market_size_year_amount_zscore_add_native",
        note="원값 유지 + market-size-year bucket z-score 추가",
        transforms=("market_size_year_zscore",),
    ),
    VariantSpec(
        name="dividend_market_size_context_add_native",
        note="dividend_payer 원값 유지 + market-size peer rate/deviation 추가",
        transforms=("dividend_context",),
    ),
    VariantSpec(
        name="size_context_all_add_native",
        note="원값 유지 + log, 산업-연도 percentile, market-size z-score, 배당 peer context 통합",
        transforms=("log_amount", "industry_year_pct", "market_size_zscore", "dividend_context"),
    ),
    VariantSpec(
        name="size_context_all_replace_native",
        note="금액 원값 제거 + log, 산업-연도 percentile, market-size z-score, 배당 peer context 통합",
        transforms=("log_amount", "industry_year_pct", "market_size_zscore", "dividend_context"),
        drop_raw_amounts=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run size-context feature experiments for the 43-feature Stage 1 XGBoost model."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--eval-years",
        type=int,
        nargs="+",
        default=ROLLING_EVAL_YEARS,
        help="Rolling OOT evaluation years. The previous year is used for calibration/threshold.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    train, valid, test = read_split_frames(args.input_dir)
    id_frames = read_id_frames(args.input_dir)
    master = read_master(args.master_path)

    metrics, segments, rolling_fold_metrics, rolling_summary, selection_table = run_experiments(
        train=train,
        valid=valid,
        test=test,
        id_frames=id_frames,
        master=master,
        eval_years=args.eval_years,
    )
    write_outputs(
        metrics,
        segments,
        rolling_fold_metrics,
        rolling_summary,
        selection_table,
        args.output_dir,
    )
    summary = summarize(metrics, rolling_summary, selection_table)
    print(
        json.dumps(
            {
                "best_by_rolling_validation": summary["best_by_rolling_validation"]["variant"],
                "best_by_test_reference_only": summary["best_by_test_reference_only"]["variant"],
                "historical_aligned_candidate": summary["historical_aligned_candidate"].get(
                    "variant"
                ),
                "report": str(
                    (args.output_dir / "size_context_feature_experiment_report.md").relative_to(
                        ROOT
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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


def read_master(master_path: Path) -> pd.DataFrame:
    master = normalize_keys(
        pd.read_csv(master_path, encoding="utf-8-sig", dtype={"stock_code": str})
    )
    duplicates = master.duplicated(ID_COLUMNS).sum()
    if duplicates:
        raise ValueError(f"feature_43_master has duplicate rows for ID columns: {duplicates}")
    return master


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "stock_code" in output.columns:
        output["stock_code"] = (
            output["stock_code"]
            .astype("string")
            .fillna("")
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True)
        )
        output["stock_code"] = output["stock_code"].where(
            ~output["stock_code"].str.isnumeric(),
            output["stock_code"].str.zfill(6),
        )
    for column in ["fiscal_year", "eval_year", "label_eval_year"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def context_frame(feature_frame: pd.DataFrame, id_frame: pd.DataFrame) -> pd.DataFrame:
    features = feature_frame.reset_index(drop=True).copy()
    ids = normalize_keys(id_frame.reset_index(drop=True)).copy()
    available_ids = [column for column in [*ID_COLUMNS, "label_eval_year"] if column in ids.columns]
    return pd.concat([ids.loc[:, available_ids], features], axis=1)


def run_experiments(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_frames: dict[str, pd.DataFrame],
    master: pd.DataFrame,
    eval_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_feature_columns = [column for column in train.columns if column != "is_speculative"]
    base_frames = {
        "train": context_frame(train, id_frames["train"]),
        "valid": context_frame(valid, id_frames["valid"]),
        "test": context_frame(test, id_frames["test"]),
    }

    metric_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    for spec in VARIANTS:
        frames, feature_columns, added_columns = build_variant_frames(
            spec,
            base_frames=base_frames,
            base_feature_columns=base_feature_columns,
        )
        variant_metrics, variant_segments = evaluate_variant(
            spec=spec,
            frames=frames,
            feature_columns=feature_columns,
            added_columns=added_columns,
        )
        metric_rows.extend(variant_metrics)
        segment_rows.extend(variant_segments)

    metrics = pd.DataFrame(metric_rows)
    segments = pd.DataFrame(segment_rows)
    rolling_fold_metrics = run_rolling_validation(
        master=master,
        base_feature_columns=base_feature_columns,
        eval_years=eval_years,
    )
    rolling_summary = summarize_rolling_metrics(rolling_fold_metrics)
    selection_table = merge_rolling_and_test(rolling_summary, metrics)
    return metrics, segments, rolling_fold_metrics, rolling_summary, selection_table


def run_rolling_validation(
    *,
    master: pd.DataFrame,
    base_feature_columns: list[str],
    eval_years: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for eval_year in eval_years:
        policy_year = eval_year - 1
        fold_base_frames = {
            "train": master.loc[master["fiscal_year"].lt(policy_year)].copy(),
            "valid": master.loc[master["fiscal_year"].eq(policy_year)].copy(),
            "test": master.loc[master["fiscal_year"].eq(eval_year)].copy(),
        }
        if any(frame.empty for frame in fold_base_frames.values()):
            counts = {scope: len(frame) for scope, frame in fold_base_frames.items()}
            raise ValueError(f"Empty rolling split for eval_year={eval_year}: {counts}")
        for spec in VARIANTS:
            frames, feature_columns, added_columns = build_variant_frames(
                spec,
                base_frames=fold_base_frames,
                base_feature_columns=base_feature_columns,
            )
            metric_rows, _ = evaluate_variant(
                spec=spec,
                frames=frames,
                feature_columns=feature_columns,
                added_columns=added_columns,
            )
            policy_row = next(row for row in metric_rows if row["evaluation_scope"] == "valid")
            eval_row = next(row for row in metric_rows if row["evaluation_scope"] == "test")
            rows.append(
                {
                    "variant": spec.name,
                    "note": spec.note,
                    "added_features": ", ".join(added_columns),
                    "added_feature_count": len(added_columns),
                    "eval_year": eval_year,
                    "policy_year": policy_year,
                    "train_year_min": int(fold_base_frames["train"]["fiscal_year"].min()),
                    "train_year_max": int(fold_base_frames["train"]["fiscal_year"].max()),
                    "train_rows": len(fold_base_frames["train"]),
                    "policy_rows": len(fold_base_frames["valid"]),
                    "eval_rows": len(fold_base_frames["test"]),
                    "threshold_tuned_on_policy_year": eval_row["threshold_tuned"],
                    "policy_pr_auc": policy_row["pr_auc"],
                    "policy_precision": policy_row["classification_precision"],
                    "policy_recall": policy_row["classification_recall"],
                    "policy_f1": policy_row["classification_f1"],
                    "eval_pr_auc": eval_row["pr_auc"],
                    "eval_roc_auc": eval_row["roc_auc"],
                    "eval_brier": eval_row["brier"],
                    "eval_logloss": eval_row["logloss"],
                    "eval_ece": eval_row["ece"],
                    "eval_precision": eval_row["classification_precision"],
                    "eval_recall": eval_row["classification_recall"],
                    "eval_f1": eval_row["classification_f1"],
                    "eval_true_negative": eval_row["classification_true_negative"],
                    "eval_false_positive": eval_row["classification_false_positive"],
                    "eval_false_negative": eval_row["classification_false_negative"],
                    "eval_true_positive": eval_row["classification_true_positive"],
                }
            )
    return pd.DataFrame(rows)


def build_variant_frames(
    spec: VariantSpec,
    *,
    base_frames: dict[str, pd.DataFrame],
    base_feature_columns: list[str],
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    frames = {scope: frame.copy() for scope, frame in base_frames.items()}
    feature_columns = list(base_feature_columns)
    added_columns: list[str] = []

    for transform in spec.transforms:
        transform_added: list[str] = []
        for scope, frame in frames.items():
            transformed, split_added = apply_transform(frame, transform)
            frames[scope] = transformed
            transform_added = split_added
        added_columns = unique_preserve_order([*added_columns, *transform_added])

    if spec.drop_raw_amounts:
        feature_columns = [column for column in feature_columns if column not in AMOUNT_COLUMNS]
    feature_columns = unique_preserve_order([*feature_columns, *added_columns])
    return frames, feature_columns, added_columns


def apply_transform(frame: pd.DataFrame, transform: str) -> tuple[pd.DataFrame, list[str]]:
    if transform == "log_amount":
        return add_signed_log_features(frame, AMOUNT_COLUMNS)
    if transform == "industry_year_pct":
        return add_group_percentile_features(
            frame,
            group_columns=INDUSTRY_YEAR_GROUP,
            value_columns=AMOUNT_COLUMNS,
            suffix="industry_year",
        )
    if transform == "market_size_zscore":
        return add_group_zscore_features(
            frame,
            group_columns=MARKET_SIZE_GROUP,
            value_columns=AMOUNT_COLUMNS,
            suffix="market_size",
        )
    if transform == "market_size_year_zscore":
        return add_group_zscore_features(
            frame,
            group_columns=MARKET_SIZE_YEAR_GROUP,
            value_columns=AMOUNT_COLUMNS,
            suffix="market_size_year",
        )
    if transform == "dividend_context":
        return add_binary_group_context_features(
            frame,
            group_columns=MARKET_SIZE_GROUP,
            value_columns=DIVIDEND_CONTEXT_COLUMNS,
            suffix="market_size",
        )
    raise ValueError(f"Unknown transform: {transform}")


def evaluate_variant(
    *,
    spec: VariantSpec,
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    added_columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y_train = frames["train"]["is_speculative"].astype(int)
    y_valid = frames["valid"]["is_speculative"].astype(int)
    model = train_xgboost(
        frames["train"].loc[:, feature_columns],
        y_train,
        frames["valid"].loc[:, feature_columns],
        y_valid,
    )
    valid_raw = as_float_array(model.predict_proba(frames["valid"].loc[:, feature_columns])[:, 1])
    calibration = fit_platt_calibration(y_valid, valid_raw)
    valid_probabilities = apply_probability_calibration(valid_raw, calibration)
    threshold = choose_tuned_threshold(y_valid, valid_probabilities, recall_floor=RECALL_FLOOR)

    metric_rows: list[dict[str, Any]] = []
    scored_frames: dict[str, pd.DataFrame] = {}
    for scope in [scope for scope in ["valid", "test"] if scope in frames]:
        frame = frames[scope]
        y_true = frame["is_speculative"].astype(int)
        raw_probabilities = as_float_array(model.predict_proba(frame.loc[:, feature_columns])[:, 1])
        probabilities = apply_probability_calibration(raw_probabilities, calibration)
        quality = probability_quality_metrics(y_true, probabilities)
        classification = classification_metrics_at_threshold(
            y_true,
            probabilities,
            threshold=threshold,
        )
        metric_rows.append(
            {
                "variant": spec.name,
                "note": spec.note,
                "evaluation_scope": scope,
                "feature_count": len(feature_columns),
                "added_feature_count": len(added_columns),
                "added_features": ", ".join(added_columns),
                "best_iteration": getattr(model, "best_iteration", None),
                "threshold_tuned": threshold,
                **quality,
                **{f"classification_{key}": value for key, value in classification.items()},
            }
        )
        scored_frames[scope] = frame.assign(
            prob_speculative=probabilities,
            prediction=(probabilities >= threshold).astype(int),
        )

    segment_rows = []
    for scope in [scope for scope in ["test"] if scope in scored_frames]:
        segment_rows.extend(build_segment_rows(spec.name, scope, scored_frames[scope]))
    return metric_rows, segment_rows


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


def build_segment_rows(variant: str, scope: str, scored: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension, segment, column, value in FOCUS_SEGMENTS:
        segment_frame = (
            scored if column is None else scored.loc[scored[column].astype(str).eq(str(value))]
        )
        if segment_frame.empty:
            continue
        y_true = segment_frame["is_speculative"].astype(int)
        probabilities = segment_frame["prob_speculative"].to_numpy(dtype=np.float64)
        predictions = segment_frame["prediction"].astype(int)
        true_positive = int(((y_true == 1) & (predictions == 1)).sum())
        true_negative = int(((y_true == 0) & (predictions == 0)).sum())
        false_positive = int(((y_true == 0) & (predictions == 1)).sum())
        false_negative = int(((y_true == 1) & (predictions == 0)).sum())
        positives = int((y_true == 1).sum())
        negatives = int((y_true == 0).sum())
        probability_quality = probability_quality_metrics(y_true, probabilities)
        rows.append(
            {
                "variant": variant,
                "evaluation_scope": scope,
                "dimension": dimension,
                "segment": segment,
                "rows": len(segment_frame),
                "positives": positives,
                "negatives": negatives,
                "precision": true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0,
                "recall": true_positive / positives if positives else 0.0,
                "f1": (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
                if 2 * true_positive + false_positive + false_negative
                else 0.0,
                "true_positive": true_positive,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "fp_rate_among_negatives": false_positive / negatives if negatives else None,
                "fn_rate_among_positives": false_negative / positives if positives else None,
                "mean_probability": probability_quality["mean_probability"],
                "ece": probability_quality["ece"],
            }
        )
    return rows


def summarize(
    metrics: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    selection_table: pd.DataFrame,
) -> dict[str, Any]:
    test = metrics.loc[metrics["evaluation_scope"].eq("test")].copy()
    baseline_test = row_by_variant(test, "baseline_43_native")
    baseline_rolling = row_by_variant(rolling_summary, "baseline_43_native")
    best_rolling = sort_rolling_rows(rolling_summary).iloc[0].to_dict()
    best_test = sort_metric_rows(test).iloc[0].to_dict()
    aligned = selection_table.loc[
        selection_table["rolling_f1_delta_vs_baseline"].gt(0.0)
        & selection_table["test_f1_delta_vs_baseline"].gt(0.0)
    ].copy()
    aligned_candidate = sort_selection_rows(aligned).iloc[0].to_dict() if not aligned.empty else {}
    fp_reduction = selection_table.loc[
        selection_table["rolling_f1_delta_vs_baseline"].ge(0.0)
        & selection_table["test_f1_delta_vs_baseline"].ge(0.0)
        & selection_table["test_fp_delta_vs_baseline"].lt(0)
    ].copy()
    fp_reduction_candidate = (
        sort_selection_rows(fp_reduction).iloc[0].to_dict() if not fp_reduction.empty else {}
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model_name": MODEL_NAME,
        "decision": "rolling_validation_and_final_test_only",
        "threshold_policy": f"policy-year max precision with recall >= {RECALL_FLOOR:.2f}",
        "rolling_eval_years": ROLLING_EVAL_YEARS,
        "baseline_test": baseline_test,
        "baseline_rolling_validation": baseline_rolling,
        "best_by_rolling_validation": best_rolling,
        "best_by_test_reference_only": best_test,
        "historical_aligned_candidate": aligned_candidate,
        "fp_reduction_candidate": fp_reduction_candidate,
    }


def summarize_rolling_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "eval_pr_auc",
        "eval_roc_auc",
        "eval_brier",
        "eval_logloss",
        "eval_ece",
        "eval_precision",
        "eval_recall",
        "eval_f1",
        "eval_false_positive",
        "eval_false_negative",
    ]
    rows: list[dict[str, Any]] = []
    for variant, group in fold_metrics.groupby("variant", sort=False):
        row: dict[str, Any] = {
            "variant": variant,
            "note": group["note"].iloc[0],
            "added_features": group["added_features"].iloc[0],
            "added_feature_count": int(group["added_feature_count"].iloc[0]),
            "folds": len(group),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_max"] = float(group[column].max())
        row["total_false_positive"] = int(group["eval_false_positive"].sum())
        row["total_false_negative"] = int(group["eval_false_negative"].sum())
        rows.append(row)
    return sort_rolling_rows(pd.DataFrame(rows))


def merge_rolling_and_test(rolling_summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["evaluation_scope"].eq("test")].copy()
    test_columns = [
        "variant",
        "pr_auc",
        "roc_auc",
        "classification_precision",
        "classification_recall",
        "classification_f1",
        "classification_false_positive",
        "classification_false_negative",
        "brier",
        "ece",
        "threshold_tuned",
    ]
    selection = rolling_summary.merge(
        test.loc[:, test_columns].rename(
            columns={
                "pr_auc": "test_pr_auc",
                "roc_auc": "test_roc_auc",
                "classification_precision": "test_precision",
                "classification_recall": "test_recall",
                "classification_f1": "test_f1",
                "classification_false_positive": "test_false_positive",
                "classification_false_negative": "test_false_negative",
                "brier": "test_brier",
                "ece": "test_ece",
            }
        ),
        on="variant",
        how="left",
    )
    baseline = selection.loc[selection["variant"].eq("baseline_43_native")].iloc[0]
    selection["rolling_f1_delta_vs_baseline"] = selection["eval_f1_mean"] - baseline["eval_f1_mean"]
    selection["rolling_pr_auc_delta_vs_baseline"] = (
        selection["eval_pr_auc_mean"] - baseline["eval_pr_auc_mean"]
    )
    selection["test_f1_delta_vs_baseline"] = selection["test_f1"] - baseline["test_f1"]
    selection["test_pr_auc_delta_vs_baseline"] = selection["test_pr_auc"] - baseline["test_pr_auc"]
    selection["test_fp_delta_vs_baseline"] = (
        selection["test_false_positive"] - baseline["test_false_positive"]
    )
    selection["test_fn_delta_vs_baseline"] = (
        selection["test_false_negative"] - baseline["test_false_negative"]
    )
    return sort_selection_rows(selection)


def sort_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "classification_f1",
            "pr_auc",
            "classification_precision",
            "classification_recall",
        ],
        ascending=[False, False, False, False],
    )


def sort_rolling_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["eval_f1_mean", "eval_pr_auc_mean", "eval_precision_mean", "eval_recall_mean"],
        ascending=[False, False, False, False],
    )


def sort_selection_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        [
            "rolling_f1_delta_vs_baseline",
            "test_f1_delta_vs_baseline",
            "test_fp_delta_vs_baseline",
            "test_pr_auc_delta_vs_baseline",
        ],
        ascending=[False, False, True, False],
    )


def row_by_variant(frame: pd.DataFrame, variant: str) -> dict[str, Any]:
    selected = frame.loc[frame["variant"].eq(variant)]
    if selected.empty:
        return {}
    return selected.iloc[0].to_dict()


def write_outputs(
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    rolling_fold_metrics: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    selection_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "size_context_feature_experiment_metrics.csv"
    segments_path = output_dir / "size_context_feature_experiment_segment_metrics.csv"
    rolling_fold_path = output_dir / "size_context_feature_rolling_fold_metrics.csv"
    rolling_summary_path = output_dir / "size_context_feature_rolling_summary.csv"
    selection_path = output_dir / "size_context_feature_selection_test_comparison.csv"
    summary_path = output_dir / "size_context_feature_experiment_summary.json"
    report_path = output_dir / "size_context_feature_experiment_report.md"
    summary = summarize(metrics, rolling_summary, selection_table)

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")
    rolling_fold_metrics.to_csv(rolling_fold_path, index=False, encoding="utf-8-sig")
    rolling_summary.to_csv(rolling_summary_path, index=False, encoding="utf-8-sig")
    selection_table.to_csv(selection_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report_path.write_text(
        build_report(metrics, segments, rolling_summary, selection_table, summary),
        encoding="utf-8",
    )


def build_report(
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    selection_table: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    baseline_test = summary["baseline_test"]
    baseline_rolling = summary["baseline_rolling_validation"]
    best_rolling = summary["best_by_rolling_validation"]
    best_test = summary["best_by_test_reference_only"]
    aligned = summary["historical_aligned_candidate"]
    fp_candidate = summary["fp_reduction_candidate"]
    test_table = metrics.loc[metrics["evaluation_scope"].eq("test")].copy()
    kosdaq_table = segments.loc[
        segments["evaluation_scope"].eq("test") & segments["segment"].eq("KOSDAQ")
    ].copy()
    return "\n".join(
        [
            "# Feature 43 Size-Context Feature Experiments",
            "",
            "절대규모 변수 과민반응을 줄이기 위해 log 변환, 산업-연도 percentile, "
            "market-size bucket z-score, dividend peer context를 비교했습니다.",
            "후보 선택은 rolling OOT validation으로 보고, final test는 마지막 확인용으로만 사용합니다.",
            "2026 external은 표본이 작아 이 리포트의 선택 기준에서 제외했습니다.",
            "",
            "## Recommendation",
            "",
            f"- Decision: `{summary['decision']}`",
            f"- Baseline rolling F1/PR-AUC: `{format_metric(baseline_rolling['eval_f1_mean'])}` / "
            f"`{format_metric(baseline_rolling['eval_pr_auc_mean'])}`",
            f"- Baseline final test F1/FP/FN: `{format_metric(baseline_test['classification_f1'])}` / "
            f"`{format_count(baseline_test['classification_false_positive'])}` / "
            f"`{format_count(baseline_test['classification_false_negative'])}`",
            f"- Best by rolling validation: `{best_rolling['variant']}` "
            f"(rolling F1 `{format_metric(best_rolling['eval_f1_mean'])}`, "
            f"rolling PR-AUC `{format_metric(best_rolling['eval_pr_auc_mean'])}`)",
            f"- Best by test reference: `{best_test['variant']}` "
            f"(test F1 `{format_metric(best_test['classification_f1'])}`)",
            "- Historical aligned candidate: "
            + (
                f"`{aligned['variant']}`"
                if aligned
                else "`none` because no candidate improved both rolling F1 and final test F1"
            ),
            "- FP-reduction candidate: "
            + (
                f"`{fp_candidate['variant']}`"
                if fp_candidate
                else "`none` under rolling F1 >= baseline and final test F1 >= baseline"
            ),
            "",
            "## Rolling Validation Summary",
            "",
            rolling_table(rolling_summary),
            "",
            "## Rolling Selection + Final Test",
            "",
            selection_table_markdown(selection_table),
            "",
            "## Final Test Metrics",
            "",
            metrics_table(test_table),
            "",
            "## KOSDAQ Test Segment",
            "",
            segment_table(kosdaq_table),
            "",
            "## Interpretation",
            "",
            "- `*_replace_native` variants remove raw amount columns and rely on relative/context features.",
            "- `industry_year_amount_pct_*` compares amount levels inside the same fiscal year and industry.",
            "- `market_size_amount_zscore_*` compares amount levels inside market and firm-size buckets.",
            "- `dividend_market_size_context_add_native` keeps the binary dividend flag but adds peer rate/deviation.",
            "- Rolling validation is the selection signal; final test is a retrospective confirmation signal.",
            "",
        ]
    )


def rolling_table(frame: pd.DataFrame) -> str:
    ordered = sort_rolling_rows(frame)
    lines = [
        "| Variant | Folds | Roll PR-AUC | Roll Precision | Roll Recall | Roll F1 | Roll FP | Roll FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['variant']} | {format_count(row['folds'])} | "
            f"{format_metric(row['eval_pr_auc_mean'])} | "
            f"{format_metric(row['eval_precision_mean'])} | "
            f"{format_metric(row['eval_recall_mean'])} | "
            f"{format_metric(row['eval_f1_mean'])} | "
            f"{format_count(row['total_false_positive'])} | "
            f"{format_count(row['total_false_negative'])} |"
        )
    return "\n".join(lines)


def selection_table_markdown(frame: pd.DataFrame) -> str:
    ordered = sort_selection_rows(frame)
    lines = [
        "| Variant | Roll F1 | Roll ΔF1 | Test PR-AUC | Test Precision | Test Recall | Test F1 | Test ΔF1 | Test ΔFP | Test ΔFN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['variant']} | {format_metric(row['eval_f1_mean'])} | "
            f"{format_metric(row['rolling_f1_delta_vs_baseline'])} | "
            f"{format_metric(row['test_pr_auc'])} | "
            f"{format_metric(row['test_precision'])} | "
            f"{format_metric(row['test_recall'])} | "
            f"{format_metric(row['test_f1'])} | "
            f"{format_metric(row['test_f1_delta_vs_baseline'])} | "
            f"{format_count(row['test_fp_delta_vs_baseline'])} | "
            f"{format_count(row['test_fn_delta_vs_baseline'])} |"
        )
    return "\n".join(lines)


def metrics_table(frame: pd.DataFrame) -> str:
    ordered = sort_metric_rows(frame)
    lines = [
        "| Variant | Features | Added | PR-AUC | Precision | Recall | F1 | FP | FN | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['variant']} | {format_count(row['feature_count'])} | "
            f"{format_count(row['added_feature_count'])} | "
            f"{format_metric(row['pr_auc'])} | "
            f"{format_metric(row['classification_precision'])} | "
            f"{format_metric(row['classification_recall'])} | "
            f"{format_metric(row['classification_f1'])} | "
            f"{format_count(row['classification_false_positive'])} | "
            f"{format_count(row['classification_false_negative'])} | "
            f"{format_metric(row['brier'])} | "
            f"{format_metric(row['ece'])} |"
        )
    return "\n".join(lines)


def segment_table(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(
        ["false_positive", "false_negative", "f1"], ascending=[True, True, False]
    )
    lines = [
        "| Variant | Rows | Precision | Recall | F1 | FP | FN | FP Rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['variant']} | {format_count(row['rows'])} | "
            f"{format_metric(row['precision'])} | "
            f"{format_metric(row['recall'])} | "
            f"{format_metric(row['f1'])} | "
            f"{format_count(row['false_positive'])} | "
            f"{format_count(row['false_negative'])} | "
            f"{format_metric(row['fp_rate_among_negatives'])} |"
        )
    return "\n".join(lines)


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_count(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(int(float(value)))


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def as_float_array(values: npt.ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


if __name__ == "__main__":
    main()
