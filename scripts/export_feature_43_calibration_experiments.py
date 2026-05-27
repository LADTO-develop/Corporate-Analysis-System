from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

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
    as_float_array,
    beta_calibration_features,
    calibration_bin_table,
    classification_metrics_at_threshold,
    probability_quality_metrics,
)

PREDICTION_SCORES_PATH = (
    ROOT / "data" / "outputs" / "dashboard" / "feature_46_mvp" / "prediction_scores.csv"
)
MODEL_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost"
MODEL_PATH = MODEL_DIR / "xgboost_model.json"
METADATA_PATH = MODEL_DIR / "model_artifact_metadata.json"
INFERENCE_2026_PATH = (
    ROOT / "data" / "input" / "credit_46_features" / "feature_46_inference_2026.csv"
)
LABELS_2026_PATH = ROOT / "data" / "evaluation" / "credit_rating_labels_2026.csv"
OUTPUT_DIR = MODEL_DIR / "diagnostics"
MODEL_NAME = "feature_46_xgboost"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"
RECALL_FLOOR = 0.85
MIN_SEGMENT_ROWS = 40

FloatArray = npt.NDArray[np.float64]
ProbabilityFn = Callable[[pd.DataFrame], FloatArray]


@dataclass(frozen=True)
class CalibrationVariant:
    """A fitted calibration variant."""

    name: str
    display_name: str
    fit_scope: str
    probability_fn: ProbabilityFn
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare probability calibration candidates for the 43-feature Stage 1 model. "
            "The 2026 labels are used only as an external audit, not for fitting."
        )
    )
    parser.add_argument("--prediction-scores", type=Path, default=PREDICTION_SCORES_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--inference-2026", type=Path, default=INFERENCE_2026_PATH)
    parser.add_argument("--labels-2026", type=Path, default=LABELS_2026_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-segment-rows", type=int, default=MIN_SEGMENT_ROWS)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    historical = read_historical_scores(args.prediction_scores)
    external = score_external_2026(
        inference_path=args.inference_2026,
        labels_path=args.labels_2026,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
    )

    variants = build_variants(historical, min_segment_rows=args.min_segment_rows)
    metrics = build_metric_table(historical=historical, external=external, variants=variants)
    bins = build_bin_table(historical=historical, external=external, variants=variants)
    segment_metrics = build_segment_metric_table(
        historical=historical,
        external=external,
        variants=variants,
    )
    summary = build_summary(
        metrics=metrics,
        bins=bins,
        segment_metrics=segment_metrics,
        min_segment_rows=args.min_segment_rows,
    )
    report = build_report(summary, metrics, bins)

    metrics_path = args.output_dir / "calibration_experiment_metrics.csv"
    bins_path = args.output_dir / "calibration_experiment_bins.csv"
    segment_path = args.output_dir / "calibration_experiment_segment_metrics.csv"
    summary_path = args.output_dir / "calibration_experiment_summary.json"
    report_path = args.output_dir / "calibration_experiment_report.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    bins.to_csv(bins_path, index=False, encoding="utf-8-sig")
    segment_metrics.to_csv(segment_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "metrics": str(metrics_path.relative_to(ROOT)),
                "bins": str(bins_path.relative_to(ROOT)),
                "segment_metrics": str(segment_path.relative_to(ROOT)),
                "summary": str(summary_path.relative_to(ROOT)),
                "report": str(report_path.relative_to(ROOT)),
                "recommended_variant": summary["recommendation"]["variant"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_historical_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    required = {
        "split",
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "label_eval_year",
        "firm_size_group",
        "industry_macro_category",
        "is_speculative",
        "prob_speculative_raw",
        "prob_speculative",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"prediction_scores.csv is missing required columns: {missing}")
    output = frame.copy()
    output["evaluation_scope"] = "historical"
    output["actual_is_speculative"] = pd.to_numeric(
        output["is_speculative"], errors="coerce"
    ).astype(int)
    output["prob_speculative_raw"] = pd.to_numeric(output["prob_speculative_raw"], errors="coerce")
    output["prob_speculative"] = pd.to_numeric(output["prob_speculative"], errors="coerce")
    for column in ["fiscal_year", "eval_year", "label_eval_year"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def score_external_2026(
    *,
    inference_path: Path,
    labels_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    import xgboost as xgb

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_columns = [str(column) for column in metadata["feature_columns"]]
    inference = pd.read_csv(inference_path, encoding="utf-8-sig", dtype={"stock_code": str})
    labels = pd.read_csv(labels_path, encoding="utf-8-sig", dtype={"stock_code": str})
    inference = normalize_keys(inference)
    labels = normalize_keys(labels)

    model_frame = build_model_frame(inference, feature_columns)
    booster = xgb.Booster()
    booster.load_model(model_path)
    raw_probabilities = as_float_array(booster.predict(xgb.DMatrix(model_frame)))
    current_probabilities = apply_probability_calibration(
        raw_probabilities,
        cast(dict[str, object], metadata.get("probability_calibration") or {}),
    )
    scored = inference.copy()
    scored["prob_speculative_raw"] = raw_probabilities
    scored["prob_speculative"] = current_probabilities

    label_columns = [
        "market",
        "stock_code",
        "fiscal_year",
        "eval_year",
        "is_speculative",
        "credit_rating",
        "credit_rating_rank",
        "rating_agency",
    ]
    merged = scored.merge(
        labels.loc[:, [column for column in label_columns if column in labels.columns]],
        on=["market", "stock_code", "fiscal_year", "eval_year"],
        how="inner",
        validate="one_to_one",
    )
    merged["split"] = "external_2026"
    merged["evaluation_scope"] = "external_2026"
    merged["label_eval_year"] = merged["eval_year"]
    merged["actual_is_speculative"] = pd.to_numeric(
        merged["is_speculative"], errors="coerce"
    ).astype(int)
    return merged


def build_model_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        output[column] = pd.to_numeric(
            frame[column] if column in frame.columns else np.nan,
            errors="coerce",
        )
    return output.loc[:, feature_columns]


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
    for column in ["fiscal_year", "eval_year"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def build_variants(scores: pd.DataFrame, *, min_segment_rows: int) -> list[CalibrationVariant]:
    valid = scores.loc[scores["split"].astype(str).eq(VALID_SPLIT)].copy()
    if valid.empty:
        raise ValueError("No validation rows available for calibration fitting.")
    y_valid = valid["actual_is_speculative"].astype(int)
    raw_valid = valid["prob_speculative_raw"].to_numpy(dtype=np.float64)

    platt = fit_platt_calibration(y_valid, raw_valid)
    isotonic = fit_isotonic(raw_valid, y_valid)
    beta = fit_beta(raw_valid, y_valid)
    rolling = fit_platt_calibration(
        scores.loc[scores["split"].astype(str).ne(TEST_SPLIT), "actual_is_speculative"].astype(int),
        scores.loc[
            scores["split"].astype(str).ne(TEST_SPLIT),
            "prob_speculative_raw",
        ].to_numpy(dtype=np.float64),
    )
    latest_year_calibrations = fit_latest_year_platt(scores)
    market_calibrations = fit_segment_platt(
        valid,
        group_columns=["market"],
        min_segment_rows=min_segment_rows,
        fallback=platt,
    )
    market_size_calibrations = fit_segment_platt(
        valid,
        group_columns=["market", "firm_size_group"],
        min_segment_rows=min_segment_rows,
        fallback=platt,
    )

    return [
        CalibrationVariant(
            name="raw_xgboost",
            display_name="Raw XGBoost",
            fit_scope="none",
            probability_fn=lambda frame: frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
            note="Uncalibrated model output.",
        ),
        CalibrationVariant(
            name="current_platt",
            display_name="Current Platt",
            fit_scope="saved_valid_platt",
            probability_fn=lambda frame: frame["prob_speculative"].to_numpy(dtype=np.float64),
            note="Current dashboard probability from saved Platt scaling.",
        ),
        CalibrationVariant(
            name="refit_platt_valid",
            display_name="Refit Platt",
            fit_scope="valid",
            probability_fn=lambda frame, calibration=platt: apply_probability_calibration(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                calibration,
            ),
            note="Refit Platt scaling on the validation split.",
        ),
        CalibrationVariant(
            name="isotonic_valid",
            display_name="Isotonic",
            fit_scope="valid",
            probability_fn=lambda frame, model=isotonic: as_float_array(
                model.predict(frame["prob_speculative_raw"].to_numpy(dtype=np.float64))
            ),
            note="Non-parametric isotonic regression fitted on validation.",
        ),
        CalibrationVariant(
            name="beta_valid",
            display_name="Beta",
            fit_scope="valid",
            probability_fn=lambda frame, model=beta: as_float_array(
                model.predict_proba(
                    beta_calibration_features(
                        frame["prob_speculative_raw"].to_numpy(dtype=np.float64)
                    )
                )[:, 1]
            ),
            note="Logistic beta calibration fitted on validation.",
        ),
        CalibrationVariant(
            name="rolling_expanding_platt",
            display_name="Rolling Expanding Platt",
            fit_scope="historical_pre_test_for_test_all_history_for_2026",
            probability_fn=lambda frame, calibration=rolling: apply_probability_calibration(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                calibration,
            ),
            note=(
                "Platt fitted on all historical rows before final test; for 2026 this uses "
                "the historical score table only."
            ),
        ),
        CalibrationVariant(
            name="latest_year_platt",
            display_name="Latest-Year Platt",
            fit_scope="latest_prior_label_year",
            probability_fn=lambda frame, calibrations=latest_year_calibrations, fallback=platt: (
                apply_latest_year_platt(frame, calibrations, fallback)
            ),
            note="Fit on the latest labeled year before the target year.",
        ),
        CalibrationVariant(
            name="segment_market_platt",
            display_name="Market Platt",
            fit_scope="valid_market_with_global_fallback",
            probability_fn=lambda frame, calibrations=market_calibrations, fallback=platt: (
                apply_segment_platt(
                    frame,
                    calibrations=calibrations,
                    fallback=fallback,
                    group_columns=["market"],
                )
            ),
            note="Market-specific Platt when validation rows are sufficient; global fallback.",
        ),
        CalibrationVariant(
            name="segment_market_size_platt",
            display_name="Market+Size Platt",
            fit_scope="valid_market_size_with_global_fallback",
            probability_fn=lambda frame, calibrations=market_size_calibrations, fallback=platt: (
                apply_segment_platt(
                    frame,
                    calibrations=calibrations,
                    fallback=fallback,
                    group_columns=["market", "firm_size_group"],
                )
            ),
            note=(
                "Market and firm-size specific Platt when validation rows are sufficient; "
                "global fallback."
            ),
        ),
    ]


def fit_isotonic(raw_probabilities: FloatArray, y_true: pd.Series) -> object:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(raw_probabilities, y_true.astype(int))
    return model


def fit_beta(raw_probabilities: FloatArray, y_true: pd.Series) -> object:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
    model.fit(beta_calibration_features(raw_probabilities), y_true.astype(int))
    return model


def fit_segment_platt(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    min_segment_rows: int,
    fallback: dict[str, object],
) -> dict[tuple[str, ...], dict[str, object]]:
    calibrations: dict[tuple[str, ...], dict[str, object]] = {}
    for keys, group in frame.groupby(group_columns, dropna=False):
        key_tuple = tuple(str(value) for value in (keys if isinstance(keys, tuple) else (keys,)))
        if len(group) < min_segment_rows or group["actual_is_speculative"].nunique() < 2:
            continue
        calibrations[key_tuple] = fit_platt_calibration(
            group["actual_is_speculative"].astype(int),
            group["prob_speculative_raw"].to_numpy(dtype=np.float64),
        )
    if not calibrations:
        calibrations[("__global__",)] = fallback
    return calibrations


def fit_latest_year_platt(scores: pd.DataFrame) -> dict[int, dict[str, object]]:
    calibrations: dict[int, dict[str, object]] = {}
    years = sorted(
        int(year) for year in scores["label_eval_year"].dropna().unique() if not pd.isna(year)
    )
    for target_year in years:
        prior_years = [year for year in years if year < target_year]
        if not prior_years:
            continue
        fit_year = max(prior_years)
        fit_frame = scores.loc[scores["label_eval_year"].eq(fit_year)].copy()
        if len(fit_frame) < 20 or fit_frame["actual_is_speculative"].nunique() < 2:
            continue
        calibrations[target_year] = fit_platt_calibration(
            fit_frame["actual_is_speculative"].astype(int),
            fit_frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
        )
    return calibrations


def apply_latest_year_platt(
    frame: pd.DataFrame,
    calibrations: dict[int, dict[str, object]],
    fallback: dict[str, object],
) -> FloatArray:
    output = pd.Series(np.nan, index=frame.index, dtype="float64")
    for year, group_index in frame.groupby("label_eval_year", dropna=False).groups.items():
        target_year = int(year) if not pd.isna(year) else -1
        calibration = calibrations.get(target_year, fallback)
        raw = frame.loc[group_index, "prob_speculative_raw"].to_numpy(dtype=np.float64)
        output.loc[group_index] = apply_probability_calibration(raw, calibration)
    return output.to_numpy(dtype=np.float64)


def apply_segment_platt(
    frame: pd.DataFrame,
    *,
    calibrations: dict[tuple[str, ...], dict[str, object]],
    fallback: dict[str, object],
    group_columns: list[str],
) -> FloatArray:
    output = pd.Series(np.nan, index=frame.index, dtype="float64")
    for keys, group_index in frame.groupby(group_columns, dropna=False).groups.items():
        key_tuple = tuple(str(value) for value in (keys if isinstance(keys, tuple) else (keys,)))
        calibration = calibrations.get(key_tuple, fallback)
        raw = frame.loc[group_index, "prob_speculative_raw"].to_numpy(dtype=np.float64)
        output.loc[group_index] = apply_probability_calibration(raw, calibration)
    return output.to_numpy(dtype=np.float64)


def build_metric_table(
    *,
    historical: pd.DataFrame,
    external: pd.DataFrame,
    variants: list[CalibrationVariant],
) -> pd.DataFrame:
    frames = {
        "valid": historical.loc[historical["split"].astype(str).eq(VALID_SPLIT)].copy(),
        "test": historical.loc[historical["split"].astype(str).eq(TEST_SPLIT)].copy(),
        "external_2026": external.copy(),
    }
    valid_labels = frames["valid"]["actual_is_speculative"].astype(int)
    rows: list[dict[str, object]] = []
    for variant in variants:
        valid_probabilities = variant.probability_fn(frames["valid"])
        threshold = float(
            choose_tuned_threshold(
                valid_labels,
                valid_probabilities,
                recall_floor=RECALL_FLOOR,
            )
        )
        for scope, frame in frames.items():
            probabilities = variant.probability_fn(frame)
            y_true = frame["actual_is_speculative"].astype(int)
            quality = probability_quality_metrics(y_true, probabilities)
            classification = classification_metrics_at_threshold(
                y_true,
                probabilities,
                threshold=threshold,
            )
            rows.append(
                {
                    "variant": variant.name,
                    "display_name": variant.display_name,
                    "fit_scope": variant.fit_scope,
                    "evaluation_scope": scope,
                    "note": variant.note,
                    **quality,
                    **{f"classification_{key}": value for key, value in classification.items()},
                }
            )
    return pd.DataFrame(rows)


def build_bin_table(
    *,
    historical: pd.DataFrame,
    external: pd.DataFrame,
    variants: list[CalibrationVariant],
) -> pd.DataFrame:
    frames = {
        "valid": historical.loc[historical["split"].astype(str).eq(VALID_SPLIT)].copy(),
        "test": historical.loc[historical["split"].astype(str).eq(TEST_SPLIT)].copy(),
        "external_2026": external.copy(),
    }
    rows: list[pd.DataFrame] = []
    for variant in variants:
        for scope, frame in frames.items():
            probabilities = pd.Series(variant.probability_fn(frame), index=frame.index)
            bins = calibration_bin_table(frame["actual_is_speculative"].astype(int), probabilities)
            bins.insert(0, "evaluation_scope", scope)
            bins.insert(0, "display_name", variant.display_name)
            bins.insert(0, "variant", variant.name)
            rows.append(bins)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_segment_metric_table(
    *,
    historical: pd.DataFrame,
    external: pd.DataFrame,
    variants: list[CalibrationVariant],
) -> pd.DataFrame:
    frames = {
        "test": historical.loc[historical["split"].astype(str).eq(TEST_SPLIT)].copy(),
        "external_2026": external.copy(),
    }
    segments = [
        ("overall", "all", None, None),
        ("market", "KOSDAQ", "market", "KOSDAQ"),
        ("market", "KOSPI", "market", "KOSPI"),
        ("firm_size_group", "large", "firm_size_group", "large"),
        ("firm_size_group", "mid_sized", "firm_size_group", "mid_sized"),
        ("firm_size_group", "small_and_medium", "firm_size_group", "small_and_medium"),
    ]
    rows: list[dict[str, object]] = []
    for variant in variants:
        for scope, frame in frames.items():
            probabilities = variant.probability_fn(frame)
            scored = frame.assign(_calibrated_probability=probabilities)
            for dimension, segment, column, value in segments:
                segment_frame = (
                    scored
                    if column is None
                    else scored.loc[scored[column].astype(str).eq(str(value))]
                )
                if segment_frame.empty:
                    continue
                metrics = probability_quality_metrics(
                    segment_frame["actual_is_speculative"].astype(int),
                    segment_frame["_calibrated_probability"].to_numpy(dtype=np.float64),
                )
                rows.append(
                    {
                        "variant": variant.name,
                        "display_name": variant.display_name,
                        "evaluation_scope": scope,
                        "dimension": dimension,
                        "segment": segment,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def build_summary(
    *,
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    min_segment_rows: int,
) -> dict[str, object]:
    test_metrics = metrics.loc[metrics["evaluation_scope"].eq("test")].copy()
    external_metrics = metrics.loc[metrics["evaluation_scope"].eq("external_2026")].copy()
    current_test = row_by_variant(test_metrics, "current_platt")
    best_test_ece = best_row(test_metrics, "ece")
    best_test_brier = best_row(test_metrics, "brier")
    best_external_ece = best_row(external_metrics, "ece")
    recommendation = choose_recommendation(test_metrics, external_metrics)
    largest_current_test_gaps = (
        bins.loc[bins["evaluation_scope"].eq("test") & bins["variant"].eq("current_platt")]
        .assign(abs_gap=lambda frame: frame["calibration_gap"].abs())
        .sort_values("abs_gap", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model_name": MODEL_NAME,
        "decision": "experiment_only_do_not_replace_dashboard_calibration_yet",
        "selection_note": (
            "Choose from validation-fitted calibrators using historical test calibration metrics; "
            "2026 labels are external audit only."
        ),
        "min_segment_rows": min_segment_rows,
        "current_test": current_test,
        "best_test_ece": best_test_ece,
        "best_test_brier": best_test_brier,
        "best_external_2026_ece": best_external_ece,
        "recommendation": recommendation,
        "largest_current_test_calibration_gaps": largest_current_test_gaps,
        "segment_metric_rows": segment_metrics.to_dict(orient="records"),
        "output_files": {
            "metrics": "calibration_experiment_metrics.csv",
            "bins": "calibration_experiment_bins.csv",
            "segment_metrics": "calibration_experiment_segment_metrics.csv",
            "summary": "calibration_experiment_summary.json",
            "report": "calibration_experiment_report.md",
        },
    }


def choose_recommendation(
    test_metrics: pd.DataFrame,
    external_metrics: pd.DataFrame,
) -> dict[str, object]:
    candidates = test_metrics.loc[~test_metrics["variant"].eq("raw_xgboost")].copy()
    current = row_by_variant(test_metrics, "current_platt")
    current_ece = float(cast(float, current["ece"]))
    current_brier = float(cast(float, current["brier"]))
    current_logloss = float(cast(float, current["logloss"]))
    current_f1 = float(cast(float, current["classification_f1"]))
    current_fp = int(cast(int, current["classification_false_positive"]))
    candidates["ece_delta_vs_current"] = candidates["ece"] - current_ece
    candidates["brier_delta_vs_current"] = candidates["brier"] - current_brier
    candidates["logloss_delta_vs_current"] = candidates["logloss"] - current_logloss
    candidates["f1_delta_vs_current"] = candidates["classification_f1"] - current_f1
    candidates["fp_delta_vs_current"] = candidates["classification_false_positive"] - current_fp
    viable = candidates.loc[
        candidates["ece_delta_vs_current"].lt(0.0)
        & candidates["brier_delta_vs_current"].le(0.0)
        & candidates["logloss_delta_vs_current"].le(0.0)
        & candidates["f1_delta_vs_current"].ge(-0.005)
        & candidates["fp_delta_vs_current"].le(0)
    ].copy()
    if viable.empty:
        selected = current
        reason = (
            "No candidate improved test ECE while also avoiding Brier/logloss and review-load "
            "regression. Keep the current Platt calibrator."
        )
    else:
        selected = viable.sort_values(["ece", "brier", "logloss"]).iloc[0].to_dict()
        reason = (
            "Lowest test ECE among candidates that also avoid Brier/logloss and review-load "
            "regression."
        )
    external_row = row_by_variant(external_metrics, str(selected["variant"]))
    return {
        "variant": selected["variant"],
        "display_name": selected["display_name"],
        "reason": reason,
        "test_ece": selected["ece"],
        "test_brier": selected["brier"],
        "test_logloss": selected["logloss"],
        "external_2026_ece": external_row.get("ece"),
        "external_2026_brier": external_row.get("brier"),
        "external_2026_logloss": external_row.get("logloss"),
    }


def row_by_variant(frame: pd.DataFrame, variant: str) -> dict[str, object]:
    selected = frame.loc[frame["variant"].eq(variant)]
    if selected.empty:
        return {}
    return selected.iloc[0].to_dict()


def best_row(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    selected = frame.loc[frame[metric].notna()].copy()
    if selected.empty:
        return {}
    return selected.sort_values([metric, "brier", "logloss"]).iloc[0].to_dict()


def build_report(summary: dict[str, object], metrics: pd.DataFrame, bins: pd.DataFrame) -> str:
    recommendation = cast(dict[str, object], summary["recommendation"])
    current = cast(dict[str, object], summary["current_test"])
    best_ece = cast(dict[str, object], summary["best_test_ece"])
    lines = [
        "# Feature 43 Calibration Experiments",
        "",
        "## Scope",
        "",
        "- Model: `feature_46_xgboost`",
        "- Fit data: historical validation split and historical rolling variants",
        "- Audit data: historical final test and 2026 external labels",
        "- Objective: improve dashboard probability reliability, not feature ranking.",
        "",
        "## Recommendation",
        "",
        f"- Recommended variant: `{recommendation['variant']}`",
        f"- Reason: {recommendation['reason']}",
        f"- Current test ECE: `{format_optional(current.get('ece'))}`",
        f"- Best test ECE: `{format_optional(best_ece.get('ece'))}` (`{best_ece.get('variant')}`)",
        f"- Recommended 2026 external ECE: `{format_optional(recommendation.get('external_2026_ece'))}`",
        "",
        "## Test Metrics",
        "",
        metrics_table(metrics.loc[metrics["evaluation_scope"].eq("test")]),
        "",
        "## 2026 External Audit",
        "",
        metrics_table(metrics.loc[metrics["evaluation_scope"].eq("external_2026")]),
        "",
        "## Current Test Bin Gaps",
        "",
        current_bins_table(
            bins.loc[bins["evaluation_scope"].eq("test") & bins["variant"].eq("current_platt")]
        ),
        "",
        "## Interpretation",
        "",
        "- `ECE` is the row-weighted absolute gap between predicted probability and realized rate.",
        "- `MCE` is the largest bin-level absolute gap and is sensitive to small bins.",
        "- Segment calibration variants use validation-only segment fits with global fallback when segment samples are too small.",
        "- 2026 labels are treated as external audit, so they do not decide the saved calibrator by themselves.",
        "",
    ]
    return "\n".join(lines)


def metrics_table(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["ece", "brier", "logloss"]).copy()
    lines = [
        "| Variant | Brier | Logloss | ECE | MCE | Mean P | Bias | Threshold | Precision | Recall | F1 | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['variant']} | {format_optional(row.get('brier'))} | "
            f"{format_optional(row.get('logloss'))} | {format_optional(row.get('ece'))} | "
            f"{format_optional(row.get('mce'))} | {format_optional(row.get('mean_probability'))} | "
            f"{format_optional(row.get('calibration_bias'))} | "
            f"{format_optional(row.get('classification_threshold'))} | "
            f"{format_optional(row.get('classification_precision'))} | "
            f"{format_optional(row.get('classification_recall'))} | "
            f"{format_optional(row.get('classification_f1'))} | "
            f"{format_count(row.get('classification_false_positive'))} | "
            f"{format_count(row.get('classification_false_negative'))} |"
        )
    return "\n".join(lines)


def current_bins_table(frame: pd.DataFrame) -> str:
    lines = [
        "| Probability Bin | Rows | Mean P | Actual Rate | Gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['probability_bin']} | {int(row['rows'])} | "
            f"{format_optional(row['mean_probability'])} | "
            f"{format_optional(row['actual_positive_rate'])} | "
            f"{format_optional(row['calibration_gap'])} |"
        )
    return "\n".join(lines)


def format_optional(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_count(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(int(float(value)))


def to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


if __name__ == "__main__":
    main()
