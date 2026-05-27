from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas.modeling.calibration import (  # noqa: E402
    DEFAULT_THRESHOLD_GRID,
    PROBABILITY_CLIP_EPSILON,
)
from cas.modeling.stage1_xgboost import (  # noqa: E402
    DEFAULT_ROLLING_EVAL_YEARS,
    DEFAULT_STAGE1_RANDOM_STATE,
    read_stage1_feature_columns,
    read_stage1_master,
    train_stage1_xgboost,
)

INPUT_DIR = ROOT / "data" / "input" / "credit_46_features"
MASTER_PATH = INPUT_DIR / "feature_46_master.csv"
FEATURE_LIST_PATH = INPUT_DIR / "feature_46_list.json"
PREDICTION_SCORES_PATH = (
    ROOT / "data" / "outputs" / "dashboard" / "feature_46_mvp" / "prediction_scores.csv"
)
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_46_xgboost" / "diagnostics"

RANDOM_STATE = DEFAULT_STAGE1_RANDOM_STATE
THRESHOLD_GRID = DEFAULT_THRESHOLD_GRID
MIN_SEGMENT_ROWS = 40
MIN_SEGMENT_POSITIVES = 5
ROLLING_CALIBRATION_YEARS = [2018, 2019, 2020, 2021, 2022]
ROLLING_EVAL_YEARS = DEFAULT_ROLLING_EVAL_YEARS
FOCUS_SEGMENTS = [
    ("overall", "all", None, None),
    ("market", "KOSDAQ", "market", "KOSDAQ"),
    ("market", "KOSPI", "market", "KOSPI"),
    ("industry", "manufacturing", "industry_macro_category", "manufacturing"),
    ("industry", "it_services", "industry_macro_category", "it_services"),
]

FloatArray = npt.NDArray[np.float64]
ProbabilityFn = Callable[[pd.DataFrame], FloatArray]


@dataclass(frozen=True)
class CalibrationVariant:
    name: str
    display_name: str
    fit_scope: str
    probability_fn: ProbabilityFn
    note: str


@dataclass(frozen=True)
class ThresholdPolicy:
    mode: str
    display_name: str
    detail: str
    apply_fn: Callable[[pd.DataFrame], pd.Series]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare calibration variants and dashboard operating threshold modes for "
            "the official 46-feature XGBoost model."
        )
    )
    parser.add_argument("--prediction-scores", type=Path, default=PREDICTION_SCORES_PATH)
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=FEATURE_LIST_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-segment-rows", type=int, default=MIN_SEGMENT_ROWS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--eval-years", type=int, nargs="+", default=ROLLING_EVAL_YEARS)
    return parser.parse_args()


def read_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
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
        "prob_speculative_raw",
        "prob_speculative",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"prediction_scores.csv is missing required columns: {missing}")
    output = frame.copy()
    output["is_speculative"] = pd.to_numeric(output["is_speculative"], errors="coerce").astype(int)
    output["prob_speculative_raw"] = pd.to_numeric(
        output["prob_speculative_raw"],
        errors="coerce",
    )
    output["prob_speculative"] = pd.to_numeric(output["prob_speculative"], errors="coerce")
    return output


def read_master(path: Path) -> pd.DataFrame:
    return read_stage1_master(
        path,
        duplicate_keys=["market", "stock_code", "corp_name", "fiscal_year"],
    )


def read_feature_columns(path: Path, master: pd.DataFrame) -> list[str]:
    return read_stage1_feature_columns(path, master)


def clip_probabilities(probabilities: npt.ArrayLike) -> FloatArray:
    return np.asarray(
        np.clip(probabilities, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON),
        dtype=np.float64,
    )


def probability_to_logit(probabilities: npt.ArrayLike) -> FloatArray:
    clipped = clip_probabilities(probabilities)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def beta_features(probabilities: npt.ArrayLike) -> FloatArray:
    clipped = clip_probabilities(probabilities)
    return np.asarray(np.column_stack([np.log(clipped), np.log1p(-clipped)]), dtype=np.float64)


def fit_platt(y_true: pd.Series, probabilities: npt.ArrayLike) -> dict[str, float]:
    model = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(probability_to_logit(probabilities).reshape(-1, 1), y_true.astype(int))
    return {"coef": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}


def apply_platt(probabilities: npt.ArrayLike, calibration: dict[str, float]) -> FloatArray:
    logits = probability_to_logit(probabilities)
    return np.asarray(
        1.0 / (1.0 + np.exp(-(calibration["intercept"] + calibration["coef"] * logits))),
        dtype=np.float64,
    )


def fit_beta(y_true: pd.Series, probabilities: npt.ArrayLike) -> LogisticRegression:
    model = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(beta_features(probabilities), y_true.astype(int))
    return model


def apply_beta(probabilities: npt.ArrayLike, model: LogisticRegression) -> FloatArray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        calibrated = model.predict_proba(beta_features(probabilities))[:, 1]
    return np.asarray(calibrated, dtype=np.float64)


def fit_isotonic(y_true: pd.Series, probabilities: npt.ArrayLike) -> IsotonicRegression:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(np.asarray(probabilities, dtype=np.float64), y_true.astype(int))
    return model


def train_xgboost(
    *,
    train: pd.DataFrame,
    policy: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> XGBClassifier:
    return train_stage1_xgboost(
        train=train,
        policy=policy,
        columns=feature_columns,
        seed=seed,
    )


def build_rolling_oof_probabilities(
    *,
    master: pd.DataFrame,
    feature_columns: list[str],
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for policy_year in ROLLING_CALIBRATION_YEARS:
        train = master.loc[master["fiscal_year"] < policy_year].copy()
        policy = master.loc[master["fiscal_year"] == policy_year].copy()
        if train.empty or policy.empty or policy["is_speculative"].nunique() < 2:
            continue
        model = train_xgboost(
            train=train,
            policy=policy,
            feature_columns=feature_columns,
            seed=seed,
        )
        raw = model.predict_proba(policy.loc[:, feature_columns])[:, 1]
        rows.append(
            pd.DataFrame(
                {
                    "fiscal_year": policy["fiscal_year"].to_numpy(),
                    "is_speculative": policy["is_speculative"].astype(int).to_numpy(),
                    "prob_speculative_raw": raw,
                }
            )
        )
    if not rows:
        raise ValueError("No rolling calibration OOF rows were generated.")
    return pd.concat(rows, ignore_index=True)


def fit_segment_platt(
    valid: pd.DataFrame,
    *,
    group_columns: list[str],
    fallback: dict[str, float],
    min_segment_rows: int,
) -> dict[tuple[str, ...], dict[str, float]]:
    calibrations: dict[tuple[str, ...], dict[str, float]] = {}
    for keys, group in valid.groupby(group_columns, dropna=False):
        key_tuple = tuple(str(value) for value in (keys if isinstance(keys, tuple) else (keys,)))
        enough_rows = len(group) >= min_segment_rows
        enough_positives = int(group["is_speculative"].sum()) >= MIN_SEGMENT_POSITIVES
        if enough_rows and enough_positives and group["is_speculative"].nunique() == 2:
            calibrations[key_tuple] = fit_platt(
                group["is_speculative"].astype(int),
                group["prob_speculative_raw"].to_numpy(dtype=np.float64),
            )
    return calibrations


def apply_segment_platt(
    frame: pd.DataFrame,
    *,
    calibrations: dict[tuple[str, ...], dict[str, float]],
    fallback: dict[str, float],
    group_columns: list[str],
) -> FloatArray:
    output = pd.Series(np.nan, index=frame.index, dtype="float64")
    for keys, group_index in frame.groupby(group_columns, dropna=False).groups.items():
        key_tuple = tuple(str(value) for value in (keys if isinstance(keys, tuple) else (keys,)))
        calibration = calibrations.get(key_tuple, fallback)
        raw = frame.loc[group_index, "prob_speculative_raw"].to_numpy(dtype=np.float64)
        output.loc[group_index] = apply_platt(raw, calibration)
    return output.to_numpy(dtype=np.float64)


def build_calibration_variants(
    *,
    scores: pd.DataFrame,
    master: pd.DataFrame,
    feature_columns: list[str],
    min_segment_rows: int,
    seed: int,
) -> list[CalibrationVariant]:
    valid = scores.loc[scores["split"].astype(str).eq("valid")].copy()
    y_valid = valid["is_speculative"].astype(int)
    raw_valid = valid["prob_speculative_raw"].to_numpy(dtype=np.float64)
    valid_platt = fit_platt(y_valid, raw_valid)
    valid_isotonic = fit_isotonic(y_valid, raw_valid)
    valid_beta = fit_beta(y_valid, raw_valid)

    rolling_oof = build_rolling_oof_probabilities(
        master=master,
        feature_columns=feature_columns,
        seed=seed,
    )
    rolling_y = rolling_oof["is_speculative"].astype(int)
    rolling_raw = rolling_oof["prob_speculative_raw"].to_numpy(dtype=np.float64)
    rolling_platt = fit_platt(rolling_y, rolling_raw)
    rolling_isotonic = fit_isotonic(rolling_y, rolling_raw)
    rolling_beta = fit_beta(rolling_y, rolling_raw)

    market_calibrations = fit_segment_platt(
        valid,
        group_columns=["market"],
        fallback=valid_platt,
        min_segment_rows=min_segment_rows,
    )
    industry_calibrations = fit_segment_platt(
        valid,
        group_columns=["industry_macro_category"],
        fallback=valid_platt,
        min_segment_rows=min_segment_rows,
    )

    return [
        CalibrationVariant(
            name="current_platt",
            display_name="Current Platt",
            fit_scope="saved_valid_platt",
            probability_fn=lambda frame: frame["prob_speculative"].to_numpy(dtype=np.float64),
            note="Saved dashboard probability.",
        ),
        CalibrationVariant(
            name="refit_platt_valid",
            display_name="Refit Platt",
            fit_scope="valid_2022",
            probability_fn=lambda frame, calibration=valid_platt: apply_platt(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                calibration,
            ),
            note="Platt refit on the official validation split.",
        ),
        CalibrationVariant(
            name="isotonic_valid",
            display_name="Isotonic Valid",
            fit_scope="valid_2022",
            probability_fn=lambda frame, model=valid_isotonic: np.asarray(
                model.predict(frame["prob_speculative_raw"].to_numpy(dtype=np.float64)),
                dtype=np.float64,
            ),
            note="Isotonic regression fitted on validation.",
        ),
        CalibrationVariant(
            name="beta_valid",
            display_name="Beta Valid",
            fit_scope="valid_2022",
            probability_fn=lambda frame, model=valid_beta: apply_beta(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                model,
            ),
            note="Beta calibration fitted on validation.",
        ),
        CalibrationVariant(
            name="rolling_oof_platt",
            display_name="Rolling OOF Platt",
            fit_scope="rolling_oof_2018_2022",
            probability_fn=lambda frame, calibration=rolling_platt: apply_platt(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                calibration,
            ),
            note="Platt fitted on out-of-fold policy-year predictions from 2018-2022.",
        ),
        CalibrationVariant(
            name="rolling_oof_isotonic",
            display_name="Rolling OOF Isotonic",
            fit_scope="rolling_oof_2018_2022",
            probability_fn=lambda frame, model=rolling_isotonic: np.asarray(
                model.predict(frame["prob_speculative_raw"].to_numpy(dtype=np.float64)),
                dtype=np.float64,
            ),
            note="Isotonic fitted on out-of-fold policy-year predictions from 2018-2022.",
        ),
        CalibrationVariant(
            name="rolling_oof_beta",
            display_name="Rolling OOF Beta",
            fit_scope="rolling_oof_2018_2022",
            probability_fn=lambda frame, model=rolling_beta: apply_beta(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                model,
            ),
            note="Beta calibration fitted on out-of-fold policy-year predictions from 2018-2022.",
        ),
        CalibrationVariant(
            name="segment_market_platt",
            display_name="Market Segment Platt",
            fit_scope="valid_2022_market",
            probability_fn=lambda frame, calibrations=market_calibrations, fallback=valid_platt: (
                apply_segment_platt(
                    frame,
                    calibrations=calibrations,
                    fallback=fallback,
                    group_columns=["market"],
                )
            ),
            note="Market-specific Platt with global fallback.",
        ),
        CalibrationVariant(
            name="segment_industry_platt",
            display_name="Industry Segment Platt",
            fit_scope="valid_2022_industry",
            probability_fn=lambda frame, calibrations=industry_calibrations, fallback=valid_platt: (
                apply_segment_platt(
                    frame,
                    calibrations=calibrations,
                    fallback=fallback,
                    group_columns=["industry_macro_category"],
                )
            ),
            note="Industry-specific Platt with global fallback.",
        ),
    ]


def build_fold_calibration_variants(
    *,
    policy: pd.DataFrame,
    min_segment_rows: int,
) -> list[CalibrationVariant]:
    y_policy = policy["is_speculative"].astype(int)
    raw_policy = policy["prob_speculative_raw"].to_numpy(dtype=np.float64)
    fold_platt = fit_platt(y_policy, raw_policy)
    fold_isotonic = fit_isotonic(y_policy, raw_policy)
    fold_beta = fit_beta(y_policy, raw_policy)
    market_calibrations = fit_segment_platt(
        policy,
        group_columns=["market"],
        fallback=fold_platt,
        min_segment_rows=min_segment_rows,
    )
    industry_calibrations = fit_segment_platt(
        policy,
        group_columns=["industry_macro_category"],
        fallback=fold_platt,
        min_segment_rows=min_segment_rows,
    )
    return [
        CalibrationVariant(
            name="fold_platt",
            display_name="Fold Platt",
            fit_scope="rolling_policy_year",
            probability_fn=lambda frame, calibration=fold_platt: apply_platt(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                calibration,
            ),
            note="Platt fitted on the fold policy year.",
        ),
        CalibrationVariant(
            name="fold_isotonic",
            display_name="Fold Isotonic",
            fit_scope="rolling_policy_year",
            probability_fn=lambda frame, model=fold_isotonic: np.asarray(
                model.predict(frame["prob_speculative_raw"].to_numpy(dtype=np.float64)),
                dtype=np.float64,
            ),
            note="Isotonic fitted on the fold policy year.",
        ),
        CalibrationVariant(
            name="fold_beta",
            display_name="Fold Beta",
            fit_scope="rolling_policy_year",
            probability_fn=lambda frame, model=fold_beta: apply_beta(
                frame["prob_speculative_raw"].to_numpy(dtype=np.float64),
                model,
            ),
            note="Beta calibration fitted on the fold policy year.",
        ),
        CalibrationVariant(
            name="fold_segment_market_platt",
            display_name="Fold Market Segment Platt",
            fit_scope="rolling_policy_year_market",
            probability_fn=lambda frame, calibrations=market_calibrations, fallback=fold_platt: (
                apply_segment_platt(
                    frame,
                    calibrations=calibrations,
                    fallback=fallback,
                    group_columns=["market"],
                )
            ),
            note="Market-specific Platt fitted on the fold policy year.",
        ),
        CalibrationVariant(
            name="fold_segment_industry_platt",
            display_name="Fold Industry Segment Platt",
            fit_scope="rolling_policy_year_industry",
            probability_fn=lambda frame, calibrations=industry_calibrations, fallback=fold_platt: (
                apply_segment_platt(
                    frame,
                    calibrations=calibrations,
                    fallback=fallback,
                    group_columns=["industry_macro_category"],
                )
            ),
            note="Industry-specific Platt fitted on the fold policy year.",
        ),
    ]


def probability_quality_metrics(y_true: pd.Series, probabilities: npt.ArrayLike) -> dict[str, Any]:
    labels = y_true.astype(int).reset_index(drop=True)
    probs = pd.Series(np.asarray(probabilities, dtype=np.float64), index=labels.index)
    clipped = clip_probabilities(probs.to_numpy(dtype=np.float64))
    bins = calibration_bin_table(labels, probs)
    weighted_gap = bins["rows"] * bins["calibration_gap"].abs()
    return {
        "rows": len(labels),
        "positive_rows": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "logloss": float(log_loss(labels, clipped)),
        "ece": float(weighted_gap.sum() / bins["rows"].sum()) if not bins.empty else 0.0,
        "mce": float(bins["calibration_gap"].abs().max()) if not bins.empty else 0.0,
        "mean_probability": float(probs.mean()),
        "calibration_bias": float(probs.mean() - labels.mean()),
    }


def calibration_bin_table(y_true: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    labels = y_true.astype(int).reset_index(drop=True)
    probs = pd.Series(np.asarray(probabilities, dtype=np.float64), index=labels.index)
    binned = pd.DataFrame({"y_true": labels, "probability": probs})
    binned["probability_bin"] = pd.cut(
        binned["probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        duplicates="drop",
    )
    grouped = (
        binned.groupby("probability_bin", observed=False)
        .agg(
            rows=("y_true", "size"),
            mean_probability=("probability", "mean"),
            actual_positive_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["rows"].gt(0)].copy()
    grouped["probability_bin"] = grouped["probability_bin"].astype(str)
    grouped["calibration_gap"] = grouped["mean_probability"] - grouped["actual_positive_rate"]
    return grouped


def classification_metrics(y_true: pd.Series, predictions: pd.Series) -> dict[str, Any]:
    labels = y_true.astype(int).reset_index(drop=True)
    pred = predictions.astype(int).reset_index(drop=True)
    return {
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "true_negative": int(((labels == 0) & (pred == 0)).sum()),
        "false_positive": int(((labels == 0) & (pred == 1)).sum()),
        "false_negative": int(((labels == 1) & (pred == 0)).sum()),
        "true_positive": int(((labels == 1) & (pred == 1)).sum()),
    }


def threshold_sweep(y_true: pd.Series, probabilities: pd.Series) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        predictions = probabilities.ge(float(threshold)).astype(int)
        rows.append({"threshold": float(threshold), **classification_metrics(y_true, predictions)})
    return pd.DataFrame(rows)


def choose_precision_at_recall_threshold(
    y_true: pd.Series,
    probabilities: pd.Series,
    recall_floor: float,
) -> float:
    sweep = threshold_sweep(y_true, probabilities)
    candidates = sweep.loc[sweep["recall"].ge(recall_floor)]
    if candidates.empty:
        row = sweep.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    else:
        row = candidates.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    return float(row["threshold"])


def choose_conservative_segment_threshold(
    segment: pd.DataFrame,
    *,
    fallback_threshold: float,
    recall_floor: float,
) -> float:
    if (
        len(segment) < MIN_SEGMENT_ROWS
        or int(segment["is_speculative"].sum()) < MIN_SEGMENT_POSITIVES
        or segment["is_speculative"].nunique() < 2
    ):
        return fallback_threshold
    sweep = threshold_sweep(segment["is_speculative"], segment["_probability"])
    conservative = sweep.loc[sweep["threshold"].ge(fallback_threshold)].copy()
    candidates = conservative.loc[conservative["recall"].ge(recall_floor)]
    if candidates.empty:
        return fallback_threshold
    row = candidates.sort_values(
        ["precision", "f1", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(row["threshold"])


def build_threshold_policies(valid_scored: pd.DataFrame) -> list[ThresholdPolicy]:
    recall_first_threshold = choose_precision_at_recall_threshold(
        valid_scored["is_speculative"],
        valid_scored["_probability"],
        recall_floor=0.90,
    )
    balanced_threshold = choose_precision_at_recall_threshold(
        valid_scored["is_speculative"],
        valid_scored["_probability"],
        recall_floor=0.85,
    )
    precision_threshold = choose_precision_at_recall_threshold(
        valid_scored["is_speculative"],
        valid_scored["_probability"],
        recall_floor=0.80,
    )
    kosdaq = valid_scored.loc[valid_scored["market"].astype(str).eq("KOSDAQ")].copy()
    kosdaq_threshold = choose_conservative_segment_threshold(
        kosdaq,
        fallback_threshold=balanced_threshold,
        recall_floor=0.80,
    )
    return [
        ThresholdPolicy(
            mode="recall_first",
            display_name="Recall 우선",
            detail=f"global:{recall_first_threshold:.3f}; valid recall floor 0.90",
            apply_fn=lambda frame, threshold=recall_first_threshold: frame["_probability"].ge(
                threshold
            ),
        ),
        ThresholdPolicy(
            mode="balanced",
            display_name="균형",
            detail=f"global:{balanced_threshold:.3f}; valid recall floor 0.85",
            apply_fn=lambda frame, threshold=balanced_threshold: frame["_probability"].ge(
                threshold
            ),
        ),
        ThresholdPolicy(
            mode="fp_reduction_global",
            display_name="FP 축소 Global",
            detail=f"global:{precision_threshold:.3f}; valid recall floor 0.80",
            apply_fn=lambda frame, threshold=precision_threshold: frame["_probability"].ge(
                threshold
            ),
        ),
        ThresholdPolicy(
            mode="fp_reduction_kosdaq",
            display_name="FP 축소 KOSDAQ",
            detail=f"market=KOSDAQ:{kosdaq_threshold:.3f}; fallback:{balanced_threshold:.3f}",
            apply_fn=lambda frame,
            segment_threshold=kosdaq_threshold,
            fallback=balanced_threshold: (
                frame["_probability"].ge(
                    np.where(
                        frame["market"].astype(str).eq("KOSDAQ"),
                        segment_threshold,
                        fallback,
                    )
                )
            ),
        ),
    ]


def build_probability_outputs(
    *,
    scores: pd.DataFrame,
    variants: list[CalibrationVariant],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    bin_rows: list[pd.DataFrame] = []
    for variant in variants:
        for split_name in ["valid", "test"]:
            frame = scores.loc[scores["split"].astype(str).eq(split_name)].copy()
            probabilities = pd.Series(variant.probability_fn(frame), index=frame.index)
            quality = probability_quality_metrics(frame["is_speculative"], probabilities)
            metrics_rows.append(
                {
                    "variant": variant.name,
                    "display_name": variant.display_name,
                    "fit_scope": variant.fit_scope,
                    "evaluation_scope": split_name,
                    "note": variant.note,
                    **quality,
                }
            )
            bins = calibration_bin_table(frame["is_speculative"], probabilities)
            bins.insert(0, "evaluation_scope", split_name)
            bins.insert(0, "display_name", variant.display_name)
            bins.insert(0, "variant", variant.name)
            bin_rows.append(bins)
    return pd.DataFrame(metrics_rows), pd.concat(bin_rows, ignore_index=True)


def build_mode_outputs(
    *,
    scores: pd.DataFrame,
    variants: list[CalibrationVariant],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    valid = scores.loc[scores["split"].astype(str).eq("valid")].copy()
    test = scores.loc[scores["split"].astype(str).eq("test")].copy()
    for variant in variants:
        valid_scored = valid.assign(_probability=variant.probability_fn(valid))
        test_scored = test.assign(_probability=variant.probability_fn(test))
        policies = build_threshold_policies(valid_scored)
        for policy in policies:
            for scope, frame in [("valid", valid_scored), ("test", test_scored)]:
                predictions = pd.Series(policy.apply_fn(frame), index=frame.index).astype(int)
                metrics = classification_metrics(frame["is_speculative"], predictions)
                mode_rows.append(
                    {
                        "variant": variant.name,
                        "display_name": variant.display_name,
                        "operating_mode": policy.mode,
                        "operating_mode_display": policy.display_name,
                        "threshold_detail": policy.detail,
                        "evaluation_scope": scope,
                        **metrics,
                    }
                )
                if scope == "test":
                    segment_rows.extend(
                        build_segment_mode_rows(
                            variant=variant,
                            policy=policy,
                            frame=frame,
                            predictions=predictions,
                        )
                    )
    return pd.DataFrame(mode_rows), pd.DataFrame(segment_rows)


def build_rolling_validation_outputs(
    *,
    master: pd.DataFrame,
    feature_columns: list[str],
    eval_years: list[int],
    min_segment_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probability_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    for eval_year in eval_years:
        policy_year = eval_year - 1
        train = master.loc[master["fiscal_year"] < policy_year].copy()
        policy = master.loc[master["fiscal_year"] == policy_year].copy()
        evaluation = master.loc[master["fiscal_year"] == eval_year].copy()
        if train.empty or policy.empty or evaluation.empty:
            raise ValueError(
                f"Empty rolling split for eval_year={eval_year}: "
                f"train={len(train)}, policy={len(policy)}, eval={len(evaluation)}"
            )
        if policy["is_speculative"].nunique() < 2 or evaluation["is_speculative"].nunique() < 2:
            raise ValueError(
                f"Rolling split for eval_year={eval_year} needs both classes in "
                "policy and evaluation years."
            )
        model = train_xgboost(
            train=train,
            policy=policy,
            feature_columns=feature_columns,
            seed=seed,
        )
        policy = policy.assign(
            prob_speculative_raw=model.predict_proba(policy.loc[:, feature_columns])[:, 1]
        )
        evaluation = evaluation.assign(
            prob_speculative_raw=model.predict_proba(evaluation.loc[:, feature_columns])[:, 1]
        )
        variants = build_fold_calibration_variants(
            policy=policy,
            min_segment_rows=min_segment_rows,
        )
        for variant in variants:
            policy_scored = policy.assign(_probability=variant.probability_fn(policy))
            eval_scored = evaluation.assign(_probability=variant.probability_fn(evaluation))
            probability_rows.append(
                {
                    "variant": variant.name,
                    "display_name": variant.display_name,
                    "fit_scope": variant.fit_scope,
                    "eval_year": eval_year,
                    "policy_year": policy_year,
                    "note": variant.note,
                    **probability_quality_metrics(
                        eval_scored["is_speculative"],
                        eval_scored["_probability"],
                    ),
                }
            )
            policies = build_threshold_policies(policy_scored)
            for policy_spec in policies:
                predictions = pd.Series(
                    policy_spec.apply_fn(eval_scored),
                    index=eval_scored.index,
                ).astype(int)
                mode_rows.append(
                    {
                        "variant": variant.name,
                        "display_name": variant.display_name,
                        "operating_mode": policy_spec.mode,
                        "operating_mode_display": policy_spec.display_name,
                        "threshold_detail": policy_spec.detail,
                        "eval_year": eval_year,
                        "policy_year": policy_year,
                        **classification_metrics(eval_scored["is_speculative"], predictions),
                    }
                )
    probability_fold_metrics = pd.DataFrame(probability_rows)
    mode_fold_metrics = pd.DataFrame(mode_rows)
    return (
        probability_fold_metrics,
        summarize_rolling_probability(probability_fold_metrics),
        mode_fold_metrics,
        summarize_rolling_modes(mode_fold_metrics),
    )


def summarize_rolling_probability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, display_name, fit_scope), group in fold_metrics.groupby(
        ["variant", "display_name", "fit_scope"],
        sort=False,
    ):
        rows.append(
            {
                "variant": variant,
                "display_name": display_name,
                "fit_scope": fit_scope,
                "folds": int(group["eval_year"].nunique()),
                "rows_sum": int(group["rows"].sum()),
                "positive_rows_sum": int(group["positive_rows"].sum()),
                "pr_auc_mean": float(group["pr_auc"].mean()),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "brier_mean": float(group["brier"].mean()),
                "logloss_mean": float(group["logloss"].mean()),
                "ece_mean": float(group["ece"].mean()),
                "mce_mean": float(group["mce"].mean()),
                "mean_probability_mean": float(group["mean_probability"].mean()),
                "calibration_bias_mean": float(group["calibration_bias"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["ece_mean", "brier_mean", "logloss_mean"])


def summarize_rolling_modes(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "variant",
        "display_name",
        "operating_mode",
        "operating_mode_display",
    ]
    for keys, group in fold_metrics.groupby(group_columns, sort=False):
        variant, display_name, operating_mode, operating_mode_display = keys
        tp = int(group["true_positive"].sum())
        fp = int(group["false_positive"].sum())
        fn = int(group["false_negative"].sum())
        tn = int(group["true_negative"].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "variant": variant,
                "display_name": display_name,
                "operating_mode": operating_mode,
                "operating_mode_display": operating_mode_display,
                "folds": int(group["eval_year"].nunique()),
                "precision_mean": float(group["precision"].mean()),
                "recall_mean": float(group["recall"].mean()),
                "f1_mean": float(group["f1"].mean()),
                "precision_pooled": precision,
                "recall_pooled": recall,
                "f1_pooled": f1,
                "true_positive_sum": tp,
                "false_positive_sum": fp,
                "false_negative_sum": fn,
                "true_negative_sum": tn,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "operating_mode"])


def build_segment_mode_rows(
    *,
    variant: CalibrationVariant,
    policy: ThresholdPolicy,
    frame: pd.DataFrame,
    predictions: pd.Series,
) -> list[dict[str, Any]]:
    rows = []
    scored = frame.assign(_prediction=predictions)
    for dimension, segment, column, value in FOCUS_SEGMENTS:
        segment_frame = (
            scored if column is None else scored.loc[scored[column].astype(str).eq(value)]
        )
        if segment_frame.empty:
            continue
        metrics = classification_metrics(
            segment_frame["is_speculative"],
            segment_frame["_prediction"],
        )
        rows.append(
            {
                "variant": variant.name,
                "display_name": variant.display_name,
                "operating_mode": policy.mode,
                "operating_mode_display": policy.display_name,
                "dimension": dimension,
                "segment": segment,
                "rows": len(segment_frame),
                "positive_rows": int(segment_frame["is_speculative"].sum()),
                **metrics,
            }
        )
    return rows


def recommend_calibration(probability_metrics: pd.DataFrame) -> dict[str, Any]:
    test = probability_metrics.loc[probability_metrics["evaluation_scope"].eq("test")].copy()
    current = test.loc[test["variant"].eq("current_platt")].iloc[0]
    candidates = test.loc[~test["variant"].eq("current_platt")].copy()
    viable = candidates.loc[
        candidates["brier"].le(float(current["brier"]))
        & candidates["logloss"].le(float(current["logloss"]))
        & candidates["ece"].lt(float(current["ece"]))
    ].copy()
    if viable.empty:
        selected = current
        reason = "No calibration variant improved ECE while preserving Brier/logloss."
    else:
        selected = viable.sort_values(["ece", "brier", "logloss"]).iloc[0]
        reason = "Improves ECE without worsening Brier/logloss on Final Test."
    return {
        "variant": selected["variant"],
        "display_name": selected["display_name"],
        "reason": reason,
    }


def recommend_rolling_calibration(rolling_probability_summary: pd.DataFrame) -> dict[str, Any]:
    current = rolling_probability_summary.loc[
        rolling_probability_summary["variant"].eq("fold_platt")
    ].iloc[0]
    candidates = rolling_probability_summary.loc[
        ~rolling_probability_summary["variant"].eq("fold_platt")
    ].copy()
    viable = candidates.loc[
        candidates["brier_mean"].le(float(current["brier_mean"]))
        & candidates["logloss_mean"].le(float(current["logloss_mean"]))
        & candidates["ece_mean"].lt(float(current["ece_mean"]))
    ].copy()
    if viable.empty:
        selected = current
        reason = (
            "No rolling calibration variant improved ECE while preserving Brier/logloss; "
            "keep fold Platt as the operating baseline."
        )
    else:
        selected = viable.sort_values(["ece_mean", "brier_mean", "logloss_mean"]).iloc[0]
        reason = "Improves rolling ECE without worsening rolling Brier/logloss."
    return {
        "variant": selected["variant"],
        "display_name": selected["display_name"],
        "reason": reason,
    }


def recommend_operating_modes(mode_metrics: pd.DataFrame) -> dict[str, Any]:
    test = mode_metrics.loc[
        mode_metrics["evaluation_scope"].eq("test") & mode_metrics["variant"].eq("current_platt")
    ].copy()
    rows = {row["operating_mode"]: row.to_dict() for _, row in test.iterrows()}
    return {
        "default": "balanced",
        "dashboard_modes": [
            "recall_first",
            "balanced",
            "fp_reduction_global",
            "fp_reduction_kosdaq",
        ],
        "reason": (
            "balanced preserves the official threshold behavior; recall_first and "
            "fp_reduction modes expose explicit review-load trade-offs without changing the "
            "saved model."
        ),
        "current_platt_test_modes": rows,
    }


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return str(int(value))


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for item in frame.to_dict(orient="records"):
        values = []
        for _, column, kind in columns:
            value = item.get(column)
            if kind == "metric":
                values.append(format_metric(value))
            elif kind == "int":
                values.append(format_int(value))
            else:
                values.append(str(value) if value is not None else "")
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def build_report(
    *,
    probability_metrics: pd.DataFrame,
    mode_metrics: pd.DataFrame,
    rolling_probability_summary: pd.DataFrame,
    rolling_mode_summary: pd.DataFrame,
    rolling_mode_fold_metrics: pd.DataFrame,
    bins: pd.DataFrame,
    segment_modes: pd.DataFrame,
    calibration_recommendation: dict[str, Any],
    rolling_calibration_recommendation: dict[str, Any],
    operating_recommendation: dict[str, Any],
) -> str:
    test_probability = probability_metrics.loc[
        probability_metrics["evaluation_scope"].eq("test")
    ].sort_values(["ece", "brier", "logloss"])
    current_modes = mode_metrics.loc[
        mode_metrics["evaluation_scope"].eq("test") & mode_metrics["variant"].eq("current_platt")
    ].copy()
    current_bins = bins.loc[
        bins["evaluation_scope"].eq("test") & bins["variant"].eq("current_platt")
    ].copy()
    important_segments = segment_modes.loc[
        segment_modes["variant"].eq("current_platt")
        & segment_modes["operating_mode"].isin(["recall_first", "balanced", "fp_reduction_kosdaq"])
        & segment_modes["dimension"].isin(["overall", "market", "industry"])
    ].copy()
    rolling_probability = rolling_probability_summary.sort_values(
        ["ece_mean", "brier_mean", "logloss_mean"]
    )
    rolling_modes = rolling_mode_summary.loc[
        rolling_mode_summary["variant"].eq("fold_platt")
        & rolling_mode_summary["operating_mode"].isin(
            ["recall_first", "balanced", "fp_reduction_global", "fp_reduction_kosdaq"]
        )
    ].copy()
    rolling_fold_modes = rolling_mode_fold_metrics.loc[
        rolling_mode_fold_metrics["variant"].eq("fold_platt")
        & rolling_mode_fold_metrics["operating_mode"].isin(
            ["recall_first", "balanced", "fp_reduction_global", "fp_reduction_kosdaq"]
        )
    ].copy()
    return "\n".join(
        [
            "# 46-Feature Calibration + Operating Threshold Experiments",
            "",
            "공식 `feature_46_xgboost` raw score는 유지하고, probability calibration과 "
            "dashboard 운영 threshold mode를 분리해 비교한 실험입니다.",
            "",
            "## 1. 결론",
            "",
            f"- Rolling calibration recommendation: `{rolling_calibration_recommendation['variant']}` "
            f"({rolling_calibration_recommendation['reason']})",
            f"- Final Test calibration check: `{calibration_recommendation['variant']}` "
            f"({calibration_recommendation['reason']})",
            f"- Operating mode default: `{operating_recommendation['default']}`",
            "- Dashboard에 `Recall 우선`, `균형`, `FP 축소 Global`, `FP 축소 KOSDAQ` 모드를 노출하면 "
            "모델 재학습 없이 review-load trade-off를 설명할 수 있습니다.",
            "",
            "## 2. Rolling Validation 운영 모드",
            "",
            "각 rolling fold는 `과거 연도 학습 -> 직전 1년 calibration/threshold 선택 -> "
            "다음 1년 평가` 구조입니다.",
            "",
            markdown_table(
                rolling_modes.sort_values(["operating_mode"]),
                [
                    ("Mode", "operating_mode", "text"),
                    ("Folds", "folds", "int"),
                    ("Precision Mean", "precision_mean", "metric"),
                    ("Recall Mean", "recall_mean", "metric"),
                    ("F1 Mean", "f1_mean", "metric"),
                    ("Pooled Precision", "precision_pooled", "metric"),
                    ("Pooled Recall", "recall_pooled", "metric"),
                    ("Pooled F1", "f1_pooled", "metric"),
                    ("FP Sum", "false_positive_sum", "int"),
                    ("FN Sum", "false_negative_sum", "int"),
                ],
            ),
            "",
            "## 3. Rolling Validation 연도별 운영 모드",
            "",
            markdown_table(
                rolling_fold_modes.sort_values(["eval_year", "operating_mode"]),
                [
                    ("Eval Year", "eval_year", "int"),
                    ("Policy Year", "policy_year", "int"),
                    ("Mode", "operating_mode", "text"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 4. Rolling Validation Calibration 비교",
            "",
            markdown_table(
                rolling_probability,
                [
                    ("Variant", "variant", "text"),
                    ("Fit", "fit_scope", "text"),
                    ("Folds", "folds", "int"),
                    ("PR-AUC Mean", "pr_auc_mean", "metric"),
                    ("Brier Mean", "brier_mean", "metric"),
                    ("Logloss Mean", "logloss_mean", "metric"),
                    ("ECE Mean", "ece_mean", "metric"),
                    ("Bias Mean", "calibration_bias_mean", "metric"),
                ],
            ),
            "",
            "## 5. Final Test Calibration 비교",
            "",
            markdown_table(
                test_probability,
                [
                    ("Variant", "variant", "text"),
                    ("Fit", "fit_scope", "text"),
                    ("PR-AUC", "pr_auc", "metric"),
                    ("Brier", "brier", "metric"),
                    ("Logloss", "logloss", "metric"),
                    ("ECE", "ece", "metric"),
                    ("MCE", "mce", "metric"),
                    ("Mean P", "mean_probability", "metric"),
                    ("Bias", "calibration_bias", "metric"),
                ],
            ),
            "",
            "## 6. Final Test Current Platt 운영 모드",
            "",
            markdown_table(
                current_modes.sort_values(["operating_mode"]),
                [
                    ("Mode", "operating_mode", "text"),
                    ("Threshold", "threshold_detail", "text"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 7. Final Test Current Platt Calibration Bin",
            "",
            markdown_table(
                current_bins,
                [
                    ("Bin", "probability_bin", "text"),
                    ("Rows", "rows", "int"),
                    ("Mean P", "mean_probability", "metric"),
                    ("Actual", "actual_positive_rate", "metric"),
                    ("Gap", "calibration_gap", "metric"),
                ],
            ),
            "",
            "## 8. Final Test Current Platt 주요 세그먼트별 운영 모드",
            "",
            markdown_table(
                important_segments.sort_values(["operating_mode", "dimension", "segment"]),
                [
                    ("Mode", "operating_mode", "text"),
                    ("Segment", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 9. 해석 주의",
            "",
            "- 운영 모드 선택은 rolling validation을 우선 기준으로 보고, Final Test는 마지막 확인용입니다.",
            "- Final Test의 segment calibration 후보 선택은 사후 확인용 결과입니다.",
            "- Operating threshold mode는 확률 자체를 바꾸지 않고 리뷰 민감도를 바꾸는 UI 정책입니다.",
        ]
    )


def write_outputs(
    *,
    output_dir: Path,
    probability_metrics: pd.DataFrame,
    mode_metrics: pd.DataFrame,
    rolling_probability_fold_metrics: pd.DataFrame,
    rolling_probability_summary: pd.DataFrame,
    rolling_mode_fold_metrics: pd.DataFrame,
    rolling_mode_summary: pd.DataFrame,
    bins: pd.DataFrame,
    segment_modes: pd.DataFrame,
    calibration_recommendation: dict[str, Any],
    rolling_calibration_recommendation: dict[str, Any],
    operating_recommendation: dict[str, Any],
    eval_years: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "probability_metrics": output_dir / "calibration_operating_policy_metrics.csv",
        "mode_metrics": output_dir / "calibration_operating_policy_mode_metrics.csv",
        "rolling_probability_fold_metrics": (
            output_dir / "calibration_operating_policy_rolling_probability_fold_metrics.csv"
        ),
        "rolling_probability_summary": (
            output_dir / "calibration_operating_policy_rolling_probability_summary.csv"
        ),
        "rolling_mode_fold_metrics": (
            output_dir / "calibration_operating_policy_rolling_mode_fold_metrics.csv"
        ),
        "rolling_mode_summary": (
            output_dir / "calibration_operating_policy_rolling_mode_summary.csv"
        ),
        "bins": output_dir / "calibration_operating_policy_bins.csv",
        "segment_modes": output_dir / "calibration_operating_policy_segment_modes.csv",
        "summary": output_dir / "calibration_operating_policy_summary.json",
        "report": output_dir / "calibration_operating_policy_report.md",
    }
    probability_metrics.to_csv(paths["probability_metrics"], index=False, encoding="utf-8-sig")
    mode_metrics.to_csv(paths["mode_metrics"], index=False, encoding="utf-8-sig")
    rolling_probability_fold_metrics.to_csv(
        paths["rolling_probability_fold_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    rolling_probability_summary.to_csv(
        paths["rolling_probability_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    rolling_mode_fold_metrics.to_csv(
        paths["rolling_mode_fold_metrics"],
        index=False,
        encoding="utf-8-sig",
    )
    rolling_mode_summary.to_csv(
        paths["rolling_mode_summary"],
        index=False,
        encoding="utf-8-sig",
    )
    bins.to_csv(paths["bins"], index=False, encoding="utf-8-sig")
    segment_modes.to_csv(paths["segment_modes"], index=False, encoding="utf-8-sig")
    report = build_report(
        probability_metrics=probability_metrics,
        mode_metrics=mode_metrics,
        rolling_probability_summary=rolling_probability_summary,
        rolling_mode_summary=rolling_mode_summary,
        rolling_mode_fold_metrics=rolling_mode_fold_metrics,
        bins=bins,
        segment_modes=segment_modes,
        calibration_recommendation=calibration_recommendation,
        rolling_calibration_recommendation=rolling_calibration_recommendation,
        operating_recommendation=operating_recommendation,
    )
    paths["report"].write_text(report, encoding="utf-8")
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "feature_46_xgboost",
        "dataset": "credit_46_features",
        "calibration_recommendation": calibration_recommendation,
        "rolling_calibration_recommendation": rolling_calibration_recommendation,
        "operating_recommendation": operating_recommendation,
        "rolling_validation": {
            "eval_years": eval_years,
            "fold_policy": (
                "train fiscal_year < eval_year-1, tune calibration/threshold on "
                "eval_year-1, evaluate eval_year"
            ),
            "fold_platt_modes": rolling_mode_summary.loc[
                rolling_mode_summary["variant"].eq("fold_platt")
            ].to_dict(orient="records"),
            "calibration_summary": rolling_probability_summary.to_dict(orient="records"),
        },
        "output_files": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
    }
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    scores = read_scores(args.prediction_scores)
    master = read_master(args.master_path)
    feature_columns = read_feature_columns(args.feature_list_path, master)
    variants = build_calibration_variants(
        scores=scores,
        master=master,
        feature_columns=feature_columns,
        min_segment_rows=args.min_segment_rows,
        seed=args.seed,
    )
    probability_metrics, bins = build_probability_outputs(scores=scores, variants=variants)
    mode_metrics, segment_modes = build_mode_outputs(scores=scores, variants=variants)
    (
        rolling_probability_fold_metrics,
        rolling_probability_summary,
        rolling_mode_fold_metrics,
        rolling_mode_summary,
    ) = build_rolling_validation_outputs(
        master=master,
        feature_columns=feature_columns,
        eval_years=args.eval_years,
        min_segment_rows=args.min_segment_rows,
        seed=args.seed,
    )
    calibration_recommendation = recommend_calibration(probability_metrics)
    rolling_calibration_recommendation = recommend_rolling_calibration(rolling_probability_summary)
    operating_recommendation = recommend_operating_modes(mode_metrics)
    write_outputs(
        output_dir=args.output_dir,
        probability_metrics=probability_metrics,
        mode_metrics=mode_metrics,
        rolling_probability_fold_metrics=rolling_probability_fold_metrics,
        rolling_probability_summary=rolling_probability_summary,
        rolling_mode_fold_metrics=rolling_mode_fold_metrics,
        rolling_mode_summary=rolling_mode_summary,
        bins=bins,
        segment_modes=segment_modes,
        calibration_recommendation=calibration_recommendation,
        rolling_calibration_recommendation=rolling_calibration_recommendation,
        operating_recommendation=operating_recommendation,
        eval_years=args.eval_years,
    )
    current_modes = operating_recommendation["current_platt_test_modes"]
    rolling_fold_platt_modes = {
        row["operating_mode"]: row
        for row in rolling_mode_summary.loc[
            rolling_mode_summary["variant"].eq("fold_platt")
        ].to_dict(orient="records")
    }
    print(
        json.dumps(
            {
                "calibration_recommendation": calibration_recommendation,
                "rolling_calibration_recommendation": rolling_calibration_recommendation,
                "rolling_balanced": rolling_fold_platt_modes["balanced"],
                "rolling_fp_reduction_kosdaq": rolling_fold_platt_modes["fp_reduction_kosdaq"],
                "balanced": current_modes["balanced"],
                "fp_reduction_kosdaq": current_modes["fp_reduction_kosdaq"],
                "report": str(
                    (args.output_dir / "calibration_operating_policy_report.md").relative_to(ROOT)
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
