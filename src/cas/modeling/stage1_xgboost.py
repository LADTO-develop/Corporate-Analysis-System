"""Shared Stage 1 XGBoost training and rolling-evaluation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cas.modeling.calibration import (
    DEFAULT_THRESHOLD_GRID,
    DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR,
    PROBABILITY_CLIP_EPSILON,
    apply_probability_calibration,
    fit_platt_calibration,
)

if TYPE_CHECKING:
    from xgboost import XGBClassifier

FloatArray = npt.NDArray[np.float64]

DEFAULT_STAGE1_RANDOM_STATE = 42
DEFAULT_STAGE1_RECALL_FLOOR = DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR
DEFAULT_ROLLING_EVAL_YEARS = [2019, 2020, 2021, 2022]
DEFAULT_STAGE1_XGBOOST_PARAMS: dict[str, object] = {
    "max_depth": 4,
    "min_child_weight": 3.0,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight_multiplier": 1.0,
}


def read_stage1_master(
    path: Path,
    *,
    duplicate_keys: Sequence[str],
    dataset_name: str = "feature_46_master",
) -> pd.DataFrame:
    """Read a Stage 1 master table and validate company-year uniqueness."""
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    if "stock_code" in frame.columns:
        stock_code = frame["stock_code"].astype("string")
        frame["stock_code"] = stock_code.where(stock_code.isna(), stock_code.str.zfill(6))
    duplicates = int(frame.duplicated(list(duplicate_keys)).sum())
    if duplicates:
        raise ValueError(
            f"{dataset_name} has duplicate rows for {list(duplicate_keys)}: {duplicates}"
        )
    return frame


def read_stage1_feature_columns(
    path: Path,
    master: pd.DataFrame,
    *,
    dataset_name: str = "feature_46_list",
) -> list[str]:
    """Read model feature columns from a Stage 1 feature spec."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = [str(column) for column in cast(Sequence[object], payload["model_features"])]
    missing = [column for column in columns if column not in master.columns]
    if missing:
        raise ValueError(f"{dataset_name} has missing model features in master: {missing}")
    return columns


def split_xy(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Split a Stage 1 frame into features and integer binary labels."""
    return frame.loc[:, list(columns)], frame["is_speculative"].astype(int)


def _float_param(params: Mapping[str, object], key: str) -> float:
    return float(cast(Any, params[key]))


def _int_param(params: Mapping[str, object], key: str) -> int:
    return int(cast(Any, params[key]))


def build_time_decay_weights(
    frame: pd.DataFrame,
    *,
    year_column: str = "fiscal_year",
    reference_year: int | None = None,
    half_life_years: float = 2.0,
    min_weight: float = 0.05,
    normalize: bool = True,
) -> FloatArray:
    """Build recency weights where each half-life halves older observations."""
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive.")
    years = pd.to_numeric(frame[year_column], errors="coerce")
    if years.isna().any():
        missing_count = int(years.isna().sum())
        raise ValueError(f"{year_column} contains {missing_count} missing/non-numeric values.")
    reference = int(reference_year if reference_year is not None else years.max())
    age = np.maximum(0.0, reference - years.to_numpy(dtype=np.float64))
    weights = np.maximum(min_weight, np.power(0.5, age / float(half_life_years)))
    if normalize:
        mean_weight = float(weights.mean())
        if mean_weight > 0:
            weights = weights / mean_weight
    return cast(FloatArray, weights.astype(np.float64))


def build_monotonic_constraints(
    columns: Sequence[str],
    directions: Mapping[str, int],
) -> tuple[int, ...]:
    """Build an XGBoost monotone-constraint tuple aligned to feature columns."""
    constraints: list[int] = []
    for column in columns:
        direction = int(directions.get(column, 0))
        if direction not in {-1, 0, 1}:
            raise ValueError(f"Invalid monotonic direction for {column}: {direction}")
        constraints.append(direction)
    return tuple(constraints)


def _sample_weight_array(values: npt.ArrayLike | None) -> FloatArray | None:
    if values is None:
        return None
    return cast(FloatArray, np.asarray(values, dtype=np.float64))


def train_stage1_xgboost(
    *,
    train: pd.DataFrame,
    policy: pd.DataFrame,
    columns: Sequence[str],
    seed: int = DEFAULT_STAGE1_RANDOM_STATE,
    params: Mapping[str, object] | None = None,
    train_sample_weight: npt.ArrayLike | None = None,
    policy_sample_weight: npt.ArrayLike | None = None,
) -> XGBClassifier:
    """Train the standard Stage 1 XGBoost classifier on a train/policy split."""
    from xgboost import XGBClassifier

    merged_params = {**DEFAULT_STAGE1_XGBOOST_PARAMS, **dict(params or {})}
    x_train, y_train = split_xy(train, columns)
    x_policy, y_policy = split_xy(policy, columns)
    train_weights = _sample_weight_array(train_sample_weight)
    policy_weights = _sample_weight_array(policy_sample_weight)
    if train_weights is not None:
        y_train_array = y_train.to_numpy(dtype=int)
        positives = float(train_weights[y_train_array == 1].sum())
        negatives = float(train_weights[y_train_array == 0].sum())
    else:
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
    base_scale_pos_weight = float(negatives / positives) if positives else 1.0
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=_int_param(merged_params, "max_depth"),
        min_child_weight=_float_param(merged_params, "min_child_weight"),
        gamma=_float_param(merged_params, "gamma"),
        reg_alpha=_float_param(merged_params, "reg_alpha"),
        reg_lambda=_float_param(merged_params, "reg_lambda"),
        subsample=_float_param(merged_params, "subsample"),
        colsample_bytree=_float_param(merged_params, "colsample_bytree"),
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        monotone_constraints=merged_params.get("monotone_constraints"),
        scale_pos_weight=base_scale_pos_weight
        * _float_param(merged_params, "scale_pos_weight_multiplier"),
        early_stopping_rounds=50,
    )
    fit_kwargs: dict[str, object] = {}
    if train_weights is not None:
        fit_kwargs["sample_weight"] = train_weights
    if policy_weights is not None:
        fit_kwargs["sample_weight_eval_set"] = [policy_weights]
    model.fit(x_train, y_train, eval_set=[(x_policy, y_policy)], verbose=False, **fit_kwargs)
    return model


def classification_counts(
    y_true: pd.Series,
    predictions: npt.ArrayLike,
) -> dict[str, int]:
    """Return confusion-matrix counts with stable Stage 1 field names."""
    y_array = y_true.to_numpy(dtype=int)
    pred_array = np.asarray(predictions, dtype=int)
    return {
        "true_negative": int(((y_array == 0) & (pred_array == 0)).sum()),
        "false_positive": int(((y_array == 0) & (pred_array == 1)).sum()),
        "false_negative": int(((y_array == 1) & (pred_array == 0)).sum()),
        "true_positive": int(((y_array == 1) & (pred_array == 1)).sum()),
    }


def classification_metrics(
    y_true: pd.Series,
    predictions: npt.ArrayLike,
) -> dict[str, float | int]:
    """Return precision, recall, F1, and confusion counts."""
    pred_array = np.asarray(predictions, dtype=int)
    return {
        "precision": float(precision_score(y_true, pred_array, zero_division=0)),
        "recall": float(recall_score(y_true, pred_array, zero_division=0)),
        "f1": float(f1_score(y_true, pred_array, zero_division=0)),
        **classification_counts(y_true, pred_array),
    }


def probability_metrics(y_true: pd.Series, probabilities: npt.ArrayLike) -> dict[str, float]:
    """Return ranking and calibration metrics for calibrated probabilities."""
    prob_array = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(prob_array, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    return {
        "pr_auc": float(average_precision_score(y_true, prob_array)),
        "roc_auc": float(roc_auc_score(y_true, prob_array)),
        "brier": float(brier_score_loss(y_true, prob_array)),
        "logloss": float(log_loss(y_true, clipped)),
    }


def choose_policy_threshold(
    y_policy: pd.Series,
    probabilities: npt.ArrayLike,
    *,
    recall_floor: float = DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR,
    threshold_grid: npt.ArrayLike = DEFAULT_THRESHOLD_GRID,
) -> tuple[float, dict[str, float | int | str]]:
    """Select the highest-precision policy threshold that clears a recall floor."""
    prob_array = np.asarray(probabilities, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for threshold in np.asarray(threshold_grid, dtype=np.float64):
        predictions = prob_array >= threshold
        rows.append(
            {"threshold": float(threshold), **classification_metrics(y_policy, predictions)}
        )
    sweep = pd.DataFrame(rows)
    candidates = sweep.loc[sweep["recall"] >= recall_floor]
    selection_rule = f"policy_max_precision_with_recall_ge_{recall_floor:.2f}"
    if candidates.empty:
        candidates = sweep
        selection_rule = "policy_best_f1_fallback"
        row = candidates.sort_values(
            ["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    else:
        row = candidates.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    metrics = cast(dict[str, float | int | str], row.drop(labels=["threshold"]).to_dict())
    metrics["threshold_selection_rule"] = selection_rule
    return float(row["threshold"]), metrics


def evaluate_calibrated_stage1_split(
    *,
    model: XGBClassifier,
    policy: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: Sequence[str],
    recall_floor: float = DEFAULT_TUNED_THRESHOLD_RECALL_FLOOR,
) -> tuple[dict[str, Any], FloatArray]:
    """Score an evaluation split using policy-year Platt calibration and thresholding."""
    x_policy, y_policy = split_xy(policy, columns)
    x_eval, y_eval = split_xy(evaluation, columns)
    policy_raw = cast(FloatArray, model.predict_proba(x_policy)[:, 1])
    eval_raw = cast(FloatArray, model.predict_proba(x_eval)[:, 1])
    calibration = fit_platt_calibration(y_policy, policy_raw)
    policy_prob = apply_probability_calibration(policy_raw, calibration)
    eval_prob = apply_probability_calibration(eval_raw, calibration)
    threshold, policy_threshold_metrics = choose_policy_threshold(
        y_policy,
        policy_prob,
        recall_floor=recall_floor,
    )
    eval_predictions = eval_prob >= threshold
    metrics = {
        "threshold_tuned": threshold,
        "threshold_selection_rule": policy_threshold_metrics["threshold_selection_rule"],
        "policy_precision_at_threshold": policy_threshold_metrics["precision"],
        "policy_recall_at_threshold": policy_threshold_metrics["recall"],
        "policy_f1_at_threshold": policy_threshold_metrics["f1"],
        **{
            f"policy_{key}": value
            for key, value in probability_metrics(y_policy, policy_prob).items()
        },
        **{f"eval_{key}": value for key, value in probability_metrics(y_eval, eval_prob).items()},
        **{
            f"eval_{key}_at_threshold": value
            for key, value in classification_metrics(y_eval, eval_predictions).items()
        },
    }
    return metrics, eval_prob
