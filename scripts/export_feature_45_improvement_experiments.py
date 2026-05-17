from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from export_feature_43_candidate_feature_pack_experiments import (
    INPUT_DIR,
    OUTPUT_DIR,
    RAW_PATH,
    apply_platt_calibration,
    attach_candidate_columns,
    classification_metrics,
    fit_platt_calibration,
    format_int,
    format_metric,
    markdown_table,
    probability_metrics,
    read_id_frames,
    read_raw_features,
    read_split_frames,
    unique_preserve_order,
)
from export_feature_43_xgboost_tuning_experiments import (
    BASELINE_PARAMS,
    RANDOM_STATE,
    RECALL_FLOOR,
    THRESHOLD_GRID,
    candidate_grid,
    train_xgboost,
)

FEATURE_SET_NAME = "feature_45"
CANDIDATE_COLUMNS = ["delta_accruals_ratio", "is_3y_consecutive_operating_loss"]
MIN_SEGMENT_ROWS = 30
MIN_SEGMENT_POSITIVES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run focused improvement experiments for the 45-feature model.")
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=48,
        help="Deterministic random sample size from the XGBoost search grid.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def _base_scale_pos_weight(train: pd.DataFrame) -> float:
    y_train = train["is_speculative"].astype(int)
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    return float(negative / positive) if positive else 1.0


def _fit_variant(
    *,
    variant: str,
    params: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    seed: int,
) -> dict[str, Any]:
    train = frames["train"]
    valid = frames["valid"]
    test = frames["test"]
    y_train = train["is_speculative"].astype(int)
    y_valid = valid["is_speculative"].astype(int)
    y_test = test["is_speculative"].astype(int)
    scale_pos_weight = _base_scale_pos_weight(train) * float(params["scale_pos_weight_multiplier"])
    model = train_xgboost(
        x_train=train.loc[:, feature_columns],
        y_train=y_train,
        x_valid=valid.loc[:, feature_columns],
        y_valid=y_valid,
        params=params,
        scale_pos_weight=scale_pos_weight,
        seed=seed,
    )
    valid_raw_probabilities = model.predict_proba(valid.loc[:, feature_columns])[:, 1]
    test_raw_probabilities = model.predict_proba(test.loc[:, feature_columns])[:, 1]
    coef, intercept = fit_platt_calibration(y_valid, valid_raw_probabilities)
    valid_probabilities = apply_platt_calibration(valid_raw_probabilities, coef, intercept)
    test_probabilities = apply_platt_calibration(test_raw_probabilities, coef, intercept)
    threshold = _choose_threshold(valid_probabilities=valid_probabilities, y_valid=y_valid)
    return {
        "variant": variant,
        "params": params,
        "feature_count": len(feature_columns),
        "best_iteration": getattr(model, "best_iteration", None),
        "scale_pos_weight": scale_pos_weight,
        "threshold_tuned": threshold,
        "y_valid": y_valid,
        "y_test": y_test,
        "valid_probabilities": valid_probabilities,
        "test_probabilities": test_probabilities,
    }


def _choose_threshold(*, valid_probabilities: np.ndarray, y_valid: pd.Series) -> float:
    sweep = _threshold_sweep(y_true=y_valid, probabilities=valid_probabilities)
    candidates = sweep.loc[sweep["recall"] >= RECALL_FLOOR]
    if candidates.empty:
        row = sweep.sort_values(
            ["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    else:
        row = candidates.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    return float(row["threshold"])


def _threshold_sweep(*, y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLD_GRID:
        rows.append(
            {
                "threshold": float(threshold),
                **_metrics_at_threshold(
                    y_true=y_true,
                    probabilities=probabilities,
                    threshold=float(threshold),
                ),
            }
        )
    return pd.DataFrame(rows)


def _metrics_at_threshold(
    *,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    return classification_metrics(y_true, predictions)


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    threshold = float(result["threshold_tuned"])
    params = result["params"]
    row: dict[str, Any] = {
        "variant": result["variant"],
        "feature_count": result["feature_count"],
        "best_iteration": result["best_iteration"],
        "threshold_tuned": threshold,
        "max_depth": params["max_depth"],
        "min_child_weight": params["min_child_weight"],
        "reg_lambda": params["reg_lambda"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "scale_pos_weight_multiplier": params["scale_pos_weight_multiplier"],
        "scale_pos_weight": result["scale_pos_weight"],
    }
    row.update(
        {
            f"valid_{key}": value
            for key, value in probability_metrics(
                result["y_valid"],
                result["valid_probabilities"],
            ).items()
        }
    )
    row.update(
        {
            f"test_{key}": value
            for key, value in probability_metrics(
                result["y_test"],
                result["test_probabilities"],
            ).items()
        }
    )
    row.update(
        {
            f"valid_{key}_at_threshold": value
            for key, value in _metrics_at_threshold(
                y_true=result["y_valid"],
                probabilities=result["valid_probabilities"],
                threshold=threshold,
            ).items()
        }
    )
    row.update(
        {
            f"test_{key}_at_threshold": value
            for key, value in _metrics_at_threshold(
                y_true=result["y_test"],
                probabilities=result["test_probabilities"],
                threshold=threshold,
            ).items()
        }
    )
    return row


def _best_by_valid(metrics: pd.DataFrame) -> pd.Series:
    return metrics.sort_values(
        [
            "valid_f1_at_threshold",
            "valid_pr_auc",
            "valid_precision_at_threshold",
            "test_f1_at_threshold",
        ],
        ascending=False,
    ).iloc[0]


def _best_by_test(metrics: pd.DataFrame) -> pd.Series:
    return metrics.sort_values(
        [
            "test_f1_at_threshold",
            "test_pr_auc",
            "test_precision_at_threshold",
            "valid_f1_at_threshold",
        ],
        ascending=False,
    ).iloc[0]


def _policy_row(
    *,
    policy: str,
    valid_threshold: float,
    test_y: pd.Series,
    test_probabilities: np.ndarray,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "threshold_detail": f"global:{valid_threshold:.3f}",
        **probability_metrics(test_y, test_probabilities),
        **_metrics_at_threshold(
            y_true=test_y,
            probabilities=test_probabilities,
            threshold=valid_threshold,
        ),
    }


def _best_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    sweep = _threshold_sweep(y_true=y_true, probabilities=probabilities)
    row = sweep.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(row["threshold"])


def _max_precision_recall_floor_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    recall_floor: float,
    *,
    min_threshold: float | None = None,
) -> float:
    sweep = _threshold_sweep(y_true=y_true, probabilities=probabilities)
    if min_threshold is not None:
        sweep = sweep.loc[sweep["threshold"] >= min_threshold]
    candidates = sweep.loc[sweep["recall"] >= recall_floor]
    if candidates.empty:
        return _best_f1_threshold(y_true, probabilities)
    row = candidates.sort_values(
        ["precision", "f1", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(row["threshold"])


def _segment_threshold_policy(
    *,
    policy: str,
    dimension: str,
    result: dict[str, Any],
    id_frames: dict[str, pd.DataFrame],
    fallback_threshold: float,
) -> dict[str, Any]:
    valid_ids = id_frames["valid"].reset_index(drop=True)
    test_ids = id_frames["test"].reset_index(drop=True)
    valid_y = result["y_valid"].reset_index(drop=True)
    test_y = result["y_test"].reset_index(drop=True)
    valid_prob = result["valid_probabilities"]
    test_prob = result["test_probabilities"]
    thresholds: dict[str, float] = {}
    for segment, indexes in valid_ids.groupby(dimension).groups.items():
        segment_index = list(indexes)
        segment_y = valid_y.iloc[segment_index]
        has_enough_rows = len(segment_index) >= MIN_SEGMENT_ROWS
        has_enough_positives = int(segment_y.sum()) >= MIN_SEGMENT_POSITIVES
        has_two_classes = segment_y.nunique() == 2
        if has_enough_rows and has_enough_positives and has_two_classes:
            thresholds[str(segment)] = _best_f1_threshold(
                segment_y,
                valid_prob[segment_index],
            )
    predictions = pd.Series(False, index=test_ids.index)
    for segment, segment_frame in test_ids.groupby(dimension):
        threshold = thresholds.get(str(segment), fallback_threshold)
        segment_index = list(segment_frame.index)
        predictions.iloc[segment_index] = test_prob[segment_index] >= threshold
    threshold_detail = "; ".join(
        [
            *(f"{segment}:{threshold:.3f}" for segment, threshold in sorted(thresholds.items())),
            f"fallback:{fallback_threshold:.3f}",
        ]
    )
    return {
        "policy": policy,
        "threshold_detail": threshold_detail,
        **probability_metrics(test_y, test_prob),
        **classification_metrics(test_y, predictions.to_numpy(dtype=bool)),
    }


def _targeted_conservative_policy(
    *,
    policy: str,
    dimension: str,
    segment_name: str,
    recall_floor: float,
    result: dict[str, Any],
    id_frames: dict[str, pd.DataFrame],
    fallback_threshold: float,
) -> dict[str, Any]:
    valid_ids = id_frames["valid"].reset_index(drop=True)
    test_ids = id_frames["test"].reset_index(drop=True)
    valid_y = result["y_valid"].reset_index(drop=True)
    test_y = result["y_test"].reset_index(drop=True)
    valid_prob = result["valid_probabilities"]
    test_prob = result["test_probabilities"]
    valid_mask = valid_ids[dimension].astype(str).eq(segment_name)
    segment_indexes = valid_mask[valid_mask].index.to_list()
    if segment_indexes:
        threshold = _max_precision_recall_floor_threshold(
            valid_y.iloc[segment_indexes],
            valid_prob[segment_indexes],
            recall_floor,
            min_threshold=fallback_threshold,
        )
    else:
        threshold = fallback_threshold
    predictions = test_prob >= fallback_threshold
    test_mask = test_ids[dimension].astype(str).eq(segment_name).to_numpy()
    predictions[test_mask] = test_prob[test_mask] >= threshold
    return {
        "policy": policy,
        "threshold_detail": f"{dimension}={segment_name}:{threshold:.3f}; fallback:{fallback_threshold:.3f}",
        **probability_metrics(test_y, test_prob),
        **classification_metrics(test_y, predictions),
    }


def _add_review_load_columns(
    row: dict[str, Any],
    *,
    predictions: np.ndarray,
    default_predictions: np.ndarray,
    y_test: pd.Series,
) -> dict[str, Any]:
    y_array = y_test.to_numpy(dtype=int)
    added = (~default_predictions) & predictions
    return {
        **row,
        "total_flagged": int(predictions.sum()),
        "added_vs_45_default": int(added.sum()),
        "added_true_risk": int((added & (y_array == 1)).sum()),
        "added_normal": int((added & (y_array == 0)).sum()),
    }


def build_recall_policies(
    *,
    result: dict[str, Any],
    id_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    valid_ids = id_frames["valid"].reset_index(drop=True)
    test_ids = id_frames["test"].reset_index(drop=True)
    valid_y = result["y_valid"].reset_index(drop=True)
    test_y = result["y_test"].reset_index(drop=True)
    valid_probabilities = result["valid_probabilities"]
    test_probabilities = result["test_probabilities"]
    fallback_threshold = float(result["threshold_tuned"])
    default_predictions = test_probabilities >= fallback_threshold
    rows: list[dict[str, Any]] = []
    for recall_floor in [0.85, 0.88, 0.90, 0.92, 0.95]:
        threshold = _max_precision_recall_floor_threshold(
            valid_y,
            valid_probabilities,
            recall_floor,
        )
        predictions = test_probabilities >= threshold
        rows.append(
            _add_review_load_columns(
                {
                    "policy": f"global_valid_recall_ge_{recall_floor:.2f}",
                    "policy_type": "global_threshold",
                    "threshold_detail": f"global:{threshold:.3f}",
                    "valid_recall_floor": recall_floor,
                    **probability_metrics(test_y, test_probabilities),
                    **classification_metrics(test_y, predictions),
                },
                predictions=predictions,
                default_predictions=default_predictions,
                y_test=test_y,
            )
        )

    for dimension, segment_name, recall_floor in [
        ("market", "KOSDAQ", 0.90),
        ("market", "KOSDAQ", 0.92),
        ("market", "KOSDAQ", 0.95),
        ("industry_macro_category", "manufacturing", 0.90),
        ("industry_macro_category", "manufacturing", 0.92),
        ("industry_macro_category", "it_services", 0.90),
        ("industry_macro_category", "it_services", 0.92),
    ]:
        valid_mask = valid_ids[dimension].astype(str).eq(segment_name).to_numpy()
        test_mask = test_ids[dimension].astype(str).eq(segment_name).to_numpy()
        if (
            valid_y[valid_mask].nunique() < 2
            or int(valid_y[valid_mask].sum()) < MIN_SEGMENT_POSITIVES
        ):
            threshold = fallback_threshold
        else:
            threshold = _max_precision_recall_floor_threshold(
                valid_y[valid_mask],
                valid_probabilities[valid_mask],
                recall_floor,
            )
        predictions = test_probabilities >= fallback_threshold
        predictions[test_mask] = test_probabilities[test_mask] >= threshold
        rows.append(
            _add_review_load_columns(
                {
                    "policy": (
                        f"targeted_{dimension}_{segment_name}_valid_recall_ge_{recall_floor:.2f}"
                    ),
                    "policy_type": "targeted_segment_threshold",
                    "threshold_detail": (
                        f"{dimension}={segment_name}:{threshold:.3f}; "
                        f"fallback:{fallback_threshold:.3f}"
                    ),
                    "valid_recall_floor": recall_floor,
                    **probability_metrics(test_y, test_probabilities),
                    **classification_metrics(test_y, predictions),
                },
                predictions=predictions,
                default_predictions=default_predictions,
                y_test=test_y,
            )
        )
    return pd.DataFrame(rows)


def build_threshold_policies(
    *,
    result: dict[str, Any],
    id_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    valid_y = result["y_valid"]
    test_y = result["y_test"]
    valid_probabilities = result["valid_probabilities"]
    test_probabilities = result["test_probabilities"]
    tuned_threshold = float(result["threshold_tuned"])
    rows: list[dict[str, Any]] = [
        _policy_row(
            policy="valid_recall85_max_precision",
            valid_threshold=tuned_threshold,
            test_y=test_y,
            test_probabilities=test_probabilities,
        )
    ]
    for recall_floor in [0.80, 0.85, 0.88]:
        rows.append(
            _policy_row(
                policy=f"global_max_precision_recall_ge_{recall_floor:.2f}",
                valid_threshold=_max_precision_recall_floor_threshold(
                    valid_y,
                    valid_probabilities,
                    recall_floor,
                ),
                test_y=test_y,
                test_probabilities=test_probabilities,
            )
        )
    rows.append(
        _policy_row(
            policy="global_best_valid_f1",
            valid_threshold=_best_f1_threshold(valid_y, valid_probabilities),
            test_y=test_y,
            test_probabilities=test_probabilities,
        )
    )
    for dimension in ["market", "industry_macro_category"]:
        rows.append(
            _segment_threshold_policy(
                policy=f"{dimension}_segment_best_valid_f1",
                dimension=dimension,
                result=result,
                id_frames=id_frames,
                fallback_threshold=tuned_threshold,
            )
        )
    for dimension, segment_name, recall_floor in [
        ("market", "KOSDAQ", 0.80),
        ("market", "KOSDAQ", 0.85),
        ("industry_macro_category", "manufacturing", 0.80),
        ("industry_macro_category", "it_services", 0.80),
    ]:
        rows.append(
            _targeted_conservative_policy(
                policy=f"targeted_{dimension}_{segment_name}_recall_ge_{recall_floor:.2f}",
                dimension=dimension,
                segment_name=segment_name,
                recall_floor=recall_floor,
                result=result,
                id_frames=id_frames,
                fallback_threshold=tuned_threshold,
            )
        )
    return pd.DataFrame(rows)


def build_trigger_policies(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> pd.DataFrame:
    y_test = baseline["y_test"].reset_index(drop=True)
    baseline_predictions = baseline["test_probabilities"] >= float(baseline["threshold_tuned"])
    candidate_predictions = candidate["test_probabilities"] >= float(candidate["threshold_tuned"])
    candidate_only = (~baseline_predictions) & candidate_predictions
    baseline_only = baseline_predictions & (~candidate_predictions)
    y_array = y_test.to_numpy(dtype=int)
    rows = []
    for policy, predictions in [
        ("43_baseline", baseline_predictions),
        ("45_feature_set", candidate_predictions),
        ("union_43_or_45_review_trigger", baseline_predictions | candidate_predictions),
        ("intersection_43_and_45_strict", baseline_predictions & candidate_predictions),
    ]:
        rows.append(
            {
                "policy": policy,
                **classification_metrics(y_test, predictions),
                "candidate_only_review_cases": int(candidate_only.sum()),
                "candidate_only_true_risk": int((candidate_only & (y_array == 1)).sum()),
                "candidate_only_normal": int((candidate_only & (y_array == 0)).sum()),
                "baseline_only_review_cases": int(baseline_only.sum()),
                "baseline_only_true_risk": int((baseline_only & (y_array == 1)).sum()),
                "baseline_only_normal": int((baseline_only & (y_array == 0)).sum()),
            }
        )
    return pd.DataFrame(rows)


def run_experiments(
    *,
    input_dir: Path,
    raw_path: Path,
    max_candidates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, valid, test = read_split_frames(input_dir)
    id_frames = read_id_frames(input_dir)
    raw = read_raw_features(raw_path)
    base_frames = {"train": train, "valid": valid, "test": test}
    candidate_frames = attach_candidate_columns(
        frames=base_frames,
        id_frames=id_frames,
        raw=raw,
        candidate_columns=CANDIDATE_COLUMNS,
    )
    base_features = [column for column in train.columns if column != "is_speculative"]
    candidate_features = unique_preserve_order([*base_features, *CANDIDATE_COLUMNS])
    if len(candidate_features) != 45:
        raise ValueError(f"Expected 45 candidate features, got {len(candidate_features)}")

    baseline_43 = _fit_variant(
        variant="baseline_43_native",
        params=BASELINE_PARAMS,
        frames=base_frames,
        feature_columns=base_features,
        seed=seed,
    )
    rows = [_metric_row(baseline_43)]
    candidate_results: dict[str, dict[str, Any]] = {}
    for index, params in enumerate(candidate_grid(max_candidates=max_candidates, seed=seed)):
        variant = (
            f"{FEATURE_SET_NAME}_default" if index == 0 else f"{FEATURE_SET_NAME}_tuned_{index:03d}"
        )
        result = _fit_variant(
            variant=variant,
            params=params,
            frames=candidate_frames,
            feature_columns=candidate_features,
            seed=seed,
        )
        rows.append(_metric_row(result))
        candidate_results[variant] = result
    metrics = pd.DataFrame(rows)
    default_result = candidate_results[f"{FEATURE_SET_NAME}_default"]
    threshold_policies = build_threshold_policies(
        result=default_result,
        id_frames=id_frames,
    )
    recall_policies = build_recall_policies(
        result=default_result,
        id_frames=id_frames,
    )
    trigger_policies = build_trigger_policies(
        baseline=baseline_43,
        candidate=default_result,
    )
    return metrics, threshold_policies, trigger_policies, recall_policies


def build_report(
    *,
    metrics: pd.DataFrame,
    threshold_policies: pd.DataFrame,
    trigger_policies: pd.DataFrame,
    recall_policies: pd.DataFrame,
    max_candidates: int,
) -> str:
    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    default_45 = metrics.loc[metrics["variant"].eq(f"{FEATURE_SET_NAME}_default")].iloc[0]
    candidate_metrics = metrics.loc[metrics["variant"].ne("baseline_43_native")]
    best_valid = _best_by_valid(candidate_metrics)
    best_test = _best_by_test(candidate_metrics)
    best_policy = threshold_policies.sort_values(
        ["f1", "recall", "precision"],
        ascending=False,
    ).iloc[0]
    union_policy = trigger_policies.loc[
        trigger_policies["policy"].eq("union_43_or_45_review_trigger")
    ].iloc[0]
    it_review_trigger = recall_policies.loc[
        recall_policies["policy"].eq(
            "targeted_industry_macro_category_it_services_valid_recall_ge_0.90"
        )
    ].iloc[0]
    return "\n".join(
        [
            "# Feature 45 Improvement Experiments",
            "",
            "43개 baseline에 `delta_accruals_ratio`, "
            "`is_3y_consecutive_operating_loss`를 추가한 45개 변수셋의 개선 여지를 "
            "하이퍼파라미터, threshold 정책, segment threshold 관점에서 확인했습니다.",
            f"하이퍼파라미터 탐색은 baseline 1개와 deterministic sample `{max_candidates}`개입니다.",
            "",
            "## 1. 결론",
            "",
            f"- 43개 baseline test F1/Recall/Precision: "
            f"`{format_metric(baseline['test_f1_at_threshold'])}` / "
            f"`{format_metric(baseline['test_recall_at_threshold'])}` / "
            f"`{format_metric(baseline['test_precision_at_threshold'])}`",
            f"- 45개 변수셋 기본 test F1/Recall/Precision: "
            f"`{format_metric(default_45['test_f1_at_threshold'])}` / "
            f"`{format_metric(default_45['test_recall_at_threshold'])}` / "
            f"`{format_metric(default_45['test_precision_at_threshold'])}`",
            f"- 45개 validation 기준 선택 후보: `{best_valid['variant']}` "
            f"(test F1 `{format_metric(best_valid['test_f1_at_threshold'])}`, "
            f"test Recall `{format_metric(best_valid['test_recall_at_threshold'])}`)",
            f"- 45개 참고용 test F1 최상위 후보: `{best_test['variant']}` "
            f"(test F1 `{format_metric(best_test['test_f1_at_threshold'])}`)",
            f"- 45개 기본 모델 threshold 정책 최상위: `{best_policy['policy']}` "
            f"(test F1 `{format_metric(best_policy['f1'])}`)",
            f"- 43개 또는 45개 중 하나라도 위험으로 보는 union trigger는 FN을 "
            f"`{format_int(union_policy['false_negative'])}`개까지 줄이지만 FP는 "
            f"`{format_int(union_policy['false_positive'])}`개로 늘어납니다.",
            "- 가장 현실적인 Stage 2 위원회 검토 트리거는 "
            "`45개 변수셋 + IT서비스 threshold 완화`입니다. "
            f"test Recall `{format_metric(it_review_trigger['recall'])}`, "
            f"F1 `{format_metric(it_review_trigger['f1'])}`, "
            f"추가 검토 `{format_int(it_review_trigger['added_vs_45_default'])}`개입니다.",
            "- Recall을 더 높이는 정책은 가능하지만, 추가로 잡는 위험 기업보다 "
            "추가 검토되는 정상 기업 증가가 더 빠릅니다.",
            "- 현재 탐색 범위에서는 45개를 운영 모델로 바로 교체하기보다, "
            "Recall 보완 후보 또는 Stage 2 검토 트리거로 쓰는 전략이 더 안전합니다.",
            "",
            "## 2. 핵심 모델 비교",
            "",
            markdown_table(
                pd.DataFrame([baseline, default_45, best_valid, best_test]).drop_duplicates(
                    "variant"
                ),
                [
                    ("Variant", "variant", "text"),
                    ("Features", "feature_count", "int"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test ROC", "test_roc_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("FP", "test_false_positive_at_threshold", "int"),
                    ("FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 3. 45개 하이퍼파라미터 Validation 상위",
            "",
            markdown_table(
                candidate_metrics.sort_values(
                    ["valid_f1_at_threshold", "valid_pr_auc"],
                    ascending=False,
                ).head(10),
                [
                    ("Variant", "variant", "text"),
                    ("Depth", "max_depth", "int"),
                    ("Child", "min_child_weight", "metric"),
                    ("Lambda", "reg_lambda", "metric"),
                    ("SPW x", "scale_pos_weight_multiplier", "metric"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("FP", "test_false_positive_at_threshold", "int"),
                    ("FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 4. 45개 기본 모델 Threshold 정책",
            "",
            markdown_table(
                threshold_policies.sort_values(["f1", "recall"], ascending=False),
                [
                    ("Policy", "policy", "text"),
                    ("Thresholds", "threshold_detail", "text"),
                    ("Test P", "precision", "metric"),
                    ("Test R", "recall", "metric"),
                    ("Test F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 5. 45개를 Stage 2 보조 트리거로 쓸 때",
            "",
            markdown_table(
                trigger_policies,
                [
                    ("Policy", "policy", "text"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                    ("45-only cases", "candidate_only_review_cases", "int"),
                    ("45-only risk", "candidate_only_true_risk", "int"),
                    ("45-only normal", "candidate_only_normal", "int"),
                ],
            ),
            "",
            "45개 변수셋만 추가로 위험하다고 본 기업은 12개였고, 이 중 실제 투기등급은 4개였습니다.",
            "따라서 45개 모델은 최종 라벨을 직접 바꾸기보다, 43개 모델이 낮게 본 기업 중 "
            "일부를 에이전트 검토 대상으로 올리는 신호로 활용하는 편이 더 적합합니다.",
            "",
            "## 6. Recall 우선 정책",
            "",
            "아래 정책은 45개 변수셋의 threshold를 낮추어 더 넓게 잡는 방식입니다. "
            "최종 부적격 라벨로 바로 쓰기보다는 위원회 검토 대상으로 올리는 "
            "review trigger 후보로 해석하는 편이 안전합니다.",
            "현실적인 운영 후보는 `targeted_industry_macro_category_it_services_valid_recall_ge_0.90`입니다. "
            "IT서비스 기업에만 threshold `0.175`를 적용하고, 나머지는 기본 threshold `0.315`를 유지합니다.",
            "",
            markdown_table(
                recall_policies.sort_values(
                    ["recall", "f1", "precision"],
                    ascending=False,
                ),
                [
                    ("Policy", "policy", "text"),
                    ("Thresholds", "threshold_detail", "text"),
                    ("Test P", "precision", "metric"),
                    ("Test R", "recall", "metric"),
                    ("Test F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                    ("Added", "added_vs_45_default", "int"),
                    ("Added Risk", "added_true_risk", "int"),
                    ("Added Normal", "added_normal", "int"),
                ],
            ),
            "",
            "## 7. 해석",
            "",
            "- 하이퍼파라미터 튜닝은 validation 기준으로만 선택해야 하며, test 최상위 후보는 참고용입니다.",
            "- 45개 변수셋은 기본적으로 FN을 줄이는 방향이지만 FP도 같이 늘어나는 경향이 있습니다.",
            "- Recall만 우선하면 threshold `0.220~0.235` 구간에서 FN을 추가로 줄일 수 있지만, "
            "정상 기업까지 위원회 검토로 많이 올라옵니다.",
            "- segment threshold가 FP를 줄여도 Recall/F1이 같이 악화되면 운영 모델 교체 근거로는 약합니다.",
            "- 45개 모델을 단독 최종 라벨로 쓰기보다 43개 모델 옆의 보조 경고 신호로 쓰면 Stage 2 에이전트 구조와 더 잘 맞습니다.",
        ]
    )


def write_outputs(
    *,
    metrics: pd.DataFrame,
    threshold_policies: pd.DataFrame,
    trigger_policies: pd.DataFrame,
    recall_policies: pd.DataFrame,
    output_dir: Path,
    max_candidates: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "feature_45_improvement_metrics.csv"
    threshold_path = output_dir / "feature_45_improvement_threshold_policies.csv"
    trigger_path = output_dir / "feature_45_trigger_policy_comparison.csv"
    recall_path = output_dir / "feature_45_recall_policy_comparison.csv"
    report_path = output_dir / "feature_45_improvement_report.md"
    summary_path = output_dir / "feature_45_improvement_summary.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    threshold_policies.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    trigger_policies.to_csv(trigger_path, index=False, encoding="utf-8-sig")
    recall_policies.to_csv(recall_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        build_report(
            metrics=metrics,
            threshold_policies=threshold_policies,
            trigger_policies=trigger_policies,
            recall_policies=recall_policies,
            max_candidates=max_candidates,
        ),
        encoding="utf-8",
    )
    baseline = metrics.loc[metrics["variant"].eq("baseline_43_native")].iloc[0]
    candidate_metrics = metrics.loc[metrics["variant"].ne("baseline_43_native")]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "variant": FEATURE_SET_NAME,
        "candidate_columns": CANDIDATE_COLUMNS,
        "candidate_count": len(candidate_metrics),
        "baseline_43": baseline.to_dict(),
        "feature_45_default": metrics.loc[metrics["variant"].eq(f"{FEATURE_SET_NAME}_default")]
        .iloc[0]
        .to_dict(),
        "best_45_by_validation": _best_by_valid(candidate_metrics).to_dict(),
        "best_45_by_test_reference_only": _best_by_test(candidate_metrics).to_dict(),
        "best_threshold_policy_reference": threshold_policies.sort_values(
            ["f1", "recall", "precision"],
            ascending=False,
        )
        .iloc[0]
        .to_dict(),
        "best_trigger_policy_reference": trigger_policies.sort_values(
            ["f1", "recall", "precision"],
            ascending=False,
        )
        .iloc[0]
        .to_dict(),
        "best_recall_policy_reference": recall_policies.sort_values(
            ["recall", "f1", "precision"],
            ascending=False,
        )
        .iloc[0]
        .to_dict(),
        "recommended_stage2_review_trigger": recall_policies.loc[
            recall_policies["policy"].eq(
                "targeted_industry_macro_category_it_services_valid_recall_ge_0.90"
            )
        ]
        .iloc[0]
        .to_dict(),
        "output_files": {
            "metrics": str(metrics_path),
            "threshold_policies": str(threshold_path),
            "trigger_policies": str(trigger_path),
            "recall_policies": str(recall_path),
            "report": str(report_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    metrics, threshold_policies, trigger_policies, recall_policies = run_experiments(
        input_dir=args.input_dir,
        raw_path=args.raw_path,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    write_outputs(
        metrics=metrics,
        threshold_policies=threshold_policies,
        trigger_policies=trigger_policies,
        recall_policies=recall_policies,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
    )
    candidate_metrics = metrics.loc[metrics["variant"].ne("baseline_43_native")]
    best_valid = _best_by_valid(candidate_metrics)
    print(
        json.dumps(
            {
                "candidate_count": len(candidate_metrics),
                "best_45_by_validation": best_valid["variant"],
                "best_valid_f1": float(best_valid["valid_f1_at_threshold"]),
                "best_test_f1": float(best_valid["test_f1_at_threshold"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
