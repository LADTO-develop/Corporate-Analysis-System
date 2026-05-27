"""Threshold policy experiment builders for model score tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd

from cas.modeling.calibration import DEFAULT_THRESHOLD_GRID

THRESHOLD_GRID = DEFAULT_THRESHOLD_GRID
DEFAULT_THRESHOLD = 0.5
MIN_SEGMENT_ROWS = 30
MIN_SEGMENT_POSITIVES = 5
TARGETED_RECALL_FLOORS = [0.85, 0.80]
KOSDAQ_TARGET = [("market", "KOSDAQ")]
FP_FOCUS_TARGETS = [
    ("market", "KOSDAQ"),
    ("industry_macro_category", "it_services"),
    ("industry_macro_category", "manufacturing"),
]
FOCUS_SEGMENTS = [
    ("market", "KOSDAQ"),
    ("market", "KOSPI"),
    ("industry_macro_category", "it_services"),
    ("industry_macro_category", "manufacturing"),
]


def _resolve_artifact_threshold(scores: pd.DataFrame) -> float:
    threshold_column = scores.get("threshold")
    if threshold_column is None:
        return DEFAULT_THRESHOLD
    thresholds = pd.to_numeric(threshold_column, errors="coerce").dropna().unique()
    if len(thresholds) == 0:
        return DEFAULT_THRESHOLD
    return float(thresholds[0])


def _classification_counts(
    y_true: pd.Series,
    predictions: pd.Series,
) -> dict[str, int]:
    true_positive = int(((y_true == 1) & (predictions == 1)).sum())
    true_negative = int(((y_true == 0) & (predictions == 0)).sum())
    false_positive = int(((y_true == 0) & (predictions == 1)).sum())
    false_negative = int(((y_true == 1) & (predictions == 0)).sum())
    return {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


def _classification_metrics(
    y_true: pd.Series,
    predictions: pd.Series,
) -> dict[str, object]:
    counts = _classification_counts(y_true, predictions)
    true_negative = counts["true_negative"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    true_positive = counts["true_positive"]

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    specificity = true_negative / specificity_denominator if specificity_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (true_positive + true_negative) / len(y_true) if len(y_true) else None
    has_two_classes = y_true.nunique() == 2
    balanced_accuracy = (recall + specificity) / 2 if has_two_classes else None

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
    }


def _probability_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    y_true = frame["is_speculative"].astype(int)
    probabilities = frame["prob_speculative"].astype(float)
    if y_true.nunique() < 2:
        return {"pr_auc": None, "roc_auc": None, "logloss": None}
    clipped_probabilities = np.clip(probabilities.to_numpy(dtype=float), 1e-15, 1 - 1e-15)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "logloss": float(log_loss(y_true, clipped_probabilities)),
    }


def _metrics_for_threshold(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    predictions = (frame["prob_speculative"] >= threshold).astype(int)
    return _classification_metrics(frame["is_speculative"].astype(int), predictions)


def _threshold_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLD_GRID:
        rows.append({"threshold": float(threshold), **_metrics_for_threshold(frame, threshold)})
    return pd.DataFrame(rows)


def _row_metrics(row: pd.Series, *, exclude_threshold: bool = True) -> dict[str, object]:
    if exclude_threshold:
        return cast(dict[str, object], row.drop(labels=["threshold"]).to_dict())
    return cast(dict[str, object], row.to_dict())


def _best_f1_threshold(frame: pd.DataFrame) -> tuple[float, dict[str, object]]:
    sweep = _threshold_sweep(frame)
    row = sweep.sort_values(
        ["f1", "recall", "precision", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(row["threshold"]), _row_metrics(row)


def _precision_at_recall_threshold(
    frame: pd.DataFrame,
    recall_floor: float,
) -> tuple[float, dict[str, object]]:
    sweep = _threshold_sweep(frame)
    candidates = sweep.loc[sweep["recall"] >= recall_floor]
    if candidates.empty:
        return _best_f1_threshold(frame)
    row = candidates.sort_values(
        ["precision", "f1", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(row["threshold"]), _row_metrics(row)


def _conservative_precision_at_recall_threshold(
    frame: pd.DataFrame,
    recall_floor: float,
    fallback_threshold: float,
) -> tuple[float, dict[str, object], bool]:
    sweep = _threshold_sweep(frame)
    conservative_sweep = sweep.loc[sweep["threshold"] >= fallback_threshold].copy()
    if conservative_sweep.empty:
        return fallback_threshold, _metrics_for_threshold(frame, fallback_threshold), True

    candidates = conservative_sweep.loc[conservative_sweep["recall"] >= recall_floor]
    if candidates.empty:
        nearest_index = (conservative_sweep["threshold"] - fallback_threshold).abs().idxmin()
        threshold_row = conservative_sweep.loc[nearest_index]
        return float(threshold_row["threshold"]), _row_metrics(threshold_row), True

    row = candidates.sort_values(
        ["precision", "f1", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(row["threshold"]), _row_metrics(row), False


def _format_threshold_detail(thresholds: Mapping[str, float], fallback: float) -> str:
    pieces = [f"{segment}:{threshold:.3f}" for segment, threshold in sorted(thresholds.items())]
    pieces.append(f"fallback:{fallback:.3f}")
    return "; ".join(pieces)


def _format_targeted_threshold_detail(
    rules: Sequence[Mapping[str, object]],
    fallback: float,
) -> str:
    pieces = [
        f"{rule['dimension']}={rule['segment']}:{float(cast(float, rule['threshold'])):.3f}"
        for rule in rules
    ]
    pieces.append(f"fallback:{fallback:.3f}")
    return "; ".join(pieces)


def _build_segment_thresholds(
    valid_scores: pd.DataFrame,
    dimension: str,
    fallback_threshold: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    thresholds: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for segment, segment_frame in valid_scores.groupby(dimension, dropna=False):
        segment_name = str(segment)
        valid_rows = len(segment_frame)
        valid_positive_rows = int(segment_frame["is_speculative"].sum())
        has_enough_rows = valid_rows >= MIN_SEGMENT_ROWS
        has_enough_positives = valid_positive_rows >= MIN_SEGMENT_POSITIVES
        has_two_classes = segment_frame["is_speculative"].nunique() == 2
        if has_enough_rows and has_enough_positives and has_two_classes:
            threshold, valid_metrics = _best_f1_threshold(segment_frame)
            thresholds[segment_name] = threshold
            fallback_used = False
        else:
            threshold = fallback_threshold
            valid_metrics = _metrics_for_threshold(segment_frame, threshold)
            fallback_used = True
        rows.append(
            {
                "dimension": dimension,
                "segment": segment_name,
                "threshold": threshold,
                "fallback_used": fallback_used,
                "valid_rows": valid_rows,
                "valid_positive_rows": valid_positive_rows,
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_f1": valid_metrics["f1"],
            }
        )
    return thresholds, rows


def _build_targeted_threshold_rules(
    valid_scores: pd.DataFrame,
    targets: Sequence[tuple[str, str]],
    fallback_threshold: float,
    recall_floor: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rules: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for dimension, segment in targets:
        segment_frame = valid_scores.loc[valid_scores[dimension].astype(str) == segment]
        valid_rows = len(segment_frame)
        valid_positive_rows = int(segment_frame["is_speculative"].sum())
        has_enough_rows = valid_rows >= MIN_SEGMENT_ROWS
        has_enough_positives = valid_positive_rows >= MIN_SEGMENT_POSITIVES
        has_two_classes = segment_frame["is_speculative"].nunique() == 2
        if has_enough_rows and has_enough_positives and has_two_classes:
            threshold, valid_metrics, fallback_used = _conservative_precision_at_recall_threshold(
                segment_frame,
                recall_floor,
                fallback_threshold,
            )
        else:
            threshold = fallback_threshold
            valid_metrics = _metrics_for_threshold(segment_frame, threshold)
            fallback_used = True

        rules.append({"dimension": dimension, "segment": segment, "threshold": threshold})
        rows.append(
            {
                "dimension": dimension,
                "segment": segment,
                "threshold": threshold,
                "fallback_used": fallback_used,
                "valid_rows": valid_rows,
                "valid_positive_rows": valid_positive_rows,
                "valid_precision": valid_metrics["precision"],
                "valid_recall": valid_metrics["recall"],
                "valid_f1": valid_metrics["f1"],
            }
        )
    return rules, rows


def _apply_global_policy(
    frame: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    return cast(pd.Series, (frame["prob_speculative"] >= threshold).astype(int))


def _apply_segment_policy(
    frame: pd.DataFrame,
    dimension: str,
    thresholds: Mapping[str, float],
    fallback_threshold: float,
) -> pd.Series:
    applied_thresholds = frame[dimension].astype(str).map(thresholds).fillna(fallback_threshold)
    return cast(pd.Series, (frame["prob_speculative"] >= applied_thresholds).astype(int))


def _apply_targeted_policy(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, object]],
    fallback_threshold: float,
) -> pd.Series:
    applied_thresholds = pd.Series(fallback_threshold, index=frame.index, dtype=float)
    for rule in rules:
        dimension = str(rule["dimension"])
        segment = str(rule["segment"])
        threshold = float(cast(float, rule["threshold"]))
        segment_mask = frame[dimension].astype(str) == segment
        applied_thresholds.loc[segment_mask] = np.maximum(
            applied_thresholds.loc[segment_mask],
            threshold,
        )
    return cast(pd.Series, (frame["prob_speculative"] >= applied_thresholds).astype(int))


def _evaluate_policy(
    *,
    scores: pd.DataFrame,
    split: str,
    policy_name: str,
    policy_type: str,
    selection_rule: str,
    threshold_detail: str,
    predictions: pd.Series,
) -> dict[str, object]:
    split_scores = scores.loc[scores["split"] == split].copy()
    y_true = split_scores["is_speculative"].astype(int)
    classification = _classification_metrics(
        y_true, predictions.loc[split_scores.index].astype(int)
    )
    probability = _probability_metrics(split_scores)
    return {
        "policy_name": policy_name,
        "policy_type": policy_type,
        "selection_rule": selection_rule,
        "split": split,
        "threshold_detail": threshold_detail,
        "rows": len(split_scores),
        "positive_rows": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        **probability,
        **classification,
    }


def _evaluate_focus_segments(
    *,
    scores: pd.DataFrame,
    split: str,
    policy_name: str,
    policy_type: str,
    selection_rule: str,
    predictions: pd.Series,
) -> list[dict[str, object]]:
    split_scores = scores.loc[scores["split"] == split].copy()
    rows: list[dict[str, object]] = []
    for dimension, segment in FOCUS_SEGMENTS:
        segment_scores = split_scores.loc[split_scores[dimension].astype(str) == segment]
        if segment_scores.empty:
            continue
        segment_predictions = predictions.loc[segment_scores.index].astype(int)
        metrics = _classification_metrics(
            segment_scores["is_speculative"].astype(int),
            segment_predictions,
        )
        rows.append(
            {
                "policy_name": policy_name,
                "policy_type": policy_type,
                "selection_rule": selection_rule,
                "split": split,
                "dimension": dimension,
                "segment": segment,
                "rows": len(segment_scores),
                "positive_rows": int(segment_scores["is_speculative"].sum()),
                "positive_rate": float(segment_scores["is_speculative"].mean()),
                **metrics,
            }
        )
    return rows


def _segment_threshold_rows(
    valid_scores: pd.DataFrame,
    global_best_f1_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    segment_threshold_rows: list[dict[str, object]] = []
    segment_policies: list[dict[str, object]] = []
    for dimension, policy_name in [
        ("market", "market_valid_best_f1_by_segment"),
        ("industry_macro_category", "industry_valid_best_f1_by_segment"),
    ]:
        thresholds, rows = _build_segment_thresholds(
            valid_scores,
            dimension,
            global_best_f1_threshold,
        )
        segment_threshold_rows.extend({"policy_name": policy_name, **row} for row in rows)
        segment_policies.append(
            {
                "policy_name": policy_name,
                "policy_type": "segment",
                "selection_rule": f"valid_best_f1_by_{dimension}_fallback_global",
                "dimension": dimension,
                "thresholds": thresholds,
                "fallback_threshold": global_best_f1_threshold,
                "threshold_detail": _format_threshold_detail(thresholds, global_best_f1_threshold),
            }
        )
    return segment_threshold_rows, segment_policies


def _targeted_threshold_rows(
    valid_scores: pd.DataFrame,
    artifact_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    segment_threshold_rows: list[dict[str, object]] = []
    targeted_policies: list[dict[str, object]] = []
    for recall_floor in TARGETED_RECALL_FLOORS:
        for policy_name, targets in [
            (f"kosdaq_conservative_recall_{recall_floor:.2f}", KOSDAQ_TARGET),
            (
                f"targeted_kosdaq_it_mfg_conservative_recall_{recall_floor:.2f}",
                FP_FOCUS_TARGETS,
            ),
        ]:
            rules, rows = _build_targeted_threshold_rules(
                valid_scores,
                targets,
                artifact_threshold,
                recall_floor,
            )
            segment_threshold_rows.extend(
                {
                    "policy_name": policy_name,
                    "target_recall_floor": recall_floor,
                    **row,
                }
                for row in rows
            )
            targeted_policies.append(
                {
                    "policy_name": policy_name,
                    "policy_type": "targeted_segment",
                    "selection_rule": (
                        f"valid_conservative_max_precision_with_recall_ge_{recall_floor:.2f}"
                    ),
                    "rules": rules,
                    "fallback_threshold": artifact_threshold,
                    "threshold_detail": _format_targeted_threshold_detail(
                        rules,
                        artifact_threshold,
                    ),
                }
            )
    return segment_threshold_rows, targeted_policies


def _global_policies(
    valid_scores: pd.DataFrame,
    artifact_threshold: float,
    global_best_f1_threshold: float,
) -> list[dict[str, object]]:
    policies: list[dict[str, object]] = [
        {
            "policy_name": "current_artifact_threshold",
            "policy_type": "global",
            "selection_rule": "saved_model_threshold",
            "threshold": artifact_threshold,
            "threshold_detail": f"{artifact_threshold:.6f}",
        },
        {
            "policy_name": "default_0_5",
            "policy_type": "global",
            "selection_rule": "fixed_0_5",
            "threshold": DEFAULT_THRESHOLD,
            "threshold_detail": f"{DEFAULT_THRESHOLD:.6f}",
        },
        {
            "policy_name": "global_valid_best_f1_grid",
            "policy_type": "global",
            "selection_rule": "valid_best_f1_grid",
            "threshold": global_best_f1_threshold,
            "threshold_detail": f"{global_best_f1_threshold:.6f}",
        },
    ]
    for recall_floor in [0.90, 0.85, 0.80, 0.75]:
        threshold, valid_metrics = _precision_at_recall_threshold(valid_scores, recall_floor)
        policies.append(
            {
                "policy_name": f"global_valid_precision_at_recall_{recall_floor:.2f}",
                "policy_type": "global",
                "selection_rule": f"valid_max_precision_with_recall_ge_{recall_floor:.2f}",
                "threshold": threshold,
                "threshold_detail": f"{threshold:.6f}",
                "valid_selection_precision": valid_metrics["precision"],
                "valid_selection_recall": valid_metrics["recall"],
                "valid_selection_f1": valid_metrics["f1"],
            }
        )
    return policies


def _evaluate_global_policies(
    scores: pd.DataFrame,
    policies: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    policy_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []
    for policy in policies:
        threshold = float(cast(float, policy["threshold"]))
        for split in ["valid", "test"]:
            split_scores = scores.loc[scores["split"] == split]
            predictions = _apply_global_policy(split_scores, threshold)
            policy_rows.append(
                _evaluate_policy(
                    scores=scores,
                    split=split,
                    policy_name=str(policy["policy_name"]),
                    policy_type=str(policy["policy_type"]),
                    selection_rule=str(policy["selection_rule"]),
                    threshold_detail=str(policy["threshold_detail"]),
                    predictions=predictions,
                )
            )
            focus_rows.extend(
                _evaluate_focus_segments(
                    scores=scores,
                    split=split,
                    policy_name=str(policy["policy_name"]),
                    policy_type=str(policy["policy_type"]),
                    selection_rule=str(policy["selection_rule"]),
                    predictions=predictions,
                )
            )
    return policy_rows, focus_rows


def _evaluate_segment_policies(
    scores: pd.DataFrame,
    policies: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    policy_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []
    for policy in policies:
        dimension = str(policy["dimension"])
        thresholds = policy["thresholds"]
        if not isinstance(thresholds, dict):
            raise TypeError("segment policy thresholds must be a dictionary")
        threshold_map = {str(key): float(cast(float, value)) for key, value in thresholds.items()}
        fallback_threshold = float(cast(float, policy["fallback_threshold"]))
        for split in ["valid", "test"]:
            split_scores = scores.loc[scores["split"] == split]
            predictions = _apply_segment_policy(
                split_scores,
                dimension,
                threshold_map,
                fallback_threshold,
            )
            policy_rows.append(
                _evaluate_policy(
                    scores=scores,
                    split=split,
                    policy_name=str(policy["policy_name"]),
                    policy_type=str(policy["policy_type"]),
                    selection_rule=str(policy["selection_rule"]),
                    threshold_detail=str(policy["threshold_detail"]),
                    predictions=predictions,
                )
            )
            focus_rows.extend(
                _evaluate_focus_segments(
                    scores=scores,
                    split=split,
                    policy_name=str(policy["policy_name"]),
                    policy_type=str(policy["policy_type"]),
                    selection_rule=str(policy["selection_rule"]),
                    predictions=predictions,
                )
            )
    return policy_rows, focus_rows


def _evaluate_targeted_policies(
    scores: pd.DataFrame,
    policies: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    policy_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []
    for policy in policies:
        rules = policy["rules"]
        if not isinstance(rules, list):
            raise TypeError("targeted policy rules must be a list")
        fallback_threshold = float(cast(float, policy["fallback_threshold"]))
        for split in ["valid", "test"]:
            split_scores = scores.loc[scores["split"] == split]
            predictions = _apply_targeted_policy(split_scores, rules, fallback_threshold)
            policy_rows.append(
                _evaluate_policy(
                    scores=scores,
                    split=split,
                    policy_name=str(policy["policy_name"]),
                    policy_type=str(policy["policy_type"]),
                    selection_rule=str(policy["selection_rule"]),
                    threshold_detail=str(policy["threshold_detail"]),
                    predictions=predictions,
                )
            )
            focus_rows.extend(
                _evaluate_focus_segments(
                    scores=scores,
                    split=split,
                    policy_name=str(policy["policy_name"]),
                    policy_type=str(policy["policy_type"]),
                    selection_rule=str(policy["selection_rule"]),
                    predictions=predictions,
                )
            )
    return policy_rows, focus_rows


def build_threshold_policy_experiments(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build threshold policy metrics from validation/test prediction scores."""
    valid_scores = scores.loc[scores["split"] == "valid"].copy()
    test_scores = scores.loc[scores["split"] == "test"].copy()
    if valid_scores.empty or test_scores.empty:
        raise ValueError("prediction_scores.csv must contain both valid and test splits.")

    artifact_threshold = _resolve_artifact_threshold(scores)
    global_best_f1_threshold, global_best_f1_valid_metrics = _best_f1_threshold(valid_scores)
    policies = _global_policies(valid_scores, artifact_threshold, global_best_f1_threshold)
    segment_threshold_rows, segment_policies = _segment_threshold_rows(
        valid_scores,
        global_best_f1_threshold,
    )
    targeted_rows, targeted_policies = _targeted_threshold_rows(valid_scores, artifact_threshold)
    segment_threshold_rows.extend(targeted_rows)

    policy_rows: list[dict[str, object]] = []
    focus_rows: list[dict[str, object]] = []
    for rows, focus in [
        _evaluate_global_policies(scores, policies),
        _evaluate_segment_policies(scores, segment_policies),
        _evaluate_targeted_policies(scores, targeted_policies),
    ]:
        policy_rows.extend(rows)
        focus_rows.extend(focus)

    metrics = pd.DataFrame(policy_rows)
    segment_thresholds = pd.DataFrame(segment_threshold_rows)
    focus_segment_metrics = pd.DataFrame(focus_rows)
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    current = test_metrics.loc[test_metrics["policy_name"] == "current_artifact_threshold"].iloc[0]
    best_by_f1 = test_metrics.sort_values(
        ["f1", "recall", "precision"],
        ascending=[False, False, False],
    ).iloc[0]
    summary: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "selection_split": "valid",
        "evaluation_split": "test",
        "threshold_grid": {
            "min": float(THRESHOLD_GRID.min()),
            "max": float(THRESHOLD_GRID.max()),
            "step": 0.005,
        },
        "segment_threshold_minimums": {
            "rows": MIN_SEGMENT_ROWS,
            "positive_rows": MIN_SEGMENT_POSITIVES,
        },
        "global_best_f1_valid": {
            "threshold": global_best_f1_threshold,
            **global_best_f1_valid_metrics,
        },
        "current_test": _row_to_summary(current),
        "best_test_f1_policy": _row_to_summary(best_by_f1),
    }
    return metrics, segment_thresholds, focus_segment_metrics, summary


def _row_to_summary(row: pd.Series) -> dict[str, object]:
    return {
        "policy_name": row["policy_name"],
        "threshold_detail": row["threshold_detail"],
        "precision": _float_or_none(row["precision"]),
        "recall": _float_or_none(row["recall"]),
        "f1": _float_or_none(row["f1"]),
        "false_positive": int(row["false_positive"]),
        "false_negative": int(row["false_negative"]),
    }


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(cast(float, value))
    if np.isnan(numeric):
        return None
    return numeric


def _format_number(value: object, digits: int = 4) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _format_int(value: object) -> str:
    if value is None:
        return "-"
    numeric = _float_or_none(value)
    if numeric is None:
        return "-"
    return str(int(numeric))


def _build_policy_table(metrics: pd.DataFrame) -> str:
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values(["f1", "recall", "precision"], ascending=False)
    rows = [
        "| Policy | Threshold | Precision | Recall | F1 | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in test_metrics.iterrows():
        rows.append(
            "| {policy} | {threshold} | {precision} | {recall} | {f1} | {fp} | {fn} |".format(
                policy=row["policy_name"],
                threshold=row["threshold_detail"],
                precision=_format_number(row["precision"]),
                recall=_format_number(row["recall"]),
                f1=_format_number(row["f1"]),
                fp=_format_int(row["false_positive"]),
                fn=_format_int(row["false_negative"]),
            )
        )
    return "\n".join(rows)


def _build_segment_table(segment_thresholds: pd.DataFrame) -> str:
    if segment_thresholds.empty:
        return "세그먼트 threshold가 생성되지 않았습니다."
    rows = [
        "| Policy | Segment | Threshold | Fallback | Valid Rows | Valid Positives | Valid F1 |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    ordered = segment_thresholds.sort_values(["policy_name", "dimension", "segment"])
    for _, row in ordered.iterrows():
        segment = f"{row['dimension']}={row['segment']}"
        rows.append(
            "| {policy} | {segment} | {threshold} | {fallback} | {rows} | {positives} | {f1} |".format(
                policy=row["policy_name"],
                segment=segment,
                threshold=_format_number(row["threshold"], 3),
                fallback="yes" if bool(row["fallback_used"]) else "no",
                rows=_format_int(row["valid_rows"]),
                positives=_format_int(row["valid_positive_rows"]),
                f1=_format_number(row["valid_f1"]),
            )
        )
    return "\n".join(rows)


def _build_focus_segment_delta_table(focus_segment_metrics: pd.DataFrame) -> str:
    if focus_segment_metrics.empty:
        return "집중 점검 세그먼트 성능이 생성되지 않았습니다."

    selected_policies = [
        "current_artifact_threshold",
        "market_valid_best_f1_by_segment",
        "industry_valid_best_f1_by_segment",
        "kosdaq_conservative_recall_0.85",
        "kosdaq_conservative_recall_0.80",
        "targeted_kosdaq_it_mfg_conservative_recall_0.85",
        "targeted_kosdaq_it_mfg_conservative_recall_0.80",
    ]
    ordered_segments = [
        ("market", "KOSDAQ"),
        ("industry_macro_category", "it_services"),
        ("industry_macro_category", "manufacturing"),
    ]
    selected_segments = set(ordered_segments)
    test_metrics = focus_segment_metrics.loc[focus_segment_metrics["split"] == "test"].copy()
    baseline = test_metrics.loc[test_metrics["policy_name"].eq("current_artifact_threshold")].copy()
    baseline_lookup = {
        (row["dimension"], row["segment"]): row
        for _, row in baseline.iterrows()
        if (row["dimension"], row["segment"]) in selected_segments
    }

    rows = [
        "| Policy | Segment | Precision | Recall | F1 | FP | FP Δ | FN | FN Δ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = test_metrics.loc[
        test_metrics["policy_name"].isin(selected_policies)
        & test_metrics.apply(
            lambda row: (row["dimension"], row["segment"]) in selected_segments,
            axis=1,
        )
    ].copy()
    ordered["policy_order"] = ordered["policy_name"].map(
        {policy: index for index, policy in enumerate(selected_policies)}
    )
    ordered["segment_order"] = ordered.apply(
        lambda row: ordered_segments.index((row["dimension"], row["segment"])),
        axis=1,
    )
    ordered = ordered.sort_values(["policy_order", "segment_order"])

    for _, row in ordered.iterrows():
        key = (row["dimension"], row["segment"])
        baseline_row = baseline_lookup.get(key)
        false_positive = int(row["false_positive"])
        false_negative = int(row["false_negative"])
        false_positive_delta = (
            false_positive - int(baseline_row["false_positive"]) if baseline_row is not None else 0
        )
        false_negative_delta = (
            false_negative - int(baseline_row["false_negative"]) if baseline_row is not None else 0
        )
        rows.append(
            "| {policy} | {segment} | {precision} | {recall} | {f1} | {fp} | {fp_delta:+d} | {fn} | {fn_delta:+d} |".format(
                policy=row["policy_name"],
                segment=f"{row['dimension']}={row['segment']}",
                precision=_format_number(row["precision"]),
                recall=_format_number(row["recall"]),
                f1=_format_number(row["f1"]),
                fp=false_positive,
                fp_delta=false_positive_delta,
                fn=false_negative,
                fn_delta=false_negative_delta,
            )
        )
    return "\n".join(rows)


def build_threshold_policy_report(
    metrics: pd.DataFrame,
    segment_thresholds: pd.DataFrame,
    focus_segment_metrics: pd.DataFrame,
    summary: Mapping[str, object],
) -> str:
    """Render a Markdown report for threshold policy experiment outputs."""
    current = cast(Mapping[str, object], summary["current_test"])
    best = cast(Mapping[str, object], summary["best_test_f1_policy"])
    current_f1 = _float_or_none(current["f1"]) or 0.0
    best_f1 = _float_or_none(best["f1"]) or 0.0
    current_fp = int(cast(int, current["false_positive"]))
    best_fp = int(cast(int, best["false_positive"]))
    current_fn = int(cast(int, current["false_negative"]))
    best_fn = int(cast(int, best["false_negative"]))
    f1_delta = best_f1 - current_f1
    fp_delta = best_fp - current_fp
    fn_delta = best_fn - current_fn

    return f"""# Feature 43 Threshold Policy Experiments

이 리포트는 기존 XGBoost 모델의 확률값은 그대로 두고, decision threshold 정책만
바꿨을 때 test 성능이 어떻게 달라지는지 비교합니다. 모든 threshold는 test가 아닌
validation split에서 선택한 뒤 test에 적용했습니다.

## 1. 핵심 결과

- 현재 artifact threshold 정책: `{current["threshold_detail"]}`
- 현재 test 성능: Precision `{_format_number(current["precision"])}`,
  Recall `{_format_number(current["recall"])}`, F1 `{_format_number(current["f1"])}`,
  FP `{current_fp}`, FN `{current_fn}`
- 이번 실험의 test F1 최상위 정책: `{best["policy_name"]}`
- 최상위 정책 test 성능: Precision `{_format_number(best["precision"])}`,
  Recall `{_format_number(best["recall"])}`, F1 `{_format_number(best["f1"])}`,
  FP `{best_fp}`, FN `{best_fn}`
- 현재 대비 변화: F1 `{f1_delta:+.4f}`, FP `{fp_delta:+d}`, FN `{fn_delta:+d}`

## 2. Test 정책 비교

{_build_policy_table(metrics)}

## 3. 세그먼트 Threshold

시장별/산업별 threshold는 validation split에서 세그먼트별 F1이 가장 높은 값을
선택했습니다. KOSDAQ/IT서비스/제조업 targeted 정책은 현재 artifact threshold보다
낮아지지 않는 보수 후보만 사용했습니다. 단, validation 표본이 `{MIN_SEGMENT_ROWS}`개
미만이거나 양성 라벨이 `{MIN_SEGMENT_POSITIVES}`개 미만이면 전체 global threshold로
fallback했습니다.

{_build_segment_table(segment_thresholds)}

## 4. FP 집중 구간 변화

아래 표는 현재 artifact threshold 대비 KOSDAQ, IT서비스, 제조업의 FP/FN이
어떻게 바뀌는지 보여줍니다.

{_build_focus_segment_delta_table(focus_segment_metrics)}

## 5. 해석

- 현재 artifact threshold는 `global_valid_precision_at_recall_0.85` 정책과 동일하며,
  Recall 0.85 이상을 유지하면서 false positive를 줄이는 단순한 운영 기준입니다.
- 더 높은 Recall을 최우선으로 두면 `global_valid_precision_at_recall_0.90`을,
  false positive 축소를 더 중시하면 `global_valid_precision_at_recall_0.80`을
  보조 기준으로 비교할 수 있습니다.
- KOSDAQ 보수 threshold는 전체 F1을 거의 유지하면서 KOSDAQ FP를 줄이는 후보입니다.
- 성능 숫자만 보면 `industry_valid_best_f1_by_segment`가 가장 좋지만, validation이
  한 해뿐이라 산업별 threshold는 추가 기간 검증 후 production 반영을 권장합니다.
- 발표에서는 "모델 확률은 그대로 두고, 경고 기준선을 목적에 따라 조정할 수 있다"는
  메시지로 설명하면 좋습니다.

## 6. 산출물

- `threshold_policy_experiment_metrics.csv`: 정책별 valid/test 성능
- `threshold_policy_segment_thresholds.csv`: 시장/산업별 threshold와 fallback 여부
- `threshold_policy_focus_segment_metrics.csv`: KOSDAQ/IT서비스/제조업 집중 성능
- `threshold_policy_experiment_summary.json`: 주요 결과 요약
"""
