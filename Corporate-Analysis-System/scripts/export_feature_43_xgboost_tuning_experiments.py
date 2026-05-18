from __future__ import annotations

import argparse
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
INPUT_DIR = ROOT / "data" / "input" / "credit_43_features"
OUTPUT_DIR = ROOT / "data" / "outputs" / "modeling" / "feature_43_xgboost" / "diagnostics"

RANDOM_STATE = 42
PROBABILITY_CLIP_EPSILON = 1e-6
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 6)
RECALL_FLOOR = 0.85
TOP_SEGMENT_CANDIDATES = 8
FOCUS_SEGMENTS = [
    ("overall", "all", None, None),
    ("market", "KOSDAQ", "market", "KOSDAQ"),
    ("market", "KOSPI", "market", "KOSPI"),
    ("industry", "manufacturing", "industry_macro_category", "manufacturing"),
    ("industry", "it_services", "industry_macro_category", "it_services"),
]

BASELINE_PARAMS = {
    "max_depth": 4,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "scale_pos_weight_multiplier": 1.0,
}

SEARCH_GRID = {
    "max_depth": [3, 4, 5],
    "min_child_weight": [1, 3, 5, 8],
    "reg_lambda": [0.5, 1.0, 3.0, 6.0],
    "subsample": [0.75, 0.9],
    "colsample_bytree": [0.75, 0.9],
    "scale_pos_weight_multiplier": [0.8, 1.0, 1.2, 1.5],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OOT-validation XGBoost hyperparameter experiments for the 43-feature credit model."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=96,
        help="Deterministic random sample size from the full search grid, excluding baseline.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def read_split_frames(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(input_dir / "xgb_train.csv", encoding="utf-8-sig"),
        pd.read_csv(input_dir / "xgb_valid.csv", encoding="utf-8-sig"),
        pd.read_csv(input_dir / "xgb_test.csv", encoding="utf-8-sig"),
    )


def read_id_frames(input_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(input_dir / f"xgb_id_{split}.csv", encoding="utf-8-sig")
        for split in ["train", "valid", "test"]
    }


def candidate_grid(max_candidates: int, seed: int) -> list[dict[str, Any]]:
    keys = list(SEARCH_GRID)
    raw_candidates = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(SEARCH_GRID[key] for key in keys))
    ]
    baseline_key = _candidate_key(BASELINE_PARAMS)
    search_candidates = [
        candidate for candidate in raw_candidates if _candidate_key(candidate) != baseline_key
    ]
    rng = np.random.default_rng(seed)
    if max_candidates > 0 and len(search_candidates) > max_candidates:
        selected_indices = sorted(
            rng.choice(len(search_candidates), size=max_candidates, replace=False).tolist()
        )
        search_candidates = [search_candidates[index] for index in selected_indices]
    return [dict(BASELINE_PARAMS), *search_candidates]


def _candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(candidate[key] for key in sorted(BASELINE_PARAMS))


def fit_platt_calibration(
    y_valid: pd.Series,
    valid_probabilities: np.ndarray,
) -> tuple[float, float]:
    clipped = np.clip(
        valid_probabilities,
        PROBABILITY_CLIP_EPSILON,
        1.0 - PROBABILITY_CLIP_EPSILON,
    )
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=RANDOM_STATE, solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, y_valid.astype(int))
    return float(calibrator.coef_[0][0]), float(calibrator.intercept_[0])


def apply_platt_calibration(
    probabilities: np.ndarray,
    coef: float,
    intercept: float,
) -> np.ndarray:
    clipped = np.clip(probabilities, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + np.exp(-(intercept + coef * logits)))


def classification_counts(y_true: pd.Series, predictions: np.ndarray) -> dict[str, int]:
    y_true_array = y_true.to_numpy(dtype=int)
    pred_array = predictions.astype(int)
    return {
        "true_negative": int(((y_true_array == 0) & (pred_array == 0)).sum()),
        "false_positive": int(((y_true_array == 0) & (pred_array == 1)).sum()),
        "false_negative": int(((y_true_array == 1) & (pred_array == 0)).sum()),
        "true_positive": int(((y_true_array == 1) & (pred_array == 1)).sum()),
    }


def classification_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float | int]:
    counts = classification_counts(y_true, predictions)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        **counts,
    }


def probability_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "logloss": float(log_loss(y_true, clipped)),
    }


def choose_threshold(
    y_valid: pd.Series,
    valid_probabilities: np.ndarray,
    recall_floor: float = RECALL_FLOOR,
) -> tuple[float, dict[str, float | int | str]]:
    rows = []
    for threshold in THRESHOLD_GRID:
        predictions = valid_probabilities >= threshold
        rows.append({"threshold": float(threshold), **classification_metrics(y_valid, predictions)})
    sweep = pd.DataFrame(rows)
    candidates = sweep.loc[sweep["recall"] >= recall_floor]
    selection_rule = f"valid_max_precision_with_recall_ge_{recall_floor:.2f}"
    if candidates.empty:
        candidates = sweep
        selection_rule = "valid_best_f1_fallback"
        row = candidates.sort_values(
            ["f1", "recall", "precision", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
    else:
        row = candidates.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    metrics = row.drop(labels=["threshold"]).to_dict()
    metrics["threshold_selection_rule"] = selection_rule
    return float(row["threshold"]), metrics


def train_xgboost(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict[str, Any],
    scale_pos_weight: float,
    seed: int,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=400,
        learning_rate=0.05,
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=0.0,
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def evaluate_candidate(
    *,
    candidate_id: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    base_scale_pos_weight: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    y_train = train["is_speculative"].astype(int)
    y_valid = valid["is_speculative"].astype(int)
    y_test = test["is_speculative"].astype(int)
    x_train = train.loc[:, feature_columns]
    x_valid = valid.loc[:, feature_columns]
    x_test = test.loc[:, feature_columns]
    scale_pos_weight = base_scale_pos_weight * float(params["scale_pos_weight_multiplier"])
    model = train_xgboost(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        params=params,
        scale_pos_weight=scale_pos_weight,
        seed=seed,
    )
    valid_raw_probabilities = model.predict_proba(x_valid)[:, 1]
    test_raw_probabilities = model.predict_proba(x_test)[:, 1]
    coef, intercept = fit_platt_calibration(y_valid, valid_raw_probabilities)
    valid_probabilities = apply_platt_calibration(valid_raw_probabilities, coef, intercept)
    test_probabilities = apply_platt_calibration(test_raw_probabilities, coef, intercept)
    threshold, valid_threshold_metrics = choose_threshold(y_valid, valid_probabilities)
    valid_predictions = valid_probabilities >= threshold
    test_predictions = test_probabilities >= threshold
    valid_probability_metrics = probability_metrics(y_valid, valid_probabilities)
    test_probability_metrics = probability_metrics(y_test, test_probabilities)
    valid_classification_metrics = classification_metrics(y_valid, valid_predictions)
    test_classification_metrics = classification_metrics(y_test, test_predictions)

    row: dict[str, Any] = {
        "candidate_id": candidate_id,
        "is_baseline": candidate_id == "baseline_current",
        **params,
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": getattr(model, "best_iteration", None),
        "threshold_tuned": threshold,
        "threshold_selection_rule": valid_threshold_metrics["threshold_selection_rule"],
        "valid_precision_at_policy": valid_threshold_metrics["precision"],
        "valid_recall_at_policy": valid_threshold_metrics["recall"],
        "valid_f1_at_policy": valid_threshold_metrics["f1"],
    }
    row.update({f"valid_{key}": value for key, value in valid_probability_metrics.items()})
    row.update(
        {f"valid_{key}_at_threshold": value for key, value in valid_classification_metrics.items()}
    )
    row.update({f"test_{key}": value for key, value in test_probability_metrics.items()})
    row.update(
        {f"test_{key}_at_threshold": value for key, value in test_classification_metrics.items()}
    )
    return row, test_probabilities


def build_segment_rows(
    *,
    metrics: pd.DataFrame,
    test_probabilities_by_candidate: dict[str, np.ndarray],
    test: pd.DataFrame,
    test_id: pd.DataFrame,
) -> pd.DataFrame:
    y_test = test["is_speculative"].astype(int).reset_index(drop=True)
    base = test_id.reset_index(drop=True).copy()
    candidate_ids = _segment_candidate_ids(metrics)
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = metrics.loc[metrics["candidate_id"].eq(candidate_id)].iloc[0]
        probabilities = test_probabilities_by_candidate[candidate_id]
        predictions = probabilities >= float(candidate["threshold_tuned"])
        frame = base.assign(
            is_speculative=y_test,
            prediction=predictions.astype(int),
            prob_speculative=probabilities,
        )
        for segment_type, segment_name, column, value in FOCUS_SEGMENTS:
            segment = frame if column is None else frame.loc[frame[column] == value]
            if segment.empty:
                continue
            segment_y = segment["is_speculative"].astype(int)
            segment_predictions = segment["prediction"].to_numpy(dtype=int)
            segment_metrics = classification_metrics(segment_y, segment_predictions)
            negatives = int((segment_y == 0).sum())
            positives = int((segment_y == 1).sum())
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "segment_type": segment_type,
                    "segment": segment_name,
                    "rows": len(segment),
                    "positives": positives,
                    "negatives": negatives,
                    "fp_rate_among_negatives": (
                        segment_metrics["false_positive"] / negatives if negatives else None
                    ),
                    "fn_rate_among_positives": (
                        segment_metrics["false_negative"] / positives if positives else None
                    ),
                    **segment_metrics,
                }
            )
    return pd.DataFrame(rows)


def _segment_candidate_ids(metrics: pd.DataFrame) -> list[str]:
    selected = ["baseline_current"]
    top_valid = (
        metrics.sort_values(
            ["valid_f1_at_threshold", "valid_pr_auc", "valid_precision_at_threshold"],
            ascending=False,
        )
        .head(TOP_SEGMENT_CANDIDATES)["candidate_id"]
        .tolist()
    )
    top_test = (
        metrics.sort_values(
            ["test_f1_at_threshold", "test_pr_auc", "test_precision_at_threshold"],
            ascending=False,
        )
        .head(3)["candidate_id"]
        .tolist()
    )
    for candidate_id in [*top_valid, *top_test]:
        if candidate_id not in selected:
            selected.append(candidate_id)
    return selected


def run_experiments(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    id_frames: dict[str, pd.DataFrame],
    max_candidates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = [column for column in train.columns if column != "is_speculative"]
    y_train = train["is_speculative"].astype(int)
    positive = int(y_train.sum())
    negative = int(len(y_train) - positive)
    base_scale_pos_weight = float(negative / positive) if positive else 1.0
    rows: list[dict[str, Any]] = []
    probabilities_by_candidate: dict[str, np.ndarray] = {}
    candidates = candidate_grid(max_candidates=max_candidates, seed=seed)
    for index, params in enumerate(candidates):
        candidate_id = "baseline_current" if index == 0 else f"candidate_{index:03d}"
        row, test_probabilities = evaluate_candidate(
            candidate_id=candidate_id,
            params=params,
            train=train,
            valid=valid,
            test=test,
            feature_columns=feature_columns,
            base_scale_pos_weight=base_scale_pos_weight,
            seed=seed,
        )
        rows.append(row)
        probabilities_by_candidate[candidate_id] = test_probabilities
    metrics = pd.DataFrame(rows)
    segments = build_segment_rows(
        metrics=metrics,
        test_probabilities_by_candidate=probabilities_by_candidate,
        test=test,
        test_id=id_frames["test"],
    )
    return metrics, segments


def format_metric(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


def format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value):,}"


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for label, _, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame.to_dict(orient="records"):
        values = []
        for _, column, kind in columns:
            value = row.get(column)
            if kind == "metric":
                values.append(format_metric(value))
            elif kind == "int":
                values.append(format_int(value))
            else:
                values.append(str(value) if value is not None else "")
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def build_report(metrics: pd.DataFrame, segments: pd.DataFrame, max_candidates: int) -> str:
    baseline = metrics.loc[metrics["candidate_id"].eq("baseline_current")].iloc[0]
    best_valid = _best_by_valid(metrics)
    best_test = _best_by_test(metrics)
    valid_delta = float(best_valid["valid_f1_at_threshold"]) - float(
        baseline["valid_f1_at_threshold"]
    )
    test_delta = float(best_valid["test_f1_at_threshold"]) - float(baseline["test_f1_at_threshold"])
    test_only_delta = float(best_test["test_f1_at_threshold"]) - float(
        baseline["test_f1_at_threshold"]
    )
    recommendation = _recommendation_text(
        best_valid=best_valid,
        baseline=baseline,
        valid_delta=valid_delta,
        test_delta=test_delta,
    )
    top_valid = _rank_by_valid(metrics).head(12)
    top_test = _rank_by_test(metrics).head(8)
    baseline_kosdaq = _segment_row(segments, "baseline_current", "KOSDAQ")
    best_valid_kosdaq = _segment_row(segments, str(best_valid["candidate_id"]), "KOSDAQ")

    return "\n".join(
        [
            "# XGBoost Hyperparameter Tuning Experiments",
            "",
            "43-feature XGBoost 모델의 하이퍼파라미터를 OOT validation 기준으로 탐색한 실험입니다.",
            "모든 후보는 XGBoost native missing, Platt scaling, validation 기준 "
            f"`recall >= {RECALL_FLOOR:.2f}` 조건에서 precision 최대 threshold를 사용했습니다.",
            f"검색 후보는 baseline 1개와 deterministic random sample `{max_candidates}`개입니다.",
            "",
            "## 1. 결론",
            "",
            f"- Baseline valid/test F1: `{format_metric(baseline['valid_f1_at_threshold'])}` / "
            f"`{format_metric(baseline['test_f1_at_threshold'])}`",
            f"- Validation 기준 선택 후보: `{best_valid['candidate_id']}` "
            f"(valid F1 `{format_metric(best_valid['valid_f1_at_threshold'])}`, "
            f"test F1 `{format_metric(best_valid['test_f1_at_threshold'])}`)",
            f"- Validation 선택 후보의 baseline 대비 변화: valid F1 `{valid_delta:+.4f}`, "
            f"test F1 `{test_delta:+.4f}`",
            f"- 참고용 test F1 최상위 후보: `{best_test['candidate_id']}` "
            f"(test F1 `{format_metric(best_test['test_f1_at_threshold'])}`, "
            f"baseline 대비 `{test_only_delta:+.4f}`)",
            recommendation,
            "",
            "## 2. Validation 기준 상위 후보",
            "",
            markdown_table(
                top_valid,
                [
                    ("Candidate", "candidate_id", "text"),
                    ("Depth", "max_depth", "int"),
                    ("Child", "min_child_weight", "metric"),
                    ("Lambda", "reg_lambda", "metric"),
                    ("Subsample", "subsample", "metric"),
                    ("Colsample", "colsample_bytree", "metric"),
                    ("SPW x", "scale_pos_weight_multiplier", "metric"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("Valid PR", "valid_pr_auc", "metric"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 3. 참고용 Test 상위 후보",
            "",
            "아래 표는 사후 점검용입니다. 모델 선택 기준으로는 사용하지 않습니다.",
            "",
            markdown_table(
                top_test,
                [
                    ("Candidate", "candidate_id", "text"),
                    ("Valid F1", "valid_f1_at_threshold", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 4. KOSDAQ 오류 관점",
            "",
            f"- Baseline KOSDAQ FP/FN: `{format_int(baseline_kosdaq.get('false_positive'))}` / "
            f"`{format_int(baseline_kosdaq.get('false_negative'))}`",
            f"- Validation 선택 후보 KOSDAQ FP/FN: "
            f"`{format_int(best_valid_kosdaq.get('false_positive'))}` / "
            f"`{format_int(best_valid_kosdaq.get('false_negative'))}`",
            "",
            "## 5. 해석 원칙",
            "",
            "- 하이퍼파라미터 선택은 validation 성능만 사용합니다.",
            "- test 성능이 좋은 후보는 참고만 하고, production 교체 전에는 재현성 검증 또는 추가 OOT split 확인이 필요합니다.",
            "- 개선 폭이 작으면 현재 production baseline을 유지하고, Stage 2 외부근거 보완으로 FN을 줄이는 전략이 더 안전합니다.",
        ]
    )


def _rank_by_valid(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(
        [
            "valid_f1_at_threshold",
            "valid_pr_auc",
            "valid_precision_at_threshold",
            "test_f1_at_threshold",
        ],
        ascending=False,
    )


def _rank_by_test(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values(
        [
            "test_f1_at_threshold",
            "test_pr_auc",
            "test_precision_at_threshold",
            "valid_f1_at_threshold",
        ],
        ascending=False,
    )


def _best_by_valid(metrics: pd.DataFrame) -> pd.Series:
    return _rank_by_valid(metrics).iloc[0]


def _best_by_test(metrics: pd.DataFrame) -> pd.Series:
    return _rank_by_test(metrics).iloc[0]


def _segment_row(segments: pd.DataFrame, candidate_id: str, segment: str) -> dict[str, Any]:
    frame = segments.loc[
        segments["candidate_id"].eq(candidate_id) & segments["segment"].eq(segment)
    ]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def _recommendation_text(
    *,
    best_valid: pd.Series,
    baseline: pd.Series,
    valid_delta: float,
    test_delta: float,
) -> str:
    if str(best_valid["candidate_id"]) == "baseline_current":
        return (
            "- Validation 기준으로도 현재 baseline이 최상위입니다. 운영 모델은 유지하고, "
            "FN 보완은 외부근거/committee_view 쪽에서 진행하는 편이 좋습니다."
        )
    if valid_delta >= 0.01 and test_delta >= 0.005:
        return (
            "- Validation과 test에서 모두 의미 있는 개선이 확인되었습니다. "
            "다음 후보 모델로 별도 재학습/대시보드 검증을 진행할 가치가 있습니다."
        )
    if valid_delta >= 0.01 and test_delta < 0:
        return (
            "- Validation에서는 좋아졌지만 test에서 악화되어 과적합 가능성이 있습니다. "
            "production 반영은 보류하는 편이 안전합니다."
        )
    return "- 개선 폭이 작습니다. 현 모델을 바로 교체하기보다 후보로 보관하고 추가 OOT 검증이 필요합니다."


def write_outputs(
    *,
    metrics: pd.DataFrame,
    segments: pd.DataFrame,
    output_dir: Path,
    max_candidates: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_sorted = _rank_by_valid(metrics)
    top_valid = metrics_sorted.head(20)
    metrics_path = output_dir / "xgboost_hyperparameter_tuning_metrics.csv"
    top_path = output_dir / "xgboost_hyperparameter_tuning_top_valid.csv"
    segments_path = output_dir / "xgboost_hyperparameter_tuning_segment_metrics.csv"
    report_path = output_dir / "xgboost_hyperparameter_tuning_report.md"
    summary_path = output_dir / "xgboost_hyperparameter_tuning_summary.json"

    metrics_sorted.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    top_valid.to_csv(top_path, index=False, encoding="utf-8-sig")
    segments.to_csv(segments_path, index=False, encoding="utf-8-sig")
    report_path.write_text(build_report(metrics, segments, max_candidates), encoding="utf-8")

    baseline = metrics.loc[metrics["candidate_id"].eq("baseline_current")].iloc[0]
    best_valid = _best_by_valid(metrics)
    best_test = _best_by_test(metrics)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_rule": "rank by validation F1 at recall>=0.85 threshold policy",
        "candidate_count": len(metrics),
        "baseline": baseline.to_dict(),
        "best_by_validation": best_valid.to_dict(),
        "best_by_test_reference_only": best_test.to_dict(),
        "output_files": {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "top_valid": str(top_path.relative_to(ROOT)),
            "segment_metrics": str(segments_path.relative_to(ROOT)),
            "report": str(report_path.relative_to(ROOT)),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    train, valid, test = read_split_frames(args.input_dir)
    id_frames = read_id_frames(args.input_dir)
    metrics, segments = run_experiments(
        train=train,
        valid=valid,
        test=test,
        id_frames=id_frames,
        max_candidates=args.max_candidates,
        seed=args.seed,
    )
    write_outputs(
        metrics=metrics,
        segments=segments,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
    )
    best_valid = _best_by_valid(metrics)
    baseline = metrics.loc[metrics["candidate_id"].eq("baseline_current")].iloc[0]
    print(
        json.dumps(
            {
                "candidate_count": len(metrics),
                "best_by_validation": str(best_valid["candidate_id"]),
                "best_valid_f1": float(best_valid["valid_f1_at_threshold"]),
                "best_test_f1": float(best_valid["test_f1_at_threshold"]),
                "baseline_valid_f1": float(baseline["valid_f1_at_threshold"]),
                "baseline_test_f1": float(baseline["test_f1_at_threshold"]),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
