from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from export_feature_43_candidate_feature_pack_experiments import (
    FOCUS_SEGMENTS,
    INPUT_DIR,
    OUTPUT_DIR,
    RAW_PATH,
    THRESHOLD_GRID,
    apply_platt_calibration,
    attach_candidate_columns,
    choose_threshold,
    classification_metrics,
    fit_platt_calibration,
    markdown_table,
    probability_metrics,
    read_id_frames,
    read_raw_features,
    read_split_frames,
    train_xgboost,
    unique_preserve_order,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_NAME = "feature_45_candidate3"
CANDIDATE_COLUMNS = ["delta_accruals_ratio", "is_3y_consecutive_operating_loss"]


def fit_variant(
    *,
    name: str,
    frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, Any]:
    y_train = frames["train"]["is_speculative"].astype(int)
    y_valid = frames["valid"]["is_speculative"].astype(int)
    y_test = frames["test"]["is_speculative"].astype(int)

    model = train_xgboost(
        frames["train"].loc[:, feature_columns],
        y_train,
        frames["valid"].loc[:, feature_columns],
        y_valid,
    )
    valid_raw = model.predict_proba(frames["valid"].loc[:, feature_columns])[:, 1]
    test_raw = model.predict_proba(frames["test"].loc[:, feature_columns])[:, 1]
    coef, intercept = fit_platt_calibration(y_valid, valid_raw)
    valid_prob = apply_platt_calibration(valid_raw, coef, intercept)
    test_prob = apply_platt_calibration(test_raw, coef, intercept)
    threshold, policy_metrics = choose_threshold(y_valid, valid_prob)

    return {
        "name": name,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "best_iteration": getattr(model, "best_iteration", None),
        "y_valid": y_valid,
        "y_test": y_test,
        "valid_prob": valid_prob,
        "test_prob": test_prob,
        "threshold": threshold,
        "threshold_policy": policy_metrics,
    }


def metrics_at_threshold(
    *,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    return {
        "threshold": float(threshold),
        **classification_metrics(y_true, predictions),
    }


def summary_row(variant: dict[str, Any]) -> dict[str, Any]:
    threshold = float(variant["threshold"])
    row: dict[str, Any] = {
        "variant": variant["name"],
        "feature_count": variant["feature_count"],
        "added_features": ", ".join(CANDIDATE_COLUMNS)
        if variant["name"] == CANDIDATE_NAME
        else "",
        "best_iteration": variant["best_iteration"],
        "threshold_tuned": threshold,
    }
    row.update({f"valid_{key}": value for key, value in probability_metrics(
        variant["y_valid"],
        variant["valid_prob"],
    ).items()})
    row.update({f"test_{key}": value for key, value in probability_metrics(
        variant["y_test"],
        variant["test_prob"],
    ).items()})
    row.update({f"valid_{key}_at_threshold": value for key, value in metrics_at_threshold(
        y_true=variant["y_valid"],
        probabilities=variant["valid_prob"],
        threshold=threshold,
    ).items()})
    row.update({f"test_{key}_at_threshold": value for key, value in metrics_at_threshold(
        y_true=variant["y_test"],
        probabilities=variant["test_prob"],
        threshold=threshold,
    ).items()})
    return row


def add_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    output = comparison.copy()
    baseline = output.loc[output["variant"].eq("baseline_43_native")].iloc[0]
    for column in [
        "test_pr_auc",
        "test_roc_auc",
        "test_precision_at_threshold",
        "test_recall_at_threshold",
        "test_f1_at_threshold",
        "test_false_positive_at_threshold",
        "test_false_negative_at_threshold",
    ]:
        output[f"delta_{column}"] = output[column] - baseline[column]
    return output


def build_threshold_sweep(candidate: dict[str, Any], baseline: dict[str, Any]) -> pd.DataFrame:
    baseline_metrics = metrics_at_threshold(
        y_true=baseline["y_test"],
        probabilities=baseline["test_prob"],
        threshold=float(baseline["threshold"]),
    )
    rows = []
    for threshold in THRESHOLD_GRID:
        row = metrics_at_threshold(
            y_true=candidate["y_test"],
            probabilities=candidate["test_prob"],
            threshold=float(threshold),
        )
        for key in ["precision", "recall", "f1", "false_positive", "false_negative"]:
            row[f"delta_{key}"] = row[key] - baseline_metrics[key]
        rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_highlights(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    sweep: pd.DataFrame,
) -> pd.DataFrame:
    baseline_metrics = metrics_at_threshold(
        y_true=baseline["y_test"],
        probabilities=baseline["test_prob"],
        threshold=float(baseline["threshold"]),
    )
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "rule": "valid_policy",
            **sweep.loc[sweep["threshold"].eq(candidate["threshold"])].iloc[0].to_dict(),
        }
    )
    rows.append(
        {
            "rule": "test_best_f1_diagnostic",
            **sweep.sort_values(
                ["f1", "recall", "precision", "threshold"],
                ascending=[False, False, False, True],
            )
            .iloc[0]
            .to_dict(),
        }
    )
    rows.append(
        {
            "rule": "test_max_precision_recall_ge_0.85_diagnostic",
            **sweep.loc[sweep["recall"].ge(0.85)]
            .sort_values(["precision", "f1", "threshold"], ascending=[False, False, False])
            .iloc[0]
            .to_dict(),
        }
    )
    rows.append(
        {
            "rule": "test_max_precision_recall_ge_baseline_diagnostic",
            **sweep.loc[sweep["recall"].ge(baseline_metrics["recall"])]
            .sort_values(["precision", "f1", "threshold"], ascending=[False, False, False])
            .iloc[0]
            .to_dict(),
        }
    )
    return pd.DataFrame(rows)


def build_segments(
    *,
    variant: dict[str, Any],
    id_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    threshold = float(variant["threshold"])
    predictions = variant["test_prob"] >= threshold
    ids = id_frames["test"].reset_index(drop=True)
    segment_base = ids.assign(
        is_speculative=variant["y_test"].reset_index(drop=True),
        prediction=predictions.astype(int),
        prob_speculative=variant["test_prob"],
    )
    rows = []
    for segment_type, segment_name, column, value in FOCUS_SEGMENTS:
        segment = segment_base if column is None else segment_base.loc[segment_base[column] == value]
        if segment.empty:
            continue
        y_true = segment["is_speculative"].astype(int)
        rows.append(
            {
                "variant": variant["name"],
                "segment_type": segment_type,
                "segment": segment_name,
                "rows": len(segment),
                "positives": int((y_true == 1).sum()),
                "negatives": int((y_true == 0).sum()),
                **classification_metrics(y_true, segment["prediction"].to_numpy(dtype=int)),
            }
        )
    return pd.DataFrame(rows)


def selected_sweep_view(sweep: pd.DataFrame, highlights: pd.DataFrame) -> pd.DataFrame:
    selected_thresholds = {
        0.25,
        0.275,
        0.3,
        0.305,
        0.31,
        0.315,
        0.32,
        0.325,
        0.33,
        0.335,
        0.34,
        0.35,
        0.375,
        0.4,
        *highlights["threshold"].tolist(),
    }
    return sweep.loc[sweep["threshold"].round(6).isin({round(value, 6) for value in selected_thresholds})]


def build_report(
    *,
    comparison: pd.DataFrame,
    threshold_highlights: pd.DataFrame,
    threshold_sweep_view: pd.DataFrame,
    segments: pd.DataFrame,
) -> str:
    baseline = comparison.loc[comparison["variant"].eq("baseline_43_native")].iloc[0]
    candidate = comparison.loc[comparison["variant"].eq(CANDIDATE_NAME)].iloc[0]
    fn_delta = int(candidate["delta_test_false_negative_at_threshold"])
    fp_delta = int(candidate["delta_test_false_positive_at_threshold"])

    return "\n".join(
        [
            "# Feature 45 Candidate 3 Experiment",
            "",
            "현재 운영 기준인 43개 변수셋에 `delta_accruals_ratio`, "
            "`is_3y_consecutive_operating_loss`를 추가한 45개 실험 후보입니다.",
            "이 산출물은 운영 모델 교체가 아니라 후보 변수 조합의 재현 가능한 기록입니다.",
            "",
            "## 1. 결론",
            "",
            f"- 43개 baseline test F1/Recall/Precision: "
            f"`{baseline['test_f1_at_threshold']:.4f}` / "
            f"`{baseline['test_recall_at_threshold']:.4f}` / "
            f"`{baseline['test_precision_at_threshold']:.4f}`",
            f"- 45개 후보3 test F1/Recall/Precision: "
            f"`{candidate['test_f1_at_threshold']:.4f}` / "
            f"`{candidate['test_recall_at_threshold']:.4f}` / "
            f"`{candidate['test_precision_at_threshold']:.4f}`",
            f"- 후보3은 FN을 `{abs(fn_delta)}`개 줄였지만 FP는 `{fp_delta}`개 늘었습니다.",
            "- 조기경보 관점에서는 의미가 있으나, F1 기준 운영 반영은 아직 보류가 안전합니다.",
            "",
            "## 2. 43개 Baseline vs 45개 후보3",
            "",
            markdown_table(
                comparison,
                [
                    ("Variant", "variant", "text"),
                    ("Features", "feature_count", "int"),
                    ("Threshold", "threshold_tuned", "metric"),
                    ("Test PR", "test_pr_auc", "metric"),
                    ("Test ROC", "test_roc_auc", "metric"),
                    ("Test P", "test_precision_at_threshold", "metric"),
                    ("Test R", "test_recall_at_threshold", "metric"),
                    ("Test F1", "test_f1_at_threshold", "metric"),
                    ("Test FP", "test_false_positive_at_threshold", "int"),
                    ("Test FN", "test_false_negative_at_threshold", "int"),
                ],
            ),
            "",
            "## 3. 45개 후보3 Threshold 진단",
            "",
            markdown_table(
                threshold_highlights,
                [
                    ("Rule", "rule", "text"),
                    ("Threshold", "threshold", "metric"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 4. Threshold별 Trade-off",
            "",
            markdown_table(
                threshold_sweep_view,
                [
                    ("Threshold", "threshold", "metric"),
                    ("Precision", "precision", "metric"),
                    ("Recall", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 5. Segment 점검",
            "",
            markdown_table(
                segments,
                [
                    ("Variant", "variant", "text"),
                    ("Segment", "segment", "text"),
                    ("Rows", "rows", "int"),
                    ("P", "precision", "metric"),
                    ("R", "recall", "metric"),
                    ("F1", "f1", "metric"),
                    ("FP", "false_positive", "int"),
                    ("FN", "false_negative", "int"),
                ],
            ),
            "",
            "## 6. 45개 기준 개선 방향",
            "",
            "1. `threshold`를 먼저 고정하지 말고 Stage 2 실행 조건과 같이 설계합니다. "
            "45개 후보3은 FN을 줄이는 대신 FP를 늘리므로, 모델 라벨 변경보다 "
            "`에이전트 검토 대상 확대` 용도로 쓰는 편이 안전합니다.",
            "2. FP가 늘어난 시장/산업 구간을 먼저 봅니다. KOSDAQ, 제조업, IT서비스처럼 "
            "오경보가 집중되는 구간은 segment threshold 또는 보류 밴드 정책을 별도로 비교합니다.",
            "3. 후보3의 추가 변수 SHAP을 오류 사례별로 확인합니다. "
            "`delta_accruals_ratio`와 `is_3y_consecutive_operating_loss`가 실제 FN을 "
            "잡는 이유인지, 아니면 특정 산업의 정상 기업을 과민하게 밀어올리는지 확인해야 합니다.",
            "4. 45개 전체 모델을 바로 운영 반영하기보다, 43개 모델 점수와 45개 후보3 점수의 "
            "차이를 `secondary_signal`로 쓰는 앙상블/룰 기반 트리거를 검토합니다.",
            "5. 최종 판단은 test 1회가 아니라 rolling validation에서 Recall 개선과 FP 증가가 "
            "반복적으로 안정적인지 확인한 뒤 결정합니다.",
            "",
        ]
    )


def main() -> None:
    train, valid, test = read_split_frames(INPUT_DIR)
    id_frames = read_id_frames(INPUT_DIR)
    raw = read_raw_features(RAW_PATH)

    base_features = [column for column in train.columns if column != "is_speculative"]
    base_frames = {"train": train, "valid": valid, "test": test}
    candidate_frames = attach_candidate_columns(
        frames=base_frames,
        id_frames=id_frames,
        raw=raw,
        candidate_columns=CANDIDATE_COLUMNS,
    )
    candidate_features = unique_preserve_order([*base_features, *CANDIDATE_COLUMNS])
    if len(candidate_features) != 45:
        raise ValueError(f"Expected 45 features, got {len(candidate_features)}")

    baseline = fit_variant(
        name="baseline_43_native",
        frames=base_frames,
        feature_columns=base_features,
    )
    candidate = fit_variant(
        name=CANDIDATE_NAME,
        frames=candidate_frames,
        feature_columns=candidate_features,
    )

    comparison = add_deltas(pd.DataFrame([summary_row(baseline), summary_row(candidate)]))
    threshold_sweep = build_threshold_sweep(candidate, baseline)
    threshold_highlights = build_threshold_highlights(
        candidate=candidate,
        baseline=baseline,
        sweep=threshold_sweep,
    )
    segments = pd.concat(
        [
            build_segments(variant=baseline, id_frames=id_frames),
            build_segments(variant=candidate, id_frames=id_frames),
        ],
        ignore_index=True,
    )
    feature_list = {
        "variant": CANDIDATE_NAME,
        "base_feature_count": len(base_features),
        "added_feature_count": len(CANDIDATE_COLUMNS),
        "feature_count": len(candidate_features),
        "added_features": CANDIDATE_COLUMNS,
        "feature_columns": candidate_features,
        "note": "Experimental 45-feature candidate; production model remains feature_43.",
    }
    threshold_view = selected_sweep_view(threshold_sweep, threshold_highlights)
    report = build_report(
        comparison=comparison,
        threshold_highlights=threshold_highlights,
        threshold_sweep_view=threshold_view,
        segments=segments,
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "variant": CANDIDATE_NAME,
        "candidate_columns": CANDIDATE_COLUMNS,
        "production_baseline": "baseline_43_native",
        "selection_principle": "Validation threshold policy is used for fair test comparison.",
        "recommendation": (
            "Keep 43-feature model as production baseline; keep 45-feature candidate3 as "
            "a recall-priority experimental candidate or Stage 2 trigger signal."
        ),
        "comparison": comparison.to_dict(orient="records"),
        "threshold_highlights": threshold_highlights.to_dict(orient="records"),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        OUTPUT_DIR / "feature_45_candidate3_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_sweep.to_csv(
        OUTPUT_DIR / "feature_45_candidate3_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    threshold_highlights.to_csv(
        OUTPUT_DIR / "feature_45_candidate3_threshold_highlights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    segments.to_csv(
        OUTPUT_DIR / "feature_45_candidate3_segment_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUTPUT_DIR / "feature_45_candidate3_feature_list.json").write_text(
        json.dumps(feature_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "feature_45_candidate3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "feature_45_candidate3_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
