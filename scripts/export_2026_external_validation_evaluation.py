"""Evaluate 2026 external credit-rating labels against Stage 1 and Stage 2 outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cas.agents.nodes import committee_node, rule_engine_node  # noqa: E402
from cas.agents.nodes.base_prediction_node import _top_risk_drivers  # noqa: E402
from cas.utils.io import read_json, read_yaml  # noqa: E402

LABELS_2026_PATH = ROOT / "data/evaluation/credit_rating_labels_2026.csv"
INFERENCE_2026_PATH = ROOT / "data/input/credit_46_features/feature_46_inference_2026.csv"
MODEL_DIR = ROOT / "data/outputs/modeling/feature_46_xgboost"
MODEL_PATH = MODEL_DIR / "xgboost_model.json"
METADATA_PATH = MODEL_DIR / "model_artifact_metadata.json"
OUTPUT_DIR = MODEL_DIR / "diagnostics"

KEY_COLUMNS = ["market", "stock_code", "fiscal_year", "eval_year"]
MODEL_LABEL_POSITIVE = "부적격"
COMMITTEE_LABEL_ELIGIBLE = "적격"
COMMITTEE_LABEL_HOLD = "보류"
COMMITTEE_LABEL_REJECT = "부적격"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-2026", type=Path, default=LABELS_2026_PATH)
    parser.add_argument("--inference-2026", type=Path, default=INFERENCE_2026_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--committee-mode",
        choices=["offline"],
        default="offline",
        help=(
            "Currently evaluates deterministic Stage 2 without live external API calls. "
            "This keeps the 2026 label benchmark reproducible."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.labels_2026)
    inference = read_inference(args.inference_2026)
    scored = score_inference(
        inference,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
    )
    evaluation = attach_labels(scored, labels)
    evaluation = run_offline_committee(evaluation)
    evaluation = add_evaluation_flags(evaluation)

    summary = build_summary(evaluation, committee_mode=args.committee_mode)
    report = build_report(summary)

    scores_path = args.output_dir / "external_validation_2026_scores.csv"
    summary_path = args.output_dir / "external_validation_2026_summary.json"
    report_path = args.output_dir / "external_validation_2026_report.md"
    evaluation.to_csv(scores_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "scores": str(scores_path.relative_to(ROOT)),
                "summary": str(summary_path.relative_to(ROOT)),
                "report": str(report_path.relative_to(ROOT)),
                "n_labeled_rows": len(evaluation),
                "stage1": summary["stage1_model"],
                "stage2_review_route": summary["stage2_review_route"],
                "stage2_review_or_reject": summary["stage2_committee_review_or_reject"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    labels = labels.copy()
    labels["stock_code"] = labels["stock_code"].map(normalize_stock_code)
    for column in ["fiscal_year", "eval_year", "is_speculative", "credit_rating_rank"]:
        labels[column] = pd.to_numeric(labels[column], errors="coerce")
    return labels


def read_inference(path: Path) -> pd.DataFrame:
    inference = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    inference = inference.copy()
    inference["stock_code"] = inference["stock_code"].map(normalize_stock_code)
    for column in ["fiscal_year", "eval_year"]:
        inference[column] = pd.to_numeric(inference[column], errors="coerce")
    return inference


def normalize_stock_code(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text.upper()


def score_inference(
    inference: pd.DataFrame,
    *,
    model_path: Path,
    metadata_path: Path,
) -> pd.DataFrame:
    import xgboost as xgb

    metadata = read_json(metadata_path)
    feature_columns = [str(column) for column in metadata["feature_columns"]]
    missing_strategy = str(metadata.get("missing_value_strategy", "xgboost_native_missing"))
    frame = build_model_frame(
        inference,
        feature_columns=feature_columns,
        fill_values=dict(metadata.get("fill_values", {})),
        native_missing=missing_strategy == "xgboost_native_missing",
    )
    booster = xgb.Booster()
    booster.load_model(model_path)
    raw_probabilities = booster.predict(xgb.DMatrix(frame))
    probabilities = np.array(
        [
            apply_probability_calibration(
                float(probability),
                metadata.get("probability_calibration"),
            )
            for probability in raw_probabilities
        ]
    )
    threshold = float(metadata.get("threshold_tuned") or metadata.get("threshold_default") or 0.5)
    cfg = read_yaml("configs/runtime/analysis.yaml")
    band_thresholds = dict(cfg.get("rule_engine", {}).get("risk_band_thresholds", {}))
    watch_threshold = float(band_thresholds.get("watch", 0.4))
    high_risk_threshold = float(band_thresholds.get("high_risk", 0.65))

    output = inference.copy()
    output["prob_speculative_raw"] = raw_probabilities
    output["prob_speculative"] = probabilities
    output["threshold"] = threshold
    output["pred_label_tuned"] = output["prob_speculative"].ge(threshold).astype(int)
    output["prediction_label"] = np.where(output["pred_label_tuned"].eq(1), "부적격", "투자적격")
    output["risk_band"] = [
        risk_band(
            probability, watch_threshold=watch_threshold, high_risk_threshold=high_risk_threshold
        )
        for probability in output["prob_speculative"]
    ]
    output["_model_feature_frame"] = list(frame.to_dict(orient="records"))
    return output


def build_model_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    fill_values: dict[str, object],
    native_missing: bool,
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        values = pd.to_numeric(frame.get(column), errors="coerce")
        if native_missing:
            output[column] = values
        else:
            output[column] = values.fillna(float(fill_values.get(column, 0.0)))
    return output.loc[:, feature_columns]


def apply_probability_calibration(probability: float, calibration: object) -> float:
    if not isinstance(calibration, dict) or calibration.get("method") != "platt_sigmoid":
        return probability
    epsilon = float(calibration.get("clip_epsilon", 1e-6))
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    logit = math.log(clipped / (1.0 - clipped))
    coef = float(calibration.get("coef"))
    intercept = float(calibration.get("intercept"))
    return float(1.0 / (1.0 + math.exp(-(intercept + coef * logit))))


def risk_band(
    probability: float,
    *,
    watch_threshold: float,
    high_risk_threshold: float,
) -> str:
    if probability >= high_risk_threshold:
        return "high_risk"
    if probability >= watch_threshold:
        return "watch"
    return "stable"


def attach_labels(scored: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_columns = [
        *KEY_COLUMNS,
        "is_speculative",
        "credit_rating",
        "credit_rating_rank",
        "rating_agency",
        "rating_agency_code",
        "rating_target",
        "rating_date",
        "current_outlook",
    ]
    merged = scored.merge(
        labels.loc[:, [column for column in label_columns if column in labels.columns]],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("", "_actual"),
        validate="one_to_one",
    )
    merged = merged.rename(columns={"is_speculative": "actual_is_speculative"})
    merged["actual_is_speculative"] = (
        pd.to_numeric(merged["actual_is_speculative"], errors="coerce").fillna(0).astype(int)
    )
    merged["actual_label"] = np.where(
        merged["actual_is_speculative"].eq(1),
        "투기등급",
        "투자적격",
    )
    return merged


def run_offline_committee(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        row_dict = row.drop(labels=["_model_feature_frame"], errors="ignore").to_dict()
        model_feature_frame = pd.DataFrame([row["_model_feature_frame"]])
        top_drivers = stage1_top_drivers(model_feature_frame)
        state: dict[str, Any] = {
            "company_id": f"{row['market']}-{row['stock_code']}-{int(row['fiscal_year'])}",
            "company_name": str(row["corp_name"]),
            "market": str(row["market"]),
            "analysis_year": int(row["fiscal_year"]),
            "company_profile": {
                "company_id": str(row["stock_code"]),
                "company_name": str(row["corp_name"]),
                "market": str(row["market"]),
            },
            "source_feature_row": row_dict,
            "model_view": stage1_model_payload(row, top_drivers=top_drivers),
            "xgboost_result": stage1_model_payload(row, top_drivers=top_drivers),
            "news_cache_snapshot": {
                "status": "disabled",
                "enabled": False,
                "items": [],
                "as_of_date": str(row.get("rating_date") or ""),
                "message": "External evidence was not collected for this reproducible benchmark.",
            },
        }
        state.update(rule_engine_node.run(state))
        state.update(committee_node.run(state))
        committee_view = dict(state.get("committee_view") or {})
        rows.append(
            {
                "market": row["market"],
                "stock_code": row["stock_code"],
                "fiscal_year": row["fiscal_year"],
                "eval_year": row["eval_year"],
                "final_committee_label": committee_view.get("final_committee_label", ""),
                "veto_triggered": bool(committee_view.get("veto_triggered", False)),
                "hidden_tail_risk_flag": bool(committee_view.get("hidden_tail_risk_flag", False)),
                "conflict_resolution": committee_view.get("conflict_resolution", ""),
                "final_review_memo": committee_view.get("final_review_memo", ""),
                "committee_confidence": state.get("final_confidence"),
            }
        )
    committee = pd.DataFrame(rows)
    return frame.drop(columns=["_model_feature_frame"]).merge(
        committee,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )


def stage1_top_drivers(model_feature_frame: pd.DataFrame) -> list[dict[str, float | str]]:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    try:
        return [
            {"name": name, "value": value}
            for name, value in _top_risk_drivers(booster, model_feature_frame)
        ]
    except Exception:
        return []


def stage1_model_payload(
    row: pd.Series, *, top_drivers: list[dict[str, float | str]]
) -> dict[str, Any]:
    return {
        "model_name": "credit_46_features",
        "model_version": "feature_46_xgboost",
        "probability_speculative": round(float(row["prob_speculative"]), 4),
        "prediction_label": str(row["prediction_label"]),
        "risk_band": str(row["risk_band"]),
        "threshold": round(float(row["threshold"]), 4),
        "top_drivers": top_drivers,
    }


def add_evaluation_flags(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    stage1_positive = output["pred_label_tuned"].astype(int).eq(1)
    actual_positive = output["actual_is_speculative"].astype(int).eq(1)
    committee_label = output["final_committee_label"].astype(str)
    output["stage1_error_type"] = np.select(
        [
            stage1_positive & actual_positive,
            stage1_positive & ~actual_positive,
            ~stage1_positive & actual_positive,
            ~stage1_positive & ~actual_positive,
        ],
        ["TP", "FP", "FN", "TN"],
        default="unknown",
    )
    output["committee_review_or_reject"] = committee_label.isin(
        [COMMITTEE_LABEL_HOLD, COMMITTEE_LABEL_REJECT]
    )
    output["committee_reject_only"] = committee_label.eq(COMMITTEE_LABEL_REJECT)
    output["stage2_review_route"] = stage1_positive | output["committee_review_or_reject"]
    output["stage2_effect"] = np.select(
        [
            output["stage1_error_type"].eq("FN") & output["committee_review_or_reject"],
            output["stage1_error_type"].eq("FN") & ~output["committee_review_or_reject"],
            output["stage1_error_type"].eq("FP") & committee_label.ne(COMMITTEE_LABEL_REJECT),
            output["stage1_error_type"].eq("FP") & committee_label.eq(COMMITTEE_LABEL_REJECT),
            output["stage1_error_type"].eq("TP") & output["committee_review_or_reject"],
            output["stage1_error_type"].eq("TP") & ~output["committee_review_or_reject"],
            output["stage1_error_type"].eq("TN") & committee_label.eq(COMMITTEE_LABEL_ELIGIBLE),
            output["stage1_error_type"].eq("TN") & committee_label.ne(COMMITTEE_LABEL_ELIGIBLE),
        ],
        [
            "fn_caught_as_review_or_reject",
            "fn_still_missed",
            "fp_softened_to_eligible_or_hold",
            "fp_kept_as_reject",
            "tp_risk_preserved",
            "tp_softened_too_much",
            "tn_kept_eligible",
            "tn_escalated_to_hold_or_reject",
        ],
        default="unknown",
    )
    return output


def build_summary(frame: pd.DataFrame, *, committee_mode: str) -> dict[str, Any]:
    y_true = frame["actual_is_speculative"].astype(int)
    stage1_pred = frame["pred_label_tuned"].astype(int)
    stage2_route_pred = frame["stage2_review_route"].astype(int)
    committee_review_pred = frame["committee_review_or_reject"].astype(int)
    committee_reject_pred = frame["committee_reject_only"].astype(int)
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "benchmark_name": "2026_external_rating_validation",
        "committee_mode": committee_mode,
        "committee_mode_note": (
            "Stage 2 was run with deterministic/offline evidence only. Live Naver/Tavily/OpenDART "
            "evidence is intentionally excluded so the benchmark is reproducible."
        ),
        "rows": len(frame),
        "positive_count": int(y_true.sum()),
        "negative_count": int((1 - y_true).sum()),
        "positive_rate": safe_rate(int(y_true.sum()), len(frame)),
        "stage1_model": classification_metrics(
            y_true=y_true,
            y_score=frame["prob_speculative"],
            y_pred=stage1_pred,
        ),
        "stage2_review_route": classification_metrics(
            y_true=y_true,
            y_score=None,
            y_pred=stage2_route_pred,
        ),
        "stage2_committee_review_or_reject": classification_metrics(
            y_true=y_true,
            y_score=None,
            y_pred=committee_review_pred,
        ),
        "stage2_committee_reject_only": classification_metrics(
            y_true=y_true,
            y_score=None,
            y_pred=committee_reject_pred,
        ),
        "stage2_effect_counts": count_dict(frame["stage2_effect"]),
        "stage1_error_counts": count_dict(frame["stage1_error_type"]),
        "committee_label_counts": count_dict(frame["final_committee_label"]),
        "review_load": {
            "stage1_reject_count": int(stage1_pred.sum()),
            "stage2_review_route_count": int(stage2_route_pred.sum()),
            "stage2_review_or_reject_count": int(committee_review_pred.sum()),
            "stage2_reject_only_count": int(committee_reject_pred.sum()),
        },
        "by_market": grouped_summary(frame, "market"),
        "by_rating_boundary": grouped_summary(frame, "credit_rating"),
        "by_rating_agency": grouped_summary(frame, "rating_agency"),
    }


def classification_metrics(
    *,
    y_true: pd.Series,
    y_score: pd.Series | None,
    y_pred: pd.Series,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
    if y_score is not None and y_true.nunique() == 2:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["pr_auc"] = None
        metrics["roc_auc"] = None
    return metrics


def grouped_summary(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(column, dropna=False):
        y_true = group["actual_is_speculative"].astype(int)
        rows.append(
            {
                column: "" if pd.isna(value) else str(value),
                "rows": len(group),
                "positive_count": int(y_true.sum()),
                "stage1_recall": safe_metric_recall(group, "pred_label_tuned"),
                "stage2_route_recall": safe_metric_recall(group, "stage2_review_route"),
                "stage2_review_recall": safe_metric_recall(group, "committee_review_or_reject"),
                "stage1_fp": int(
                    (group["actual_is_speculative"].eq(0) & group["pred_label_tuned"].eq(1)).sum()
                ),
                "stage2_route_fp": int(
                    (
                        group["actual_is_speculative"].eq(0)
                        & group["stage2_review_route"].astype(bool)
                    ).sum()
                ),
                "stage2_review_fp": int(
                    (
                        group["actual_is_speculative"].eq(0)
                        & group["committee_review_or_reject"].astype(bool)
                    ).sum()
                ),
            }
        )
    return sorted(rows, key=lambda row: (-int(row["rows"]), str(row[column])))


def safe_metric_recall(frame: pd.DataFrame, pred_column: str) -> float | None:
    positives = frame["actual_is_speculative"].astype(int).eq(1)
    if int(positives.sum()) == 0:
        return None
    preds = frame[pred_column].astype(int if pred_column == "pred_label_tuned" else bool)
    return safe_rate(int((positives & preds.astype(bool)).sum()), int(positives.sum()))


def count_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def build_report(summary: dict[str, Any]) -> str:
    stage1 = summary["stage1_model"]
    stage2_route = summary["stage2_review_route"]
    stage2_review = summary["stage2_committee_review_or_reject"]
    stage2_reject = summary["stage2_committee_reject_only"]
    effect = summary["stage2_effect_counts"]
    lines = [
        "# 2026 External Rating Validation",
        "",
        "## Scope",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Speculative-grade labels: `{summary['positive_count']}`",
        f"- Investment-grade labels: `{summary['negative_count']}`",
        f"- Positive rate: `{summary['positive_rate']:.1%}`",
        f"- Committee mode: `{summary['committee_mode']}`",
        f"- Note: {summary['committee_mode_note']}",
        "",
        "## Overall Metrics",
        "",
        "| View | PR-AUC | ROC-AUC | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        metric_row("Stage 1 model reject", stage1),
        metric_row("Stage 2 review route", stage2_route),
        metric_row("Stage 2 hold/reject as review", stage2_review),
        metric_row("Stage 2 reject only", stage2_reject),
        "",
        "## Review Load",
        "",
        f"- Stage 1 reject count: `{summary['review_load']['stage1_reject_count']}`",
        f"- Stage 2 review route count: `{summary['review_load']['stage2_review_route_count']}`",
        f"- Stage 2 hold/reject count: `{summary['review_load']['stage2_review_or_reject_count']}`",
        f"- Stage 2 reject-only count: `{summary['review_load']['stage2_reject_only_count']}`",
        "",
        "## Stage 2 Effect Counts",
        "",
        *[f"- `{key}`: `{value}`" for key, value in effect.items()],
        "",
        "## Interpretation",
        "",
        "- `Stage 1 model reject`는 XGBoost 43개 공식 모델만 사용한 이진 판단입니다.",
        "- `Stage 2 review route`는 1차 모델 경고 기업을 검토 대상으로 유지하면서, 위원회가 보류/부적격으로 올린 기업도 추가합니다.",
        "- `Stage 2 hold/reject as review`는 조기경보 관점에서 보류와 부적격을 모두 추가 검토 대상으로 봅니다.",
        "- `Stage 2 reject only`는 위원회가 최종 부적격까지 올린 경우만 위험 판단으로 봅니다.",
        "- 현재 평가는 live 외부 API를 사용하지 않은 재현 가능한 offline 평가입니다. 뉴스/웹/OpenDART를 실제로 켠 평가는 별도 실험으로 분리해야 합니다.",
        "",
    ]
    return "\n".join(lines)


def metric_row(label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {label} | {format_optional(metrics.get('pr_auc'))} | "
        f"{format_optional(metrics.get('roc_auc'))} | {metrics['precision']:.4f} | "
        f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['tp']} | "
        f"{metrics['fp']} | {metrics['fn']} | {metrics['tn']} |"
    )


def format_optional(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
