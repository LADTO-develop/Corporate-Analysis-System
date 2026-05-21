"""Export rolling OOT validation samples for Stage 2 committee tuning.

The generated samples are for Stage 2 prompt/rule tuning only. Each fiscal year
is predicted by a model trained without that year or future years, so these rows
are safer than reusing in-sample train predictions.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from export_committee_review_evaluation_plan import (
    add_rating_segments,
    company_selection_payload,
    normalize_stock_code,
    read_labels,
    success_question,
)
from export_feature_43_candidate_feature_pack_experiments import (
    INPUT_DIR,
    JOIN_KEYS,
    RECALL_FLOOR,
    apply_platt_calibration,
    choose_threshold,
    fit_platt_calibration,
    train_xgboost,
)

ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = INPUT_DIR / "feature_43_master.csv"
TARGET_LABEL_REFERENCE_PATH = ROOT / "data/evaluation/target_label_reference.csv"
OUTPUT_DIR = ROOT / "data/outputs/modeling/feature_43_xgboost/diagnostics"
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
POLICY_COLUMNS = {
    "rolling_stage1_or_near_threshold_0_10": "rolling_committee_review_trigger",
    "rolling_recall_first_mid_mfg_prob_0_10": "rolling_recall_first_review_trigger",
}
SAMPLE_COLUMNS = [
    "evaluation_mode",
    "split",
    "rolling_eval_year",
    "policy_year",
    "committee_policy",
    "sample_category",
    "market",
    "stock_code",
    "corp_name",
    "fiscal_year",
    "eval_year",
    "as_of_date",
    "actual_label_name",
    "model_predicted_label_name",
    "model_error_type",
    "credit_rating",
    "rating_boundary_group",
    "rating_agency_group",
    "industry_macro_category",
    "firm_size_group",
    "prob_speculative",
    "threshold",
    "stage1_probability_margin",
    "trigger_reason_code",
    "trigger_reason",
    "evidence_cutoff_rule",
    "committee_success_question",
    "company_selection_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-path", type=Path, default=MASTER_PATH)
    parser.add_argument("--target-label-reference", type=Path, default=TARGET_LABEL_REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-years", type=int, nargs="+", default=ROLLING_EVAL_YEARS)
    parser.add_argument("--per-category", type=int, default=15)
    return parser.parse_args()


def read_master(path: Path) -> pd.DataFrame:
    master = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    master = master.copy()
    master["stock_code"] = normalize_stock_code(master["stock_code"])
    duplicates = int(master.duplicated(JOIN_KEYS).sum())
    if duplicates:
        raise ValueError(f"feature_43_master has duplicate rows: {duplicates}")
    for column in ["fiscal_year", "eval_year", "is_speculative"]:
        master[column] = pd.to_numeric(master[column], errors="coerce")
    return master


def feature_columns(master: pd.DataFrame) -> list[str]:
    excluded = {*ID_COLUMNS, "label_eval_year", "is_speculative"}
    return [column for column in master.columns if column not in excluded]


def attach_rating_reference(master: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels = read_labels(labels_path)
    label_columns = [
        *JOIN_KEYS,
        "credit_rating",
        "credit_rating_rank",
        "rating_agency_group",
        "rating_target",
        "rating_date",
    ]
    labels = labels.loc[:, [column for column in label_columns if column in labels.columns]]
    labels = labels.drop_duplicates(JOIN_KEYS)
    merged = master.merge(labels, on=JOIN_KEYS, how="left", validate="many_to_one")
    return add_rating_segments(merged)


def rolling_scores(master: pd.DataFrame, eval_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_columns(master)
    score_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for eval_year in eval_years:
        policy_year = eval_year - 1
        train_frame = master.loc[master["fiscal_year"] < policy_year].copy()
        policy_frame = master.loc[master["fiscal_year"] == policy_year].copy()
        eval_frame = master.loc[master["fiscal_year"] == eval_year].copy()
        if train_frame.empty or policy_frame.empty or eval_frame.empty:
            raise ValueError(
                f"Empty rolling split for eval_year={eval_year}: "
                f"train={len(train_frame)}, policy={len(policy_frame)}, eval={len(eval_frame)}"
            )

        model = train_xgboost(
            train_frame.loc[:, features],
            train_frame["is_speculative"].astype(int),
            policy_frame.loc[:, features],
            policy_frame["is_speculative"].astype(int),
        )
        policy_raw = model.predict_proba(policy_frame.loc[:, features])[:, 1]
        eval_raw = model.predict_proba(eval_frame.loc[:, features])[:, 1]
        coef, intercept = fit_platt_calibration(policy_frame["is_speculative"], policy_raw)
        policy_prob = apply_platt_calibration(policy_raw, coef, intercept)
        eval_prob = apply_platt_calibration(eval_raw, coef, intercept)
        threshold, threshold_metrics = choose_threshold(
            policy_frame["is_speculative"].astype(int),
            policy_prob,
            recall_floor=RECALL_FLOOR,
        )

        fold_rows.append(
            {
                "rolling_eval_year": eval_year,
                "policy_year": policy_year,
                "train_year_min": int(train_frame["fiscal_year"].min()),
                "train_year_max": int(train_frame["fiscal_year"].max()),
                "train_rows": len(train_frame),
                "policy_rows": len(policy_frame),
                "eval_rows": len(eval_frame),
                "threshold": threshold,
                "policy_precision": threshold_metrics["precision"],
                "policy_recall": threshold_metrics["recall"],
                "policy_f1": threshold_metrics["f1"],
                "best_iteration": getattr(model, "best_iteration", None),
            }
        )
        score_columns = [
            column
            for column in [*ID_COLUMNS, "label_eval_year", "is_speculative"]
            if column in eval_frame.columns
        ]
        scored = eval_frame.loc[:, score_columns].copy()
        scored["split"] = "rolling_validation"
        scored["rolling_eval_year"] = eval_year
        scored["policy_year"] = policy_year
        scored["prob_speculative_raw"] = eval_raw
        scored["prob_speculative"] = eval_prob
        scored["threshold"] = threshold
        scored["pred_label_tuned"] = (eval_prob >= threshold).astype(int)
        score_frames.append(scored)
    return pd.concat(score_frames, ignore_index=True), pd.DataFrame(fold_rows)


def add_sample_policy_flags(scores: pd.DataFrame) -> pd.DataFrame:
    output = scores.copy()
    base = output["pred_label_tuned"].astype(int).eq(1)
    actual = output["is_speculative"].astype(int).eq(1)
    near_threshold = output["prob_speculative"].sub(output["threshold"]).abs().le(0.10)
    mid_mfg = output["industry_macro_category"].astype(str).eq("manufacturing") & output[
        "firm_size_group"
    ].astype(str).eq("mid_sized")
    output["rolling_committee_review_trigger"] = base | near_threshold
    output["rolling_recall_first_review_trigger"] = (
        base | near_threshold | ((~base) & mid_mfg & output["prob_speculative"].ge(0.10))
    )
    output["stage1_probability_margin"] = output["prob_speculative"] - output["threshold"]
    output["actual_label_name"] = np.where(actual, "투기등급", "투자적격")
    output["model_predicted_label_name"] = np.where(base, "투기등급", "투자적격")
    output["model_error_type"] = np.select(
        [
            base & actual,
            base & ~actual,
            (~base) & actual,
            (~base) & ~actual,
        ],
        ["true_positive", "false_positive", "false_negative", "true_negative"],
        default="unknown",
    )
    output["as_of_date"] = output["fiscal_year"].astype(int).astype(str) + "-12-31"
    return output


def build_samples(scores: pd.DataFrame, per_category: int) -> pd.DataFrame:
    frame = add_sample_policy_flags(scores)
    sample_frames: list[pd.DataFrame] = []
    for policy_name, trigger_column in POLICY_COLUMNS.items():
        triggered = frame[trigger_column].astype(bool)
        base = frame["pred_label_tuned"].astype(int).eq(1)
        actual = frame["is_speculative"].astype(int).eq(1)
        near_threshold = frame["prob_speculative"].sub(frame["threshold"]).abs().le(0.10)
        category_masks = {
            "fn_caught_by_stage2_review": triggered & (~base) & actual,
            "fp_needing_committee_mitigation": triggered & base & (~actual),
            "bbb_minus_bb_plus_boundary": triggered
            & frame["is_exact_boundary_bbb_minus_bb_plus"].astype(bool),
            "true_positive_risk_explanation": triggered & base & actual,
            "true_negative_overescalation_guardrail": triggered & (~base) & (~actual) & near_threshold,
        }
        sort_orders = {
            "fn_caught_by_stage2_review": ("prob_speculative", False),
            "fp_needing_committee_mitigation": ("stage1_probability_margin", True),
            "bbb_minus_bb_plus_boundary": ("stage1_probability_margin", False),
            "true_positive_risk_explanation": ("prob_speculative", False),
            "true_negative_overescalation_guardrail": ("stage1_probability_margin", False),
        }
        for category, mask in category_masks.items():
            sort_column, ascending = sort_orders[category]
            subset = frame.loc[mask].copy().sort_values(
                [sort_column, "rolling_eval_year", "stock_code"],
                ascending=[ascending, True, True],
            )
            subset = subset.head(per_category)
            subset["evaluation_mode"] = "rolling_validation_tuning"
            subset["committee_policy"] = policy_name
            subset["sample_category"] = category
            subset["trigger_reason_code"] = np.where(
                subset["pred_label_tuned"].astype(int).eq(1),
                "rolling_stage1_risk",
                "rolling_near_threshold_or_segment_review",
            )
            subset["trigger_reason"] = np.where(
                subset["pred_label_tuned"].astype(int).eq(1),
                "rolling OOT 43개 모델이 위험 기준선을 넘어 위원회 검토 대상으로 올렸습니다.",
                "rolling OOT 43개 모델은 투자적격이지만 기준선 근처 또는 취약 세그먼트여서 추가 검토 대상으로 올렸습니다.",
            )
            sample_frames.append(subset)
    if not sample_frames:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    samples = pd.concat(sample_frames, ignore_index=True)
    samples = samples.drop_duplicates(
        ["committee_policy", "market", "stock_code", "fiscal_year", "eval_year"],
        keep="first",
    )
    samples["evidence_cutoff_rule"] = (
        "Use only evidence published on or before fiscal_year-12-31. "
        "For rolling validation, undated web evidence is excluded."
    )
    samples["committee_success_question"] = samples.apply(success_question, axis=1)
    samples["company_selection_json"] = samples.apply(company_selection_payload, axis=1)
    return samples.loc[:, [column for column in SAMPLE_COLUMNS if column in samples.columns]]


def summarize(scores: pd.DataFrame, samples: pd.DataFrame, folds: pd.DataFrame) -> dict[str, Any]:
    case_counts = (
        scores.groupby(["rolling_eval_year", "model_error_type"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_dict("records")
    )
    sample_counts = (
        samples.groupby(["committee_policy", "sample_category"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_dict("records")
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_mode": "rolling_validation_tuning",
        "eval_years": [int(year) for year in sorted(scores["rolling_eval_year"].unique())],
        "score_rows": len(scores),
        "sample_rows": len(samples),
        "folds": folds.to_dict("records"),
        "case_counts": case_counts,
        "sample_counts": sample_counts,
        "split_usage_policy": (
            "Use rolling validation for Stage 2 tuning; keep fixed test holdout and "
            "2026 external labels for final confirmation only."
        ),
    }


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    preview = frame.head(max_rows).copy()
    columns = [str(column) for column in preview.columns]
    rows = preview.astype(object).where(pd.notna(preview), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(value.replace("|", "/") for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_report(scores: pd.DataFrame, samples: pd.DataFrame, summary: dict[str, Any]) -> str:
    case_counts = pd.DataFrame(summary["case_counts"])
    sample_counts = pd.DataFrame(summary["sample_counts"])
    preview_columns = [
        "committee_policy",
        "sample_category",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "actual_label_name",
        "model_predicted_label_name",
        "credit_rating",
        "prob_speculative",
    ]
    return "\n".join(
        [
            "# Stage 2 Rolling Validation Tuning Samples",
            "",
            "rolling OOT 예측값을 기준으로 Stage 2 에이전트 튜닝 샘플을 구성했습니다.",
            "",
            "## 원칙",
            "",
            "- 각 rolling_eval_year는 그 이전 데이터만 사용한 모델로 예측합니다.",
            "- 이 파일은 에이전트 규칙/프롬프트 개선용 validation pool입니다.",
            "- test holdout과 2026 외부검증 라벨은 튜닝에 사용하지 않습니다.",
            "",
            "## Fold Summary",
            "",
            markdown_table(pd.DataFrame(summary["folds"])),
            "",
            "## Case Counts",
            "",
            markdown_table(case_counts),
            "",
            "## Sample Counts",
            "",
            markdown_table(sample_counts),
            "",
            "## Sample Preview",
            "",
            markdown_table(samples.loc[:, [c for c in preview_columns if c in samples.columns]]),
            "",
            f"Total rolling score rows: {len(scores)}",
            f"Total tuning sample rows: {len(samples)}",
            "",
        ]
    )


def write_outputs(
    *,
    scores: pd.DataFrame,
    samples: pd.DataFrame,
    folds: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "stage2_rolling_validation_scores.csv"
    samples_path = output_dir / "committee_review_rolling_validation_tuning_samples.csv"
    summary_path = output_dir / "stage2_rolling_validation_summary.json"
    report_path = output_dir / "stage2_rolling_validation_report.md"
    summary = summarize(scores, samples, folds)
    summary["paths"] = {
        "scores": str(scores_path.relative_to(ROOT)),
        "samples": str(samples_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(scores, samples, summary), encoding="utf-8")
    print(f"[Saved] {scores_path}")
    print(f"[Saved] {samples_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {report_path}")


def main() -> None:
    args = parse_args()
    master = read_master(args.master_path)
    scores, folds = rolling_scores(master, args.eval_years)
    scores = attach_rating_reference(scores, args.target_label_reference)
    scores = add_sample_policy_flags(scores)
    samples = build_samples(scores, args.per_category)
    write_outputs(scores=scores, samples=samples, folds=folds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
