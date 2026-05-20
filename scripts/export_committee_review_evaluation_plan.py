"""Prepare leakage-safe committee review evaluation samples."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_SCORES_PATH = ROOT / "data/outputs/dashboard/feature_43_mvp/prediction_scores.csv"
TARGET_LABEL_REFERENCE_PATH = ROOT / "data/evaluation/target_label_reference.csv"
LABELS_2026_PATH = ROOT / "data/evaluation/credit_rating_labels_2026.csv"
INFERENCE_2026_PATH = ROOT / "data/input/credit_43_features/feature_43_inference_2026.csv"
OUTPUT_DIR = ROOT / "data/outputs/modeling/feature_43_xgboost/diagnostics"

KEY_COLUMNS = ["market", "stock_code", "corp_name", "fiscal_year", "eval_year"]

RATING_RANK = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-scores", type=Path, default=PREDICTION_SCORES_PATH)
    parser.add_argument("--target-label-reference", type=Path, default=TARGET_LABEL_REFERENCE_PATH)
    parser.add_argument("--labels-2026", type=Path, default=LABELS_2026_PATH)
    parser.add_argument("--inference-2026", type=Path, default=INFERENCE_2026_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--per-category", type=int, default=15)
    return parser.parse_args()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(6)
    )


def read_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    scores = scores.copy()
    scores["stock_code"] = normalize_stock_code(scores["stock_code"])
    for column in [
        "fiscal_year",
        "eval_year",
        "is_speculative",
        "prob_speculative",
        "threshold",
        "prob_speculative_45",
        "threshold_45",
        "threshold_45_it_services_review",
    ]:
        if column in scores.columns:
            scores[column] = pd.to_numeric(scores[column], errors="coerce")
    return scores


def read_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
    labels = labels.copy()
    labels["stock_code"] = normalize_stock_code(labels["stock_code"])
    for column in ["fiscal_year", "eval_year", "is_speculative", "credit_rating_rank"]:
        if column in labels.columns:
            labels[column] = pd.to_numeric(labels[column], errors="coerce")
    if "credit_rating_rank" not in labels.columns and "credit_rating" in labels.columns:
        labels["credit_rating_rank"] = labels["credit_rating"].map(RATING_RANK)
    keep_columns = [
        *KEY_COLUMNS,
        "is_speculative",
        "credit_rating",
        "credit_rating_rank",
        "rating_agency",
        "rating_agency_group",
        "rating_agency_code",
        "rating_target",
        "rating_date",
        "selection_scope",
        "source_label_set",
    ]
    return labels.loc[:, [column for column in keep_columns if column in labels.columns]]


def attach_rating_reference(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.drop(columns=["is_speculative"], errors="ignore")
    score_columns_to_drop = [
        column
        for column in [
            "credit_rating",
            "credit_rating_rank",
            "rating_agency",
            "rating_agency_group",
            "rating_target",
            "rating_date",
            "selection_scope",
            "source_label_set",
        ]
        if column in scores.columns
    ]
    scores = scores.drop(columns=score_columns_to_drop)
    merged = scores.merge(labels, on=KEY_COLUMNS, how="left", validate="many_to_one")
    return add_rating_segments(merged)


def add_rating_segments(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    rank = pd.to_numeric(output.get("credit_rating_rank"), errors="coerce")
    rating = output.get("credit_rating", pd.Series(index=output.index, dtype="string"))
    rating = rating.astype("string").str.strip()
    output["is_exact_boundary_bbb_minus_bb_plus"] = rating.isin(["BBB-", "BB+"])
    output["rating_boundary_group"] = np.select(
        [
            rank.le(7),
            rank.between(8, 10, inclusive="both"),
            rank.between(11, 13, inclusive="both"),
            rank.ge(14),
        ],
        [
            "upper_investment_A_or_above",
            "near_investment_BBB_plus_to_BBB_minus",
            "near_speculative_BB_plus_to_BB_minus",
            "deep_speculative_B_plus_or_lower",
        ],
        default="missing_rating",
    )
    return output


def stage1_risk(frame: pd.DataFrame) -> pd.Series:
    if "pred_label_tuned" in frame.columns:
        return pd.to_numeric(frame["pred_label_tuned"], errors="coerce").fillna(0).astype(int).eq(1)
    return frame["prob_speculative"].ge(frame["threshold"])


def secondary_45_risk(frame: pd.DataFrame, base_risk: pd.Series) -> pd.Series:
    if "stage2_secondary_trigger" in frame.columns:
        return frame["stage2_secondary_trigger"].astype(str).str.lower().isin({"true", "1"})
    if not {"prob_speculative_45", "threshold_45"}.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    feature45_risk = frame["prob_speculative_45"].ge(frame["threshold_45"])
    it_services_review = pd.Series(False, index=frame.index)
    if "threshold_45_it_services_review" in frame.columns:
        it_services_review = frame["industry_macro_category"].astype(str).eq("it_services") & frame[
            "prob_speculative_45"
        ].ge(frame["threshold_45_it_services_review"])
    return (~base_risk) & (feature45_risk | it_services_review)


def add_policy_flags(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    base = stage1_risk(output)
    secondary = secondary_45_risk(output, base)
    near_threshold = output["prob_speculative"].sub(output["threshold"]).abs().le(0.10)
    mid_mfg = output["industry_macro_category"].astype(str).eq("manufacturing") & output[
        "firm_size_group"
    ].astype(str).eq("mid_sized")
    output["stage1_review_trigger"] = base
    output["current_45_secondary_radar_trigger"] = base | secondary
    output["balanced_committee_review_trigger"] = base | secondary | ((~base) & near_threshold)
    output["recall_first_committee_review_trigger"] = (
        base | secondary | ((~base) & mid_mfg & output["prob_speculative"].ge(0.10))
    )
    output["stage1_probability_margin"] = output["prob_speculative"] - output["threshold"]
    output["model_predicted_label_name"] = np.where(base, "투기등급", "투자적격")
    output["actual_label_name"] = np.where(
        pd.to_numeric(output["is_speculative"], errors="coerce").fillna(0).astype(int).eq(1),
        "투기등급",
        "투자적격",
    )
    output["model_error_type"] = np.select(
        [
            base & output["is_speculative"].astype(int).eq(1),
            base & output["is_speculative"].astype(int).eq(0),
            (~base) & output["is_speculative"].astype(int).eq(1),
            (~base) & output["is_speculative"].astype(int).eq(0),
        ],
        ["true_positive", "false_positive", "false_negative", "true_negative"],
        default="unknown",
    )
    return output


def historical_as_of_date(frame: pd.DataFrame) -> pd.Series:
    return frame["fiscal_year"].astype(int).astype(str) + "-12-31"


def company_selection_payload(row: pd.Series) -> str:
    payload = {
        "request_id": (
            f"committee-eval-{row['market']}-{str(row['stock_code']).zfill(6)}-"
            f"{int(row['fiscal_year'])}"
        ),
        "source": "csv_upload",
        "selected_at": datetime.now(UTC).isoformat(),
        "as_of_date": str(row["as_of_date"]),
        "company": {
            "market": str(row["market"]),
            "stock_code": str(row["stock_code"]).zfill(6),
            "corp_name": str(row["corp_name"]),
        },
        "analysis": {
            "fiscal_year": int(row["fiscal_year"]),
            "eval_year": int(row["eval_year"]),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def success_question(row: pd.Series) -> str:
    error_type = str(row.get("model_error_type", ""))
    if error_type == "false_negative":
        return "실제 투기등급을 1차 모델이 놓친 사례입니다. 위원회가 보류/부적격으로 끌어올리는지 확인합니다."
    if error_type == "false_positive":
        return "실제 투자적격을 1차 모델이 과민 경고한 사례입니다. 위원회가 적격/보류로 완화하는지 확인합니다."
    if error_type == "true_positive":
        return "1차 모델이 맞춘 위험 사례입니다. 위원회가 외부근거로 위험 설명을 보강하는지 확인합니다."
    return "1차 모델이 안정으로 본 사례입니다. 위원회가 근거 없이 과도하게 위험을 키우지 않는지 확인합니다."


def build_historical_samples(scores: pd.DataFrame, per_category: int) -> pd.DataFrame:
    test = scores.loc[scores["split"].astype(str).eq("test")].copy()
    test = add_policy_flags(test)
    test["as_of_date"] = historical_as_of_date(test)
    sample_frames: list[pd.DataFrame] = []
    policy_columns = {
        "balanced_current_45_or_near_threshold_0_10": "balanced_committee_review_trigger",
        "recall_first_current_45_or_fn_mid_mfg_prob_0_10": "recall_first_committee_review_trigger",
    }
    for policy_name, trigger_column in policy_columns.items():
        triggered = test[trigger_column].astype(bool)
        category_masks = {
            "fn_caught_by_stage2_review": (
                triggered & test["stage1_review_trigger"].eq(False) & test["is_speculative"].eq(1)
            ),
            "fp_needing_committee_mitigation": (
                triggered & test["stage1_review_trigger"].eq(True) & test["is_speculative"].eq(0)
            ),
            "bbb_minus_bb_plus_boundary": (
                triggered & test["is_exact_boundary_bbb_minus_bb_plus"].astype(bool)
            ),
            "true_positive_risk_explanation": (
                triggered & test["stage1_review_trigger"].eq(True) & test["is_speculative"].eq(1)
            ),
        }
        sort_orders = {
            "fn_caught_by_stage2_review": ("prob_speculative", False),
            "fp_needing_committee_mitigation": ("stage1_probability_margin", True),
            "bbb_minus_bb_plus_boundary": ("stage1_probability_margin", False),
            "true_positive_risk_explanation": ("prob_speculative", False),
        }
        for category, mask in category_masks.items():
            sort_column, ascending = sort_orders[category]
            subset = test.loc[mask].copy().sort_values(sort_column, ascending=ascending)
            subset = subset.head(per_category)
            subset["evaluation_mode"] = "historical_test_replay"
            subset["committee_policy"] = policy_name
            subset["sample_category"] = category
            sample_frames.append(subset)

    if not sample_frames:
        return pd.DataFrame()
    samples = pd.concat(sample_frames, ignore_index=True)
    samples = samples.drop_duplicates(
        ["committee_policy", "market", "stock_code", "fiscal_year", "eval_year"],
        keep="first",
    )
    samples["evidence_cutoff_rule"] = (
        "Use only evidence published on or before fiscal_year-12-31. "
        "For historical replay, undated web evidence is excluded."
    )
    samples["committee_success_question"] = samples.apply(success_question, axis=1)
    samples["company_selection_json"] = samples.apply(company_selection_payload, axis=1)
    columns = [
        "evaluation_mode",
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
        "prob_speculative_45",
        "threshold_45",
        "trigger_reason_code",
        "trigger_reason",
        "evidence_cutoff_rule",
        "committee_success_question",
        "company_selection_json",
    ]
    return samples.loc[:, [column for column in columns if column in samples.columns]]


def build_2026_candidates(labels_path: Path, inference_path: Path) -> pd.DataFrame:
    if not labels_path.exists() or not inference_path.exists():
        return pd.DataFrame()
    labels = read_labels(labels_path)
    inference = pd.read_csv(inference_path, encoding="utf-8-sig", dtype={"stock_code": str})
    inference = inference.copy()
    inference["stock_code"] = normalize_stock_code(inference["stock_code"])
    for column in ["fiscal_year", "eval_year"]:
        inference[column] = pd.to_numeric(inference[column], errors="coerce")
    feature_columns = [
        "industry_macro_category",
        "firm_size_group",
        "market_to_book",
        "current_ratio",
        "interest_coverage_ratio",
        "debt_ratio",
    ]
    merged = labels.merge(
        inference.loc[
            :, [column for column in [*KEY_COLUMNS, *feature_columns] if column in inference]
        ],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    merged = add_rating_segments(merged)
    merged["evaluation_mode"] = "current_2026_external_validation"
    merged["as_of_date"] = date.today().isoformat()
    merged["actual_label_name"] = np.where(
        merged["is_speculative"].astype(int).eq(1), "투기등급", "투자적격"
    )
    merged["model_score_status"] = np.where(
        merged["industry_macro_category"].notna(),
        "feature_row_ready_score_not_exported",
        "missing_2026_feature_row",
    )
    merged["evidence_cutoff_rule"] = (
        "Current/2026 external validation may use evidence available as of the run date."
    )
    columns = [
        "evaluation_mode",
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "as_of_date",
        "actual_label_name",
        "credit_rating",
        "rating_boundary_group",
        "rating_agency",
        "rating_agency_code",
        "rating_target",
        "rating_date",
        "industry_macro_category",
        "firm_size_group",
        "model_score_status",
        "evidence_cutoff_rule",
    ]
    return merged.loc[:, [column for column in columns if column in merged.columns]]


def summarize_samples(historical: pd.DataFrame, current_2026: pd.DataFrame) -> dict[str, object]:
    historical_counts = (
        historical.groupby(["committee_policy", "sample_category"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_dict(orient="records")
        if not historical.empty
        else []
    )
    current_counts = (
        current_2026.groupby(["model_score_status", "actual_label_name"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_dict(orient="records")
        if not current_2026.empty
        else []
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "historical_test_replay_rows": len(historical),
        "current_2026_external_validation_rows": len(current_2026),
        "historical_counts": historical_counts,
        "current_2026_counts": current_counts,
        "leakage_guardrail": {
            "historical_as_of_date": "fiscal_year-12-31",
            "naver_tavily_filter": "published_at must be <= as_of_date; undated web results are excluded in historical mode",
            "opendart_filter": "query window ends at as_of_date",
            "current_2026_mode": "uses evidence available as of run date",
        },
    }


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    preview = frame.head(max_rows).copy()
    columns = [str(column) for column in preview.columns]
    rows = preview.astype(object).where(pd.notna(preview), "").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(value.replace("|", "/") for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_report(
    historical: pd.DataFrame, current_2026: pd.DataFrame, summary: dict[str, object]
) -> str:
    historical_counts = pd.DataFrame(summary["historical_counts"])
    current_counts = pd.DataFrame(summary["current_2026_counts"])
    return "\n".join(
        [
            "# Committee Review Evaluation Plan",
            "",
            "Stage 2 위원회가 모델 판단을 얼마나 보완하는지 평가하기 위한 샘플과 실행 기준입니다.",
            "",
            "## 1. Historical Test Replay",
            "",
            "- 목적: 과거 test 구간에서 위원회가 FN을 보류/부적격으로 끌어올리고, FP를 적격/보류로 완화하는지 확인합니다.",
            "- 기준일: 각 행의 `as_of_date = fiscal_year-12-31`입니다.",
            "- 누수 방지: Naver/Tavily는 기준일 이후 결과를 제외하고, 과거 모드에서는 날짜 없는 웹 결과도 제외합니다. OpenDART는 조회 종료일을 기준일로 고정합니다.",
            "",
            "### Historical Sample Counts",
            "",
            markdown_table(historical_counts),
            "",
            "### Historical Sample Preview",
            "",
            markdown_table(
                historical[
                    [
                        "committee_policy",
                        "sample_category",
                        "corp_name",
                        "fiscal_year",
                        "eval_year",
                        "as_of_date",
                        "actual_label_name",
                        "model_predicted_label_name",
                        "credit_rating",
                        "prob_speculative",
                    ]
                ]
                if not historical.empty
                else historical
            ),
            "",
            "## 2. Current/2026 External Validation",
            "",
            "- 목적: 2026 inference 기업을 현재 시점에서 실제 외부 검증 정답셋과 비교할 준비를 합니다.",
            "- 기준일: 실행일 기준 현재 사용 가능한 뉴스/공시를 사용할 수 있습니다.",
            "- 다음 단계: 2026 prediction score를 export한 뒤 `model_score_status=feature_row_ready_score_not_exported` 행과 결합해 위원회 판단을 비교합니다.",
            "",
            "### 2026 Candidate Counts",
            "",
            markdown_table(current_counts),
            "",
            "## Evaluation Questions",
            "",
            "- FN 보완: 실제 투기등급인데 1차 모델이 투자적격으로 본 기업을 위원회가 보류/부적격으로 끌어올리는가?",
            "- FP 완화: 실제 투자적격인데 1차 모델이 위험하다고 본 기업을 위원회가 적격/보류로 완화하는가?",
            "- 근거 신뢰도: veto나 숨은 꼬리위험 판단이 실제 기업 직접 근거에 기반하는가?",
            "- 발표 표현: 과거 test 재현 평가는 look-ahead bias를 막기 위해 기준일 이전 공개 정보만 사용합니다.",
            "",
        ]
    )


def write_outputs(
    *,
    historical: pd.DataFrame,
    current_2026: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    historical_path = output_dir / "committee_review_historical_test_replay_samples.csv"
    current_2026_path = output_dir / "committee_review_2026_external_validation_candidates.csv"
    summary_path = output_dir / "committee_review_evaluation_summary.json"
    report_path = output_dir / "committee_review_evaluation_plan.md"
    summary = summarize_samples(historical, current_2026)
    summary["paths"] = {
        "historical_samples": str(historical_path.relative_to(ROOT)),
        "current_2026_candidates": str(current_2026_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }

    historical.to_csv(historical_path, index=False, encoding="utf-8-sig")
    current_2026.to_csv(current_2026_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report(historical, current_2026, summary), encoding="utf-8")
    print(f"[Saved] {historical_path}")
    print(f"[Saved] {current_2026_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {report_path}")


def main() -> None:
    args = parse_args()
    scores = read_scores(args.prediction_scores)
    labels = read_labels(args.target_label_reference)
    scores = attach_rating_reference(scores, labels)
    historical = build_historical_samples(scores, args.per_category)
    current_2026 = build_2026_candidates(args.labels_2026, args.inference_2026)
    write_outputs(historical=historical, current_2026=current_2026, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
