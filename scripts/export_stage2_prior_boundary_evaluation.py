"""Export Stage 2 diagnostics for prior BBB-/BB+ boundary ratings."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = ROOT / "data/outputs/modeling/feature_43_xgboost/diagnostics"
DEFAULT_SCORES = ROOT / "data/outputs/dashboard/feature_43_mvp/prediction_scores.csv"
DEFAULT_PRIOR_REFERENCE = ROOT / "data/evaluation/prior_rating_reference.csv"
DEFAULT_OUTPUT_PREFIX = DIAGNOSTICS_DIR / "stage2_prior_boundary_evaluation"

BOUNDARY_GROUP = "exact_bbb_minus_bb_plus_boundary"
POSITIVE_LABELS = {"1", "true", "투기등급", "부적격", "speculative"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-scores",
        type=Path,
        default=DEFAULT_SCORES,
        help="Stage 1 prediction scores with Stage 2 trigger columns.",
    )
    parser.add_argument(
        "--prior-reference",
        type=Path,
        default=DEFAULT_PRIOR_REFERENCE,
        help="Non-leaky prior rating reference built as of each fiscal-year end.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix for CSV, JSON, and Markdown report files.",
    )
    return parser.parse_args()


def load_boundary_frame(prediction_scores: Path, prior_reference: Path) -> pd.DataFrame:
    """Load model scores and keep rows whose prior rating was exactly BBB- or BB+."""
    if not prediction_scores.exists():
        raise FileNotFoundError(prediction_scores)
    if not prior_reference.exists():
        raise FileNotFoundError(prior_reference)

    scores = pd.read_csv(prediction_scores, encoding="utf-8-sig", dtype={"stock_code": str})
    prior = pd.read_csv(prior_reference, encoding="utf-8-sig", dtype={"stock_code": str})
    prior = prior.loc[prior["universe"].eq("model_v1")].copy()

    required_score_columns = {
        "stock_code",
        "fiscal_year",
        "eval_year",
        "is_speculative",
        "predicted_label",
        "stage2_review_trigger",
        "stage2_secondary_trigger",
        "stage2_overwarning_filter_candidate",
    }
    required_prior_columns = {
        "stock_code",
        "fiscal_year",
        "eval_year",
        "prior_credit_rating",
        "prior_rating_boundary_group",
    }
    _ensure_columns(scores, required_score_columns, prediction_scores)
    _ensure_columns(prior, required_prior_columns, prior_reference)

    merged = scores.merge(
        prior,
        on=["stock_code", "fiscal_year", "eval_year"],
        how="left",
        suffixes=("", "_prior_reference"),
    )
    boundary = merged.loc[
        merged["prior_rating_boundary_group"].eq(BOUNDARY_GROUP)
    ].copy()
    return enrich_boundary_frame(boundary)


def enrich_boundary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized flags and diagnostic labels for prior-boundary evaluation."""
    output = frame.copy()
    output["actual_is_speculative"] = output["is_speculative"].map(_bool_value)
    output["stage1_predicts_risk"] = output.apply(_stage1_risk, axis=1)
    output["stage2_review_trigger_bool"] = output["stage2_review_trigger"].map(_bool_value)
    output["stage2_secondary_trigger_bool"] = output["stage2_secondary_trigger"].map(
        _bool_value
    )
    output["stage2_overwarning_filter_candidate_bool"] = output[
        "stage2_overwarning_filter_candidate"
    ].map(_bool_value)
    output["stage2_cautious_review"] = (
        output["stage2_review_trigger_bool"]
        | output["stage2_overwarning_filter_candidate_bool"]
    )
    output["stage2_risk_signal_proxy"] = (
        output["stage1_predicts_risk"]
        & ~output["stage2_overwarning_filter_candidate_bool"]
    ) | output["stage2_secondary_trigger_bool"]
    output["stage1_error_type"] = output.apply(_stage1_error_type, axis=1)
    output["stage2_boundary_role"] = output.apply(_stage2_boundary_role, axis=1)
    output["actual_label_name"] = output["actual_is_speculative"].map(
        {True: "투기등급", False: "투자적격"}
    )
    output["stage1_label_name"] = output["stage1_predicts_risk"].map(
        {True: "투기등급", False: "투자적격"}
    )
    return output


def overall_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare Stage 1 and Stage 2 prior-boundary signals."""
    actual = frame["actual_is_speculative"]
    rows = [
        _classification_metric_row(
            "1차 모델 위험 판단",
            actual,
            frame["stage1_predicts_risk"],
        ),
        _classification_metric_row(
            "2차 조심검토 게이트",
            actual,
            frame["stage2_cautious_review"],
        ),
        _classification_metric_row(
            "2차 위험신호 근사",
            actual,
            frame["stage2_risk_signal_proxy"],
        ),
    ]
    return pd.DataFrame(rows)


def subset_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Build metrics for all, validation/test, and test-only boundary subsets."""
    subsets = {
        "all_boundary": frame,
        "valid_test_boundary": frame.loc[frame["split"].isin(["valid", "test"])],
        "test_boundary": frame.loc[frame["split"].eq("test")],
    }
    rows: list[dict[str, Any]] = []
    for subset_name, subset in subsets.items():
        if subset.empty:
            continue
        for row in overall_metrics(subset).to_dict(orient="records"):
            row = dict(row)
            row["subset"] = subset_name
            rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    columns = ["subset", *[column for column in output.columns if column != "subset"]]
    return output.loc[:, columns]


def boundary_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize Stage 2 caution behavior by split and prior rating."""
    rows: list[dict[str, Any]] = []
    for (split, prior_rating), group in frame.groupby(["split", "prior_credit_rating"]):
        rows.append(_boundary_summary_row(str(split), str(prior_rating), group))
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    split_order = {"train": 0, "valid": 1, "test": 2}
    rating_order = {"BBB-": 0, "BB+": 1}
    output["_split_order"] = output["split"].map(split_order).fillna(99)
    output["_rating_order"] = output["prior_credit_rating"].map(rating_order).fillna(99)
    return (
        output.sort_values(["_split_order", "_rating_order"])
        .drop(columns=["_split_order", "_rating_order"])
        .reset_index(drop=True)
    )


def rating_transition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize how prior BBB-/BB+ rows moved in the actual label year."""
    rows: list[dict[str, Any]] = []
    for prior_rating, group in frame.groupby("prior_credit_rating"):
        rows.append(_rating_summary_row(str(prior_rating), group))
    for prior_rating, group in frame.loc[frame["split"].isin(["valid", "test"])].groupby(
        "prior_credit_rating"
    ):
        row = _rating_summary_row(str(prior_rating), group)
        row["scope"] = "valid_test"
        rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["scope"] = output["scope"].fillna("all")
    return output.sort_values(["scope", "prior_credit_rating"]).reset_index(drop=True)


def write_outputs(
    *,
    frame: pd.DataFrame,
    output_prefix: Path,
    prediction_scores: Path,
    prior_reference: Path,
) -> dict[str, Path]:
    """Write prior-boundary diagnostics."""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    cases_path = output_prefix.with_name(output_prefix.name + "_cases.csv")
    subset_metrics_path = output_prefix.with_name(output_prefix.name + "_subset_metrics.csv")
    by_group_path = output_prefix.with_name(output_prefix.name + "_by_split_rating.csv")
    transition_path = output_prefix.with_name(output_prefix.name + "_rating_transition.csv")
    summary_path = output_prefix.with_suffix(".json")
    report_path = output_prefix.with_suffix(".md")

    cases = _case_columns(frame)
    subset = subset_metrics(frame)
    by_group = boundary_group_summary(frame)
    transition = rating_transition_summary(frame)

    cases.to_csv(cases_path, index=False, encoding="utf-8-sig")
    subset.to_csv(subset_metrics_path, index=False, encoding="utf-8-sig")
    by_group.to_csv(by_group_path, index=False, encoding="utf-8-sig")
    transition.to_csv(transition_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "boundary_rule": "prior_credit_rating in {BBB-, BB+} as of fiscal_year-12-31",
        "rows": len(frame),
        "valid_test_rows": int(frame["split"].isin(["valid", "test"]).sum()),
        "test_rows": int(frame["split"].eq("test").sum()),
        "prediction_scores_path": str(prediction_scores.relative_to(ROOT)),
        "prior_reference_path": str(prior_reference.relative_to(ROOT)),
        "cases_path": str(cases_path.relative_to(ROOT)),
        "subset_metrics_path": str(subset_metrics_path.relative_to(ROOT)),
        "by_split_rating_path": str(by_group_path.relative_to(ROOT)),
        "rating_transition_path": str(transition_path.relative_to(ROOT)),
        "report_path": str(report_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        build_report(
            frame=frame,
            subset=subset,
            by_group=by_group,
            transition=transition,
            summary=summary,
        ),
        encoding="utf-8",
    )
    return {
        "cases": cases_path,
        "subset_metrics": subset_metrics_path,
        "by_split_rating": by_group_path,
        "rating_transition": transition_path,
        "summary": summary_path,
        "report": report_path,
    }


def build_report(
    *,
    frame: pd.DataFrame,
    subset: pd.DataFrame,
    by_group: pd.DataFrame,
    transition: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    """Build a Markdown report for prior-boundary Stage 2 evaluation."""
    valid_test = frame.loc[frame["split"].isin(["valid", "test"])]
    vt_stage1 = _metric_lookup(subset, "valid_test_boundary", "1차 모델 위험 판단")
    vt_review = _metric_lookup(subset, "valid_test_boundary", "2차 조심검토 게이트")
    vt_risk = _metric_lookup(subset, "valid_test_boundary", "2차 위험신호 근사")

    lines = [
        "# Stage 2 Prior BBB-/BB+ Boundary Evaluation",
        "",
        (
            "이 리포트는 `prior_rating_reference` 기준으로 직전 공개 신용등급이 "
            "`BBB-` 또는 `BB+`였던 기업-연도만 따로 뽑아 Stage 2가 얼마나 "
            "조심스럽게 검토 대상으로 올리는지 점검합니다."
        ),
        "",
        "## 핵심 요약",
        "",
        (
            f"- 전체 prior 경계등급 표본은 {len(frame)}건이고, "
            f"valid/test 구간은 {len(valid_test)}건입니다."
        ),
        (
            f"- valid/test 경계등급에서 1차 모델 Recall은 {vt_stage1['Recall']:.4f}, "
            f"Stage 2 조심검토 게이트 Recall은 {vt_review['Recall']:.4f}입니다."
        ),
        (
            f"- 같은 구간에서 1차 모델 FN은 {int(vt_stage1['FN'])}건이고, "
            f"Stage 2 조심검토 게이트 기준 FN은 {int(vt_review['FN'])}건입니다."
        ),
        (
            f"- 2차 위험신호 근사는 Precision {vt_risk['Precision']:.4f}, "
            f"Recall {vt_risk['Recall']:.4f}로, 보류 전체가 아니라 실제 위험신호로 "
            "볼 만한 케이스를 따로 본 값입니다."
        ),
        (
            "- 따라서 발표에서는 `BBB-/BB+ 경계등급은 확정 분류보다 추가 확인이 필요한 "
            "구간이며, Stage 2는 이 구간의 누락 위험을 줄이는 보수적 검토 장치`라고 "
            "설명하는 것이 가장 안전합니다."
        ),
        "",
        "## Subset Metrics",
        "",
        _markdown_table(subset),
        "",
        "## Split And Prior Rating Breakdown",
        "",
        _markdown_table(by_group),
        "",
        "## Prior Rating Transition",
        "",
        _markdown_table(transition),
        "",
        "## 해석 가이드",
        "",
        "- `1차 모델 위험 판단`은 XGBoost 공식 모델의 원판단입니다.",
        "- `2차 조심검토 게이트`는 Stage 2가 경계기업을 보류/추가 검토 후보로 올리는 넓은 그물입니다.",
        "- `2차 위험신호 근사`는 과민경고 완화 후보를 제외하고, 1차 위험 판단 또는 보조 위험 트리거가 남은 경우입니다.",
        "- 이 평가는 실제 Claude/Agno 호출 결과가 아니라, 전체 표본에 적용 가능한 Stage 2 정책 신호 평가입니다. 실제 위원회 문장 품질은 별도 파일럿 배치로 확인하면 됩니다.",
        "",
        "## Source",
        "",
        f"- Prediction scores: `{summary['prediction_scores_path']}`",
        f"- Prior rating reference: `{summary['prior_reference_path']}`",
        "",
    ]
    return "\n".join(lines)


def _boundary_summary_row(split: str, prior_rating: str, group: pd.DataFrame) -> dict[str, Any]:
    actual = group["actual_is_speculative"]
    stage1 = group["stage1_predicts_risk"]
    review = group["stage2_cautious_review"]
    secondary = group["stage2_secondary_trigger_bool"]
    overwarning = group["stage2_overwarning_filter_candidate_bool"]
    fn = actual & ~stage1
    fp = ~actual & stage1
    return {
        "split": split,
        "prior_credit_rating": prior_rating,
        "n": len(group),
        "actual_speculative": int(actual.sum()),
        "actual_speculative_rate": _safe_div(int(actual.sum()), len(group)),
        "stage1_risk_count": int(stage1.sum()),
        "stage1_risk_rate": _safe_div(int(stage1.sum()), len(group)),
        "stage2_cautious_review_count": int(review.sum()),
        "stage2_cautious_review_rate": _safe_div(int(review.sum()), len(group)),
        "stage2_secondary_trigger_count": int(secondary.sum()),
        "stage2_secondary_trigger_rate": _safe_div(int(secondary.sum()), len(group)),
        "overwarning_soften_candidate_count": int(overwarning.sum()),
        "overwarning_soften_candidate_rate": _safe_div(int(overwarning.sum()), len(group)),
        "stage1_FN": int(fn.sum()),
        "stage1_FP": int(fp.sum()),
        "stage2_FN_caught_by_review": int((fn & review).sum()),
        "stage2_FP_soften_candidate": int((fp & overwarning).sum()),
    }


def _rating_summary_row(prior_rating: str, group: pd.DataFrame) -> dict[str, Any]:
    actual = group["actual_is_speculative"]
    return {
        "scope": "all",
        "prior_credit_rating": prior_rating,
        "n": len(group),
        "actual_speculative": int(actual.sum()),
        "actual_investment": int((~actual).sum()),
        "actual_speculative_rate": _safe_div(int(actual.sum()), len(group)),
        "stage2_cautious_review_rate": _safe_div(
            int(group["stage2_cautious_review"].sum()), len(group)
        ),
        "stage2_overwarning_soften_rate": _safe_div(
            int(group["stage2_overwarning_filter_candidate_bool"].sum()), len(group)
        ),
    }


def _case_columns(frame: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "market",
        "stock_code",
        "corp_name",
        "fiscal_year",
        "eval_year",
        "split",
        "prior_credit_rating",
        "prior_rating_date",
        "prior_rating_age_days",
        "prior_rating_agency",
        "actual_label_name",
        "stage1_label_name",
        "stage1_error_type",
        "prob_speculative",
        "threshold",
        "risk_band",
        "stage2_review_trigger_bool",
        "stage2_secondary_trigger_bool",
        "stage2_overwarning_filter_candidate_bool",
        "stage2_cautious_review",
        "stage2_risk_signal_proxy",
        "stage2_boundary_role",
        "trigger_reason_code",
        "trigger_reason",
        "overwarning_filter_reason_code",
        "overwarning_filter_reason",
        "industry_macro_category",
        "firm_size_group",
    ]
    available = [column for column in wanted if column in frame.columns]
    return frame.loc[:, available].sort_values(
        ["split", "prior_credit_rating", "stage1_error_type", "prob_speculative"],
        ascending=[True, True, True, False],
    )


def _metric_lookup(frame: pd.DataFrame, subset: str, scope: str) -> pd.Series:
    rows = frame.loc[frame["subset"].eq(subset) & frame["metric_scope"].eq(scope)]
    if rows.empty:
        raise KeyError(f"No metric row for subset={subset!r}, scope={scope!r}")
    return rows.iloc[0]


def _classification_metric_row(
    scope: str,
    actual_positive: pd.Series,
    predicted_positive: pd.Series,
) -> dict[str, Any]:
    actual = actual_positive.astype(bool)
    predicted = predicted_positive.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "metric_scope": scope,
        "n": len(actual),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": _safe_div(2 * precision * recall, precision + recall),
        "Accuracy": _safe_div(tp + tn, len(actual)),
        "PositiveRate": _safe_div(int(predicted.sum()), len(actual)),
    }


def _stage1_risk(row: pd.Series) -> bool:
    if "predicted_label" in row and pd.notna(row["predicted_label"]):
        return str(row["predicted_label"]).strip().lower() in POSITIVE_LABELS
    probability = pd.to_numeric(row.get("prob_speculative"), errors="coerce")
    threshold = pd.to_numeric(row.get("threshold"), errors="coerce")
    if pd.notna(probability) and pd.notna(threshold):
        return bool(probability >= threshold)
    return False


def _stage1_error_type(row: pd.Series) -> str:
    actual = bool(row["actual_is_speculative"])
    predicted = bool(row["stage1_predicts_risk"])
    if actual and predicted:
        return "TP"
    if not actual and predicted:
        return "FP"
    if actual and not predicted:
        return "FN"
    return "TN"


def _stage2_boundary_role(row: pd.Series) -> str:
    actual = bool(row["actual_is_speculative"])
    stage1 = bool(row["stage1_predicts_risk"])
    review = bool(row["stage2_cautious_review"])
    overwarning = bool(row["stage2_overwarning_filter_candidate_bool"])
    if actual and not stage1 and review:
        return "missed_risk_caught_by_stage2_review"
    if actual and not review:
        return "risk_not_routed_to_stage2"
    if actual and stage1:
        return "risk_already_flagged_by_stage1"
    if not actual and stage1 and overwarning:
        return "overwarning_soften_candidate"
    if not actual and stage1 and review:
        return "investment_grade_model_warning_kept_for_review"
    if not actual and review:
        return "investment_grade_boundary_review"
    return "investment_grade_not_flagged"


def _ensure_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "투기등급", "부적격"}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    headers = list(display.columns)
    rows = [[_format_cell(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    """Run export."""
    args = parse_args()
    frame = load_boundary_frame(args.prediction_scores, args.prior_reference)
    outputs = write_outputs(
        frame=frame,
        output_prefix=args.output_prefix,
        prediction_scores=args.prediction_scores,
        prior_reference=args.prior_reference,
    )
    print({key: str(value) for key, value in outputs.items()})


if __name__ == "__main__":
    main()
