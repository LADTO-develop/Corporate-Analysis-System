"""Export Stage 2 committee decision-type performance diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = ROOT / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents"
DEFAULT_INPUT = (
    DIAGNOSTICS_DIR
    / "committee_review_balanced_error_risk_30_agno_live_v6"
    / "committee_review_batch_results.csv"
)
DEFAULT_OUTPUT_PREFIX = DIAGNOSTICS_DIR / "stage2_committee_decision_type_performance"

POSITIVE_LABELS = {"투기등급", "부적격", "speculative", "1", "true"}
RISK_DECISION_TYPES = {"위험 보류", "부적격"}
REVIEW_OR_REJECT_LABELS = {"보류", "부적격"}
INVESTMENT_FRIENDLY_TYPES = {"적격", "과민경고 완화 보류"}
AMBIGUOUS_REVIEW_TYPES = {"경계등급 보류", "확인필요 보류"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        default=None,
        help=(
            "Committee batch result CSV. Can be passed multiple times. "
            "Defaults to the latest 30-case Agno v6 pilot."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix for CSV, JSON, and Markdown report files.",
    )
    return parser.parse_args()


def read_results(paths: Iterable[Path]) -> pd.DataFrame:
    """Read and concatenate committee batch result files."""
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"stock_code": str})
        frame = frame.copy()
        frame["source_run"] = path.parent.name
        frame["source_path"] = str(path.relative_to(ROOT))
        frames.append(frame)
    if not frames:
        raise ValueError("No input files were provided.")
    return pd.concat(frames, ignore_index=True, sort=False)


def enrich_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized boolean columns used by performance diagnostics."""
    output = frame.copy()
    output["actual_is_speculative"] = output.apply(_actual_positive, axis=1)
    output["stage1_predicts_risk"] = output.apply(_stage1_predicts_risk, axis=1)
    output["committee_review_or_reject"] = output["final_committee_label"].map(
        lambda value: str(value).strip() in REVIEW_OR_REJECT_LABELS
    )
    output["committee_risk_signal_bool"] = output.apply(_committee_risk_signal, axis=1)
    output["committee_reject_only"] = output["final_committee_label"].map(
        lambda value: str(value).strip() == "부적격"
    )
    output["committee_decision_type_label"] = output["committee_decision_type_label"].fillna(
        output["final_committee_label"]
    )
    return output


def overall_signal_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Build overall Stage 1 and Stage 2 risk-signal metrics."""
    actual = frame["actual_is_speculative"]
    rows = [
        _classification_metric_row("1차 모델", actual, frame["stage1_predicts_risk"]),
        _classification_metric_row(
            "2차 검토대상(보류+부적격)",
            actual,
            frame["committee_review_or_reject"],
        ),
        _classification_metric_row(
            "2차 위험신호(risk_signal)",
            actual,
            frame["committee_risk_signal_bool"],
        ),
        _classification_metric_row("2차 부적격만", actual, frame["committee_reject_only"]),
    ]
    return pd.DataFrame(rows)


def decision_type_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Build one-vs-rest and semantic alignment metrics by committee decision type."""
    actual = frame["actual_is_speculative"]
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("committee_decision_type_label", dropna=False):
        label_text = str(label)
        predicted_this_type = frame["committee_decision_type_label"].astype(str).eq(label_text)
        one_vs_rest = _classification_metric_row(label_text, actual, predicted_this_type)
        actual_speculative = int(group["actual_is_speculative"].sum())
        actual_investment = int((~group["actual_is_speculative"]).sum())
        n = len(group)
        expected_label, alignment_rate = _semantic_alignment(label_text, group)
        rows.append(
            {
                "committee_decision_type_label": label_text,
                "n": n,
                "actual_speculative": actual_speculative,
                "actual_investment": actual_investment,
                "actual_speculative_rate": _safe_div(actual_speculative, n),
                "actual_investment_rate": _safe_div(actual_investment, n),
                "expected_alignment_label": expected_label,
                "expected_alignment_rate": alignment_rate,
                "risk_one_vs_rest_TP": one_vs_rest["TP"],
                "risk_one_vs_rest_FP": one_vs_rest["FP"],
                "risk_one_vs_rest_TN": one_vs_rest["TN"],
                "risk_one_vs_rest_FN": one_vs_rest["FN"],
                "risk_one_vs_rest_precision": one_vs_rest["Precision"],
                "risk_one_vs_rest_recall": one_vs_rest["Recall"],
                "risk_one_vs_rest_f1": one_vs_rest["F1"],
                "category_breakdown": json.dumps(
                    group.get("sample_category", pd.Series(dtype=str))
                    .fillna("unknown")
                    .astype(str)
                    .value_counts()
                    .to_dict(),
                    ensure_ascii=False,
                ),
                "interpretation": _decision_type_interpretation(label_text),
            }
        )
    output = pd.DataFrame(rows)
    type_order = {
        "부적격": 0,
        "위험 보류": 1,
        "경계등급 보류": 2,
        "과민경고 완화 보류": 3,
        "확인필요 보류": 4,
        "적격": 5,
    }
    output["_order"] = output["committee_decision_type_label"].map(type_order).fillna(99)
    output = output.sort_values(["_order", "n"], ascending=[True, False]).drop(columns="_order")
    return output.reset_index(drop=True)


def category_performance(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize committee decision behavior by sampled case category."""
    if "sample_category" not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for category, group in frame.groupby("sample_category", dropna=False):
        actual = group["actual_is_speculative"]
        risk_signal = _classification_metric_row(
            str(category),
            actual,
            group["committee_risk_signal_bool"],
        )
        rows.append(
            {
                "sample_category": str(category),
                "n": len(group),
                "actual_speculative": int(actual.sum()),
                "stage1_risk_count": int(group["stage1_predicts_risk"].sum()),
                "committee_review_or_reject_count": int(group["committee_review_or_reject"].sum()),
                "committee_risk_signal_count": int(group["committee_risk_signal_bool"].sum()),
                "committee_risk_signal_precision": risk_signal["Precision"],
                "committee_risk_signal_recall": risk_signal["Recall"],
                "committee_risk_signal_f1": risk_signal["F1"],
                "decision_type_breakdown": json.dumps(
                    group["committee_decision_type_label"]
                    .fillna("unknown")
                    .astype(str)
                    .value_counts()
                    .to_dict(),
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_category").reset_index(drop=True)


def write_outputs(
    *,
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    by_type: pd.DataFrame,
    by_category: pd.DataFrame,
    output_prefix: Path,
) -> dict[str, Path]:
    """Write CSV, JSON summary, and Markdown report outputs."""
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    overall_path = output_prefix.with_name(output_prefix.name + "_overall_metrics.csv")
    type_path = output_prefix.with_name(output_prefix.name + "_by_type.csv")
    category_path = output_prefix.with_name(output_prefix.name + "_by_category.csv")
    summary_path = output_prefix.with_suffix(".json")
    report_path = output_prefix.with_suffix(".md")

    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    by_type.to_csv(type_path, index=False, encoding="utf-8-sig")
    if not by_category.empty:
        by_category.to_csv(category_path, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rows": len(frame),
        "source_paths": sorted(frame["source_path"].dropna().unique().tolist()),
        "overall_metrics_path": str(overall_path.relative_to(ROOT)),
        "decision_type_metrics_path": str(type_path.relative_to(ROOT)),
        "category_metrics_path": str(category_path.relative_to(ROOT))
        if not by_category.empty
        else None,
        "report_path": str(report_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        build_report(
            frame=frame,
            overall=overall,
            by_type=by_type,
            by_category=by_category,
            summary=summary,
        ),
        encoding="utf-8",
    )
    return {
        "overall": overall_path,
        "by_type": type_path,
        "by_category": category_path,
        "summary": summary_path,
        "report": report_path,
    }


def build_report(
    *,
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    by_type: pd.DataFrame,
    by_category: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    """Build a presentation-ready Markdown report."""
    risk_signal_row = overall.loc[overall["metric_scope"].eq("2차 위험신호(risk_signal)")].iloc[0]
    review_row = overall.loc[overall["metric_scope"].eq("2차 검토대상(보류+부적격)")].iloc[0]
    stage1_row = overall.loc[overall["metric_scope"].eq("1차 모델")].iloc[0]
    lines = [
        "# Stage 2 Committee Decision-Type Performance",
        "",
        "이 리포트는 2차 에이전트 위원회 판단을 유형별로 나누어 실제 투기등급 라벨과 비교합니다.",
        "",
        "## 핵심 요약",
        "",
        (
            f"- 분석 대상은 {len(frame)}건이며, 1차 모델 F1은 "
            f"{stage1_row['F1']:.4f}, 2차 위험신호 F1은 {risk_signal_row['F1']:.4f}입니다."
        ),
        (
            f"- 2차 위험신호 기준 Precision은 {risk_signal_row['Precision']:.4f}, "
            f"Recall은 {risk_signal_row['Recall']:.4f}입니다."
        ),
        (
            f"- 2차 검토대상(보류+부적격)은 Recall {review_row['Recall']:.4f}로 "
            "놓치지 않는 방향의 넓은 그물 역할을 합니다."
        ),
        (
            "- `경계등급 보류`, `과민경고 완화 보류`, `확인필요 보류`는 모두 "
            "투기등급 확정 신호가 아니므로, one-vs-rest 위험 Precision만으로 "
            "좋고 나쁨을 해석하면 안 됩니다."
        ),
        "",
        "## Overall Risk-Signal Metrics",
        "",
        _markdown_table(overall),
        "",
        "## Decision-Type Breakdown",
        "",
        _markdown_table(_report_type_columns(by_type)),
        "",
        "## Category Breakdown",
        "",
        _markdown_table(by_category) if not by_category.empty else "분류 카테고리 컬럼이 없습니다.",
        "",
        "## Interpretation Guide",
        "",
        "- `위험 보류`와 `부적격`은 실제 위험신호로 읽습니다. 여기서는 actual speculative rate가 높을수록 좋습니다.",
        "- `과민경고 완화 보류`와 `적격`은 과도한 위험 경고를 낮추는 방향입니다. 여기서는 actual investment rate가 높을수록 좋습니다.",
        "- `경계등급 보류`와 `확인필요 보류`는 확정 분류라기보다 검토 상태입니다. 실제 위험/정상 혼합이 자연스럽고, 발표에서는 경계·근거부족 케이스를 분리 관리한다는 의미로 설명하는 것이 안전합니다.",
        "",
        "## Source",
        "",
        *[f"- `{path}`" for path in summary["source_paths"]],
        "",
    ]
    return "\n".join(lines)


def _report_type_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "committee_decision_type_label",
        "n",
        "actual_speculative",
        "actual_investment",
        "actual_speculative_rate",
        "actual_investment_rate",
        "expected_alignment_label",
        "expected_alignment_rate",
        "risk_one_vs_rest_precision",
        "risk_one_vs_rest_recall",
        "risk_one_vs_rest_f1",
        "interpretation",
    ]
    return frame.loc[:, columns].copy()


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
    }


def _semantic_alignment(label: str, group: pd.DataFrame) -> tuple[str, float | None]:
    n = len(group)
    if n == 0:
        return "not_applicable", None
    if label in RISK_DECISION_TYPES:
        return "투기등급", _safe_div(int(group["actual_is_speculative"].sum()), n)
    if label in INVESTMENT_FRIENDLY_TYPES:
        return "투자적격", _safe_div(int((~group["actual_is_speculative"]).sum()), n)
    if label in AMBIGUOUS_REVIEW_TYPES:
        return "혼합/검토", None
    return "unknown", None


def _decision_type_interpretation(label: str) -> str:
    descriptions = {
        "부적격": "위원회가 명확한 위험신호로 본 케이스입니다.",
        "위험 보류": "투기등급 가능성을 놓치지 않기 위해 보류로 올린 케이스입니다.",
        "경계등급 보류": "등급·확률 경계에 있어 확정보다 추가 확인을 택한 케이스입니다.",
        "과민경고 완화 보류": "모델 경고를 바로 부적격으로 확정하지 않은 케이스입니다.",
        "확인필요 보류": "근거 부족 또는 판단 충돌 때문에 추가 확인을 남긴 케이스입니다.",
        "적격": "위원회가 추가 위험신호를 강하게 보지 않은 케이스입니다.",
    }
    return descriptions.get(label, "위원회 판단 유형입니다.")


def _actual_positive(row: pd.Series) -> bool:
    if "actual_label_name" in row and pd.notna(row["actual_label_name"]):
        return str(row["actual_label_name"]).strip().lower() in POSITIVE_LABELS
    if "is_speculative" in row and pd.notna(row["is_speculative"]):
        return _bool_value(row["is_speculative"])
    raise KeyError("Input must contain actual_label_name or is_speculative.")


def _stage1_predicts_risk(row: pd.Series) -> bool:
    for column in ["model_predicted_label_name", "graph_model_label"]:
        if column in row and pd.notna(row[column]):
            return str(row[column]).strip().lower() in POSITIVE_LABELS
    probability = pd.to_numeric(row.get("sample_prob_speculative"), errors="coerce")
    threshold = pd.to_numeric(row.get("sample_threshold"), errors="coerce")
    if pd.notna(probability) and pd.notna(threshold):
        return bool(probability >= threshold)
    return False


def _committee_risk_signal(row: pd.Series) -> bool:
    if "committee_risk_signal" in row and pd.notna(row["committee_risk_signal"]):
        return _bool_value(row["committee_risk_signal"])
    label = str(row.get("committee_decision_type_label") or row.get("final_committee_label") or "")
    return label.strip() in RISK_DECISION_TYPES


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "위험신호 있음"}


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
    inputs = args.inputs or [DEFAULT_INPUT]
    frame = enrich_results(read_results(inputs))
    overall = overall_signal_metrics(frame)
    by_type = decision_type_performance(frame)
    by_category = category_performance(frame)
    outputs = write_outputs(
        frame=frame,
        overall=overall,
        by_type=by_type,
        by_category=by_category,
        output_prefix=args.output_prefix,
    )
    print({key: str(value) for key, value in outputs.items()})


if __name__ == "__main__":
    main()
