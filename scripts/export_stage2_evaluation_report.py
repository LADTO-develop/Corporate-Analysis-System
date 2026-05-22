"""Export a consolidated Stage 2 evaluation report.

This script does not call live LLM or external-evidence APIs.  It collects the
existing Stage 2 diagnostics, recomputes key comparison tables, and writes a
presentation-ready Markdown report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = (
    ROOT / "data/outputs/modeling/feature_43_xgboost/diagnostics/stage2_agents"
)
DEFAULT_OUTPUT_PREFIX = DIAGNOSTICS_DIR / "stage2_evaluation_report"

AGNO_HOLD_METRICS = "stage2_agent_agno_hold_subtype_metrics.csv"
ERROR_RISK_METRICS = "stage2_agent_error_risk_10_agno_metrics.csv"
EXPERIMENT_LOG = "stage2_agent_performance_experiment_log.csv"
SPEED_LOG = "stage2_agent_speed_experiment_log.csv"
RECOMPUTED_SUMMARY = "stage2_agent_all_pilots_recomputed_summary.csv"
LATEST_BATCH_RESULTS = "committee_review_batch_results.csv"
VALIDATION_TEST_POLICY_METRICS = "stage2_validation_test_policy_metrics.csv"
TRACE_GATE_CONTRIBUTION = "stage2_validation_test_trace_gate_contribution.csv"
OPENAI_AGNO_COMPARISON_DETAILS = "stage2_openai_agno_explanation_comparison_details.csv"

POSITIVE_LABELS = {"투기등급", "부적격", "speculative", "1", "true"}
REVIEW_OR_REJECT_LABELS = {"보류", "부적격"}
RISK_SIGNAL_TYPES = {"위험 보류", "부적격"}


@dataclass(frozen=True, slots=True)
class SourceStatus:
    """File-read status used in the generated evaluation report."""

    name: str
    path: Path
    exists: bool
    rows: int
    columns: int
    modified_at: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=DIAGNOSTICS_DIR,
        help="Directory containing Stage 2 diagnostic CSV/JSON/Markdown files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix. The script writes .md, .json, and *_combined_metrics.csv.",
    )
    return parser.parse_args()


def read_source(diagnostics_dir: Path, filename: str) -> tuple[pd.DataFrame, SourceStatus]:
    """Read one optional CSV diagnostics file and return its status."""
    path = diagnostics_dir / filename
    if not path.exists():
        return (
            pd.DataFrame(),
            SourceStatus(
                name=filename,
                path=path,
                exists=False,
                rows=0,
                columns=0,
                modified_at=None,
            ),
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(timespec="seconds")
    return (
        frame,
        SourceStatus(
            name=filename,
            path=path,
            exists=True,
            rows=len(frame),
            columns=len(frame.columns),
            modified_at=modified_at.replace("+00:00", "Z"),
        ),
    )


def normalize_metrics(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine Stage 2 metric tables into one consistent schema."""
    outputs: list[pd.DataFrame] = []
    for frame in frames:
        if frame.empty:
            continue
        output = frame.copy()
        rename_map = {
            "tp": "TP",
            "fp": "FP",
            "tn": "TN",
            "fn": "FN",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "accuracy": "Accuracy",
        }
        output = output.rename(columns=rename_map)
        if "run" not in output.columns:
            output["run"] = "unknown"
        if "target" not in output.columns:
            output["target"] = "unknown"
        metric_columns = [
            "run",
            "target",
            "n",
            "TP",
            "FP",
            "TN",
            "FN",
            "Precision",
            "Recall",
            "F1",
            "Accuracy",
        ]
        for column in metric_columns:
            if column not in output.columns:
                output[column] = pd.NA
        outputs.append(output.loc[:, metric_columns])
    if not outputs:
        return pd.DataFrame(
            columns=[
                "run",
                "target",
                "n",
                "TP",
                "FP",
                "TN",
                "FN",
                "Precision",
                "Recall",
                "F1",
                "Accuracy",
            ]
        )
    combined = pd.concat(outputs, ignore_index=True, sort=False)
    for column in ["n", "TP", "FP", "TN", "FN"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce").astype("Int64")
    for column in ["Precision", "Recall", "F1", "Accuracy"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def summarize_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create one compact comparison row per Stage 2 evaluation run."""
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for run, group in metrics.groupby("run", dropna=False):
        stage1 = _metric_row(group, "1차 모델")
        review = _metric_row(group, "2차 검토대상(보류+부적격)")
        risk = _metric_row(group, "2차 위험신호(risk_signal)")
        reject = _metric_row(group, "2차 부적격만")
        rows.append(
            {
                "run": str(run),
                "n": _first_numeric(group.get("n")),
                "stage1_f1": _value(stage1, "F1"),
                "review_recall": _value(review, "Recall"),
                "risk_precision": _value(risk, "Precision"),
                "risk_recall": _value(risk, "Recall"),
                "risk_f1": _value(risk, "F1"),
                "reject_precision": _value(reject, "Precision"),
                "reject_recall": _value(reject, "Recall"),
                "risk_f1_delta_vs_stage1": _delta(_value(risk, "F1"), _value(stage1, "F1")),
                "review_recall_delta_vs_stage1": _delta(
                    _value(review, "Recall"),
                    _value(stage1, "Recall"),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("run").reset_index(drop=True)


def latest_batch_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute Stage 1 and Stage 2 classification metrics for latest batch output."""
    if frame.empty:
        return pd.DataFrame()
    enriched = frame.copy()
    enriched["actual_is_speculative"] = enriched.apply(_actual_positive, axis=1)
    enriched["stage1_predicts_risk"] = enriched.apply(_stage1_predicts_risk, axis=1)
    enriched["committee_review_or_reject"] = enriched["final_committee_label"].map(
        lambda value: str(value).strip() in REVIEW_OR_REJECT_LABELS
    )
    if "committee_risk_signal" in enriched.columns:
        enriched["committee_risk_signal_bool"] = enriched["committee_risk_signal"].map(_bool_value)
        risk_signal_label = "2차 위험신호(risk_signal)"
    elif "committee_decision_type_label" in enriched.columns:
        enriched["committee_risk_signal_bool"] = enriched["committee_decision_type_label"].map(
            lambda value: str(value).strip() in RISK_SIGNAL_TYPES
        )
        risk_signal_label = "2차 위험신호(risk_signal)"
    else:
        enriched["committee_risk_signal_bool"] = enriched["committee_review_or_reject"]
        risk_signal_label = "2차 위험신호(risk_signal 미제공; 보류+부적격 대체)"
    enriched["committee_reject_only"] = enriched["final_committee_label"].map(
        lambda value: str(value).strip() == "부적격"
    )
    actual = enriched["actual_is_speculative"]
    return pd.DataFrame(
        [
            _classification_metric_row("1차 모델", actual, enriched["stage1_predicts_risk"]),
            _classification_metric_row(
                "2차 검토대상(보류+부적격)",
                actual,
                enriched["committee_review_or_reject"],
            ),
            _classification_metric_row(
                risk_signal_label,
                actual,
                enriched["committee_risk_signal_bool"],
            ),
            _classification_metric_row("2차 부적격만", actual, enriched["committee_reject_only"]),
        ]
    )


def write_outputs(
    *,
    output_prefix: Path,
    source_statuses: list[SourceStatus],
    combined_metrics: pd.DataFrame,
    run_summary: pd.DataFrame,
    latest_batch: pd.DataFrame,
    experiment_log: pd.DataFrame,
    speed_log: pd.DataFrame,
    recomputed_summary: pd.DataFrame,
    validation_policy_metrics: pd.DataFrame,
    trace_gate_contribution: pd.DataFrame,
    openai_agno_comparison: pd.DataFrame,
) -> dict[str, Path]:
    """Write the report, summary JSON, and combined CSV."""
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_prefix.with_name(output_prefix.name + "_combined_metrics.csv")
    summary_path = output_prefix.with_suffix(".json")
    report_path = output_prefix.with_suffix(".md")

    combined_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary = build_summary(
        source_statuses=source_statuses,
        combined_metrics=combined_metrics,
        run_summary=run_summary,
        latest_batch=latest_batch,
        trace_gate_contribution=trace_gate_contribution,
        metrics_path=metrics_path,
        report_path=report_path,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(
        build_report(
            source_statuses=source_statuses,
            combined_metrics=combined_metrics,
            run_summary=run_summary,
            latest_batch=latest_batch,
            experiment_log=experiment_log,
            speed_log=speed_log,
            recomputed_summary=recomputed_summary,
            validation_policy_metrics=validation_policy_metrics,
            trace_gate_contribution=trace_gate_contribution,
            openai_agno_comparison=openai_agno_comparison,
            summary=summary,
        ),
        encoding="utf-8",
    )
    return {
        "report": report_path,
        "summary": summary_path,
        "combined_metrics": metrics_path,
    }


def build_summary(
    *,
    source_statuses: list[SourceStatus],
    combined_metrics: pd.DataFrame,
    run_summary: pd.DataFrame,
    latest_batch: pd.DataFrame,
    trace_gate_contribution: pd.DataFrame,
    metrics_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Build a machine-readable summary for downstream checks."""
    headline = _headline_metrics(run_summary)
    latest_review = (
        latest_batch.loc[latest_batch["target"].eq("2차 검토대상(보류+부적격)")]
        if not latest_batch.empty and "target" in latest_batch.columns
        else pd.DataFrame()
    )
    if not latest_review.empty:
        headline["latest_batch_review_recall"] = float(latest_review.iloc[0]["Recall"])
    headline.update(_trace_gate_headline(trace_gate_contribution))
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_statuses": [
            {
                "name": status.name,
                "path": _relative(status.path),
                "exists": status.exists,
                "rows": status.rows,
                "columns": status.columns,
                "modified_at_utc": status.modified_at,
            }
            for status in source_statuses
        ],
        "combined_metric_rows": len(combined_metrics),
        "run_summary_rows": len(run_summary),
        "latest_batch_metric_rows": len(latest_batch),
        "trace_gate_contribution_rows": len(trace_gate_contribution),
        "headline": headline,
        "outputs": {
            "report": _relative(report_path),
            "combined_metrics": _relative(metrics_path),
        },
    }


def build_report(
    *,
    source_statuses: list[SourceStatus],
    combined_metrics: pd.DataFrame,
    run_summary: pd.DataFrame,
    latest_batch: pd.DataFrame,
    experiment_log: pd.DataFrame,
    speed_log: pd.DataFrame,
    recomputed_summary: pd.DataFrame,
    validation_policy_metrics: pd.DataFrame,
    trace_gate_contribution: pd.DataFrame,
    openai_agno_comparison: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    """Build a presentation-ready Stage 2 evaluation report."""
    headline = summary.get("headline", {})
    lines = [
        "# Stage 2 Evaluation Report",
        "",
        f"- 생성시각(UTC): `{summary['generated_at_utc']}`",
        "- 목적: Stage 2 에이전트 위원회의 보완 효과, 위험신호 성능, 실행 안정성, 속도 개선을 한 번에 점검한다.",
        "",
        "## 해석 주의",
        "",
        "- 이 리포트는 대시보드에 노출하는 사용자용 정확도 지표가 아니다.",
        "- 아래 수치는 과거 validation/test 기업-연도 replay와 Agno 파일럿 샘플을 기준으로 한다.",
        "- 2026년 추론 대상 기업 전체의 실제 정답률이나, 현재 선택 기업의 개별 정확도로 해석하면 안 된다.",
        "- Agno 파일럿 표본은 FN/FP/경계등급 등 어려운 케이스를 의도적으로 포함하므로 전체 모집단 성능으로 해석하지 않는다.",
        "- 발표에서는 “과거 오류 사례에서 Stage 2가 1차 모델 판단을 얼마나 보완했는지 보는 검증 자료”로 설명한다.",
        "",
        "## 핵심 요약",
        "",
        *_headline_lines(headline),
        "",
        "## Stage 2 성능 요약",
        "",
        "아래 표는 Agno/Claude 파일럿 및 오류위험 샘플의 성능표를 합쳐 run 단위로 재정리한 결과다.",
        "",
        _markdown_table(_report_run_summary_columns(run_summary)),
        "",
        "## 통합 분류 성능표",
        "",
        _markdown_table(combined_metrics),
        "",
        "## 최신 배치 결과 재계산",
        "",
        "현재 `committee_review_batch_results.csv`가 남아 있으면 같은 기준으로 즉시 재계산한다.",
        "",
        _markdown_table(latest_batch),
        "",
        "## 파일럿 성공률 로그",
        "",
        _markdown_table(_report_experiment_columns(experiment_log)),
        "",
        "## 속도 로그",
        "",
        _markdown_table(_report_speed_columns(speed_log)),
        "",
        "## 전체 파일럿 재검증 요약",
        "",
        _markdown_table(_report_recomputed_columns(recomputed_summary)),
        "",
        "## Validation/Test 정책 성능",
        "",
        "정책 선택은 validation 기준으로 보고, test는 확인용으로만 해석한다.",
        "",
        _markdown_table(_report_validation_policy_columns(validation_policy_metrics)),
        "",
        "## Decision Trace 게이트 기여도",
        "",
        "아래 표는 deterministic committee replay의 `decision_trace`를 이용해, 어떤 게이트가 1차 모델의 FN 끌어올림 또는 FP 완화에 함께 작동했는지 집계한 결과다.",
        "한 기업에서 여러 게이트가 동시에 켜질 수 있으므로 게이트별 건수는 서로 배타적이지 않다.",
        "",
        _markdown_table(_report_trace_gate_contribution_columns(trace_gate_contribution)),
        "",
        "## OpenAI Agno 설명 품질 비교",
        "",
        "같은 샘플을 deterministic과 OpenAI Agno로 각각 실행한 뒤 저장된 결과가 있으면, 최종 라벨 변화와 설명 품질 점수를 비교한다.",
        "현재 Codex 세션에서 실제 OpenAI 호출이 차단된 경우 이 표는 비어 있을 수 있다.",
        "",
        _markdown_table(_report_openai_agno_comparison_columns(openai_agno_comparison)),
        "",
        "## 해석 가이드",
        "",
        "- `2차 검토대상(보류+부적격)`은 조기경보 관점의 넓은 그물이다. Recall이 높을수록 위험 기업을 검토망에 올리는 능력이 좋다.",
        "- `2차 위험신호(risk_signal)`은 실제 빨간 경고에 가까운 신호다. Precision과 Recall을 함께 본다.",
        "- `2차 부적격만`은 가장 엄격한 확정 판단이다. Precision은 높을 수 있지만 Recall이 낮아질 수 있다.",
        "- `과민경고 완화 보류`, `확인필요 보류`, `경계등급 보류`는 위험 확정이 아니라 추가 확인 상태로 해석한다.",
        "",
        "## 입력 파일 상태",
        "",
        _markdown_table(pd.DataFrame([_source_status_row(status) for status in source_statuses])),
        "",
    ]
    return "\n".join(lines)


def _headline_lines(headline: dict[str, Any]) -> list[str]:
    if not headline:
        return ["- 아직 요약 가능한 Stage 2 성능표가 없습니다."]
    lines: list[str] = []
    if headline.get("best_risk_signal_run"):
        lines.append(
            "- 파일럿 표본 내 위험신호 F1 최고값: "
            f"`{headline['best_risk_signal_run']}` "
            f"F1 {headline['best_risk_signal_f1']:.4f}, "
            f"Precision {headline['best_risk_signal_precision']:.4f}, "
            f"Recall {headline['best_risk_signal_recall']:.4f}"
        )
    if headline.get("best_review_recall_run"):
        lines.append(
            "- 파일럿 표본 내 검토대상 Recall 최고값: "
            f"`{headline['best_review_recall_run']}` "
            f"Recall {headline['best_review_recall']:.4f}"
        )
    if headline.get("latest_batch_review_recall") is not None:
        lines.append(
            f"- 최신 배치 기준 검토대상 Recall: {headline['latest_batch_review_recall']:.4f}"
        )
    if headline.get("top_fn_gate"):
        lines.append(
            "- validation/test trace 기준 FN 보완 최다 게이트: "
            f"`{headline['top_fn_gate']}` "
            f"{headline['top_fn_count']}건"
        )
    if headline.get("top_fp_gate"):
        lines.append(
            "- validation/test trace 기준 FP 완화 최다 게이트: "
            f"`{headline['top_fp_gate']}` "
            f"{headline['top_fp_count']}건"
        )
    return lines or ["- 아직 요약 가능한 Stage 2 성능표가 없습니다."]


def _headline_metrics(run_summary: pd.DataFrame) -> dict[str, Any]:
    if run_summary.empty:
        return {}
    output: dict[str, Any] = {}
    risk = run_summary.dropna(subset=["risk_f1"])
    if not risk.empty:
        best = risk.sort_values(["risk_f1", "risk_recall", "risk_precision"], ascending=False).iloc[
            0
        ]
        output.update(
            {
                "best_risk_signal_run": best["run"],
                "best_risk_signal_f1": float(best["risk_f1"]),
                "best_risk_signal_precision": float(best["risk_precision"]),
                "best_risk_signal_recall": float(best["risk_recall"]),
            }
        )
    review = run_summary.dropna(subset=["review_recall"])
    if not review.empty:
        best_review = review.sort_values(["review_recall", "risk_f1"], ascending=False).iloc[0]
        output.update(
            {
                "best_review_recall_run": best_review["run"],
                "best_review_recall": float(best_review["review_recall"]),
            }
        )
    return output


def _trace_gate_headline(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    output: dict[str, Any] = {}
    fn_frame = frame.loc[pd.to_numeric(frame.get("fn_escalated_count"), errors="coerce").gt(0)]
    if not fn_frame.empty:
        best_fn = fn_frame.sort_values(
            ["fn_escalated_count", "triggered_count"],
            ascending=False,
        ).iloc[0]
        output["top_fn_gate"] = str(best_fn.get("gate_label") or best_fn.get("gate"))
        output["top_fn_count"] = int(best_fn["fn_escalated_count"])
    fp_frame = frame.loc[pd.to_numeric(frame.get("fp_softened_count"), errors="coerce").gt(0)]
    if not fp_frame.empty:
        best_fp = fp_frame.sort_values(
            ["fp_softened_count", "triggered_count"],
            ascending=False,
        ).iloc[0]
        output["top_fp_gate"] = str(best_fp.get("gate_label") or best_fp.get("gate"))
        output["top_fp_count"] = int(best_fp["fp_softened_count"])
    return output


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
        "target": scope,
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


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "위험신호 있음"}


def _metric_row(frame: pd.DataFrame, target: str) -> pd.Series | None:
    matched = frame.loc[frame["target"].astype(str).eq(target)]
    if matched.empty:
        return None
    return matched.iloc[0]


def _value(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def _delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return round(value - base, 4)


def _first_numeric(series: pd.Series | None) -> int | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return int(numeric.iloc[0])


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _report_run_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run",
        "n",
        "stage1_f1",
        "review_recall",
        "risk_precision",
        "risk_recall",
        "risk_f1",
        "risk_f1_delta_vs_stage1",
        "review_recall_delta_vs_stage1",
    ]
    return _select_columns(frame, columns)


def _report_experiment_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_group",
        "rows",
        "strict_success_rate",
        "review_safe_success_rate",
        "run_failures",
        "note",
    ]
    return _select_columns(frame, columns)


def _report_speed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment_group",
        "runner",
        "rows",
        "batch_wall_time_seconds",
        "case_elapsed_seconds_mean",
        "throughput_cases_per_minute",
        "note",
    ]
    return _select_columns(frame, columns)


def _report_recomputed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "artifact_dir",
        "rows",
        "strict_success_rate",
        "review_safe_success_rate",
        "run_failures",
        "speed_wall_sec",
        "throughput_cases_per_minute",
    ]
    return _select_columns(frame, columns)


def _report_validation_policy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "split",
        "policy",
        "n",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    if set(preferred).issubset(frame.columns):
        return frame.loc[:, preferred]
    return frame.head(20)


def _report_trace_gate_contribution_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "split",
        "gate_label",
        "triggered_count",
        "fn_escalated_count",
        "fn_escalation_share",
        "fp_softened_count",
        "fp_softening_share",
        "dominant_effect",
    ]
    if not set(preferred).issubset(frame.columns):
        return frame.head(20)
    output = frame.loc[:, preferred].copy()
    output["_split_order"] = output["split"].map({"valid": 0, "test": 1}).fillna(9)
    output = output.sort_values(
        ["_split_order", "fn_escalated_count", "fp_softened_count", "triggered_count"],
        ascending=[True, False, False, False],
    )
    return output.drop(columns=["_split_order"]).head(20)


def _report_openai_agno_comparison_columns(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "corp_name",
        "model_error_type",
        "stage1_label",
        "deterministic_label",
        "agno_label",
        "deterministic_quality_score",
        "agno_quality_score",
        "quality_delta",
    ]
    return _select_columns(frame, preferred).head(20)


def _select_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    existing = [column for column in columns if column in frame.columns]
    return frame.loc[:, existing].copy()


def _source_status_row(status: SourceStatus) -> dict[str, Any]:
    return {
        "file": status.name,
        "exists": status.exists,
        "rows": status.rows,
        "columns": status.columns,
        "modified_at_utc": status.modified_at or "",
    }


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


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    """Run the report export."""
    args = parse_args()
    diagnostics_dir = args.diagnostics_dir.resolve()
    sources: dict[str, pd.DataFrame] = {}
    statuses: list[SourceStatus] = []
    for filename in [
        AGNO_HOLD_METRICS,
        ERROR_RISK_METRICS,
        EXPERIMENT_LOG,
        SPEED_LOG,
        RECOMPUTED_SUMMARY,
        LATEST_BATCH_RESULTS,
        VALIDATION_TEST_POLICY_METRICS,
        TRACE_GATE_CONTRIBUTION,
        OPENAI_AGNO_COMPARISON_DETAILS,
    ]:
        frame, status = read_source(diagnostics_dir, filename)
        sources[filename] = frame
        statuses.append(status)

    combined_metrics = normalize_metrics(
        sources[AGNO_HOLD_METRICS],
        sources[ERROR_RISK_METRICS],
    )
    run_summary = summarize_runs(combined_metrics)
    latest_batch = latest_batch_metrics(sources[LATEST_BATCH_RESULTS])

    outputs = write_outputs(
        output_prefix=args.output_prefix,
        source_statuses=statuses,
        combined_metrics=combined_metrics,
        run_summary=run_summary,
        latest_batch=latest_batch,
        experiment_log=sources[EXPERIMENT_LOG],
        speed_log=sources[SPEED_LOG],
        recomputed_summary=sources[RECOMPUTED_SUMMARY],
        validation_policy_metrics=sources[VALIDATION_TEST_POLICY_METRICS],
        trace_gate_contribution=sources[TRACE_GATE_CONTRIBUTION],
        openai_agno_comparison=sources[OPENAI_AGNO_COMPARISON_DETAILS],
    )
    print(json.dumps({key: _relative(value) for key, value in outputs.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
