"""Shared metrics for Stage 2 provider evaluation harnesses."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_COLUMNS = [
    "run_id",
    "runner",
    "provider",
    "model",
    "run_status",
    "skip_reason",
    "rows",
    "strict_success_rate",
    "review_safe_success_rate",
    "fn_rescue_success_rate",
    "fn_cases",
    "fp_over_hold_rate",
    "fp_over_hold_count",
    "fp_cases",
    "stage2_policy_version",
    "stage2_latency_mean_seconds",
    "stage2_latency_p95_seconds",
    "stage2_latency_max_seconds",
    "case_latency_mean_seconds",
    "case_latency_p95_seconds",
    "case_latency_max_seconds",
    "review_qa_trigger_rate",
    "review_qa_trigger_rows",
    "risk_recall_qa_trigger_rate",
    "risk_recall_qa_trigger_rows",
    "any_qa_trigger_rate",
    "any_qa_trigger_rows",
    "llm_cache_hit_rate",
    "llm_cache_hit_rows",
    "any_cache_hit_rate",
    "any_cache_hit_rows",
    "run_failure_rows",
    "result_path",
]


def summarize_batch_results(
    results: pd.DataFrame,
    *,
    run_id: str,
    runner: str,
    provider: str,
    model: str,
    result_path: str = "",
    stage2_policy_version: str = "",
) -> dict[str, Any]:
    """Summarize one deterministic or LLM-backed Stage 2 batch result."""
    frame = results.copy()
    rows = len(frame)
    model_error_type = frame.get("model_error_type", pd.Series(dtype=object)).astype(str)
    final_label = frame.get("final_committee_label", pd.Series(dtype=object)).astype(str)
    committee_effect = frame.get("committee_effect", pd.Series(dtype=object)).astype(str)
    fn_mask = model_error_type.eq("false_negative")
    fp_mask = model_error_type.eq("false_positive")
    fp_over_hold = fp_mask & final_label.eq("보류")
    stage2_latency = _numeric_column(frame, "stage2_total_elapsed_seconds")
    case_latency = _numeric_column(frame, "case_elapsed_seconds")
    llm_cache_hit = _bool_column(frame, "stage2_llm_cache_hit")
    review_qa_cache_hit = _bool_column(frame, "stage2_review_qa_cache_hit")
    risk_recall_cache_hit = _bool_column(frame, "stage2_risk_recall_qa_cache_hit")
    review_qa_triggered = _bool_column(frame, "stage2_review_qa_triggered")
    risk_recall_qa_triggered = _bool_column(frame, "stage2_risk_recall_qa_triggered")
    any_qa_triggered = review_qa_triggered | risk_recall_qa_triggered
    any_cache_hit = llm_cache_hit | review_qa_cache_hit | risk_recall_cache_hit
    run_failed = (
        committee_effect.eq("run_failed")
        | _non_empty_column(frame, "error_message")
        | _non_empty_column(frame, "stage2_error_message")
    )
    return {
        "run_id": run_id,
        "runner": runner,
        "provider": provider,
        "model": model,
        "run_status": "completed",
        "skip_reason": "",
        "rows": rows,
        "strict_success_rate": _bool_mean(frame, "committee_success"),
        "review_safe_success_rate": _bool_mean(frame, "committee_review_safe_success"),
        "fn_rescue_success_rate": _rate(committee_effect[fn_mask].eq("fn_escalated")),
        "fn_cases": int(fn_mask.sum()),
        "fp_over_hold_rate": _rate(fp_over_hold[fp_mask]),
        "fp_over_hold_count": int(fp_over_hold.sum()),
        "fp_cases": int(fp_mask.sum()),
        "stage2_policy_version": stage2_policy_version,
        "stage2_latency_mean_seconds": _series_mean(stage2_latency),
        "stage2_latency_p95_seconds": _series_quantile(stage2_latency, 0.95),
        "stage2_latency_max_seconds": _series_max(stage2_latency),
        "case_latency_mean_seconds": _series_mean(case_latency),
        "case_latency_p95_seconds": _series_quantile(case_latency, 0.95),
        "case_latency_max_seconds": _series_max(case_latency),
        "review_qa_trigger_rate": _rate(review_qa_triggered),
        "review_qa_trigger_rows": int(review_qa_triggered.sum()),
        "risk_recall_qa_trigger_rate": _rate(risk_recall_qa_triggered),
        "risk_recall_qa_trigger_rows": int(risk_recall_qa_triggered.sum()),
        "any_qa_trigger_rate": _rate(any_qa_triggered),
        "any_qa_trigger_rows": int(any_qa_triggered.sum()),
        "llm_cache_hit_rate": _rate(llm_cache_hit),
        "llm_cache_hit_rows": int(llm_cache_hit.sum()),
        "any_cache_hit_rate": _rate(any_cache_hit),
        "any_cache_hit_rows": int(any_cache_hit.sum()),
        "run_failure_rows": int(run_failed.sum()),
        "result_path": result_path,
    }


def summarize_by_category(results: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    """Return strict/review-safe success by sample category for one run."""
    if results.empty or "sample_category" not in results.columns:
        return pd.DataFrame(
            columns=[
                "run_id",
                "sample_category",
                "rows",
                "strict_success_rate",
                "review_safe_success_rate",
            ]
        )
    rows: list[dict[str, Any]] = []
    for category, group in results.groupby("sample_category", dropna=False):
        rows.append(
            {
                "run_id": run_id,
                "sample_category": str(category),
                "rows": len(group),
                "strict_success_rate": _bool_mean(group, "committee_success"),
                "review_safe_success_rate": _bool_mean(group, "committee_review_safe_success"),
            }
        )
    return pd.DataFrame(rows).sort_values(["run_id", "sample_category"]).reset_index(drop=True)


def provider_summary_frame(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize provider summaries into a stable column order."""
    if not summaries:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    frame = pd.DataFrame(summaries)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, SUMMARY_COLUMNS].copy()


def build_harness_report(
    *,
    provider_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    sample_summary: dict[str, Any],
    skipped_runs: list[dict[str, str]],
    output_dir: Path,
    stage2_policy_version: str,
    prompt_contract_versions: Mapping[str, str] | None = None,
) -> str:
    """Build a compact Markdown report for the feature_46/full_review_trigger_73 harness."""
    prompt_versions = dict(prompt_contract_versions or {})
    lines = [
        "# Stage 2 Feature 46 / full_review_trigger_73 Evaluation Harness",
        "",
        f"- 생성시각(UTC): `{datetime.now(UTC).isoformat(timespec='seconds')}`",
        "- 기준: 공식 Stage1 `feature_46_xgboost` + Stage2 trigger `full_review_trigger_73`",
        f"- Stage2 policy version: `{stage2_policy_version}`",
        f"- Prompt contract versions: `{json_like(prompt_versions)}`",
        "- 목적: rolling validation 전체 샘플에서 deterministic/OpenAI/Gemini/multi-role 실행을 같은 지표로 비교한다.",
        "",
        "## Sample Pool",
        "",
        _markdown_table(pd.DataFrame([sample_summary])),
        "",
        "## Provider Summary",
        "",
        _markdown_table(_report_provider_columns(provider_summary)),
        "",
        "## Category Summary",
        "",
        _markdown_table(category_summary),
        "",
        "## Skipped Runs",
        "",
        _markdown_table(pd.DataFrame(skipped_runs)),
        "",
        "## Output Directory",
        "",
        f"`{output_dir}`",
        "",
        "## Metric Notes",
        "",
        "- strict success: 오류유형별 기대 최종 라벨을 엄격히 만족한 비율",
        "- review-safe success: 정상기업을 부적격으로 악화시키지 않는 넓은 검토 안전 기준",
        "- FN rescue 성공률: 1차 모델 false negative가 Stage2에서 보류/부적격으로 올라간 비율",
        "- FP over-hold: 1차 모델 false positive가 Stage2에서 부적격은 피했지만 보류로 남은 건수",
        "- latency p95: provider별 긴 꼬리 지연을 보기 위한 95백분위 실행 시간",
        "- QA trigger rate: ReviewQA 또는 RiskRecallQA가 실제로 트리거된 행 비율",
        "- cache hit: Stage2 본 실행, ReviewQA, RiskRecallQA cache hit 중 하나라도 켜진 행을 별도로 집계",
        "",
    ]
    return "\n".join(lines)


def json_like(value: Mapping[str, str]) -> str:
    """Return a compact deterministic mapping string for Markdown reports."""
    return ", ".join(f"{key}={value[key]}" for key in sorted(value)) or "not_recorded"


def _report_provider_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "role_assignment_id",
        "selected_default",
        "run_status",
        "skip_reason",
        "rows",
        "strict_success_rate",
        "review_safe_success_rate",
        "fn_rescue_success_rate",
        "review_or_reject_precision",
        "review_or_reject_recall",
        "review_or_reject_f1",
        "risk_review_or_reject_precision",
        "risk_review_or_reject_recall",
        "risk_review_or_reject_f1",
        "fp_over_hold_count",
        "stage2_policy_version",
        "stage2_latency_mean_seconds",
        "stage2_latency_p95_seconds",
        "stage2_latency_max_seconds",
        "explanation_quality_score",
        "estimated_cost_usd",
        "review_qa_trigger_rate",
        "risk_recall_qa_trigger_rate",
        "any_qa_trigger_rate",
        "llm_cache_hit_rows",
        "any_cache_hit_rows",
        "run_failure_rows",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.loc[:, [column for column in columns if column in frame.columns]].copy()


def _bool_mean(frame: pd.DataFrame, column: str) -> float:
    return _rate(_bool_column(frame, column))


def _rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return round(float(series.astype(bool).mean()), 4)


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].map(_bool_value).fillna(False).astype(bool)


def _non_empty_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].map(_non_empty_value).fillna(False).astype(bool)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _series_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return round(float(series.mean()), 4)


def _series_max(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return round(float(series.max()), 4)


def _series_quantile(series: pd.Series, quantile: float) -> float | None:
    if series.empty:
        return None
    return round(float(series.quantile(quantile)), 4)


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _non_empty_value(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.astype(object).where(pd.notna(frame), "")
    headers = [str(column) for column in display.columns]
    rows = [[_format_cell(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


__all__ = [
    "SUMMARY_COLUMNS",
    "build_harness_report",
    "provider_summary_frame",
    "summarize_batch_results",
    "summarize_by_category",
]
