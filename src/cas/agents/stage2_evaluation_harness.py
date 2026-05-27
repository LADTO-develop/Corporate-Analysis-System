"""Shared metrics for Stage 2 provider evaluation harnesses."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
    "quality_gate_pass",
    "quality_gate_fail_reasons",
    "explanation_quality_score",
    "memo_quality_score",
    "evidence_grounding_score",
    "decision_consistency_score",
    "hallucination_flag_rate",
    "hallucination_flag_rows",
    "quality_rubric_method",
    "quality_judge_source",
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
    "risk_recall_guardrail_rate",
    "risk_recall_guardrail_rows",
    "any_qa_trigger_rate",
    "any_qa_trigger_rows",
    "llm_cache_hit_rate",
    "llm_cache_hit_rows",
    "any_cache_hit_rate",
    "any_cache_hit_rows",
    "run_failure_rows",
    "result_path",
]

QUALITY_GATE_THRESHOLDS = {
    "memo_quality_score": 0.55,
    "evidence_grounding_score": 0.50,
    "decision_consistency_score": 0.75,
    "hallucination_flag_rate": 0.05,
}


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
    risk_recall_guardrail_applied = _bool_column(
        frame,
        "stage2_risk_recall_guardrail_applied",
    )
    any_qa_triggered = review_qa_triggered | risk_recall_qa_triggered
    any_cache_hit = llm_cache_hit | review_qa_cache_hit | risk_recall_cache_hit
    run_failed = (
        committee_effect.eq("run_failed")
        | _non_empty_column(frame, "error_message")
        | _non_empty_column(frame, "stage2_error_message")
    )
    quality_metrics = explanation_quality_metrics(frame)
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
        **quality_metrics,
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
        "risk_recall_guardrail_rate": _rate(risk_recall_guardrail_applied),
        "risk_recall_guardrail_rows": int(risk_recall_guardrail_applied.sum()),
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
        "- RiskRecall guardrail: LLM QA 이후에도 적격으로 남은 고위험 누락 후보를 deterministic 규칙으로 위험 보류 승격한 비율",
        "- cache hit: Stage2 본 실행, ReviewQA, RiskRecallQA cache hit 중 하나라도 켜진 행을 별도로 집계",
        "- quality gate: memo/evidence/decision consistency와 hallucination flag를 함께 보는 보고서 품질 회귀 게이트",
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
        "quality_gate_pass",
        "quality_gate_fail_reasons",
        "explanation_quality_score",
        "memo_quality_score",
        "evidence_grounding_score",
        "decision_consistency_score",
        "hallucination_flag_rate",
        "stage2_policy_version",
        "stage2_latency_mean_seconds",
        "stage2_latency_p95_seconds",
        "stage2_latency_max_seconds",
        "explanation_quality_score",
        "estimated_cost_usd",
        "review_qa_trigger_rate",
        "risk_recall_qa_trigger_rate",
        "risk_recall_guardrail_rate",
        "any_qa_trigger_rate",
        "llm_cache_hit_rows",
        "any_cache_hit_rows",
        "run_failure_rows",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    return frame.loc[:, [column for column in columns if column in frame.columns]].copy()


def explanation_quality_metrics(results: pd.DataFrame) -> dict[str, Any]:
    """Return report-quality metrics for the official Stage 2 harness."""
    if results.empty:
        metrics = {
            "explanation_quality_score": 0.0,
            "memo_quality_score": 0.0,
            "evidence_grounding_score": 0.0,
            "financial_specificity_score": 0.0,
            "actionability_score": 0.0,
            "disagreement_resolution_score": 0.0,
            "decision_consistency_score": 0.0,
            "hallucination_flag_rate": 0.0,
            "hallucination_flag_rows": 0,
            "quality_rubric_method": "deterministic_text_and_diagnostics_v2",
            "quality_judge_source": "no_rows",
        }
        return {**metrics, **_quality_gate(metrics)}

    scores = results.apply(_explanation_quality_row, axis=1, result_type="expand")
    llm_metrics = _llm_judge_quality_metrics(results)
    quality_source = "llm_judge_columns" if llm_metrics else "deterministic_text_and_diagnostics"
    memo_quality = llm_metrics.get(
        "memo_quality_score",
        round(float(scores["memo_quality_score"].mean()), 4),
    )
    evidence_grounding = llm_metrics.get(
        "evidence_grounding_score",
        round(float(scores["evidence_grounding_score"].mean()), 4),
    )
    decision_consistency = llm_metrics.get(
        "decision_consistency_score",
        round(float(scores["decision_consistency_score"].mean()), 4),
    )
    hallucination_flag_rate = llm_metrics.get(
        "hallucination_flag_rate",
        _rate(scores["hallucination_flag"].astype(bool)),
    )
    hallucination_flag_rows = int(
        llm_metrics.get("hallucination_flag_rows", int(scores["hallucination_flag"].sum()))
    )
    explanation_quality = round(
        0.30 * float(memo_quality)
        + 0.30 * float(evidence_grounding)
        + 0.25 * float(decision_consistency)
        + 0.15 * (1.0 - float(hallucination_flag_rate)),
        4,
    )
    metrics = {
        "explanation_quality_score": explanation_quality,
        "memo_quality_score": memo_quality,
        "evidence_grounding_score": evidence_grounding,
        "financial_specificity_score": round(float(scores["financial_specificity_score"].mean()), 4),
        "actionability_score": round(float(scores["actionability_score"].mean()), 4),
        "disagreement_resolution_score": round(
            float(scores["disagreement_resolution_score"].mean()), 4
        ),
        "decision_consistency_score": decision_consistency,
        "hallucination_flag_rate": hallucination_flag_rate,
        "hallucination_flag_rows": hallucination_flag_rows,
        "quality_rubric_method": "official_report_quality_gate_v1",
        "quality_judge_source": quality_source,
    }
    return {**metrics, **_quality_gate(metrics)}


def _quality_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    fail_reasons: list[str] = []
    for metric_name in (
        "memo_quality_score",
        "evidence_grounding_score",
        "decision_consistency_score",
    ):
        value = _safe_float(metrics.get(metric_name), default=0.0)
        threshold = QUALITY_GATE_THRESHOLDS[metric_name]
        if value < threshold:
            fail_reasons.append(f"{metric_name}<{threshold}")
    hallucination_rate = _safe_float(metrics.get("hallucination_flag_rate"), default=1.0)
    hallucination_threshold = QUALITY_GATE_THRESHOLDS["hallucination_flag_rate"]
    if hallucination_rate > hallucination_threshold:
        fail_reasons.append(f"hallucination_flag_rate>{hallucination_threshold}")
    return {
        "quality_gate_pass": not fail_reasons,
        "quality_gate_fail_reasons": " / ".join(fail_reasons),
        "quality_gate_thresholds": json_like(
            {key: str(value) for key, value in QUALITY_GATE_THRESHOLDS.items()}
        ),
    }


def _explanation_quality_row(row: pd.Series) -> dict[str, float | bool]:
    text = _combined_explanation_text(row)
    memo_quality = _memo_quality_score(text)
    evidence_grounding = _evidence_grounding_score(row, text)
    financial_specificity = _financial_specificity_score(text)
    actionability = _actionability_score(row, text)
    disagreement_resolution = _disagreement_resolution_score(row, text)
    decision_consistency = _decision_consistency_score(row, text)
    hallucination_flag = _hallucination_flag(row, text)
    overall = (
        0.25 * memo_quality
        + 0.25 * evidence_grounding
        + 0.15 * financial_specificity
        + 0.15 * actionability
        + 0.10 * disagreement_resolution
        + 0.10 * decision_consistency
    )
    if hallucination_flag:
        overall = min(overall, 0.4)
    return {
        "memo_quality_score": round(memo_quality, 4),
        "evidence_grounding_score": round(evidence_grounding, 4),
        "financial_specificity_score": round(financial_specificity, 4),
        "actionability_score": round(actionability, 4),
        "disagreement_resolution_score": round(disagreement_resolution, 4),
        "decision_consistency_score": round(decision_consistency, 4),
        "hallucination_flag": bool(hallucination_flag),
        "explanation_quality_score": round(overall, 4),
    }


def _combined_explanation_text(row: pd.Series) -> str:
    columns = [
        "final_review_memo",
        "decision_trace",
        "risk_hold_reason_summary",
        "agent_disagreement_summary",
        "agent_disagreement_reasons",
        "conflict_resolution",
        "top_evidence_titles",
        "materiality_top_basis",
        "stage2_review_qa_advisory_apply_reason",
        "stage2_risk_recall_qa_advisory_apply_reason",
    ]
    return " ".join(str(row.get(column) or "") for column in columns)


def _memo_quality_score(text: str) -> float:
    length = len(text.strip())
    if length < 80:
        return 0.25
    if length < 180:
        return 0.55
    if length <= 1400:
        return 0.9
    return 0.7


def _evidence_grounding_score(row: pd.Series, text: str) -> float:
    evidence_status = str(row.get("evidence_status") or "").strip().lower()
    evidence_items = _safe_int(row.get("evidence_items"))
    evidence_unavailable = evidence_status in {
        "disabled",
        "missing_credentials",
        "not_implemented",
        "not_requested",
        "placeholder",
    } or evidence_items == 0
    if evidence_unavailable:
        if _overclaims_external_evidence(text):
            return 0.25
        if any(term in text for term in ("미수집", "비활성", "근거 없음", "확인된 외부")):
            return 0.85
        return 0.65

    score = 0.0
    if _bool_value(row.get("evidence_audit_structured_found")):
        score += 0.25
    if _safe_int(row.get("evidence_audit_critical_evidence_count")) > 0:
        score += 0.15
    if _safe_int(row.get("materiality_event_count")) > 0:
        score += 0.20
    if any(term in text for term in ("공시", "기사", "근거", "외부", "materiality", "evidence")):
        score += 0.20
    if str(row.get("top_evidence_titles") or "").strip():
        score += 0.20
    return min(score, 1.0)


def _financial_specificity_score(text: str) -> float:
    financial_terms = (
        "부채",
        "차입",
        "현금",
        "이자",
        "매출",
        "영업",
        "손실",
        "자본",
        "현금흐름",
        "ROA",
        "coverage",
        "ratio",
        "percentile",
    )
    term_hits = sum(1 for term in financial_terms if term.lower() in text.lower())
    numeric_hits = sum(char.isdigit() for char in text)
    return min(1.0, 0.18 * term_hits + (0.25 if numeric_hits >= 3 else 0.0))


def _actionability_score(row: pd.Series, text: str) -> float:
    score = 0.0
    if str(row.get("final_committee_label") or "").strip():
        score += 0.25
    if str(row.get("committee_decision_type") or "").strip():
        score += 0.20
    if str(row.get("risk_hold_reason_tags") or "").strip():
        score += 0.20
    if any(term in text for term in ("보류", "부적격", "적격", "검토", "확인", "모니터링")):
        score += 0.25
    if str(row.get("stage2_review_qa_recommended_action") or "").strip():
        score += 0.10
    return min(score, 1.0)


def _disagreement_resolution_score(row: pd.Series, text: str) -> float:
    disagreement = _safe_float(row.get("agent_disagreement_score"), default=0.0)
    if disagreement <= 0:
        return 0.8 if "conflict" not in text.lower() else 0.6
    score = 0.0
    if str(row.get("agent_disagreement_summary") or "").strip():
        score += 0.35
    if str(row.get("conflict_resolution") or "").strip():
        score += 0.35
    if any(term in text for term in ("상충", "충돌", "조정", "해소", "conflict")):
        score += 0.20
    return min(score, 1.0)


def _decision_consistency_score(row: pd.Series, text: str) -> float:
    final_label = str(row.get("final_committee_label") or "").strip()
    decision_type = str(row.get("committee_decision_type") or "").strip()
    if not final_label:
        return 0.0
    score = 0.25
    if _decision_type_matches_label(final_label, decision_type):
        score += 0.35
    if final_label in text:
        score += 0.20
    if str(row.get("decision_trace") or "").strip():
        score += 0.10
    if final_label == "보류" and str(row.get("risk_hold_reason_summary") or "").strip():
        score += 0.10
    if _has_conflicting_label(final_label, text):
        score -= 0.35
    return min(max(score, 0.0), 1.0)


def _hallucination_flag(row: pd.Series, text: str) -> bool:
    judge_flag = _first_existing_value(
        row,
        (
            "llm_judge_hallucination_flag",
            "judge_hallucination_flag",
            "hallucination_flag",
        ),
    )
    if judge_flag is not None:
        return _bool_value(judge_flag)
    official_overclaim_terms = (
        "신용등급을 확정",
        "등급을 확정",
        "최종 승인",
        "등급을 부여",
        "공식 신용등급",
        "부도 확정",
    )
    if any(term in text for term in official_overclaim_terms):
        return True
    evidence_status = str(row.get("evidence_status") or "").strip().lower()
    evidence_items = _safe_int(row.get("evidence_items"))
    if evidence_status in {"disabled", "missing_credentials", "not_requested"} or evidence_items == 0:
        return _overclaims_external_evidence(text)
    return False


def _overclaims_external_evidence(text: str) -> bool:
    if any(term in text for term in ("미수집", "비활성", "없음", "제한", "확인되지")):
        return False
    return any(term in text for term in ("확인된 공시", "공시 확인", "기사 확인", "보도 확인"))


def _decision_type_matches_label(final_label: str, decision_type: str) -> bool:
    if final_label == "적격":
        return decision_type == "eligible"
    if final_label == "보류":
        return decision_type in {
            "risk_hold",
            "boundary_hold",
            "mitigation_hold",
            "review_hold",
        }
    if final_label == "부적격":
        return decision_type == "reject"
    return False


def _has_conflicting_label(final_label: str, text: str) -> bool:
    labels = {"적격", "보류", "부적격"} - {final_label}
    return any(f"최종 {label}" in text or f"라벨은 {label}" in text for label in labels)


def _llm_judge_quality_metrics(results: pd.DataFrame) -> dict[str, Any]:
    memo = _first_numeric_column_mean(
        results,
        (
            "llm_judge_memo_quality",
            "llm_judge_memo_quality_score",
            "judge_memo_quality_score",
        ),
    )
    evidence = _first_numeric_column_mean(
        results,
        (
            "llm_judge_evidence_grounding",
            "llm_judge_evidence_grounding_score",
            "judge_evidence_grounding_score",
        ),
    )
    consistency = _first_numeric_column_mean(
        results,
        (
            "llm_judge_decision_consistency",
            "llm_judge_decision_consistency_score",
            "judge_decision_consistency_score",
        ),
    )
    hallucination = _first_bool_column_rate(
        results,
        (
            "llm_judge_hallucination_flag",
            "judge_hallucination_flag",
        ),
    )
    metrics: dict[str, Any] = {}
    if memo is not None:
        metrics["memo_quality_score"] = memo
    if evidence is not None:
        metrics["evidence_grounding_score"] = evidence
    if consistency is not None:
        metrics["decision_consistency_score"] = consistency
    if hallucination is not None:
        rate, rows = hallucination
        metrics["hallucination_flag_rate"] = rate
        metrics["hallucination_flag_rows"] = rows
    return metrics


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


def _first_existing_value(row: pd.Series, columns: tuple[str, ...]) -> object | None:
    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if _non_empty_value(value):
            return cast(object, value)
    return None


def _first_numeric_column_mean(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> float | None:
    for column in columns:
        values = _numeric_column(frame, column)
        if values.empty:
            continue
        normalized = values
        if float(values.max()) > 1.0:
            normalized = values / 5.0
        return round(float(normalized.clip(lower=0.0, upper=1.0).mean()), 4)
    return None


def _first_bool_column_rate(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[float, int] | None:
    for column in columns:
        if column not in frame.columns:
            continue
        values = _bool_column(frame, column)
        return _rate(values), int(values.sum())
    return None


def _safe_int(value: object) -> int:
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object, *, default: float) -> float:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


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
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
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
        max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)
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
    "explanation_quality_metrics",
    "provider_summary_frame",
    "summarize_batch_results",
    "summarize_by_category",
]
