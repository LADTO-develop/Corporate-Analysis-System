"""Compare Stage 2 multi-role provider assignments on a small shared sample."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import export_stage2_feature46_full_review_trigger_harness as harness  # noqa: E402
import export_stage2_rolling_validation_samples as sample_export  # noqa: E402

OUTPUT_DIR = (
    ROOT
    / "data/outputs/modeling/feature_46_xgboost/diagnostics/stage2_agents/"
    "feature46_full_review_trigger_73_role_assignment_20"
)
PROVIDER_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1-mini",
    "google": "gemini-2.5-flash",
}
MODEL_PRICING_USD_PER_1M = {
    "anthropic:claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "openai:gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "google:gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}
ROLE_TOKEN_ESTIMATES = {
    "quant_credit": {"input": 2600, "output": 550},
    "evidence_audit": {"input": 3200, "output": 700},
    "chair_report": {"input": 2800, "output": 550},
    "review_qa": {"input": 2600, "output": 450},
    "risk_recall_qa": {"input": 2600, "output": 450},
}
SELECTED_ROLE_ASSIGNMENT_ID = "gemini_quant_claude_evidence_openai_chair"
ASSIGNMENTS = [
    ("claude_quant_openai_evidence_gemini_chair", "anthropic", "openai", "google"),
    ("claude_quant_gemini_evidence_openai_chair", "anthropic", "google", "openai"),
    ("openai_quant_claude_evidence_gemini_chair", "openai", "anthropic", "google"),
    ("openai_quant_gemini_evidence_claude_chair", "openai", "google", "anthropic"),
    ("gemini_quant_claude_evidence_openai_chair", "google", "anthropic", "openai"),
    ("gemini_quant_openai_evidence_claude_chair", "google", "openai", "anthropic"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--policy", default=harness.DEFAULT_POLICY)
    parser.add_argument("--eval-years", type=int, nargs="+", default=sample_export.ROLLING_EVAL_YEARS)
    parser.add_argument("--sample-per-category", type=int, default=15)
    parser.add_argument("--batch-per-category", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed-attempts", type=int, default=2)
    parser.add_argument("--retry-failed-delay-seconds", type=float, default=20.0)
    parser.add_argument("--stage2-llm-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--stage2-fallback-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--live-external-evidence", action="store_true")
    parser.add_argument("--reuse-existing-runs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path, sample_summary = harness.generate_rolling_samples(output_dir, _harness_args(args))

    summaries: list[dict[str, Any]] = []
    category_frames: list[pd.DataFrame] = []
    skipped_runs: list[dict[str, str]] = []
    for assignment_id, quant_provider, evidence_provider, chair_provider in ASSIGNMENTS:
        run_args = _harness_args(
            args,
            assignment_id=assignment_id,
            quant_provider=quant_provider,
            evidence_provider=evidence_provider,
            chair_provider=chair_provider,
        )
        summary, category_summary, skipped = harness.run_provider(
            provider="multi_role",
            samples_path=samples_path,
            output_dir=output_dir,
            args=run_args,
        )
        if skipped is not None:
            skipped_runs.append(skipped)
            continue
        if summary is not None:
            summaries.append(summary)
        if not category_summary.empty:
            category_frames.append(category_summary)

    provider_summary = harness.provider_summary_frame(summaries)
    provider_summary.insert(4, "role_assignment_id", provider_summary["run_id"].str.removeprefix("multi_role_"))
    provider_summary.insert(
        5,
        "selected_default",
        provider_summary["role_assignment_id"].eq(SELECTED_ROLE_ASSIGNMENT_ID),
    )
    provider_summary = _attach_assignment_metrics(provider_summary, output_dir)
    category_summary = (
        pd.concat(category_frames, ignore_index=True) if category_frames else pd.DataFrame()
    )
    outputs = harness.write_harness_outputs(
        output_dir=output_dir,
        provider_summary=provider_summary,
        category_summary=category_summary,
        sample_summary={
            **sample_summary,
            "experiment": "multi_role_provider_assignment_20",
            "assignment_rows": len(ASSIGNMENTS),
        },
        skipped_runs=skipped_runs,
    )
    assignment_path = output_dir / "stage2_role_assignment_manifest.csv"
    _assignment_manifest().to_csv(assignment_path, index=False, encoding="utf-8-sig")
    summary_path = output_dir / "stage2_role_assignment_experiment_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "selected_role_assignment_id": SELECTED_ROLE_ASSIGNMENT_ID,
                "outputs": outputs,
                "assignment_manifest": harness._relative(assignment_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **outputs,
                "assignment_manifest": harness._relative(assignment_path),
                "experiment_summary": harness._relative(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _harness_args(
    args: argparse.Namespace,
    *,
    assignment_id: str | None = None,
    quant_provider: str | None = None,
    evidence_provider: str | None = None,
    chair_provider: str | None = None,
) -> argparse.Namespace:
    return SimpleNamespace(
        policy=args.policy,
        eval_years=args.eval_years,
        sample_per_category=args.sample_per_category,
        batch_per_category=args.batch_per_category,
        max_cases=args.max_cases,
        workers=args.workers,
        retry_failed_attempts=args.retry_failed_attempts,
        retry_failed_delay_seconds=args.retry_failed_delay_seconds,
        stage2_fallback_on_error=args.stage2_fallback_on_error,
        stage2_llm_cache=args.stage2_llm_cache,
        live_external_evidence=args.live_external_evidence,
        strict_provider_keys=True,
        reuse_existing_runs=args.reuse_existing_runs,
        openai_model=PROVIDER_MODELS["openai"],
        gemini_model=PROVIDER_MODELS["google"],
        role_assignment_id=assignment_id,
        quant_provider=quant_provider,
        quant_model=_model_for_provider(quant_provider),
        evidence_provider=evidence_provider,
        evidence_model=_model_for_provider(evidence_provider),
        chair_provider=chair_provider,
        chair_model=_model_for_provider(chair_provider),
        mark_skipped=[],
    )


def _model_for_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    return PROVIDER_MODELS[provider]


def _assignment_manifest() -> pd.DataFrame:
    rows = []
    for assignment_id, quant_provider, evidence_provider, chair_provider in ASSIGNMENTS:
        rows.append(
            {
                "role_assignment_id": assignment_id,
                "selected_default": assignment_id == SELECTED_ROLE_ASSIGNMENT_ID,
                "quant_credit": f"{quant_provider}:{PROVIDER_MODELS[quant_provider]}",
                "evidence_audit": f"{evidence_provider}:{PROVIDER_MODELS[evidence_provider]}",
                "chair_report": f"{chair_provider}:{PROVIDER_MODELS[chair_provider]}",
            }
        )
    return pd.DataFrame(rows)


def _attach_assignment_metrics(provider_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if provider_summary.empty:
        return provider_summary
    enriched = provider_summary.copy()
    metric_rows: dict[str, dict[str, Any]] = {}
    for row in enriched.to_dict(orient="records"):
        run_id = str(row.get("run_id") or "")
        result_path = output_dir / "runs" / run_id / "committee_review_batch_results.csv"
        if not result_path.exists():
            continue
        results = pd.read_csv(result_path)
        assignment = _assignment_from_model_label(str(row.get("model") or ""))
        metric_rows[run_id] = {
            **_review_or_reject_metrics(results),
            **_explanation_quality_metrics(results),
            **_assignment_cost_estimate(results, assignment),
        }
    if not metric_rows:
        return enriched
    metrics = pd.DataFrame(
        [{"run_id": run_id, **values} for run_id, values in metric_rows.items()]
    )
    return enriched.merge(metrics, on="run_id", how="left")


def _review_or_reject_metrics(results: pd.DataFrame) -> dict[str, Any]:
    actual_positive = results["actual_label_name"].astype(str).eq("투기등급")
    predicted_positive = results["final_committee_label"].astype(str).isin({"보류", "부적격"})
    risk_review_positive = _risk_review_positive(results)
    reject_positive = results["final_committee_label"].astype(str).eq("부적격")
    review_metrics = _binary_metrics(actual_positive, predicted_positive)
    risk_review_metrics = _binary_metrics(actual_positive, risk_review_positive)
    reject_metrics = _binary_metrics(actual_positive, reject_positive)
    return {
        "review_or_reject_precision": review_metrics["precision"],
        "review_or_reject_recall": review_metrics["recall"],
        "review_or_reject_f1": review_metrics["f1"],
        "review_or_reject_tp": review_metrics["tp"],
        "review_or_reject_fp": review_metrics["fp"],
        "review_or_reject_fn": review_metrics["fn"],
        "review_positive_definition": "final_committee_label in {보류, 부적격}",
        "risk_review_or_reject_precision": risk_review_metrics["precision"],
        "risk_review_or_reject_recall": risk_review_metrics["recall"],
        "risk_review_or_reject_f1": risk_review_metrics["f1"],
        "risk_review_or_reject_tp": risk_review_metrics["tp"],
        "risk_review_or_reject_fp": risk_review_metrics["fp"],
        "risk_review_or_reject_fn": risk_review_metrics["fn"],
        "risk_review_positive_definition": (
            "final_committee_label=부적격 or risk-signal hold "
            "(risk_hold/review_hold/boundary_hold or risk/critical hold tags)"
        ),
        "reject_only_precision": reject_metrics["precision"],
        "reject_only_recall": reject_metrics["recall"],
        "reject_only_f1": reject_metrics["f1"],
    }


def _risk_review_positive(results: pd.DataFrame) -> pd.Series:
    final_label = results["final_committee_label"].astype(str)
    decision_type = results.get("committee_decision_type", pd.Series("", index=results.index)).astype(str)
    risk_signal = results.get("committee_risk_signal", pd.Series("", index=results.index)).astype(str)
    hold_tags = results.get("risk_hold_reason_tags", pd.Series("", index=results.index)).astype(str)
    risk_hold_types = decision_type.isin({"risk_hold", "review_hold", "boundary_hold", "reject"})
    risk_tagged_hold = final_label.eq("보류") & (
        risk_signal.str.strip().astype(bool)
        | hold_tags.str.contains("risk|critical|distress|tail|boundary", case=False, regex=True)
    )
    return final_label.eq("부적격") | (final_label.eq("보류") & risk_hold_types) | risk_tagged_hold


def _binary_metrics(actual_positive: pd.Series, predicted_positive: pd.Series) -> dict[str, Any]:
    actual = actual_positive.astype(bool)
    predicted = predicted_positive.astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _assignment_cost_estimate(
    results: pd.DataFrame,
    assignment: dict[str, str],
) -> dict[str, Any]:
    rows = len(results)
    review_qa_rows = int(_bool_column(results, "stage2_review_qa_triggered").sum())
    risk_recall_qa_rows = int(_bool_column(results, "stage2_risk_recall_qa_triggered").sum())
    token_counts: dict[str, dict[str, int]] = {}
    for role in ("quant_credit", "evidence_audit", "chair_report"):
        _add_role_tokens(token_counts, assignment[role], role, rows)
    _add_role_tokens(token_counts, assignment["chair_report"], "review_qa", review_qa_rows)
    _add_role_tokens(token_counts, assignment["chair_report"], "risk_recall_qa", risk_recall_qa_rows)
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    for model_id, counts in token_counts.items():
        pricing = MODEL_PRICING_USD_PER_1M[model_id]
        total_input_tokens += counts["input"]
        total_output_tokens += counts["output"]
        total_cost += (
            counts["input"] * pricing["input"] / 1_000_000
            + counts["output"] * pricing["output"] / 1_000_000
        )
    return {
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(total_cost, 6),
        "cost_estimate_method": (
            "role_call_token_estimate_no_cache; excludes failed provider attempts before retry"
        ),
    }


def _explanation_quality_metrics(results: pd.DataFrame) -> dict[str, Any]:
    scores = results.apply(_explanation_quality_row, axis=1, result_type="expand")
    return {
        "explanation_quality_score": round(float(scores["explanation_quality_score"].mean()), 4),
        "memo_quality_score": round(float(scores["memo_quality_score"].mean()), 4),
        "evidence_grounding_score": round(float(scores["evidence_grounding_score"].mean()), 4),
        "financial_specificity_score": round(
            float(scores["financial_specificity_score"].mean()), 4
        ),
        "actionability_score": round(float(scores["actionability_score"].mean()), 4),
        "disagreement_resolution_score": round(
            float(scores["disagreement_resolution_score"].mean()), 4
        ),
        "quality_rubric_method": "deterministic_text_and_diagnostics_v1",
    }


def _explanation_quality_row(row: pd.Series) -> dict[str, float]:
    text = _combined_explanation_text(row)
    memo_quality = _memo_quality_score(text)
    evidence_grounding = _evidence_grounding_score(row, text)
    financial_specificity = _financial_specificity_score(text)
    actionability = _actionability_score(row, text)
    disagreement_resolution = _disagreement_resolution_score(row, text)
    overall = (
        0.25 * memo_quality
        + 0.25 * evidence_grounding
        + 0.20 * financial_specificity
        + 0.20 * actionability
        + 0.10 * disagreement_resolution
    )
    return {
        "memo_quality_score": round(memo_quality, 4),
        "evidence_grounding_score": round(evidence_grounding, 4),
        "financial_specificity_score": round(financial_specificity, 4),
        "actionability_score": round(actionability, 4),
        "disagreement_resolution_score": round(disagreement_resolution, 4),
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
    score = 0.0
    if _bool_value(row.get("evidence_audit_structured_found")):
        score += 0.35
    if _safe_int(row.get("evidence_audit_critical_evidence_count")) > 0:
        score += 0.20
    if _safe_int(row.get("materiality_event_count")) > 0:
        score += 0.20
    if any(term in text for term in ("공시", "기사", "근거", "외부", "materiality", "evidence")):
        score += 0.20
    if str(row.get("top_evidence_titles") or "").strip():
        score += 0.15
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
    disagreement = _safe_float(row.get("agent_disagreement_score"))
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


def _assignment_from_model_label(model_label: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for part in model_label.split(";"):
        role, sep, provider_model = part.strip().partition("=")
        if sep:
            output[role.strip()] = provider_model.strip()
    for role, provider in {
        "quant_credit": "anthropic",
        "evidence_audit": "openai",
        "chair_report": "google",
    }.items():
        output.setdefault(role, f"{provider}:{PROVIDER_MODELS[provider]}")
    return output


def _add_role_tokens(
    token_counts: dict[str, dict[str, int]],
    model_id: str,
    role: str,
    calls: int,
) -> None:
    estimates = ROLE_TOKEN_ESTIMATES[role]
    bucket = token_counts.setdefault(model_id, {"input": 0, "output": 0})
    bucket["input"] += estimates["input"] * calls
    bucket["output"] += estimates["output"] * calls


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].map(_bool_value).fillna(False).astype(bool)


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


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


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
