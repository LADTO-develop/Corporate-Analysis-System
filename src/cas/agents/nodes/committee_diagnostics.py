"""Committee runtime diagnostic helpers."""

from __future__ import annotations

from typing import Any

from cas.agents.nodes.evidence_profile import _safe_float
from cas.agents.signals import evaluate_agent_disagreement
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput


def _runtime_diagnostic_metrics(diagnostics: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source_key, metric_key in (
        ("stage2_total_elapsed_seconds", "stage2_total_elapsed_seconds"),
        ("agent_elapsed_seconds_sum", "stage2_agent_elapsed_seconds_sum"),
    ):
        value = _safe_float(diagnostics.get(source_key))
        if value is not None:
            metrics[metric_key] = value
    agent_timings = diagnostics.get("agent_elapsed_seconds")
    if isinstance(agent_timings, dict):
        for role in (
            "quant_credit",
            "evidence_audit",
            "chair_report",
            "review_qa",
            "risk_recall_qa",
            "llm_client",
        ):
            value = _safe_float(agent_timings.get(role))
            if value is not None:
                metrics[f"stage2_{role}_elapsed_seconds"] = value
    metrics["stage2_llm_cache_hit"] = 1.0 if diagnostics.get("cache_hit") is True else 0.0
    metrics["stage2_response_cache_hit"] = (
        1.0 if diagnostics.get("response_cache_hit") is True else 0.0
    )
    metrics["stage2_degraded"] = 1.0 if diagnostics.get("degraded") is True else 0.0
    retry_count = _safe_float(diagnostics.get("retry_count"))
    if retry_count is not None:
        metrics["stage2_retry_count"] = retry_count
    role_cache_hits = diagnostics.get("role_cache_hits")
    if isinstance(role_cache_hits, dict):
        for role in ("quant_credit", "evidence_audit", "chair_report"):
            metrics[f"stage2_{role}_cache_hit"] = 1.0 if role_cache_hits.get(role) is True else 0.0
    role_fallback_used = diagnostics.get("role_fallback_used")
    if isinstance(role_fallback_used, dict):
        for role in ("quant_credit", "evidence_audit", "chair_report"):
            metrics[f"stage2_{role}_fallback_used"] = (
                1.0 if role_fallback_used.get(role) is True else 0.0
            )
    role_cache_count = _safe_float(diagnostics.get("role_cache_hit_count"))
    if role_cache_count is not None:
        metrics["stage2_role_cache_hit_count"] = role_cache_count
    _attach_usage_metrics(metrics, diagnostics)
    metrics["stage2_review_qa_triggered"] = (
        1.0 if diagnostics.get("review_qa_triggered") is True else 0.0
    )
    metrics["stage2_review_qa_cache_hit"] = (
        1.0 if diagnostics.get("review_qa_cache_hit") is True else 0.0
    )
    metrics["stage2_review_qa_advisory_applied"] = (
        1.0 if diagnostics.get("review_qa_advisory_applied") is True else 0.0
    )
    metrics["stage2_risk_recall_qa_triggered"] = (
        1.0 if diagnostics.get("risk_recall_qa_triggered") is True else 0.0
    )
    metrics["stage2_risk_recall_qa_cache_hit"] = (
        1.0 if diagnostics.get("risk_recall_qa_cache_hit") is True else 0.0
    )
    metrics["stage2_risk_recall_qa_advisory_applied"] = (
        1.0 if diagnostics.get("risk_recall_qa_advisory_applied") is True else 0.0
    )
    metrics["stage2_risk_recall_guardrail_applied"] = (
        1.0 if diagnostics.get("risk_recall_guardrail_applied") is True else 0.0
    )
    return metrics


def _attach_usage_metrics(metrics: dict[str, float], diagnostics: dict[str, Any]) -> None:
    token_totals = diagnostics.get("token_usage_totals")
    if isinstance(token_totals, dict):
        for source_key, metric_key in (
            ("input_tokens", "stage2_input_tokens"),
            ("output_tokens", "stage2_output_tokens"),
            ("total_tokens", "stage2_total_tokens"),
            ("billable_input_tokens", "stage2_billable_input_tokens"),
            ("billable_output_tokens", "stage2_billable_output_tokens"),
            ("billable_total_tokens", "stage2_billable_total_tokens"),
            ("cost_usd", "stage2_cost_usd"),
            ("billable_cost_usd", "stage2_billable_cost_usd"),
        ):
            value = _safe_float(token_totals.get(source_key))
            if value is not None:
                metrics[metric_key] = value
    role_usage = diagnostics.get("role_token_usage")
    if not isinstance(role_usage, dict):
        return
    for role in (
        "quant_credit",
        "evidence_audit",
        "chair_report",
        "review_qa",
        "risk_recall_qa",
        "llm_client",
    ):
        usage = role_usage.get(role)
        if not isinstance(usage, dict):
            continue
        for source_key, suffix in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
            ("billable_input_tokens", "billable_input_tokens"),
            ("billable_output_tokens", "billable_output_tokens"),
            ("billable_total_tokens", "billable_total_tokens"),
            ("cost_usd", "cost_usd"),
            ("billable_cost_usd", "billable_cost_usd"),
        ):
            value = _safe_float(usage.get(source_key))
            if value is not None:
                metrics[f"stage2_{role}_{suffix}"] = value


def _attach_agent_disagreement(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    structured_outputs: tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput],
    runtime_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    disagreement = evaluate_agent_disagreement(
        bundle=bundle,
        committee_view=committee_view,
        quant_credit=structured_outputs[0],
        evidence_audit=structured_outputs[1],
        chair_report=structured_outputs[2],
    )
    payload = disagreement.as_payload()
    runtime_diagnostics.update(payload)
    return {**committee_view, **payload}
