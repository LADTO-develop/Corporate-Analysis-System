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
    return metrics


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
