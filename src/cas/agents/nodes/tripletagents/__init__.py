"""Agno-backed Stage 2 triplet agent package."""

from __future__ import annotations

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.state import Recommendation

from .chair_report_agent import run_chair_report_agent
from .evidence_audit_agent import run_evidence_audit_agent
from .quant_credit_agent import run_quant_credit_agent


def run_triplet_agents(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    model_name: str,
    max_tokens: int,
) -> tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput]:
    """Run QuantCredit, EvidenceAudit, and ChairReport Agno agents in order."""
    quant_credit = run_quant_credit_agent(
        bundle=bundle,
        model_name=model_name,
        max_tokens=max_tokens,
    )
    evidence_audit = run_evidence_audit_agent(
        bundle=bundle,
        model_name=model_name,
        max_tokens=max_tokens,
    )
    chair_report = run_chair_report_agent(
        bundle=bundle,
        recommendation=recommendation,
        confidence=confidence,
        quant_credit=quant_credit,
        evidence_audit=evidence_audit,
        model_name=model_name,
        max_tokens=max_tokens,
    )
    return quant_credit, evidence_audit, chair_report


__all__ = [
    "run_chair_report_agent",
    "run_evidence_audit_agent",
    "run_quant_credit_agent",
    "run_triplet_agents",
]
