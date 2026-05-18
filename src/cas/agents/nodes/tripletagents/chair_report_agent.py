"""Agno-backed ChairReportAgent adapter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.state import Recommendation

from .runtime import build_agno_agent, clamp, json_payload, run_structured_agent


class AgnoChairReportResponse(BaseModel):
    """Structured response produced by the Agno ChairReportAgent."""

    model_config = ConfigDict(extra="forbid")

    final_committee_label: str = Field(
        description="Committee-level label or recommendation after reviewing all Stage 2 perspectives."
    )
    veto_triggered: bool = Field(
        description="Whether critical external evidence triggers a veto-style escalation."
    )
    conflict_resolution: str = Field(
        description="How conflicts between model view and external evidence are resolved."
    )
    executive_summary: str = Field(
        description="Final executive summary for the committee report."
    )


def run_chair_report_agent(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    model_name: str,
    max_tokens: int,
) -> ChairReportOutput:
    """Run the Agno ChairReportAgent and map it to the CAS Stage 2 schema."""
    agent = build_agno_agent(
        name="ChairReport_Agent",
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoChairReportResponse,
        instructions=[
            "You are the CAS ChairReportAgent.",
            "Synthesize QuantCreditAgent and EvidenceAuditAgent outputs into committee-ready language.",
            "Preserve the Stage 1 model label and explain any committee qualification separately.",
            "Return concise Korean business review prose in the structured response fields only.",
        ],
    )
    result = run_structured_agent(
        agent=agent,
        query=_query(
            bundle=bundle,
            recommendation=recommendation,
            confidence=confidence,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
        ),
        response_model=AgnoChairReportResponse,
    )
    return ChairReportOutput(
        report_summary=result.executive_summary,
        model_preservation_note=(
            f"Stage 1 model label is preserved as {bundle.prediction_label}; "
            "committee qualification is recorded separately."
        ),
        committee_scope_note=(
            f"Agno chair label={result.final_committee_label}; "
            f"veto_triggered={result.veto_triggered}; recommendation={recommendation}."
        ),
        final_review_memo_seed=result.conflict_resolution,
        confidence=_chair_confidence(
            base_confidence=confidence,
            quant_confidence=quant_credit.confidence,
            evidence_confidence=evidence_audit.confidence,
            veto_triggered=result.veto_triggered,
        ),
    )


def _query(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
) -> str:
    prompt_payload = {
        "company": {
            "company_id": bundle.company_id,
            "company_name": bundle.company_name,
            "market": bundle.market,
            "analysis_year": bundle.analysis_year,
        },
        "stage1_model": {
            "prediction_label": bundle.prediction_label,
            "probability_speculative": bundle.probability_speculative,
            "recommendation": recommendation,
            "confidence": confidence,
        },
        "quant_credit": quant_credit.model_dump(mode="json"),
        "evidence_audit": evidence_audit.model_dump(mode="json"),
    }
    return (
        "Run ChairReportAgent for CAS Stage 2. "
        "Resolve model/evidence conflict and write the final committee synthesis. "
        "Return only the AgnoChairReportResponse fields.\n\n"
        f"{json_payload(prompt_payload)}"
    )


def _chair_confidence(
    *,
    base_confidence: float,
    quant_confidence: float,
    evidence_confidence: float,
    veto_triggered: bool,
) -> float:
    blended = 0.4 * base_confidence + 0.3 * quant_confidence + 0.3 * evidence_confidence
    if veto_triggered:
        blended += 0.04
    return round(clamp(blended, minimum=0.45, maximum=0.9), 4)


__all__ = ["AgnoChairReportResponse", "run_chair_report_agent"]
