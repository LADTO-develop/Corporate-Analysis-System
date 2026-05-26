"""Agno-backed RiskRecallQAAgent adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    RiskRecallQAOutput,
)

from .runtime import build_agno_agent, clamp, json_payload, provider_label, run_structured_agent


class AgnoRiskRecallQAResponse(BaseModel):
    """Structured response produced by the Agno RiskRecallQAAgent."""

    model_config = ConfigDict(extra="forbid")

    qa_summary: str = Field(description="Concise Korean summary of the recall QA audit.")
    eligible_safety_assessment: Literal[
        "safe_to_keep_eligible",
        "needs_boundary_review",
        "material_missed_risk",
        "not_applicable",
    ] = Field(description="Whether the eligible final decision appears safe enough.")
    financial_resilience_check: str = Field(
        description="Check liquidity, cash-flow, interest coverage, and capital defenses."
    )
    evidence_recall_check: str = Field(
        description="Check whether external evidence suggests missed downside risk."
    )
    rating_boundary_check: str = Field(
        description="Check model/rating boundary context such as BBB-/BB+ or threshold margin."
    )
    recommended_action: Literal[
        "keep_committee_view",
        "escalate_eligible_to_boundary_hold",
        "escalate_eligible_to_risk_hold",
        "request_manual_review",
        "memo_only_fix",
    ] = Field(description="Advisory recall QA recommendation.")
    confidence: float = Field(ge=0.0, le=1.0)


def run_risk_recall_qa_agent(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
    model_name: str,
    model_provider: str = "openai",
    max_tokens: int,
) -> RiskRecallQAOutput:
    """Run the Agno RiskRecallQAAgent and map it to the CAS Stage 2 schema."""
    model_label = provider_label(model_provider)
    agent = build_agno_agent(
        name=f"{model_label}_RiskRecallQA_Agent",
        model_provider=model_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoRiskRecallQAResponse,
        instructions=[
            f"You are the CAS RiskRecallQAAgent speaking from the {model_label} perspective.",
            "Audit only already-eligible committee decisions for missed-risk recall safety.",
            "Do not rewrite model_view. Treat committee_view as decision-support, not an official rating.",
            "Do not invent external news, DART filings, macro events, or industry events not present in the input.",
            "Escalate to risk_hold only when verified adverse evidence or severe financial stress is present.",
            "Use boundary_hold or manual review for near-threshold uncertainty without confirmed adverse evidence.",
            "If financial defenses and external evidence are adequate, keep committee_view unchanged.",
            "Return concise Korean review prose in the structured response fields only.",
        ],
    )
    result = run_structured_agent(
        agent=agent,
        query=_query(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
        ),
        response_model=AgnoRiskRecallQAResponse,
    )
    return RiskRecallQAOutput(
        qa_summary=_safe_qa_text(result.qa_summary),
        trigger_reasons=trigger_reasons[:5],
        eligible_safety_assessment=result.eligible_safety_assessment,
        financial_resilience_check=_safe_qa_text(result.financial_resilience_check),
        evidence_recall_check=_safe_qa_text(result.evidence_recall_check),
        rating_boundary_check=_safe_qa_text(result.rating_boundary_check),
        recommended_action=result.recommended_action,
        confidence=round(clamp(result.confidence, minimum=0.3, maximum=0.9), 4),
    )


def _query(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
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
            "threshold": bundle.threshold,
            "stage2_secondary_trigger": bundle.model_view.get("stage2_secondary_trigger"),
            "stage2_review_priority": bundle.model_view.get("stage2_review_priority"),
        },
        "source_feature_row": bundle.source_feature_row,
        "prior_rating_reference": bundle.prior_rating_reference,
        "news_cache_snapshot": bundle.news_cache_snapshot,
        "committee_view": committee_view,
        "agent_outputs": {
            "quant_credit": quant_credit.model_dump(mode="json"),
            "evidence_audit": evidence_audit.model_dump(mode="json"),
            "chair_report": chair_report.model_dump(mode="json"),
        },
        "qa_trigger_reasons": trigger_reasons,
        "qa_checks": [
            "final_committee_label must already be eligible",
            "near-threshold eligible decisions need recall safety if financial defenses are weak",
            "repeated financing, guarantee, audit, litigation, suspension, or contract-cancellation evidence needs materiality context",
            "risk_hold requires verified adverse evidence or severe financial stress",
            "boundary_hold is preferred for uncertainty without confirmed adverse evidence",
        ],
    }
    return (
        "Run RiskRecallQAAgent for CAS Stage 2. "
        "Audit the resolved eligible committee_view for missed-risk recall safety. "
        "Return only the AgnoRiskRecallQAResponse fields.\n\n"
        f"{json_payload(prompt_payload)}"
    )


def _safe_qa_text(text: str) -> str:
    cleaned = text.strip()
    replacements = {
        "신용등급을 확정": "신용위험 검토 의견을 정리",
        "등급을 확정": "검토 의견을 정리",
        "최종 승인": "검토 의견",
        "확정합니다": "검토 의견을 제시합니다",
        "승인합니다": "의견을 제시합니다",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


__all__ = ["AgnoRiskRecallQAResponse", "run_risk_recall_qa_agent"]
