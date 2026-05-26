"""Agno-backed ReviewQAAgent adapter."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    ReviewQAOutput,
)

from .runtime import build_agno_agent, clamp, json_payload, provider_label, run_structured_agent


class AgnoReviewQAResponse(BaseModel):
    """Structured response produced by the Agno ReviewQAAgent."""

    model_config = ConfigDict(extra="forbid")

    qa_summary: str = Field(description="Concise Korean summary of the QA audit.")
    label_memo_consistency: str = Field(
        description="Whether final committee label and memo language are consistent."
    )
    risk_hold_assessment: Literal["adequate", "overstated", "not_applicable"] = Field(
        description="Whether a risk_hold subtype is justified by the evidence."
    )
    evidence_cutoff_check: str = Field(
        description="Whether external evidence respects the historical cutoff."
    )
    overhold_guardrail_assessment: str = Field(
        description="Whether the normal-company over-hold guardrail should be considered."
    )
    recommended_action: Literal[
        "keep_committee_view",
        "downgrade_risk_hold_to_boundary_hold",
        "downgrade_reject_to_boundary_hold",
        "request_manual_review",
        "memo_only_fix",
    ] = Field(description="Advisory QA recommendation; do not rewrite committee_view directly.")
    confidence: float = Field(ge=0.0, le=1.0)


def run_review_qa_agent(
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
) -> ReviewQAOutput:
    """Run the Agno ReviewQAAgent and map it to the CAS Stage 2 schema."""
    model_label = provider_label(model_provider)
    agent = build_agno_agent(
        name=f"{model_label}_ReviewQA_Agent",
        model_provider=model_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoReviewQAResponse,
        instructions=[
            f"You are the CAS ReviewQAAgent speaking from the {model_label} perspective.",
            "Audit the already-resolved committee_view; do not rewrite model_view.",
            "Check label/memo consistency, risk_hold subtype quality, evidence cutoff discipline, and normal-company over-hold risk.",
            "Treat your output as advisory QA only. Do not claim an official credit rating decision.",
            "Do not invent external news, DART filings, macro events, or industry events not present in the input.",
            "For historical replay, use only evidence that passes the supplied cutoff context.",
            "Use EvidenceAudit recommended_evidence_treatment before prose when judging whether evidence is watch-context or substantive.",
            "If a single medium financing, procedural halt, or routine audit filing is the only concern, prefer subtype downgrade or manual review over risk escalation.",
            "If final reject relies on model confidence plus financial weakness but external evidence is only routine/caution/watch-context, consider downgrade_reject_to_boundary_hold instead of a hard reject.",
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
        response_model=AgnoReviewQAResponse,
    )
    return ReviewQAOutput(
        qa_summary=_safe_qa_text(result.qa_summary),
        trigger_reasons=trigger_reasons[:5],
        label_memo_consistency=_safe_qa_text(result.label_memo_consistency),
        risk_hold_assessment=result.risk_hold_assessment,
        evidence_cutoff_check=_safe_qa_text(result.evidence_cutoff_check),
        overhold_guardrail_assessment=_safe_qa_text(result.overhold_guardrail_assessment),
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
    prompt_context = bundle.to_compact_prompt_payload(role="review_qa")
    prompt_payload = {
        "company": prompt_context["company"],
        "stage1_model": prompt_context["stage1_model"],
        "source_feature_row": prompt_context["financial_metrics"],
        "news_cache_snapshot": prompt_context["news_cache_snapshot"],
        "materiality_summary": prompt_context["materiality_summary"],
        "committee_view": committee_view,
        "agent_outputs": {
            "quant_credit": quant_credit.model_dump(mode="json"),
            "evidence_audit": evidence_audit.model_dump(mode="json"),
            "chair_report": chair_report.model_dump(mode="json"),
        },
        "qa_trigger_reasons": trigger_reasons,
        "qa_checks": [
            "final_committee_label and final_review_memo must not contradict each other",
            "risk_hold requires verified adverse evidence or severe financial stress",
            "hard reject requires stronger support than routine/caution/watch-context filings",
            "external evidence must respect historical cutoff context",
            "single medium financing, resolved procedural halt, or routine audit filing may support boundary_hold/manual_review instead of risk_hold",
            "normal-company over-hold guardrail should be considered when Stage 1 is investment-grade and severe evidence is absent",
        ],
    }
    return (
        "Run ReviewQAAgent for CAS Stage 2. "
        "Audit the resolved committee_view and return advisory QA only. "
        "Return only the AgnoReviewQAResponse fields.\n\n"
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


__all__ = ["AgnoReviewQAResponse", "run_review_qa_agent"]
