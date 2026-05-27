"""Agno-backed RiskRecallQAAgent adapter."""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    RiskRecallQAOutput,
)
from cas.agents.stage2_prompt_contracts import (
    build_stage2_role_instructions,
    build_stage2_role_query,
)
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig

from .runtime import build_agno_agent, clamp, provider_label, run_structured_agent


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
    runtime_config: Stage2RuntimeConfig | None = None,
) -> RiskRecallQAOutput:
    """Run the Agno RiskRecallQAAgent and map it to the CAS Stage 2 schema."""
    model_label = provider_label(model_provider)
    agent = build_agno_agent(
        name=f"{model_label}_RiskRecallQA_Agent",
        model_provider=model_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoRiskRecallQAResponse,
        runtime_config=runtime_config,
        instructions=build_stage2_role_instructions(
            "risk_recall_qa",
            provider_label=model_label,
        ),
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
        runtime_config=runtime_config,
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
    prompt_context = bundle.to_compact_prompt_payload(role="risk_recall_qa")
    prompt_payload = {
        "company": prompt_context["company"],
        "stage1_model": prompt_context["stage1_model"],
        "source_feature_row": prompt_context["financial_metrics"],
        "prior_rating_reference": prompt_context["prior_rating_reference"],
        "news_cache_snapshot": prompt_context["news_cache_snapshot"],
        "materiality_summary": prompt_context["materiality_summary"],
        "committee_view": committee_view,
        "agent_outputs": {
            "quant_credit": quant_credit.model_dump(mode="json"),
            "evidence_audit": evidence_audit.model_dump(mode="json"),
            "chair_report": chair_report.model_dump(mode="json"),
        },
        "qa_trigger_reasons": trigger_reasons,
    }
    return cast(str, build_stage2_role_query("risk_recall_qa", prompt_payload=prompt_payload))


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
