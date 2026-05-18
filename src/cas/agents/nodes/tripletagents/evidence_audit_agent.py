"""Agno-backed EvidenceAuditAgent adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import EvidenceAuditOutput

from .runtime import build_agno_agent, clamp, compact_items, json_payload, run_structured_agent

_EvidenceStrength = Literal["none", "weak", "moderate", "strong", "critical"]


class AgnoEvidenceAuditResponse(BaseModel):
    """Structured response produced by the Agno EvidenceAuditAgent."""

    model_config = ConfigDict(extra="forbid")

    macro_environmental_impact: str = Field(
        description="Macro, industry, and market sensitivity affecting credit risk."
    )
    critical_off_balance_risk: str = Field(
        description="Critical disclosure, DART, litigation, covenant, or off-balance-sheet risk."
    )
    has_critical_risk: bool = Field(
        description="Whether direct and material external evidence creates critical tail risk."
    )
    external_risk_level: str = Field(
        description="External evidence risk level such as low, medium, high, or equivalent Korean label."
    )


def run_evidence_audit_agent(
    *,
    bundle: Stage2InputBundle,
    model_name: str,
    max_tokens: int,
) -> EvidenceAuditOutput:
    """Run the Agno EvidenceAuditAgent and map it to the CAS Stage 2 schema."""
    agent = build_agno_agent(
        name="EvidenceAudit_Agent",
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoEvidenceAuditResponse,
        instructions=[
            "You are the CAS EvidenceAuditAgent.",
            "Audit external evidence, debt/liquidity context, macro risk, and tail-risk indicators.",
            "Do not invent external evidence; if evidence is missing, say the evidence is pending.",
            "Return concise Korean business review prose in the structured response fields only.",
        ],
    )
    result = run_structured_agent(
        agent=agent,
        query=_query(bundle),
        response_model=AgnoEvidenceAuditResponse,
    )
    strength = _evidence_strength(result=result, status=bundle.news_status)
    model_challenge = _model_challenge(result=result, bundle=bundle, strength=strength)
    return EvidenceAuditOutput(
        evidence_summary=(
            "Agno EvidenceAuditAgent reviewed external evidence and tail-risk context. "
            f"External risk level: {result.external_risk_level}. "
            f"{result.macro_environmental_impact}"
        ),
        evidence_status=bundle.news_status,
        evidence_reliability=_evidence_reliability(result=result, status=bundle.news_status),
        evidence_strength=strength,
        model_challenge=model_challenge,
        audit_conclusion=_audit_conclusion(result=result, strength=strength),
        debt_liquidity_cross_check=compact_items(result.critical_off_balance_risk),
        macro_industry_sensitivity=compact_items(result.macro_environmental_impact),
        external_evidence_findings=compact_items(
            result.critical_off_balance_risk,
            f"External risk level: {result.external_risk_level}",
        ),
        confidence=_confidence_for_strength(strength),
    )


def _query(bundle: Stage2InputBundle) -> str:
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
        },
        "source_feature_row": bundle.source_feature_row,
        "news_cache_snapshot": bundle.news_cache_snapshot,
    }
    return (
        "Run EvidenceAuditAgent for CAS Stage 2. "
        "Focus on external evidence, DART/news context, macro sensitivity, and veto-grade tail risk. "
        "Return only the AgnoEvidenceAuditResponse fields.\n\n"
        f"{json_payload(prompt_payload)}"
    )


def _evidence_strength(
    *,
    result: AgnoEvidenceAuditResponse,
    status: str,
) -> _EvidenceStrength:
    if status in {"disabled", "not_implemented", "placeholder", "missing_credentials"}:
        return "critical" if result.has_critical_risk else "none"
    if result.has_critical_risk:
        return "critical"
    normalized = result.external_risk_level.strip().lower()
    if any(marker in normalized for marker in ("high", "critical", "elevated", "고위험")):
        return "strong"
    if any(marker in normalized for marker in ("medium", "moderate", "중간")):
        return "moderate"
    return "weak"


def _model_challenge(
    *,
    result: AgnoEvidenceAuditResponse,
    bundle: Stage2InputBundle,
    strength: _EvidenceStrength,
) -> str:
    if strength in {"critical", "strong"}:
        return (
            "External evidence may challenge the Stage 1 model view. "
            f"Model label preserved: {bundle.prediction_label}. "
            f"{result.critical_off_balance_risk}"
        )
    return (
        "External evidence does not materially overturn the Stage 1 model view at this stage. "
        f"Model label preserved: {bundle.prediction_label}."
    )


def _audit_conclusion(
    *,
    result: AgnoEvidenceAuditResponse,
    strength: _EvidenceStrength,
) -> str:
    if strength == "critical":
        return f"Critical external evidence requires chair-level review: {result.critical_off_balance_risk}"
    if strength == "strong":
        return f"Strong external evidence should be reflected in committee qualification: {result.critical_off_balance_risk}"
    return "External evidence should be treated as contextual support until stronger direct evidence is available."


def _evidence_reliability(*, result: AgnoEvidenceAuditResponse, status: str) -> str:
    return (
        f"status={status}; has_critical_risk={result.has_critical_risk}; "
        f"external_risk_level={result.external_risk_level}"
    )


def _confidence_for_strength(strength: _EvidenceStrength) -> float:
    return clamp(
        {
            "none": 0.5,
            "weak": 0.6,
            "moderate": 0.7,
            "strong": 0.78,
            "critical": 0.84,
        }[strength],
        minimum=0.35,
        maximum=0.88,
    )


__all__ = ["AgnoEvidenceAuditResponse", "run_evidence_audit_agent"]
