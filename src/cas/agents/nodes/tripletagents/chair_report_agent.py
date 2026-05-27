"""Agno-backed ChairReportAgent adapter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.stage2_prompt_contracts import (
    build_stage2_role_instructions,
    build_stage2_role_query,
)
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.agents.state import Recommendation

from .runtime import build_agno_agent, clamp, provider_label, run_structured_agent


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
    executive_summary: str = Field(description="Final executive summary for the committee report.")


def run_chair_report_agent(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    model_name: str,
    model_provider: str = "openai",
    max_tokens: int,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> ChairReportOutput:
    """Run the Agno ChairReportAgent and map it to the CAS Stage 2 schema."""
    model_label = provider_label(model_provider)
    agent = build_agno_agent(
        name=f"{model_label}_ChairReport_Agent",
        model_provider=model_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoChairReportResponse,
        runtime_config=runtime_config,
        instructions=build_stage2_role_instructions(
            "chair_report",
            provider_label=model_label,
        ),
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
        runtime_config=runtime_config,
    )
    report_summary = _safe_committee_text(result.executive_summary)
    conflict_resolution = _safe_committee_text(result.conflict_resolution)
    return ChairReportOutput(
        report_summary=report_summary,
        model_preservation_note=(
            f"Stage 1 모델 라벨은 {bundle.prediction_label}으로 보존하며, "
            "위원회 검토 의견은 별도로 기록합니다."
        ),
        committee_scope_note=(
            f"Agno {model_label} chair label={result.final_committee_label}; "
            f"veto_triggered={result.veto_triggered}; recommendation={recommendation}."
        ),
        final_review_memo_seed=conflict_resolution,
        confidence=_chair_confidence(
            base_confidence=confidence,
            quant_confidence=quant_credit.confidence,
            evidence_confidence=evidence_audit.confidence,
            veto_triggered=result.veto_triggered,
        ),
    )


def _policy_summary(bundle: Stage2InputBundle) -> dict[str, object]:
    snapshot = bundle.credit_policy_snapshot
    return {
        "policy_version": snapshot.get("policy_version"),
        "label_override_allowed": snapshot.get("label_override_allowed", False),
        "risk_signal_count": snapshot.get("risk_signal_count", 0),
        "mitigating_signal_count": snapshot.get("mitigating_signal_count", 0),
        "critical_signal_count": snapshot.get("critical_signal_count", 0),
    }


def _query(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
) -> str:
    prompt_context = bundle.to_compact_prompt_payload(role="chair_report")
    prompt_payload = {
        "company": prompt_context["company"],
        "stage1_model": prompt_context["stage1_model"],
        "prior_rating_reference": prompt_context["prior_rating_reference"],
        "rule_engine": {
            "recommendation": recommendation,
            "rule_engine_confidence": confidence,
            "confidence_explanation_kr": (
                "이 값은 규칙엔진/위원회 검토 보조 신뢰도이며, XGBoost 모델 확률이나 "
                "공식 신용등급 신뢰도가 아니다."
            ),
        },
        "quant_credit": quant_credit.model_dump(mode="json"),
        "evidence_audit": evidence_audit.model_dump(mode="json"),
        "materiality_summary": prompt_context["materiality_summary"],
        "credit_policy_summary": _policy_summary(bundle),
    }
    return build_stage2_role_query(
        "chair_report",
        prompt_payload=prompt_payload,
    )


def _safe_committee_text(text: str) -> str:
    """Soften official-rating language and clean common Korean josa/punctuation issues."""
    cleaned = text.strip()
    replacements = {
        "투자적격 등급을 확정합니다": "투자적격 검토 의견을 제시합니다",
        "부적격 등급을 확정합니다": "부적격 검토 의견을 제시합니다",
        "신용등급을 확정합니다": "신용위험 검토 의견을 제시합니다",
        "등급을 확정합니다": "검토 의견을 제시합니다",
        "최종 승인합니다": "검토 의견으로 정리합니다",
        "최종 승인": "검토 의견",
        "확정합니다": "검토 의견을 제시합니다",
        "승인합니다": "의견을 제시합니다",
        "적격로": "적격으로",
        "부적격로": "부적격으로",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    return cleaned


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
