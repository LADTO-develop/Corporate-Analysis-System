"""Agno-backed EvidenceAuditAgent adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.signals.evidence_treatment_signals import (
    EvidenceTreatmentSignals,
    evaluate_evidence_treatment,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import EvidenceAuditOutput, EvidenceTreatment
from cas.agents.stage2_prompt_contracts import (
    build_stage2_role_instructions,
    build_stage2_role_query,
)
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig

from .runtime import (
    build_agno_agent,
    clamp,
    compact_items,
    provider_label,
    run_structured_agent,
)

_EvidenceStrength = Literal["none", "weak", "moderate", "strong", "critical"]
_UNAVAILABLE_EVIDENCE_STATUSES = {
    "disabled",
    "missing_credentials",
    "not_implemented",
    "not_requested",
    "placeholder",
}


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
        description=(
            "LLM-proposed critical-risk flag. Final criticality is gated by "
            "structured_evidence_decision, not this flag alone."
        )
    )
    external_risk_level: str = Field(
        description="External evidence risk level such as low, medium, high, or equivalent Korean label."
    )
    evidence_limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence-bundle limitations such as missing providers, date filters, "
            "weak company relevance, or unverified snippets."
        ),
    )
    critical_evidence_count: int = Field(
        default=0,
        ge=0,
        description="Count of direct external items that should be treated as substantive or critical.",
    )
    watch_context_count: int = Field(
        default=0,
        ge=0,
        description="Count of direct external items that are watch/context rather than critical.",
    )
    hard_distress_detected: bool = Field(
        default=False,
        description="Whether hard distress terms such as delisting, insolvency, fraud, or default are detected.",
    )
    recommended_evidence_treatment: EvidenceTreatment = Field(
        default="context_only",
        description=(
            "Recommended treatment: context_only, watch_context, substantive_review, "
            "or critical_veto_review."
        ),
    )


def run_evidence_audit_agent(
    *,
    bundle: Stage2InputBundle,
    model_name: str,
    model_provider: str = "openai",
    max_tokens: int,
    runtime_config: Stage2RuntimeConfig | None = None,
) -> EvidenceAuditOutput:
    """Run the Agno EvidenceAuditAgent and map it to the CAS Stage 2 schema."""
    if _external_evidence_unavailable(bundle.news_status):
        return _unavailable_evidence_output(bundle)

    model_label = provider_label(model_provider)
    agent = build_agno_agent(
        name=f"{model_label}_EvidenceAudit_Agent",
        model_provider=model_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoEvidenceAuditResponse,
        runtime_config=runtime_config,
        instructions=build_stage2_role_instructions(
            "evidence_audit",
            provider_label=model_label,
        ),
    )
    result = run_structured_agent(
        agent=agent,
        query=_query(bundle),
        response_model=AgnoEvidenceAuditResponse,
        runtime_config=runtime_config,
    )
    prompt_context = bundle.to_compact_prompt_payload(role="evidence_audit")
    treatment = evaluate_evidence_treatment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
        materiality_summary=prompt_context["materiality_summary"],
    )
    strength = _evidence_strength(
        result=result,
        status=bundle.news_status,
        treatment=treatment,
    )
    model_challenge = _model_challenge(result=result, bundle=bundle, strength=strength)
    return EvidenceAuditOutput(
        evidence_summary=(
            "Agno EvidenceAuditAgent가 외부근거와 꼬리위험 맥락을 검토했습니다. "
            f"외부위험 수준: {result.external_risk_level}. "
            f"{result.macro_environmental_impact}"
        ),
        evidence_status=bundle.news_status,
        evidence_reliability=_evidence_reliability(
            result=result,
            status=bundle.news_status,
            treatment=treatment,
        ),
        evidence_strength=strength,
        model_challenge=model_challenge,
        audit_conclusion=_audit_conclusion(result=result, strength=strength),
        debt_liquidity_cross_check=compact_items(result.critical_off_balance_risk),
        macro_industry_sensitivity=compact_items(result.macro_environmental_impact),
        external_evidence_findings=compact_items(
            result.critical_off_balance_risk,
            f"External risk level: {result.external_risk_level}",
            (
                "Structured evidence treatment: "
                f"{treatment.recommended_evidence_treatment}; "
                f"critical={treatment.critical_evidence_count}; "
                f"watch={treatment.watch_context_count}"
            ),
        ),
        evidence_limitations=compact_items(*result.evidence_limitations),
        critical_evidence_count=treatment.critical_evidence_count,
        watch_context_count=treatment.watch_context_count,
        materiality_summary=treatment.materiality_summary,
        hard_distress_detected=treatment.hard_distress_detected,
        recommended_evidence_treatment=treatment.recommended_evidence_treatment,
        confidence=_confidence_for_strength(strength),
    )


def _query(bundle: Stage2InputBundle) -> str:
    prompt_context = bundle.to_compact_prompt_payload(role="evidence_audit")
    compact_news = prompt_context["news_cache_snapshot"]
    treatment = evaluate_evidence_treatment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
        materiality_summary=prompt_context["materiality_summary"],
    )
    prompt_payload = {
        "company": prompt_context["company"],
        "stage1_model": prompt_context["stage1_model"],
        "prior_rating_reference": prompt_context["prior_rating_reference"],
        "source_feature_row": prompt_context["financial_metrics"],
        "news_cache_snapshot": compact_news,
        "materiality_summary": prompt_context["materiality_summary"],
        "structured_evidence_decision": treatment.as_payload(),
        "evidence_guardrail": {
            "news_status": bundle.news_status,
            "as_of_date": compact_news.get("as_of_date", ""),
            "external_evidence_available": not _external_evidence_unavailable(bundle.news_status),
            "disclosure_calibration_rule_kr": (
                "공시가 caution/procedural_or_one_off/routine_context로 분류된 경우에는 "
                "그 자체만으로 실질 부실 또는 tail risk로 확정하지 않는다. "
                "materiality_basis가 있으면 자금조달/채무보증/소송/계약해지/영업정지의 "
                "기업 규모 대비 중요도를 우선 반영하고, dilution_basis가 있으면 희석률도 함께 본다. "
                "adverse/veto, 반복 공시, 미해소 사건, 재무 차단 신호와 결합될 때만 "
                "보수적 재검토 신호로 강화한다."
            ),
            "structured_output_rule_kr": (
                "critical_evidence_count, watch_context_count, hard_distress_detected, "
                "recommended_evidence_treatment는 structured_evidence_decision을 기준으로 "
                "일관되게 채운다. watch_context는 위험 확정이 아니라 관찰/설명 보완으로 둔다."
            ),
            "rule_kr": (
                "외부근거가 없거나 비활성화된 상태라면 특정 뉴스, 공시, 업황 사건을 "
                "확인 사실처럼 쓰지 말고 '외부근거 미수집'으로만 판단한다."
            ),
        },
    }
    return build_stage2_role_query(
        "evidence_audit",
        prompt_payload=prompt_payload,
    )


def _evidence_strength(
    *,
    result: AgnoEvidenceAuditResponse,
    status: str,
    treatment: EvidenceTreatmentSignals,
) -> _EvidenceStrength:
    if _external_evidence_unavailable(status):
        return "critical" if _has_structured_critical_evidence(treatment) else "none"
    if _has_structured_critical_evidence(treatment):
        return "critical"
    normalized = result.external_risk_level.strip().lower()
    if any(marker in normalized for marker in ("high", "critical", "elevated", "고위험")):
        return "strong"
    if any(marker in normalized for marker in ("medium", "moderate", "중간")):
        return "moderate"
    return "weak"


def _has_structured_critical_evidence(treatment: EvidenceTreatmentSignals) -> bool:
    return bool(
        treatment.recommended_evidence_treatment == "critical_veto_review"
        or treatment.hard_distress_detected
        or treatment.critical_evidence_count > 0
    )


def _model_challenge(
    *,
    result: AgnoEvidenceAuditResponse,
    bundle: Stage2InputBundle,
    strength: _EvidenceStrength,
) -> str:
    if strength in {"critical", "strong"}:
        return (
            "외부근거가 Stage 1 모델 판단에 보수적 재검토 신호를 줄 수 있습니다. "
            f"모델 라벨은 {bundle.prediction_label}으로 보존합니다. "
            f"{result.critical_off_balance_risk}"
        )
    return (
        "현재 확인된 외부근거만으로는 Stage 1 모델 판단을 실질적으로 뒤집기 어렵습니다. "
        f"모델 라벨은 {bundle.prediction_label}으로 보존합니다."
    )


def _audit_conclusion(
    *,
    result: AgnoEvidenceAuditResponse,
    strength: _EvidenceStrength,
) -> str:
    if strength == "critical":
        return f"치명적 외부근거가 있어 위원장 단계 검토가 필요합니다: {result.critical_off_balance_risk}"
    if strength == "strong":
        return f"강한 외부근거를 위원회 보수 의견에 반영해야 합니다: {result.critical_off_balance_risk}"
    return "더 강한 직접 외부근거가 확보되기 전까지는 참고 맥락으로만 처리합니다."


def _evidence_reliability(
    *,
    result: AgnoEvidenceAuditResponse,
    status: str,
    treatment: EvidenceTreatmentSignals,
) -> str:
    return (
        f"status={status}; has_critical_risk={result.has_critical_risk}; "
        f"external_risk_level={result.external_risk_level}; "
        f"structured_critical_evidence={_has_structured_critical_evidence(treatment)}"
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


def _external_evidence_unavailable(status: str) -> bool:
    """Return whether no external evidence was collected for this run."""
    return status.strip().lower() in _UNAVAILABLE_EVIDENCE_STATUSES


def _unavailable_evidence_output(bundle: Stage2InputBundle) -> EvidenceAuditOutput:
    """Return a guarded output when news/DART evidence collection is not active."""
    status = bundle.news_status
    summary = (
        "외부 뉴스·공시 근거 수집이 비활성화되어 확인된 외부 사건 기반 판단은 수행하지 "
        f"않았습니다. 현재 news_status는 `{status}`입니다."
    )
    return EvidenceAuditOutput(
        evidence_summary=summary,
        evidence_status=status,
        evidence_reliability=f"status={status}; 외부근거 미수집",
        evidence_strength="none",
        model_challenge=(
            "외부근거 미수집 상태이므로 Stage 1 모델 판단을 뒤집을 확인 근거는 없습니다. "
            f"모델 라벨은 {bundle.prediction_label}으로 보존합니다."
        ),
        audit_conclusion="뉴스·공시·DART 수집을 활성화한 뒤 외부 리스크를 재검토해야 합니다.",
        debt_liquidity_cross_check=[
            "외부근거 미수집으로 부채·유동성 관련 외부 교차검증은 보류합니다."
        ],
        macro_industry_sensitivity=[
            "거시·산업 관련 외부근거가 제공되지 않아 정성 판단은 제한적입니다."
        ],
        external_evidence_findings=["확인된 외부 뉴스·공시 항목 없음"],
        evidence_limitations=[f"외부근거 수집 상태가 `{status}`라서 확인 가능한 근거가 없습니다."],
        critical_evidence_count=0,
        watch_context_count=0,
        materiality_summary=evaluate_evidence_treatment(
            bundle.news_cache_snapshot,
            source_feature_row=bundle.source_feature_row,
        ).materiality_summary,
        hard_distress_detected=False,
        recommended_evidence_treatment="context_only",
        confidence=0.45,
    )


__all__ = ["AgnoEvidenceAuditResponse", "run_evidence_audit_agent"]
