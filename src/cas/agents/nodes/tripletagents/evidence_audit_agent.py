"""Agno-backed EvidenceAuditAgent adapter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import EvidenceAuditOutput

from .runtime import (
    build_agno_agent,
    clamp,
    compact_items,
    json_payload,
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
        description="Whether direct and material external evidence creates critical tail risk."
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


def run_evidence_audit_agent(
    *,
    bundle: Stage2InputBundle,
    model_name: str,
    model_provider: str = "anthropic",
    max_tokens: int,
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
        instructions=[
            f"You are the CAS EvidenceAuditAgent, a highly skeptical Corporate Compliance and Risk Investigator (Model: {model_label}).",
            "Your SOLE purpose is to audit non-financial tail risks, external shocks, public disclosures, and management red flags that may not be fully captured in the financial statements.",
            "Do not perform basic financial ratio analysis such as net margin, debt ratio, current ratio, interest coverage, or cash-flow coverage. Leave accounting-ratio interpretation to the QuantCreditAgent.",
            "Focus entirely on: litigation, embezzlement, breach of trust, trading halt, delisting risk, audit opinion issues, going-concern uncertainty, sudden DART filings, refinancing events, liquidity crisis disclosures, regulatory sanctions, and industry-level external shocks.",
            "Rule of Evidence: If an event is not explicitly written in the provided news_cache_snapshot, prior_rating_reference, source_feature_row, or other supplied evidence fields, IT DOES NOT EXIST for this review. Do not hallucinate.",
            "Treat credit_policy_snapshot, if present, only as financial-policy context from the Quant side. It is not news, not a DART filing, not a legal event, and not external evidence.",
            "If the external evidence feed is empty, disabled, not requested, not implemented, missing credentials, or placeholder-only, you must clearly state: '외부 근거 데이터 부재로 인한 검토 불가' and lower your confidence score.",
            "Separate confirmed external facts from evidence limitations. Use explicit labels such as '확인된 외부근거', '미확인 영역', and '검토 한계'.",
            "If you find a critical external risk such as bankruptcy filing, trading halt, delisting procedure, adverse audit opinion, major embezzlement, breach of trust, or court-supervised restructuring, aggressively flag it as a Veto Candidate regardless of how strong the financial ratios look.",
            "If no critical external risk is confirmed, do not imply that the company is safe. Instead, state that no veto-grade external evidence was found within the provided evidence scope.",
            "If external evidence conflicts with the Stage 1 model or QuantCreditAgent-style financial signals, describe the conflict without changing the Stage 1 prediction_label or probability_speculative.",
            "Write in a sharp, fact-based Korean investigative tone. Be concise, skeptical, and explicit about what is confirmed versus what is not available.",
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
            "Agno EvidenceAuditAgent가 외부근거와 꼬리위험 맥락을 검토했습니다. "
            f"외부위험 수준: {result.external_risk_level}. "
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
        evidence_limitations=compact_items(*result.evidence_limitations),
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
        "prior_rating_reference": bundle.prior_rating_reference,
        "source_feature_row": bundle.source_feature_row,
        "news_cache_snapshot": bundle.news_cache_snapshot,
        "evidence_guardrail": {
            "news_status": bundle.news_status,
            "as_of_date": bundle.news_cache_snapshot.get("as_of_date", ""),
            "external_evidence_available": not _external_evidence_unavailable(bundle.news_status),
            "rule_kr": (
                "외부근거가 없거나 비활성화된 상태라면 특정 뉴스, 공시, 업황 사건을 "
                "확인 사실처럼 쓰지 말고 '외부근거 미수집'으로만 판단한다."
            ),
        },
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
    if _external_evidence_unavailable(status):
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
        confidence=0.45,
    )


__all__ = ["AgnoEvidenceAuditResponse", "run_evidence_audit_agent"]
