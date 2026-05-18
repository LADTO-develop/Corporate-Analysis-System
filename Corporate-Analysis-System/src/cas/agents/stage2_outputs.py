from pydantic import BaseModel, Field

# 1. Quant 에이전트 스키마
class QuantCreditOutput(BaseModel):
    quant_summary: str = Field(description="기업의 기본 맥락과 XGBoost 예측 확률, 주요 위험/완화 요인을 포함한 한 줄 요약")
    model_rationale: str = Field(description="모델이 왜 이런 판단을 했는지에 대한 요약")
    key_risk_factors: list[str] = Field(description="SHAP 기반 위험 증가 요인 리스트 (보조 신호 포함)")
    mitigating_factors: list[str] = Field(description="SHAP 기반 위험 완화 요인 리스트")
    confidence: float = Field(description="분석 신뢰도 (0.0 ~ 1.0)")

# 2. Evidence 에이전트 스키마
class EvidenceAuditOutput(BaseModel):
    evidence_summary: str = Field(description="뉴스/공시 등 외부 근거와 유동성 신호 점검 결과 요약")
    evidence_status: str = Field(description="외부 뉴스 캐시 상태 (예: active, disabled 등)")
    evidence_reliability: str = Field(description="수집된 근거의 신뢰성, 강도, 검증 여부 설명 텍스트")
    evidence_strength: str = Field(description="외부 근거 강도 (none, weak, moderate, strong, critical)")
    model_challenge: str = Field(description="정량 모델의 판단과 정성적 외부/유동성 판단 사이의 충돌(Challenge) 여부")
    audit_conclusion: str = Field(description="증거 분석 최종 결론 및 권고사항")
    debt_liquidity_cross_check: list[str] = Field(description="부채/유동성 관련 세부 발견 사항")
    macro_industry_sensitivity: list[str] = Field(description="거시경제 및 산업 민감도 발견 사항")
    external_evidence_findings: list[str] = Field(description="외부 뉴스/공시 관련 직접 발견 사항")
    confidence: float = Field(description="분석 신뢰도 (0.0 ~ 1.0)")

# 3. Chair(의장) 에이전트 스키마
class ChairReportOutput(BaseModel):
    report_summary: str = Field(description="위원회의 최종 평가 및 권고안 요약")
    model_preservation_note: str = Field(description="Stage 1 모델 판단 보존 여부에 대한 안내문")
    committee_scope_note: str = Field(description="최종 위원회 심사 범위 및 기준 안내문")
    final_review_memo_seed: str = Field(description="심사역이 읽게 될 최종 메모의 초안(Seed)")
    confidence: float = Field(description="위원장 최종 신뢰도 (0.0 ~ 1.0)")

"""Agent-specific Stage 2 output schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.state import AgentOutput


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QuantCreditOutput(_StrictModel):
    """Structured output produced by QuantCreditAgent before dashboard flattening."""

    role: Literal["quant_credit"] = "quant_credit"
    quant_summary: str
    model_rationale: str
    key_risk_factors: list[str]
    mitigating_factors: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    def to_agent_output(self) -> AgentOutput:
        """Convert the structured output into the common graph payload."""
        return AgentOutput(
            role=self.role,
            summary=self.quant_summary,
            findings=[
                f"정량 해석 요약: {self.model_rationale}",
                "핵심 위험 요인: "
                + _join_items(
                    self.key_risk_factors,
                    fallback="상위 변수 기준 뚜렷한 위험 가중 요인은 제한적입니다.",
                ),
                "완화 요인: "
                + _join_items(
                    self.mitigating_factors,
                    fallback="상위 변수 기준 완화 요인은 제한적입니다.",
                ),
            ],
            confidence=self.confidence,
        )


class EvidenceAuditOutput(_StrictModel):
    """Structured output produced by EvidenceAuditAgent before dashboard flattening."""

    role: Literal["evidence_audit"] = "evidence_audit"
    evidence_summary: str
    evidence_status: str
    evidence_reliability: str
    evidence_strength: Literal["none", "weak", "moderate", "strong", "critical"]
    model_challenge: str
    audit_conclusion: str
    debt_liquidity_cross_check: list[str]
    macro_industry_sensitivity: list[str]
    external_evidence_findings: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    def to_agent_output(self) -> AgentOutput:
        """Convert the structured output into the common graph payload."""
        return AgentOutput(
            role=self.role,
            summary=self.evidence_summary,
            findings=[
                f"외부 근거 상태: 현재 뉴스/공시 근거 번들 상태는 `{self.evidence_status}`입니다.",
                f"근거 검증 원칙: {self.evidence_reliability}",
                f"외부근거 강도: {self.evidence_strength}",
                *([f"모델-근거 충돌 점검: {self.model_challenge}"] if self.model_challenge else []),
                *(
                    [f"EvidenceAudit 검토 결론: {self.audit_conclusion}"]
                    if self.audit_conclusion
                    else []
                ),
                *self.debt_liquidity_cross_check,
                *self.macro_industry_sensitivity,
                *self.external_evidence_findings,
            ],
            confidence=self.confidence,
        )


class ChairReportOutput(_StrictModel):
    """Structured output produced by ChairReportAgent before dashboard flattening."""

    role: Literal["chair_report"] = "chair_report"
    report_summary: str
    model_preservation_note: str
    committee_scope_note: str
    final_review_memo_seed: str
    confidence: float = Field(ge=0.0, le=1.0)

    def to_agent_output(self) -> AgentOutput:
        """Convert the structured output into the common graph payload."""
        return AgentOutput(
            role=self.role,
            summary=self.report_summary,
            findings=[
                self.model_preservation_note,
                self.committee_scope_note,
                self.final_review_memo_seed,
            ],
            confidence=self.confidence,
        )


Stage2StructuredOutput = QuantCreditOutput | EvidenceAuditOutput | ChairReportOutput


def _join_items(items: list[str], *, fallback: str) -> str:
    values = [item for item in items if item]
    if not values:
        return fallback
    return " / ".join(values[:3])


__all__ = [
    "ChairReportOutput",
    "EvidenceAuditOutput",
    "QuantCreditOutput",
    "Stage2StructuredOutput",
]
