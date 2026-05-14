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
