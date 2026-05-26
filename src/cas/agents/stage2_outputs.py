"""Agent-specific Stage 2 output schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.state import AgentOutput


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EvidenceTreatment = Literal[
    "context_only",
    "watch_context",
    "substantive_review",
    "critical_veto_review",
]


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
    evidence_limitations: list[str] = Field(default_factory=list)
    critical_evidence_count: int = Field(default=0, ge=0)
    watch_context_count: int = Field(default=0, ge=0)
    materiality_summary: dict[str, Any] = Field(default_factory=dict)
    hard_distress_detected: bool = False
    recommended_evidence_treatment: EvidenceTreatment = "context_only"
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
                (
                    "구조화 근거 판정: "
                    f"critical_evidence_count={self.critical_evidence_count}; "
                    f"watch_context_count={self.watch_context_count}; "
                    f"hard_distress_detected={self.hard_distress_detected}; "
                    f"recommended_evidence_treatment={self.recommended_evidence_treatment}"
                    + _materiality_basis_note(self.materiality_summary)
                ),
                *([f"모델-근거 충돌 점검: {self.model_challenge}"] if self.model_challenge else []),
                *(
                    [f"EvidenceAudit 검토 결론: {self.audit_conclusion}"]
                    if self.audit_conclusion
                    else []
                ),
                *self.debt_liquidity_cross_check,
                *self.macro_industry_sensitivity,
                *self.external_evidence_findings,
                *[f"근거 한계: {item}" for item in self.evidence_limitations],
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


class ReviewQAOutput(_StrictModel):
    """Structured advisory output produced by the optional ReviewQAAgent."""

    role: Literal["review_qa"] = "review_qa"
    qa_summary: str
    trigger_reasons: list[str] = Field(default_factory=list)
    label_memo_consistency: str
    risk_hold_assessment: Literal["adequate", "overstated", "not_applicable"]
    evidence_cutoff_check: str
    overhold_guardrail_assessment: str
    recommended_action: Literal[
        "keep_committee_view",
        "downgrade_risk_hold_to_boundary_hold",
        "downgrade_reject_to_boundary_hold",
        "request_manual_review",
        "memo_only_fix",
    ]
    confidence: float = Field(ge=0.0, le=1.0)

    def to_agent_output(self) -> AgentOutput:
        """Convert the QA output into the common graph payload."""
        return AgentOutput(
            role=self.role,
            summary=self.qa_summary,
            findings=[
                "QA 트리거: "
                + _join_items(
                    self.trigger_reasons,
                    fallback="명시적 QA 트리거는 기록되지 않았습니다.",
                ),
                f"라벨-메모 일관성: {self.label_memo_consistency}",
                f"위험 보류 적정성: {self.risk_hold_assessment}",
                f"외부근거 기준일 점검: {self.evidence_cutoff_check}",
                f"정상기업 과잉 보류 guardrail 점검: {self.overhold_guardrail_assessment}",
                f"QA 권고: {self.recommended_action}",
            ],
            confidence=self.confidence,
        )


class RiskRecallQAOutput(_StrictModel):
    """Structured advisory output produced by the optional RiskRecallQAAgent."""

    role: Literal["risk_recall_qa"] = "risk_recall_qa"
    qa_summary: str
    trigger_reasons: list[str] = Field(default_factory=list)
    eligible_safety_assessment: Literal[
        "safe_to_keep_eligible",
        "needs_boundary_review",
        "material_missed_risk",
        "not_applicable",
    ]
    financial_resilience_check: str
    evidence_recall_check: str
    rating_boundary_check: str
    recommended_action: Literal[
        "keep_committee_view",
        "escalate_eligible_to_boundary_hold",
        "escalate_eligible_to_risk_hold",
        "request_manual_review",
        "memo_only_fix",
    ]
    confidence: float = Field(ge=0.0, le=1.0)

    def to_agent_output(self) -> AgentOutput:
        """Convert the recall QA output into the common graph payload."""
        return AgentOutput(
            role=self.role,
            summary=self.qa_summary,
            findings=[
                "Recall QA 트리거: "
                + _join_items(
                    self.trigger_reasons,
                    fallback="명시적 recall QA 트리거는 기록되지 않았습니다.",
                ),
                f"적격 안전성 평가: {self.eligible_safety_assessment}",
                f"재무 방어축 점검: {self.financial_resilience_check}",
                f"외부근거 누락위험 점검: {self.evidence_recall_check}",
                f"등급/기준선 경계 점검: {self.rating_boundary_check}",
                f"Recall QA 권고: {self.recommended_action}",
            ],
            confidence=self.confidence,
        )


Stage2StructuredOutput = (
    QuantCreditOutput
    | EvidenceAuditOutput
    | ChairReportOutput
    | ReviewQAOutput
    | RiskRecallQAOutput
)


def _join_items(items: list[str], *, fallback: str) -> str:
    values = [item for item in items if item]
    if not values:
        return fallback
    return " / ".join(values[:3])


def _materiality_basis_note(materiality_summary: dict[str, Any]) -> str:
    basis = str(materiality_summary.get("top_materiality_basis") or "").strip()
    if not basis:
        return ""
    return f"; top_materiality_basis={basis}"


__all__ = [
    "ChairReportOutput",
    "EvidenceAuditOutput",
    "EvidenceTreatment",
    "QuantCreditOutput",
    "ReviewQAOutput",
    "RiskRecallQAOutput",
    "Stage2StructuredOutput",
]
