"""Strict schema for Stage 2 committee_view outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CommitteeLabel = Literal["적격", "보류", "부적격"]
CommitteeDecisionType = Literal[
    "eligible",
    "risk_hold",
    "boundary_hold",
    "mitigation_hold",
    "review_hold",
    "reject",
]
RiskHoldReasonTag = Literal[
    "combined_watch_hold",
    "financial_stress_hold",
    "external_materiality_hold",
    "secondary_radar_hold",
    "model_reject_confirmation_hold",
    "model_risk_hold",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceSummaryItem(_StrictModel):
    """One evidence item used by the committee view."""

    source: str
    summary: str
    reliability: str


class DecisionTraceItem(_StrictModel):
    """One deterministic gate check behind the committee decision."""

    gate: str
    label: str
    triggered: bool
    severity: Literal["info", "watch", "risk", "mitigation"] = "info"
    summary: str


class CommitteeViewPayload(_StrictModel):
    """Final committee-facing decision-support view."""

    final_committee_label: CommitteeLabel
    committee_decision_type: CommitteeDecisionType = "review_hold"
    committee_decision_type_label: str = "확인필요 보류"
    committee_risk_signal: bool = True
    risk_hold_reason_tags: list[RiskHoldReasonTag] = Field(default_factory=list)
    risk_hold_reason_labels: list[str] = Field(default_factory=list)
    risk_hold_reason_summary: str = ""
    veto_triggered: bool
    hidden_tail_risk_flag: bool = False
    hidden_tail_risk_reason: str = ""
    conflict_resolution: str
    key_risk_factors: list[str]
    mitigating_factors: list[str]
    evidence_summary: list[EvidenceSummaryItem]
    decision_trace: list[DecisionTraceItem] = Field(default_factory=list)
    final_review_memo: str


__all__ = [
    "CommitteeDecisionType",
    "CommitteeLabel",
    "CommitteeViewPayload",
    "DecisionTraceItem",
    "EvidenceSummaryItem",
    "RiskHoldReasonTag",
]
