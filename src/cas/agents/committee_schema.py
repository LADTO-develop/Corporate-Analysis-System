"""Strict schema for Stage 2 committee_view outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

CommitteeLabel = Literal["적격", "보류", "부적격"]
CommitteeDecisionType = Literal[
    "eligible",
    "risk_hold",
    "boundary_hold",
    "mitigation_hold",
    "review_hold",
    "reject",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceSummaryItem(_StrictModel):
    """One evidence item used by the committee view."""

    source: str
    summary: str
    reliability: str


class CommitteeViewPayload(_StrictModel):
    """Final committee-facing decision-support view."""

    final_committee_label: CommitteeLabel
    committee_decision_type: CommitteeDecisionType = "review_hold"
    committee_decision_type_label: str = "확인필요 보류"
    committee_risk_signal: bool = True
    veto_triggered: bool
    hidden_tail_risk_flag: bool = False
    hidden_tail_risk_reason: str = ""
    conflict_resolution: str
    key_risk_factors: list[str]
    mitigating_factors: list[str]
    evidence_summary: list[EvidenceSummaryItem]
    final_review_memo: str


__all__ = [
    "CommitteeDecisionType",
    "CommitteeLabel",
    "CommitteeViewPayload",
    "EvidenceSummaryItem",
]
