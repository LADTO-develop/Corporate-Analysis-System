"""LangGraph state schema for the corporate analysis scaffold."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

Recommendation = Literal["priority", "watch", "review", "defer"]
NodeName = Literal[
    "data",
    "feature",
    "base_prediction",
    "market_overlay",
    "news_overlay",
    "committee",
    "report",
]


class AuditEntry(BaseModel):
    """A structured audit-trail record emitted by every node."""

    node: NodeName
    timestamp: str
    summary: str
    payload_ref: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class BaseAssessment(BaseModel):
    """Output of a single analysis lens."""

    lens_name: str
    score: float = Field(ge=0.0, le=1.0)
    summary: str
    drivers: list[tuple[str, float]] = Field(default_factory=list)


class OverlayAssessment(BaseModel):
    """Contextual adjustment applied on top of the base score."""

    label: str
    adjustment: float = 0.0
    rationale: str = ""
    signals: dict[str, Any] = Field(default_factory=dict)


class CommitteeReview(BaseModel):
    """One committee perspective reviewing the current candidate."""

    perspective: str
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


def append_audit(
    current: list[AuditEntry] | None, new: list[AuditEntry] | None
) -> list[AuditEntry]:
    """Append-only reducer for the audit log."""
    return [*(current or []), *(new or [])]


def append_opinions(
    current: list[CommitteeReview] | None, new: list[CommitteeReview] | None
) -> list[CommitteeReview]:
    """Append-only reducer for committee reviews."""
    return [*(current or []), *(new or [])]


def merge_dict(current: dict[str, Any] | None, new: dict[str, Any] | None) -> dict[str, Any]:
    """Dict-merge reducer used for artifacts and assessment collections."""
    out: dict[str, Any] = dict(current or {})
    out.update(new or {})
    return out


class AgentState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""

    company_id: str
    company_name: str
    market: str
    analysis_year: int

    company_profile: dict[str, Any]
    raw_financials: dict[str, Any]
    normalized_features: dict[str, float]

    base_assessments: Annotated[dict[str, BaseAssessment], merge_dict]
    market_overlay: OverlayAssessment
    news_overlay: OverlayAssessment
    overall_score: float

    committee_reviews: Annotated[list[CommitteeReview], append_opinions]
    final_recommendation: Recommendation
    final_confidence: float

    audit: Annotated[list[AuditEntry], append_audit]
    artifacts: Annotated[dict[str, str], merge_dict]
    insufficient_data: bool


__all__ = [
    "AgentState",
    "AuditEntry",
    "BaseAssessment",
    "CommitteeReview",
    "merge_dict",
    "NodeName",
    "OverlayAssessment",
    "Recommendation",
    "append_audit",
    "append_opinions",
]
