"""LangGraph state schema for the corporate analysis scaffold."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

Recommendation = Literal["priority", "watch", "review", "defer"]
RiskBand = Literal["stable", "watch", "high_risk", "insufficient_data"]
NodeName = Literal[
    "data",
    "feature",
    "feature_store",
    "base_prediction",
    "xgboost_inference",
    "market_overlay",
    "news_overlay",
    "news_cache",
    "rule_engine",
    "committee",
    "agno_agents",
    "json_schema",
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


class ModelResult(BaseModel):
    """Realtime model inference result loaded from the model registry."""

    model_name: str
    model_version: str
    probability_speculative: float = Field(ge=0.0, le=1.0)
    prediction_label: str
    risk_band: RiskBand
    threshold: float
    top_drivers: list[tuple[str, float]] = Field(default_factory=list)


class RuleResult(BaseModel):
    """Rule-engine decision layered on top of model inference."""

    risk_band: RiskBand
    label: str
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    blocking_flags: list[str] = Field(default_factory=list)


class CommitteeReview(BaseModel):
    """One committee perspective reviewing the current candidate."""

    perspective: str
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class AgentOutput(BaseModel):
    """One role-fixed agent output in the Agno-style committee."""

    role: Literal[
        "quant_credit",
        "evidence_audit",
        "chair_report",
    ]
    summary: str
    findings: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


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


def append_agent_outputs(
    current: list[AgentOutput] | None, new: list[AgentOutput] | None
) -> list[AgentOutput]:
    """Append-only reducer for role-fixed agent outputs."""
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
    company_selection: dict[str, Any]
    selection_errors: list[str]

    company_profile: dict[str, Any]
    raw_financials: dict[str, Any]
    source_feature_row: dict[str, Any]
    peer_comparison_rows: list[dict[str, Any]]
    normalized_features: dict[str, float]
    model_features: dict[str, float]
    processed_company: dict[str, Any]
    processed_company_list_ref: str
    feature_store_snapshot: dict[str, Any]
    news_cache_snapshot: dict[str, Any]
    model_registry_ref: dict[str, Any]

    base_assessments: Annotated[dict[str, BaseAssessment], merge_dict]
    market_overlay: OverlayAssessment
    news_overlay: OverlayAssessment
    overall_score: float
    model_view: dict[str, Any]
    xgboost_result: ModelResult
    rule_result: RuleResult

    committee_reviews: Annotated[list[CommitteeReview], append_opinions]
    agent_outputs: Annotated[list[AgentOutput], append_agent_outputs]
    agent_summary: dict[str, Any]
    committee_view: dict[str, Any]
    final_recommendation: Recommendation
    final_confidence: float
    response_json: dict[str, Any]
    json_schema_errors: list[str]

    audit: Annotated[list[AuditEntry], append_audit]
    artifacts: Annotated[dict[str, str], merge_dict]
    insufficient_data: bool


__all__ = [
    "AgentOutput",
    "AgentState",
    "AuditEntry",
    "BaseAssessment",
    "CommitteeReview",
    "ModelResult",
    "NodeName",
    "OverlayAssessment",
    "Recommendation",
    "RiskBand",
    "RuleResult",
    "append_agent_outputs",
    "append_audit",
    "append_opinions",
    "merge_dict",
]
