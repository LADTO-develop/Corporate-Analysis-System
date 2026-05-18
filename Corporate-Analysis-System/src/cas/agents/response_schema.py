"""Strict service response schema for dashboard rendering."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.committee_schema import CommitteeViewPayload


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompanyOverview(_StrictModel):
    """Company identity and selection metadata."""

    company_id: str
    company_name: str
    market: str
    analysis_year: int
    summary: str


class Driver(_StrictModel):
    """Model driver shown in the dashboard."""

    name: str
    value: float


class ModelResultPayload(_StrictModel):
    """Realtime XGBoost result plus rule-engine label."""

    model_name: str
    model_version: str
    prediction_label: str
    risk_band: str
    probability_speculative: float = Field(ge=0.0, le=1.0)
    top_drivers: list[Driver]
    rule_label: str


class NewsAnalysisPayload(_StrictModel):
    """Placeholder news/crawling analysis summary."""

    status: str
    summary: str


class AgentRolePayload(_StrictModel):
    """One role-fixed agent's dashboard-safe output."""

    summary: str
    findings: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AgentSummaryPayload(_StrictModel):
    """Synthesized multi-agent explanation payload."""

    final_recommendation: str
    final_confidence: float = Field(ge=0.0, le=1.0)
    synthesis: str
    agents: dict[str, AgentRolePayload]


class DashboardResponse(_StrictModel):
    """Strict JSON emitted to the web dashboard."""

    company_overview: CompanyOverview
    model_result: ModelResultPayload
    news_analysis: NewsAnalysisPayload
    agent_summary: AgentSummaryPayload
    committee_view: CommitteeViewPayload
