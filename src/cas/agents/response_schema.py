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


class Stage2RuntimePayload(_StrictModel):
    """Dashboard-safe Stage 2 execution diagnostics."""

    backend_name: str = ""
    cache_hit: bool = False
    fallback_used: bool = False
    fallback_reason: str = ""
    stage2_total_elapsed_seconds: float | None = None
    agent_elapsed_seconds: dict[str, float] = Field(default_factory=dict)
    agent_elapsed_seconds_sum: float | None = None
    parallel_independent_agents: bool | None = None
    review_qa_triggered: bool = False
    review_qa_trigger_reasons: list[str] = Field(default_factory=list)
    review_qa_recommended_action: str = ""
    review_qa_cache_hit: bool = False
    review_qa_advisory_applied: bool = False
    review_qa_advisory_apply_reason: str = ""
    review_qa_error_message: str = ""
    risk_recall_qa_triggered: bool = False
    risk_recall_qa_trigger_reasons: list[str] = Field(default_factory=list)
    risk_recall_qa_recommended_action: str = ""
    risk_recall_qa_cache_hit: bool = False
    risk_recall_qa_advisory_applied: bool = False
    risk_recall_qa_advisory_apply_reason: str = ""
    risk_recall_qa_error_message: str = ""


class AgentSummaryPayload(_StrictModel):
    """Synthesized multi-agent explanation payload."""

    final_recommendation: str
    final_confidence: float = Field(ge=0.0, le=1.0)
    synthesis: str
    agents: dict[str, AgentRolePayload]
    runtime: Stage2RuntimePayload = Field(default_factory=Stage2RuntimePayload)


class DashboardResponse(_StrictModel):
    """Strict JSON emitted to the web dashboard."""

    company_overview: CompanyOverview
    model_result: ModelResultPayload
    news_analysis: NewsAnalysisPayload
    agent_summary: AgentSummaryPayload
    committee_view: CommitteeViewPayload
