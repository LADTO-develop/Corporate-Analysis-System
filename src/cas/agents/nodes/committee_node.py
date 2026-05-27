"""Compatibility facade for Stage 2 committee orchestration."""

from __future__ import annotations

from typing import Any, cast

from cas.agents.nodes.deterministic_agents import (
    _evidence_audit_agent,
    _quant_credit_agent,
)
from cas.agents.nodes.post_committee_qa import (
    _apply_deterministic_risk_recall_guardrail,
    _apply_review_qa_advisory,
    _apply_risk_recall_qa_advisory,
    _review_qa_trigger_reasons,
    _risk_recall_qa_trigger_reasons,
    _run_review_qa_agent_with_cache,
)
from cas.agents.post_committee_pipeline import validate_agent_order as _validate_agent_order
from cas.agents.stage2_orchestrator import (
    _chair_summary,
    _committee_confidence,
    _recommendation_from_score,
    _stage2_fallback_on_error,
    _stage2_max_tokens,
    _stage2_runner,
    _stage2_runner_name,
    _stage2_runtime_config,
    run_stage2_committee,
    stage2_runner_override,
    stage2_runtime_config_override,
)
from cas.agents.state import AgentState

__all__ = [
    "_apply_review_qa_advisory",
    "_apply_deterministic_risk_recall_guardrail",
    "_apply_risk_recall_qa_advisory",
    "_chair_summary",
    "_committee_confidence",
    "_evidence_audit_agent",
    "_quant_credit_agent",
    "_recommendation_from_score",
    "_review_qa_trigger_reasons",
    "_risk_recall_qa_trigger_reasons",
    "_run_review_qa_agent_with_cache",
    "_stage2_fallback_on_error",
    "_stage2_max_tokens",
    "_stage2_runner",
    "_stage2_runner_name",
    "_stage2_runtime_config",
    "_validate_agent_order",
    "run",
    "stage2_runner_override",
    "stage2_runtime_config_override",
]


def run(state: AgentState) -> dict[str, Any]:
    """Run the Stage 2 committee scaffold through the shared orchestrator."""
    return cast(dict[str, Any], run_stage2_committee(state))
