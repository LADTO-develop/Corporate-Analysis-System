"""Post-committee QA orchestration for Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cas.agents.nodes.committee_diagnostics import _attach_agent_disagreement
from cas.agents.nodes.post_committee_qa import (
    _apply_deterministic_risk_recall_guardrail,
    _apply_review_qa_advisory,
    _apply_risk_recall_qa_advisory,
    _maybe_run_review_qa,
    _maybe_run_risk_recall_qa,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    ReviewQAOutput,
    RiskRecallQAOutput,
)
from cas.agents.stage2_specs import STAGE2_AGENT_ROLES
from cas.agents.state import AgentOutput

Stage2StructuredTriplet = tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput]


@dataclass(frozen=True)
class PostCommitteePipelineResult:
    """Outputs after optional post-committee QA agents have run."""

    committee_view: dict[str, Any]
    agents: list[AgentOutput]
    runtime_diagnostics: dict[str, Any]


def run_post_committee_pipeline(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    structured_outputs: Stage2StructuredTriplet,
    runtime_backend_name: str,
    runtime_diagnostics: dict[str, Any],
) -> PostCommitteePipelineResult:
    """Attach diagnostics and run optional ReviewQA/RiskRecallQA agents."""
    diagnostics = runtime_diagnostics
    agents = [output.to_agent_output() for output in structured_outputs]
    committee_view = _attach_agent_disagreement(
        bundle=bundle,
        committee_view=committee_view,
        structured_outputs=structured_outputs,
        runtime_diagnostics=diagnostics,
    )

    review_qa_output = _maybe_run_review_qa(
        bundle=bundle,
        committee_view=committee_view,
        structured_outputs=structured_outputs,
        runtime_backend_name=runtime_backend_name,
        runtime_diagnostics=diagnostics,
    )
    if review_qa_output is not None:
        committee_view = _apply_review_qa_advisory(
            committee_view=committee_view,
            review_qa_output=review_qa_output,
            bundle=bundle,
            news_cache_snapshot=bundle.news_cache_snapshot,
            runtime_diagnostics=diagnostics,
        )
        committee_view = _attach_qa_action_plan(
            committee_view=committee_view,
            qa_output=review_qa_output,
        )
        committee_view = _attach_agent_disagreement(
            bundle=bundle,
            committee_view=committee_view,
            structured_outputs=structured_outputs,
            runtime_diagnostics=diagnostics,
        )
        agents.append(review_qa_output.to_agent_output())

    risk_recall_qa_output = _maybe_run_risk_recall_qa(
        bundle=bundle,
        committee_view=committee_view,
        structured_outputs=structured_outputs,
        runtime_backend_name=runtime_backend_name,
        runtime_diagnostics=diagnostics,
    )
    if risk_recall_qa_output is not None:
        committee_view = _apply_risk_recall_qa_advisory(
            committee_view=committee_view,
            risk_recall_qa_output=risk_recall_qa_output,
            bundle=bundle,
            runtime_diagnostics=diagnostics,
        )
        committee_view = _attach_qa_action_plan(
            committee_view=committee_view,
            qa_output=risk_recall_qa_output,
        )
        committee_view = _attach_agent_disagreement(
            bundle=bundle,
            committee_view=committee_view,
            structured_outputs=structured_outputs,
            runtime_diagnostics=diagnostics,
        )
        agents.append(risk_recall_qa_output.to_agent_output())

    committee_view = _apply_deterministic_risk_recall_guardrail(
        committee_view=committee_view,
        bundle=bundle,
        runtime_diagnostics=diagnostics,
    )
    if diagnostics.get("risk_recall_guardrail_applied") is True:
        committee_view = _attach_agent_disagreement(
            bundle=bundle,
            committee_view=committee_view,
            structured_outputs=structured_outputs,
            runtime_diagnostics=diagnostics,
        )

    validate_agent_order(agents)
    return PostCommitteePipelineResult(
        committee_view=committee_view,
        agents=agents,
        runtime_diagnostics=diagnostics,
    )


def validate_agent_order(agents: list[AgentOutput]) -> None:
    """Validate fixed Stage 2 role order with optional QA agents at the end."""
    actual_roles = tuple(agent.role for agent in agents)
    required_count = len(STAGE2_AGENT_ROLES)
    optional_suffix = actual_roles[required_count:]
    allowed_suffixes = {
        (),
        ("review_qa",),
        ("risk_recall_qa",),
        ("review_qa", "risk_recall_qa"),
    }
    if (
        actual_roles[:required_count] != STAGE2_AGENT_ROLES
        or optional_suffix not in allowed_suffixes
    ):
        expected = ", ".join(STAGE2_AGENT_ROLES)
        actual = ", ".join(actual_roles)
        raise ValueError(
            "Stage 2 agent order mismatch: "
            f"expected {expected} with optional QA agents at the end, got {actual}"
        )


def _attach_qa_action_plan(
    *,
    committee_view: dict[str, Any],
    qa_output: ReviewQAOutput | RiskRecallQAOutput,
) -> dict[str, Any]:
    updated = dict(committee_view)
    for field in ("manual_review_tasks", "missing_evidence", "monitoring_triggers"):
        updated[field] = _merge_text_items(
            list(updated.get(field, []) or []),
            list(getattr(qa_output, field, []) or []),
        )
    if qa_output.recommended_action == "request_manual_review":
        updated["manual_review_tasks"] = _merge_text_items(
            list(updated.get("manual_review_tasks", []) or []),
            [
                "QA 권고가 수동 검토를 요청했으므로 보류 사유, 근거 누락, 모니터링 조건을 담당자가 확정합니다."
            ],
        )
    return updated


def _merge_text_items(left: list[object], right: list[object], *, limit: int = 8) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in [*left, *right]:
        text = str(item).strip()
        if not text or text in seen:
            continue
        output.append(text)
        seen.add(text)
        if len(output) >= limit:
            break
    return output


__all__ = [
    "PostCommitteePipelineResult",
    "Stage2StructuredTriplet",
    "run_post_committee_pipeline",
    "validate_agent_order",
]
