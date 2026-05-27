"""Role contracts for the Stage 2 three-agent review scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Stage2AgentRole = Literal[
    "quant_credit",
    "evidence_audit",
    "chair_report",
    "review_qa",
    "risk_recall_qa",
]


@dataclass(frozen=True)
class Stage2AgentSpec:
    """Provider-neutral role contract that can later be mapped to Agno agents."""

    role: Stage2AgentRole
    display_name: str
    purpose: str
    required_inputs: tuple[str, ...]
    output_fields: tuple[str, ...]
    future_agno_instruction: str


STAGE2_AGENT_ROLES: tuple[Stage2AgentRole, ...] = (
    "quant_credit",
    "evidence_audit",
    "chair_report",
)

STAGE2_OPTIONAL_AGENT_ROLES: tuple[Stage2AgentRole, ...] = (
    "review_qa",
    "risk_recall_qa",
)

STAGE2_AGENT_SPECS: tuple[Stage2AgentSpec, ...] = (
    Stage2AgentSpec(
        role="quant_credit",
        display_name="QuantCreditAgent",
        purpose=(
            "Explain the immutable Stage 1 XGBoost decision using SHAP drivers, "
            "core financial values, and peer context."
        ),
        required_inputs=(
            "model_view",
            "xgboost_result",
            "source_feature_row",
            "peer_comparison_rows",
        ),
        output_fields=(
            "quant_summary",
            "model_rationale",
            "key_risk_factors",
            "mitigating_factors",
            "confidence",
        ),
        future_agno_instruction=(
            "Treat model_view as read-only. Explain why the model reached its "
            "label, connect SHAP drivers to raw metrics, and avoid changing the "
            "model prediction."
        ),
    ),
    Stage2AgentSpec(
        role="evidence_audit",
        display_name="EvidenceAuditAgent",
        purpose=(
            "Combine external evidence, macro/industry context, and debt-liquidity "
            "signals to surface hidden tail risks or verified mitigants."
        ),
        required_inputs=(
            "model_view",
            "news_cache_snapshot",
            "source_feature_row",
            "company_profile",
            "macro_market_context",
        ),
        output_fields=(
            "evidence_summary",
            "evidence_status",
            "evidence_reliability",
            "evidence_strength",
            "model_challenge",
            "audit_conclusion",
            "debt_liquidity_cross_check",
            "macro_industry_sensitivity",
            "external_evidence_findings",
            "evidence_limitations",
            "critical_evidence_count",
            "watch_context_count",
            "materiality_summary",
            "hard_distress_detected",
            "recommended_evidence_treatment",
            "confidence",
        ),
        future_agno_instruction=(
            "Audit evidence quality before reasoning from it. Separate verified "
            "facts from weak signals, and flag veto-grade events such as fraud, "
            "delisting risk, or severe liquidity stress."
        ),
    ),
    Stage2AgentSpec(
        role="chair_report",
        display_name="ChairReportAgent",
        purpose=(
            "Synthesize the quantitative explanation and evidence audit into a "
            "committee_view without overwriting model_view."
        ),
        required_inputs=(
            "quant_credit_output",
            "evidence_audit_output",
            "model_view",
            "rule_result",
        ),
        output_fields=(
            "final_committee_label",
            "committee_decision_type",
            "committee_decision_type_label",
            "committee_risk_signal",
            "risk_hold_reason_tags",
            "risk_hold_reason_labels",
            "risk_hold_reason_summary",
            "agent_disagreement_score",
            "agent_disagreement_level",
            "agent_disagreement_reasons",
            "agent_disagreement_summary",
            "veto_triggered",
            "hidden_tail_risk_flag",
            "hidden_tail_risk_reason",
            "conflict_resolution",
            "key_risk_factors",
            "mitigating_factors",
            "evidence_summary",
            "decision_trace",
            "manual_review_tasks",
            "missing_evidence",
            "monitoring_triggers",
            "final_review_memo",
        ),
        future_agno_instruction=(
            "Resolve conflicts conservatively, explain any gap between model_view "
            "and committee_view, avoid turning low-absolute-risk near-threshold "
            "cases into holds without severe financial or verified evidence support, "
            "and emit the strict committee_view fields."
        ),
    ),
)

STAGE2_OPTIONAL_AGENT_SPECS: tuple[Stage2AgentSpec, ...] = (
    Stage2AgentSpec(
        role="review_qa",
        display_name="ReviewQAAgent",
        purpose=(
            "Audit the resolved committee_view for label/memo consistency, evidence "
            "cutoff discipline, over-hold risk, and risk-hold subtype quality."
        ),
        required_inputs=(
            "committee_view",
            "quant_credit_output",
            "evidence_audit_output",
            "chair_report_output",
            "model_view",
            "news_cache_snapshot",
        ),
        output_fields=(
            "qa_summary",
            "trigger_reasons",
            "label_memo_consistency",
            "risk_hold_assessment",
            "evidence_cutoff_check",
            "overhold_guardrail_assessment",
            "recommended_action",
            "manual_review_tasks",
            "missing_evidence",
            "monitoring_triggers",
            "confidence",
        ),
        future_agno_instruction=(
            "Treat model_view and committee_view as inputs to audit, not as "
            "official ratings. Check whether a hold on an investment-grade model "
            "call is justified by verified pre-cutoff evidence or severe financial "
            "stress, and recommend keep, subtype downgrade, memo-only fix, or manual review."
        ),
    ),
    Stage2AgentSpec(
        role="risk_recall_qa",
        display_name="RiskRecallQAAgent",
        purpose=(
            "Audit eligible committee decisions for missed-risk recall safety when the "
            "model probability, financial axes, external evidence, or rating boundary "
            "suggest residual downside risk."
        ),
        required_inputs=(
            "committee_view",
            "quant_credit_output",
            "evidence_audit_output",
            "chair_report_output",
            "model_view",
            "source_feature_row",
            "news_cache_snapshot",
            "prior_rating_reference",
        ),
        output_fields=(
            "qa_summary",
            "trigger_reasons",
            "eligible_safety_assessment",
            "financial_resilience_check",
            "evidence_recall_check",
            "rating_boundary_check",
            "recommended_action",
            "manual_review_tasks",
            "missing_evidence",
            "monitoring_triggers",
            "confidence",
        ),
        future_agno_instruction=(
            "Treat eligible committee decisions as provisional when they are near the "
            "threshold, financially weak, or supported by ambiguous external evidence. "
            "Recommend keep, boundary hold, risk hold, manual review, or memo-only fix "
            "without inventing evidence, and do not escalate from low-quality news "
            "snippets unless structured evidence or severe financial stress corroborates them."
        ),
    ),
)


def get_stage2_agent_spec(role: Stage2AgentRole) -> Stage2AgentSpec:
    """Return the fixed role contract for a Stage 2 agent."""
    for spec in (*STAGE2_AGENT_SPECS, *STAGE2_OPTIONAL_AGENT_SPECS):
        if spec.role == role:
            return spec
    raise ValueError(f"Unknown Stage 2 agent role: {role}")


__all__ = [
    "STAGE2_AGENT_ROLES",
    "STAGE2_AGENT_SPECS",
    "STAGE2_OPTIONAL_AGENT_ROLES",
    "STAGE2_OPTIONAL_AGENT_SPECS",
    "Stage2AgentRole",
    "Stage2AgentSpec",
    "get_stage2_agent_spec",
]
