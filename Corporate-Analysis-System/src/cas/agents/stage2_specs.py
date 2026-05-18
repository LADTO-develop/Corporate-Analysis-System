"""Role contracts for the Stage 2 three-agent review scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Stage2AgentRole = Literal["quant_credit", "evidence_audit", "chair_report"]


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
            "evidence_reliability",
            "hidden_tail_risk",
            "fn_supplement_flag",
            "debt_liquidity_cross_check",
            "macro_industry_sensitivity",
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
            "veto_triggered",
            "hidden_tail_risk_flag",
            "hidden_tail_risk_reason",
            "conflict_resolution",
            "key_risk_factors",
            "mitigating_factors",
            "evidence_summary",
            "final_review_memo",
        ),
        future_agno_instruction=(
            "Resolve conflicts conservatively, explain any gap between model_view "
            "and committee_view, and emit the strict committee_view fields."
        ),
    ),
)


def get_stage2_agent_spec(role: Stage2AgentRole) -> Stage2AgentSpec:
    """Return the fixed role contract for a Stage 2 agent."""
    for spec in STAGE2_AGENT_SPECS:
        if spec.role == role:
            return spec
    raise ValueError(f"Unknown Stage 2 agent role: {role}")


__all__ = [
    "STAGE2_AGENT_ROLES",
    "STAGE2_AGENT_SPECS",
    "Stage2AgentRole",
    "Stage2AgentSpec",
    "get_stage2_agent_spec",
]
