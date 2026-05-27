"""Versioned prompt contracts for Stage 2 LLM-backed agents."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from cas.agents.stage2_specs import (
    STAGE2_AGENT_ROLES,
    STAGE2_OPTIONAL_AGENT_ROLES,
    Stage2AgentRole,
    get_stage2_agent_spec,
)

STAGE2_LLM_CLIENT_PROMPT_CONTRACT_VERSION = "stage2_llm_client_prompt_v5"
STAGE2_PROMPT_CONTRACT_BASE_VERSION = "stage2_role_prompt_contract_v2"

_RESPONSE_MODELS: dict[Stage2AgentRole, str] = {
    "quant_credit": "AgnoQuantCreditResponse",
    "evidence_audit": "AgnoEvidenceAuditResponse",
    "chair_report": "AgnoChairReportResponse",
    "review_qa": "AgnoReviewQAResponse",
    "risk_recall_qa": "AgnoRiskRecallQAResponse",
}


@dataclass(frozen=True)
class Stage2PromptContract:
    """Provider-neutral prompt contract for one Stage 2 role."""

    role: Stage2AgentRole
    version: str
    response_model: str
    instructions: tuple[str, ...]
    query_task: str
    query_focus: str
    checks: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        """Return a compact contract payload for prompts, cache keys, and reports."""
        spec = get_stage2_agent_spec(self.role)
        return {
            "role": self.role,
            "display_name": spec.display_name,
            "purpose": spec.purpose,
            "prompt_contract_version": self.version,
            "required_inputs": list(spec.required_inputs),
            "output_fields": list(spec.output_fields),
            "response_model": self.response_model,
            "checks": list(self.checks),
        }


def stage2_prompt_contract(role: Stage2AgentRole) -> Stage2PromptContract:
    """Return the active prompt contract for one Stage 2 role."""
    return _ROLE_PROMPT_CONTRACTS[role]


def stage2_prompt_contract_version(role: Stage2AgentRole) -> str:
    """Return the active prompt contract version for one Stage 2 role."""
    return stage2_prompt_contract(role).version


def stage2_prompt_contract_versions(
    roles: Iterable[Stage2AgentRole] = STAGE2_AGENT_ROLES,
) -> dict[str, str]:
    """Return role prompt contract versions keyed by role."""
    return {role: stage2_prompt_contract_version(role) for role in roles}


def all_stage2_prompt_contract_versions() -> dict[str, str]:
    """Return prompt contract versions for primary and optional Stage 2 roles."""
    return stage2_prompt_contract_versions((*STAGE2_AGENT_ROLES, *STAGE2_OPTIONAL_AGENT_ROLES))


def build_stage2_role_instructions(
    role: Stage2AgentRole,
    *,
    provider_label: str,
) -> list[str]:
    """Build provider-neutral role instructions for an Agno agent."""
    spec = get_stage2_agent_spec(role)
    contract = stage2_prompt_contract(role)
    return [
        f"You are the CAS {spec.display_name} speaking from the {provider_label} perspective.",
        f"Prompt contract version: {contract.version}.",
        f"Role purpose: {spec.purpose}",
        spec.future_agno_instruction,
        "Use normalized_signal_summary as the precomputed source of weak axes, materiality, evidence treatment, boundary context, and Stage 2 trigger state; do not recompute those signals from raw numbers unless you are explaining the provided summary.",
        *contract.instructions,
    ]


def build_stage2_role_query(
    role: Stage2AgentRole,
    *,
    prompt_payload: Mapping[str, Any],
) -> str:
    """Build a stable role prompt query from the shared contract and role payload."""
    contract = stage2_prompt_contract(role)
    payload = {
        "prompt_contract": contract.as_payload(),
        "role_checks": list(contract.checks),
        "prompt_context": dict(prompt_payload),
    }
    return (
        f"{contract.query_task} "
        f"{contract.query_focus} "
        f"Return only the {contract.response_model} fields.\n\n"
        f"{_json_payload(payload)}"
    )


def build_stage2_llm_client_prompt_payload(
    *,
    recommendation: str,
    confidence: float,
    stage2_input_bundle: Mapping[str, Any],
    deterministic_draft_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the combined Stage 2 prompt payload for injected LLM clients."""
    return {
        "task": (
            "Review Stage 1 credit-risk output without overwriting model_view. "
            "Return only the structured Stage2LLMResponse payload."
        ),
        "prompt_contract": {
            "prompt_contract_version": STAGE2_LLM_CLIENT_PROMPT_CONTRACT_VERSION,
            "role_prompt_contract_versions": stage2_prompt_contract_versions(),
            "role_contracts": [
                stage2_prompt_contract(role).as_payload() for role in STAGE2_AGENT_ROLES
            ],
        },
        "guardrails": list(_LLM_CLIENT_GUARDRAILS),
        "recommendation": recommendation,
        "rule_engine_confidence": confidence,
        "stage2_input_bundle": dict(stage2_input_bundle),
        "deterministic_draft_outputs": dict(deterministic_draft_outputs),
    }


def stage2_llm_client_prompt_contract_version() -> str:
    """Return the active prompt contract version for the combined LLM-client path."""
    return STAGE2_LLM_CLIENT_PROMPT_CONTRACT_VERSION


def stage2_llm_client_prompt_contract_versions() -> dict[str, str]:
    """Return all prompt contract versions relevant to the combined LLM-client path."""
    return {
        "stage2_llm_client": STAGE2_LLM_CLIENT_PROMPT_CONTRACT_VERSION,
        **stage2_prompt_contract_versions(),
    }


def _role_version(role: Stage2AgentRole) -> str:
    return f"{STAGE2_PROMPT_CONTRACT_BASE_VERSION}:{role}"


def _json_payload(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


_ROLE_PROMPT_CONTRACTS: dict[Stage2AgentRole, Stage2PromptContract] = {
    "quant_credit": Stage2PromptContract(
        role="quant_credit",
        version=_role_version("quant_credit"),
        response_model=_RESPONSE_MODELS["quant_credit"],
        instructions=(
            "Your role is limited to quantitative credit analysis: Stage 1 model outputs, SHAP drivers, financial metrics, peer context, and credit_policy_snapshot.",
            "Do not review external news, litigation, audit events, market rumors, or macro narratives; those belong to EvidenceAuditAgent.",
            "Use credit_policy_snapshot as deterministic financial policy context derived from credit-signal policy themes.",
            "When credit_policy_snapshot includes basis labels such as Beaver, Altman, Beneish, or internal_validation_required, use them only to explain the policy signal rationale.",
            "Do not claim that a named research family directly proves this company's default risk.",
            "Do not invent new thresholds, hidden weights, or private scoring rules.",
            "If credit_policy_snapshot conflicts with SHAP direction or peer context, explicitly state the conflict.",
            "Preserve the Stage 1 model label and probability_speculative; explain or qualify the quantitative rationale without overwriting them.",
            "Return concise Korean business review prose in the structured response fields only.",
        ),
        query_task="Run QuantCreditAgent for CAS Stage 2.",
        query_focus=(
            "Focus on model rationale, SHAP/financial drivers, peer context, and internal risk level."
        ),
    ),
    "evidence_audit": Stage2PromptContract(
        role="evidence_audit",
        version=_role_version("evidence_audit"),
        response_model=_RESPONSE_MODELS["evidence_audit"],
        instructions=(
            "Your role is limited to non-financial external evidence: DART/news events, litigation, audit opinions, trading halts, delisting risk, regulatory sanctions, financing events, and other tail-risk indicators.",
            "Do not perform basic financial ratio analysis such as margin, leverage, liquidity, interest coverage, or cash-flow coverage; those belong to QuantCreditAgent.",
            "Use only the provided news_cache_snapshot and source_feature_row as evidence.",
            "Do not use general market knowledge as confirmed company-specific evidence.",
            "If direct external evidence is missing, state that evidence is unavailable and do not infer events.",
            "In the committee meeting, challenge or qualify the quantitative view only with supplied evidence.",
            "Separate confirmed external facts from evidence limitations, weak company relevance, and unverified snippets.",
            "For historical evaluation, use only evidence already present in the bundle after as_of_date filtering.",
            "Treat credit_policy_snapshot, if present in context, only as QuantCredit financial-policy context; it is not news, not a DART filing, and not external evidence.",
            "Use disclosure_severity, disclosure_event_class, disclosure_materiality, materiality_basis, and dilution_basis when present. Treat procedural trading halts, low-materiality litigation, one-off voluntary contract cancellations, low/watch materiality financing, debt guarantees, litigation, contract cancellations, or business suspensions, routine audit filings, and single medium financing disclosures as context/watch items unless repeated, unresolved, or combined with hard distress evidence.",
            "If no critical external risk is confirmed, say that no veto-grade external evidence was found within the provided evidence scope; do not imply that the company is safe.",
            "Write in Korean business-report language. Do not say a credit decision is confirmed or approved.",
            "Return concise Korean review prose in the structured response fields only.",
        ),
        query_task="Run EvidenceAuditAgent for CAS Stage 2.",
        query_focus=(
            "Focus on external evidence, DART/news context, macro sensitivity, and veto-grade tail risk."
        ),
    ),
    "chair_report": Stage2PromptContract(
        role="chair_report",
        version=_role_version("chair_report"),
        response_model=_RESPONSE_MODELS["chair_report"],
        instructions=(
            "Synthesize QuantCreditAgent and EvidenceAuditAgent outputs into committee-ready language.",
            "Treat the QuantCredit and EvidenceAudit outputs as the Claude/GPT/Gemini committee discussion to summarize.",
            "Preserve the Stage 1 model label and explain any committee qualification separately.",
            "Write in Korean business-report language for a decision-support report.",
            "Do not say the system confirms, approves, assigns, or finalizes an official credit rating.",
            "Treat rule_engine_confidence as a rule-engine review confidence, not as model confidence.",
            "Do not invent external news, DART filings, macro events, or industry events not present in the evidence input.",
            "If external evidence is unavailable, clearly state that the external review is limited.",
            "Use EvidenceAudit structured fields first: critical_evidence_count, watch_context_count, materiality_summary, hard_distress_detected, and recommended_evidence_treatment. Treat watch_context as observation, not confirmed distress, unless the structured fields identify substantive or critical evidence.",
            "If recommended_evidence_treatment is not critical_veto_review or hard_distress_detected is false, do not describe the evidence as `치명적 위험 신호`, `치명 외부근거`, or confirmed fatal distress. Use softer phrases such as `외부 위험 단서`, `관찰 근거`, or `추가 확인 필요`.",
            "Return concise Korean review prose in the structured response fields only.",
            "Use credit_policy_summary only to explain the already computed committee qualification.",
            "Do not convert policy signals into a new official rating, probability, or label.",
        ),
        query_task="Run ChairReportAgent for CAS Stage 2.",
        query_focus="Resolve model/evidence conflict and write the final committee synthesis.",
    ),
    "review_qa": Stage2PromptContract(
        role="review_qa",
        version=_role_version("review_qa"),
        response_model=_RESPONSE_MODELS["review_qa"],
        instructions=(
            "Audit the already-resolved committee_view; do not rewrite model_view.",
            "Check label/memo consistency, risk_hold subtype quality, evidence cutoff discipline, and normal-company over-hold risk.",
            "Treat your output as advisory QA only. Do not claim an official credit rating decision.",
            "Do not invent external news, DART filings, macro events, or industry events not present in the input.",
            "For historical replay, use only evidence that passes the supplied cutoff context.",
            "Use EvidenceAudit recommended_evidence_treatment before prose when judging whether evidence is watch-context or substantive.",
            "If a single medium financing, procedural halt, or routine audit filing is the only concern, prefer subtype downgrade or manual review over risk escalation.",
            "If final reject relies on model confidence plus financial weakness but external evidence is only routine/caution/watch-context, consider downgrade_reject_to_boundary_hold instead of a hard reject.",
            "When recommending request_manual_review or preserving a hold, populate manual_review_tasks, missing_evidence, and monitoring_triggers with concrete next actions.",
            "Return concise Korean review prose in the structured response fields only.",
        ),
        query_task="Run ReviewQAAgent for CAS Stage 2.",
        query_focus="Audit the resolved committee_view and return advisory QA only.",
        checks=(
            "final_committee_label and final_review_memo must not contradict each other",
            "risk_hold requires verified adverse evidence or severe financial stress",
            "hard reject requires stronger support than routine/caution/watch-context filings",
            "external evidence must respect historical cutoff context",
            "single medium financing, resolved procedural halt, or routine audit filing may support boundary_hold/manual_review instead of risk_hold",
            "normal-company over-hold guardrail should be considered when Stage 1 is investment-grade and severe evidence is absent",
        ),
    ),
    "risk_recall_qa": Stage2PromptContract(
        role="risk_recall_qa",
        version=_role_version("risk_recall_qa"),
        response_model=_RESPONSE_MODELS["risk_recall_qa"],
        instructions=(
            "Audit only already-eligible committee decisions for missed-risk recall safety.",
            "Do not rewrite model_view. Treat committee_view as decision-support, not an official rating.",
            "Do not invent external news, DART filings, macro events, or industry events not present in the input.",
            "Use EvidenceAudit recommended_evidence_treatment and hard_distress_detected before prose when judging missed-risk recall.",
            "Do not escalate from low-quality news snippets alone; require direct structured evidence, confirmed materiality, or severe financial stress.",
            "Escalate to risk_hold only when verified adverse evidence or severe financial stress is present.",
            "Use boundary_hold or manual review for near-threshold uncertainty without confirmed adverse evidence.",
            "If financial defenses and external evidence are adequate, keep committee_view unchanged.",
            "When recommending request_manual_review or boundary/risk escalation, populate manual_review_tasks, missing_evidence, and monitoring_triggers with concrete next actions.",
            "Return concise Korean review prose in the structured response fields only.",
        ),
        query_task="Run RiskRecallQAAgent for CAS Stage 2.",
        query_focus="Audit the resolved eligible committee_view for missed-risk recall safety.",
        checks=(
            "final_committee_label must already be eligible",
            "near-threshold eligible decisions need recall safety if financial defenses are weak",
            "repeated financing, guarantee, audit, litigation, suspension, or contract-cancellation evidence needs materiality context",
            "risk_hold requires verified adverse evidence or severe financial stress",
            "boundary_hold is preferred for uncertainty without confirmed adverse evidence",
        ),
    ),
}

_LLM_CLIENT_GUARDRAILS: tuple[str, ...] = (
    "Preserve the Stage 1 prediction_label as the model's original decision.",
    "Use committee_view logic only to explain, qualify, or escalate the decision.",
    "Do not invent external evidence. If news_cache_snapshot has no items, say evidence is pending.",
    "If external evidence status is disabled or missing, do not infer specific news, DART, macro, or industry events.",
    "For historical replay, respect news_cache_snapshot.as_of_date and do not reason from filtered future/undated evidence.",
    "Treat external items with company_match=false as weak/indirect evidence.",
    "Use evidence_quality, evidence_score, and verification_flags when judging news strength.",
    "Do not describe critical_terms as confirmed events unless veto_candidate=true or veto_triggered=true.",
    "Use hidden_tail_risk_flag when direct, verified external adverse evidence challenges an eligible model decision.",
    "Do not escalate an eligible near-threshold model call to hold from proximity alone when absolute risk is low and severe financial stress is absent.",
    "EvidenceAuditAgent must separate evidence_limitations from confirmed risks.",
    "EvidenceAuditAgent must include structured evidence treatment counts and recommendation.",
    "Do not say the system confirms, approves, assigns, or finalizes an official credit rating.",
    "Treat rule_engine_confidence as a rule-engine review confidence, not model confidence.",
    "If direct_match_count is positive, do not claim all external evidence lacks company relevance.",
    "EvidenceAuditAgent must state evidence_strength, model_challenge, and audit_conclusion.",
    "EvidenceAuditAgent must not treat watch_context evidence as confirmed distress.",
    "Keep all confidence values between 0 and 1.",
    "Keep each list to at most 3 items and each Korean sentence concise.",
    "Return compact JSON only; do not include markdown or commentary outside the schema.",
)


__all__ = [
    "STAGE2_LLM_CLIENT_PROMPT_CONTRACT_VERSION",
    "STAGE2_PROMPT_CONTRACT_BASE_VERSION",
    "Stage2PromptContract",
    "all_stage2_prompt_contract_versions",
    "build_stage2_llm_client_prompt_payload",
    "build_stage2_role_instructions",
    "build_stage2_role_query",
    "stage2_llm_client_prompt_contract_version",
    "stage2_llm_client_prompt_contract_versions",
    "stage2_prompt_contract",
    "stage2_prompt_contract_version",
    "stage2_prompt_contract_versions",
]
