"""Agno-backed QuantCreditAgent adapter."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import QuantCreditOutput

from .runtime import (
    build_agno_agent,
    clamp,
    compact_items,
    json_payload,
    provider_label,
    run_structured_agent,
)


class AgnoQuantCreditResponse(BaseModel):
    """Structured response produced by the Agno QuantCreditAgent."""

    model_config = ConfigDict(extra="forbid")

    quantitative_interpretation: str = Field(
        description="Business explanation of the model label, probability, and SHAP drivers."
    )
    fundamental_defense_capacity: str = Field(
        description="Assessment of liquidity, cash-flow, leverage, and peer defenses."
    )
    key_risk_and_mitigation: str = Field(
        description="Main risk factors and mitigating factors from the quantitative view."
    )
    internal_risk_level: str = Field(
        description="Internal risk level such as low, medium, high, or equivalent Korean label."
    )


def run_quant_credit_agent(
    *,
    bundle: Stage2InputBundle,
    model_name: str,
    model_provider: str = "anthropic",
    max_tokens: int,
) -> QuantCreditOutput:
    """Run the Agno QuantCreditAgent and map it to the CAS Stage 2 schema."""
    model_label = provider_label(model_provider)
    agent = build_agno_agent(
        name=f"{model_label}_QuantCredit_Agent",
        model_provider=model_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        response_model=AgnoQuantCreditResponse,
        instructions=[
            f"You are the CAS QuantCreditAgent, a cold, numbers-driven Financial Risk Analyst (Model: {model_label}).",
            "Your SOLE purpose is to evaluate the mathematical deterioration of the company's financial statements and validate the Stage 1 XGBoost model's decision.",
            "Do not review external news, market rumors, litigation, audit events, or macro sentiment narratives. Those belong to the EvidenceAuditAgent.",
            "Focus entirely on: XGBoost probability, SHAP top drivers, capital impairment, interest coverage, cash-flow coverage, leverage, liquidity buffers, peer comparison, and credit_policy_snapshot.",
            "Use credit_policy_snapshot as the deterministic financial-policy guideline layer. Treat it as structured internal credit guidance derived from classic financial distress research themes such as cash-flow-to-debt, interest coverage, leverage, liquidity, accrual quality, and earnings manipulation risk.",
            "Do not invent new thresholds, hidden weights, or private scoring rules. If a threshold or signal is not present in credit_policy_snapshot, describe it as an observation rather than a policy trigger.",
            "When credit_policy_snapshot contains basis labels such as Beaver, Altman, Beneish, accounting_ratio_distress_literature, or internal_validation_required, use them only to explain the rationale of the policy signal. Do not claim that the paper directly proves this company's default risk.",
            "Preserve Stage 1 prediction_label and probability_speculative. You may challenge, qualify, or contextualize the Stage 1 decision, but you must not overwrite it.",
            "If SHAP drivers, peer comparison, and credit_policy_snapshot point in the same direction, present this as a convergent quantitative signal.",
            "If SHAP drivers and credit_policy_snapshot conflict, explicitly explain the conflict. For example, distinguish between model-driven risk, accounting-policy risk, and financial buffer evidence.",
            "1. For 'quantitative_interpretation': Translate raw SHAP contributions and Stage 1 probability into a logical financial narrative. Explain exactly why the machine learning model flagged or defended this company based on the numbers.",
            "2. For 'fundamental_defense_capacity': Analyze core financial buffers such as operating cash flow, cash equivalents, current ratio, low leverage, equity ratio, and debt-service capacity. If these are stable despite a high-risk ML flag, emphasize this as a potential False Positive or 과민경고 candidate.",
            "3. For 'key_risk_and_mitigation': List the exact financial metrics causing the risk and any numeric buffers defending against it. Use percentages, ratios, SHAP direction, peer percentile, and policy signal severity whenever available.",
            "4. For 'internal_risk_level': Classify strictly as 'High', 'Medium', or 'Low' based solely on financial health, model evidence, peer comparison, and credit_policy_snapshot.",
            "Write the analysis in highly professional Korean accounting and credit-risk terminology. Be precise with percentages, ratios, SHAP values, and policy signal names.",
        ],
    )
    result = run_structured_agent(
        agent=agent,
        query=_query(bundle),
        response_model=AgnoQuantCreditResponse,
    )
    return QuantCreditOutput(
        quant_summary=(
            f"Agno {model_label} QuantCreditAgent reviewed the Stage 1 model view. "
            f"Internal risk level: {result.internal_risk_level}. "
            f"{result.quantitative_interpretation}"
        ),
        model_rationale=result.quantitative_interpretation,
        key_risk_factors=compact_items(
            result.key_risk_and_mitigation,
            f"Internal risk level: {result.internal_risk_level}",
        ),
        mitigating_factors=compact_items(result.fundamental_defense_capacity),
        confidence=_confidence_from_risk_level(result.internal_risk_level),
    )


def _query(bundle: Stage2InputBundle) -> str:
    prompt_payload = {
        "company": {
            "company_id": bundle.company_id,
            "company_name": bundle.company_name,
            "market": bundle.market,
            "analysis_year": bundle.analysis_year,
        },
        "stage1_model": {
            "prediction_label": bundle.prediction_label,
            "probability_speculative": bundle.probability_speculative,
            "xgboost_result": bundle.xgboost_result,
            "model_view": bundle.model_view,
        },
        "prior_rating_reference": bundle.prior_rating_reference,
        "source_feature_row": bundle.source_feature_row,
        "peer_comparison_rows": list(bundle.peer_comparison_rows),
        "credit_policy_snapshot": bundle.credit_policy_snapshot,
        "policy_guardrail": {
            "label_override_allowed": False,
            "rule_kr": (
                "credit_policy_snapshot은 정량 해석 보조 근거이며, "
                "Stage 1 모델 라벨과 확률을 수정하는 근거가 아니다."
            ),
        },
    }
    return (
        "Run QuantCreditAgent for CAS Stage 2. "
        "Focus on model rationale, SHAP/financial drivers, peer context, and internal risk level. "
        "Return only the AgnoQuantCreditResponse fields.\n\n"
        f"{json_payload(prompt_payload)}"
    )


def _confidence_from_risk_level(risk_level: str) -> float:
    normalized = risk_level.strip().lower()
    if any(marker in normalized for marker in ("high", "critical", "elevated", "고위험")):
        return 0.82
    if any(marker in normalized for marker in ("medium", "moderate", "중간")):
        return 0.76
    if any(marker in normalized for marker in ("low", "낮")):
        return 0.68
    return clamp(0.72, minimum=0.45, maximum=0.86)


__all__ = ["AgnoQuantCreditResponse", "run_quant_credit_agent"]
