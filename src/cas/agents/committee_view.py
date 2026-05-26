"""Build the dashboard-facing Stage 2 committee_view payload."""

from __future__ import annotations

from typing import Any, Literal, cast

from cas.agents.committee_assessments import (
    ADVERSE_EVIDENCE_QUALITY,
    ADVERSE_PROVIDER_RELEVANCE,
    BoundaryReviewAssessment,
    FinancialResilienceAssessment,
    HiddenTailRiskAssessment,
    NoncriticalEvidenceAssessment,
    OverwarningMitigationAssessment,
    RejectConfirmationAssessment,
    SecondaryReviewRiskAssessment,
)
from cas.agents.committee_schema import (
    CommitteeDecisionType,
    CommitteeLabel,
    CommitteeViewPayload,
    DecisionTraceItem,
    RiskHoldReasonTag,
)
from cas.agents.committee_utils import (
    clean_evidence_summary_items as _clean_evidence_summary_items,
)
from cas.agents.committee_utils import (
    clean_korean_review_text as _clean_korean_review_text,
)
from cas.agents.committee_utils import (
    clean_text_items as _clean_text_items,
)
from cas.agents.committee_utils import (
    flag_is_false as _flag_is_false,
)
from cas.agents.committee_utils import (
    flag_is_true as _flag_is_true,
)
from cas.agents.committee_utils import (
    metric_above as _metric_above,
)
from cas.agents.committee_utils import (
    metric_at_least as _metric_at_least,
)
from cas.agents.committee_utils import (
    metric_at_most as _metric_at_most,
)
from cas.agents.committee_utils import (
    metric_below as _metric_below,
)
from cas.agents.committee_utils import (
    safe_float as _safe_float,
)
from cas.agents.committee_utils import (
    safe_int as _safe_int,
)
from cas.agents.signals.evidence_treatment_signals import (
    evaluate_evidence_treatment as _evaluate_evidence_treatment,
)
from cas.agents.signals.materiality_signals import (
    financing_evidence_items as _shared_financing_evidence_items,
)
from cas.agents.signals.materiality_signals import (
    has_hard_distress_terms as _shared_has_hard_distress_terms,
)
from cas.agents.signals.materiality_signals import (
    hidden_tail_evidence_requires_risk_signal as _shared_hidden_tail_evidence_requires_risk_signal,
)
from cas.agents.signals.materiality_signals import (
    high_risk_financing_evidence_count as _shared_high_risk_financing_evidence_count,
)
from cas.agents.signals.materiality_signals import (
    is_material_financing_or_guarantee_item as _shared_is_material_financing_or_guarantee_item,
)
from cas.agents.signals.materiality_signals import (
    is_uncorroborated_material_financing_or_guarantee_item as _shared_is_uncorroborated_material_financing_or_guarantee_item,
)
from cas.agents.signals.materiality_signals import (
    material_financing_evidence_blocks_tn_hold as _shared_material_financing_evidence_blocks_tn_hold,
)
from cas.agents.signals.materiality_signals import (
    material_financing_or_guarantee_has_financial_corroboration as _shared_material_financing_or_guarantee_has_financial_corroboration,
)
from cas.agents.signals.materiality_signals import (
    material_financing_or_guarantee_has_severe_financial_corroboration as _shared_material_financing_or_guarantee_has_severe_financial_corroboration,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.state import AgentOutput, Recommendation
from cas.veto_rules import (
    VetoRules,
    critical_terms_in_text,
    external_evidence_veto_triggered,
    flag_contains_veto_marker,
    load_veto_rules,
)


def build_committee_view(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    agents: list[AgentOutput],
) -> dict[str, Any]:
    """Build and serialize committee_view without calling external LLMs."""
    payload = build_committee_view_model(
        bundle=bundle,
        recommendation=recommendation,
        agents=agents,
    )
    return cast(dict[str, Any], payload.model_dump(mode="json"))


def build_committee_view_model(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    agents: list[AgentOutput],
) -> CommitteeViewPayload:
    """Build the strict Pydantic committee_view payload."""
    prediction_label = bundle.prediction_label
    committee_label = _committee_label_from_recommendation(recommendation)
    veto_rules = load_veto_rules()
    veto_triggered = _veto_triggered(bundle, veto_rules=veto_rules)
    if veto_triggered:
        committee_label = _veto_triggered_label(veto_rules)

    hidden_tail_risk = _hidden_tail_risk_assessment(bundle)
    risk_factors = _collect_committee_factors(agents, target="risk")
    if hidden_tail_risk.triggered:
        risk_factors = [hidden_tail_risk.reason, *risk_factors]
    secondary_review_risk = _secondary_review_risk_assessment(bundle)
    if secondary_review_risk.triggered:
        risk_factors = [secondary_review_risk.reason, *risk_factors]
    mitigating_factors = _collect_committee_factors(agents, target="mitigation")
    secondary_overhold_guardrail_reason = _secondary_overhold_guardrail_reason(bundle)
    if secondary_overhold_guardrail_reason:
        mitigating_factors = [secondary_overhold_guardrail_reason, *mitigating_factors]
    if not veto_triggered:
        committee_label = _committee_label_with_evidence_escalation(
            committee_label,
            bundle=bundle,
            agents=agents,
            hidden_tail_risk=hidden_tail_risk,
        )
    if (
        not veto_triggered
        and not hidden_tail_risk.triggered
        and secondary_review_risk.triggered
        and committee_label == "적격"
    ):
        committee_label = "보류"
    committee_label = _committee_label_with_model_alignment(
        committee_label,
        bundle=bundle,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
    )
    committee_label = _committee_label_with_investment_evidence_alignment(
        committee_label,
        bundle=bundle,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
    )
    if prediction_label == "투자적격" and committee_label == "부적격" and not veto_triggered:
        committee_label = "적격" if secondary_overhold_guardrail_reason else "보류"
    overwarning_mitigation = _overwarning_mitigation_assessment(
        bundle,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        mitigating_factors=mitigating_factors,
    )
    if (
        not veto_triggered
        and not hidden_tail_risk.triggered
        and overwarning_mitigation.triggered
        and committee_label == "부적격"
    ):
        committee_label = "보류"
        mitigating_factors = [overwarning_mitigation.reason, *mitigating_factors]
    prior_boundary_reason = _prior_rating_boundary_hold_reason(
        bundle,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
    )
    if (
        prior_boundary_reason
        and (committee_label == "부적격" or _prior_rating_boundary_requires_hold(bundle))
        and not secondary_overhold_guardrail_reason
    ):
        committee_label = "보류"
    reject_confirmation = _reject_confirmation_assessment(
        bundle,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
    )
    if committee_label == "부적격" and not reject_confirmation.confirmed:
        committee_label = "보류"
        risk_factors = [reject_confirmation.reason, *risk_factors]
    else:
        reject_confirmation = RejectConfirmationAssessment(
            reject_confirmation.confirmed,
            False,
            reject_confirmation.reason,
            reject_confirmation.signal_count,
            reject_confirmation.signals,
        )
    if prediction_label == "부적격" and committee_label == "적격" and not veto_triggered:
        committee_label = "보류"
    boundary_review = _boundary_review_assessment(
        bundle,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
        reject_confirmation=reject_confirmation,
    )
    if boundary_review.triggered:
        risk_factors = [boundary_review.reason, *risk_factors]
    evidence_summary = _evidence_summary_items(bundle, agents)
    conflict_resolution = _conflict_resolution(
        prediction_label=prediction_label,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        boundary_review=boundary_review,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
        reject_confirmation=reject_confirmation,
    )
    final_review_memo = _final_review_memo(
        prediction_label=prediction_label,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        boundary_review=boundary_review,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
        reject_confirmation=reject_confirmation,
        risk_factors=risk_factors,
        mitigating_factors=mitigating_factors,
    )
    final_review_memo = _with_chair_report_memo(
        final_review_memo,
        _chair_report_memo_seed(agents, committee_label=committee_label),
    )
    decision_type = _committee_decision_type(
        committee_label=committee_label,
        prediction_label=prediction_label,
        hidden_tail_risk=hidden_tail_risk,
        boundary_review=boundary_review,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
        reject_confirmation=reject_confirmation,
    )
    risk_hold_reason_tags = _risk_hold_reason_tags(
        bundle=bundle,
        decision_type=decision_type,
        hidden_tail_risk=hidden_tail_risk,
        secondary_review_risk=secondary_review_risk,
        reject_confirmation=reject_confirmation,
    )
    risk_hold_reason_labels = _risk_hold_reason_labels(risk_hold_reason_tags)
    risk_hold_reason_summary = _risk_hold_reason_summary(
        tags=risk_hold_reason_tags,
        labels=risk_hold_reason_labels,
    )
    if risk_hold_reason_summary:
        final_review_memo = f"{final_review_memo} {risk_hold_reason_summary}"

    risk_factors = _clean_text_items(risk_factors)
    mitigating_factors = _clean_text_items(mitigating_factors)
    evidence_summary = _clean_evidence_summary_items(evidence_summary)
    conflict_resolution = _clean_korean_review_text(conflict_resolution)
    final_review_memo = _clean_korean_review_text(final_review_memo)

    return CommitteeViewPayload(
        final_committee_label=committee_label,
        committee_decision_type=decision_type,
        committee_decision_type_label=_committee_decision_type_label(decision_type),
        committee_risk_signal=_committee_risk_signal(decision_type),
        risk_hold_reason_tags=risk_hold_reason_tags,
        risk_hold_reason_labels=risk_hold_reason_labels,
        risk_hold_reason_summary=risk_hold_reason_summary,
        veto_triggered=veto_triggered,
        hidden_tail_risk_flag=hidden_tail_risk.triggered,
        hidden_tail_risk_reason=hidden_tail_risk.reason,
        conflict_resolution=conflict_resolution,
        key_risk_factors=risk_factors or ["현재 scaffold 기준 추가 위험 요인은 제한적입니다."],
        mitigating_factors=mitigating_factors
        or ["현재 scaffold 기준 명시적 완화 요인은 제한적입니다."],
        evidence_summary=evidence_summary,
        decision_trace=_decision_trace_items(
            bundle=bundle,
            committee_label=committee_label,
            decision_type=decision_type,
            veto_triggered=veto_triggered,
            hidden_tail_risk=hidden_tail_risk,
            boundary_review=boundary_review,
            secondary_review_risk=secondary_review_risk,
            overwarning_mitigation=overwarning_mitigation,
            reject_confirmation=reject_confirmation,
            risk_hold_reason_tags=risk_hold_reason_tags,
            risk_hold_reason_summary=risk_hold_reason_summary,
        ),
        final_review_memo=final_review_memo,
    )


def _committee_label_from_recommendation(recommendation: Recommendation) -> CommitteeLabel:
    if recommendation == "priority":
        return "적격"
    if recommendation in {"watch", "review"}:
        return "보류"
    return "부적격"


def _committee_decision_type(
    *,
    committee_label: CommitteeLabel,
    prediction_label: str,
    hidden_tail_risk: HiddenTailRiskAssessment,
    boundary_review: BoundaryReviewAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
) -> CommitteeDecisionType:
    if committee_label == "적격":
        return "eligible"
    if committee_label == "부적격":
        return "reject"
    if hidden_tail_risk.triggered:
        return "risk_hold" if hidden_tail_risk.risk_signal else "review_hold"
    if overwarning_mitigation.triggered and prediction_label == "부적격":
        return "mitigation_hold"
    if reject_confirmation.triggered:
        return "risk_hold" if reject_confirmation.review_risk_signal else "review_hold"
    if boundary_review.triggered:
        return "boundary_hold"
    if secondary_review_risk.triggered:
        return "risk_hold" if secondary_review_risk.risk_signal else "review_hold"
    if overwarning_mitigation.triggered:
        return "mitigation_hold"
    if prediction_label == "부적격":
        return "risk_hold"
    return "review_hold"


def _committee_decision_type_label(decision_type: CommitteeDecisionType) -> str:
    labels: dict[CommitteeDecisionType, str] = {
        "eligible": "적격",
        "risk_hold": "위험 보류",
        "boundary_hold": "경계등급 보류",
        "mitigation_hold": "과민경고 완화 보류",
        "review_hold": "확인필요 보류",
        "reject": "부적격",
    }
    return labels[decision_type]


def _committee_risk_signal(decision_type: CommitteeDecisionType) -> bool:
    return decision_type in {"risk_hold", "reject"}


_RISK_HOLD_REASON_LABELS: dict[RiskHoldReasonTag, str] = {
    "combined_watch_hold": "재무+외부 복합 관찰",
    "financial_stress_hold": "재무 스트레스",
    "external_materiality_hold": "외부 중요도 근거",
    "secondary_radar_hold": "2차 보조 레이더",
    "model_reject_confirmation_hold": "부적격 확정 전 보류",
    "model_risk_hold": "모델 위험 보류",
}


def _risk_hold_reason_tags(
    *,
    bundle: Stage2InputBundle,
    decision_type: CommitteeDecisionType,
    hidden_tail_risk: HiddenTailRiskAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    reject_confirmation: RejectConfirmationAssessment,
) -> list[RiskHoldReasonTag]:
    """Explain why a risk_hold remained a hold without changing the decision."""
    if decision_type != "risk_hold":
        return []

    financial_stress = _risk_hold_has_financial_stress(
        bundle,
        secondary_review_risk=secondary_review_risk,
        reject_confirmation=reject_confirmation,
    )
    external_materiality = _risk_hold_has_external_materiality(
        bundle,
        hidden_tail_risk=hidden_tail_risk,
    )
    tags: list[RiskHoldReasonTag] = []
    if financial_stress and external_materiality:
        tags.append("combined_watch_hold")
    if financial_stress:
        tags.append("financial_stress_hold")
    if external_materiality:
        tags.append("external_materiality_hold")
    if secondary_review_risk.triggered:
        tags.append("secondary_radar_hold")
    if reject_confirmation.triggered:
        tags.append("model_reject_confirmation_hold")
    if not tags:
        tags.append("model_risk_hold")
    return list(dict.fromkeys(tags))[:4]


def _risk_hold_has_financial_stress(
    bundle: Stage2InputBundle,
    *,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    reject_confirmation: RejectConfirmationAssessment,
) -> bool:
    row = bundle.source_feature_row
    if _has_severe_financial_watch_signal(row):
        return True
    if _has_secondary_overhold_guardrail_blocker(row):
        return True
    if reject_confirmation.signal_count >= 2:
        return True
    financial_flags = [
        _flag_is_true(row.get("icr_under_1")) or _metric_below(row, "interest_coverage_ratio", 1.0),
        _flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
        or _metric_below(row, "cashflow_coverage_ratio", 0.0)
        or _metric_below(row, "ocf_to_total_liabilities", 0.0)
        or _metric_below(row, "ocf_to_sales", 0.0),
        _flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        or _metric_below(row, "net_margin", -0.10),
        _metric_above(row, "capital_impairment_ratio", 0.0)
        or (_metric_below(row, "equity_ratio", 0.25) and _metric_above(row, "debt_ratio", 1.50)),
        _metric_below(row, "current_ratio", 1.0) and _metric_below(row, "cash_ratio", 0.10),
    ]
    if sum(1 for flag in financial_flags if flag) >= 2:
        return True
    return bool(secondary_review_risk.triggered and secondary_review_risk.risk_signal)


def _risk_hold_has_external_materiality(
    bundle: Stage2InputBundle,
    *,
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> bool:
    if hidden_tail_risk.triggered:
        return True
    treatment = _evaluate_evidence_treatment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    if treatment.recommended_evidence_treatment in {
        "substantive_review",
        "critical_veto_review",
    }:
        return True
    summary = treatment.materiality_summary
    return bool(_safe_int(summary.get("high_risk_financing_evidence_count")) or 0) or bool(
        summary.get("material_financing_blocks_tn_hold")
    )


def _risk_hold_reason_labels(tags: list[RiskHoldReasonTag]) -> list[str]:
    return [_RISK_HOLD_REASON_LABELS[tag] for tag in tags]


def _risk_hold_reason_summary(
    *,
    tags: list[RiskHoldReasonTag],
    labels: list[str],
) -> str:
    if not tags:
        return ""
    label_text = ", ".join(labels)
    if "combined_watch_hold" in tags:
        return (
            "위험 보류 이유 태그는 "
            f"{label_text}입니다. 재무 스트레스와 외부 중요도 근거가 함께 남아 있어, "
            "정상기업 과잉 보류 guardrail을 바로 적용하지 않고 위험 보류로 유지했습니다."
        )
    if "financial_stress_hold" in tags:
        return (
            "위험 보류 이유 태그는 "
            f"{label_text}입니다. 치명 외부근거가 약하더라도 현금흐름, 이자보상, 손익, "
            "유동성 중 방어가 충분하지 않은 축이 있어 적격으로 바로 낮추지 않았습니다."
        )
    if "external_materiality_hold" in tags:
        return (
            "위험 보류 이유 태그는 "
            f"{label_text}입니다. 공시나 외부근거의 기업 규모 대비 중요도가 남아 있어 "
            "단순 경계 보류보다 강한 재검토 신호로 유지했습니다."
        )
    return (
        "위험 보류 이유 태그는 "
        f"{label_text}입니다. 부적격 확정까지는 아니지만 적격으로 낮추기에는 "
        "잔여 위험 신호가 남아 보류로 유지했습니다."
    )


def _veto_triggered(bundle: Stage2InputBundle, *, veto_rules: VetoRules) -> bool:
    if not veto_rules.enabled:
        return False
    blocking_flags = [
        str(flag).lower() for flag in bundle.rule_result.get("blocking_flags", []) or []
    ]
    if any(flag_contains_veto_marker(flag, rules=veto_rules) for flag in blocking_flags):
        return True
    return bool(
        external_evidence_veto_triggered(
            bundle.news_cache_snapshot,
            company_name=bundle.company_name,
            stock_code=str(bundle.source_feature_row.get("stock_code") or bundle.company_id),
            rules=veto_rules,
        )
    )


def _veto_triggered_label(veto_rules: VetoRules) -> CommitteeLabel:
    label = veto_rules.triggered_label
    if label in {"적격", "보류", "부적격"}:
        return cast(CommitteeLabel, label)
    return "부적격"


def _committee_label_with_evidence_escalation(
    committee_label: CommitteeLabel,
    *,
    bundle: Stage2InputBundle,
    agents: list[AgentOutput],
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> CommitteeLabel:
    """Escalate non-veto EvidenceAudit red flags without overwriting model_view."""
    if committee_label != "적격":
        return committee_label
    if hidden_tail_risk.triggered:
        return "보류"
    evidence_agent = next((agent for agent in agents if agent.role == "evidence_audit"), None)
    if evidence_agent is None:
        return committee_label
    if _external_evidence_unavailable(bundle.news_status):
        return committee_label
    if _evidence_agent_requires_hold(evidence_agent):
        return "보류"
    return committee_label


def _committee_label_with_model_alignment(
    committee_label: CommitteeLabel,
    *,
    bundle: Stage2InputBundle,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> CommitteeLabel:
    """Keep low-probability model-eligible cases eligible unless hard evidence overrides."""
    if committee_label != "보류" or bundle.prediction_label != "투자적격":
        return committee_label
    if veto_triggered or hidden_tail_risk.triggered:
        return committee_label
    if not _external_evidence_unavailable(bundle.news_status):
        return committee_label
    if _secondary_overhold_guardrail_reason(bundle):
        return "적격"
    if _secondary_review_requires_hold(bundle):
        return committee_label
    if _has_blocking_flags(bundle):
        return committee_label
    if not bundle.source_feature_row:
        return committee_label

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    near_threshold = probability >= max(0.28, threshold - 0.05)
    if near_threshold:
        return committee_label
    if _has_severe_financial_watch_signal(bundle.source_feature_row):
        return committee_label
    return "적격"


def _committee_label_with_investment_evidence_alignment(
    committee_label: CommitteeLabel,
    *,
    bundle: Stage2InputBundle,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> CommitteeLabel:
    """Keep eligible model calls eligible when collected evidence is only non-critical context."""
    if committee_label != "보류" or bundle.prediction_label != "투자적격":
        return committee_label
    if veto_triggered or hidden_tail_risk.triggered:
        return committee_label
    if _external_evidence_unavailable(bundle.news_status):
        return committee_label
    if _secondary_overhold_guardrail_reason(bundle):
        return "적격"
    if _secondary_review_requires_hold(bundle):
        return committee_label
    if _has_blocking_flags(bundle):
        return committee_label
    if _has_severe_financial_watch_signal(bundle.source_feature_row):
        return committee_label
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability >= threshold:
        return committee_label
    noncritical_evidence = _noncritical_external_evidence_assessment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    if not noncritical_evidence.triggered:
        return committee_label
    return "적격"


def _evidence_agent_requires_hold(agent: AgentOutput) -> bool:
    for finding in agent.findings:
        text = str(finding)
        if text.startswith("외부근거 강도:"):
            strength = text.removeprefix("외부근거 강도:").strip().lower()
            if strength in {"strong", "critical"}:
                return True
        if _committee_factor_value(text, target="risk"):
            if _non_escalating_risk_text(text):
                continue
            return True
    return False


def _external_evidence_unavailable(status: str) -> bool:
    return status.strip().lower() in {
        "disabled",
        "missing_credentials",
        "not_implemented",
        "not_requested",
        "placeholder",
        "no_results",
    }


def _has_stage2_secondary_trigger(bundle: Stage2InputBundle) -> bool:
    for source in (bundle.model_view, bundle.xgboost_result):
        if bool(source.get("stage2_secondary_trigger")):
            return True
    return False


def _secondary_review_requires_hold(bundle: Stage2InputBundle) -> bool:
    """Return whether a secondary trigger is strong enough to block eligible alignment."""
    if not _has_stage2_secondary_trigger(bundle):
        return False
    if _secondary_overhold_guardrail_reason(bundle):
        return False
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    probability_floor = max(0.28, threshold - 0.10)
    secondary_liquidity_watch = _has_secondary_rule_liquidity_watch_signal(bundle)
    confident_secondary_liquidity_watch = secondary_liquidity_watch and (
        probability >= probability_floor
        or (threshold >= 0.28 and _rule_confidence_at_least(bundle, 0.60))
    )
    return (
        probability >= probability_floor
        or _has_severe_financial_watch_signal(bundle.source_feature_row)
        or confident_secondary_liquidity_watch
    )


def _has_blocking_flags(bundle: Stage2InputBundle) -> bool:
    flags = bundle.rule_result.get("blocking_flags", []) or []
    return any(str(flag).strip() for flag in flags)


def _has_isolated_interest_cover_defense(bundle: Stage2InputBundle) -> bool:
    """Allow TN guardrail when ICR is the only hard flag and OCF coverage is strong."""
    raw_flags = bundle.rule_result.get("blocking_flags", []) or []
    flags = {str(flag).strip().lower() for flag in raw_flags if str(flag).strip()}
    if not flags or not flags.issubset(
        {"interest_coverage_under_1", "icr_under_1", "interest_coverage_ratio_under_1"}
    ):
        return False
    return _has_isolated_interest_cover_row_defense(bundle.source_feature_row)


def _has_isolated_interest_cover_row_defense(row: dict[str, Any]) -> bool:
    """Return whether cash flow and low borrowings offset a single-year ICR dip."""
    return bool(
        (
            _flag_is_true(row.get("icr_under_1"))
            or _metric_below(row, "interest_coverage_ratio", 1.0)
        )
        and _metric_at_least(row, "current_ratio", 1.2)
        and _metric_at_least(row, "cash_ratio", 0.15)
        and _metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and _metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and _metric_at_most(row, "total_borrowings_ratio", 0.10)
        and _metric_at_most(row, "capital_impairment_ratio", 0.0)
        and not _flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        and not _flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
        and not _metric_below(row, "net_margin", -0.05)
    )


def _has_isolated_icr_review_buffer(row: dict[str, Any]) -> bool:
    """Downgrade risk display when an ICR dip is offset by OCF, capital, and low debt."""
    if not (
        _flag_is_true(row.get("icr_under_1")) or _metric_below(row, "interest_coverage_ratio", 1.0)
    ):
        return False
    if _flag_is_true(row.get("is_2y_consecutive_operating_loss")) or _flag_is_true(
        row.get("is_2y_consecutive_ocf_deficit")
    ):
        return False
    if _metric_above(row, "capital_impairment_ratio", 0.0):
        return False
    if _metric_below(row, "net_margin", -0.05):
        return False
    return bool(
        _metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and _metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and _metric_at_least(row, "equity_ratio", 0.70)
        and _metric_at_most(row, "debt_ratio", 0.50)
        and _metric_at_most(row, "total_borrowings_ratio", 0.20)
    )


def _has_secondary_rule_liquidity_watch_signal(bundle: Stage2InputBundle) -> bool:
    """Preserve hold for low-but-near-threshold eligible calls with liquidity rule watch."""
    if not _has_stage2_secondary_trigger(bundle):
        return False
    if _has_financial_statement_missing_placeholder(bundle.source_feature_row):
        return False
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability < threshold - 0.10:
        return False
    review_priority = _stage2_review_priority(bundle)
    if review_priority not in {"medium", "high", "critical"}:
        return False

    risk_band = str(bundle.rule_result.get("risk_band") or "").strip().lower()
    recommendation = str(bundle.rule_result.get("recommendation") or "").strip().lower()
    if risk_band not in {"watch", "관찰"} and recommendation not in {"watch", "review"}:
        return False

    raw_reasons = bundle.rule_result.get("reasons", [])
    reason_text = " ".join(str(reason) for reason in raw_reasons if str(reason).strip()).lower()
    liquidity_markers = (
        "current_ratio",
        "cash_ratio",
        "liquidity",
        "유동비율",
        "현금비율",
        "유동성",
    )
    has_reported_liquidity_weakness = _metric_below(
        bundle.source_feature_row, "current_ratio", 1.0
    ) or _metric_below(bundle.source_feature_row, "cash_ratio", 0.10)
    if has_reported_liquidity_weakness and _has_cashflow_backed_liquidity_buffer(
        bundle.source_feature_row
    ):
        return False
    if (
        any(marker in reason_text for marker in liquidity_markers)
        and has_reported_liquidity_weakness
    ):
        return True
    return bool(has_reported_liquidity_weakness)


def _has_cashflow_backed_liquidity_buffer(row: dict[str, Any]) -> bool:
    """Allow a current-ratio watch through when cash, OCF, and capital are strong."""
    return bool(
        _metric_below(row, "current_ratio", 1.0)
        and _metric_at_least(row, "cash_ratio", 0.25)
        and _metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and _metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and _metric_at_least(row, "ocf_to_sales", 0.0)
        and _metric_at_least(row, "interest_coverage_ratio", 3.0)
        and _metric_at_least(row, "equity_ratio", 0.40)
        and _metric_at_most(row, "debt_ratio", 1.50)
        and (
            _metric_at_most(row, "short_term_borrowings_share", 0.80)
            or _metric_at_most(row, "total_borrowings_ratio", 0.30)
        )
        and _metric_at_most(row, "capital_impairment_ratio", 0.0)
        and not _flag_is_true(row.get("icr_under_1"))
        and not _flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        and not _flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
        and not _metric_below(row, "net_margin", -0.05)
    )


def _secondary_overhold_guardrail_reason(bundle: Stage2InputBundle) -> str:
    """Keep defensive investment-grade cases from being held by secondary radar alone."""
    if bundle.prediction_label != "투자적격" or not _has_stage2_secondary_trigger(bundle):
        return ""

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability >= threshold:
        return ""
    if not bundle.source_feature_row:
        return ""
    stable_prior_cashflow_reason = _stable_prior_cashflow_overhold_guardrail_reason(bundle)
    if stable_prior_cashflow_reason:
        return stable_prior_cashflow_reason
    if _has_blocking_flags(bundle) and not _has_isolated_interest_cover_defense(bundle):
        return ""
    if _has_severe_financial_watch_signal(
        bundle.source_feature_row
    ) and not _has_isolated_interest_cover_defense(bundle):
        return ""
    if _has_extreme_financial_distress_signal(bundle.source_feature_row):
        return ""
    if _has_secondary_overhold_guardrail_blocker(
        bundle.source_feature_row
    ) and not _has_isolated_interest_cover_defense(bundle):
        return ""
    if _has_secondary_rule_liquidity_watch_signal(bundle):
        return ""
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if _material_financing_evidence_blocks_tn_hold(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if _prior_rating_is_speculative(bundle.prior_rating_reference):
        return ""

    supports = _secondary_overhold_guardrail_supports(bundle.source_feature_row)
    if len(supports) < 2 or "현금흐름" not in supports:
        return ""

    return (
        "정상기업 과잉 보류 방어 guardrail: 1차 모델은 투자적격이고 "
        f"투기등급 확률 {probability:.1%}가 기준선 {threshold:.1%} 아래입니다. "
        "직접 검증된 외부 치명근거와 강한 재무 부실 신호가 없고 "
        f"{', '.join(supports[:3])} 축이 방어적이어서 45개 보조 레이더 단독 신호만으로는 "
        "위험 보류나 경계 보류로 올리지 않습니다."
    )


def _stable_prior_cashflow_overhold_guardrail_reason(bundle: Stage2InputBundle) -> str:
    """Lower near-threshold TN holds when prior rating and OCF defense are strong."""
    if not _prior_rating_is_stable_investment_non_boundary(bundle.prior_rating_reference):
        return ""
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if _material_financing_evidence_blocks_tn_hold(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if _has_extreme_financial_distress_signal(bundle.source_feature_row):
        return ""
    if not _has_cashflow_backed_near_threshold_tn_defense(bundle.source_feature_row):
        return ""

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    prior = bundle.prior_rating_reference
    rating = str(prior.get("prior_credit_rating") or "").strip()
    rating_date = str(prior.get("prior_rating_date") or "").strip()
    agency = str(prior.get("prior_rating_agency") or "").strip()
    agency_text = f"{agency} " if agency else ""
    return (
        "정상기업 과잉 보류 방어 guardrail v2: 1차 모델은 투자적격이고 "
        f"투기등급 확률 {probability:.1%}가 기준선 {threshold:.1%} 아래입니다. "
        f"평가 기준일 이전 {agency_text}공개등급도 {rating}({rating_date})로 "
        "BBB-/BB+ 경계보다 위의 투자등급 영역입니다. 이자보상배율 단기 저하는 있으나 "
        "영업현금흐름·부채상환 현금흐름·자본잠식 부재·반복 손실 부재가 확인되고, "
        "직접 검증된 외부 치명근거도 없어 45개 보조 레이더의 경계 보류를 적격으로 "
        "낮춥니다."
    )


def _prior_rating_is_stable_investment_non_boundary(prior: dict[str, Any]) -> bool:
    if not prior or prior.get("has_prior_rating") is not True:
        return False
    if (
        str(prior.get("prior_rating_boundary_group") or "").strip()
        != "investment_grade_non_boundary"
    ):
        return False
    rank = _safe_int(prior.get("prior_credit_rating_rank"))
    if rank is not None:
        return bool(rank <= 8)
    rating = str(prior.get("prior_credit_rating") or "").strip().upper()
    return rating in {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+"}


def _has_cashflow_backed_near_threshold_tn_defense(row: dict[str, Any]) -> bool:
    """Allow eligible alignment when a single ICR dip is offset by cash generation."""
    if not (
        _flag_is_true(row.get("icr_under_1")) or _metric_below(row, "interest_coverage_ratio", 1.0)
    ):
        return False
    if _flag_is_true(row.get("is_2y_consecutive_operating_loss")) or _flag_is_true(
        row.get("is_2y_consecutive_ocf_deficit")
    ):
        return False
    if _metric_above(row, "capital_impairment_ratio", 0.0):
        return False
    if _metric_below(row, "net_margin", -0.10):
        return False

    cashflow_support = (
        _metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and _metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and _metric_at_least(row, "ocf_to_sales", 0.0)
    )
    balance_or_borrowing_support = _metric_at_least(row, "cash_ratio", 0.05) or _metric_at_most(
        row, "total_borrowings_ratio", 0.55
    )
    return bool(cashflow_support and balance_or_borrowing_support)


def _secondary_overhold_guardrail_supports(row: dict[str, Any]) -> list[str]:
    """Return broad financial-defense categories for TN over-hold prevention."""
    supports: list[str] = []
    liquidity_support = _metric_at_least(row, "current_ratio", 1.2) or _metric_at_least(
        row, "cash_ratio", 0.15
    )
    if liquidity_support:
        supports.append("유동성")

    cashflow_signal = (
        _metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        or _metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        or _metric_at_least(row, "ocf_to_sales", 0.0)
    )
    interest_service_signal = _metric_at_least(row, "interest_coverage_ratio", 1.0) and not (
        _flag_is_true(row.get("icr_under_1"))
    )
    if cashflow_signal and (
        interest_service_signal or _has_isolated_interest_cover_row_defense(row)
    ):
        supports.append("현금흐름")

    capital_support = (
        _metric_at_least(row, "equity_ratio", 0.40)
        and (
            _metric_at_most(row, "debt_ratio", 1.50)
            or _metric_at_most(row, "total_borrowings_ratio", 0.50)
        )
        and not _metric_above(row, "capital_impairment_ratio", 0.0)
    )
    if capital_support:
        supports.append("자본")
    return supports


def _has_secondary_overhold_guardrail_blocker(row: dict[str, Any]) -> bool:
    """Return moderate stress signals that should keep a near-boundary FN on hold."""
    if _metric_below(row, "net_margin", -0.10):
        return True
    if _metric_below(row, "ocf_to_sales", 0.0) and _metric_below(
        row, "ocf_to_total_liabilities", 0.0
    ):
        return True
    weak_interest_cover = _metric_below(row, "interest_coverage_ratio", 3.0)
    weak_capital_buffer = _metric_below(row, "equity_ratio", 0.40) and _metric_above(
        row, "debt_ratio", 1.50
    )
    return bool(weak_interest_cover and weak_capital_buffer)


def _has_financial_statement_missing_placeholder(row: dict[str, Any]) -> bool:
    """Detect rows where absent statements are encoded as zero/capped ratios."""
    return bool(
        _metric_at_most(row, "assets_total", 0.0)
        and _metric_at_most(row, "gross_profit", 0.0)
        and _metric_at_least(row, "interest_coverage_ratio", 999_999.0)
        and _metric_at_least(row, "cashflow_coverage_ratio", 999_999.0)
    )


def _rule_confidence_at_least(bundle: Stage2InputBundle, threshold: float) -> bool:
    confidence = _safe_float(bundle.rule_result.get("confidence"))
    return confidence is not None and confidence >= threshold


def _has_severe_financial_watch_signal(row: dict[str, Any]) -> bool:
    hard_stress_flags = [
        _flag_is_true(row.get("icr_under_1")),
        _flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        and _flag_is_true(row.get("is_2y_consecutive_ocf_deficit")),
        _metric_above(row, "capital_impairment_ratio", 0.0),
        _metric_below(row, "interest_coverage_ratio", 1.0),
    ]
    if any(hard_stress_flags):
        return True
    weak_liquidity = _metric_below(row, "current_ratio", 0.7) and _metric_below(
        row, "cash_ratio", 0.05
    )
    weak_cashflow = _metric_below(row, "cashflow_coverage_ratio", 0.0) or _metric_below(
        row, "ocf_to_total_liabilities", 0.0
    )
    return bool(weak_liquidity and weak_cashflow)


def _hidden_tail_risk_assessment(bundle: Stage2InputBundle) -> HiddenTailRiskAssessment:
    """Flag likely FN cases where external adverse evidence challenges an eligible model call."""
    if bundle.prediction_label != "투자적격":
        return HiddenTailRiskAssessment(False, "", 0, 0)
    adverse_items = _adverse_external_items(bundle.news_cache_snapshot)
    if not adverse_items:
        return HiddenTailRiskAssessment(False, "", 0, 0)

    verified_items = [
        item
        for item in adverse_items
        if _is_actionable_verified_adverse_external_item(item, bundle.news_cache_snapshot)
        and not _is_uncorroborated_material_financing_or_guarantee_item(
            item,
            source_feature_row=bundle.source_feature_row,
        )
    ]
    if not verified_items:
        return HiddenTailRiskAssessment(False, "", len(adverse_items), 0)

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    source_names = sorted({str(item.get("source", "external")) for item in verified_items})
    terms = sorted({term for item in adverse_items for term in _item_critical_terms(item)})
    terms_text = f" 위험 키워드: {', '.join(terms[:4])}." if terms else ""
    risk_signal = _hidden_tail_evidence_requires_risk_signal(
        verified_items,
        source_feature_row=bundle.source_feature_row,
    )
    if risk_signal:
        reason = (
            f"숨은 꼬리위험 보완 플래그: 모델은 투자적격(투기등급 확률 {probability:.1%}, "
            f"기준선 {threshold:.1%})으로 봤지만, 기업 직접 관련 외부 위험 근거 "
            f"{len(adverse_items)}건 중 검증 가능 근거 {len(verified_items)}건이 확인되어 "
            f"FN 가능성을 보수적으로 점검해야 합니다. 출처: {', '.join(source_names)}."
            f"{terms_text}"
        )
    else:
        reason = (
            f"외부근거 확인필요 보류 플래그: 모델은 투자적격(투기등급 확률 {probability:.1%}, "
            f"기준선 {threshold:.1%})으로 봤지만, 자금조달·채무보증 등 규모성 공시 "
            f"{len(verified_items)}건이 확인되어 추가 점검이 필요합니다. 다만 치명 문맥이나 "
            "현금흐름 악화가 함께 확인된 실질 부실 근거는 제한적이므로 위험 보류가 아닌 "
            f"확인필요 보류로 분리합니다. 출처: {', '.join(source_names)}."
        )
    return HiddenTailRiskAssessment(
        True,
        reason,
        len(adverse_items),
        len(verified_items),
        risk_signal,
    )


def _secondary_review_risk_assessment(bundle: Stage2InputBundle) -> SecondaryReviewRiskAssessment:
    """Flag likely FN cases surfaced by the 45-feature Stage 2 review radar."""
    if bundle.prediction_label != "투자적격" or not _has_stage2_secondary_trigger(bundle):
        return SecondaryReviewRiskAssessment(False, "", "none")

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    review_priority = _stage2_review_priority(bundle)
    # Near-threshold review should not automatically turn every stable call into
    # "보류". Rolling validation showed low absolute probabilities near a low
    # fold threshold can create unnecessary holds, so keep a minimum risk floor.
    probability_floor = max(0.28, threshold - 0.10)
    meets_probability_floor = probability >= probability_floor
    near_threshold = probability >= threshold - 0.10 and meets_probability_floor
    priority_requires_hold = review_priority in {"medium", "high", "critical"}
    severe_watch = _has_severe_financial_watch_signal(bundle.source_feature_row)
    secondary_liquidity_watch = _has_secondary_rule_liquidity_watch_signal(bundle)
    rule_liquidity_watch = secondary_liquidity_watch and (
        meets_probability_floor or (threshold >= 0.28 and _rule_confidence_at_least(bundle, 0.60))
    )
    if _secondary_overhold_guardrail_reason(bundle):
        return SecondaryReviewRiskAssessment(False, "", review_priority)
    risk_signal_floor = max(0.28, threshold - 0.04)
    risk_signal_corroborated = _secondary_review_risk_signal_corroborated(
        bundle,
        severe_watch=severe_watch,
        rule_liquidity_watch=rule_liquidity_watch,
    )
    risk_signal = probability >= risk_signal_floor and risk_signal_corroborated
    if not (
        ((near_threshold or priority_requires_hold) and meets_probability_floor)
        or severe_watch
        or rule_liquidity_watch
    ):
        return SecondaryReviewRiskAssessment(False, "", review_priority)

    trigger_reason = _stage2_trigger_reason(bundle)
    reason_parts = [
        "2차 보조 레이더 플래그: 43개 모델은 투자적격으로 봤지만 "
        "45개 보조 변수셋이 추가 검토 대상으로 올렸습니다.",
        f"투기등급 확률은 {probability:.1%}, 기준선은 {threshold:.1%}, "
        f"검토 우선순위는 {review_priority}, 최소 보류 검토 확률선은 "
        f"{probability_floor:.1%}입니다.",
    ]
    if risk_signal:
        reason_parts.append(
            f"확률이 위험신호 표시 기준선({risk_signal_floor:.1%}) 이상이라 "
            "사용자 화면에서는 위험 보류로 표시합니다."
        )
    elif probability >= risk_signal_floor:
        reason_parts.append(
            f"확률은 위험신호 표시 기준선({risk_signal_floor:.1%}) 이상이지만 "
            "직접 adverse 외부근거, 반복·고위험 자금조달, 심각 재무 watch 같은 위험 보강 "
            "근거가 부족해 사용자 화면에서는 확인필요 보류로 분리합니다."
        )
    else:
        reason_parts.append(
            f"확률이 위험신호 표시 기준선({risk_signal_floor:.1%}) 미만이라 "
            "사용자 화면에서는 확인필요 보류로 분리합니다."
        )
    if trigger_reason:
        reason_parts.append(trigger_reason)
    if rule_liquidity_watch and risk_signal:
        reason_parts.append(
            "룰 엔진도 유동성 watch 신호를 냈기 때문에 낮은 확률 바닥선만으로 "
            "적격 확정하지 않고 보류를 유지합니다."
        )
    elif rule_liquidity_watch:
        reason_parts.append(
            "룰 엔진의 유동성 watch 신호는 보류 근거로 반영하되, 단독으로는 "
            "위험 보류 확정 신호로 보지 않습니다."
        )
    reason_parts.append("따라서 2차 위원회는 이를 최종 적격으로 확정하지 않고 보류로 재점검합니다.")
    return SecondaryReviewRiskAssessment(
        True,
        " ".join(reason_parts),
        review_priority,
        risk_signal=risk_signal,
    )


def _secondary_review_risk_signal_corroborated(
    bundle: Stage2InputBundle,
    *,
    severe_watch: bool,
    rule_liquidity_watch: bool,
) -> bool:
    """Require corroboration before showing a secondary review hold as a risk signal."""
    if severe_watch:
        return not _has_isolated_icr_review_buffer(bundle.source_feature_row)
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return True
    if _material_financing_evidence_blocks_tn_hold(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return True
    prior = bundle.prior_rating_reference
    return _prior_rating_is_speculative(prior) or _prior_rating_is_exact_boundary(prior)


def _boundary_review_assessment(
    bundle: Stage2InputBundle,
    *,
    committee_label: CommitteeLabel,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
) -> BoundaryReviewAssessment:
    """Separate near-boundary holds from hard risk or over-warning holds."""
    if committee_label != "보류" or veto_triggered or hidden_tail_risk.triggered:
        return BoundaryReviewAssessment(False, "")
    if reject_confirmation.triggered:
        return BoundaryReviewAssessment(False, "")

    prior_boundary_reason = _prior_rating_boundary_hold_reason(
        bundle,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
    )
    if prior_boundary_reason:
        return BoundaryReviewAssessment(True, prior_boundary_reason)

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    margin = probability - threshold
    boundary_margin = 0.04
    if abs(margin) > boundary_margin:
        return BoundaryReviewAssessment(False, "")
    if not (secondary_review_risk.triggered or overwarning_mitigation.triggered):
        return BoundaryReviewAssessment(False, "")
    if secondary_review_risk.triggered and secondary_review_risk.risk_signal:
        return BoundaryReviewAssessment(False, "")

    direction = "투기등급 쪽" if margin >= 0 else "투자적격 쪽"
    reason = (
        f"경계등급 보류 플래그: 투기등급 확률이 {probability:.1%}, 기준선이 "
        f"{threshold:.1%}로 차이가 {abs(margin):.1%}p에 불과합니다. "
        "모델 기준선 경계에서 판단이 흔들릴 수 있는 구간이므로, "
        f"{direction}으로 확정하기보다 추가 근거 확인 대상으로 분리합니다."
    )
    return BoundaryReviewAssessment(True, reason)


def _prior_rating_boundary_hold_reason(
    bundle: Stage2InputBundle,
    *,
    committee_label: CommitteeLabel,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> str:
    """Return a non-leaky prior-rating boundary hold reason when applicable."""
    if veto_triggered or hidden_tail_risk.triggered:
        return ""
    prior = bundle.prior_rating_reference
    if not _prior_rating_is_exact_boundary(prior):
        return ""
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if committee_label == "부적격" and probability > max(threshold + 0.20, 0.55):
        return ""

    rating = str(prior.get("prior_credit_rating") or "").strip()
    rating_date = str(prior.get("prior_rating_date") or "").strip()
    agency = str(prior.get("prior_rating_agency") or "").strip()
    age_days = _safe_int(prior.get("prior_rating_age_days"))
    age_text = f", 기준일 대비 {age_days}일 전 공개" if age_days is not None else ""
    source_text = f"{agency} " if agency else ""
    return (
        f"경계등급 보류 플래그: {source_text}이전 공개등급이 {rating}"
        f"({rating_date}{age_text})로 BBB-/BB+ 경계권에 있습니다. "
        "이 정보는 평가 대상 시점 이전에 공개된 prior rating reference에서만 가져온 "
        "비누수 입력입니다. 모델 확률과 외부근거를 함께 보더라도 즉시 확정하기보다 "
        "투자적격/투기등급 경계 재확인 대상으로 분리합니다."
    )


def _prior_rating_is_exact_boundary(prior: dict[str, Any]) -> bool:
    if not prior or prior.get("has_prior_rating") is not True:
        return False
    group = str(prior.get("prior_rating_boundary_group") or "").strip()
    if group == "exact_bbb_minus_bb_plus_boundary":
        return True
    rating = str(prior.get("prior_credit_rating") or "").strip()
    return rating in {"BBB-", "BB+"}


def _prior_rating_boundary_requires_hold(bundle: Stage2InputBundle) -> bool:
    """Hold prior BBB-/BB+ cases only when the model is not clearly far from risk."""
    if bundle.prediction_label == "부적격":
        return True
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability >= max(threshold - 0.10, 0.20):
        return True
    return bool(
        bundle.model_view.get("stage2_review_trigger")
        or bundle.model_view.get("stage2_secondary_trigger")
    )


def _overwarning_mitigation_assessment(
    bundle: Stage2InputBundle,
    *,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    mitigating_factors: list[str],
) -> OverwarningMitigationAssessment:
    """Soften likely over-warning cases to hold, not eligible."""
    if bundle.prediction_label != "부적격" or veto_triggered or hidden_tail_risk.triggered:
        return OverwarningMitigationAssessment(False, "")
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return OverwarningMitigationAssessment(False, "")

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    near_threshold = probability <= threshold + 0.10
    watch_band = str(
        bundle.model_view.get("risk_band")
        or bundle.xgboost_result.get("risk_band")
        or bundle.rule_result.get("risk_band")
        or ""
    ).lower() in {"watch", "관찰"}
    explicit_overwarning = bool(bundle.model_view.get("stage2_overwarning_filter_candidate"))
    financial_resilience = _financial_resilience_overwarning_assessment(bundle.source_feature_row)
    cash_rich_loss_stage_buffer_reason = _cash_rich_loss_stage_overwarning_buffer_reason(bundle)
    noncritical_evidence = _noncritical_external_evidence_assessment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    near_threshold_buffer = near_threshold and (
        noncritical_evidence.triggered
        or _no_direct_external_items(bundle.news_cache_snapshot)
        or _external_evidence_unavailable(bundle.news_status)
        or bool(bundle.news_cache_snapshot.get("items"))
    )
    model_only_buffer_reason = _model_only_overwarning_buffer_reason(
        bundle,
        mitigating_factors=mitigating_factors,
    )
    prior_boundary_buffer_reason = _prior_boundary_overwarning_buffer_reason(
        bundle,
        noncritical_evidence=noncritical_evidence,
    )
    if not (
        near_threshold
        or watch_band
        or explicit_overwarning
        or financial_resilience.triggered
        or cash_rich_loss_stage_buffer_reason
        or near_threshold_buffer
        or model_only_buffer_reason
        or prior_boundary_buffer_reason
    ):
        return OverwarningMitigationAssessment(False, "")
    if (
        not mitigating_factors
        and not explicit_overwarning
        and not financial_resilience.triggered
        and not cash_rich_loss_stage_buffer_reason
        and not noncritical_evidence.triggered
        and not near_threshold_buffer
        and not model_only_buffer_reason
        and not prior_boundary_buffer_reason
    ):
        return OverwarningMitigationAssessment(False, "")

    reason_parts = [
        "과민 경고 완화 검토: 1차 모델은 부적격이지만 강한 외부 위험 근거는 확인되지 않았습니다."
    ]
    if near_threshold:
        reason_parts.append(
            f"위험확률이 기준선 근처입니다({probability:.1%} vs 기준선 {threshold:.1%})."
        )
    if near_threshold_buffer and not noncritical_evidence.triggered:
        reason_parts.append(
            "기준선 근처 경고이고 외부근거에서 부적격 확정을 막을 실질 adverse 근거가 "
            "확인되지 않아 즉시 위험 보류보다 과민경고 완화 보류로 재점검합니다."
        )
    if explicit_overwarning:
        reason = str(bundle.model_view.get("overwarning_filter_reason") or "").strip()
        if reason:
            reason_parts.append(reason)
    if financial_resilience.triggered:
        reason_parts.append(financial_resilience.reason)
    if cash_rich_loss_stage_buffer_reason:
        reason_parts.append(cash_rich_loss_stage_buffer_reason)
    if noncritical_evidence.triggered:
        reason_parts.append(noncritical_evidence.reason)
    if model_only_buffer_reason:
        reason_parts.append(model_only_buffer_reason)
    if prior_boundary_buffer_reason:
        reason_parts.append(prior_boundary_buffer_reason)
    return OverwarningMitigationAssessment(True, " ".join(reason_parts))


def _cash_rich_loss_stage_overwarning_buffer_reason(bundle: Stage2InputBundle) -> str:
    """Soften high model warnings when losses are buffered by unusually strong liquidity."""
    row = bundle.source_feature_row
    probability = bundle.probability_speculative
    if probability < 0.85:
        return ""
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if not (
        _metric_at_least(row, "current_ratio", 2.0)
        and _metric_at_least(row, "cash_ratio", 0.50)
        and _metric_at_least(row, "equity_ratio", 0.60)
        and _metric_at_most(row, "debt_ratio", 0.50)
        and _metric_at_most(row, "total_borrowings_ratio", 0.10)
    ):
        return ""
    if _flag_is_true(row.get("is_2y_consecutive_operating_loss")) or _flag_is_true(
        row.get("is_2y_consecutive_ocf_deficit")
    ):
        return ""
    if not (
        _metric_at_least(row, "cashflow_coverage_ratio", 0.0)
        or _metric_at_least(row, "ocf_to_total_liabilities", 0.0)
        or _metric_at_least(row, "ocf_to_sales", 0.0)
    ):
        return ""
    return (
        f"현금·자본 버퍼 기반 과민경고 완화: 투기등급 확률은 {probability:.1%}로 높지만 "
        "유동비율·현금비율·자기자본비율이 높고 차입 부담이 낮으며, 반복 영업손실이나 "
        "반복 OCF 적자는 확인되지 않았습니다. 현재 손익성 악화가 즉시 부도위험으로 "
        "연결되는지 추가 확인이 필요하므로 부적격 확정보다는 보류로 완화합니다."
    )


def _prior_boundary_overwarning_buffer_reason(
    bundle: Stage2InputBundle,
    *,
    noncritical_evidence: NoncriticalEvidenceAssessment,
) -> str:
    """Soften high-probability boundary-grade warnings unless distress is decisive."""
    prior = bundle.prior_rating_reference
    if not _prior_rating_is_exact_boundary(prior):
        return ""
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability <= max(threshold + 0.20, 0.55):
        return ""
    if _has_extreme_financial_distress_signal(bundle.source_feature_row):
        return ""
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if not (
        noncritical_evidence.triggered
        or _has_nonblocking_external_context(bundle.news_cache_snapshot)
        or _no_direct_external_items(bundle.news_cache_snapshot)
        or _external_evidence_unavailable(bundle.news_status)
    ):
        return ""

    rating = str(prior.get("prior_credit_rating") or "").strip()
    rating_date = str(prior.get("prior_rating_date") or "").strip()
    agency = str(prior.get("prior_rating_agency") or "").strip()
    source_text = f"{agency} " if agency else ""
    return (
        f"경계등급 과민경고 완화: 모델 확률은 {probability:.1%}로 높지만 "
        f"평가 기준일 이전 {source_text}공개등급이 {rating}({rating_date})로 "
        "BBB-/BB+ 경계권에 있고, 직접 관련 외부근거도 치명·adverse 수준으로 "
        "확인되지 않았습니다. 자본잠식·극단적 레버리지·만기집중 현금흐름 악화 같은 "
        "결정적 차단 신호가 없는 경우에는 즉시 부적격 확정보다 보류로 완화해 "
        "과민경고 여부를 재점검합니다."
    )


def _reject_confirmation_assessment(
    bundle: Stage2InputBundle,
    *,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> RejectConfirmationAssessment:
    """Require at least two strong signals before converting a model warning to reject."""
    if veto_triggered:
        return RejectConfirmationAssessment(True, False, "veto gate", 1, ("veto",))
    if bundle.prediction_label != "부적격" or hidden_tail_risk.triggered:
        return RejectConfirmationAssessment(True, False, "", 0, ())

    probability = bundle.probability_speculative
    very_high_model_warning = probability >= 0.90
    direct_adverse_items = _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    direct_adverse_evidence = bool(direct_adverse_items)
    hard_external_confirmation = _has_hard_reject_external_confirmation(direct_adverse_items)
    extreme_financial_distress = _has_extreme_financial_distress_signal(bundle.source_feature_row)
    severe_financial_watch = very_high_model_warning and _has_severe_financial_watch_signal(
        bundle.source_feature_row
    )
    signals = []
    if very_high_model_warning:
        signals.append(f"고확률 모델 경고({probability:.1%})")
    if direct_adverse_evidence:
        signals.append("직접 adverse 외부근거")
    if extreme_financial_distress:
        signals.append("극단 재무위험")
    elif severe_financial_watch:
        signals.append("강한 재무위험")

    signal_count = len(signals)
    if signal_count >= 2 and (hard_external_confirmation or extreme_financial_distress):
        reason = (
            "부적격 확정 게이트 통과: "
            f"{', '.join(signals)} 신호가 함께 확인되어 모델의 부적격 경고를 "
            "위원회 부적격 의견으로 확정할 수 있습니다."
        )
        return RejectConfirmationAssessment(True, False, reason, signal_count, tuple(signals))

    if signal_count >= 2:
        reason = (
            "부적격 확정 게이트 부분 충족: "
            f"{', '.join(signals)} 신호가 확인되었지만, 치명 외부근거나 극단 재무위험이 "
            "함께 확인되지는 않았습니다. 부적격 확정은 유보하고 위험 보류로 "
            "추가 근거 확인이 필요합니다."
        )
        return RejectConfirmationAssessment(
            False,
            True,
            reason,
            signal_count,
            tuple(signals),
            True,
            "고확률 모델 경고와 재무/외부 watch 신호는 있으나 확정형 부실 근거는 제한적입니다.",
        )

    review_risk_signal, review_risk_reason = _unconfirmed_reject_review_risk_assessment(
        bundle,
        probability=probability,
    )
    if review_risk_signal:
        reason = (
            "부적격 확정 게이트 미충족: 1차 모델은 부적격으로 봤지만, "
            "고확률 모델 경고·직접 adverse 외부근거·극단 재무위험 중 "
            f"{signal_count}개 신호만 확인되었습니다. 따라서 부적격 확정은 유보합니다. "
            f"다만 {review_risk_reason} 부적격 확정보다는 낮은 단계인 위험 보류로 "
            "표시하고 추가 근거 확인이 필요합니다."
        )
        return RejectConfirmationAssessment(
            False,
            True,
            reason,
            signal_count,
            tuple(signals),
            True,
            review_risk_reason,
        )

    reason = (
        "부적격 확정 게이트 미충족: 1차 모델은 부적격으로 봤지만, "
        "고확률 모델 경고·직접 adverse 외부근거·극단 재무위험 중 "
        f"{signal_count}개 신호만 확인되었습니다. 따라서 부적격 확정보다는 "
        "확인필요 보류로 두고 추가 근거 확인이 필요합니다."
    )
    return RejectConfirmationAssessment(False, True, reason, signal_count, tuple(signals))


def _has_hard_reject_external_confirmation(items: list[dict[str, Any]]) -> bool:
    """Return whether adverse external evidence is hard enough to confirm reject."""
    for item in items:
        if item.get("veto_candidate") is True or item.get("critical_context_confirmed") is True:
            return True
        if _shared_has_hard_distress_terms(item):
            return True
    return False


def _unconfirmed_reject_review_risk_assessment(
    bundle: Stage2InputBundle,
    *,
    probability: float,
) -> tuple[bool, str]:
    """Upgrade unconfirmed reject holds only when corroborating watch signals exist."""
    repeated_financing_count = _repeated_financing_evidence_count(bundle.news_cache_snapshot)
    threshold = _model_threshold(bundle)
    near_very_high_probability = probability >= max(0.88, threshold + 0.55)
    strong_probability = probability >= max(0.80, threshold + 0.35)
    prior_speculative = _prior_rating_is_speculative(bundle.prior_rating_reference)

    reasons: list[str] = []
    if near_very_high_probability and repeated_financing_count >= 2:
        reasons.append(
            f"투기등급 확률이 {probability:.1%}로 높고, 전환사채·유상증자 등 "
            f"자금조달성 공시가 {repeated_financing_count}건 반복 확인되었습니다."
        )
    if strong_probability and prior_speculative:
        prior = bundle.prior_rating_reference
        rating = str(prior.get("prior_credit_rating") or "").strip()
        rating_date = str(prior.get("prior_rating_date") or "").strip()
        reasons.append(
            f"투기등급 확률이 {probability:.1%}이고, 평가 기준일 이전 공개등급이 "
            f"{rating}({rating_date})로 이미 투기등급 영역에 있었습니다."
        )

    if not reasons:
        return False, ""
    return True, " ".join(reasons)


def _repeated_financing_evidence_count(news_cache: dict[str, Any]) -> int:
    return len(_financing_evidence_items(news_cache))


def _material_financing_evidence_blocks_tn_hold(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    """Block TN overhold relief only for repeated or explicitly high-risk financing."""
    return bool(
        _shared_material_financing_evidence_blocks_tn_hold(
            news_cache,
            source_feature_row=source_feature_row,
        )
    )


def _high_risk_financing_evidence_count(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> int:
    return int(
        _shared_high_risk_financing_evidence_count(
            news_cache,
            source_feature_row=source_feature_row,
        )
    )


def _financing_evidence_items(news_cache: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], _shared_financing_evidence_items(news_cache))


def _is_uncorroborated_material_financing_or_guarantee_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None,
) -> bool:
    """Treat material financing/guarantee as contextual unless distress corroborates it."""
    return bool(
        _shared_is_uncorroborated_material_financing_or_guarantee_item(
            item,
            source_feature_row=source_feature_row,
        )
    )


def _hidden_tail_evidence_requires_risk_signal(
    items: list[dict[str, Any]],
    *,
    source_feature_row: dict[str, Any],
) -> bool:
    return bool(
        _shared_hidden_tail_evidence_requires_risk_signal(
            items,
            source_feature_row=source_feature_row,
        )
    )


def _is_material_financing_or_guarantee_item(item: dict[str, Any]) -> bool:
    return bool(_shared_is_material_financing_or_guarantee_item(item))


def _has_hard_distress_terms(item: dict[str, Any]) -> bool:
    return bool(_shared_has_hard_distress_terms(item))


def _material_financing_or_guarantee_has_financial_corroboration(
    row: dict[str, Any],
) -> bool:
    return bool(_shared_material_financing_or_guarantee_has_financial_corroboration(row))


def _material_financing_or_guarantee_has_severe_financial_corroboration(
    row: dict[str, Any],
) -> bool:
    return bool(_shared_material_financing_or_guarantee_has_severe_financial_corroboration(row))


def _financial_observation_count(row: dict[str, Any]) -> int:
    keys = (
        "cashflow_coverage_ratio",
        "ocf_to_total_liabilities",
        "ocf_to_sales",
        "interest_coverage_ratio",
        "equity_ratio",
        "debt_ratio",
        "total_borrowings_ratio",
        "current_ratio",
        "cash_ratio",
        "net_margin",
    )
    return sum(1 for key in keys if row.get(key) is not None)


def _prior_rating_is_speculative(prior: dict[str, Any]) -> bool:
    if not prior or prior.get("has_prior_rating") is not True:
        return False
    rank = _safe_int(prior.get("prior_credit_rating_rank"))
    if rank is not None:
        return bool(rank >= 11)
    rating = str(prior.get("prior_credit_rating") or "").strip().upper()
    return rating in {"BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"}


def _compact_text(text: str) -> str:
    return "".join(str(text).lower().split())


def _model_only_overwarning_buffer_reason(
    bundle: Stage2InputBundle,
    *,
    mitigating_factors: list[str],
) -> str:
    """Downgrade unsupported high-probability reject calls to hold, not eligible."""
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability <= threshold + 0.10:
        return ""
    # Above this level, keep the model's severe warning unless another explicit
    # mitigation screen such as financial resilience or non-critical evidence fires.
    if probability >= 0.90:
        return ""
    if not mitigating_factors:
        return ""
    cashflow_backed_resilience = _has_cashflow_backed_fp_resilience(bundle.source_feature_row)
    if not cashflow_backed_resilience:
        return ""
    if _has_blocking_flags(bundle) and not cashflow_backed_resilience:
        return ""
    if (
        _has_severe_financial_watch_signal(bundle.source_feature_row)
        and not cashflow_backed_resilience
    ):
        return ""
    if _overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""

    news_status = bundle.news_status
    if not _external_evidence_unavailable(news_status) and not _no_direct_external_items(
        bundle.news_cache_snapshot
    ):
        return ""
    if cashflow_backed_resilience:
        return (
            f"고확률 모델 단독 경고 완화: 투기등급 확률은 {probability:.1%}로 높고 "
            "일부 손익·이자보상 스트레스가 있지만, OCF와 자본/부채 구조가 방어력을 "
            "제공합니다. 직접 외부 치명근거도 확인되지 않았으므로 즉시 부적격 "
            "확정보다는 보류로 재점검합니다."
        )
    return (
        f"고확률 모델 단독 경고 완화: 투기등급 확률은 {probability:.1%}로 높지만 "
        "부적격 확정을 뒷받침하는 veto·직접 외부 치명근거·강한 재무 차단 신호는 "
        "확인되지 않았습니다. 완화 요인이 일부 있으므로 즉시 부적격 확정보다는 "
        "보류로 재점검합니다."
    )


def _has_cashflow_backed_fp_resilience(row: dict[str, Any]) -> bool:
    """Allow hold, not reject, when stress is offset by cash-flow and balance-sheet buffers."""
    if _flag_is_true(row.get("is_2y_consecutive_ocf_deficit")):
        return False
    if _metric_above(row, "capital_impairment_ratio", 0.0):
        return False
    cashflow_support = (
        _metric_at_least(row, "cashflow_coverage_ratio", 0.0)
        or _metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        or _metric_at_least(row, "ocf_to_sales", 0.0)
    )
    balance_sheet_support = _metric_at_least(row, "equity_ratio", 0.40) and _metric_at_most(
        row, "debt_ratio", 1.50
    )
    borrowing_support = _metric_at_most(row, "total_borrowings_ratio", 0.40) or _metric_at_most(
        row, "short_term_borrowings_share", 0.70
    )
    return bool(cashflow_support and balance_sheet_support and borrowing_support)


def _has_extreme_financial_distress_signal(row: dict[str, Any]) -> bool:
    """Return whether financial stress is too severe to soften a boundary warning."""
    if _metric_above(row, "capital_impairment_ratio", 0.50):
        return True
    if _metric_below(row, "equity_ratio", 0.15):
        return True
    if _metric_above(row, "debt_ratio", 5.0):
        return True

    short_term_maturity_wall = _metric_at_least(row, "short_term_borrowings_share", 0.95)
    weak_cashflow = (
        _metric_below(row, "cashflow_coverage_ratio", 0.0)
        or _metric_below(row, "ocf_to_total_liabilities", 0.0)
        or _metric_below(row, "ocf_to_sales", 0.0)
    )
    recurring_loss_or_ocf_deficit = _flag_is_true(
        row.get("is_2y_consecutive_operating_loss")
    ) or _flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
    interest_blocked = _flag_is_true(row.get("icr_under_1")) or _metric_below(
        row, "interest_coverage_ratio", 1.0
    )
    return bool(
        short_term_maturity_wall
        and weak_cashflow
        and recurring_loss_or_ocf_deficit
        and interest_blocked
    )


def _noncritical_external_evidence_assessment(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> NoncriticalEvidenceAssessment:
    """Treat verified but non-critical external evidence as a reason to hold, not reject."""
    status = str(news_cache.get("status", "")).strip().lower()
    if status != "ready":
        return NoncriticalEvidenceAssessment(False, "", 0, 0)
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return NoncriticalEvidenceAssessment(False, "", 0, 0)
    direct_items = [
        item for item in raw_items if isinstance(item, dict) and item.get("company_match") is True
    ]
    if not direct_items:
        return NoncriticalEvidenceAssessment(False, "", 0, 0)
    blocking_items = [
        item
        for item in direct_items
        if _is_blocking_external_adverse_item(
            item,
            source_feature_row=source_feature_row,
        )
    ]
    if blocking_items:
        return NoncriticalEvidenceAssessment(False, "", len(direct_items), len(blocking_items))
    contextual_items = [
        item
        for item in direct_items
        if str(item.get("disclosure_severity", "")).lower() in {"routine", "caution"}
        or str(item.get("provider_relevance", "")).lower() in {"routine", "context", "caution"}
        or str(item.get("source", "")).lower() in {"opendart", "naver_news", "tavily"}
    ]
    if not contextual_items:
        return NoncriticalEvidenceAssessment(False, "", len(direct_items), 0)
    reason = (
        f"외부근거 완화 신호: 직접 관련 근거 {len(direct_items)}건을 수집했지만 "
        "강제 경고·치명 키워드·실질 adverse 공시는 확인되지 않았습니다. "
        "따라서 2차 위원회는 모델의 부적격 경고를 확정하기보다 보류로 재점검합니다."
    )
    return NoncriticalEvidenceAssessment(True, reason, len(direct_items), 0)


def _no_direct_external_items(news_cache: dict[str, Any]) -> bool:
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return True
    return not any(
        isinstance(item, dict) and item.get("company_match") is True for item in raw_items
    )


def _has_nonblocking_external_context(news_cache: dict[str, Any]) -> bool:
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    has_direct_item = any(
        isinstance(item, dict) and item.get("company_match") is True for item in raw_items
    )
    return has_direct_item and not _overwarning_blocking_external_items(news_cache)


def _financial_resilience_overwarning_assessment(
    row: dict[str, Any],
) -> FinancialResilienceAssessment:
    """Detect high-risk model calls that still show broad financial defense capacity."""
    support_checks = [
        ("유동비율 1.2배 이상", _metric_at_least(row, "current_ratio", 1.2)),
        ("현금비율 15% 이상", _metric_at_least(row, "cash_ratio", 0.15)),
        ("자기자본비율 40% 이상", _metric_at_least(row, "equity_ratio", 0.40)),
        ("부채비율 150% 이하", _metric_at_most(row, "debt_ratio", 1.50)),
        ("총차입금 비중 50% 이하", _metric_at_most(row, "total_borrowings_ratio", 0.50)),
        ("자본잠식 신호 없음", _metric_at_most(row, "capital_impairment_ratio", 0.0)),
        ("이자보상배율 1배 이상", _metric_at_least(row, "interest_coverage_ratio", 1.0)),
        ("순이익률 흑자", _metric_at_least(row, "net_margin", 0.0)),
        ("OCF/매출액 양수", _metric_at_least(row, "ocf_to_sales", 0.0)),
        ("2년 연속 영업손실 아님", _flag_is_false(row.get("is_2y_consecutive_operating_loss"))),
        ("2년 연속 OCF 적자 아님", _flag_is_false(row.get("is_2y_consecutive_ocf_deficit"))),
        ("ICR 1 미만 플래그 없음", _flag_is_false(row.get("icr_under_1"))),
        ("단기차입금 비중 80% 이하", _metric_at_most(row, "short_term_borrowings_share", 0.80)),
    ]
    blocker_checks = [
        _flag_is_true(row.get("is_2y_consecutive_operating_loss")),
        _flag_is_true(row.get("is_2y_consecutive_ocf_deficit")),
        _flag_is_true(row.get("icr_under_1")),
        _metric_below(row, "net_margin", -0.10),
        _metric_below(row, "equity_ratio", 0.25),
        _metric_above(row, "capital_impairment_ratio", 0.0),
        _metric_above(row, "total_borrowings_ratio", 0.65),
        _metric_above(row, "short_term_borrowings_share", 0.90),
    ]
    active_supports = [label for label, passed in support_checks if passed]
    support_count = len(active_supports)
    blocker_count = sum(1 for passed in blocker_checks if passed)
    core_defense = (
        _metric_at_least(row, "current_ratio", 1.2)
        and _metric_at_least(row, "cash_ratio", 0.15)
        and _metric_at_least(row, "equity_ratio", 0.40)
        and _metric_at_most(row, "debt_ratio", 1.50)
        and _metric_at_least(row, "interest_coverage_ratio", 1.0)
        and _metric_at_least(row, "net_margin", 0.0)
    )
    triggered = core_defense and support_count >= 8 and blocker_count == 0
    if not triggered:
        return FinancialResilienceAssessment(False, "", support_count, blocker_count)
    reason = (
        f"고확률 과민 경고 방어 신호: 유동성·현금·자본·이자보상·순이익률 핵심 조건과 "
        f"재무 방어 조건 {support_count}개가 충족되고 "
        f"강한 차단 신호는 {blocker_count}개입니다. "
        f"대표 완화 신호는 {', '.join(active_supports[:4])}입니다."
    )
    return FinancialResilienceAssessment(True, reason, support_count, blocker_count)


def _adverse_external_items(news_cache: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return []
    adverse_items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if item.get("company_match") is not True:
            continue
        if _is_adverse_external_item(item):
            adverse_items.append(item)
    return adverse_items


def _verified_adverse_external_items(news_cache: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in _adverse_external_items(news_cache)
        if _is_verified_adverse_external_item(item)
    ]


def _is_actionable_verified_adverse_external_item(
    item: dict[str, Any],
    news_cache: dict[str, Any],
) -> bool:
    """Return whether a verified adverse item is actionable for committee escalation."""
    return (
        _is_verified_adverse_external_item(item)
        and not _is_noisy_aggregated_news_item(item)
        and not _is_resolved_procedural_trading_halt_item(item, news_cache)
    )


def _overwarning_blocking_external_items(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return external items strong enough to block FP mitigation."""
    return [
        item
        for item in _verified_adverse_external_items(news_cache)
        if _is_actionable_verified_adverse_external_item(item, news_cache)
        and not _is_uncorroborated_material_financing_or_guarantee_item(
            item,
            source_feature_row=source_feature_row,
        )
    ]


def _is_adverse_external_item(item: dict[str, Any]) -> bool:
    if item.get("veto_candidate") is True:
        return True
    severity = str(item.get("disclosure_severity", "")).lower()
    if severity in {"veto", "adverse"}:
        if str(item.get("source", "")).lower() == "opendart":
            return True
        return item.get("critical_context_confirmed") is True
    if severity in {"routine", "caution"}:
        return False
    if item.get("critical_context_confirmed") is True:
        return True
    if str(item.get("provider_relevance", "")).lower() in ADVERSE_PROVIDER_RELEVANCE:
        return True
    terms = _item_critical_terms(item)
    if not terms:
        return False
    return str(item.get("evidence_quality", "")).lower() in ADVERSE_EVIDENCE_QUALITY


def _is_blocking_external_adverse_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    """Return whether an evidence item should prevent FP mitigation."""
    if _is_uncorroborated_material_financing_or_guarantee_item(
        item,
        source_feature_row=source_feature_row,
    ):
        return False
    if item.get("veto_candidate") is True:
        return True
    source = str(item.get("source", "")).lower()
    severity = str(item.get("disclosure_severity", "")).lower()
    if severity in {"veto", "adverse"}:
        return source == "opendart" or item.get("critical_context_confirmed") is True
    # Keyword hits from aggregated news snippets are noisy. They should block
    # over-warning mitigation only when the collector confirmed the risky context.
    return item.get("critical_context_confirmed") is True


def _is_noisy_aggregated_news_item(item: dict[str, Any]) -> bool:
    """Detect multi-company market wrap snippets where risk terms can belong elsewhere."""
    source = str(item.get("source", "")).lower()
    if source not in {"naver_news", "tavily"}:
        return False
    if not _item_critical_terms(item):
        return False
    title = _compact_text(str(item.get("title", "")))
    noisy_title_markers = (
        "주요공시",
        "기업공시",
        "주요종목뉴스",
        "장마감후주요종목뉴스",
        "전일주요공시",
    )
    return any(marker in title for marker in noisy_title_markers)


def _is_resolved_procedural_trading_halt_item(
    item: dict[str, Any],
    news_cache: dict[str, Any],
) -> bool:
    """Do not treat resolved procedural trading-halt checks as hard adverse blockers."""
    if str(item.get("source", "")).lower() != "opendart":
        return False
    text = _compact_text(" ".join(str(item.get(key, "")) for key in ("title", "summary")))
    if "거래정지" not in text:
        return False
    hard_markers = ("관리종목", "상장폐지", "감사의견", "회생", "파산", "불성실공시")
    if any(marker in text for marker in hard_markers):
        return False
    procedural_capital_action_markers = (
        "무상증자",
        "주식분할",
        "액면분할",
        "권리락",
        "주식병합",
    )
    if any(marker in text for marker in procedural_capital_action_markers):
        return True
    if _is_resolved_spac_merger_halt_item(text, news_cache):
        return True
    if "우회상장" not in text:
        return False
    return _has_resolved_reverse_listing_halt(news_cache)


def _is_resolved_spac_merger_halt_item(text: str, news_cache: dict[str, Any]) -> bool:
    if "합병" not in text:
        return False
    spac_markers = ("spac", "스팩", "기업인수목적")
    if not any(marker in text.lower() for marker in spac_markers):
        return False
    return _has_resolved_spac_merger_halt(news_cache)


def _has_resolved_spac_merger_halt(news_cache: dict[str, Any]) -> bool:
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    for raw_item in raw_items:
        if not isinstance(raw_item, dict) or raw_item.get("company_match") is not True:
            continue
        text = _compact_text(" ".join(str(raw_item.get(key, "")) for key in ("title", "summary")))
        if "거래정지해제" in text and "상장예비심사결과" in text and "승인" in text:
            return True
    return False


def _has_resolved_reverse_listing_halt(news_cache: dict[str, Any]) -> bool:
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    for raw_item in raw_items:
        if not isinstance(raw_item, dict) or raw_item.get("company_match") is not True:
            continue
        text = _compact_text(" ".join(str(raw_item.get(key, "")) for key in ("title", "summary")))
        if "거래정지해제" in text and "우회상장" in text and "미해당" in text:
            return True
    return False


def _is_verified_adverse_external_item(item: dict[str, Any]) -> bool:
    quality = str(item.get("evidence_quality", "")).lower()
    if quality in ADVERSE_EVIDENCE_QUALITY:
        return True
    score = _safe_float(item.get("evidence_score"))
    return score is not None and score >= 0.55


def _item_critical_terms(item: dict[str, Any]) -> list[str]:
    raw_terms = item.get("critical_terms", [])
    if isinstance(raw_terms, list | tuple):
        return [str(term) for term in raw_terms if str(term).strip()]
    text = " ".join(str(item.get(key, "")) for key in ("title", "summary"))
    return cast(list[str], critical_terms_in_text(text))


def _model_threshold(bundle: Stage2InputBundle) -> float:
    for source in (bundle.xgboost_result, bundle.model_view, bundle.rule_result):
        for key in ("threshold", "threshold_tuned", "decision_threshold"):
            value = _safe_float(source.get(key))
            if value is not None and value > 0:
                return float(value)
    return 0.315


def _stage2_review_priority(bundle: Stage2InputBundle) -> str:
    for source in (bundle.model_view, bundle.xgboost_result, bundle.rule_result):
        value = str(source.get("stage2_review_priority") or "").strip().lower()
        if value:
            return value
    return "none"


def _stage2_trigger_reason(bundle: Stage2InputBundle) -> str:
    for source in (bundle.model_view, bundle.xgboost_result, bundle.rule_result):
        value = str(source.get("trigger_reason") or "").strip()
        if value:
            return value
    return ""


def _collect_committee_factors(
    agents: list[AgentOutput],
    *,
    target: Literal["risk", "mitigation"],
) -> list[str]:
    factors: list[str] = []
    for agent in agents:
        for finding in agent.findings:
            text = str(finding)
            value = _committee_factor_value(text, target=target)
            if value and "제한적입니다" not in value:
                factors.append(value)
    return factors[:5]


def _committee_factor_value(text: str, *, target: Literal["risk", "mitigation"]) -> str | None:
    """Classify flattened agent findings into risk or mitigation buckets."""
    if target == "risk" and text.startswith("핵심 위험 요인:"):
        return text.removeprefix("핵심 위험 요인:").strip()
    if target == "mitigation" and text.startswith("완화 요인:"):
        return text.removeprefix("완화 요인:").strip()
    for prefix in (
        "부채·유동성 검증 의견:",
        "EvidenceAudit 검토 결론:",
        "모델-근거 충돌 점검:",
        "외부근거 위험:",
        "외부근거 점검:",
    ):
        if text.startswith(prefix):
            value = text.removeprefix(prefix).strip()
            if target == "risk" and _non_escalating_risk_text(value):
                return None
            classification = _classify_committee_factor(value)
            if classification == target:
                return value
            return None
    return None


def _non_escalating_risk_text(text: str) -> bool:
    """Keep missing or unconfirmed evidence from escalating an eligible company to hold."""
    neutral_markers = (
        "외부근거 미수집",
        "외부 뉴스·공시 근거 수집이 비활성화",
        "확인된 외부 뉴스·공시 항목 없음",
        "외부근거가 제공되지 않아",
        "외부 교차검증은 보류",
        "확정되지 않았",
        "중대한 충돌은 제한적",
        "현재 연결된 뉴스/공시 항목은 없습니다",
        "확인 가능한 외부 근거가 제한적",
        "수집 상태가 `not_requested`",
        "수집 상태가 `disabled`",
        "수집 상태가 `no_results`",
        "모델 판단을 실질적으로 뒤집기 어렵",
        "모델 원판단을 실질적으로 뒤집기 어렵",
        "stage 1 모델 판단을 실질적으로 뒤집기 어렵",
        "모델 라벨은 투자적격으로 보존",
    )
    lowered = text.lower()
    return any(marker in lowered for marker in neutral_markers)


def _classify_committee_factor(
    text: str,
) -> Literal["risk", "mitigation", "neutral"]:
    risk_markers = (
        "추가 경계",
        "추가 점검",
        "보수 검토",
        "보수적인",
        "부적격 판단을 보수적으로 뒷받침",
        "부족",
        "취약",
        "제한",
        "어렵습니다",
        "약합니다",
        "차환 리스크",
        "치명 리스크",
        "위험 근거",
        "veto",
    )
    mitigation_markers = (
        "완충 근거",
        "완화 신호",
        "방어력",
        "양호",
        "확보",
        "여력",
        "과도하지",
        "완화 요인",
    )
    if any(marker in text for marker in risk_markers):
        return "risk"
    if any(marker in text for marker in mitigation_markers):
        return "mitigation"
    return "neutral"


def _evidence_summary_items(
    bundle: Stage2InputBundle,
    agents: list[AgentOutput],
) -> list[dict[str, str]]:
    evidence_agent = next((agent for agent in agents if agent.role == "evidence_audit"), None)
    quant_agent = next((agent for agent in agents if agent.role == "quant_credit"), None)
    items = [
        {
            "source": "model_view",
            "summary": quant_agent.summary if quant_agent else "Stage 1 model_view was reviewed.",
            "reliability": "high",
        },
        {
            "source": "feature_snapshot",
            "summary": evidence_agent.summary
            if evidence_agent
            else "Feature snapshot risk checks were not produced.",
            "reliability": "high",
        },
        {
            "source": "news_cache",
            "summary": f"뉴스/공시 근거 번들 상태는 `{bundle.news_status}`입니다.",
            "reliability": "pending",
        },
    ]
    prior = bundle.prior_rating_reference
    if prior.get("has_prior_rating") is True:
        items.append(
            {
                "source": "prior_rating_reference",
                "summary": (
                    "이전 공개등급 "
                    f"{prior.get('prior_credit_rating')} "
                    f"({prior.get('prior_rating_agency')}, {prior.get('prior_rating_date')})을 "
                    f"{prior.get('as_of_date')} 기준으로 확인했습니다. "
                    f"경계구간: {prior.get('prior_rating_boundary_group')}."
                ),
                "reliability": "high",
            }
        )
    else:
        items.append(
            {
                "source": "prior_rating_reference",
                "summary": "기준일 이전 공개 신용등급 reference가 없어 경계등급 입력은 비어 있습니다.",
                "reliability": "context",
            }
        )
    evidence_limitations = _evidence_limitations_from_agents(agents)
    if evidence_limitations:
        items.append(
            {
                "source": "evidence_limitations",
                "summary": " / ".join(evidence_limitations[:3]),
                "reliability": "context",
            }
        )
    raw_items = bundle.news_cache_snapshot.get("items", [])
    if isinstance(raw_items, list):
        for item in raw_items[:3]:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or item.get("title") or "")
            reliability = str(item.get("reliability", "unknown"))
            evidence_quality = str(item.get("evidence_quality", "unknown"))
            if evidence_quality != "unknown":
                summary = f"검증품질 {evidence_quality}: {summary}"
            if item.get("company_match") is False:
                summary = f"직접 관련성 낮음: {summary}"
                reliability = "low_relevance"
            elif item.get("company_match") is not True:
                summary = f"직접 관련성 미확인: {summary}"
            critical_terms = [str(term) for term in item.get("critical_terms", []) or []]
            if critical_terms and item.get("veto_candidate") is not True:
                summary = f"{summary} (미확인 키워드 히트: {', '.join(critical_terms)})"
            items.append(
                {
                    "source": str(item.get("source", "external")),
                    "summary": summary,
                    "reliability": reliability,
                }
            )
    return items


def _decision_trace_items(
    *,
    bundle: Stage2InputBundle,
    committee_label: CommitteeLabel,
    decision_type: CommitteeDecisionType,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    boundary_review: BoundaryReviewAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
    risk_hold_reason_tags: list[RiskHoldReasonTag],
    risk_hold_reason_summary: str,
) -> list[DecisionTraceItem]:
    """Build an auditable deterministic gate trace for Stage 2 decisions."""
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    review_priority = _stage2_review_priority(bundle)
    trigger_reason = _stage2_trigger_reason(bundle)
    trace = [
        DecisionTraceItem(
            gate="stage1_model_view",
            label="1차 모델 원판단",
            triggered=bundle.prediction_label == "부적격",
            severity="risk" if bundle.prediction_label == "부적격" else "info",
            summary=(
                f"1차 모델은 {bundle.prediction_label}으로 판단했습니다 "
                f"(투기등급 확률 {probability:.1%}, 기준선 {threshold:.1%})."
            ),
        ),
        DecisionTraceItem(
            gate="veto_rule",
            label="강제 경고 게이트",
            triggered=veto_triggered,
            severity="risk" if veto_triggered else "info",
            summary=(
                "다중 출처 또는 고신뢰 치명 근거가 확인되어 강제 경고 조건을 충족했습니다."
                if veto_triggered
                else "강제 경고 조건은 충족하지 않았습니다."
            ),
        ),
        DecisionTraceItem(
            gate="hidden_tail_risk",
            label="숨은 꼬리위험 점검",
            triggered=hidden_tail_risk.triggered,
            severity="risk"
            if hidden_tail_risk.triggered and hidden_tail_risk.risk_signal
            else ("watch" if hidden_tail_risk.triggered else "info"),
            summary=hidden_tail_risk.reason
            or "직접 관련 외부 위험 근거로 모델 투자적격 판단을 뒤집을 신호는 제한적입니다.",
        ),
        DecisionTraceItem(
            gate="secondary_review_trigger",
            label="2차 보조 레이더",
            triggered=secondary_review_risk.triggered,
            severity="risk"
            if secondary_review_risk.risk_signal
            else ("watch" if secondary_review_risk.triggered else "info"),
            summary=secondary_review_risk.reason
            or (
                "보조 검토 우선순위는 "
                f"{review_priority or 'none'}입니다."
                + (f" 사유: {trigger_reason}" if trigger_reason else "")
            ),
        ),
        DecisionTraceItem(
            gate="boundary_rating_review",
            label="경계등급 점검",
            triggered=boundary_review.triggered,
            severity="watch" if boundary_review.triggered else "info",
            summary=boundary_review.reason
            or "이전 공개등급 또는 확률 기준의 BBB-/BB+ 경계 보류 조건은 제한적입니다.",
        ),
        DecisionTraceItem(
            gate="overwarning_mitigation",
            label="과민경고 완화 점검",
            triggered=overwarning_mitigation.triggered,
            severity="mitigation" if overwarning_mitigation.triggered else "info",
            summary=overwarning_mitigation.reason
            or "모델 과민경고를 완화할 만큼의 방어력/비위험 근거는 제한적입니다.",
        ),
        DecisionTraceItem(
            gate="reject_confirmation",
            label="부적격 확정 게이트",
            triggered=reject_confirmation.confirmed,
            severity="risk"
            if reject_confirmation.confirmed
            else ("watch" if reject_confirmation.triggered else "info"),
            summary=reject_confirmation.reason
            or "부적격 확정을 위한 복수의 강한 근거 조건은 충족하지 않았습니다.",
        ),
        DecisionTraceItem(
            gate="risk_hold_reason_tagging",
            label="위험 보류 이유 태그",
            triggered=bool(risk_hold_reason_tags),
            severity="risk" if risk_hold_reason_tags else "info",
            summary=risk_hold_reason_summary
            or "위험 보류가 아니므로 별도 위험 보류 이유 태그는 남기지 않았습니다.",
        ),
        DecisionTraceItem(
            gate="final_committee_decision",
            label="최종 위원회 판단",
            triggered=True,
            severity=_decision_trace_final_severity(decision_type),
            summary=(
                f"최종 위원회 판단은 {committee_label}이며, "
                f"세부 유형은 {_committee_decision_type_label(decision_type)}입니다."
            ),
        ),
    ]
    return trace


def _decision_trace_final_severity(
    decision_type: CommitteeDecisionType,
) -> Literal["info", "watch", "risk", "mitigation"]:
    if decision_type == "reject":
        return "risk"
    if decision_type == "risk_hold":
        return "risk"
    if decision_type == "mitigation_hold":
        return "mitigation"
    if decision_type in {"boundary_hold", "review_hold"}:
        return "watch"
    return "info"


def _evidence_limitations_from_agents(agents: list[AgentOutput]) -> list[str]:
    evidence_agent = next((agent for agent in agents if agent.role == "evidence_audit"), None)
    if evidence_agent is None:
        return []
    limitations: list[str] = []
    for finding in evidence_agent.findings:
        text = str(finding)
        if text.startswith("근거 한계:"):
            value = text.removeprefix("근거 한계:").strip()
            if value:
                limitations.append(value)
    return limitations


def _conflict_resolution(
    *,
    prediction_label: str,
    committee_label: str,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    boundary_review: BoundaryReviewAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
) -> str:
    if veto_triggered:
        return (
            "치명적 외부 위험 신호가 확인되어 모델 원판단과 무관하게 "
            "위원회 의견을 부적격으로 보수 조정했습니다."
        )
    if hidden_tail_risk.triggered:
        if not hidden_tail_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}이지만, 직접 관련 외부 규모성 공시가 "
                "추가 확인 대상으로 확인되어 위원회 의견은 보류로 정리했습니다. 다만 "
                "치명 문맥이나 실질 부실 전이 근거는 제한적이어서 위험 보류가 아닌 "
                "확인필요 보류로 구분했습니다."
            )
        return (
            f"모델 원판단은 {prediction_label}이지만, 직접 관련 외부 위험 근거가 "
            "모델이 놓칠 수 있는 숨은 꼬리위험을 보완해 위원회 의견은 보류로 정리했습니다."
        )
    if overwarning_mitigation.triggered:
        return (
            f"모델 원판단은 {prediction_label}이지만, 과민 경고 가능성과 완화 근거를 함께 보아 "
            "위원회 의견은 부적격이 아닌 보류로 정리했습니다."
        )
    if reject_confirmation.triggered:
        hold_type = "위험 보류" if reject_confirmation.review_risk_signal else "확인필요 보류"
        return (
            f"모델 원판단은 {prediction_label}이지만, 부적격 확정 게이트를 통과하지 못해 "
            f"위원회 의견은 부적격 확정이 아닌 {hold_type}로 정리했습니다. "
            f"{reject_confirmation.reason}"
        )
    if boundary_review.triggered:
        return (
            f"모델 원판단은 {prediction_label}이지만, 경계등급 확인 신호가 있어 "
            "위원회 의견은 경계등급 보류로 정리했습니다. 이는 위험 확정이나 과민경고 완화가 "
            f"아니라, 추가 근거 확인이 필요한 상태입니다. {boundary_review.reason}"
        )
    if secondary_review_risk.triggered:
        if not secondary_review_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}이지만, 45개 보조 변수셋이 추가 확인 대상으로 "
                "올린 케이스라 위원회 의견은 보류로 정리했습니다. 다만 확률 수준은 위험신호 "
                "표시 기준선보다 낮아 확인필요 보류로 구분합니다."
            )
        return (
            f"모델 원판단은 {prediction_label}이지만, 45개 보조 변수셋의 추가 검토 신호가 "
            "FN 가능성을 보완해 위원회 의견은 보류로 정리했습니다."
        )
    model_label = "적격" if prediction_label == "투자적격" else "부적격"
    if committee_label == "보류":
        return (
            f"모델 원판단은 {prediction_label}이지만, 정량 해석과 외부/유동성 검증 사이에 "
            "추가 점검 여지가 있어 위원회 의견은 보류로 정리했습니다."
        )
    if committee_label != model_label:
        return (
            f"모델 원판단({prediction_label})과 위원회 라벨({committee_label})이 달라, "
            "외부 검증 근거와 완화 요인을 함께 고려해 최종 의견을 조정했습니다."
        )
    return (
        f"모델 원판단({prediction_label})과 위원회 라벨({committee_label})이 대체로 일치하며, "
        "Stage 2는 판단을 덮어쓰기보다 근거와 설명을 보완했습니다."
    )


def _final_review_memo(
    *,
    prediction_label: str,
    committee_label: str,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    boundary_review: BoundaryReviewAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
    risk_factors: list[str],
    mitigating_factors: list[str],
) -> str:
    if veto_triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존하지만, 강제 경고 조건을 충족하는 "
            "외부 또는 정책 위험 신호가 있어 위원회 의견을 부적격으로 정리했습니다."
        )
    if hidden_tail_risk.triggered:
        if not hidden_tail_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 직접 관련 "
                "규모성 공시가 확인되어 최종 적격으로 바로 확정하지 않고 보류로 "
                "정리했습니다. 치명 문맥이나 현금흐름 악화가 함께 확인된 실질 부실 "
                f"근거는 제한적이므로 세부 유형은 확인필요 보류입니다. {hidden_tail_risk.reason}"
            )
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 직접 관련 외부 위험 "
            f"근거가 확인되어 재무제표 기반 모델이 놓칠 수 있는 FN 가능성을 보완했습니다. "
            f"위원회는 최종 의견을 {committee_label}로 정리했습니다. {hidden_tail_risk.reason}"
        )
    if overwarning_mitigation.triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 강한 외부 위험 근거가 "
            f"확인되지 않았고 완화 근거가 있어 위원회는 최종 의견을 {committee_label}로 "
            f"낮춰 정리했습니다. {overwarning_mitigation.reason}"
        )
    if reject_confirmation.triggered:
        hold_type = "위험 보류" if reject_confirmation.review_risk_signal else "확인필요 보류"
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 위원회는 부적격을 "
            "확정하기 위한 복수의 강한 근거가 충분하지 않다고 보고 최종 의견을 "
            f"{committee_label}로 정리했으며, 세부 유형은 {hold_type}입니다. "
            f"{reject_confirmation.reason}"
        )
    if boundary_review.triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 투자적격/투기등급 "
            "경계에서 판단 불확실성이 큰 케이스라 위원회는 최종 의견을 보류로 "
            f"정리하고, 세부 유형은 경계등급 보류로 표시했습니다. {boundary_review.reason}"
        )
    if secondary_review_risk.triggered:
        if not secondary_review_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 45개 보조 변수셋이 "
                "추가 확인 대상으로 올린 케이스라 최종 적격으로 바로 확정하지 않고 보류로 "
                "정리했습니다. 확률 수준은 위험신호 표시 기준선보다 낮아 확인필요 보류로 "
                f"구분합니다. {secondary_review_risk.reason}"
            )
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 45개 보조 변수셋이 "
            f"추가 검토 대상으로 올린 케이스라 FN 가능성을 보수적으로 보완했습니다. "
            f"위원회는 최종 의견을 {committee_label}로 정리했습니다. "
            f"{secondary_review_risk.reason}"
        )
    risk_note = (
        f"주요 위험은 {risk_factors[0]}"
        if risk_factors
        else "추가로 확정된 핵심 위험 요인은 제한적입니다"
    )
    mitigation_note = (
        f"완화 요인은 {mitigating_factors[0]}"
        if mitigating_factors
        else "명시적 완화 요인은 제한적입니다"
    )
    return (
        f"모델 원판단은 {prediction_label}으로 보존합니다. 위원회는 정량 해석, "
        f"부채/유동성 교차 검증, 외부 근거 상태를 함께 검토해 최종 의견을 "
        f"{committee_label}로 정리했습니다. {risk_note}. {mitigation_note}."
    )


def _chair_report_memo_seed(agents: list[AgentOutput], *, committee_label: CommitteeLabel) -> str:
    chair = next((agent for agent in agents if agent.role == "chair_report"), None)
    if chair is None:
        return ""
    candidates = [*chair.findings[::-1], chair.summary]
    for candidate in candidates:
        cleaned = cast(str, _clean_korean_review_text(str(candidate or "")))
        if _is_informative_chair_report_memo(
            cleaned,
            committee_label=committee_label,
        ):
            return cleaned
    return ""


def _is_informative_chair_report_memo(text: str, *, committee_label: CommitteeLabel) -> bool:
    if len(text.strip()) < 40:
        return False
    generic_markers = (
        "ChairReportAgent는 정량 해석과 검증 근거를 사람이 읽는 심사 메모로 연결합니다",
        "정량 판단은 model_view로 보존",
        "committee_view에서는 해석과 보완 의견만 추가합니다",
        "최종 보고서는 적격/보류/부적격 3단 위원회 의견",
        "Agno ",
        "chair label=",
        "ChairReportAgent는 모델 원판단",
        "현재 서비스 recommendation은",
    )
    if any(marker in text for marker in generic_markers):
        return False
    return not _chair_report_memo_conflicts_with_final_label(
        text,
        committee_label=committee_label,
    )


def _chair_report_memo_conflicts_with_final_label(
    text: str, *, committee_label: CommitteeLabel
) -> bool:
    """Avoid appending agent prose that contradicts the resolved committee label."""
    if committee_label == "적격":
        return False
    investment_keep_markers = (
        "투자적격 판단을 유지",
        "투자적격 라벨을 유지",
        "투자적격 등급을 유지",
        "투자적격 분류를 유지",
        "투자적격 유지",
        "모델 라벨을 유지",
        "모델 라벨 유지",
        "모델 라벨을 존중",
        "모델 라벨 존중",
        "최종 라벨은 투자적격",
        "최종 라벨을 투자적격",
        "기존 투자적격",
    )
    return any(marker in text for marker in investment_keep_markers)


def _with_chair_report_memo(base_memo: str, chair_memo_seed: str) -> str:
    base = cast(str, _clean_korean_review_text(base_memo))
    seed = cast(str, _clean_korean_review_text(chair_memo_seed))
    if not seed:
        return base
    normalized_base = " ".join(base.split())
    normalized_seed = " ".join(seed.split())
    if normalized_seed in normalized_base:
        return base
    if normalized_base in normalized_seed:
        return seed
    return f"{base} 위원회 보강 의견: {seed}"


__all__ = ["build_committee_view", "build_committee_view_model"]
