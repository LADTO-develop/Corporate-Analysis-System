"""Deterministic Stage 2 committee decision policy."""

from __future__ import annotations

from typing import Any, cast

from cas.agents.committee_assessments import (
    BoundaryReviewAssessment,
    HiddenTailRiskAssessment,
    OverwarningMitigationAssessment,
    RejectConfirmationAssessment,
    SecondaryReviewRiskAssessment,
)
from cas.agents.committee_external_evidence import (
    adverse_external_items,
    is_actionable_verified_adverse_external_item,
    item_critical_terms,
    no_direct_external_items,
    noncritical_external_evidence_assessment,
    overwarning_blocking_external_items,
)
from cas.agents.committee_financial_guardrails import (
    cash_rich_loss_stage_overwarning_buffer_reason,
    financial_resilience_overwarning_assessment,
    has_blocking_flags,
    has_extreme_financial_distress_signal,
    has_severe_financial_watch_signal,
    mitigation_hold_model_risk_label_reason,
    mitigation_hold_residual_risk_reason,
    model_only_overwarning_buffer_reason,
    prior_boundary_overwarning_buffer_reason,
    prior_hard_distress_risk_label_reason,
    prior_rating_boundary_requires_hold,
    prior_rating_has_hard_distress_context,
    prior_rating_is_exact_boundary,
    prior_rating_is_speculative,
    review_hold_model_risk_label_reason,
    risk_hold_has_financial_stress,
    secondary_overhold_guardrail_reason,
    secondary_review_requires_hold,
    secondary_review_risk_assessment,
)
from cas.agents.committee_schema import (
    CommitteeDecisionType,
    CommitteeLabel,
    RiskHoldReasonTag,
)
from cas.agents.committee_utils import safe_float, safe_int
from cas.agents.signals.evidence_treatment_signals import evaluate_evidence_treatment
from cas.agents.signals.materiality_signals import (
    confirmed_external_veto_item,
    confirmed_hard_distress_item,
    financing_evidence_items,
    hidden_tail_evidence_requires_risk_signal,
    is_uncorroborated_material_financing_or_guarantee_item,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.state import Recommendation
from cas.veto_rules import (
    VetoRules,
    external_evidence_veto_triggered,
    flag_contains_veto_marker,
)


def committee_label_from_recommendation(recommendation: Recommendation) -> CommitteeLabel:
    """Map the committee recommendation enum to the dashboard label."""
    if recommendation == "priority":
        return "적격"
    if recommendation in {"watch", "review"}:
        return "보류"
    return "부적격"


def committee_decision_type(
    *,
    committee_label: CommitteeLabel,
    prediction_label: str,
    hidden_tail_risk: HiddenTailRiskAssessment,
    boundary_review: BoundaryReviewAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
    mitigation_residual_risk_reason: str = "",
    prior_hard_distress_risk_label_reason: str = "",
    mitigation_model_risk_label_reason: str = "",
    review_model_risk_label_reason: str = "",
) -> CommitteeDecisionType:
    """Return the dashboard-facing subtype for a final committee label."""
    if committee_label == "적격":
        return "eligible"
    if committee_label == "부적격":
        return "reject"
    if hidden_tail_risk.triggered:
        return "risk_hold" if hidden_tail_risk.risk_signal else "review_hold"
    if prior_hard_distress_risk_label_reason:
        return "risk_hold"
    if (
        overwarning_mitigation.triggered
        and prediction_label == "부적격"
        and (mitigation_residual_risk_reason or mitigation_model_risk_label_reason)
    ):
        return "risk_hold"
    if overwarning_mitigation.triggered and prediction_label == "부적격":
        return "mitigation_hold"
    if reject_confirmation.triggered:
        return (
            "risk_hold"
            if reject_confirmation.review_risk_signal or review_model_risk_label_reason
            else "review_hold"
        )
    if boundary_review.triggered:
        return "boundary_hold"
    if secondary_review_risk.triggered:
        return "risk_hold" if secondary_review_risk.risk_signal else "review_hold"
    if overwarning_mitigation.triggered:
        return "mitigation_hold"
    if prediction_label == "부적격":
        return "risk_hold"
    return "review_hold"


def committee_decision_type_label(decision_type: CommitteeDecisionType) -> str:
    """Return the Korean display label for a decision subtype."""
    labels: dict[CommitteeDecisionType, str] = {
        "eligible": "적격",
        "risk_hold": "위험 보류",
        "boundary_hold": "경계등급 보류",
        "mitigation_hold": "과민경고 완화 보류",
        "review_hold": "확인필요 보류",
        "reject": "부적격",
    }
    return labels[decision_type]


def committee_risk_signal(decision_type: CommitteeDecisionType) -> bool:
    """Return whether the subtype should be treated as a risk signal."""
    return decision_type in {"risk_hold", "reject"}


_RISK_HOLD_REASON_LABELS: dict[RiskHoldReasonTag, str] = {
    "combined_watch_hold": "재무+외부 복합 관찰",
    "prior_hard_distress_hold": "기준일 이전 severe 등급",
    "financial_stress_hold": "재무 스트레스",
    "external_materiality_hold": "외부 중요도 근거",
    "secondary_radar_hold": "2차 보조 레이더",
    "model_reject_confirmation_hold": "부적격 확정 전 보류",
    "model_risk_hold": "모델 위험 보류",
}


def risk_hold_reason_tags(
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

    financial_stress = risk_hold_has_financial_stress(
        bundle,
        secondary_review_risk=secondary_review_risk,
        reject_confirmation=reject_confirmation,
    )
    external_materiality = _risk_hold_has_external_materiality(
        bundle,
        hidden_tail_risk=hidden_tail_risk,
    )
    prior_hard_distress = prior_rating_has_hard_distress_context(bundle.prior_rating_reference)
    tags: list[RiskHoldReasonTag] = []
    if financial_stress and external_materiality:
        tags.append("combined_watch_hold")
    if prior_hard_distress:
        tags.append("prior_hard_distress_hold")
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


def _risk_hold_has_external_materiality(
    bundle: Stage2InputBundle,
    *,
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> bool:
    if hidden_tail_risk.triggered:
        return True
    treatment = evaluate_evidence_treatment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    if treatment.recommended_evidence_treatment in {
        "substantive_review",
        "critical_veto_review",
    }:
        return True
    summary = treatment.materiality_summary
    return bool(safe_int(summary.get("high_risk_financing_evidence_count")) or 0) or bool(
        summary.get("material_financing_blocks_tn_hold")
    )


def risk_hold_reason_labels(tags: list[RiskHoldReasonTag]) -> list[str]:
    """Return Korean display labels for risk-hold reason tags."""
    return [_RISK_HOLD_REASON_LABELS[tag] for tag in tags]


def risk_hold_reason_summary(
    *,
    tags: list[RiskHoldReasonTag],
    labels: list[str],
) -> str:
    """Return a concise Korean explanation for risk-hold reason tags."""
    if not tags:
        return ""
    label_text = ", ".join(labels)
    if "combined_watch_hold" in tags:
        return (
            "위험 보류 이유 태그는 "
            f"{label_text}입니다. 재무 스트레스와 외부 중요도 근거가 함께 남아 있어, "
            "정상기업 과잉 보류 guardrail을 바로 적용하지 않고 위험 보류로 유지했습니다."
        )
    if "prior_hard_distress_hold" in tags:
        return (
            "위험 보류 이유 태그는 "
            f"{label_text}입니다. 기준일 이전 공개등급이 CCC/C/D 등 심각한 신용위험 "
            "영역에 있어, 현재 재무와 외부근거가 완전히 해소를 확인하기 전까지 "
            "위험 보류로 유지했습니다."
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


def veto_triggered(bundle: Stage2InputBundle, *, veto_rules: VetoRules) -> bool:
    """Return whether deterministic veto rules force a reject-style label."""
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


def veto_triggered_label(veto_rules: VetoRules) -> CommitteeLabel:
    """Return the configured committee label for a veto hit."""
    label = veto_rules.triggered_label
    if label in {"적격", "보류", "부적격"}:
        return cast(CommitteeLabel, label)
    return "부적격"


def committee_label_with_evidence_escalation(
    committee_label: CommitteeLabel,
    *,
    bundle: Stage2InputBundle,
    evidence_agent_requires_hold: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
) -> CommitteeLabel:
    """Escalate non-veto EvidenceAudit red flags without overwriting model_view."""
    if committee_label != "적격":
        return committee_label
    if hidden_tail_risk.triggered:
        return "보류"
    if _external_evidence_unavailable(bundle.news_status):
        return committee_label
    if evidence_agent_requires_hold:
        return "보류"
    return committee_label


def committee_label_with_model_alignment(
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
    if secondary_overhold_guardrail_reason(bundle):
        return "적격"
    if secondary_review_requires_hold(bundle):
        return committee_label
    if has_blocking_flags(bundle):
        return committee_label
    if not bundle.source_feature_row:
        return committee_label

    probability = bundle.probability_speculative
    threshold = model_threshold(bundle)
    near_threshold = probability >= max(0.28, threshold - 0.05)
    if near_threshold:
        return committee_label
    if has_severe_financial_watch_signal(bundle.source_feature_row):
        return committee_label
    return "적격"


def committee_label_with_investment_evidence_alignment(
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
    if secondary_overhold_guardrail_reason(bundle):
        return "적격"
    if secondary_review_requires_hold(bundle):
        return committee_label
    if has_blocking_flags(bundle):
        return committee_label
    if has_severe_financial_watch_signal(bundle.source_feature_row):
        return committee_label
    probability = bundle.probability_speculative
    threshold = model_threshold(bundle)
    if probability >= threshold:
        return committee_label
    noncritical_evidence = noncritical_external_evidence_assessment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    if not noncritical_evidence.triggered:
        return committee_label
    return "적격"


def hidden_tail_risk_assessment(bundle: Stage2InputBundle) -> HiddenTailRiskAssessment:
    """Flag likely FN cases where external adverse evidence challenges an eligible model call."""
    if bundle.prediction_label != "투자적격":
        return HiddenTailRiskAssessment(False, "", 0, 0)
    adverse_items = adverse_external_items(bundle.news_cache_snapshot)
    if not adverse_items:
        return HiddenTailRiskAssessment(False, "", 0, 0)

    verified_items = [
        item
        for item in adverse_items
        if is_actionable_verified_adverse_external_item(item, bundle.news_cache_snapshot)
        and not _is_uncorroborated_material_financing_or_guarantee_item(
            item,
            source_feature_row=bundle.source_feature_row,
        )
    ]
    if not verified_items:
        return HiddenTailRiskAssessment(False, "", len(adverse_items), 0)

    probability = bundle.probability_speculative
    threshold = model_threshold(bundle)
    source_names = sorted({str(item.get("source", "external")) for item in verified_items})
    terms = sorted({term for item in adverse_items for term in item_critical_terms(item)})
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


def boundary_review_assessment(
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

    prior_boundary_reason = prior_rating_boundary_hold_reason(
        bundle,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
    )
    if prior_boundary_reason:
        return BoundaryReviewAssessment(True, prior_boundary_reason)

    probability = bundle.probability_speculative
    threshold = model_threshold(bundle)
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


def prior_rating_boundary_hold_reason(
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
    if not prior_rating_is_exact_boundary(prior):
        return ""
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""

    probability = bundle.probability_speculative
    threshold = model_threshold(bundle)
    if committee_label == "부적격" and probability > max(threshold + 0.20, 0.55):
        return ""

    rating = str(prior.get("prior_credit_rating") or "").strip()
    rating_date = str(prior.get("prior_rating_date") or "").strip()
    agency = str(prior.get("prior_rating_agency") or "").strip()
    age_days = safe_int(prior.get("prior_rating_age_days"))
    age_text = f", 기준일 대비 {age_days}일 전 공개" if age_days is not None else ""
    source_text = f"{agency} " if agency else ""
    return (
        f"경계등급 보류 플래그: {source_text}이전 공개등급이 {rating}"
        f"({rating_date}{age_text})로 BBB-/BB+ 경계권에 있습니다. "
        "이 정보는 평가 대상 시점 이전에 공개된 prior rating reference에서만 가져온 "
        "비누수 입력입니다. 모델 확률과 외부근거를 함께 보더라도 즉시 확정하기보다 "
        "투자적격/투기등급 경계 재확인 대상으로 분리합니다."
    )


def overwarning_mitigation_assessment(
    bundle: Stage2InputBundle,
    *,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    mitigating_factors: list[str],
) -> OverwarningMitigationAssessment:
    """Soften likely over-warning cases to hold, not eligible."""
    if bundle.prediction_label != "부적격" or veto_triggered or hidden_tail_risk.triggered:
        return OverwarningMitigationAssessment(False, "")
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return OverwarningMitigationAssessment(False, "")

    probability = bundle.probability_speculative
    threshold = model_threshold(bundle)
    near_threshold = probability <= threshold + 0.10
    watch_band = str(
        bundle.model_view.get("risk_band")
        or bundle.xgboost_result.get("risk_band")
        or bundle.rule_result.get("risk_band")
        or ""
    ).lower() in {"watch", "관찰"}
    explicit_overwarning = bool(bundle.model_view.get("stage2_overwarning_filter_candidate"))
    financial_resilience = financial_resilience_overwarning_assessment(bundle.source_feature_row)
    cash_rich_loss_stage_buffer_reason = cash_rich_loss_stage_overwarning_buffer_reason(bundle)
    noncritical_evidence = noncritical_external_evidence_assessment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    near_threshold_buffer = near_threshold and (
        noncritical_evidence.triggered
        or no_direct_external_items(bundle.news_cache_snapshot)
        or _external_evidence_unavailable(bundle.news_status)
        or bool(bundle.news_cache_snapshot.get("items"))
    )
    model_only_buffer_reason = model_only_overwarning_buffer_reason(
        bundle,
        mitigating_factors=mitigating_factors,
    )
    prior_boundary_buffer_reason = prior_boundary_overwarning_buffer_reason(
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


def reject_confirmation_assessment(
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
    direct_adverse_items = overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    direct_adverse_evidence = bool(direct_adverse_items)
    hard_external_confirmation = _has_hard_reject_external_confirmation(direct_adverse_items)
    extreme_financial_distress = has_extreme_financial_distress_signal(bundle.source_feature_row)
    severe_financial_watch = very_high_model_warning and has_severe_financial_watch_signal(
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
        if confirmed_external_veto_item(item):
            return True
        if confirmed_hard_distress_item(item):
            return True
    return False


def _unconfirmed_reject_review_risk_assessment(
    bundle: Stage2InputBundle,
    *,
    probability: float,
) -> tuple[bool, str]:
    """Upgrade unconfirmed reject holds only when corroborating watch signals exist."""
    repeated_financing_count = _repeated_financing_evidence_count(bundle.news_cache_snapshot)
    threshold = model_threshold(bundle)
    near_very_high_probability = probability >= max(0.88, threshold + 0.55)
    strong_probability = probability >= max(0.80, threshold + 0.35)
    prior_hard_distress = prior_rating_has_hard_distress_context(bundle.prior_rating_reference)
    prior_speculative = prior_rating_is_speculative(bundle.prior_rating_reference)

    reasons: list[str] = []
    if near_very_high_probability and repeated_financing_count >= 2:
        reasons.append(
            f"투기등급 확률이 {probability:.1%}로 높고, 전환사채·유상증자 등 "
            f"자금조달성 공시가 {repeated_financing_count}건 반복 확인되었습니다."
        )
    if strong_probability and prior_hard_distress:
        prior = bundle.prior_rating_reference
        rating = str(prior.get("prior_credit_rating") or "").strip()
        rating_date = str(prior.get("prior_rating_date") or "").strip()
        reasons.append(
            f"투기등급 확률이 {probability:.1%}이고, 평가 기준일 이전 공개등급이 "
            f"{rating}({rating_date})로 CCC/C/D 등 심각한 신용위험 영역에 있었습니다."
        )
    elif strong_probability and prior_speculative:
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


def _financing_evidence_items(news_cache: dict[str, Any]) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], financing_evidence_items(news_cache))


def _is_uncorroborated_material_financing_or_guarantee_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None,
) -> bool:
    """Treat material financing/guarantee as contextual unless distress corroborates it."""
    return bool(
        is_uncorroborated_material_financing_or_guarantee_item(
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
        hidden_tail_evidence_requires_risk_signal(
            items,
            source_feature_row=source_feature_row,
        )
    )


def model_threshold(bundle: Stage2InputBundle) -> float:
    """Return the model threshold from the available Stage 2 inputs."""
    for source in (bundle.xgboost_result, bundle.model_view, bundle.rule_result):
        for key in ("threshold", "threshold_tuned", "decision_threshold"):
            value = safe_float(source.get(key))
            if value is not None and value > 0:
                return float(value)
    return 0.315


def stage2_review_priority(bundle: Stage2InputBundle) -> str:
    """Return the Stage 2 review priority string."""
    for source in (bundle.model_view, bundle.xgboost_result, bundle.rule_result):
        value = str(source.get("stage2_review_priority") or "").strip().lower()
        if value:
            return value
    return "none"


def stage2_trigger_reason(bundle: Stage2InputBundle) -> str:
    """Return the Stage 2 trigger reason string."""
    for source in (bundle.model_view, bundle.xgboost_result, bundle.rule_result):
        value = str(source.get("trigger_reason") or "").strip()
        if value:
            return value
    return ""


def _external_evidence_unavailable(status: str) -> bool:
    return status.strip().lower() in {
        "disabled",
        "missing_credentials",
        "not_implemented",
        "not_requested",
        "placeholder",
        "no_results",
    }


__all__ = [
    "boundary_review_assessment",
    "committee_decision_type",
    "committee_decision_type_label",
    "committee_label_from_recommendation",
    "committee_label_with_evidence_escalation",
    "committee_label_with_investment_evidence_alignment",
    "committee_label_with_model_alignment",
    "committee_risk_signal",
    "hidden_tail_risk_assessment",
    "mitigation_hold_model_risk_label_reason",
    "mitigation_hold_residual_risk_reason",
    "model_threshold",
    "overwarning_mitigation_assessment",
    "prior_hard_distress_risk_label_reason",
    "prior_rating_boundary_hold_reason",
    "prior_rating_boundary_requires_hold",
    "reject_confirmation_assessment",
    "review_hold_model_risk_label_reason",
    "risk_hold_reason_labels",
    "risk_hold_reason_summary",
    "risk_hold_reason_tags",
    "secondary_overhold_guardrail_reason",
    "secondary_review_risk_assessment",
    "stage2_review_priority",
    "stage2_trigger_reason",
    "veto_triggered",
    "veto_triggered_label",
]
