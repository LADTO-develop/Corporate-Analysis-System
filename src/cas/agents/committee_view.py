"""Build the dashboard-facing Stage 2 committee_view payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from cas.agents.committee_schema import (
    CommitteeDecisionType,
    CommitteeLabel,
    CommitteeViewPayload,
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

_ADVERSE_PROVIDER_RELEVANCE = {"risk"}
_ADVERSE_EVIDENCE_QUALITY = {"medium", "high"}


@dataclass(frozen=True)
class HiddenTailRiskAssessment:
    """Model-aware external-evidence flag for likely false-negative risk."""

    triggered: bool
    reason: str
    adverse_item_count: int
    verified_adverse_item_count: int


@dataclass(frozen=True)
class SecondaryReviewRiskAssessment:
    """Model-aware Stage 2 review flag for near-threshold false-negative risk."""

    triggered: bool
    reason: str
    review_priority: str


@dataclass(frozen=True)
class OverwarningMitigationAssessment:
    """Model-aware mitigation flag for likely false-positive review cases."""

    triggered: bool
    reason: str


@dataclass(frozen=True)
class FinancialResilienceAssessment:
    """Financial-defense screen for high-probability over-warning cases."""

    triggered: bool
    reason: str
    support_count: int
    blocker_count: int


@dataclass(frozen=True)
class NoncriticalEvidenceAssessment:
    """External-evidence screen for severe model warnings without decisive corroboration."""

    triggered: bool
    reason: str
    direct_item_count: int
    blocking_item_count: int


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
        committee_label = "보류"
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
    if prediction_label == "부적격" and committee_label == "적격" and not veto_triggered:
        committee_label = "보류"
    evidence_summary = _evidence_summary_items(bundle, agents)
    conflict_resolution = _conflict_resolution(
        prediction_label=prediction_label,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
    )
    final_review_memo = _final_review_memo(
        prediction_label=prediction_label,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
        risk_factors=risk_factors,
        mitigating_factors=mitigating_factors,
    )
    risk_factors = _clean_text_items(risk_factors)
    mitigating_factors = _clean_text_items(mitigating_factors)
    evidence_summary = _clean_evidence_summary_items(evidence_summary)
    conflict_resolution = _clean_korean_review_text(conflict_resolution)
    final_review_memo = _clean_korean_review_text(final_review_memo)
    decision_type = _committee_decision_type(
        committee_label=committee_label,
        hidden_tail_risk=hidden_tail_risk,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
    )

    return CommitteeViewPayload(
        final_committee_label=committee_label,
        committee_decision_type=decision_type,
        committee_decision_type_label=_committee_decision_type_label(decision_type),
        committee_risk_signal=_committee_risk_signal(decision_type),
        veto_triggered=veto_triggered,
        hidden_tail_risk_flag=hidden_tail_risk.triggered,
        hidden_tail_risk_reason=hidden_tail_risk.reason,
        conflict_resolution=conflict_resolution,
        key_risk_factors=risk_factors or ["현재 scaffold 기준 추가 위험 요인은 제한적입니다."],
        mitigating_factors=mitigating_factors
        or ["현재 scaffold 기준 명시적 완화 요인은 제한적입니다."],
        evidence_summary=evidence_summary,
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
    hidden_tail_risk: HiddenTailRiskAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
) -> CommitteeDecisionType:
    if committee_label == "적격":
        return "eligible"
    if committee_label == "부적격":
        return "reject"
    if hidden_tail_risk.triggered or secondary_review_risk.triggered:
        return "risk_hold"
    if overwarning_mitigation.triggered:
        return "mitigation_hold"
    return "review_hold"


def _committee_decision_type_label(decision_type: CommitteeDecisionType) -> str:
    labels: dict[CommitteeDecisionType, str] = {
        "eligible": "적격",
        "risk_hold": "위험 보류",
        "mitigation_hold": "과민경고 완화 보류",
        "review_hold": "확인필요 보류",
        "reject": "부적격",
    }
    return labels[decision_type]


def _committee_risk_signal(decision_type: CommitteeDecisionType) -> bool:
    return decision_type in {"risk_hold", "reject"}


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
    noncritical_evidence = _noncritical_external_evidence_assessment(bundle.news_cache_snapshot)
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


def _has_secondary_rule_liquidity_watch_signal(bundle: Stage2InputBundle) -> bool:
    """Preserve hold for low-but-near-threshold eligible calls with liquidity rule watch."""
    if not _has_stage2_secondary_trigger(bundle):
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
    if any(marker in reason_text for marker in liquidity_markers):
        return True
    return _metric_below(bundle.source_feature_row, "current_ratio", 1.0) or _metric_below(
        bundle.source_feature_row, "cash_ratio", 0.10
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

    verified_items = [item for item in adverse_items if _is_verified_adverse_external_item(item)]
    if not verified_items:
        return HiddenTailRiskAssessment(False, "", len(adverse_items), 0)

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    source_names = sorted({str(item.get("source", "external")) for item in verified_items})
    terms = sorted({term for item in adverse_items for term in _item_critical_terms(item)})
    terms_text = f" 위험 키워드: {', '.join(terms[:4])}." if terms else ""
    reason = (
        f"숨은 꼬리위험 보완 플래그: 모델은 투자적격(투기등급 확률 {probability:.1%}, "
        f"기준선 {threshold:.1%})으로 봤지만, 기업 직접 관련 외부 위험 근거 "
        f"{len(adverse_items)}건 중 검증 가능 근거 {len(verified_items)}건이 확인되어 "
        f"FN 가능성을 보수적으로 점검해야 합니다. 출처: {', '.join(source_names)}."
        f"{terms_text}"
    )
    return HiddenTailRiskAssessment(True, reason, len(adverse_items), len(verified_items))


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
    if trigger_reason:
        reason_parts.append(trigger_reason)
    if rule_liquidity_watch:
        reason_parts.append(
            "룰 엔진도 유동성 watch 신호를 냈기 때문에 낮은 확률 바닥선만으로 "
            "적격 확정하지 않고 보류를 유지합니다."
        )
    reason_parts.append("따라서 2차 위원회는 이를 최종 적격으로 확정하지 않고 보류로 재점검합니다.")
    return SecondaryReviewRiskAssessment(True, " ".join(reason_parts), review_priority)


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
    if _verified_adverse_external_items(bundle.news_cache_snapshot):
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
    noncritical_evidence = _noncritical_external_evidence_assessment(bundle.news_cache_snapshot)
    model_only_buffer_reason = _model_only_overwarning_buffer_reason(
        bundle,
        mitigating_factors=mitigating_factors,
    )
    if not (
        near_threshold
        or watch_band
        or explicit_overwarning
        or financial_resilience.triggered
        or noncritical_evidence.triggered
        or model_only_buffer_reason
    ):
        return OverwarningMitigationAssessment(False, "")
    if (
        not mitigating_factors
        and not explicit_overwarning
        and not financial_resilience.triggered
        and not noncritical_evidence.triggered
        and not model_only_buffer_reason
    ):
        return OverwarningMitigationAssessment(False, "")

    reason_parts = [
        "과민 경고 완화 검토: 1차 모델은 부적격이지만 강한 외부 위험 근거는 확인되지 않았습니다."
    ]
    if near_threshold:
        reason_parts.append(
            f"위험확률이 기준선 근처입니다({probability:.1%} vs 기준선 {threshold:.1%})."
        )
    if explicit_overwarning:
        reason = str(bundle.model_view.get("overwarning_filter_reason") or "").strip()
        if reason:
            reason_parts.append(reason)
    if financial_resilience.triggered:
        reason_parts.append(financial_resilience.reason)
    if noncritical_evidence.triggered:
        reason_parts.append(noncritical_evidence.reason)
    if model_only_buffer_reason:
        reason_parts.append(model_only_buffer_reason)
    return OverwarningMitigationAssessment(True, " ".join(reason_parts))


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
    if _verified_adverse_external_items(bundle.news_cache_snapshot):
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


def _noncritical_external_evidence_assessment(
    news_cache: dict[str, Any],
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
    blocking_items = [item for item in direct_items if _is_blocking_external_adverse_item(item)]
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
    if str(item.get("provider_relevance", "")).lower() in _ADVERSE_PROVIDER_RELEVANCE:
        return True
    terms = _item_critical_terms(item)
    if not terms:
        return False
    return str(item.get("evidence_quality", "")).lower() in _ADVERSE_EVIDENCE_QUALITY


def _is_blocking_external_adverse_item(item: dict[str, Any]) -> bool:
    """Return whether an evidence item should prevent FP mitigation."""
    if item.get("veto_candidate") is True:
        return True
    source = str(item.get("source", "")).lower()
    severity = str(item.get("disclosure_severity", "")).lower()
    if severity in {"veto", "adverse"}:
        return source == "opendart" or item.get("critical_context_confirmed") is True
    # Keyword hits from aggregated news snippets are noisy. They should block
    # over-warning mitigation only when the collector confirmed the risky context.
    return item.get("critical_context_confirmed") is True


def _is_verified_adverse_external_item(item: dict[str, Any]) -> bool:
    quality = str(item.get("evidence_quality", "")).lower()
    if quality in _ADVERSE_EVIDENCE_QUALITY:
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
                return value
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
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
) -> str:
    if veto_triggered:
        return (
            "치명적 외부 위험 신호가 확인되어 모델 원판단과 무관하게 "
            "위원회 의견을 부적격으로 보수 조정했습니다."
        )
    if hidden_tail_risk.triggered:
        return (
            f"모델 원판단은 {prediction_label}이지만, 직접 관련 외부 위험 근거가 "
            "모델이 놓칠 수 있는 숨은 꼬리위험을 보완해 위원회 의견은 보류로 정리했습니다."
        )
    if secondary_review_risk.triggered:
        return (
            f"모델 원판단은 {prediction_label}이지만, 45개 보조 변수셋의 추가 검토 신호가 "
            "FN 가능성을 보완해 위원회 의견은 보류로 정리했습니다."
        )
    if overwarning_mitigation.triggered:
        return (
            f"모델 원판단은 {prediction_label}이지만, 과민 경고 가능성과 완화 근거를 함께 보아 "
            "위원회 의견은 부적격이 아닌 보류로 정리했습니다."
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
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    risk_factors: list[str],
    mitigating_factors: list[str],
) -> str:
    if veto_triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존하지만, 강제 경고 조건을 충족하는 "
            "외부 또는 정책 위험 신호가 있어 위원회 의견을 부적격으로 정리했습니다."
        )
    if hidden_tail_risk.triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 직접 관련 외부 위험 "
            f"근거가 확인되어 재무제표 기반 모델이 놓칠 수 있는 FN 가능성을 보완했습니다. "
            f"위원회는 최종 의견을 {committee_label}로 정리했습니다. {hidden_tail_risk.reason}"
        )
    if secondary_review_risk.triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 45개 보조 변수셋이 "
            f"추가 검토 대상으로 올린 케이스라 FN 가능성을 보수적으로 보완했습니다. "
            f"위원회는 최종 의견을 {committee_label}로 정리했습니다. "
            f"{secondary_review_risk.reason}"
        )
    if overwarning_mitigation.triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 강한 외부 위험 근거가 "
            f"확인되지 않았고 완화 근거가 있어 위원회는 최종 의견을 {committee_label}로 "
            f"낮춰 정리했습니다. {overwarning_mitigation.reason}"
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


def _safe_float(value: object) -> float | None:
    try:
        if value is None or not isinstance(value, int | float | str):
            return None
        numeric = float(value)
        if numeric != numeric:
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _metric_at_least(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value >= threshold


def _metric_at_most(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value <= threshold


def _metric_above(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value > threshold


def _metric_below(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value < threshold


def _flag_is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric >= 0.5
    return str(value).strip().lower() in {"true", "yes", "y", "on"}


def _flag_is_false(value: object) -> bool:
    if isinstance(value, bool):
        return not value
    numeric = _safe_float(value)
    if numeric is not None:
        return numeric < 0.5
    return str(value).strip().lower() in {"false", "no", "n", "off"}


def _clean_text_items(items: list[str]) -> list[str]:
    return [_clean_korean_review_text(item) for item in items]


def _clean_evidence_summary_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            **item,
            "summary": _clean_korean_review_text(str(item.get("summary", ""))),
        }
        for item in items
    ]


def _clean_korean_review_text(text: str) -> str:
    """Clean committee prose for Korean report output."""
    cleaned = str(text).strip()
    replacements = {
        "적격로": "적격으로",
        "부적격로": "부적격으로",
        "투자적격 등급을 확정합니다": "투자적격 검토 의견을 제시합니다",
        "부적격 등급을 확정합니다": "부적격 검토 의견을 제시합니다",
        "신용등급을 확정합니다": "신용위험 검토 의견을 제시합니다",
        "등급을 확정합니다": "검토 의견을 제시합니다",
        "최종 승인합니다": "검토 의견으로 정리합니다",
        "최종 승인": "검토 의견",
        "확정합니다": "검토 의견을 제시합니다",
        "승인합니다": "의견을 제시합니다",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    return cleaned


__all__ = ["build_committee_view", "build_committee_view_model"]
