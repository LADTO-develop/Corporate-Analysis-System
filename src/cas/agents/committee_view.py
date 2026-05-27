"""Build the dashboard-facing Stage 2 committee_view payload."""

from __future__ import annotations

from typing import Any, Literal, cast

from cas.agents.committee_assessments import (
    BoundaryReviewAssessment,
    HiddenTailRiskAssessment,
    OverwarningMitigationAssessment,
    RejectConfirmationAssessment,
    SecondaryReviewRiskAssessment,
)
from cas.agents.committee_decision_policy import (
    boundary_review_assessment as _boundary_review_assessment,
)
from cas.agents.committee_decision_policy import (
    committee_decision_type as _committee_decision_type,
)
from cas.agents.committee_decision_policy import (
    committee_decision_type_label as _committee_decision_type_label,
)
from cas.agents.committee_decision_policy import (
    committee_label_from_recommendation as _committee_label_from_recommendation,
)
from cas.agents.committee_decision_policy import (
    committee_label_with_evidence_escalation as _committee_label_with_evidence_escalation,
)
from cas.agents.committee_decision_policy import (
    committee_label_with_investment_evidence_alignment as _committee_label_with_investment_evidence_alignment,
)
from cas.agents.committee_decision_policy import (
    committee_label_with_model_alignment as _committee_label_with_model_alignment,
)
from cas.agents.committee_decision_policy import (
    committee_risk_signal as _committee_risk_signal,
)
from cas.agents.committee_decision_policy import (
    hidden_tail_risk_assessment as _hidden_tail_risk_assessment,
)
from cas.agents.committee_decision_policy import (
    mitigation_hold_residual_risk_reason as _mitigation_hold_residual_risk_reason,
)
from cas.agents.committee_decision_policy import (
    model_threshold as _model_threshold,
)
from cas.agents.committee_decision_policy import (
    overwarning_mitigation_assessment as _overwarning_mitigation_assessment,
)
from cas.agents.committee_decision_policy import (
    prior_rating_boundary_hold_reason as _prior_rating_boundary_hold_reason,
)
from cas.agents.committee_decision_policy import (
    prior_rating_boundary_requires_hold as _prior_rating_boundary_requires_hold,
)
from cas.agents.committee_decision_policy import (
    reject_confirmation_assessment as _reject_confirmation_assessment,
)
from cas.agents.committee_decision_policy import (
    risk_hold_reason_labels as _risk_hold_reason_labels,
)
from cas.agents.committee_decision_policy import (
    risk_hold_reason_summary as _risk_hold_reason_summary,
)
from cas.agents.committee_decision_policy import (
    risk_hold_reason_tags as _risk_hold_reason_tags,
)
from cas.agents.committee_decision_policy import (
    secondary_overhold_guardrail_reason as _secondary_overhold_guardrail_reason,
)
from cas.agents.committee_decision_policy import (
    secondary_review_risk_assessment as _secondary_review_risk_assessment,
)
from cas.agents.committee_decision_policy import (
    stage2_review_priority as _stage2_review_priority,
)
from cas.agents.committee_decision_policy import (
    stage2_trigger_reason as _stage2_trigger_reason,
)
from cas.agents.committee_decision_policy import (
    veto_triggered as _veto_triggered,
)
from cas.agents.committee_decision_policy import (
    veto_triggered_label as _veto_triggered_label,
)
from cas.agents.committee_financial_guardrails import (
    prior_rating_has_hard_distress_context as _prior_rating_has_hard_distress_context,
)
from cas.agents.committee_memo import (
    chair_report_memo_seed as _chair_report_memo_seed,
)
from cas.agents.committee_memo import (
    conflict_resolution as _conflict_resolution,
)
from cas.agents.committee_memo import (
    evidence_limitations_from_agents as _evidence_limitations_from_agents,
)
from cas.agents.committee_memo import (
    final_review_memo as _final_review_memo,
)
from cas.agents.committee_memo import (
    with_chair_report_memo as _with_chair_report_memo,
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
from cas.agents.signals.evidence_treatment_signals import (
    evaluate_evidence_treatment as _evaluate_evidence_treatment,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.state import AgentOutput, Recommendation
from cas.veto_rules import load_veto_rules


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
            evidence_agent_requires_hold=_evidence_agents_require_hold(agents),
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
    mitigation_residual_risk_reason = (
        _mitigation_hold_residual_risk_reason(bundle)
        if overwarning_mitigation.triggered
        and not veto_triggered
        and not hidden_tail_risk.triggered
        else ""
    )
    if mitigation_residual_risk_reason:
        risk_factors = [mitigation_residual_risk_reason, *risk_factors]
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
    evidence_treatment = _evaluate_evidence_treatment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    final_review_memo = _with_chair_report_memo(
        final_review_memo,
        _chair_report_memo_seed(
            agents,
            committee_label=committee_label,
            evidence_treatment=evidence_treatment,
        ),
    )
    decision_type = _committee_decision_type(
        committee_label=committee_label,
        prediction_label=prediction_label,
        hidden_tail_risk=hidden_tail_risk,
        boundary_review=boundary_review,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
        reject_confirmation=reject_confirmation,
        mitigation_residual_risk_reason=mitigation_residual_risk_reason,
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
    if mitigation_residual_risk_reason:
        conflict_resolution = f"{conflict_resolution} 다만 {mitigation_residual_risk_reason}"

    risk_factors = _clean_text_items(risk_factors)
    mitigating_factors = _clean_text_items(mitigating_factors)
    evidence_summary = _clean_evidence_summary_items(evidence_summary)
    conflict_resolution = _clean_korean_review_text(conflict_resolution)
    final_review_memo = _clean_korean_review_text(final_review_memo)
    action_plan = _committee_action_plan(
        bundle=bundle,
        committee_label=committee_label,
        decision_type=decision_type,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        boundary_review=boundary_review,
        secondary_review_risk=secondary_review_risk,
        overwarning_mitigation=overwarning_mitigation,
        reject_confirmation=reject_confirmation,
    )

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
            mitigation_residual_risk_reason=mitigation_residual_risk_reason,
            risk_hold_reason_tags=risk_hold_reason_tags,
            risk_hold_reason_summary=risk_hold_reason_summary,
        ),
        manual_review_tasks=action_plan["manual_review_tasks"],
        missing_evidence=action_plan["missing_evidence"],
        monitoring_triggers=action_plan["monitoring_triggers"],
        final_review_memo=final_review_memo,
    )


def _evidence_agents_require_hold(agents: list[AgentOutput]) -> bool:
    evidence_agent = next((agent for agent in agents if agent.role == "evidence_audit"), None)
    return bool(evidence_agent and _evidence_agent_requires_hold(evidence_agent))


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


def _committee_action_plan(
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
) -> dict[str, list[str]]:
    """Return dashboard-facing next actions for hold/reject/manual-review cases."""
    tasks: list[str] = []
    missing: list[str] = []
    monitoring: list[str] = []

    if committee_label == "보류":
        tasks.append("보류 사유와 최종 메모가 같은 결론을 가리키는지 담당자가 재확인합니다.")
    elif committee_label == "부적격":
        tasks.append("부적격 확정 전 재무 원자료와 직접 공시 근거를 승인권자가 교차검토합니다.")

    if decision_type == "risk_hold":
        tasks.append("위험 보류 근거가 재무 스트레스인지 외부 중요도 근거인지 분리해 확인합니다.")
        missing.append("위험 보류를 뒷받침하는 직접 공시, 감사의견, 차입/보증, 거래정지 근거")
    elif decision_type == "boundary_hold":
        tasks.append("모델 기준선 근접도와 공개등급 경계구간(BBB-/BB+) 맥락을 재확인합니다.")
    elif decision_type == "mitigation_hold":
        tasks.append("과민경고 완화 근거가 원자료와 외부근거에서 동시에 지지되는지 확인합니다.")
    elif decision_type == "review_hold":
        tasks.append("보류를 해소하기 위해 필요한 재무, 등급, 외부근거 누락 항목을 확인합니다.")
    elif decision_type == "reject":
        tasks.append("veto 또는 부적격 확정 게이트의 원문 근거와 기준일 적합성을 확인합니다.")

    if veto_triggered:
        tasks.append("veto 발동 근거의 회사 일치성, 기준일, 원문 출처를 수동 검증합니다.")
        monitoring.append("veto 관련 정정공시, 소송/제재 확정, 거래정지 해소 공시 발생 시 재검토")
    if hidden_tail_risk.triggered:
        tasks.append("숨은 꼬리위험 근거가 해당 회사 직접 사건인지 원문 기준으로 확인합니다.")
        monitoring.append("동일 사건의 후속 공시, 감사의견 변형, 차입/보증 확대 발생 시 재검토")
    if boundary_review.triggered:
        missing.append("최근 공개 신용등급, 등급전망, 기준선 근접 사유를 확인할 수 있는 reference")
    if secondary_review_risk.triggered:
        missing.append("2차 보조 레이더가 감지한 유동성/현금흐름 약점의 최신 원자료")
    if overwarning_mitigation.triggered:
        monitoring.append(
            "방어축이 약화되거나 투기등급 확률이 기준선을 재상회하면 보류 해소 판단 재검토"
        )
    if reject_confirmation.triggered and not reject_confirmation.confirmed:
        missing.append("부적격 확정에 필요한 치명 외부근거 또는 복수 재무부실 신호")

    if _prior_rating_has_hard_distress_context(bundle.prior_rating_reference):
        tasks.append("기준일 이전 CCC/C/D 등 severe 공개등급의 원문과 이후 해소 근거를 확인합니다.")
        missing.append("기준일 이전 severe 공개등급 원문, 등급전망, 후속 등급조정 또는 해소 공시")
        monitoring.append(
            "등급하향, 회생/상장폐지/거래정지, 감사의견 변형 후속 이벤트 발생 시 재심사"
        )

    if _external_evidence_unavailable(bundle.news_status):
        tasks.append("외부근거 수집을 활성화한 뒤 기준일 이전 직접 공시/뉴스를 재조회합니다.")
        missing.append("기준일 이전 직접 공시/뉴스 원문과 회사명/종목코드 일치 확인")
    if bundle.prior_rating_reference.get("has_prior_rating") is not True:
        missing.append("기준일 이전 공개 신용등급 reference")

    if committee_label == "보류":
        monitoring.append(
            "신규 DART 수시공시, 감사의견 변형, 거래정지, 대규모 차입/보증 발생 시 재심사"
        )
    elif committee_label == "적격" and _external_evidence_unavailable(bundle.news_status):
        monitoring.append("외부근거 수집이 가능해지면 적격 판단의 누락위험을 재점검")

    return {
        "manual_review_tasks": _unique_action_items(tasks),
        "missing_evidence": _unique_action_items(missing),
        "monitoring_triggers": _unique_action_items(monitoring),
    }


def _external_evidence_unavailable(status: str) -> bool:
    return status.strip().lower() in {
        "disabled",
        "missing_credentials",
        "not_implemented",
        "not_requested",
        "placeholder",
        "no_results",
    }


def _unique_action_items(items: list[str], *, limit: int = 5) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_korean_review_text(str(item))
        if not cleaned or cleaned in seen:
            continue
        output.append(cleaned)
        seen.add(cleaned)
        if len(output) >= limit:
            break
    return output


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
    mitigation_residual_risk_reason: str,
    risk_hold_reason_tags: list[RiskHoldReasonTag],
    risk_hold_reason_summary: str,
) -> list[DecisionTraceItem]:
    """Build an auditable deterministic gate trace for Stage 2 decisions."""
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    review_priority = _stage2_review_priority(bundle)
    trigger_reason = _stage2_trigger_reason(bundle)
    prior_hard_distress = _prior_rating_has_hard_distress_context(bundle.prior_rating_reference)
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
            gate="prior_hard_distress_context",
            label="기준일 이전 severe 등급 컨텍스트",
            triggered=prior_hard_distress,
            severity="risk" if prior_hard_distress else "info",
            summary=_prior_hard_distress_trace_summary(bundle.prior_rating_reference),
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
            gate="mitigation_residual_risk",
            label="과민경고 완화 잔여위험",
            triggered=bool(mitigation_residual_risk_reason),
            severity="risk" if mitigation_residual_risk_reason else "info",
            summary=mitigation_residual_risk_reason
            or "과민경고 완화 보류 안에서 위험 보류로 되돌릴 잔여 재무위험은 제한적입니다.",
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


def _prior_hard_distress_trace_summary(prior: dict[str, Any]) -> str:
    if not _prior_rating_has_hard_distress_context(prior):
        return "기준일 이전 CCC/C/D 등 severe 공개등급 컨텍스트는 확인되지 않았습니다."
    rating = str(prior.get("prior_credit_rating") or prior.get("credit_rating") or "").strip()
    rating_date = str(prior.get("prior_rating_date") or "").strip()
    agency = str(prior.get("prior_rating_agency") or "").strip()
    source = f"{agency} " if agency else ""
    date_text = f"({rating_date})" if rating_date else ""
    return (
        f"기준일 이전 {source}공개등급이 {rating}{date_text}로 CCC/C/D 등 "
        "심각한 신용위험 영역에 있어, 현재 재무·외부근거에서 해소 여부를 확인해야 합니다."
    )


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


__all__ = ["build_committee_view", "build_committee_view_model"]
