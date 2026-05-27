"""Deterministic Stage 2 agent implementations."""

from __future__ import annotations

from cas.agents.nodes.committee_feature_formatting import (
    describe_top_drivers,
    humanize_category,
    humanize_industry,
    humanize_size_group,
)
from cas.agents.nodes.evidence_profile import (
    _evidence_audit_conclusion,
    _evidence_audit_confidence,
    _evidence_limitations,
    _evidence_reliability_text,
    _external_evidence_profile,
    _model_evidence_challenge,
)
from cas.agents.signals import (
    evaluate_debt_liquidity,
    evaluate_evidence_treatment,
    evaluate_external_evidence,
    evaluate_macro_market,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.state import Recommendation


def _quant_credit_agent(bundle: Stage2InputBundle) -> QuantCreditOutput:
    source_row = bundle.source_feature_row
    peer_by_feature = bundle.peer_rows_by_feature
    company_name = bundle.company_name
    market = humanize_category(source_row.get("market"), fallback=bundle.market)
    industry = humanize_industry(source_row.get("industry_macro_category"))
    size_group = humanize_size_group(source_row.get("firm_size_group"))

    probability = bundle.probability_speculative
    prediction_label = bundle.prediction_label
    # Stage 1의 top_drivers와 source row 원값, peer comparison을 같이 묶어서
    # "모델이 왜 그렇게 판단했는지"를 사람 문장으로 바꾸는 것이 QuantCreditAgent의 핵심 역할이다.
    driver_details = describe_top_drivers(bundle.xgboost_result, source_row, peer_by_feature)
    risk_items = [item for item in driver_details if item["direction"] == "risk"]
    support_items = [item for item in driver_details if item["direction"] == "support"]
    secondary_triggered = bool(bundle.model_view.get("stage2_secondary_trigger"))
    review_priority = str(bundle.model_view.get("stage2_review_priority") or "none")
    trigger_reason = str(bundle.model_view.get("trigger_reason") or "")
    overwarning_candidate = bool(bundle.model_view.get("stage2_overwarning_filter_candidate"))
    overwarning_reason = str(bundle.model_view.get("overwarning_filter_reason") or "")

    if risk_items:
        primary_risk = f"{risk_items[0]['feature']}이(가) 위험을 높이는 요인으로 해석됩니다."
    else:
        primary_risk = "현재 상위 SHAP 변수에서 뚜렷한 위험 가중 요인은 제한적으로 관찰됩니다."

    if support_items:
        primary_support = f"{support_items[0]['feature']}이(가) 완화 요인으로 작용하고 있습니다."
    else:
        primary_support = "상위 변수 기준 완화 요인은 제한적으로 확인됩니다."

    summary = (
        f"QuantCreditAgent는 {company_name}이(가) {market} 시장의 {industry} "
        f"{size_group} 분류에 속한다는 맥락에서 Stage 1 결과를 해석했습니다. "
        f"모델은 현재 기업을 {prediction_label}으로 판단했습니다. "
        f"투기등급 위험확률은 {probability:.1%}이며, {primary_risk} {primary_support}"
    )
    if secondary_triggered:
        summary += (
            f" 다만 full_review_trigger_73(stage2_review_aux) 신호가 `{review_priority}` 우선순위의 "
            f"추가 위원회 검토 대상으로 표시했습니다."
        )
    if overwarning_candidate:
        summary += (
            " 한편 조합형 재무 스트레스 필터는 1차 위험 경고가 과민할 가능성을 "
            "완화 요인으로 재확인하라고 표시했습니다."
        )
    key_risk_factors = [str(item.get("detail", "")) for item in risk_items if item.get("detail")]
    if secondary_triggered and trigger_reason:
        key_risk_factors.insert(
            0,
            f"full_review_trigger_73(stage2_review_aux) 검토 신호: {trigger_reason}",
        )
    mitigating_factors = [
        str(item.get("detail", "")) for item in support_items if item.get("detail")
    ]
    if overwarning_candidate and overwarning_reason:
        mitigating_factors.insert(0, f"과민 경고 가능성 검토 신호: {overwarning_reason}")

    return QuantCreditOutput(
        quant_summary=summary,
        model_rationale=(
            f"상위 SHAP 변수 {min(len(driver_details), 3)}개를 기준으로 모델 판단의 근거를 정리했습니다."
        ),
        key_risk_factors=key_risk_factors,
        mitigating_factors=mitigating_factors,
        confidence=0.82 if bundle.xgboost_result else 0.35,
    )


def _evidence_audit_agent(bundle: Stage2InputBundle) -> EvidenceAuditOutput:
    status = bundle.news_status
    debt_signals = evaluate_debt_liquidity(bundle)
    macro_signals = evaluate_macro_market(bundle)
    external_signals = evaluate_external_evidence(bundle.news_cache_snapshot)
    evidence_profile = _external_evidence_profile(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    model_challenge = _model_evidence_challenge(
        bundle=bundle,
        debt_findings=debt_signals.findings,
        evidence_profile=evidence_profile,
    )
    audit_conclusion = _evidence_audit_conclusion(
        bundle=bundle,
        debt_findings=debt_signals.findings,
        evidence_profile=evidence_profile,
    )
    evidence_treatment = evaluate_evidence_treatment(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    summary = (
        "EvidenceAuditAgent는 뉴스·공시·거시환경·산업 맥락과 부채/유동성 신호를 "
        "결합해 재무제표에 덜 드러난 꼬리 위험을 점검했습니다. "
        f"{debt_signals.summary} {model_challenge}"
    )
    return EvidenceAuditOutput(
        evidence_summary=summary,
        evidence_status=status,
        evidence_reliability=_evidence_reliability_text(evidence_profile),
        evidence_strength=evidence_profile["strength"],
        model_challenge=model_challenge,
        audit_conclusion=audit_conclusion,
        debt_liquidity_cross_check=debt_signals.findings,
        macro_industry_sensitivity=macro_signals.findings,
        external_evidence_findings=[
            str(evidence_profile["finding"]),
            (
                "구조화 근거 판정: "
                f"recommended_evidence_treatment={evidence_treatment.recommended_evidence_treatment}; "
                f"critical={evidence_treatment.critical_evidence_count}; "
                f"watch={evidence_treatment.watch_context_count}"
            ),
            *external_signals.findings,
        ],
        evidence_limitations=_evidence_limitations(
            bundle.news_cache_snapshot,
            evidence_profile=evidence_profile,
        ),
        critical_evidence_count=evidence_treatment.critical_evidence_count,
        watch_context_count=evidence_treatment.watch_context_count,
        materiality_summary=evidence_treatment.materiality_summary,
        hard_distress_detected=evidence_treatment.hard_distress_detected,
        recommended_evidence_treatment=evidence_treatment.recommended_evidence_treatment,
        confidence=_evidence_audit_confidence(
            status=status,
            debt_confidence=debt_signals.confidence,
            evidence_profile=evidence_profile,
        ),
    )


def _chair_report_agent(
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
) -> ChairReportOutput:
    prediction_label = bundle.prediction_label
    probability = bundle.probability_speculative
    summary = (
        f"ChairReportAgent는 모델 원판단 {prediction_label}과 위험확률 {probability:.1%}를 "
        "그대로 보존하면서, QuantCreditAgent의 정량 해석과 EvidenceAuditAgent의 "
        f"검증 근거를 종합했습니다. 현재 서비스 recommendation은 {recommendation}입니다."
    )
    return ChairReportOutput(
        report_summary=summary,
        model_preservation_note=(
            "정량 판단은 model_view로 보존하고, committee_view에서는 해석과 보완 의견만 추가합니다."
        ),
        committee_scope_note=(
            "최종 보고서는 적격/보류/부적격 3단 위원회 의견과 주요 위험/완화 요인을 함께 제시합니다."
        ),
        final_review_memo_seed=(
            "ChairReportAgent는 정량 해석과 검증 근거를 사람이 읽는 심사 메모로 연결합니다."
        ),
        confidence=max(0.5, confidence),
    )
