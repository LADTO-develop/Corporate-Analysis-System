"""Financial guardrails used by Stage 2 committee decisions."""

from __future__ import annotations

from typing import Any

from cas.agents.committee_assessments import (
    FinancialResilienceAssessment,
    NoncriticalEvidenceAssessment,
    RejectConfirmationAssessment,
    SecondaryReviewRiskAssessment,
)
from cas.agents.committee_external_evidence import (
    has_nonblocking_external_context,
    no_direct_external_items,
    overwarning_blocking_external_items,
)
from cas.agents.committee_utils import (
    flag_is_false,
    flag_is_true,
    metric_above,
    metric_at_least,
    metric_at_most,
    metric_below,
    safe_float,
    safe_int,
)
from cas.agents.signals.materiality_signals import material_financing_evidence_blocks_tn_hold
from cas.agents.stage2_bundle import Stage2InputBundle


def has_stage2_secondary_trigger(bundle: Stage2InputBundle) -> bool:
    """Return whether Stage 1 marked the case for secondary Stage 2 review."""
    for source in (bundle.model_view, bundle.xgboost_result):
        if bool(source.get("stage2_secondary_trigger")):
            return True
    return False


def secondary_review_requires_hold(bundle: Stage2InputBundle) -> bool:
    """Return whether a secondary trigger is strong enough to block eligible alignment."""
    if not has_stage2_secondary_trigger(bundle):
        return False
    if secondary_overhold_guardrail_reason(bundle):
        return False
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    probability_floor = max(0.28, threshold - 0.10)
    secondary_liquidity_watch = has_secondary_rule_liquidity_watch_signal(bundle)
    confident_secondary_liquidity_watch = secondary_liquidity_watch and (
        probability >= probability_floor
        or (threshold >= 0.28 and _rule_confidence_at_least(bundle, 0.60))
    )
    return (
        probability >= probability_floor
        or has_severe_financial_watch_signal(bundle.source_feature_row)
        or confident_secondary_liquidity_watch
    )


def has_blocking_flags(bundle: Stage2InputBundle) -> bool:
    """Return whether the rule engine emitted any blocking flags."""
    flags = bundle.rule_result.get("blocking_flags", []) or []
    return any(str(flag).strip() for flag in flags)


def has_isolated_interest_cover_defense(bundle: Stage2InputBundle) -> bool:
    """Allow TN guardrail when ICR is the only hard flag and OCF coverage is strong."""
    raw_flags = bundle.rule_result.get("blocking_flags", []) or []
    flags = {str(flag).strip().lower() for flag in raw_flags if str(flag).strip()}
    if not flags or not flags.issubset(
        {"interest_coverage_under_1", "icr_under_1", "interest_coverage_ratio_under_1"}
    ):
        return False
    return has_isolated_interest_cover_row_defense(bundle.source_feature_row)


def has_isolated_interest_cover_row_defense(row: dict[str, Any]) -> bool:
    """Return whether cash flow and low borrowings offset a single-year ICR dip."""
    return bool(
        (
            flag_is_true(row.get("icr_under_1"))
            or metric_below(row, "interest_coverage_ratio", 1.0)
        )
        and metric_at_least(row, "current_ratio", 1.2)
        and metric_at_least(row, "cash_ratio", 0.15)
        and metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and metric_at_most(row, "total_borrowings_ratio", 0.10)
        and metric_at_most(row, "capital_impairment_ratio", 0.0)
        and not flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        and not flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
        and not metric_below(row, "net_margin", -0.05)
    )


def has_isolated_icr_review_buffer(row: dict[str, Any]) -> bool:
    """Downgrade risk display when an ICR dip is offset by OCF, capital, and low debt."""
    if not (
        flag_is_true(row.get("icr_under_1")) or metric_below(row, "interest_coverage_ratio", 1.0)
    ):
        return False
    if flag_is_true(row.get("is_2y_consecutive_operating_loss")) or flag_is_true(
        row.get("is_2y_consecutive_ocf_deficit")
    ):
        return False
    if metric_above(row, "capital_impairment_ratio", 0.0):
        return False
    if metric_below(row, "net_margin", -0.05):
        return False
    return bool(
        metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and metric_at_least(row, "equity_ratio", 0.70)
        and metric_at_most(row, "debt_ratio", 0.50)
        and metric_at_most(row, "total_borrowings_ratio", 0.20)
    )


def has_secondary_rule_liquidity_watch_signal(bundle: Stage2InputBundle) -> bool:
    """Preserve hold for low-but-near-threshold eligible calls with liquidity rule watch."""
    if not has_stage2_secondary_trigger(bundle):
        return False
    if has_financial_statement_missing_placeholder(bundle.source_feature_row):
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
    has_reported_liquidity_weakness = metric_below(
        bundle.source_feature_row, "current_ratio", 1.0
    ) or metric_below(bundle.source_feature_row, "cash_ratio", 0.10)
    if has_reported_liquidity_weakness and has_cashflow_backed_liquidity_buffer(
        bundle.source_feature_row
    ):
        return False
    if (
        any(marker in reason_text for marker in liquidity_markers)
        and has_reported_liquidity_weakness
    ):
        return True
    return bool(has_reported_liquidity_weakness)


def has_cashflow_backed_liquidity_buffer(row: dict[str, Any]) -> bool:
    """Allow a current-ratio watch through when cash, OCF, and capital are strong."""
    return bool(
        metric_below(row, "current_ratio", 1.0)
        and metric_at_least(row, "cash_ratio", 0.25)
        and metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and metric_at_least(row, "ocf_to_sales", 0.0)
        and metric_at_least(row, "interest_coverage_ratio", 3.0)
        and metric_at_least(row, "equity_ratio", 0.40)
        and metric_at_most(row, "debt_ratio", 1.50)
        and (
            metric_at_most(row, "short_term_borrowings_share", 0.80)
            or metric_at_most(row, "total_borrowings_ratio", 0.30)
        )
        and metric_at_most(row, "capital_impairment_ratio", 0.0)
        and not flag_is_true(row.get("icr_under_1"))
        and not flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        and not flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
        and not metric_below(row, "net_margin", -0.05)
    )


def secondary_overhold_guardrail_reason(bundle: Stage2InputBundle) -> str:
    """Keep defensive investment-grade cases from being held by secondary radar alone."""
    if bundle.prediction_label != "투자적격" or not has_stage2_secondary_trigger(bundle):
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
    if has_blocking_flags(bundle) and not has_isolated_interest_cover_defense(bundle):
        return ""
    if has_severe_financial_watch_signal(
        bundle.source_feature_row
    ) and not has_isolated_interest_cover_defense(bundle):
        return ""
    if has_extreme_financial_distress_signal(bundle.source_feature_row):
        return ""
    if has_secondary_overhold_guardrail_blocker(
        bundle.source_feature_row
    ) and not has_isolated_interest_cover_defense(bundle):
        return ""
    if has_secondary_rule_liquidity_watch_signal(bundle):
        return ""
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if material_financing_evidence_blocks_tn_hold(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if prior_rating_is_speculative(bundle.prior_rating_reference):
        return ""

    supports = secondary_overhold_guardrail_supports(bundle.source_feature_row)
    if len(supports) < 2 or "현금흐름" not in supports:
        return ""

    return (
        "정상기업 과잉 보류 방어 guardrail: 1차 모델은 투자적격이고 "
        f"투기등급 확률 {probability:.1%}가 기준선 {threshold:.1%} 아래입니다. "
        "직접 검증된 외부 치명근거와 강한 재무 부실 신호가 없고 "
        f"{', '.join(supports[:3])} 축이 방어적이어서 45개 보조 레이더 단독 신호만으로는 "
        "위험 보류나 경계 보류로 올리지 않습니다."
    )


def secondary_overhold_guardrail_supports(row: dict[str, Any]) -> list[str]:
    """Return broad financial-defense categories for TN over-hold prevention."""
    supports: list[str] = []
    liquidity_support = metric_at_least(row, "current_ratio", 1.2) or metric_at_least(
        row, "cash_ratio", 0.15
    )
    if liquidity_support:
        supports.append("유동성")

    cashflow_signal = (
        metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        or metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        or metric_at_least(row, "ocf_to_sales", 0.0)
    )
    interest_service_signal = metric_at_least(row, "interest_coverage_ratio", 1.0) and not (
        flag_is_true(row.get("icr_under_1"))
    )
    if cashflow_signal and (
        interest_service_signal or has_isolated_interest_cover_row_defense(row)
    ):
        supports.append("현금흐름")

    capital_support = (
        metric_at_least(row, "equity_ratio", 0.40)
        and (
            metric_at_most(row, "debt_ratio", 1.50)
            or metric_at_most(row, "total_borrowings_ratio", 0.50)
        )
        and not metric_above(row, "capital_impairment_ratio", 0.0)
    )
    if capital_support:
        supports.append("자본")
    return supports


def has_secondary_overhold_guardrail_blocker(row: dict[str, Any]) -> bool:
    """Return moderate stress signals that should keep a near-boundary FN on hold."""
    if metric_below(row, "net_margin", -0.10):
        return True
    if metric_below(row, "ocf_to_sales", 0.0) and metric_below(
        row, "ocf_to_total_liabilities", 0.0
    ):
        return True
    weak_interest_cover = metric_below(row, "interest_coverage_ratio", 3.0)
    weak_capital_buffer = metric_below(row, "equity_ratio", 0.40) and metric_above(
        row, "debt_ratio", 1.50
    )
    return bool(weak_interest_cover and weak_capital_buffer)


def has_financial_statement_missing_placeholder(row: dict[str, Any]) -> bool:
    """Detect rows where absent statements are encoded as zero/capped ratios."""
    return bool(
        metric_at_most(row, "assets_total", 0.0)
        and metric_at_most(row, "gross_profit", 0.0)
        and metric_at_least(row, "interest_coverage_ratio", 999_999.0)
        and metric_at_least(row, "cashflow_coverage_ratio", 999_999.0)
    )


def has_severe_financial_watch_signal(row: dict[str, Any]) -> bool:
    """Return whether row-level stress is severe enough to block simple mitigation."""
    hard_stress_flags = [
        flag_is_true(row.get("icr_under_1")),
        flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        and flag_is_true(row.get("is_2y_consecutive_ocf_deficit")),
        metric_above(row, "capital_impairment_ratio", 0.0),
        metric_below(row, "interest_coverage_ratio", 1.0),
    ]
    if any(hard_stress_flags):
        return True
    weak_liquidity = metric_below(row, "current_ratio", 0.7) and metric_below(
        row, "cash_ratio", 0.05
    )
    weak_cashflow = metric_below(row, "cashflow_coverage_ratio", 0.0) or metric_below(
        row, "ocf_to_total_liabilities", 0.0
    )
    return bool(weak_liquidity and weak_cashflow)


def risk_hold_has_financial_stress(
    bundle: Stage2InputBundle,
    *,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    reject_confirmation: RejectConfirmationAssessment,
) -> bool:
    """Return whether a risk_hold label is supported by financial stress."""
    row = bundle.source_feature_row
    if has_severe_financial_watch_signal(row):
        return True
    if has_secondary_overhold_guardrail_blocker(row):
        return True
    if reject_confirmation.signal_count >= 2:
        return True
    financial_flags = [
        flag_is_true(row.get("icr_under_1")) or metric_below(row, "interest_coverage_ratio", 1.0),
        flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
        or metric_below(row, "cashflow_coverage_ratio", 0.0)
        or metric_below(row, "ocf_to_total_liabilities", 0.0)
        or metric_below(row, "ocf_to_sales", 0.0),
        flag_is_true(row.get("is_2y_consecutive_operating_loss"))
        or metric_below(row, "net_margin", -0.10),
        metric_above(row, "capital_impairment_ratio", 0.0)
        or (metric_below(row, "equity_ratio", 0.25) and metric_above(row, "debt_ratio", 1.50)),
        metric_below(row, "current_ratio", 1.0) and metric_below(row, "cash_ratio", 0.10),
    ]
    if sum(1 for flag in financial_flags if flag) >= 2:
        return True
    return bool(secondary_review_risk.triggered and secondary_review_risk.risk_signal)


def secondary_review_risk_assessment(bundle: Stage2InputBundle) -> SecondaryReviewRiskAssessment:
    """Flag likely FN cases surfaced by the 45-feature Stage 2 review radar."""
    if bundle.prediction_label != "투자적격" or not has_stage2_secondary_trigger(bundle):
        return SecondaryReviewRiskAssessment(False, "", "none")

    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    review_priority = _stage2_review_priority(bundle)
    probability_floor = max(0.28, threshold - 0.10)
    meets_probability_floor = probability >= probability_floor
    near_threshold = probability >= threshold - 0.10 and meets_probability_floor
    priority_requires_hold = review_priority in {"medium", "high", "critical"}
    severe_watch = has_severe_financial_watch_signal(bundle.source_feature_row)
    secondary_liquidity_watch = has_secondary_rule_liquidity_watch_signal(bundle)
    rule_liquidity_watch = secondary_liquidity_watch and (
        meets_probability_floor or (threshold >= 0.28 and _rule_confidence_at_least(bundle, 0.60))
    )
    if secondary_overhold_guardrail_reason(bundle):
        return SecondaryReviewRiskAssessment(False, "", review_priority)
    risk_signal_floor = max(0.28, threshold - 0.04)
    risk_signal_corroborated = secondary_review_risk_signal_corroborated(
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


def secondary_review_risk_signal_corroborated(
    bundle: Stage2InputBundle,
    *,
    severe_watch: bool,
    rule_liquidity_watch: bool,
) -> bool:
    """Require corroboration before showing a secondary review hold as a risk signal."""
    if severe_watch:
        return not has_isolated_icr_review_buffer(bundle.source_feature_row)
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return True
    if material_financing_evidence_blocks_tn_hold(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return True
    prior = bundle.prior_rating_reference
    return prior_rating_is_speculative(prior) or prior_rating_is_exact_boundary(prior)


def prior_rating_is_exact_boundary(prior: dict[str, Any]) -> bool:
    """Return whether prior rating is exactly at the BBB-/BB+ boundary."""
    if not prior or prior.get("has_prior_rating") is not True:
        return False
    group = str(prior.get("prior_rating_boundary_group") or "").strip()
    if group == "exact_bbb_minus_bb_plus_boundary":
        return True
    rating = str(prior.get("prior_credit_rating") or "").strip()
    return rating in {"BBB-", "BB+"}


def prior_rating_boundary_requires_hold(bundle: Stage2InputBundle) -> bool:
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


def cash_rich_loss_stage_overwarning_buffer_reason(bundle: Stage2InputBundle) -> str:
    """Soften high model warnings when losses are buffered by unusually strong liquidity."""
    row = bundle.source_feature_row
    probability = bundle.probability_speculative
    if probability < 0.85:
        return ""
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if not (
        metric_at_least(row, "current_ratio", 2.0)
        and metric_at_least(row, "cash_ratio", 0.50)
        and metric_at_least(row, "equity_ratio", 0.60)
        and metric_at_most(row, "debt_ratio", 0.50)
        and metric_at_most(row, "total_borrowings_ratio", 0.10)
    ):
        return ""
    if flag_is_true(row.get("is_2y_consecutive_operating_loss")) or flag_is_true(
        row.get("is_2y_consecutive_ocf_deficit")
    ):
        return ""
    if not (
        metric_at_least(row, "cashflow_coverage_ratio", 0.0)
        or metric_at_least(row, "ocf_to_total_liabilities", 0.0)
        or metric_at_least(row, "ocf_to_sales", 0.0)
    ):
        return ""
    return (
        f"현금·자본 버퍼 기반 과민경고 완화: 투기등급 확률은 {probability:.1%}로 높지만 "
        "유동비율·현금비율·자기자본비율이 높고 차입 부담이 낮으며, 반복 영업손실이나 "
        "반복 OCF 적자는 확인되지 않았습니다. 현재 손익성 악화가 즉시 부도위험으로 "
        "연결되는지 추가 확인이 필요하므로 부적격 확정보다는 보류로 완화합니다."
    )


def prior_boundary_overwarning_buffer_reason(
    bundle: Stage2InputBundle,
    *,
    noncritical_evidence: NoncriticalEvidenceAssessment,
) -> str:
    """Soften high-probability boundary-grade warnings unless distress is decisive."""
    prior = bundle.prior_rating_reference
    if not prior_rating_is_exact_boundary(prior):
        return ""
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability <= max(threshold + 0.20, 0.55):
        return ""
    if has_extreme_financial_distress_signal(bundle.source_feature_row):
        return ""
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if not (
        noncritical_evidence.triggered
        or has_nonblocking_external_context(bundle.news_cache_snapshot)
        or no_direct_external_items(bundle.news_cache_snapshot)
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


def prior_rating_is_speculative(prior: dict[str, Any]) -> bool:
    """Return whether prior public rating is in speculative-grade territory."""
    if not prior or prior.get("has_prior_rating") is not True:
        return False
    rank = safe_int(prior.get("prior_credit_rating_rank"))
    if rank is not None:
        return bool(rank >= 11)
    rating = str(prior.get("prior_credit_rating") or "").strip().upper()
    return rating in {"BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"}


def model_only_overwarning_buffer_reason(
    bundle: Stage2InputBundle,
    *,
    mitigating_factors: list[str],
) -> str:
    """Downgrade unsupported high-probability reject calls to hold, not eligible."""
    probability = bundle.probability_speculative
    threshold = _model_threshold(bundle)
    if probability <= threshold + 0.10:
        return ""
    if probability >= 0.90:
        return ""
    if not mitigating_factors:
        return ""
    cashflow_backed_resilience = has_cashflow_backed_fp_resilience(bundle.source_feature_row)
    if not cashflow_backed_resilience:
        return ""
    if has_blocking_flags(bundle) and not cashflow_backed_resilience:
        return ""
    if has_severe_financial_watch_signal(bundle.source_feature_row) and not cashflow_backed_resilience:
        return ""
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""

    news_status = bundle.news_status
    if not _external_evidence_unavailable(news_status) and not no_direct_external_items(
        bundle.news_cache_snapshot
    ):
        return ""
    return (
        f"고확률 모델 단독 경고 완화: 투기등급 확률은 {probability:.1%}로 높고 "
        "일부 손익·이자보상 스트레스가 있지만, OCF와 자본/부채 구조가 방어력을 "
        "제공합니다. 직접 외부 치명근거도 확인되지 않았으므로 즉시 부적격 "
        "확정보다는 보류로 재점검합니다."
    )


def has_cashflow_backed_fp_resilience(row: dict[str, Any]) -> bool:
    """Allow hold, not reject, when stress is offset by cash-flow and balance-sheet buffers."""
    if flag_is_true(row.get("is_2y_consecutive_ocf_deficit")):
        return False
    if metric_above(row, "capital_impairment_ratio", 0.0):
        return False
    cashflow_support = (
        metric_at_least(row, "cashflow_coverage_ratio", 0.0)
        or metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        or metric_at_least(row, "ocf_to_sales", 0.0)
    )
    balance_sheet_support = metric_at_least(row, "equity_ratio", 0.40) and metric_at_most(
        row, "debt_ratio", 1.50
    )
    borrowing_support = metric_at_most(row, "total_borrowings_ratio", 0.40) or metric_at_most(
        row, "short_term_borrowings_share", 0.70
    )
    return bool(cashflow_support and balance_sheet_support and borrowing_support)


def has_extreme_financial_distress_signal(row: dict[str, Any]) -> bool:
    """Return whether financial stress is too severe to soften a boundary warning."""
    if metric_above(row, "capital_impairment_ratio", 0.50):
        return True
    if metric_below(row, "equity_ratio", 0.15):
        return True
    if metric_above(row, "debt_ratio", 5.0):
        return True

    short_term_maturity_wall = metric_at_least(row, "short_term_borrowings_share", 0.95)
    weak_cashflow = (
        metric_below(row, "cashflow_coverage_ratio", 0.0)
        or metric_below(row, "ocf_to_total_liabilities", 0.0)
        or metric_below(row, "ocf_to_sales", 0.0)
    )
    recurring_loss_or_ocf_deficit = flag_is_true(
        row.get("is_2y_consecutive_operating_loss")
    ) or flag_is_true(row.get("is_2y_consecutive_ocf_deficit"))
    interest_blocked = flag_is_true(row.get("icr_under_1")) or metric_below(
        row, "interest_coverage_ratio", 1.0
    )
    return bool(
        short_term_maturity_wall
        and weak_cashflow
        and recurring_loss_or_ocf_deficit
        and interest_blocked
    )


def financial_resilience_overwarning_assessment(
    row: dict[str, Any],
) -> FinancialResilienceAssessment:
    """Detect high-risk model calls that still show broad financial defense capacity."""
    support_checks = [
        ("유동비율 1.2배 이상", metric_at_least(row, "current_ratio", 1.2)),
        ("현금비율 15% 이상", metric_at_least(row, "cash_ratio", 0.15)),
        ("자기자본비율 40% 이상", metric_at_least(row, "equity_ratio", 0.40)),
        ("부채비율 150% 이하", metric_at_most(row, "debt_ratio", 1.50)),
        ("총차입금 비중 50% 이하", metric_at_most(row, "total_borrowings_ratio", 0.50)),
        ("자본잠식 신호 없음", metric_at_most(row, "capital_impairment_ratio", 0.0)),
        ("이자보상배율 1배 이상", metric_at_least(row, "interest_coverage_ratio", 1.0)),
        ("순이익률 흑자", metric_at_least(row, "net_margin", 0.0)),
        ("OCF/매출액 양수", metric_at_least(row, "ocf_to_sales", 0.0)),
        ("2년 연속 영업손실 아님", flag_is_false(row.get("is_2y_consecutive_operating_loss"))),
        ("2년 연속 OCF 적자 아님", flag_is_false(row.get("is_2y_consecutive_ocf_deficit"))),
        ("ICR 1 미만 플래그 없음", flag_is_false(row.get("icr_under_1"))),
        ("단기차입금 비중 80% 이하", metric_at_most(row, "short_term_borrowings_share", 0.80)),
    ]
    blocker_checks = [
        flag_is_true(row.get("is_2y_consecutive_operating_loss")),
        flag_is_true(row.get("is_2y_consecutive_ocf_deficit")),
        flag_is_true(row.get("icr_under_1")),
        metric_below(row, "net_margin", -0.10),
        metric_below(row, "equity_ratio", 0.25),
        metric_above(row, "capital_impairment_ratio", 0.0),
        metric_above(row, "total_borrowings_ratio", 0.65),
        metric_above(row, "short_term_borrowings_share", 0.90),
    ]
    active_supports = [label for label, passed in support_checks if passed]
    support_count = len(active_supports)
    blocker_count = sum(1 for passed in blocker_checks if passed)
    core_defense = (
        metric_at_least(row, "current_ratio", 1.2)
        and metric_at_least(row, "cash_ratio", 0.15)
        and metric_at_least(row, "equity_ratio", 0.40)
        and metric_at_most(row, "debt_ratio", 1.50)
        and metric_at_least(row, "interest_coverage_ratio", 1.0)
        and metric_at_least(row, "net_margin", 0.0)
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


def _stable_prior_cashflow_overhold_guardrail_reason(bundle: Stage2InputBundle) -> str:
    """Lower near-threshold TN holds when prior rating and OCF defense are strong."""
    if not _prior_rating_is_stable_investment_non_boundary(bundle.prior_rating_reference):
        return ""
    if overwarning_blocking_external_items(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if material_financing_evidence_blocks_tn_hold(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    ):
        return ""
    if has_extreme_financial_distress_signal(bundle.source_feature_row):
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
    rank = safe_int(prior.get("prior_credit_rating_rank"))
    if rank is not None:
        return bool(rank <= 8)
    rating = str(prior.get("prior_credit_rating") or "").strip().upper()
    return rating in {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+"}


def _has_cashflow_backed_near_threshold_tn_defense(row: dict[str, Any]) -> bool:
    """Allow eligible alignment when a single ICR dip is offset by cash generation."""
    if not (
        flag_is_true(row.get("icr_under_1")) or metric_below(row, "interest_coverage_ratio", 1.0)
    ):
        return False
    if flag_is_true(row.get("is_2y_consecutive_operating_loss")) or flag_is_true(
        row.get("is_2y_consecutive_ocf_deficit")
    ):
        return False
    if metric_above(row, "capital_impairment_ratio", 0.0):
        return False
    if metric_below(row, "net_margin", -0.10):
        return False

    cashflow_support = (
        metric_at_least(row, "cashflow_coverage_ratio", 1.0)
        and metric_at_least(row, "ocf_to_total_liabilities", 0.05)
        and metric_at_least(row, "ocf_to_sales", 0.0)
    )
    balance_or_borrowing_support = metric_at_least(row, "cash_ratio", 0.05) or metric_at_most(
        row, "total_borrowings_ratio", 0.55
    )
    return bool(cashflow_support and balance_or_borrowing_support)


def _rule_confidence_at_least(bundle: Stage2InputBundle, threshold: float) -> bool:
    confidence = safe_float(bundle.rule_result.get("confidence"))
    return confidence is not None and confidence >= threshold


def _model_threshold(bundle: Stage2InputBundle) -> float:
    for source in (bundle.xgboost_result, bundle.model_view, bundle.rule_result):
        for key in ("threshold", "threshold_tuned", "decision_threshold"):
            value = safe_float(source.get(key))
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
    "cash_rich_loss_stage_overwarning_buffer_reason",
    "financial_resilience_overwarning_assessment",
    "has_blocking_flags",
    "has_extreme_financial_distress_signal",
    "has_isolated_icr_review_buffer",
    "has_isolated_interest_cover_defense",
    "has_secondary_overhold_guardrail_blocker",
    "has_severe_financial_watch_signal",
    "has_stage2_secondary_trigger",
    "model_only_overwarning_buffer_reason",
    "prior_boundary_overwarning_buffer_reason",
    "prior_rating_boundary_requires_hold",
    "prior_rating_is_exact_boundary",
    "prior_rating_is_speculative",
    "risk_hold_has_financial_stress",
    "secondary_overhold_guardrail_reason",
    "secondary_review_requires_hold",
    "secondary_review_risk_assessment",
]
