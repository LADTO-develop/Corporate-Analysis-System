"""RiskRecallQA trigger and advisory policy for eligible-case recall checks."""

from __future__ import annotations

from typing import Any

from cas.agents.nodes.evidence_profile import (
    _external_evidence_profile,
    _has_substantive_external_risk,
    _is_substantive_external_risk_item,
    _metric_above_value,
    _metric_below_value,
    _safe_float,
    _truthy,
)
from cas.agents.nodes.qa_cache import _run_risk_recall_qa_agent_with_cache, _stage2_runtime_config
from cas.agents.nodes.qa_diagnostics import (
    _append_sentence,
    _merge_post_committee_qa_diagnostics,
    _prepend_unique_text,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    RiskRecallQAOutput,
)
from cas.agents.stage2_policy import load_stage2_policy


def _maybe_run_risk_recall_qa(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    structured_outputs: tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput],
    runtime_backend_name: str,
    runtime_diagnostics: dict[str, Any],
) -> RiskRecallQAOutput | None:
    trigger_reasons = _risk_recall_qa_trigger_reasons(
        bundle=bundle,
        committee_view=committee_view,
    )
    runtime_diagnostics["risk_recall_qa_triggered"] = bool(trigger_reasons)
    runtime_diagnostics["risk_recall_qa_trigger_reasons"] = trigger_reasons
    if not trigger_reasons or not _stage2_risk_recall_qa_enabled(runtime_backend_name):
        return None
    try:
        risk_recall_qa, diagnostics = _run_risk_recall_qa_agent_with_cache(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=structured_outputs[0],
            evidence_audit=structured_outputs[1],
            chair_report=structured_outputs[2],
            trigger_reasons=trigger_reasons,
        )
    except Exception as error:
        if not _stage2_risk_recall_qa_fallback_on_error():
            raise
        runtime_diagnostics["risk_recall_qa_error_message"] = str(error)
        runtime_diagnostics["risk_recall_qa_cache_hit"] = False
        return None
    _merge_post_committee_qa_diagnostics(
        runtime_diagnostics,
        diagnostics,
        role="risk_recall_qa",
    )
    runtime_diagnostics["risk_recall_qa_recommended_action"] = risk_recall_qa.recommended_action
    return risk_recall_qa


def _apply_risk_recall_qa_advisory(
    *,
    committee_view: dict[str, Any],
    risk_recall_qa_output: RiskRecallQAOutput,
    bundle: Stage2InputBundle,
    runtime_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    runtime_diagnostics["risk_recall_qa_advisory_applied"] = False
    runtime_diagnostics["risk_recall_qa_advisory_apply_reason"] = ""
    if not _stage2_risk_recall_qa_apply_advisory():
        return committee_view
    if str(committee_view.get("final_committee_label") or "") != "적격":
        return committee_view
    if str(committee_view.get("committee_decision_type") or "") != "eligible":
        return committee_view

    apply_reason = _risk_recall_qa_advisory_apply_reason(
        risk_recall_qa_output=risk_recall_qa_output,
        bundle=bundle,
    )
    if not apply_reason:
        return committee_view

    target_type = (
        "risk_hold"
        if risk_recall_qa_output.recommended_action == "escalate_eligible_to_risk_hold"
        else "boundary_hold"
    )
    adjusted = dict(committee_view)
    adjusted["final_committee_label"] = "보류"
    adjusted["committee_decision_type"] = target_type
    adjusted["committee_decision_type_label"] = (
        "위험 보류" if target_type == "risk_hold" else "경계등급 보류"
    )
    adjusted["committee_risk_signal"] = target_type == "risk_hold"
    if target_type == "risk_hold":
        reason_tags, reason_labels, reason_summary = _risk_recall_hold_reason_fields(apply_reason)
        adjusted["risk_hold_reason_tags"] = reason_tags
        adjusted["risk_hold_reason_labels"] = reason_labels
        adjusted["risk_hold_reason_summary"] = reason_summary
    else:
        adjusted["risk_hold_reason_tags"] = []
        adjusted["risk_hold_reason_labels"] = []
        adjusted["risk_hold_reason_summary"] = ""
    adjustment_note = (
        "RiskRecallQAAgent는 최종 적격 판단을 유지하기에는 기준선/재무/외부근거의 "
        "잔여 위험이 남아 있다고 보아, 최종 라벨을 보류로 재검수하도록 권고했습니다."
    )
    adjusted["conflict_resolution"] = _append_sentence(
        str(adjusted.get("conflict_resolution") or ""),
        adjustment_note,
    )
    adjusted["final_review_memo"] = _append_sentence(
        _neutralize_prior_eligible_final_memo(str(adjusted.get("final_review_memo") or "")),
        (
            "RiskRecallQA 보강 의견: 적격 판단의 위험 누락 가능성을 재점검해 "
            "최종 표시 라벨을 보류로 올립니다."
        ),
    )
    adjusted["key_risk_factors"] = _prepend_unique_text(
        adjusted.get("key_risk_factors"),
        "RiskRecallQA 적격 재검수 경고",
    )
    risk_hold_reason_trace = (
        [
            {
                "gate": "risk_hold_reason_tagging",
                "label": "위험 보류 이유 태그",
                "triggered": True,
                "severity": "risk",
                "summary": adjusted.get("risk_hold_reason_summary") or "",
            }
        ]
        if target_type == "risk_hold"
        else []
    )
    adjusted["decision_trace"] = [
        *list(adjusted.get("decision_trace") or []),
        {
            "gate": "risk_recall_qa_escalation",
            "label": "RiskRecallQA 적격 재검수",
            "triggered": True,
            "severity": "risk" if target_type == "risk_hold" else "watch",
            "summary": adjustment_note,
        },
        *risk_hold_reason_trace,
    ]
    runtime_diagnostics["risk_recall_qa_advisory_applied"] = True
    runtime_diagnostics["risk_recall_qa_adjusted_decision_type"] = target_type
    runtime_diagnostics["risk_recall_qa_advisory_apply_reason"] = apply_reason
    return adjusted


def _neutralize_prior_eligible_final_memo(memo: str) -> str:
    replacements = {
        "최종 위원회 판단은 적격입니다.": "초기 위원회 판단은 적격이었습니다.",
        "최종 의견을 적격으로 정리했습니다.": "초기 의견을 적격으로 정리했습니다.",
        "최종 의견은 적격입니다.": "초기 의견은 적격이었습니다.",
        "최종 라벨은 적격입니다.": "초기 라벨은 적격이었습니다.",
    }
    updated = memo
    for before, after in replacements.items():
        updated = updated.replace(before, after)
    return updated


def _risk_recall_qa_advisory_apply_reason(
    *,
    risk_recall_qa_output: RiskRecallQAOutput,
    bundle: Stage2InputBundle,
) -> str:
    policy = load_stage2_policy()
    action = risk_recall_qa_output.recommended_action
    assessment = risk_recall_qa_output.eligible_safety_assessment
    trigger_reasons = {str(reason) for reason in risk_recall_qa_output.trigger_reasons}
    weak_axes = _risk_recall_weak_financial_axes(bundle)
    confirmed_external_evidence = _risk_recall_confirmed_external_escalation_evidence(bundle)
    if action == "escalate_eligible_to_risk_hold":
        if assessment != "material_missed_risk" or risk_recall_qa_output.confidence < policy.float(
            "risk_recall_qa",
            "advisory",
            "risk_hold_min_confidence",
        ):
            return ""
        if confirmed_external_evidence:
            return "risk_recall_substantive_external_risk"
        if len(weak_axes) >= policy.int(
            "risk_recall_qa",
            "advisory",
            "severe_financial_weakness_min_axes",
        ):
            return "risk_recall_severe_financial_weakness"
        return ""

    if action != "escalate_eligible_to_boundary_hold":
        return ""
    if assessment not in {"needs_boundary_review", "material_missed_risk"}:
        return ""
    if risk_recall_qa_output.confidence < policy.float(
        "risk_recall_qa",
        "advisory",
        "boundary_hold_min_confidence",
    ):
        return ""
    if not trigger_reasons.intersection(
        {
            "eligible_near_threshold",
            "eligible_near_threshold_with_weak_financials",
            "eligible_with_multiple_weak_financial_axes",
            "eligible_with_recall_watch_evidence",
            "eligible_boundary_rating_context",
        }
    ):
        return ""
    near_threshold_weak = bool(
        trigger_reasons.intersection(
            {
                "eligible_near_threshold",
                "eligible_near_threshold_with_weak_financials",
            }
        )
        and len(weak_axes)
        >= policy.int("risk_recall_qa", "advisory", "near_threshold_min_weak_axes")
    )
    multi_axis_weak = "eligible_with_multiple_weak_financial_axes" in trigger_reasons and len(
        weak_axes
    ) >= policy.int("risk_recall_qa", "advisory", "multi_axis_min_weak_axes")
    boundary_weak = "eligible_boundary_rating_context" in trigger_reasons and len(
        weak_axes
    ) >= policy.int("risk_recall_qa", "advisory", "boundary_rating_min_weak_axes")
    if not (confirmed_external_evidence or near_threshold_weak or multi_axis_weak or boundary_weak):
        return ""
    return "risk_recall_boundary_safety_review"


def _risk_recall_confirmed_external_escalation_evidence(bundle: Stage2InputBundle) -> bool:
    """Require confirmed, structured evidence before RiskRecallQA escalates eligible cases."""
    news_cache = bundle.news_cache_snapshot
    profile = _external_evidence_profile(
        news_cache,
        source_feature_row=bundle.source_feature_row,
    )
    if profile["veto_candidate_count"] > 0 or profile["high_confidence_critical_count"] > 0:
        return True

    raw_items = news_cache.get("items", [])
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    for item in items:
        if item.get("company_match") is not True:
            continue
        if not _risk_recall_evidence_item_is_confirmed(item):
            continue
        if _is_substantive_external_risk_item(
            item,
            source_feature_row=bundle.source_feature_row,
        ):
            return True
    return False


def _risk_recall_evidence_item_is_confirmed(item: dict[str, Any]) -> bool:
    policy = load_stage2_policy()
    if item.get("veto_candidate") is True or item.get("critical_context_confirmed") is True:
        return True

    source = str(item.get("source") or "").lower()
    quality = str(item.get("evidence_quality") or "").lower()
    score = _safe_float(item.get("evidence_score"))
    if source == "opendart":
        return score is None or score >= policy.float(
            "risk_recall_qa",
            "evidence",
            "opendart_min_score",
        )
    if quality == "low":
        return False
    if quality in {"medium", "high"} and (
        score is None
        or score >= policy.float("risk_recall_qa", "evidence", "medium_high_min_score")
    ):
        return True
    return score is not None and score >= policy.float(
        "risk_recall_qa",
        "evidence",
        "fallback_min_score",
    )


def _risk_recall_hold_reason_fields(apply_reason: str) -> tuple[list[str], list[str], str]:
    if apply_reason == "risk_recall_substantive_external_risk":
        return (
            ["external_materiality_hold"],
            ["외부 중요도 근거"],
            (
                "위험 보류 이유 태그는 외부 중요도 근거입니다. RiskRecallQA가 적격 판단에서 "
                "놓칠 수 있는 실질 외부근거를 확인해 위험 보류로 올렸습니다."
            ),
        )
    if apply_reason == "risk_recall_severe_financial_weakness":
        return (
            ["financial_stress_hold"],
            ["재무 스트레스"],
            (
                "위험 보류 이유 태그는 재무 스트레스입니다. RiskRecallQA가 현금흐름, "
                "이자보상, 손익, 유동성 중 복수 약점을 확인해 위험 보류로 올렸습니다."
            ),
        )
    return (
        ["model_risk_hold"],
        ["모델 위험 보류"],
        (
            "위험 보류 이유 태그는 모델 위험 보류입니다. 적격으로 유지하기에는 잔여 위험 "
            "신호가 남아 위험 보류로 올렸습니다."
        ),
    )


def _risk_recall_qa_trigger_reasons(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
) -> list[str]:
    if str(committee_view.get("final_committee_label") or "") != "적격":
        return []
    if str(committee_view.get("committee_decision_type") or "") != "eligible":
        return []

    reasons: list[str] = []
    policy = load_stage2_policy()
    near_threshold = _risk_recall_near_threshold(bundle)
    weak_axes = _risk_recall_weak_financial_axes(bundle)
    has_watch_evidence = _has_risk_recall_watch_evidence(bundle.news_cache_snapshot)
    has_substantive_evidence = _has_substantive_external_risk(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    has_boundary_context = _has_rating_boundary_context(bundle)

    near_threshold_min_axes = policy.int(
        "risk_recall_qa",
        "advisory",
        "near_threshold_min_weak_axes",
    )
    multi_axis_min_axes = policy.int(
        "risk_recall_qa",
        "advisory",
        "multi_axis_min_weak_axes",
    )
    if near_threshold and (len(weak_axes) >= near_threshold_min_axes or has_substantive_evidence):
        reasons.append("eligible_near_threshold")
    if near_threshold and len(weak_axes) >= near_threshold_min_axes:
        reasons.append("eligible_near_threshold_with_weak_financials")
    if len(weak_axes) >= multi_axis_min_axes:
        reasons.append("eligible_with_multiple_weak_financial_axes")
    if has_watch_evidence and (
        (near_threshold and len(weak_axes) >= near_threshold_min_axes)
        or len(weak_axes) >= multi_axis_min_axes
        or has_substantive_evidence
    ):
        reasons.append("eligible_with_recall_watch_evidence")
    if has_substantive_evidence:
        reasons.append("eligible_with_substantive_evidence")
    if has_boundary_context and (
        (near_threshold and len(weak_axes) >= near_threshold_min_axes)
        or len(weak_axes) >= multi_axis_min_axes
        or has_substantive_evidence
    ):
        reasons.append("eligible_boundary_rating_context")
    return reasons[:5]


def _risk_recall_near_threshold(bundle: Stage2InputBundle) -> bool:
    policy = load_stage2_policy()
    probability = _safe_float(bundle.probability_speculative)
    threshold = _safe_float(bundle.threshold)
    if threshold is None or threshold <= 0:
        return False
    margin = threshold - (probability or 0.0)
    return bool(0.0 <= margin <= policy.float("risk_recall_qa", "trigger", "near_threshold_margin"))


def _risk_recall_weak_financial_axes(bundle: Stage2InputBundle) -> list[str]:
    policy = load_stage2_policy()
    section = ("risk_recall_qa", "trigger")
    row = bundle.source_feature_row
    axes: list[str] = []
    if _metric_below_value(row, "current_ratio", policy.float(*section, "current_ratio_floor")):
        axes.append("low_current_ratio")
    if _metric_below_value(row, "cash_ratio", policy.float(*section, "cash_ratio_floor")):
        axes.append("low_cash_ratio")
    if (
        _metric_below_value(
            row,
            "cashflow_coverage_ratio",
            policy.float(*section, "cashflow_coverage_ratio_floor"),
        )
        or _metric_below_value(
            row,
            "ocf_to_total_liabilities",
            policy.float(*section, "ocf_to_total_liabilities_floor"),
        )
        or _metric_below_value(row, "ocf_to_sales", policy.float(*section, "ocf_to_sales_floor"))
        or _truthy(row.get("is_2y_consecutive_ocf_deficit"))
    ):
        axes.append("weak_cashflow")
    if _metric_below_value(
        row,
        "interest_coverage_ratio",
        policy.float(*section, "interest_coverage_ratio_floor"),
    ) or _truthy(row.get("icr_under_1")):
        axes.append("weak_interest_coverage")
    if _metric_above_value(row, "debt_ratio", policy.float(*section, "debt_ratio_floor")):
        axes.append("high_debt_ratio")
    if _metric_above_value(
        row,
        "total_borrowings_ratio",
        policy.float(*section, "total_borrowings_ratio_floor"),
    ) or _metric_above_value(
        row,
        "short_term_borrowings_share",
        policy.float(*section, "short_term_borrowings_share_floor"),
    ):
        axes.append("high_borrowing_pressure")
    return axes


def _has_risk_recall_watch_evidence(news_cache: dict[str, Any]) -> bool:
    policy = load_stage2_policy()
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return False
    markers = (
        "유상증자",
        "전환사채",
        "신주인수권",
        "채무보증",
        "감사보고서",
        "소송",
        "계약해지",
        "영업정지",
        "거래정지",
    )
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if item.get("company_match") is not True:
            continue
        score = _safe_float(item.get("evidence_score")) or 0.0
        if score < policy.float("risk_recall_qa", "evidence", "watch_min_score"):
            continue
        text = " ".join(str(item.get(key) or "") for key in ("title", "summary"))
        if any(marker in text for marker in markers):
            return True
    return False


def _has_rating_boundary_context(bundle: Stage2InputBundle) -> bool:
    prior = bundle.prior_rating_reference
    group = str(prior.get("prior_rating_boundary_group") or "").lower()
    rating = str(prior.get("prior_credit_rating") or prior.get("credit_rating") or "").upper()
    if "boundary" in group:
        return True
    return rating in {"BBB-", "BB+"}


def _stage2_risk_recall_qa_enabled(runtime_backend_name: str) -> bool:
    config = _stage2_runtime_config()
    if config.risk_recall_qa_enabled is not None:
        return bool(config.risk_recall_qa_enabled)
    backend = runtime_backend_name.strip().lower()
    return backend.startswith("agno") and "fallback" not in backend


def _stage2_risk_recall_qa_apply_advisory() -> bool:
    return bool(_stage2_runtime_config().risk_recall_qa_apply_advisory)


def _stage2_risk_recall_qa_fallback_on_error() -> bool:
    return bool(_stage2_runtime_config().risk_recall_qa_fallback_on_error)


__all__ = [
    "_apply_risk_recall_qa_advisory",
    "_has_rating_boundary_context",
    "_has_risk_recall_watch_evidence",
    "_maybe_run_risk_recall_qa",
    "_neutralize_prior_eligible_final_memo",
    "_risk_recall_confirmed_external_escalation_evidence",
    "_risk_recall_evidence_item_is_confirmed",
    "_risk_recall_hold_reason_fields",
    "_risk_recall_near_threshold",
    "_risk_recall_qa_advisory_apply_reason",
    "_risk_recall_qa_trigger_reasons",
    "_risk_recall_weak_financial_axes",
    "_stage2_risk_recall_qa_apply_advisory",
    "_stage2_risk_recall_qa_enabled",
    "_stage2_risk_recall_qa_fallback_on_error",
]
