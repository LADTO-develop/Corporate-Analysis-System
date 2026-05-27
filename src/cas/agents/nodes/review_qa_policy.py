"""ReviewQA trigger and advisory policy for post-committee mitigation checks."""

from __future__ import annotations

from typing import Any

from cas.agents.nodes.evidence_profile import (
    _external_evidence_profile,
    _has_substantive_external_risk,
    _is_substantive_external_risk_item,
    _metric_above_value,
    _metric_at_least_value,
    _metric_at_most_value,
    _metric_below_value,
    _safe_float,
    _truthy,
)
from cas.agents.nodes.qa_cache import _run_review_qa_agent_with_cache, _stage2_runtime_config
from cas.agents.nodes.qa_diagnostics import (
    _append_sentence,
    _merge_review_qa_diagnostics,
    _prepend_unique_text,
)
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    ReviewQAOutput,
)
from cas.agents.stage2_policy import load_stage2_policy


def _maybe_run_review_qa(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    structured_outputs: tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput],
    runtime_backend_name: str,
    runtime_diagnostics: dict[str, Any],
) -> ReviewQAOutput | None:
    trigger_reasons = _review_qa_trigger_reasons(bundle=bundle, committee_view=committee_view)
    runtime_diagnostics["review_qa_triggered"] = bool(trigger_reasons)
    runtime_diagnostics["review_qa_trigger_reasons"] = trigger_reasons
    if not trigger_reasons or not _stage2_review_qa_enabled(runtime_backend_name):
        return None
    try:
        review_qa, diagnostics = _run_review_qa_agent_with_cache(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=structured_outputs[0],
            evidence_audit=structured_outputs[1],
            chair_report=structured_outputs[2],
            trigger_reasons=trigger_reasons,
        )
    except Exception as error:
        if not _stage2_review_qa_fallback_on_error():
            raise
        runtime_diagnostics["review_qa_error_message"] = str(error)
        runtime_diagnostics["review_qa_cache_hit"] = False
        return None
    _merge_review_qa_diagnostics(runtime_diagnostics, diagnostics)
    runtime_diagnostics["review_qa_recommended_action"] = review_qa.recommended_action
    return review_qa


def _apply_review_qa_advisory(
    *,
    committee_view: dict[str, Any],
    review_qa_output: ReviewQAOutput,
    bundle: Stage2InputBundle | None = None,
    news_cache_snapshot: dict[str, Any] | None = None,
    runtime_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    runtime_diagnostics["review_qa_advisory_applied"] = False
    runtime_diagnostics["review_qa_advisory_apply_reason"] = ""
    if not _stage2_review_qa_apply_advisory():
        return committee_view
    if bool(committee_view.get("veto_triggered", False)) or bool(
        committee_view.get("hidden_tail_risk_flag", False)
    ):
        return committee_view

    final_label = str(committee_view.get("final_committee_label") or "")
    decision_type = str(committee_view.get("committee_decision_type") or "")
    if final_label == "보류" and decision_type == "risk_hold":
        if review_qa_output.recommended_action != "downgrade_risk_hold_to_boundary_hold":
            return committee_view
        return _apply_review_qa_risk_hold_advisory(
            committee_view=committee_view,
            review_qa_output=review_qa_output,
            bundle=bundle,
            news_cache_snapshot=news_cache_snapshot,
            runtime_diagnostics=runtime_diagnostics,
        )
    if final_label == "부적격" and decision_type == "reject":
        return _apply_review_qa_reject_advisory(
            committee_view=committee_view,
            review_qa_output=review_qa_output,
            bundle=bundle,
            news_cache_snapshot=news_cache_snapshot,
            runtime_diagnostics=runtime_diagnostics,
        )
    return committee_view


def _apply_review_qa_risk_hold_advisory(
    *,
    committee_view: dict[str, Any],
    review_qa_output: ReviewQAOutput,
    bundle: Stage2InputBundle | None,
    news_cache_snapshot: dict[str, Any] | None,
    runtime_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    apply_reason = _review_qa_advisory_apply_reason(
        review_qa_output=review_qa_output,
        news_cache_snapshot=news_cache_snapshot,
        source_feature_row=bundle.source_feature_row if bundle else None,
    )
    if not apply_reason:
        return committee_view

    adjusted = dict(committee_view)
    adjusted["committee_decision_type"] = "boundary_hold"
    adjusted["committee_decision_type_label"] = "경계등급 보류"
    adjusted["committee_risk_signal"] = False
    adjusted["risk_hold_reason_tags"] = []
    adjusted["risk_hold_reason_labels"] = []
    adjusted["risk_hold_reason_summary"] = ""
    if apply_reason == "watch_context_only_risk_hold_override":
        adjustment_note = (
            "ReviewQAAgent는 외부 공시가 caution/watch_context 수준에 머무르고 "
            "치명 위험 근거가 확인되지 않아, 최종 라벨은 보류로 유지하되 "
            "세부유형을 경계등급 보류로 낮추도록 권고했습니다."
        )
    else:
        adjustment_note = (
            "ReviewQAAgent는 최종 라벨을 보류로 유지하되, 위험 보류 근거가 과도하다고 "
            "판단해 세부유형을 경계등급 보류로 낮추도록 권고했습니다."
        )
    adjusted["conflict_resolution"] = _append_sentence(
        str(adjusted.get("conflict_resolution") or ""),
        adjustment_note,
    )
    adjusted["final_review_memo"] = _append_sentence(
        str(adjusted.get("final_review_memo") or ""),
        "ReviewQA 보강 의견: 최종 라벨은 보류로 유지하고 위험신호 표시는 경계등급 보류로 낮춥니다.",
    )
    adjusted["mitigating_factors"] = _prepend_unique_text(
        adjusted.get("mitigating_factors"),
        "ReviewQA 위험 보류 세부유형 보정 권고",
    )
    adjusted["decision_trace"] = [
        *list(adjusted.get("decision_trace") or []),
        {
            "gate": "review_qa_subtype_adjustment",
            "label": "ReviewQA 세부유형 보정",
            "triggered": True,
            "severity": "mitigation",
            "summary": adjustment_note,
        },
    ]
    runtime_diagnostics["review_qa_advisory_applied"] = True
    runtime_diagnostics["review_qa_adjusted_decision_type"] = "boundary_hold"
    runtime_diagnostics["review_qa_advisory_apply_reason"] = apply_reason
    return adjusted


def _apply_review_qa_reject_advisory(
    *,
    committee_view: dict[str, Any],
    review_qa_output: ReviewQAOutput,
    bundle: Stage2InputBundle | None,
    news_cache_snapshot: dict[str, Any] | None,
    runtime_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    apply_reason = _review_qa_reject_advisory_apply_reason(
        review_qa_output=review_qa_output,
        bundle=bundle,
        news_cache_snapshot=news_cache_snapshot,
    )
    if not apply_reason:
        return committee_view

    adjusted = dict(committee_view)
    adjusted["final_committee_label"] = "보류"
    adjusted["committee_decision_type"] = "boundary_hold"
    adjusted["committee_decision_type_label"] = "경계등급 보류"
    adjusted["committee_risk_signal"] = False
    adjusted["risk_hold_reason_tags"] = []
    adjusted["risk_hold_reason_labels"] = []
    adjusted["risk_hold_reason_summary"] = ""
    adjustment_note = (
        "ReviewQAAgent는 부적격 확정 근거가 모델 고확률과 재무 약점에 치우쳐 있고, "
        "외부 공시는 routine/caution/watch-context 수준에 머문다고 보아 "
        "부적격 확정보다는 경계등급 보류로 재검수하도록 권고했습니다."
    )
    adjusted["conflict_resolution"] = _append_sentence(
        str(adjusted.get("conflict_resolution") or ""),
        adjustment_note,
    )
    adjusted["final_review_memo"] = _append_sentence(
        str(adjusted.get("final_review_memo") or ""),
        "ReviewQA 보강 의견: 치명 외부근거가 약한 부적격 확정은 경계등급 보류로 낮춰 재검수합니다.",
    )
    adjusted["mitigating_factors"] = _prepend_unique_text(
        adjusted.get("mitigating_factors"),
        "ReviewQA 부적격 확정 완화 권고",
    )
    adjusted["decision_trace"] = [
        *list(adjusted.get("decision_trace") or []),
        {
            "gate": "review_qa_reject_adjustment",
            "label": "ReviewQA 부적격 확정 보정",
            "triggered": True,
            "severity": "mitigation",
            "summary": adjustment_note,
        },
    ]
    runtime_diagnostics["review_qa_advisory_applied"] = True
    runtime_diagnostics["review_qa_adjusted_decision_type"] = "boundary_hold"
    runtime_diagnostics["review_qa_advisory_apply_reason"] = apply_reason
    return adjusted


def _review_qa_advisory_apply_reason(
    *,
    review_qa_output: ReviewQAOutput,
    news_cache_snapshot: dict[str, Any] | None,
    source_feature_row: dict[str, Any] | None = None,
) -> str:
    policy = load_stage2_policy()
    if (
        review_qa_output.risk_hold_assessment == "overstated"
        and review_qa_output.confidence
        >= policy.float("review_qa", "advisory", "overstated_risk_hold_min_confidence")
    ):
        return "review_qa_overstated_risk_hold"

    trigger_reasons = {str(reason) for reason in review_qa_output.trigger_reasons}
    if "risk_hold_without_critical_evidence" not in trigger_reasons:
        return ""
    if review_qa_output.confidence < policy.float(
        "review_qa",
        "advisory",
        "risk_hold_min_confidence",
    ):
        return ""
    if _external_evidence_is_watch_context_only(
        news_cache_snapshot or {},
        source_feature_row=source_feature_row,
    ):
        return "watch_context_only_risk_hold_override"
    return ""


def _review_qa_reject_advisory_apply_reason(
    *,
    review_qa_output: ReviewQAOutput,
    bundle: Stage2InputBundle | None,
    news_cache_snapshot: dict[str, Any] | None,
) -> str:
    policy = load_stage2_policy()
    if review_qa_output.confidence < policy.float(
        "review_qa",
        "advisory",
        "reject_min_confidence",
    ):
        return ""
    if review_qa_output.recommended_action in {"request_manual_review", "memo_only_fix"}:
        return ""
    trigger_reasons = {str(reason) for reason in review_qa_output.trigger_reasons}
    if (
        review_qa_output.recommended_action != "downgrade_reject_to_boundary_hold"
        and "reject_without_critical_evidence" not in trigger_reasons
    ):
        return ""

    news_cache = news_cache_snapshot or {}
    profile = _external_evidence_profile(
        news_cache,
        source_feature_row=bundle.source_feature_row if bundle else None,
    )
    if profile["strength"] in {"strong", "critical"}:
        return ""
    if profile["veto_candidate_count"] > 0 or profile["high_confidence_critical_count"] > 0:
        return ""
    if _has_substantive_external_risk(
        news_cache,
        source_feature_row=bundle.source_feature_row if bundle else None,
    ):
        return ""
    if not _external_evidence_is_watch_context_only(
        news_cache,
        source_feature_row=bundle.source_feature_row if bundle else None,
    ):
        return ""
    if not _has_review_qa_reject_boundary_defense(bundle):
        return ""
    if review_qa_output.recommended_action == "downgrade_reject_to_boundary_hold":
        return "review_qa_reject_watch_context_only_override"
    return "review_qa_reject_defensive_boundary_override"


def _has_review_qa_reject_boundary_defense(bundle: Stage2InputBundle | None) -> bool:
    """Require balance-sheet defense before lowering a high-probability reject."""
    if bundle is None:
        return False
    row = bundle.source_feature_row
    if _has_review_qa_extreme_financial_distress(row):
        return False
    policy = load_stage2_policy()
    section = ("review_qa", "reject_boundary_defense")

    axes = [
        _metric_at_least_value(row, "current_ratio", policy.float(*section, "current_ratio_floor"))
        and _metric_at_least_value(row, "cash_ratio", policy.float(*section, "cash_ratio_floor")),
        _metric_at_least_value(row, "equity_ratio", policy.float(*section, "equity_ratio_floor"))
        and _metric_at_most_value(row, "debt_ratio", policy.float(*section, "debt_ratio_ceiling"))
        and _metric_at_most_value(
            row,
            "capital_impairment_ratio",
            policy.float(*section, "capital_impairment_ratio_ceiling"),
        ),
        _metric_at_most_value(
            row,
            "total_borrowings_ratio",
            policy.float(*section, "total_borrowings_ratio_ceiling"),
        )
        or _metric_at_most_value(
            row,
            "short_term_borrowings_share",
            policy.float(*section, "short_term_borrowings_share_ceiling"),
        ),
        not _truthy(row.get("is_2y_consecutive_operating_loss"))
        and not _truthy(row.get("is_2y_consecutive_ocf_deficit")),
        _metric_at_least_value(
            row,
            "cashflow_coverage_ratio",
            policy.float(*section, "cashflow_coverage_ratio_floor"),
        )
        or _metric_at_least_value(
            row,
            "ocf_to_total_liabilities",
            policy.float(*section, "ocf_to_total_liabilities_floor"),
        )
        or _metric_at_least_value(row, "ocf_to_sales", policy.float(*section, "ocf_to_sales_floor")),
    ]
    return sum(1 for passed in axes if passed) >= policy.int(
        *section,
        "min_defensive_axes",
    )


def _has_review_qa_extreme_financial_distress(row: dict[str, Any]) -> bool:
    policy = load_stage2_policy()
    section = ("review_qa", "extreme_distress")
    if _metric_above_value(
        row,
        "capital_impairment_ratio",
        policy.float(*section, "capital_impairment_ratio_floor"),
    ):
        return True
    if _metric_below_value(row, "equity_ratio", policy.float(*section, "equity_ratio_ceiling")):
        return True
    if _metric_above_value(row, "debt_ratio", policy.float(*section, "debt_ratio_floor")):
        return True

    short_term_maturity_wall = _metric_at_least_value(
        row,
        "short_term_borrowings_share",
        policy.float(*section, "short_term_borrowings_share_floor"),
    )
    weak_cashflow = (
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
    )
    recurring_loss_or_ocf_deficit = _truthy(row.get("is_2y_consecutive_operating_loss")) or _truthy(
        row.get("is_2y_consecutive_ocf_deficit")
    )
    interest_blocked = _truthy(row.get("icr_under_1")) or _metric_below_value(
        row,
        "interest_coverage_ratio",
        policy.float(*section, "interest_coverage_ratio_ceiling"),
    )
    return bool(
        short_term_maturity_wall
        and weak_cashflow
        and recurring_loss_or_ocf_deficit
        and interest_blocked
    )


def _external_evidence_is_watch_context_only(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    raw_items = news_cache.get("items", [])
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    if not items:
        return False

    profile = _external_evidence_profile(
        news_cache,
        source_feature_row=source_feature_row,
    )
    if profile["veto_candidate_count"] > 0 or profile["high_confidence_critical_count"] > 0:
        return False

    watch_context_count = 0
    for item in items:
        if _is_substantive_external_risk_item(
            item,
            source_feature_row=source_feature_row,
        ):
            return False
        if _is_watch_context_external_item(item):
            watch_context_count += 1
    return watch_context_count > 0


def _is_watch_context_external_item(item: dict[str, Any]) -> bool:
    severity = str(item.get("disclosure_severity", "")).lower()
    event_class = str(item.get("disclosure_event_class", "")).lower()
    materiality = str(item.get("disclosure_materiality", "")).lower()
    provider_relevance = str(item.get("provider_relevance", "")).lower()
    return (
        severity in {"routine", "caution"}
        or event_class in {"routine_context", "watch_context", "procedural_or_one_off"}
        or materiality in {"routine_context", "watch_context", "procedural_or_one_off"}
        or provider_relevance in {"routine", "caution", "context"}
    )


def _review_qa_trigger_reasons(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    final_label = str(committee_view.get("final_committee_label") or "")
    decision_type = str(committee_view.get("committee_decision_type") or "")
    evidence_profile = _external_evidence_profile(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )

    has_critical_evidence = (
        evidence_profile["strength"] in {"strong", "critical"}
        or evidence_profile["veto_candidate_count"] > 0
        or evidence_profile["high_confidence_critical_count"] > 0
    )
    watch_context_only = _external_evidence_is_watch_context_only(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    disagreement_level = _agent_disagreement_level_from_committee_view(committee_view)
    disagreement_reasons = _agent_disagreement_reasons_from_committee_view(committee_view)
    high_disagreement = disagreement_level == "high"
    memo_conflict_candidate = _review_qa_memo_conflict_possible(committee_view)
    risk_hold_boundary_defense = _has_review_qa_risk_hold_boundary_defense(bundle)
    risk_hold_from_investment_model = bundle.prediction_label == "투자적격"
    risk_hold_review_candidate = final_label == "보류" and decision_type == "risk_hold"
    risk_hold_actionable_candidate = memo_conflict_candidate or (
        risk_hold_from_investment_model and (watch_context_only or risk_hold_boundary_defense)
    )
    if risk_hold_review_candidate:
        if high_disagreement and not has_critical_evidence and risk_hold_actionable_candidate:
            reasons.append("agent_disagreement_high_without_critical_evidence")
        if (
            not has_critical_evidence
            and risk_hold_actionable_candidate
            and _review_qa_disagreement_allows_risk_hold(
                disagreement_level=disagreement_level,
                disagreement_reasons=disagreement_reasons,
            )
        ):
            reasons.append("risk_hold_without_critical_evidence")
            if memo_conflict_candidate:
                reasons.append("label_memo_conflict_candidate")
    if (
        final_label == "부적격"
        and decision_type == "reject"
        and not has_critical_evidence
        and high_disagreement
    ):
        reasons.append("agent_disagreement_high_without_critical_evidence")
    if (
        final_label == "부적격"
        and decision_type == "reject"
        and not has_critical_evidence
        and watch_context_only
        and _has_review_qa_reject_boundary_defense(bundle)
        and _review_qa_disagreement_allows_reject(
            disagreement_level=disagreement_level,
            disagreement_reasons=disagreement_reasons,
        )
    ):
        reasons.append("reject_without_critical_evidence")
    return reasons[:5]


def _agent_disagreement_level_from_committee_view(committee_view: dict[str, Any]) -> str:
    level = str(committee_view.get("agent_disagreement_level") or "").strip().lower()
    if level in {"low", "medium", "high"}:
        return level
    policy = load_stage2_policy()
    score = _safe_float(committee_view.get("agent_disagreement_score")) or 0.0
    if score >= policy.float("review_qa", "disagreement", "high_score_floor"):
        return "high"
    if score >= policy.float("review_qa", "disagreement", "medium_score_floor"):
        return "medium"
    return "low"


def _agent_disagreement_reasons_from_committee_view(
    committee_view: dict[str, Any],
) -> set[str]:
    raw_reasons = committee_view.get("agent_disagreement_reasons")
    if not isinstance(raw_reasons, list | tuple | set):
        return set()
    return {str(reason).strip() for reason in raw_reasons if str(reason).strip()}


def _review_qa_disagreement_allows_risk_hold(
    *,
    disagreement_level: str,
    disagreement_reasons: set[str],
) -> bool:
    if disagreement_level == "high":
        return True
    if disagreement_level != "medium":
        return False
    return bool(
        disagreement_reasons.intersection(
            {
                "chair_risk_without_critical_evidence",
                "committee_label_memo_conflict",
            }
        )
    )


def _review_qa_disagreement_allows_reject(
    *,
    disagreement_level: str,
    disagreement_reasons: set[str],
) -> bool:
    if disagreement_level == "high":
        return True
    if disagreement_level != "medium":
        return False
    return bool(
        disagreement_reasons.intersection(
            {
                "chair_reject_without_critical_evidence",
                "committee_label_memo_conflict",
            }
        )
    )


def _has_review_qa_risk_hold_boundary_defense(bundle: Stage2InputBundle) -> bool:
    """Return whether a risk hold has enough defense to justify optional QA."""
    row = bundle.source_feature_row
    if not row or _has_review_qa_extreme_financial_distress(row):
        return False
    policy = load_stage2_policy()
    section = ("review_qa", "boundary_defense")

    defensive_axes = [
        _metric_at_least_value(row, "current_ratio", policy.float(*section, "current_ratio_floor"))
        or _metric_at_least_value(row, "cash_ratio", policy.float(*section, "cash_ratio_floor")),
        _metric_at_least_value(
            row,
            "cashflow_coverage_ratio",
            policy.float(*section, "cashflow_coverage_ratio_floor"),
        )
        or _metric_at_least_value(
            row,
            "ocf_to_total_liabilities",
            policy.float(*section, "ocf_to_total_liabilities_floor"),
        )
        or _metric_at_least_value(row, "ocf_to_sales", policy.float(*section, "ocf_to_sales_floor")),
        _metric_at_least_value(
            row,
            "interest_coverage_ratio",
            policy.float(*section, "interest_coverage_ratio_floor"),
        )
        and not _truthy(row.get("icr_under_1")),
        _metric_at_least_value(row, "equity_ratio", policy.float(*section, "equity_ratio_floor"))
        and _metric_at_most_value(row, "debt_ratio", policy.float(*section, "debt_ratio_ceiling"))
        and _metric_at_most_value(
            row,
            "capital_impairment_ratio",
            policy.float(*section, "capital_impairment_ratio_ceiling"),
        ),
        _metric_at_most_value(
            row,
            "total_borrowings_ratio",
            policy.float(*section, "total_borrowings_ratio_ceiling"),
        )
        or _metric_at_most_value(
            row,
            "short_term_borrowings_share",
            policy.float(*section, "short_term_borrowings_share_ceiling"),
        ),
        not _truthy(row.get("is_2y_consecutive_operating_loss"))
        and not _truthy(row.get("is_2y_consecutive_ocf_deficit")),
    ]
    return sum(1 for passed in defensive_axes if passed) >= policy.int(
        *section,
        "min_defensive_axes",
    )


def _review_qa_memo_conflict_possible(committee_view: dict[str, Any]) -> bool:
    if str(committee_view.get("final_committee_label") or "") == "적격":
        return False
    memo = str(committee_view.get("final_review_memo") or "")
    conflict_markers = (
        "투자적격 판단을 유지",
        "투자적격 라벨을 유지",
        "투자적격 유지",
        "모델 라벨을 유지",
        "모델 라벨 유지",
        "모델 라벨을 존중",
        "모델 라벨 존중",
        "최종 라벨은 투자적격",
        "최종 라벨을 투자적격",
    )
    return any(marker in memo for marker in conflict_markers)


def _stage2_review_qa_enabled(runtime_backend_name: str) -> bool:
    config = _stage2_runtime_config()
    if config.review_qa_enabled is not None:
        return bool(config.review_qa_enabled)
    backend = runtime_backend_name.strip().lower()
    return backend.startswith("agno") and "fallback" not in backend


def _stage2_review_qa_apply_advisory() -> bool:
    return bool(_stage2_runtime_config().review_qa_apply_advisory)


def _stage2_review_qa_fallback_on_error() -> bool:
    return bool(_stage2_runtime_config().review_qa_fallback_on_error)


__all__ = [
    "_agent_disagreement_level_from_committee_view",
    "_agent_disagreement_reasons_from_committee_view",
    "_apply_review_qa_advisory",
    "_apply_review_qa_reject_advisory",
    "_apply_review_qa_risk_hold_advisory",
    "_external_evidence_is_watch_context_only",
    "_has_review_qa_extreme_financial_distress",
    "_has_review_qa_reject_boundary_defense",
    "_has_review_qa_risk_hold_boundary_defense",
    "_is_watch_context_external_item",
    "_maybe_run_review_qa",
    "_review_qa_advisory_apply_reason",
    "_review_qa_disagreement_allows_reject",
    "_review_qa_disagreement_allows_risk_hold",
    "_review_qa_memo_conflict_possible",
    "_review_qa_reject_advisory_apply_reason",
    "_review_qa_trigger_reasons",
    "_stage2_review_qa_apply_advisory",
    "_stage2_review_qa_enabled",
    "_stage2_review_qa_fallback_on_error",
]
