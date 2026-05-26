"""Run the Stage 2 three-agent review scaffold."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, Literal, TypedDict, cast

from cas.agents.committee_view import build_committee_view
from cas.agents.nodes.committee_feature_formatting import (
    describe_top_drivers,
    humanize_category,
    humanize_industry,
    humanize_size_group,
)
from cas.agents.signals import (
    evaluate_agent_disagreement,
    evaluate_debt_liquidity,
    evaluate_evidence_treatment,
    evaluate_external_evidence,
    evaluate_macro_market,
)
from cas.agents.signals.credit_policy_signals import evaluate_credit_policy
from cas.agents.signals.materiality_signals import (
    has_substantive_external_risk as _shared_has_substantive_external_risk,
)
from cas.agents.signals.materiality_signals import (
    substantive_external_risk_item as _shared_substantive_external_risk_item,
)
from cas.agents.stage2_bundle import Stage2InputBundle, build_stage2_input_bundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    ReviewQAOutput,
    RiskRecallQAOutput,
)
from cas.agents.stage2_runner import (
    AgnoStage2AgentRunner,
    DeterministicStage2AgentRunner,
    Stage2AgentRunner,
)
from cas.agents.stage2_specs import STAGE2_AGENT_ROLES
from cas.agents.state import (
    AgentOutput,
    AgentState,
    AuditEntry,
    CommitteeReview,
    Recommendation,
)
from cas.utils.live_cache import read_json_cache, stable_cache_key, write_json_cache

_EvidenceStrength = Literal["none", "weak", "moderate", "strong", "critical"]
_REVIEW_QA_CACHE_VERSION = "stage2_review_qa_v4"
_RISK_RECALL_QA_CACHE_VERSION = "stage2_risk_recall_qa_v3"


class _EvidenceProfile(TypedDict):
    status: str
    strength: _EvidenceStrength
    finding: str
    item_count: int
    direct_count: int
    verified_count: int
    weak_count: int
    adverse_count: int
    verified_adverse_count: int
    veto_candidate_count: int
    high_confidence_critical_count: int
    critical_terms: list[str]
    score: float


def run(state: AgentState) -> dict[str, Any]:
    """Run the three-agent Stage 2 scaffold over Stage 1 outputs."""
    credit_policy_snapshot = evaluate_credit_policy(
        source_feature_row=dict(state.get("source_feature_row") or {}),
        peer_comparison_rows=list(state.get("peer_comparison_rows") or []),
    ).model_dump(mode="json")

    state_with_policy = cast(
        AgentState,
        {
            **state,
            "credit_policy_snapshot": credit_policy_snapshot,
        },
    )

    bundle = build_stage2_input_bundle(state_with_policy)
    recommendation = cast(
        Recommendation,
        bundle.rule_result.get("recommendation") or state.get("final_recommendation") or "review",
    )
    rule_confidence = round(
        float(bundle.rule_result.get("confidence", state.get("final_confidence", 0.0)) or 0.0),
        4,
    )

    # Stage 2 execution goes through a runner adapter. Today it is deterministic
    # for CI stability; later it can be swapped for an Agno-backed runner.
    runner = _stage2_runner()
    structured_outputs = runner.run(
        bundle=bundle,
        recommendation=recommendation,
        confidence=rule_confidence,
    )
    runtime_backend_name = str(getattr(runner, "last_run_backend_name", runner.backend_name))
    runtime_diagnostics = dict(getattr(runner, "last_run_diagnostics", {}) or {})
    runtime_diagnostics.setdefault("backend_name", runtime_backend_name)
    runtime_diagnostics.setdefault("cache_hit", False)
    agents = [output.to_agent_output() for output in structured_outputs]
    committee_view = build_committee_view(
        bundle=bundle,
        recommendation=recommendation,
        agents=agents,
    )
    committee_view = _attach_agent_disagreement(
        bundle=bundle,
        committee_view=committee_view,
        structured_outputs=structured_outputs,
        runtime_diagnostics=runtime_diagnostics,
    )
    review_qa_output = _maybe_run_review_qa(
        bundle=bundle,
        committee_view=committee_view,
        structured_outputs=structured_outputs,
        runtime_backend_name=runtime_backend_name,
        runtime_diagnostics=runtime_diagnostics,
    )
    if review_qa_output is not None:
        committee_view = _apply_review_qa_advisory(
            committee_view=committee_view,
            review_qa_output=review_qa_output,
            bundle=bundle,
            news_cache_snapshot=bundle.news_cache_snapshot,
            runtime_diagnostics=runtime_diagnostics,
        )
        committee_view = _attach_agent_disagreement(
            bundle=bundle,
            committee_view=committee_view,
            structured_outputs=structured_outputs,
            runtime_diagnostics=runtime_diagnostics,
        )
        agents.append(review_qa_output.to_agent_output())
    risk_recall_qa_output = _maybe_run_risk_recall_qa(
        bundle=bundle,
        committee_view=committee_view,
        structured_outputs=structured_outputs,
        runtime_backend_name=runtime_backend_name,
        runtime_diagnostics=runtime_diagnostics,
    )
    if risk_recall_qa_output is not None:
        committee_view = _apply_risk_recall_qa_advisory(
            committee_view=committee_view,
            risk_recall_qa_output=risk_recall_qa_output,
            bundle=bundle,
            runtime_diagnostics=runtime_diagnostics,
        )
        committee_view = _attach_agent_disagreement(
            bundle=bundle,
            committee_view=committee_view,
            structured_outputs=structured_outputs,
            runtime_diagnostics=runtime_diagnostics,
        )
        agents.append(risk_recall_qa_output.to_agent_output())
    _validate_agent_order(agents)
    reviews = [
        CommitteeReview(
            perspective=agent.role,
            recommendation=recommendation,
            confidence=agent.confidence,
            rationale=agent.summary,
        )
        for agent in agents
    ]
    committee_confidence = _committee_confidence(
        bundle=bundle,
        agents=agents,
        committee_view=committee_view,
        rule_confidence=rule_confidence,
        runtime_backend_name=runtime_backend_name,
    )
    # agent_summary는 대시보드/리포트에서 바로 읽기 쉬운 dict 구조이고,
    # agent_outputs / committee_reviews는 schema와 audit trail 쪽에서 쓰는 정규화 결과다.
    agent_summary = {
        "final_recommendation": recommendation,
        "final_confidence": committee_confidence,
        "synthesis": _chair_summary(agents),
        "agents": {
            agent.role: {
                "summary": agent.summary,
                "findings": agent.findings,
                "confidence": agent.confidence,
            }
            for agent in agents
        },
        "runtime": runtime_diagnostics,
    }

    audit_metrics = {
        "n_agents": float(len(agents)),
        "rule_confidence": rule_confidence,
        "final_confidence": committee_confidence,
    }
    audit_metrics.update(_runtime_diagnostic_metrics(runtime_diagnostics))
    audit = AuditEntry(
        node="agno_agents",
        timestamp=_now(),
        summary=(
            f"Stage 2 scaffold completed via {runtime_backend_name} runner: "
            f"{', '.join(agent.role for agent in agents)}"
        ),
        metrics=audit_metrics,
    )
    return {
        "agent_outputs": agents,
        "committee_reviews": reviews,
        "agent_summary": agent_summary,
        "committee_view": committee_view,
        "stage2_runtime_diagnostics": runtime_diagnostics,
        "credit_policy_snapshot": credit_policy_snapshot,
        "final_recommendation": recommendation,
        "final_confidence": committee_confidence,
        "audit": [audit],
    }


def _runtime_diagnostic_metrics(diagnostics: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source_key, metric_key in (
        ("stage2_total_elapsed_seconds", "stage2_total_elapsed_seconds"),
        ("agent_elapsed_seconds_sum", "stage2_agent_elapsed_seconds_sum"),
    ):
        value = _safe_float(diagnostics.get(source_key))
        if value is not None:
            metrics[metric_key] = value
    agent_timings = diagnostics.get("agent_elapsed_seconds")
    if isinstance(agent_timings, dict):
        for role in (
            "quant_credit",
            "evidence_audit",
            "chair_report",
            "review_qa",
            "risk_recall_qa",
            "llm_client",
        ):
            value = _safe_float(agent_timings.get(role))
            if value is not None:
                metrics[f"stage2_{role}_elapsed_seconds"] = value
    metrics["stage2_llm_cache_hit"] = 1.0 if diagnostics.get("cache_hit") is True else 0.0
    metrics["stage2_review_qa_triggered"] = (
        1.0 if diagnostics.get("review_qa_triggered") is True else 0.0
    )
    metrics["stage2_review_qa_cache_hit"] = (
        1.0 if diagnostics.get("review_qa_cache_hit") is True else 0.0
    )
    metrics["stage2_review_qa_advisory_applied"] = (
        1.0 if diagnostics.get("review_qa_advisory_applied") is True else 0.0
    )
    metrics["stage2_risk_recall_qa_triggered"] = (
        1.0 if diagnostics.get("risk_recall_qa_triggered") is True else 0.0
    )
    metrics["stage2_risk_recall_qa_cache_hit"] = (
        1.0 if diagnostics.get("risk_recall_qa_cache_hit") is True else 0.0
    )
    metrics["stage2_risk_recall_qa_advisory_applied"] = (
        1.0 if diagnostics.get("risk_recall_qa_advisory_applied") is True else 0.0
    )
    return metrics


def _validate_agent_order(agents: list[AgentOutput]) -> None:
    actual_roles = tuple(agent.role for agent in agents)
    required_count = len(STAGE2_AGENT_ROLES)
    optional_suffix = actual_roles[required_count:]
    allowed_suffixes = {
        (),
        ("review_qa",),
        ("risk_recall_qa",),
        ("review_qa", "risk_recall_qa"),
    }
    if (
        actual_roles[:required_count] != STAGE2_AGENT_ROLES
        or optional_suffix not in allowed_suffixes
    ):
        expected = ", ".join(STAGE2_AGENT_ROLES)
        actual = ", ".join(actual_roles)
        raise ValueError(
            "Stage 2 agent order mismatch: "
            f"expected {expected} with optional QA agents at the end, got {actual}"
        )


def _chair_summary(agents: list[AgentOutput]) -> str:
    for agent in agents:
        if agent.role == "chair_report":
            return str(agent.summary)
    return str(agents[-1].summary) if agents else ""


def _attach_agent_disagreement(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    structured_outputs: tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput],
    runtime_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    disagreement = evaluate_agent_disagreement(
        bundle=bundle,
        committee_view=committee_view,
        quant_credit=structured_outputs[0],
        evidence_audit=structured_outputs[1],
        chair_report=structured_outputs[2],
    )
    payload = disagreement.as_payload()
    runtime_diagnostics.update(payload)
    return {**committee_view, **payload}


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
    if (
        review_qa_output.risk_hold_assessment == "overstated"
        and review_qa_output.confidence >= 0.55
    ):
        return "review_qa_overstated_risk_hold"

    trigger_reasons = {str(reason) for reason in review_qa_output.trigger_reasons}
    if "risk_hold_without_critical_evidence" not in trigger_reasons:
        return ""
    if review_qa_output.confidence < 0.45:
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
    if review_qa_output.confidence < 0.45:
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

    axes = [
        _metric_at_least_value(row, "current_ratio", 1.2)
        and _metric_at_least_value(row, "cash_ratio", 0.15),
        _metric_at_least_value(row, "equity_ratio", 0.40)
        and _metric_at_most_value(row, "debt_ratio", 1.50)
        and _metric_at_most_value(row, "capital_impairment_ratio", 0.0),
        _metric_at_most_value(row, "total_borrowings_ratio", 0.50)
        or _metric_at_most_value(row, "short_term_borrowings_share", 0.70),
        not _truthy(row.get("is_2y_consecutive_operating_loss"))
        and not _truthy(row.get("is_2y_consecutive_ocf_deficit")),
        _metric_at_least_value(row, "cashflow_coverage_ratio", 0.0)
        or _metric_at_least_value(row, "ocf_to_total_liabilities", 0.0)
        or _metric_at_least_value(row, "ocf_to_sales", 0.0),
    ]
    return sum(1 for passed in axes if passed) >= 3


def _has_review_qa_extreme_financial_distress(row: dict[str, Any]) -> bool:
    if _metric_above_value(row, "capital_impairment_ratio", 0.50):
        return True
    if _metric_below_value(row, "equity_ratio", 0.15):
        return True
    if _metric_above_value(row, "debt_ratio", 5.0):
        return True

    short_term_maturity_wall = _metric_at_least_value(row, "short_term_borrowings_share", 0.95)
    weak_cashflow = (
        _metric_below_value(row, "cashflow_coverage_ratio", 0.0)
        or _metric_below_value(row, "ocf_to_total_liabilities", 0.0)
        or _metric_below_value(row, "ocf_to_sales", 0.0)
    )
    recurring_loss_or_ocf_deficit = _truthy(row.get("is_2y_consecutive_operating_loss")) or _truthy(
        row.get("is_2y_consecutive_ocf_deficit")
    )
    interest_blocked = _truthy(row.get("icr_under_1")) or _metric_below_value(
        row,
        "interest_coverage_ratio",
        1.0,
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


def _is_substantive_external_risk_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    return bool(
        _shared_substantive_external_risk_item(
            item,
            source_feature_row=source_feature_row,
        )
    )


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


def _append_sentence(base: str, sentence: str) -> str:
    cleaned = base.strip()
    if not cleaned:
        return sentence
    if sentence in cleaned:
        return cleaned
    if cleaned.endswith((".", "!", "?")):
        return f"{cleaned} {sentence}"
    return f"{cleaned}. {sentence}"


def _prepend_unique_text(raw_items: object, item: str) -> list[str]:
    if isinstance(raw_items, list | tuple | set):
        existing = [str(value) for value in raw_items if str(value)]
    else:
        existing = []
    return [item, *[value for value in existing if value != item]]


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
    risk_hold_actionable_candidate = (
        memo_conflict_candidate
        or (
            risk_hold_from_investment_model
            and (watch_context_only or risk_hold_boundary_defense)
        )
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
    score = _safe_float(committee_view.get("agent_disagreement_score")) or 0.0
    if score >= 0.55:
        return "high"
    if score >= 0.25:
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

    defensive_axes = [
        _metric_at_least_value(row, "current_ratio", 1.2)
        or _metric_at_least_value(row, "cash_ratio", 0.15),
        _metric_at_least_value(row, "cashflow_coverage_ratio", 1.0)
        or _metric_at_least_value(row, "ocf_to_total_liabilities", 0.05)
        or _metric_at_least_value(row, "ocf_to_sales", 0.0),
        _metric_at_least_value(row, "interest_coverage_ratio", 1.0)
        and not _truthy(row.get("icr_under_1")),
        _metric_at_least_value(row, "equity_ratio", 0.40)
        and _metric_at_most_value(row, "debt_ratio", 1.50)
        and _metric_at_most_value(row, "capital_impairment_ratio", 0.0),
        _metric_at_most_value(row, "total_borrowings_ratio", 0.50)
        or _metric_at_most_value(row, "short_term_borrowings_share", 0.70),
        not _truthy(row.get("is_2y_consecutive_operating_loss"))
        and not _truthy(row.get("is_2y_consecutive_ocf_deficit")),
    ]
    return sum(1 for passed in defensive_axes if passed) >= 3


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
    value = os.environ.get("CAS_STAGE2_REVIEW_QA_ENABLED")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off"}
    backend = runtime_backend_name.strip().lower()
    return backend.startswith("agno") and "fallback" not in backend


def _stage2_review_qa_apply_advisory() -> bool:
    value = os.environ.get("CAS_STAGE2_REVIEW_QA_APPLY_ADVISORY", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _stage2_review_qa_fallback_on_error() -> bool:
    value = os.environ.get("CAS_STAGE2_REVIEW_QA_FALLBACK_ON_ERROR", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _run_review_qa_agent_with_cache(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
) -> tuple[ReviewQAOutput, dict[str, Any]]:
    started_at = time.perf_counter()
    model_provider = _stage2_review_qa_provider()
    model_name = _stage2_review_qa_model()
    cache_payload = _review_qa_cache_payload(
        bundle=bundle,
        committee_view=committee_view,
        quant_credit=quant_credit,
        evidence_audit=evidence_audit,
        chair_report=chair_report,
        trigger_reasons=trigger_reasons,
        model_provider=model_provider,
        model_name=model_name,
    )
    cache_key = stable_cache_key(cache_payload)
    cached_payload = read_json_cache(
        "llm_stage2_review_qa",
        cache_key,
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
    )
    if cached_payload is not None:
        response_payload = cached_payload.get("response", cached_payload)
        review_qa = ReviewQAOutput.model_validate(response_payload)
        return review_qa, {
            "review_qa_cache_hit": True,
            "review_qa_cache_key": cache_key,
            "agent_elapsed_seconds": {"review_qa": round(time.perf_counter() - started_at, 4)},
        }

    review_module = import_module("cas.agents.nodes.tripletagents.review_qa_agent")
    review_qa = cast(
        ReviewQAOutput,
        review_module.run_review_qa_agent(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=model_provider,
            model_name=model_name,
            max_tokens=_stage2_review_qa_max_tokens(),
        ),
    )
    write_json_cache(
        "llm_stage2_review_qa",
        cache_key,
        {
            "cache_version": _REVIEW_QA_CACHE_VERSION,
            "response": review_qa.model_dump(mode="json"),
        },
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
    )
    return review_qa, {
        "review_qa_cache_hit": False,
        "review_qa_cache_key": cache_key,
        "agent_elapsed_seconds": {"review_qa": round(time.perf_counter() - started_at, 4)},
    }


def _review_qa_cache_payload(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
    model_provider: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "cache_version": _REVIEW_QA_CACHE_VERSION,
        "model_provider": model_provider,
        "model_name": model_name,
        "stage2_input_bundle": bundle.to_compact_prompt_payload(role="review_qa"),
        "committee_view": committee_view,
        "agent_outputs": {
            "quant_credit": quant_credit.model_dump(mode="json"),
            "evidence_audit": evidence_audit.model_dump(mode="json"),
            "chair_report": chair_report.model_dump(mode="json"),
        },
        "trigger_reasons": trigger_reasons,
    }


def _merge_review_qa_diagnostics(
    runtime_diagnostics: dict[str, Any],
    review_qa_diagnostics: dict[str, Any],
) -> None:
    existing_timings = runtime_diagnostics.get("agent_elapsed_seconds")
    if not isinstance(existing_timings, dict):
        existing_timings = {}
    qa_timings = review_qa_diagnostics.get("agent_elapsed_seconds")
    qa_elapsed_seconds = 0.0
    if isinstance(qa_timings, dict):
        existing_timings.update(qa_timings)
        qa_elapsed = _safe_float(qa_timings.get("review_qa"))
        if qa_elapsed is not None:
            qa_elapsed_seconds = qa_elapsed
    runtime_diagnostics["agent_elapsed_seconds"] = existing_timings
    runtime_diagnostics["agent_elapsed_seconds_sum"] = round(
        sum(float(value) for value in existing_timings.values()),
        4,
    )
    current_total = _safe_float(runtime_diagnostics.get("stage2_total_elapsed_seconds"))
    if current_total is not None and qa_elapsed_seconds:
        runtime_diagnostics["stage2_total_elapsed_seconds"] = round(
            current_total + qa_elapsed_seconds,
            4,
        )
    runtime_diagnostics["review_qa_cache_hit"] = bool(
        review_qa_diagnostics.get("review_qa_cache_hit", False)
    )
    if review_qa_diagnostics.get("review_qa_cache_key"):
        runtime_diagnostics["review_qa_cache_key"] = review_qa_diagnostics["review_qa_cache_key"]


def _stage2_review_qa_provider() -> str:
    return (
        os.environ.get("CAS_STAGE2_REVIEW_QA_PROVIDER")
        or os.environ.get("CAS_STAGE2_CHAIR_PROVIDER")
        or os.environ.get("CAS_STAGE2_MODEL_PROVIDER")
        or "openai"
    )


def _stage2_review_qa_model() -> str:
    return (
        os.environ.get("CAS_STAGE2_REVIEW_QA_MODEL")
        or os.environ.get("CAS_STAGE2_CHAIR_MODEL")
        or os.environ.get("CAS_STAGE2_MODEL")
        or "gpt-4.1-mini"
    )


def _stage2_review_qa_max_tokens() -> int:
    try:
        return int(os.environ.get("CAS_STAGE2_REVIEW_QA_MAX_TOKENS", "3000"))
    except ValueError:
        return 3000


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
    action = risk_recall_qa_output.recommended_action
    assessment = risk_recall_qa_output.eligible_safety_assessment
    trigger_reasons = {str(reason) for reason in risk_recall_qa_output.trigger_reasons}
    weak_axes = _risk_recall_weak_financial_axes(bundle)
    confirmed_external_evidence = _risk_recall_confirmed_external_escalation_evidence(bundle)
    if action == "escalate_eligible_to_risk_hold":
        if assessment != "material_missed_risk" or risk_recall_qa_output.confidence < 0.70:
            return ""
        if confirmed_external_evidence:
            return "risk_recall_substantive_external_risk"
        if len(weak_axes) >= 4:
            return "risk_recall_severe_financial_weakness"
        return ""

    if action != "escalate_eligible_to_boundary_hold":
        return ""
    if assessment not in {"needs_boundary_review", "material_missed_risk"}:
        return ""
    if risk_recall_qa_output.confidence < 0.60:
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
        and len(weak_axes) >= 2
    )
    multi_axis_weak = "eligible_with_multiple_weak_financial_axes" in trigger_reasons and len(
        weak_axes
    ) >= 3
    boundary_weak = "eligible_boundary_rating_context" in trigger_reasons and len(weak_axes) >= 2
    if not (
        confirmed_external_evidence
        or near_threshold_weak
        or multi_axis_weak
        or boundary_weak
    ):
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
    if item.get("veto_candidate") is True or item.get("critical_context_confirmed") is True:
        return True

    source = str(item.get("source") or "").lower()
    quality = str(item.get("evidence_quality") or "").lower()
    score = _safe_float(item.get("evidence_score"))
    if source == "opendart":
        return score is None or score >= 0.55
    if quality == "low":
        return False
    if quality in {"medium", "high"} and (score is None or score >= 0.55):
        return True
    return score is not None and score >= 0.65


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
    near_threshold = _risk_recall_near_threshold(bundle)
    weak_axes = _risk_recall_weak_financial_axes(bundle)
    has_watch_evidence = _has_risk_recall_watch_evidence(bundle.news_cache_snapshot)
    has_substantive_evidence = _has_substantive_external_risk(
        bundle.news_cache_snapshot,
        source_feature_row=bundle.source_feature_row,
    )
    has_boundary_context = _has_rating_boundary_context(bundle)

    if near_threshold and (len(weak_axes) >= 2 or has_substantive_evidence):
        reasons.append("eligible_near_threshold")
    if near_threshold and len(weak_axes) >= 2:
        reasons.append("eligible_near_threshold_with_weak_financials")
    if len(weak_axes) >= 3:
        reasons.append("eligible_with_multiple_weak_financial_axes")
    if has_watch_evidence and (
        (near_threshold and len(weak_axes) >= 2) or len(weak_axes) >= 3 or has_substantive_evidence
    ):
        reasons.append("eligible_with_recall_watch_evidence")
    if has_substantive_evidence:
        reasons.append("eligible_with_substantive_evidence")
    if has_boundary_context and (
        (near_threshold and len(weak_axes) >= 2) or len(weak_axes) >= 3 or has_substantive_evidence
    ):
        reasons.append("eligible_boundary_rating_context")
    return reasons[:5]


def _risk_recall_near_threshold(bundle: Stage2InputBundle) -> bool:
    probability = _safe_float(bundle.probability_speculative)
    threshold = _safe_float(bundle.threshold)
    if threshold is None or threshold <= 0:
        return False
    margin = threshold - (probability or 0.0)
    return 0.0 <= margin <= 0.10


def _risk_recall_weak_financial_axes(bundle: Stage2InputBundle) -> list[str]:
    row = bundle.source_feature_row
    axes: list[str] = []
    if _metric_below_value(row, "current_ratio", 1.0):
        axes.append("low_current_ratio")
    if _metric_below_value(row, "cash_ratio", 0.10):
        axes.append("low_cash_ratio")
    if (
        _metric_below_value(row, "cashflow_coverage_ratio", 0.0)
        or _metric_below_value(row, "ocf_to_total_liabilities", 0.0)
        or _metric_below_value(row, "ocf_to_sales", 0.0)
        or _truthy(row.get("is_2y_consecutive_ocf_deficit"))
    ):
        axes.append("weak_cashflow")
    if _metric_below_value(row, "interest_coverage_ratio", 1.0) or _truthy(row.get("icr_under_1")):
        axes.append("weak_interest_coverage")
    if _metric_above_value(row, "debt_ratio", 2.0):
        axes.append("high_debt_ratio")
    if _metric_above_value(row, "total_borrowings_ratio", 0.65) or _metric_above_value(
        row,
        "short_term_borrowings_share",
        0.90,
    ):
        axes.append("high_borrowing_pressure")
    return axes


def _has_risk_recall_watch_evidence(news_cache: dict[str, Any]) -> bool:
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
        if score < 0.45:
            continue
        text = " ".join(str(item.get(key) or "") for key in ("title", "summary"))
        if any(marker in text for marker in markers):
            return True
    return False


def _has_substantive_external_risk(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    return bool(
        _shared_has_substantive_external_risk(
            news_cache,
            source_feature_row=source_feature_row,
        )
    )


def _has_rating_boundary_context(bundle: Stage2InputBundle) -> bool:
    prior = bundle.prior_rating_reference
    group = str(prior.get("prior_rating_boundary_group") or "").lower()
    rating = str(prior.get("prior_credit_rating") or prior.get("credit_rating") or "").upper()
    if "boundary" in group:
        return True
    return rating in {"BBB-", "BB+"}


def _metric_below_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value < threshold


def _metric_at_least_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value >= threshold


def _metric_above_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value > threshold


def _metric_at_most_value(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _safe_float(row.get(key))
    return value is not None and value <= threshold


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _stage2_risk_recall_qa_enabled(runtime_backend_name: str) -> bool:
    value = os.environ.get("CAS_STAGE2_RISK_RECALL_QA_ENABLED")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off"}
    backend = runtime_backend_name.strip().lower()
    return backend.startswith("agno") and "fallback" not in backend


def _stage2_risk_recall_qa_apply_advisory() -> bool:
    value = os.environ.get("CAS_STAGE2_RISK_RECALL_QA_APPLY_ADVISORY", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _stage2_risk_recall_qa_fallback_on_error() -> bool:
    value = os.environ.get("CAS_STAGE2_RISK_RECALL_QA_FALLBACK_ON_ERROR", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _run_risk_recall_qa_agent_with_cache(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
) -> tuple[RiskRecallQAOutput, dict[str, Any]]:
    started_at = time.perf_counter()
    model_provider = _stage2_risk_recall_qa_provider()
    model_name = _stage2_risk_recall_qa_model()
    cache_payload = _risk_recall_qa_cache_payload(
        bundle=bundle,
        committee_view=committee_view,
        quant_credit=quant_credit,
        evidence_audit=evidence_audit,
        chair_report=chair_report,
        trigger_reasons=trigger_reasons,
        model_provider=model_provider,
        model_name=model_name,
    )
    cache_key = stable_cache_key(cache_payload)
    cached_payload = read_json_cache(
        "llm_stage2_risk_recall_qa",
        cache_key,
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
    )
    if cached_payload is not None:
        response_payload = cached_payload.get("response", cached_payload)
        risk_recall_qa = RiskRecallQAOutput.model_validate(response_payload)
        return risk_recall_qa, {
            "risk_recall_qa_cache_hit": True,
            "risk_recall_qa_cache_key": cache_key,
            "agent_elapsed_seconds": {"risk_recall_qa": round(time.perf_counter() - started_at, 4)},
        }

    review_module = import_module("cas.agents.nodes.tripletagents.risk_recall_qa_agent")
    risk_recall_qa = cast(
        RiskRecallQAOutput,
        review_module.run_risk_recall_qa_agent(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=model_provider,
            model_name=model_name,
            max_tokens=_stage2_risk_recall_qa_max_tokens(),
        ),
    )
    write_json_cache(
        "llm_stage2_risk_recall_qa",
        cache_key,
        {
            "cache_version": _RISK_RECALL_QA_CACHE_VERSION,
            "response": risk_recall_qa.model_dump(mode="json"),
        },
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
    )
    return risk_recall_qa, {
        "risk_recall_qa_cache_hit": False,
        "risk_recall_qa_cache_key": cache_key,
        "agent_elapsed_seconds": {"risk_recall_qa": round(time.perf_counter() - started_at, 4)},
    }


def _risk_recall_qa_cache_payload(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
    model_provider: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "cache_version": _RISK_RECALL_QA_CACHE_VERSION,
        "model_provider": model_provider,
        "model_name": model_name,
        "stage2_input_bundle": bundle.to_compact_prompt_payload(role="risk_recall_qa"),
        "committee_view": committee_view,
        "agent_outputs": {
            "quant_credit": quant_credit.model_dump(mode="json"),
            "evidence_audit": evidence_audit.model_dump(mode="json"),
            "chair_report": chair_report.model_dump(mode="json"),
        },
        "trigger_reasons": trigger_reasons,
    }


def _merge_post_committee_qa_diagnostics(
    runtime_diagnostics: dict[str, Any],
    qa_diagnostics: dict[str, Any],
    *,
    role: str,
) -> None:
    existing_timings = runtime_diagnostics.get("agent_elapsed_seconds")
    if not isinstance(existing_timings, dict):
        existing_timings = {}
    qa_timings = qa_diagnostics.get("agent_elapsed_seconds")
    qa_elapsed_seconds = 0.0
    if isinstance(qa_timings, dict):
        existing_timings.update(qa_timings)
        qa_elapsed = _safe_float(qa_timings.get(role))
        if qa_elapsed is not None:
            qa_elapsed_seconds = qa_elapsed
    runtime_diagnostics["agent_elapsed_seconds"] = existing_timings
    runtime_diagnostics["agent_elapsed_seconds_sum"] = round(
        sum(float(value) for value in existing_timings.values()),
        4,
    )
    current_total = _safe_float(runtime_diagnostics.get("stage2_total_elapsed_seconds"))
    if current_total is not None and qa_elapsed_seconds:
        runtime_diagnostics["stage2_total_elapsed_seconds"] = round(
            current_total + qa_elapsed_seconds,
            4,
        )
    runtime_diagnostics[f"{role}_cache_hit"] = bool(qa_diagnostics.get(f"{role}_cache_hit", False))
    if qa_diagnostics.get(f"{role}_cache_key"):
        runtime_diagnostics[f"{role}_cache_key"] = qa_diagnostics[f"{role}_cache_key"]


def _stage2_risk_recall_qa_provider() -> str:
    return (
        os.environ.get("CAS_STAGE2_RISK_RECALL_QA_PROVIDER")
        or os.environ.get("CAS_STAGE2_CHAIR_PROVIDER")
        or os.environ.get("CAS_STAGE2_MODEL_PROVIDER")
        or "openai"
    )


def _stage2_risk_recall_qa_model() -> str:
    return (
        os.environ.get("CAS_STAGE2_RISK_RECALL_QA_MODEL")
        or os.environ.get("CAS_STAGE2_CHAIR_MODEL")
        or os.environ.get("CAS_STAGE2_MODEL")
        or "gpt-4.1-mini"
    )


def _stage2_risk_recall_qa_max_tokens() -> int:
    try:
        return int(os.environ.get("CAS_STAGE2_RISK_RECALL_QA_MAX_TOKENS", "3000"))
    except ValueError:
        return 3000


def _committee_confidence(
    *,
    bundle: Stage2InputBundle,
    agents: list[AgentOutput],
    committee_view: dict[str, Any],
    rule_confidence: float,
    runtime_backend_name: str,
) -> float:
    """Blend model certainty, evidence quality, and agent certainty into one score."""
    probability = _clamp(bundle.probability_speculative)
    model_confidence = 0.45 + 0.35 * min(abs(probability - 0.5) * 2.0, 1.0)
    agent_confidence = _average_agent_confidence(agents)
    evidence_confidence = _external_evidence_quality(
        bundle.news_cache_snapshot,
        veto_triggered=bool(committee_view.get("veto_triggered", False)),
    )
    alignment_adjustment = _committee_alignment_adjustment(
        bundle=bundle,
        committee_label=str(committee_view.get("final_committee_label", "")),
    )
    fallback_penalty = -0.07 if "fallback" in runtime_backend_name else 0.0
    score = (
        0.35 * _clamp(rule_confidence)
        + 0.35 * model_confidence
        + 0.20 * agent_confidence
        + 0.10 * evidence_confidence
        + alignment_adjustment
        + fallback_penalty
    )
    return round(_clamp(score, minimum=0.2, maximum=0.95), 4)


def _average_agent_confidence(agents: list[AgentOutput]) -> float:
    if not agents:
        return 0.35
    return _clamp(sum(agent.confidence for agent in agents) / len(agents))


def _external_evidence_quality(
    news_cache: dict[str, Any],
    *,
    veto_triggered: bool,
) -> float:
    status = str(news_cache.get("status", "not_implemented"))
    if status in {"disabled", "not_implemented", "placeholder", "missing_credentials"}:
        return 0.35
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list) or not raw_items:
        return 0.4

    verified_count = _safe_int(news_cache.get("verified_item_count"))
    direct_count = sum(
        1 for item in raw_items if isinstance(item, dict) and item.get("company_match") is True
    )
    weak_count = sum(
        1 for item in raw_items if isinstance(item, dict) and item.get("company_match") is False
    )
    high_reliability_count = sum(
        1
        for item in raw_items
        if isinstance(item, dict)
        and (
            str(item.get("reliability", "")).lower() == "high"
            or str(item.get("source", "")).lower() == "opendart"
        )
    )
    average_item_score = _average_evidence_item_score(raw_items)
    score = 0.38 + 0.07 * min(verified_count, 3) + 0.04 * min(direct_count, 3)
    score += 0.08 * min(high_reliability_count, 2) + 0.15 * average_item_score
    score -= 0.05 * min(weak_count, 3)
    if veto_triggered:
        score += 0.15
    elif news_cache.get("has_critical_risk"):
        score -= 0.08
    return _clamp(score, minimum=0.2, maximum=0.85)


def _average_evidence_item_score(raw_items: list[object]) -> float:
    scores: list[float] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        score = item.get("evidence_score")
        if isinstance(score, int | float | str):
            try:
                scores.append(_clamp(float(score)))
            except ValueError:
                continue
    if not scores:
        return 0.35
    return sum(scores) / len(scores)


def _safe_int(value: object) -> int:
    try:
        return int(value) if isinstance(value, int | float | str) else 0
    except (TypeError, ValueError):
        return 0


def _committee_alignment_adjustment(
    *,
    bundle: Stage2InputBundle,
    committee_label: str,
) -> float:
    model_label = "적격" if bundle.prediction_label == "투자적격" else "부적격"
    if committee_label == model_label:
        return 0.08
    if committee_label == "보류":
        risk_band = str(bundle.rule_result.get("risk_band", ""))
        return 0.03 if risk_band == "watch" else 0.0
    return -0.06


def _clamp(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(max(value, minimum), maximum)


def _stage2_runner() -> Stage2AgentRunner:
    deterministic_runner = DeterministicStage2AgentRunner(
        quant_credit_agent=_quant_credit_agent,
        evidence_audit_agent=_evidence_audit_agent,
        chair_report_agent=_chair_report_agent,
    )
    runner_name = _stage2_runner_name()
    if runner_name in {"", "deterministic", "local", "offline"}:
        return deterministic_runner
    if runner_name == "agno":
        return AgnoStage2AgentRunner(
            deterministic_runner=deterministic_runner,
            routing_mode=os.environ.get("CAS_STAGE2_AGNO_MODE", "single"),
            model_provider=os.environ.get("CAS_STAGE2_MODEL_PROVIDER", "openai"),
            model_name=os.environ.get("CAS_STAGE2_MODEL", "gpt-4.1-mini"),
            quant_model_provider=os.environ.get("CAS_STAGE2_QUANT_PROVIDER") or None,
            quant_model_name=os.environ.get("CAS_STAGE2_QUANT_MODEL") or None,
            evidence_model_provider=os.environ.get("CAS_STAGE2_EVIDENCE_PROVIDER") or None,
            evidence_model_name=os.environ.get("CAS_STAGE2_EVIDENCE_MODEL") or None,
            chair_model_provider=os.environ.get("CAS_STAGE2_CHAIR_PROVIDER") or None,
            chair_model_name=os.environ.get("CAS_STAGE2_CHAIR_MODEL") or None,
            max_tokens=_stage2_max_tokens(),
            fallback_on_error=_stage2_fallback_on_error(),
        )
    raise ValueError(
        f"Unsupported CAS_STAGE2_RUNNER value. Use 'deterministic' or 'agno', got {runner_name!r}."
    )


def _stage2_max_tokens() -> int:
    try:
        return int(os.environ.get("CAS_STAGE2_MAX_TOKENS", "6000"))
    except ValueError:
        return 6000


def _stage2_runner_name() -> str:
    if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get(
        "CAS_ALLOW_LIVE_STAGE2_IN_TESTS", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        return "deterministic"
    return os.environ.get("CAS_STAGE2_RUNNER", "deterministic").strip().lower()


def _stage2_fallback_on_error() -> bool:
    value = os.environ.get("CAS_STAGE2_FALLBACK_ON_ERROR", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


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
            f" 다만 45개 보조 변수셋 신호가 `{review_priority}` 우선순위의 "
            f"추가 위원회 검토 대상으로 표시했습니다."
        )
    if overwarning_candidate:
        summary += (
            " 한편 조합형 재무 스트레스 필터는 1차 위험 경고가 과민할 가능성을 "
            "완화 요인으로 재확인하라고 표시했습니다."
        )
    key_risk_factors = [str(item.get("detail", "")) for item in risk_items if item.get("detail")]
    if secondary_triggered and trigger_reason:
        key_risk_factors.insert(0, f"45개 보조 변수셋 검토 신호: {trigger_reason}")
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


def _external_evidence_profile(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> _EvidenceProfile:
    status = str(news_cache.get("status", "not_implemented"))
    raw_items = news_cache.get("items", [])
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    item_count = len(items)
    direct_count = _safe_int(news_cache.get("direct_match_count"))
    if direct_count == 0:
        direct_count = sum(1 for item in items if item.get("company_match") is True)
    weak_count = _safe_int(news_cache.get("weak_evidence_count"))
    if weak_count == 0:
        weak_count = sum(1 for item in items if item.get("company_match") is not True)
    verified_count = _safe_int(news_cache.get("verified_item_count"))
    if verified_count == 0:
        verified_count = sum(1 for item in items if _is_verified_evidence_item(item))
    adverse_items = [
        item
        for item in items
        if _is_adverse_evidence_item(item, source_feature_row=source_feature_row)
    ]
    adverse_count = len(adverse_items)
    verified_adverse_count = sum(1 for item in adverse_items if _is_verified_evidence_item(item))
    veto_candidate_count = _safe_int(news_cache.get("veto_candidate_count"))
    if veto_candidate_count == 0:
        veto_candidate_count = sum(1 for item in items if item.get("veto_candidate") is True)
    high_confidence_critical_count = _safe_int(news_cache.get("high_confidence_critical_count"))
    if high_confidence_critical_count == 0:
        high_confidence_critical_count = sum(
            1 for item in items if item.get("critical_context_confirmed") is True
        )
    critical_terms = [str(term) for term in news_cache.get("critical_terms", []) or []]
    strength = _evidence_strength(
        status=status,
        item_count=item_count,
        direct_count=direct_count,
        verified_count=verified_count,
        adverse_count=adverse_count,
        verified_adverse_count=verified_adverse_count,
        veto_candidate_count=veto_candidate_count,
        high_confidence_critical_count=high_confidence_critical_count,
    )
    score = _evidence_strength_score(strength)
    return {
        "status": status,
        "strength": strength,
        "finding": _evidence_profile_finding(
            status=status,
            strength=strength,
            item_count=item_count,
            direct_count=direct_count,
            verified_count=verified_count,
            weak_count=weak_count,
            adverse_count=adverse_count,
            verified_adverse_count=verified_adverse_count,
            veto_candidate_count=veto_candidate_count,
            critical_terms=critical_terms,
        ),
        "item_count": item_count,
        "direct_count": direct_count,
        "verified_count": verified_count,
        "weak_count": weak_count,
        "adverse_count": adverse_count,
        "verified_adverse_count": verified_adverse_count,
        "veto_candidate_count": veto_candidate_count,
        "high_confidence_critical_count": high_confidence_critical_count,
        "critical_terms": critical_terms,
        "score": score,
    }


def _is_verified_evidence_item(item: dict[str, Any]) -> bool:
    score = _safe_float(item.get("evidence_score"))
    return score is not None and score >= 0.55


def _is_adverse_evidence_item(
    item: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
) -> bool:
    return bool(
        _shared_substantive_external_risk_item(
            item,
            source_feature_row=source_feature_row,
        )
    )


def _evidence_strength(
    *,
    status: str,
    item_count: int,
    direct_count: int,
    verified_count: int,
    adverse_count: int,
    verified_adverse_count: int,
    veto_candidate_count: int,
    high_confidence_critical_count: int,
) -> _EvidenceStrength:
    if status in {"disabled", "not_implemented", "placeholder", "missing_credentials"}:
        return "none"
    if item_count <= 0:
        return "none"
    if veto_candidate_count >= 2 and high_confidence_critical_count >= 1:
        return "critical"
    if veto_candidate_count >= 1 or high_confidence_critical_count >= 1:
        return "strong"
    if verified_adverse_count >= 1:
        return "strong"
    if adverse_count >= 1:
        return "moderate"
    if direct_count >= 1 and verified_count >= 1:
        return "weak"
    return "weak"


def _evidence_strength_score(strength: _EvidenceStrength) -> float:
    return {
        "none": 0.0,
        "weak": 0.18,
        "moderate": 0.38,
        "strong": 0.62,
        "critical": 0.85,
    }[strength]


def _evidence_profile_finding(
    *,
    status: str,
    strength: _EvidenceStrength,
    item_count: int,
    direct_count: int,
    verified_count: int,
    weak_count: int,
    adverse_count: int,
    verified_adverse_count: int,
    veto_candidate_count: int,
    critical_terms: list[str],
) -> str:
    if strength == "none":
        if status == "disabled":
            return "외부근거 점검: 외부 뉴스/공시 수집이 꺼져 있어 정성 근거는 판단 보류입니다."
        return f"외부근거 점검: 수집 상태가 `{status}`라서 확인 가능한 외부 근거가 제한적입니다."

    terms = ", ".join(critical_terms[:4]) if critical_terms else "configured critical terms"
    counts = (
        f"총 {item_count}건 중 직접 관련 {direct_count}건, 검증 가능 {verified_count}건, "
        f"위험 후보 {adverse_count}건, 검증된 위험 후보 {verified_adverse_count}건, "
        f"약한/간접 근거 {weak_count}건"
    )
    if strength in {"critical", "strong"}:
        return (
            f"외부근거 위험: {counts}이며, 위험 키워드 후보 {veto_candidate_count}건이 "
            f"감지되었습니다({terms}). 다중 출처·고신뢰 조건 충족 여부를 보수적으로 확인해야 합니다."
        )
    if strength == "moderate":
        return (
            f"외부근거 점검: {counts}입니다. 강한 위험 신호로 확인된 항목은 없으며 "
            "모델 판단을 보완할 참고 근거로 활용합니다."
        )
    return (
        f"외부근거 점검: {counts}입니다. 현재 확인된 항목은 routine/context 성격이거나 "
        "약한 근거이므로 모델 판단을 뒤집는 근거로 쓰지 않습니다."
    )


def _evidence_reliability_text(evidence_profile: _EvidenceProfile) -> str:
    return (
        "출처 신뢰도, 기업 직접 관련성, 최신성, 중복 여부, 위험 키워드의 문맥 확인 여부를 "
        "나눠 검증합니다. "
        f"현재 외부근거 강도는 `{evidence_profile['strength']}`이며, "
        f"직접 관련 {evidence_profile['direct_count']}건, "
        f"검증 가능 {evidence_profile['verified_count']}건으로 요약됩니다."
    )


def _evidence_limitations(
    news_cache: dict[str, Any],
    *,
    evidence_profile: _EvidenceProfile,
) -> list[str]:
    """Explain evidence coverage limits so Stage 2 does not overstate weak signals."""
    limitations: list[str] = []
    status = evidence_profile["status"]
    if status in {"disabled", "not_implemented", "placeholder", "missing_credentials"}:
        limitations.append(
            f"외부근거 수집 상태가 `{status}`라서 뉴스·웹·공시 기반 검증은 제한적입니다."
        )
    elif evidence_profile["item_count"] > 0 and evidence_profile["direct_count"] == 0:
        limitations.append(
            "수집 항목은 있지만 기업명 또는 종목코드 직접 관련성이 확인된 근거가 없습니다."
        )
    elif evidence_profile["weak_count"] > evidence_profile["direct_count"]:
        limitations.append(
            "간접/약한 근거가 직접 관련 근거보다 많아, 위험 신호를 확정 사실로 보지 않습니다."
        )

    date_filter_note = _historical_evidence_filter_note(news_cache)
    if date_filter_note:
        limitations.append(date_filter_note)

    provider_note = _provider_coverage_limitation_note(news_cache.get("providers"))
    if provider_note:
        limitations.append(provider_note)

    if not limitations and evidence_profile["strength"] in {"none", "weak"}:
        limitations.append(
            "현재 외부근거 강도는 낮아 모델 판단을 뒤집기보다 설명 보완용으로만 사용합니다."
        )
    return limitations[:3]


def _historical_evidence_filter_note(news_cache: dict[str, Any]) -> str:
    providers = news_cache.get("providers")
    if not isinstance(providers, dict):
        return ""
    end_dates: set[str] = set()
    filtered_after_cutoff = 0
    filtered_undated = 0
    historical_mode = False
    for provider in providers.values():
        if not isinstance(provider, dict):
            continue
        date_filter = provider.get("as_of_date_filter")
        if isinstance(date_filter, dict):
            historical_mode = historical_mode or bool(date_filter.get("historical_mode", False))
            end_date = str(date_filter.get("end_date") or "")
            if end_date:
                end_dates.add(end_date)
            filtered_after_cutoff += _safe_int(date_filter.get("filtered_after_cutoff_count"))
            filtered_undated += _safe_int(date_filter.get("filtered_undated_count"))
        query_window = provider.get("query_window")
        if isinstance(query_window, dict):
            end_date = str(query_window.get("end_date") or "")
            if end_date:
                end_dates.add(end_date)
    if not historical_mode:
        return ""
    cutoff = sorted(end_dates)[-1] if end_dates else str(news_cache.get("as_of_date") or "")
    filtered_count = filtered_after_cutoff + filtered_undated
    if filtered_count <= 0:
        return f"과거 기준일 {cutoff} 이전 공개 근거만 사용하도록 날짜 필터를 적용했습니다."
    return (
        f"과거 기준일 {cutoff} 이후 또는 날짜 미확인 근거 {filtered_count}건을 제외해 "
        "look-ahead bias를 줄였습니다."
    )


def _provider_coverage_limitation_note(providers: object) -> str:
    if not isinstance(providers, dict) or not providers:
        return ""
    limited: list[str] = []
    for provider_name, raw_provider in providers.items():
        if not isinstance(raw_provider, dict):
            continue
        status = str(raw_provider.get("status") or "")
        if status in {"missing_key", "error", "partial_error", "missing_corp_code"}:
            limited.append(f"{provider_name}:{status}")
    if not limited:
        return ""
    return "일부 수집 경로에 제한이 있습니다(" + ", ".join(limited[:3]) + ")."


def _model_evidence_challenge(
    *,
    bundle: Stage2InputBundle,
    debt_findings: list[str],
    evidence_profile: _EvidenceProfile,
) -> str:
    prediction_label = bundle.prediction_label
    strength = evidence_profile["strength"]
    has_debt_risk = _contains_any(
        debt_findings,
        ("추가 경계", "부족", "취약", "제한적", "어렵습니다", "약합니다", "차환 리스크"),
    )
    has_debt_support = _contains_any(
        debt_findings,
        ("완충 근거", "완화 신호", "방어력", "양호", "확보", "여력"),
    )
    if prediction_label == "투자적격" and strength in {"strong", "critical"}:
        return (
            "정량상 투자적격이지만 직접 관련 외부 위험 근거가 있어 위원회 보수 검토가 필요합니다."
        )
    has_offsetting_support = (
        has_debt_support and strength in {"none", "weak"} and bundle.probability_speculative < 0.10
    )
    if prediction_label == "투자적격" and has_debt_risk and not has_offsetting_support:
        return "정량상 투자적격이지만 유동성·상환여력 신호가 일부 충돌해 추가 점검이 필요합니다."
    if prediction_label == "투자적격" and has_debt_risk and has_offsetting_support:
        return (
            "일부 부채·유동성 경고 신호가 있으나 현금흐름과 상환여력 완화 요인이 더 커 "
            "현재 모델 원판단을 뒤집을 수준은 아닙니다."
        )
    if prediction_label == "부적격" and has_debt_support and strength in {"none", "weak"}:
        return "정량상 부적격 판단은 유지하되, 부채·현금흐름 일부 지표는 완화 근거로 재검토할 수 있습니다."
    return "정량 모델 판단과 외부/유동성 검증 사이의 중대한 충돌은 제한적입니다."


def _evidence_audit_conclusion(
    *,
    bundle: Stage2InputBundle,
    debt_findings: list[str],
    evidence_profile: _EvidenceProfile,
) -> str:
    strength = evidence_profile["strength"]
    if strength == "critical":
        return "외부 근거가 치명 리스크 후보에 가까워 veto 규칙 충족 여부를 최우선으로 확인해야 합니다."
    if strength == "strong":
        return "외부 근거가 강하므로 모델 원판단보다 보수적인 보류 또는 부적격 검토가 필요합니다."
    if _contains_any(debt_findings, ("추가 경계", "차환 리스크", "상환 재원", "1배 미만")):
        if strength in {"none", "weak"} and _contains_any(
            debt_findings,
            ("완충 근거", "완화 신호", "현금 여력", "상환 방어력", "양호", "확보", "여력"),
        ):
            return (
                "부채·유동성 일부 경고는 있으나 현금흐름과 상환여력 완화 요인이 함께 확인되어 "
                "모델 원판단을 뒤집기보다 참고 점검 포인트로 처리합니다."
            )
        return "외부 치명 리스크는 확정되지 않았지만 부채·유동성 측면에서 보류 의견을 강화합니다."
    if bundle.prediction_label == "부적격" and _contains_any(
        debt_findings,
        ("완화 신호", "현금 여력", "상환 방어력", "양호"),
    ):
        return "부적격 원판단은 보존하되, 현금흐름과 상환여력 완화 요인을 함께 표시해야 합니다."
    return "현재 확인된 외부 근거는 모델 원판단을 뒤집기보다 설명과 점검 포인트를 보완합니다."


def _evidence_audit_confidence(
    *,
    status: str,
    debt_confidence: float,
    evidence_profile: _EvidenceProfile,
) -> float:
    if status in {"not_implemented", "disabled", "placeholder"}:
        return round(_clamp(max(0.25, debt_confidence - 0.08), maximum=0.62), 4)
    score = 0.28 + 0.35 * _clamp(debt_confidence) + 0.32 * float(evidence_profile["score"])
    if evidence_profile["direct_count"] > 0:
        score += 0.05
    if evidence_profile["weak_count"] > evidence_profile["direct_count"]:
        score -= 0.04
    return round(_clamp(score, minimum=0.28, maximum=0.88), 4)


def _contains_any(values: list[str], markers: tuple[str, ...]) -> bool:
    text = " ".join(values)
    return any(marker in text for marker in markers)


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


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if not isinstance(value, int | float | str):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _recommendation_from_score(score: float, thresholds: dict[str, float]) -> Recommendation:
    """Map a numeric suitability score to the legacy recommendation buckets."""
    if score >= float(thresholds["priority"]):
        return "priority"
    if score >= float(thresholds["watch"]):
        return "watch"
    if score >= float(thresholds["review"]):
        return "review"
    return "defer"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
