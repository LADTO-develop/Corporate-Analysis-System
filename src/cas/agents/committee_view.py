"""Build the dashboard-facing Stage 2 committee_view payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from cas.agents.committee_schema import CommitteeLabel, CommitteeViewPayload
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
    mitigating_factors = _collect_committee_factors(agents, target="mitigation")
    if not veto_triggered:
        committee_label = _committee_label_with_evidence_escalation(
            committee_label,
            agents=agents,
            hidden_tail_risk=hidden_tail_risk,
        )
    evidence_summary = _evidence_summary_items(bundle, agents)
    conflict_resolution = _conflict_resolution(
        prediction_label=prediction_label,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
    )
    final_review_memo = _final_review_memo(
        prediction_label=prediction_label,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
        hidden_tail_risk=hidden_tail_risk,
        risk_factors=risk_factors,
        mitigating_factors=mitigating_factors,
    )

    return CommitteeViewPayload(
        final_committee_label=committee_label,
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
    if _evidence_agent_requires_hold(evidence_agent):
        return "보류"
    return committee_label


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


def _is_adverse_external_item(item: dict[str, Any]) -> bool:
    if item.get("veto_candidate") is True:
        return True
    if item.get("critical_context_confirmed") is True:
        return True
    if str(item.get("provider_relevance", "")).lower() in _ADVERSE_PROVIDER_RELEVANCE:
        return True
    terms = _item_critical_terms(item)
    if not terms:
        return False
    return str(item.get("evidence_quality", "")).lower() in _ADVERSE_EVIDENCE_QUALITY


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
    return critical_terms_in_text(text)


def _model_threshold(bundle: Stage2InputBundle) -> float:
    for source in (bundle.xgboost_result, bundle.model_view, bundle.rule_result):
        for key in ("threshold", "threshold_tuned", "decision_threshold"):
            value = _safe_float(source.get(key))
            if value is not None and value > 0:
                return value
    return 0.315


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
        "확정되지 않았",
        "중대한 충돌은 제한적",
        "현재 연결된 뉴스/공시 항목은 없습니다",
        "확인 가능한 외부 근거가 제한적",
        "수집 상태가 `not_requested`",
        "수집 상태가 `disabled`",
        "수집 상태가 `no_results`",
    )
    return any(marker in text for marker in neutral_markers)


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


def _conflict_resolution(
    *,
    prediction_label: str,
    committee_label: str,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
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
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["build_committee_view", "build_committee_view_model"]
