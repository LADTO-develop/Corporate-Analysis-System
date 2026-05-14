"""Build the dashboard-facing Stage 2 committee_view payload."""

from __future__ import annotations

from typing import Any, Literal, cast

from cas.agents.committee_schema import CommitteeLabel, CommitteeViewPayload
from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.state import AgentOutput, Recommendation
from cas.veto_rules import (
    VetoRules,
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

    risk_factors = _collect_committee_factors(agents, target="risk")
    mitigating_factors = _collect_committee_factors(agents, target="mitigation")
    evidence_summary = _evidence_summary_items(bundle, agents)
    conflict_resolution = _conflict_resolution(
        prediction_label=prediction_label,
        committee_label=committee_label,
        veto_triggered=veto_triggered,
    )
    final_review_memo = (
        f"모델 원판단은 {prediction_label}으로 보존합니다. 위원회는 정량 해석, "
        f"부채/유동성 교차 검증, 외부 근거 상태를 함께 검토해 최종 의견을 "
        f"{committee_label}로 정리했습니다."
    )

    return CommitteeViewPayload(
        final_committee_label=committee_label,
        veto_triggered=veto_triggered,
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
    if not text.startswith("부채·유동성 검증 의견:"):
        return None

    value = text.removeprefix("부채·유동성 검증 의견:").strip()
    classification = _classify_debt_liquidity_validation(value)
    if classification == target:
        return value
    return None


def _classify_debt_liquidity_validation(
    text: str,
) -> Literal["risk", "mitigation", "neutral"]:
    risk_markers = (
        "추가 경계",
        "추가 점검",
        "부적격 판단을 보수적으로 뒷받침",
        "부족",
        "취약",
        "제한",
        "어렵습니다",
        "약합니다",
    )
    mitigation_markers = (
        "완충 근거",
        "완화 신호",
        "방어력",
        "양호",
        "확보",
        "여력",
        "과도하지",
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
) -> str:
    if veto_triggered:
        return (
            "치명적 외부 위험 신호가 확인되어 모델 원판단과 무관하게 "
            "위원회 의견을 부적격으로 보수 조정했습니다."
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


__all__ = ["build_committee_view", "build_committee_view_model"]
