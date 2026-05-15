"""Run the Stage 2 three-agent review scaffold."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from cas.agents.committee_view import build_committee_view
from cas.agents.signals import (
    evaluate_debt_liquidity,
    evaluate_external_evidence,
    evaluate_macro_market,
)
from cas.agents.stage2_bundle import Stage2InputBundle, build_stage2_input_bundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
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
from cas.utils.io import read_json

_FEATURE_METADATA_PATH = Path("data/input/credit_43_features/feature_43_dictionary_metadata.json")

_INDUSTRY_LABELS = {
    "manufacturing": "제조업",
    "construction": "건설업",
    "retail_wholesale": "도소매업",
    "it_services": "IT·서비스업",
    "transport_storage": "운수·창고업",
    "other": "기타",
}

_SIZE_LABELS = {
    "large": "대기업",
    "mid_sized": "중견기업",
    "small_medium": "중소기업",
    "other": "기타",
}

_POLARITY: dict[str, Literal["higher_better", "lower_better", "contextual", "flag_positive"]] = {
    "current_ratio": "higher_better",
    "cash_ratio": "higher_better",
    "equity_ratio": "higher_better",
    "debt_ratio": "lower_better",
    "total_borrowings_ratio": "lower_better",
    "capital_impairment_ratio": "lower_better",
    "net_margin": "higher_better",
    "gross_profit": "higher_better",
    "interest_coverage_ratio": "higher_better",
    "pretax_roa": "higher_better",
    "operating_roa": "higher_better",
    "pretax_roe": "higher_better",
    "ocf_to_total_liabilities": "higher_better",
    "ocf_to_total_borrowings": "higher_better",
    "ocf_to_sales": "higher_better",
    "cashflow_coverage_ratio": "higher_better",
    "accruals_ratio": "lower_better",
    "intangible_assets_ratio": "lower_better",
    "total_debt_turnover": "higher_better",
    "dividend_payer": "flag_positive",
    "market_to_book": "contextual",
    "spec_spread": "lower_better",
    "short_term_borrowings_share": "lower_better",
    "total_assets_growth": "contextual",
    "net_margin_diff": "higher_better",
    "is_2y_consecutive_ocf_deficit": "lower_better",
    "icr_under_1": "lower_better",
    "is_2y_consecutive_operating_loss": "lower_better",
}

_EvidenceStrength = Literal["none", "weak", "moderate", "strong", "critical"]


class _EvidenceProfile(TypedDict):
    status: str
    strength: _EvidenceStrength
    finding: str
    item_count: int
    direct_count: int
    verified_count: int
    weak_count: int
    veto_candidate_count: int
    high_confidence_critical_count: int
    critical_terms: list[str]
    score: float


def run(state: AgentState) -> dict[str, Any]:
    """Run the three-agent Stage 2 scaffold over Stage 1 outputs."""
    bundle = build_stage2_input_bundle(state)

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
    agents = [output.to_agent_output() for output in structured_outputs]
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
    committee_view = build_committee_view(
        bundle=bundle,
        recommendation=recommendation,
        agents=agents,
    )
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
        "synthesis": agents[-1].summary,
        "agents": {
            agent.role: {
                "summary": agent.summary,
                "findings": agent.findings,
                "confidence": agent.confidence,
            }
            for agent in agents
        },
    }

    audit = AuditEntry(
        node="agno_agents",
        timestamp=_now(),
        summary=(
            f"Three-agent Stage 2 scaffold completed via {runtime_backend_name} runner: "
            f"{', '.join(agent.role for agent in agents)}"
        ),
        metrics={
            "n_agents": float(len(agents)),
            "rule_confidence": rule_confidence,
            "final_confidence": committee_confidence,
        },
    )
    return {
        "agent_outputs": agents,
        "committee_reviews": reviews,
        "agent_summary": agent_summary,
        "committee_view": committee_view,
        "final_recommendation": recommendation,
        "final_confidence": committee_confidence,
        "audit": [audit],
    }


def _validate_agent_order(agents: list[AgentOutput]) -> None:
    actual_roles = tuple(agent.role for agent in agents)
    if actual_roles != STAGE2_AGENT_ROLES:
        expected = ", ".join(STAGE2_AGENT_ROLES)
        actual = ", ".join(actual_roles)
        raise ValueError(f"Stage 2 agent order mismatch: expected {expected}, got {actual}")


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
            model_name=os.environ.get("CAS_STAGE2_MODEL", "claude-sonnet-4-5-20250929"),
            max_tokens=_stage2_max_tokens(),
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


def _quant_credit_agent(bundle: Stage2InputBundle) -> QuantCreditOutput:
    source_row = bundle.source_feature_row
    peer_by_feature = bundle.peer_rows_by_feature
    company_name = bundle.company_name
    market = _humanize_category(source_row.get("market"), fallback=bundle.market)
    industry = _humanize_category(
        source_row.get("industry_macro_category"),
        mapping=_INDUSTRY_LABELS,
        fallback="업종 정보 미확인",
    )
    size_group = _humanize_category(
        source_row.get("firm_size_group"),
        mapping=_SIZE_LABELS,
        fallback="규모 정보 미확인",
    )

    probability = bundle.probability_speculative
    prediction_label = bundle.prediction_label
    # Stage 1의 top_drivers와 source row 원값, peer comparison을 같이 묶어서
    # "모델이 왜 그렇게 판단했는지"를 사람 문장으로 바꾸는 것이 QuantCreditAgent의 핵심 역할이다.
    driver_details = _describe_top_drivers(bundle.xgboost_result, source_row, peer_by_feature)
    risk_items = [item for item in driver_details if item["direction"] == "risk"]
    support_items = [item for item in driver_details if item["direction"] == "support"]

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

    return QuantCreditOutput(
        quant_summary=summary,
        model_rationale=(
            f"상위 SHAP 변수 {min(len(driver_details), 3)}개를 기준으로 모델 판단의 근거를 정리했습니다."
        ),
        key_risk_factors=[str(item.get("detail", "")) for item in risk_items if item.get("detail")],
        mitigating_factors=[
            str(item.get("detail", "")) for item in support_items if item.get("detail")
        ],
        confidence=0.82 if bundle.xgboost_result else 0.35,
    )


def _evidence_audit_agent(bundle: Stage2InputBundle) -> EvidenceAuditOutput:
    status = bundle.news_status
    debt_signals = evaluate_debt_liquidity(bundle)
    macro_signals = evaluate_macro_market(bundle)
    external_signals = evaluate_external_evidence(bundle.news_cache_snapshot)
    evidence_profile = _external_evidence_profile(bundle.news_cache_snapshot)
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
            *external_signals.findings,
        ],
        confidence=_evidence_audit_confidence(
            status=status,
            debt_confidence=debt_signals.confidence,
            evidence_profile=evidence_profile,
        ),
    )


def _external_evidence_profile(news_cache: dict[str, Any]) -> _EvidenceProfile:
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
            veto_candidate_count=veto_candidate_count,
            critical_terms=critical_terms,
        ),
        "item_count": item_count,
        "direct_count": direct_count,
        "verified_count": verified_count,
        "weak_count": weak_count,
        "veto_candidate_count": veto_candidate_count,
        "high_confidence_critical_count": high_confidence_critical_count,
        "critical_terms": critical_terms,
        "score": score,
    }


def _is_verified_evidence_item(item: dict[str, Any]) -> bool:
    score = _safe_float(item.get("evidence_score"))
    return score is not None and score >= 0.55


def _evidence_strength(
    *,
    status: str,
    item_count: int,
    direct_count: int,
    verified_count: int,
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
    if direct_count >= 1 and verified_count >= 1:
        return "moderate"
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
        f"외부근거 점검: {counts}입니다. 현재 근거는 약하거나 간접적이므로 "
        "모델 판단을 뒤집는 근거로 쓰지 않습니다."
    )


def _evidence_reliability_text(evidence_profile: _EvidenceProfile) -> str:
    return (
        "출처 신뢰도, 기업 직접 관련성, 최신성, 중복 여부, 위험 키워드의 문맥 확인 여부를 "
        "나눠 검증합니다. "
        f"현재 외부근거 강도는 `{evidence_profile['strength']}`이며, "
        f"직접 관련 {evidence_profile['direct_count']}건, "
        f"검증 가능 {evidence_profile['verified_count']}건으로 요약됩니다."
    )


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
    if prediction_label == "투자적격" and has_debt_risk:
        return "정량상 투자적격이지만 유동성·상환여력 신호가 일부 충돌해 추가 점검이 필요합니다."
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


def _describe_top_drivers(
    xgb: dict[str, Any],
    source_row: dict[str, Any],
    peer_by_feature: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for name, shap_value in _driver_pairs(xgb):
        metadata = _feature_metadata().get(name, {})
        feature_name = str(metadata.get("korean_name") or _prettify_feature_name(name))
        unit = str(metadata.get("unit") or "")
        raw_value = source_row.get(name)
        direction = _driver_direction(name, shap_value)
        details.append(
            {
                "name": name,
                "feature": feature_name,
                "shap_value": shap_value,
                "direction": direction,
                # Stage 2의 첫 에이전트는 "모델이 왜 그렇게 봤는지"를 설명하는 역할이므로,
                # 값 자체와 peer context를 한 문장으로 합쳐 findings에 넘긴다.
                "detail": _feature_point_text(
                    feature_name=feature_name,
                    feature_key=name,
                    raw_value=raw_value,
                    unit=unit,
                    shap_value=shap_value,
                    peer_row=peer_by_feature.get(name),
                ),
            }
        )
    return details


def _feature_point_text(
    *,
    feature_name: str,
    feature_key: str,
    raw_value: object,
    unit: str,
    shap_value: float,
    peer_row: dict[str, Any] | None,
) -> str:
    direction = _driver_direction(feature_key, shap_value)
    value_text = _format_feature_value(raw_value, unit)
    comparison_text = _peer_comparison_text(peer_row=peer_row, unit=unit)
    if direction == "risk":
        return (
            f"{feature_name}({value_text})이(가) 현재 모델에서 위험을 높이는 방향으로 작용했습니다."
            f"{comparison_text}"
        )
    return (
        f"{feature_name}({value_text})이(가) 현재 모델에서 위험을 낮추는 방향으로 작용했습니다."
        f"{comparison_text}"
    )


def _driver_direction(feature_key: str, shap_value: float) -> Literal["risk", "support"]:
    polarity = _POLARITY.get(feature_key, "contextual")
    if polarity in {"higher_better", "flag_positive"}:
        return "risk" if shap_value > 0 else "support"
    if polarity == "lower_better":
        return "risk" if shap_value > 0 else "support"
    return "risk" if shap_value > 0 else "support"


def _driver_pairs(xgb: dict[str, Any]) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for item in xgb.get("top_drivers", []) or []:
        if isinstance(item, dict):
            name = str(item.get("name", item.get("feature", "")))
            value = float(item.get("value", item.get("score", 0.0)) or 0.0)
        else:
            name = str(item[0])
            value = float(item[1])
        if name:
            pairs.append((name, value))
    return pairs


@lru_cache(maxsize=1)
def _feature_metadata() -> dict[str, dict[str, Any]]:
    metadata = read_json(_FEATURE_METADATA_PATH)
    columns = metadata.get("columns", [])
    return {
        str(column.get("variable_name")): dict(column)
        for column in columns
        if isinstance(column, dict) and column.get("variable_name")
    }


def _prettify_feature_name(name: str) -> str:
    return name.replace("_", " ")


def _humanize_category(
    value: object,
    *,
    mapping: dict[str, str] | None = None,
    fallback: str = "unknown",
) -> str:
    if value is None:
        return fallback
    raw = str(value)
    if not raw:
        return fallback
    if mapping and raw in mapping:
        return mapping[raw]
    return raw


def _format_feature_value(value: object, unit: str) -> str:
    if value is None:
        return "값 없음"
    if unit == "0/1":
        numeric = _safe_float(value)
        if numeric is None:
            return "값 없음"
        return "예" if numeric >= 1.0 else "아니오"
    numeric = _safe_float(value)
    return _format_number(numeric, unit)


def _format_number(value: float | None, unit: str) -> str:
    if value is None:
        return "값 없음"
    if unit == "KRW thousand":
        return f"{value:,.0f}"
    if unit == "%p":
        return f"{value:.2f}%p"
    if unit == "ratio":
        return f"{value:.3f}"
    return f"{value:.3f}"


def _peer_comparison_text(*, peer_row: dict[str, Any] | None, unit: str) -> str:
    if not peer_row:
        return ""

    industry_median = _safe_float(peer_row.get("industry_median"))
    market_median = _safe_float(peer_row.get("market_median"))
    industry_percentile = _safe_float(peer_row.get("industry_percentile"))

    parts: list[str] = []
    if industry_median is not None:
        parts.append(f"산업 중앙값은 {_format_number(industry_median, unit)}입니다")
    if market_median is not None:
        parts.append(f"시장 중앙값은 {_format_number(market_median, unit)}입니다")
    if industry_percentile is not None:
        parts.append(f"산업 내 위치는 {industry_percentile:.1f}백분위입니다")

    if not parts:
        return ""
    return " " + " ".join(parts) + "."


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
