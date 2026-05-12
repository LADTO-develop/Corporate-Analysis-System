"""Run the Stage 2 five-agent review scaffold."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

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


def run(state: AgentState) -> dict[str, Any]:
    """Run the five-agent Stage 2 scaffold over Stage 1 outputs."""
    xgb = dict(state.get("xgboost_result") or {})
    rule = dict(state.get("rule_result") or {})

    recommendation = cast(
        Recommendation,
        rule.get("recommendation") or state.get("final_recommendation") or "review",
    )
    confidence = round(
        float(rule.get("confidence", state.get("final_confidence", 0.0)) or 0.0),
        4,
    )

    # 지금은 FinancialModelAgent만 실제 정량 해석을 수행하고,
    # 나머지 에이전트는 후속 구현을 위한 Stage 2 골격을 유지한다.
    agents = [
        _financial_model_agent(state, xgb),
        _debt_liquidity_agent(state),
        _macro_market_agent(state),
        _evidence_audit_agent(state),
        _chair_investment_agent(state, recommendation, confidence),
    ]
    reviews = [
        CommitteeReview(
            perspective=agent.role,
            recommendation=recommendation,
            confidence=agent.confidence,
            rationale=agent.summary,
        )
        for agent in agents
    ]
    # agent_summary는 대시보드/리포트에서 바로 읽기 쉬운 dict 구조이고,
    # agent_outputs / committee_reviews는 schema와 audit trail 쪽에서 쓰는 정규화 결과다.
    agent_summary = {
        "final_recommendation": recommendation,
        "final_confidence": confidence,
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
        summary=f"Five-agent Stage 2 scaffold completed: {', '.join(agent.role for agent in agents)}",
        metrics={"n_agents": float(len(agents)), "final_confidence": confidence},
    )
    return {
        "agent_outputs": agents,
        "committee_reviews": reviews,
        "agent_summary": agent_summary,
        "final_recommendation": recommendation,
        "final_confidence": confidence,
        "audit": [audit],
    }


def _financial_model_agent(state: AgentState, xgb: dict[str, Any]) -> AgentOutput:
    source_row = dict(state.get("source_feature_row") or {})
    peer_rows = list(state.get("peer_comparison_rows") or [])
    # feature별 peer row를 미리 맵으로 만들어 두면, SHAP 상위 변수 설명에
    # 산업/시장 중앙값과 백분위를 바로 붙일 수 있다.
    peer_by_feature = {
        str(row.get("feature")): row for row in peer_rows if isinstance(row.get("feature"), str)
    }
    company_name = str(state.get("company_name") or state.get("company_id", "unknown"))
    market = _humanize_category(
        source_row.get("market"), fallback=str(state.get("market", "UNKNOWN"))
    )
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

    probability = float(xgb.get("probability_speculative", 0.0) or 0.0)
    prediction_label = str(xgb.get("prediction_label", "unknown"))
    # Stage 1의 top_drivers와 source row 원값, peer comparison을 같이 묶어서
    # "모델이 왜 그렇게 판단했는지"를 사람 문장으로 바꾸는 것이 FinancialModelAgent의 핵심 역할이다.
    driver_details = _describe_top_drivers(xgb, source_row, peer_by_feature)
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
        f"{company_name}은(는) {market} 시장의 {industry} {size_group} 분류에 속하며, "
        f"모델은 현재 기업을 {prediction_label}으로 판단했습니다. "
        f"투기등급 위험확률은 {probability:.1%}이며, {primary_risk} {primary_support}"
    )

    findings = [
        f"정량 해석 요약: 상위 SHAP 변수 {min(len(driver_details), 3)}개를 기준으로 모델 판단의 근거를 정리했습니다.",
        f"핵심 위험 요인: {_join_feature_points(risk_items) or '상위 변수 기준 뚜렷한 위험 가중 요인은 제한적입니다.'}",
        f"완화 요인: {_join_feature_points(support_items) or '상위 변수 기준 완화 요인은 제한적입니다.'}",
    ]

    return AgentOutput(
        role="financial_model",
        summary=summary,
        findings=findings,
        confidence=0.82 if xgb else 0.35,
    )


def _debt_liquidity_agent(state: AgentState) -> AgentOutput:
    source_row = dict(state.get("source_feature_row") or {})
    xgb = dict(state.get("xgboost_result") or {})
    model_view = dict(state.get("model_view") or {})
    prediction_label = str(
        xgb.get("prediction_label") or model_view.get("prediction_label") or "unknown"
    )

    current_ratio = _safe_float(source_row.get("current_ratio"))
    cash_ratio = _safe_float(source_row.get("cash_ratio"))
    debt_ratio = _safe_float(source_row.get("debt_ratio"))
    short_term_share = _safe_float(source_row.get("short_term_borrowings_share"))
    cashflow_coverage = _safe_float(source_row.get("cashflow_coverage_ratio"))
    interest_coverage = _safe_float(source_row.get("interest_coverage_ratio"))
    ocf_to_liabilities = _safe_float(source_row.get("ocf_to_total_liabilities"))
    ocf_to_borrowings = _safe_float(source_row.get("ocf_to_total_borrowings"))
    ocf_deficit_flag = _safe_float(source_row.get("is_2y_consecutive_ocf_deficit"))
    icr_under_1_flag = _safe_float(source_row.get("icr_under_1"))

    liquidity_risks: list[str] = []
    liquidity_supports: list[str] = []
    repayment_risks: list[str] = []
    repayment_supports: list[str] = []

    if current_ratio is not None:
        if current_ratio < 1.0:
            liquidity_risks.append("유동비율이 1.0 미만으로 단기 상환 재원이 부족합니다.")
        elif current_ratio < 1.5:
            liquidity_risks.append("유동비율이 1.5 미만으로 단기 완충력이 제한적입니다.")
        elif current_ratio >= 2.0:
            liquidity_supports.append(
                "유동비율이 2.0 이상으로 단기 유동성 방어력이 확보되어 있습니다."
            )

    if cash_ratio is not None:
        if cash_ratio < 0.2:
            liquidity_risks.append("현금비율이 0.2 미만으로 즉시 사용 가능한 현금 버퍼가 약합니다.")
        elif cash_ratio >= 0.5:
            liquidity_supports.append(
                "현금비율이 0.5 이상으로 즉시 대응 가능한 현금 여력이 있습니다."
            )

    if short_term_share is not None:
        if short_term_share >= 0.6:
            liquidity_risks.append("단기차입금 비중이 높아 차환 리스크에 취약합니다.")
        elif short_term_share <= 0.35:
            liquidity_supports.append("단기차입금 비중이 낮아 만기구조 부담이 상대적으로 덜합니다.")

    if debt_ratio is not None:
        if debt_ratio >= 2.5:
            repayment_risks.append("부채비율이 높아 자본 완충력이 얇습니다.")
        elif debt_ratio <= 1.0:
            repayment_supports.append("부채비율이 1.0 이하로 레버리지 부담이 과도하지 않습니다.")

    if (icr_under_1_flag is not None and icr_under_1_flag >= 1.0) or (
        interest_coverage is not None and interest_coverage < 1.0
    ):
        repayment_risks.append(
            "이자보상배율이 1배 미만이어서 영업이익만으로 금융비용을 감당하기 어렵습니다."
        )

    if ocf_deficit_flag is not None and ocf_deficit_flag >= 1.0:
        repayment_risks.append("영업현금흐름 적자가 이어져 상환 재원의 지속성이 약합니다.")
    elif ocf_to_liabilities is not None:
        if ocf_to_liabilities < 0.0:
            repayment_risks.append(
                "영업현금흐름이 총부채 대비 음수로 상환 재원 창출력이 부족합니다."
            )
        elif ocf_to_liabilities >= 0.1:
            repayment_supports.append(
                "영업현금흐름이 총부채 대비 0.1 이상으로 상환 재원 창출력이 확인됩니다."
            )

    if cashflow_coverage is not None:
        if cashflow_coverage < 1.0:
            repayment_risks.append(
                "현금흐름 커버리지가 1배 미만으로 상환 부담 흡수력이 제한적입니다."
            )
        elif cashflow_coverage >= 5.0:
            repayment_supports.append("현금흐름 커버리지가 5배 이상으로 상환 방어력이 양호합니다.")

    if ocf_to_borrowings is not None and ocf_to_borrowings >= 0.2:
        repayment_supports.append(
            "영업현금흐름이 차입금 대비 0.2 이상으로 차입 상환 여력이 뒷받침됩니다."
        )

    if interest_coverage is not None and interest_coverage >= 3.0:
        repayment_supports.append("이자보상배율이 3배 이상으로 이자 부담 흡수력이 양호합니다.")

    validation = _debt_liquidity_validation(
        prediction_label=prediction_label,
        liquidity_risks=liquidity_risks,
        liquidity_supports=liquidity_supports,
        repayment_risks=repayment_risks,
        repayment_supports=repayment_supports,
    )

    summary = (
        f"DebtLiquidityAgent는 1단계 {prediction_label} 판단을 부채상환능력과 "
        f"유동성 관점에서 검증했습니다. {validation}"
    )
    findings = [
        f"모델 검증 의견: {validation}",
        "유동성 점검: "
        + _join_signal_text(
            risks=liquidity_risks,
            supports=liquidity_supports,
            fallback=(
                f"유동비율은 {_format_number(current_ratio, 'ratio')}이고 "
                f"현금비율은 {_format_number(cash_ratio, 'ratio')}로, 즉시 두드러진 유동성 경고는 제한적입니다."
            ),
        ),
        "상환여력 점검: "
        + _join_signal_text(
            risks=repayment_risks,
            supports=repayment_supports,
            fallback=(
                "현금흐름 커버리지와 이자보상배율을 함께 보더라도 "
                "즉시 상환여력 신호는 중립적인 수준입니다."
            ),
        ),
    ]

    return AgentOutput(
        role="debt_liquidity",
        summary=summary,
        findings=findings,
        confidence=_debt_liquidity_confidence(
            [
                current_ratio,
                cash_ratio,
                debt_ratio,
                short_term_share,
                cashflow_coverage,
                interest_coverage,
                ocf_to_liabilities,
                ocf_to_borrowings,
            ]
        ),
    )


def _macro_market_agent(state: AgentState) -> AgentOutput:
    source_row = dict(state.get("source_feature_row") or {})
    spec_spread = _safe_float(source_row.get("spec_spread"))
    market = _humanize_category(
        source_row.get("market"), fallback=str(state.get("market", "UNKNOWN"))
    )

    summary = (
        f"MacroMarketAgent는 현재 {market} 시장과 거시 변수 중 즉시 연결된 "
        "투기경계 스프레드 수준을 확인했습니다."
    )
    findings = [
        f"투기경계 스프레드는 {_format_number(spec_spread, '%p')}입니다."
        if spec_spread is not None
        else "현재 source row에는 투기경계 스프레드 값이 없습니다.",
        "추후 금리, 환율, 회사채 스프레드 묶음을 연결하면 거시 해석을 확장할 수 있습니다.",
    ]
    return AgentOutput(
        role="macro_market",
        summary=summary,
        findings=findings,
        confidence=0.5,
    )


def _evidence_audit_agent(state: AgentState) -> AgentOutput:
    news_cache = dict(state.get("news_cache_snapshot") or {})
    status = str(news_cache.get("status", "not_implemented"))
    summary = "EvidenceAuditAgent는 외부 뉴스·공시 근거 번들의 연결 상태를 점검했습니다."
    findings = [
        f"현재 뉴스/공시 근거 상태는 `{status}`입니다.",
        "외부 근거 수집 파이프라인이 연결되면 기사·공시·주석의 신뢰도 검증을 담당합니다.",
    ]
    return AgentOutput(
        role="evidence_audit",
        summary=summary,
        findings=findings,
        confidence=0.2 if status == "not_implemented" else 0.45,
    )


def _chair_investment_agent(
    state: AgentState,
    recommendation: Recommendation,
    confidence: float,
) -> AgentOutput:
    xgb = dict(state.get("xgboost_result") or {})
    prediction_label = str(xgb.get("prediction_label", "unknown"))
    probability = float(xgb.get("probability_speculative", 0.0) or 0.0)
    summary = (
        f"ChairInvestmentAgent는 현재 단계에서 모델 원판단 {prediction_label}과 "
        f"위험확률 {probability:.1%}를 유지하되, 후속 에이전트 근거가 연결되면 "
        f"최종 위원회 의견을 보완하도록 정리했습니다. 현재 서비스 recommendation은 "
        f"{recommendation}입니다."
    )
    findings = [
        "정량 판단은 model_view로 보존하고, committee_view에서는 해석과 보완 의견만 추가합니다.",
        "외부 근거 에이전트가 연결되면 적격/보류/부적격 3단 위원회 의견으로 확장합니다.",
    ]
    return AgentOutput(
        role="chair_investment",
        summary=summary,
        findings=findings,
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


def _join_feature_points(items: list[dict[str, Any]]) -> str:
    points = [str(item.get("detail", "")) for item in items[:3] if item.get("detail")]
    return " / ".join(points)


def _join_signal_text(*, risks: list[str], supports: list[str], fallback: str) -> str:
    points = [*risks[:3], *supports[:2]]
    if points:
        return " ".join(points)
    return fallback


def _debt_liquidity_validation(
    *,
    prediction_label: str,
    liquidity_risks: list[str],
    liquidity_supports: list[str],
    repayment_risks: list[str],
    repayment_supports: list[str],
) -> str:
    risk_count = len(liquidity_risks) + len(repayment_risks)
    support_count = len(liquidity_supports) + len(repayment_supports)

    if prediction_label == "투자적격" and risk_count >= 2:
        return "정량상 투자적격이더라도 단기 유동성과 상환여력에는 추가 경계가 필요합니다."
    if prediction_label == "부적격" and support_count >= 2 and risk_count <= 1:
        return "정량상 부적격 판단은 유지하되, 현금흐름과 상환여력 측면의 완화 신호가 확인됩니다."
    if risk_count > support_count:
        return f"부채 및 유동성 지표는 현재 {prediction_label} 판단을 보수적으로 뒷받침합니다."
    if support_count > risk_count:
        return f"부채 및 유동성 지표는 현재 {prediction_label} 판단에 일부 완충 근거를 제공합니다."
    return "부채 및 유동성 지표는 현재 모델 판단과 대체로 중립적으로 맞물립니다."


def _debt_liquidity_confidence(metrics: list[float | None]) -> float:
    available = sum(value is not None for value in metrics)
    return min(0.78, 0.42 + available * 0.04)


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
