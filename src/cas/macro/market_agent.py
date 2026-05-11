"""Interpret current macro-market context for the agent committee."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from cas.macro.schemas import (
    MacroDirection,
    MacroFactorSignal,
    MacroGroupAdjustment,
    MacroMarketAgentOutput,
    MacroMarketContext,
    MacroRiskLevel,
    MacroStance,
)


def evaluate_macro_market_context(context: MacroMarketContext) -> MacroMarketAgentOutput:
    """Create a deterministic MacroMarketAgent opinion from macro context."""
    factor_signals = _build_factor_signals(context)
    risk_score = _macro_risk_score(factor_signals)
    risk_level = _risk_level(risk_score)
    stance = _stance_from_risk(risk_level)
    group_adjustments = _build_group_adjustments(factor_signals)

    risk_delta = round(max(-0.03, min(0.08, risk_score * 0.012)), 4)
    if context.missing_indicators:
        risk_delta = round(min(0.08, risk_delta + 0.01), 4)

    return MacroMarketAgentOutput(
        produced_at=_now(),
        as_of_date=context.as_of_date,
        stance=stance,
        macro_risk_level=risk_level,
        current_context_risk_delta=risk_delta,
        macro_summary_kr=_summary_kr(risk_level, factor_signals, context),
        key_macro_factors=factor_signals,
        group_adjustments=group_adjustments,
        limitations_kr=_limitations(context),
        source_context_refs=_source_refs(context),
        model_handling_note_kr=(
            "이 결과는 XGBoost의 y_proba를 직접 변경하지 않는다. "
            "위원회 단계에서 부채상환능력, 유동성, 수익성 해석의 가중 검토 의견으로만 사용한다."
        ),
    )


def _build_factor_signals(context: MacroMarketContext) -> list[MacroFactorSignal]:
    """Interpret each available macro observation or derived metric."""
    values: dict[str, tuple[str, float, str]] = {}
    for observation in context.observations:
        values[observation.code] = (observation.name_kr, observation.value, observation.unit)
    for metric in context.derived_metrics:
        values[metric.code] = (metric.name_kr, metric.value, metric.unit)

    signals: list[MacroFactorSignal] = []
    for code, (name_kr, value, unit) in values.items():
        signal = _signal_for_code(code, name_kr, value, unit)
        if signal is not None:
            signals.append(signal)
    return signals


def _signal_for_code(
    code: str,
    name_kr: str,
    value: float,
    unit: str,
) -> MacroFactorSignal | None:
    """Return a rule-based interpretation for one macro value."""
    if code == "BaseRate":
        return _make_signal(
            code,
            name_kr,
            value,
            unit,
            ["debt_liquidity", "interest_coverage"],
            high_cut=3.25,
            moderate_cut=2.75,
            interpretation_prefix="정책금리가 높을수록 차입비용과 이자보상 부담이 커진다.",
        )
    if code == "3YRate_CorpBond_BBB":
        return _make_signal(
            code,
            name_kr,
            value,
            unit,
            ["debt_liquidity", "market_refinancing"],
            high_cut=9.5,
            moderate_cut=8.0,
            interpretation_prefix="BBB- 회사채 금리는 저신용 기업의 시장 조달비용을 직접 반영한다.",
        )
    if code == "Spread_Credit":
        return _make_signal(
            code,
            name_kr,
            value,
            unit,
            ["market_refinancing", "debt_liquidity"],
            high_cut=0.8,
            moderate_cut=0.55,
            interpretation_prefix="신용스프레드 확대는 회사채 시장의 위험 프리미엄 상승을 뜻한다.",
        )
    if code == "Spread_Quality":
        return _make_signal(
            code,
            name_kr,
            value,
            unit,
            ["market_refinancing", "credit_boundary"],
            high_cut=5.0,
            moderate_cut=4.0,
            interpretation_prefix="투기경계 스프레드 확대는 낮은 등급 기업에 대한 회피 심리를 뜻한다.",
        )
    if code == "USDKRW":
        return _make_signal(
            code,
            name_kr,
            value,
            unit,
            ["cost_pressure", "fx_exposure"],
            high_cut=1450.0,
            moderate_cut=1350.0,
            interpretation_prefix="원/달러 환율 상승은 수입원가와 외화부채 부담을 키울 수 있다.",
        )
    if code == "PPI":
        return _make_signal(
            code,
            name_kr,
            value,
            unit,
            ["cost_pressure", "profitability_cashflow"],
            high_cut=130.0,
            moderate_cut=124.0,
            interpretation_prefix="생산자물가 수준이 높으면 원가 부담과 마진 압박을 점검해야 한다.",
        )
    return None


def _make_signal(
    code: str,
    name_kr: str,
    value: float,
    unit: str,
    affected_groups: list[str],
    *,
    high_cut: float,
    moderate_cut: float,
    interpretation_prefix: str,
) -> MacroFactorSignal:
    """Build a factor signal using high and moderate risk thresholds."""
    if value >= high_cut:
        severity: MacroRiskLevel = "high"
        direction: MacroDirection = "downgrade"
        suffix = "현재 값은 높은 위험 구간이다."
    elif value >= moderate_cut:
        severity = "moderate"
        direction = "downgrade"
        suffix = "현재 값은 주의가 필요한 구간이다."
    else:
        severity = "low"
        direction = "neutral"
        suffix = "현재 값만으로는 강한 하향 근거로 보기 어렵다."

    return MacroFactorSignal(
        code=code,
        name_kr=name_kr,
        value=round(value, 6),
        unit=unit,
        direction=direction,
        severity=severity,
        affected_variable_groups=affected_groups,
        interpretation_kr=f"{interpretation_prefix} {suffix}",
        source_ref=f"macro_context:{code}",
    )


def _macro_risk_score(signals: list[MacroFactorSignal]) -> float:
    """Aggregate factor severities into a compact macro-risk score."""
    severity_score: dict[MacroRiskLevel, float] = {
        "low": 0.0,
        "moderate": 1.0,
        "high": 2.0,
        "very_high": 3.0,
    }
    score = 0.0
    for signal in signals:
        multiplier = 1.0 if signal.direction == "downgrade" else -0.25
        score += severity_score[signal.severity] * multiplier
    return max(-1.0, score)


def _risk_level(score: float) -> MacroRiskLevel:
    """Map the aggregate risk score to a macro-risk level."""
    if score >= 7.0:
        return "very_high"
    if score >= 4.0:
        return "high"
    if score >= 1.5:
        return "moderate"
    return "low"


def _stance_from_risk(risk_level: MacroRiskLevel) -> MacroStance:
    """Map macro-risk level to an agent stance label."""
    if risk_level == "very_high":
        return "stress"
    if risk_level == "high":
        return "cautious"
    if risk_level == "moderate":
        return "neutral"
    return "supportive"


def _build_group_adjustments(
    signals: list[MacroFactorSignal],
) -> list[MacroGroupAdjustment]:
    """Convert factor signals into committee weight-guidance by variable group."""
    group_refs: defaultdict[str, list[str]] = defaultdict(list)
    group_scores: defaultdict[str, float] = defaultdict(float)
    for signal in signals:
        for group in signal.affected_variable_groups:
            group_refs[group].append(signal.source_ref)
            if signal.severity == "high":
                group_scores[group] += 2.0
            elif signal.severity == "moderate":
                group_scores[group] += 1.0

    adjustments: list[MacroGroupAdjustment] = []
    for group, refs in sorted(group_refs.items()):
        score = group_scores[group]
        direction: MacroDirection = "downgrade" if score >= 1.0 else "neutral"
        multiplier = round(1.0 + min(0.15, score * 0.03), 4)
        rationale = _group_rationale(group, direction, multiplier)
        adjustments.append(
            MacroGroupAdjustment(
                variable_group=group,
                direction=direction,
                weight_multiplier=multiplier,
                rationale_kr=rationale,
                evidence_refs=sorted(set(refs)),
            )
        )
    return adjustments


def _group_rationale(
    group: str,
    direction: MacroDirection,
    multiplier: float,
) -> str:
    """Create a Korean rationale for one variable-group adjustment."""
    if direction == "neutral":
        return f"{group} 그룹은 현재 거시지표만으로 추가 가중을 크게 줄 필요가 없다."
    return f"{group} 그룹은 현재 거시환경에서 더 중요하게 검토해야 하므로 가중 참고값을 {multiplier:.2f}로 둔다."


def _summary_kr(
    risk_level: MacroRiskLevel,
    signals: list[MacroFactorSignal],
    context: MacroMarketContext,
) -> str:
    """Create a concise Korean macro-market summary."""
    risky = [signal for signal in signals if signal.direction == "downgrade"]
    if not risky:
        return (
            f"{context.as_of_date} 기준 ECOS 거시지표는 전반적으로 급격한 하향 신호가 약하다. "
            "다만 기업별 환율 민감도와 차입 만기구조는 별도로 확인해야 한다."
        )
    top_factors = ", ".join(signal.name_kr for signal in risky[:3])
    return (
        f"{context.as_of_date} 기준 거시위험 수준은 {risk_level}로 판단된다. "
        f"주요 점검 요인은 {top_factors}이며, 부채상환능력과 시장 조달환경 해석에 반영해야 한다."
    )


def _limitations(context: MacroMarketContext) -> list[str]:
    """Return known limitations of the macro agent output."""
    limitations = [
        "ECOS 지표는 과거에 공표된 최신 관측값이며 실시간 시장 호가와 다를 수 있다.",
        "생산자물가지수는 수준값만으로 해석하며 전년동월대비 증가율은 별도 계산하지 않았다.",
        "기업별 외화부채, 수출입 비중, 변동금리 차입 비중이 없으면 환율·금리 민감도는 정성 의견에 머문다.",
    ]
    if context.missing_indicators:
        limitations.append("일부 지표 수집 실패가 있어 거시 해석 신뢰도가 낮아질 수 있다.")
    if context.stale_indicators:
        limitations.append("일부 지표의 최신 관측시차가 길어 공표 지연 여부를 확인해야 한다.")
    return limitations


def _source_refs(context: MacroMarketContext) -> list[str]:
    """Return context references used by the macro agent."""
    observation_refs = [f"macro_context:{item.code}" for item in context.observations]
    derived_refs = [f"macro_context:{item.code}" for item in context.derived_metrics]
    return [*observation_refs, *derived_refs]


def _now() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
