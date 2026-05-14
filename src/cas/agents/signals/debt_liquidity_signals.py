"""Debt, liquidity, and cash-flow signal extraction for EvidenceAuditAgent."""

from __future__ import annotations

from dataclasses import dataclass

from cas.agents.stage2_bundle import Stage2InputBundle


@dataclass(frozen=True)
class DebtLiquiditySignals:
    """Debt and liquidity findings used by EvidenceAuditAgent."""

    summary: str
    findings: list[str]
    confidence: float


def evaluate_debt_liquidity(bundle: Stage2InputBundle) -> DebtLiquiditySignals:
    """Evaluate debt repayment capacity and short-term liquidity signals."""
    source_row = bundle.source_feature_row
    prediction_label = bundle.prediction_label

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

    summary = f"부채상환능력과 유동성 관점에서는 1단계 {prediction_label} 판단에 대해 {validation}"
    findings = [
        f"부채·유동성 검증 의견: {validation}",
        "단기 방어력 점검: "
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

    confidence = _debt_liquidity_confidence(
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
    )
    return DebtLiquiditySignals(summary=summary, findings=findings, confidence=confidence)


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


def _format_number(value: float | None, unit: str) -> str:
    if value is None:
        return "값 없음"
    if unit == "%p":
        return f"{value:.2f}%p"
    if unit == "ratio":
        return f"{value:.3f}"
    return f"{value:.3f}"


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if not isinstance(value, int | float | str):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
