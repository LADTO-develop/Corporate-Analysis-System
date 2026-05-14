"""Macro and market-context signal extraction for EvidenceAuditAgent."""

from __future__ import annotations

from dataclasses import dataclass

from cas.agents.stage2_bundle import Stage2InputBundle


@dataclass(frozen=True)
class MacroMarketSignals:
    """Macro and market findings used by EvidenceAuditAgent."""

    findings: list[str]


def evaluate_macro_market(bundle: Stage2InputBundle) -> MacroMarketSignals:
    """Evaluate currently available macro and market context signals."""
    source_row = bundle.source_feature_row
    spec_spread = _safe_float(source_row.get("spec_spread"))
    market = _humanize_category(source_row.get("market"), fallback=bundle.market)

    findings = [
        f"거시·시장 점검: 현재 {market} 시장 기준으로 즉시 연결 가능한 거시 변수는 투기경계 스프레드입니다.",
        f"투기경계 스프레드는 {_format_number(spec_spread, '%p')}입니다."
        if spec_spread is not None
        else "현재 source row에는 투기경계 스프레드 값이 없습니다.",
        "추후 금리, 환율, 회사채 스프레드 묶음을 연결하면 거시 해석을 확장할 수 있습니다.",
    ]
    return MacroMarketSignals(findings=findings)


def _humanize_category(value: object, *, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    raw = str(value)
    if not raw:
        return fallback
    return raw


def _format_number(value: float | None, unit: str) -> str:
    if value is None:
        return "값 없음"
    if unit == "%p":
        return f"{value:.2f}%p"
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
