"""Macro overlay node — adjusts base P(borderline) using ECOS context.

The adjustment is intentionally small and interpretable: a firm whose base
risk probability sits near the decision boundary can be nudged by a widening
credit spread or a compressed real rate, but the macro signal cannot by
itself flip a confidently-healthy firm to borderline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bfd.agents.state import AgentState, AuditEntry, MacroOverlay
from bfd.data.alignment import as_of_ecos_snapshot
from bfd.data.loaders.ecos import ECOSClient
from bfd.utils.logging import get_logger
from bfd.utils.time import fiscal_year_end

logger = get_logger(__name__)


def run(state: AgentState) -> dict[str, Any]:
    """Compute macro features and derive an adjustment to ensemble_proba."""
    fiscal_year = state["fiscal_year"]
    try:
        client = ECOSClient()
    except RuntimeError as exc:
        audit = AuditEntry(
            node="macro_overlay",
            timestamp=_now(),
            summary=f"ECOS client unavailable: {exc}. Skipping macro overlay.",
        )
        return {"macro_overlay": MacroOverlay().model_dump(), "audit": [audit]}

    fy_end = fiscal_year_end(fiscal_year)
    start = f"{fiscal_year - 1}0101"
    end = fy_end.strftime("%Y%m%d")

    try:
        series = client.fetch_all_configured(start=start, end=end)
    except Exception as exc:  # noqa: BLE001
        audit = AuditEntry(
            node="macro_overlay",
            timestamp=_now(),
            summary=f"ECOS fetch failed ({exc}); skipping macro overlay.",
        )
        return {"macro_overlay": MacroOverlay().model_dump(), "audit": [audit]}

    snapshots: dict[str, float | None] = {
        name: as_of_ecos_snapshot(df, fiscal_year) for name, df in series.items()
    }

    credit_spread = _safe_diff(
        snapshots.get("corp_bond_3y_bbb_minus"),
        snapshots.get("corp_bond_3y_aa_minus"),
    )
    real_rate = _safe_diff(snapshots.get("base_rate"), snapshots.get("cpi"))

    # Heuristic adjustment — bounded in [-0.05, +0.05].
    # Positive adjustment increases P(borderline).
    adjustment = 0.0
    rationale: list[str] = []
    if credit_spread is not None and credit_spread > 2.0:
        adjustment += 0.02
        rationale.append(f"BBB-/AA- 스프레드 확대({credit_spread:.2f}%p)")
    if real_rate is not None and real_rate > 3.0:
        adjustment += 0.01
        rationale.append(f"실질금리 상승({real_rate:.2f}%p)")

    adjustment = float(max(-0.05, min(0.05, adjustment)))

    overlay = MacroOverlay(
        credit_spread=credit_spread,
        real_rate=real_rate,
        fx_volatility_30d=snapshots.get("usd_krw"),  # raw level; volatility calc done upstream
        adjustment=adjustment,
        rationale_ko="; ".join(rationale) or "특이 거시 이벤트 없음",
    )

    audit = AuditEntry(
        node="macro_overlay",
        timestamp=_now(),
        summary=f"Macro adjustment={adjustment:+.3f} ({overlay.rationale_ko})",
        metrics={"macro_adjustment": adjustment},
    )
    return {"macro_overlay": overlay.model_dump(), "audit": [audit]}


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
