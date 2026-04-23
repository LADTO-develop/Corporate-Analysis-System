"""Load a local company profile and validate the minimum input shape."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cas.agents.state import AgentState, AuditEntry
from cas.utils.io import read_yaml
from cas.utils.logging import get_logger

logger = get_logger(__name__)

_PROFILE_ROOT = Path("data/input/companies")
_REQUIRED_FINANCIALS = {
    "revenue_growth_pct",
    "operating_margin_pct",
    "debt_to_equity",
    "current_ratio",
    "free_cash_flow_margin_pct",
    "interest_coverage",
}
_REQUIRED_QUALITATIVE = {"governance_score", "product_momentum_score"}


def run(state: AgentState) -> dict[str, Any]:
    """Load a company profile from ``data/input/companies``."""
    company_id = state["company_id"]
    profile_path = _PROFILE_ROOT / f"{company_id}.yaml"
    logger.info("data_node_run", company_id=company_id, path=str(profile_path))

    if not profile_path.exists():
        audit = AuditEntry(
            node="data",
            timestamp=_now(),
            summary=f"Company profile not found: {profile_path}",
        )
        return {"insufficient_data": True, "audit": [audit]}

    profile = read_yaml(profile_path)
    company = profile.get("company", {})
    financials = profile.get("financials", {})
    qualitative = profile.get("qualitative", {})
    missing = sorted(
        [
            *(key for key in _REQUIRED_FINANCIALS if key not in financials),
            *(key for key in _REQUIRED_QUALITATIVE if key not in qualitative),
        ]
    )
    if missing:
        audit = AuditEntry(
            node="data",
            timestamp=_now(),
            summary=f"Missing required input fields: {missing}",
            metrics={"missing_fields": float(len(missing))},
        )
        return {"insufficient_data": True, "audit": [audit]}

    audit = AuditEntry(
        node="data",
        timestamp=_now(),
        summary=f"Loaded company profile for {company.get('name', company_id)}",
        metrics={"n_financial_fields": float(len(financials))},
    )
    return {
        "company_name": company.get("name", company_id),
        "market": company.get("market", state.get("market", "UNKNOWN")),
        "analysis_year": int(profile.get("analysis_year") or state.get("analysis_year") or 0),
        "company_profile": profile,
        "raw_financials": financials,
        "insufficient_data": False,
        "audit": [audit],
    }


def has_enough_data(state: AgentState) -> Literal["enough", "insufficient"]:
    """Conditional-edge predicate referenced by the graph config."""
    return "insufficient" if state.get("insufficient_data") else "enough"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
