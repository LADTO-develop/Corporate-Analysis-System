"""Data ingestion node.

Loads financials for the requested (corp_code, fiscal_year), performs
a minimum-data sanity check, and records an audit entry.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from bfd.agents.state import AgentState, AuditEntry
from bfd.data.loaders.ts2000 import TS2000Loader
from bfd.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum required columns to go forward — the downstream feature functions
# will fail without these.
_REQUIRED_COLUMNS = {
    "total_assets",
    "total_liabilities",
    "total_equity",
    "revenue",
    "operating_income",
    "interest_expense",
    "cfo",
}


def run(state: AgentState) -> dict[str, Any]:
    """Node entry point."""
    corp_code = state["corp_code"]
    fiscal_year = state["fiscal_year"]
    logger.info("data_node_run", corp_code=corp_code, fiscal_year=fiscal_year)

    loader = TS2000Loader()
    wide = loader.load_wide(year=fiscal_year)
    firm = wide.loc[wide["corp_code"] == corp_code]

    if firm.empty:
        audit = AuditEntry(
            node="data",
            timestamp=_now(),
            summary=f"No TS2000 record found for {corp_code} / FY{fiscal_year}",
            metrics={"rows": 0.0},
        )
        return {"insufficient_data": True, "audit": [audit]}

    # Keep annual-consolidated row
    firm = firm[
        (firm["fiscal_quarter"] == 4) & (firm["report_type"] == "consolidated")
    ].reset_index(drop=True)

    if firm.empty:
        audit = AuditEntry(
            node="data",
            timestamp=_now(),
            summary=f"No annual consolidated row for {corp_code} / FY{fiscal_year}",
            metrics={"rows": 0.0},
        )
        return {"insufficient_data": True, "audit": [audit]}

    row = firm.iloc[0].to_dict()
    missing = [c for c in _REQUIRED_COLUMNS if c not in row or row[c] is None]
    if missing:
        audit = AuditEntry(
            node="data",
            timestamp=_now(),
            summary=f"Missing required columns: {missing}",
            metrics={"n_missing": float(len(missing))},
        )
        return {"insufficient_data": True, "audit": [audit], "raw_financials": row}

    audit = AuditEntry(
        node="data",
        timestamp=_now(),
        summary=f"Loaded TS2000 annual consolidated for {corp_code} / FY{fiscal_year}",
        metrics={"n_fields": float(len(row))},
    )
    return {
        "raw_financials": row,
        "insufficient_data": False,
        "audit": [audit],
    }


def has_enough_data(state: AgentState) -> Literal["enough", "insufficient"]:
    """Conditional-edge predicate referenced by configs/agent/graph.yaml."""
    return "insufficient" if state.get("insufficient_data") else "enough"


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
