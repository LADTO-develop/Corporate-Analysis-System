"""Report node — assembles the final audited report for the firm."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from bfd.agents.state import AgentState, AuditEntry
from bfd.reporting.export import render_report
from bfd.utils.io import ensure_dir, write_json
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


def run(state: AgentState) -> dict[str, Any]:
    """Write the final report JSON + markdown to ``data/processed/reports/``."""
    corp_code = state.get("corp_code", "unknown")
    fiscal_year = state.get("fiscal_year", 0)

    report_dir = Path("data/processed/reports") / corp_code
    ensure_dir(report_dir)

    payload = render_report(state)
    json_path = report_dir / f"{fiscal_year}.json"
    md_path = report_dir / f"{fiscal_year}.md"

    write_json(payload, json_path)
    md_path.write_text(payload["markdown"], encoding="utf-8")

    audit = AuditEntry(
        node="report",
        timestamp=_now(),
        summary=(
            f"Report written to {json_path}. "
            f"final_verdict={state.get('final_verdict', 'n/a')}, "
            f"insufficient_data={state.get('insufficient_data', False)}"
        ),
        payload_ref=str(json_path),
    )
    return {
        "audit": [audit],
        "artifacts": {"report_json": str(json_path), "report_md": str(md_path)},
    }


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
