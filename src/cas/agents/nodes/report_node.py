"""Write the final report for the current company."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cas.agents.state import AgentState, AuditEntry
from cas.reporting.export import render_report
from cas.utils.io import ensure_dir, write_json


def run(state: AgentState) -> dict[str, Any]:
    """Write the final report JSON and markdown to ``data/outputs/reports``."""
    company_id = state.get("company_id", "unknown")
    report_dir = Path("data/outputs/reports") / company_id
    ensure_dir(report_dir)

    payload = render_report(state)
    json_path = report_dir / "latest.json"
    md_path = report_dir / "latest.md"

    write_json(payload, json_path)
    md_path.write_text(payload["markdown"], encoding="utf-8")

    audit = AuditEntry(
        node="report",
        timestamp=_now(),
        summary=(
            f"Report written to {json_path}. "
            f"final_recommendation={state.get('final_recommendation', 'n/a')}, "
            f"insufficient_data={state.get('insufficient_data', False)}"
        ),
        payload_ref=str(json_path),
    )
    return {
        "audit": [audit],
        "artifacts": {"report_json": str(json_path), "report_md": str(md_path)},
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
