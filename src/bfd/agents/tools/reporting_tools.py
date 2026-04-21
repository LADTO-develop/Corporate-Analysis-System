"""Reporting-layer tools exposed to agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from bfd.agents.state import AuditEntry


@tool
def build_audit_entry(
    node: str,
    summary: str,
    metrics: dict[str, float] | None = None,
    payload_ref: str | None = None,
) -> dict[str, Any]:
    """Construct a validated audit entry as a plain dict.

    Args:
        node: one of the pipeline's node names.
        summary: one-line human-readable summary.
        metrics: optional numeric metrics to log alongside.
        payload_ref: optional path or URI pointing to a larger artifact.
    """
    entry = AuditEntry(
        node=node,  # type: ignore[arg-type]
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        summary=summary,
        metrics=metrics or {},
        payload_ref=payload_ref,
    )
    return entry.model_dump()


@tool
def render_final_report(state: dict[str, Any]) -> dict[str, Any]:
    """Render the final JSON + markdown report from an ``AgentState`` dict."""
    from bfd.reporting.export import render_report

    return render_report(state)  # type: ignore[arg-type]
