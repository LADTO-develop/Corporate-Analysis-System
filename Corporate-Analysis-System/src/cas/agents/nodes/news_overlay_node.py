"""Optional external evidence collection node for Stage 2."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

from cas.agents.state import AgentState, AuditEntry
from cas.evidence import collect_external_evidence, external_evidence_enabled


def run(state: AgentState) -> dict[str, Any]:
    """Collect external evidence only when the environment explicitly enables it."""
    if external_evidence_enabled():
        processed_company = dict(state.get("processed_company") or {})
        company_name = str(
            state.get("company_name")
            or processed_company.get("company_name")
            or state.get("company_id", "unknown")
        )
        snapshot = collect_external_evidence(
            company_name=company_name,
            stock_code=str(processed_company.get("stock_code") or state.get("company_id") or ""),
            corp_code=_optional_text(processed_company.get("corp_code")),
            as_of_date=_as_of_date(processed_company, state),
        )
        audit = AuditEntry(
            node="news_cache",
            timestamp=_now(),
            summary=f"External evidence collection completed: {snapshot.get('status', 'unknown')}",
            metrics={"n_evidence_items": float(len(snapshot.get("items", []) or []))},
        )
        return {"news_cache_snapshot": snapshot, "audit": [audit]}

    audit = AuditEntry(
        node="news_cache",
        timestamp=_now(),
        summary=(
            "External evidence collection is disabled; "
            "set CAS_ENABLE_EXTERNAL_EVIDENCE=1 to enable it."
        ),
    )
    return {
        "news_cache_snapshot": {
            "status": "disabled",
            "source": "external_evidence",
            "enabled": False,
            "items": [],
            "has_critical_risk": False,
            "message": "Set CAS_ENABLE_EXTERNAL_EVIDENCE=1 to enable live evidence calls.",
        },
        "audit": [audit],
    }


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_of_date(processed_company: dict[str, Any], state: AgentState) -> str:
    explicit = _optional_text(processed_company.get("as_of_date"))
    if explicit:
        return explicit
    analysis_year = _optional_int(
        processed_company.get("eval_year")
        or processed_company.get("analysis_year")
        or state.get("analysis_year")
    )
    if analysis_year is None:
        return date.today().isoformat()
    candidate = date(analysis_year, 12, 31)
    return min(candidate, date.today()).isoformat()


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if isinstance(value, int | float | str) else None
    except (TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
