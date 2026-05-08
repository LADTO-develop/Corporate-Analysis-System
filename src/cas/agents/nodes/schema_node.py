"""Validate and emit the strict dashboard response JSON."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from cas.agents.response_schema import DashboardResponse
from cas.agents.state import AgentState, AuditEntry


def run(state: AgentState) -> dict[str, Any]:
    """Generate the service response and validate it against the JSON schema."""
    payload = _build_response_payload(state)
    errors: list[str] = []
    try:
        response = DashboardResponse.model_validate(payload).model_dump(mode="json")
        summary = "Dashboard response JSON validated against strict schema."
    except ValidationError as error:
        response = payload
        errors = [err["msg"] for err in error.errors()]
        summary = f"Dashboard response JSON failed schema validation: {errors}"

    audit = AuditEntry(
        node="json_schema",
        timestamp=_now(),
        summary=summary,
        metrics={"schema_errors": float(len(errors))},
    )
    return {"response_json": response, "json_schema_errors": errors, "audit": [audit]}


def _build_response_payload(state: AgentState) -> dict[str, Any]:
    profile = dict(state.get("company_profile") or {})
    company = dict(profile.get("company") or {})
    xgb = dict(state.get("xgboost_result") or {})
    rule = dict(state.get("rule_result") or {})
    news_cache = dict(state.get("news_cache_snapshot") or {})
    agent_summary = dict(state.get("agent_summary") or {})

    insufficient = bool(state.get("insufficient_data", False))
    company_name = str(state.get("company_name") or company.get("name") or state["company_id"])
    company_summary = str(
        company.get("summary")
        or ("Analysis deferred because required inputs are missing." if insufficient else "")
    )
    return {
        "company_overview": {
            "company_id": str(state.get("company_id", "unknown")),
            "company_name": company_name,
            "market": str(state.get("market") or company.get("market") or "UNKNOWN"),
            "analysis_year": int(state.get("analysis_year") or 0),
            "summary": company_summary,
        },
        "model_result": {
            "model_name": str(xgb.get("model_name", "xgboost_realtime")),
            "model_version": str(xgb.get("model_version", "unavailable")),
            "prediction_label": str(xgb.get("prediction_label", "unknown")),
            "risk_band": str(rule.get("risk_band", xgb.get("risk_band", "insufficient_data"))),
            "probability_speculative": float(xgb.get("probability_speculative", 0.0) or 0.0),
            "top_drivers": _driver_payloads(xgb.get("top_drivers", [])),
            "rule_label": str(rule.get("label", "insufficient_data")),
        },
        "news_analysis": {
            "status": str(news_cache.get("status", "not_implemented")),
            "summary": _news_summary(news_cache, insufficient=insufficient),
        },
        "agent_summary": {
            "final_recommendation": str(
                agent_summary.get("final_recommendation")
                or state.get("final_recommendation")
                or "review"
            ),
            "final_confidence": float(
                agent_summary.get("final_confidence", state.get("final_confidence", 0.0)) or 0.0
            ),
            "synthesis": str(
                agent_summary.get("synthesis")
                or ("Analysis deferred because required inputs are missing." if insufficient else "")
            ),
            "agents": _agent_payloads(agent_summary),
        },
    }


def _driver_payloads(raw_drivers: object) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    if not isinstance(raw_drivers, list):
        return drivers
    for item in raw_drivers:
        if isinstance(item, dict):
            name = str(item.get("name", item.get("feature", "")))
            value = float(item.get("value", item.get("score", 0.0)) or 0.0)
        else:
            name = str(item[0])
            value = float(item[1])
        if name:
            drivers.append({"name": name, "value": value})
    return drivers


def _agent_payloads(agent_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    agents = agent_summary.get("agents")
    if isinstance(agents, dict) and agents:
        return {
            str(role): {
                "summary": str(payload.get("summary", "")),
                "findings": [str(item) for item in payload.get("findings", [])],
                "confidence": float(payload.get("confidence", 0.0) or 0.0),
            }
            for role, payload in agents.items()
            if isinstance(payload, dict)
        }
    return {
        "synthesis_format": {
            "summary": "Analysis deferred before multi-agent synthesis.",
            "findings": ["No agent output was produced."],
            "confidence": 0.0,
        }
    }


def _news_summary(news_cache: dict[str, Any], *, insufficient: bool) -> str:
    if insufficient:
        return "News cache was not queried because required company inputs are missing."
    if news_cache.get("status") == "placeholder":
        return "News/crawling integration is a placeholder only."
    return "News/crawling integration is not implemented."


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
