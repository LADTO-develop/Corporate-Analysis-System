"""Render a final markdown report from the strict dashboard response."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState
from cas.reporting.audit_trail import to_markdown as audit_to_md


def render_report(state: AgentState | dict[str, Any]) -> dict[str, Any]:
    """Build a markdown companion for the validated response JSON."""
    s = dict(state)
    response = dict(s.get("response_json") or _fallback_response(s))
    company = dict(response.get("company_overview") or {})
    model_result = dict(response.get("model_result") or {})
    news = dict(response.get("news_analysis") or {})
    agent_summary = dict(response.get("agent_summary") or {})
    committee_view = dict(response.get("committee_view") or {})
    audit = s.get("audit") or []
    schema_errors = [str(error) for error in s.get("json_schema_errors", [])]

    md_lines = [
        f"# Corporate Review: {company.get('company_name', s.get('company_id', '?'))}",
        "",
        f"- **Company ID**: `{company.get('company_id', s.get('company_id', '?'))}`",
        f"- **Market**: {company.get('market', '?')}",
        f"- **Analysis Year**: {company.get('analysis_year', '?')}",
        (
            f"- **Generated At**: "
            f"{datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}"
        ),
        "",
        "## Dashboard JSON",
        "",
        f"- **Recommendation**: `{agent_summary.get('final_recommendation', 'review')}`",
        f"- **Confidence**: {float(agent_summary.get('final_confidence', 0.0) or 0.0):.3f}",
        f"- **Risk Band**: `{model_result.get('risk_band', 'insufficient_data')}`",
        (
            f"- **Speculative Probability**: "
            f"{float(model_result.get('probability_speculative', 0.0) or 0.0):.3f}"
        ),
        "",
        "## Model Result",
        "",
        f"- **Model**: {model_result.get('model_name', 'n/a')} "
        f"({model_result.get('model_version', 'n/a')})",
        f"- **Prediction Label**: `{model_result.get('prediction_label', 'unknown')}`",
        f"- **Rule Label**: `{model_result.get('rule_label', 'unknown')}`",
        "",
        "### Top Drivers",
        "",
    ]

    drivers = model_result.get("top_drivers", []) or []
    if drivers:
        for driver in drivers:
            md_lines.append(f"- `{driver.get('name', '')}`: {float(driver.get('value', 0.0)):.3f}")
    else:
        md_lines.append("_(No model drivers)_")
    md_lines += [
        "",
        "## News Analysis",
        "",
        f"- **Status**: `{news.get('status', 'not_implemented')}`",
        f"- **Summary**: {news.get('summary', '')}",
        "",
        "## Committee View",
        "",
        f"- **Final Committee Label**: `{committee_view.get('final_committee_label', '보류')}`",
        f"- **Veto Triggered**: `{bool(committee_view.get('veto_triggered', False))}`",
        f"- **Hidden Tail Risk Flag**: `{bool(committee_view.get('hidden_tail_risk_flag', False))}`",
        f"- **Hidden Tail Risk Reason**: {committee_view.get('hidden_tail_risk_reason', '')}",
        f"- **Conflict Resolution**: {committee_view.get('conflict_resolution', '')}",
        f"- **Final Review Memo**: {committee_view.get('final_review_memo', '')}",
        "",
    ]

    risk_factors = [str(item) for item in committee_view.get("key_risk_factors", []) or []]
    mitigating_factors = [str(item) for item in committee_view.get("mitigating_factors", []) or []]
    if risk_factors:
        md_lines += ["### Key Risk Factors", "", *[f"- {item}" for item in risk_factors], ""]
    if mitigating_factors:
        md_lines += [
            "### Mitigating Factors",
            "",
            *[f"- {item}" for item in mitigating_factors],
            "",
        ]

    md_lines += [
        "## Agent Summary",
        "",
        str(agent_summary.get("synthesis", "")),
        "",
    ]
    agents = dict(agent_summary.get("agents") or {})
    if agents:
        md_lines += ["| Agent | Confidence | Summary |", "|---|---:|---|"]
        for role, payload in agents.items():
            payload_dict = dict(payload)
            summary = str(payload_dict.get("summary", "")).replace("|", r"\|")
            confidence = float(payload_dict.get("confidence", 0.0) or 0.0)
            md_lines.append(f"| `{role}` | {confidence:.3f} | {summary} |")
        md_lines.append("")

    if schema_errors:
        md_lines += [
            "## Schema Errors",
            "",
            *[f"- {error}" for error in schema_errors],
            "",
        ]

    md_lines += [
        "## Audit Trail",
        "",
        audit_to_md(audit),
        "",
        "---",
        "_This report is a decision-support artifact, not an automatic investment decision._",
    ]

    return {**response, "markdown": "\n".join(md_lines)}


def _fallback_response(state: dict[str, Any]) -> dict[str, Any]:
    company_id = str(state.get("company_id", "unknown"))
    return {
        "company_overview": {
            "company_id": company_id,
            "company_name": str(state.get("company_name", company_id)),
            "market": str(state.get("market", "UNKNOWN")),
            "analysis_year": int(state.get("analysis_year", 0) or 0),
            "summary": "",
        },
        "model_result": {
            "model_name": "xgboost_realtime",
            "model_version": "unavailable",
            "prediction_label": "unknown",
            "risk_band": "insufficient_data",
            "probability_speculative": 0.0,
            "top_drivers": [],
            "rule_label": "insufficient_data",
        },
        "news_analysis": {
            "status": "not_implemented",
            "summary": "No response JSON was generated.",
        },
        "agent_summary": {
            "final_recommendation": str(state.get("final_recommendation", "review")),
            "final_confidence": float(state.get("final_confidence", 0.0) or 0.0),
            "synthesis": "No agent summary was generated.",
            "agents": {},
        },
        "committee_view": {
            "final_committee_label": "보류",
            "veto_triggered": False,
            "hidden_tail_risk_flag": False,
            "hidden_tail_risk_reason": "",
            "conflict_resolution": "No committee_view was generated.",
            "key_risk_factors": [],
            "mitigating_factors": [],
            "evidence_summary": [],
            "final_review_memo": "No committee_view was generated.",
        },
    }
