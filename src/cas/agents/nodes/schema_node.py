"""Validate and emit the strict dashboard response JSON."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from pydantic import ValidationError

from cas.agents.response_schema import DashboardResponse
from cas.agents.state import AgentState, AuditEntry
from cas.veto_rules import external_evidence_veto_triggered


def run(state: AgentState) -> dict[str, Any]:
    """Generate the service response and validate it against the JSON schema."""
    payload = _build_response_payload(state)
    errors: list[str] = []
    try:
        response = DashboardResponse.model_validate(payload).model_dump(mode="json")
        summary = "Dashboard response JSON validated against strict schema."
    except ValidationError as error:
        errors = [err["msg"] for err in error.errors()]
        response = _build_schema_failure_response(state, payload, errors)
        summary = (
            "Dashboard response JSON failed schema validation; "
            "emitted strict fallback response instead."
        )

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
    committee_view = dict(state.get("committee_view") or {})
    processed_company = dict(state.get("processed_company") or {})

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
            "summary": _news_summary(
                news_cache,
                insufficient=insufficient,
                company_name=company_name,
                stock_code=str(processed_company.get("stock_code") or state.get("company_id", "")),
            ),
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
                or (
                    "Analysis deferred because required inputs are missing." if insufficient else ""
                )
            ),
            "agents": _agent_payloads(agent_summary),
            "runtime": _runtime_payload(
                agent_summary.get("runtime") or state.get("stage2_runtime_diagnostics")
            ),
        },
        "committee_view": _committee_view_payload(
            committee_view,
            prediction_label=str(xgb.get("prediction_label", "unknown")),
            insufficient=insufficient,
        ),
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


def _runtime_payload(raw_runtime: object) -> dict[str, Any]:
    if not isinstance(raw_runtime, dict):
        return {}
    return _json_safe_dict(raw_runtime)


def _json_safe_dict(value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, raw_value in value.items():
        output[str(key)] = _json_safe_value(raw_value)
    return output


def _json_safe_value(value: object) -> object:
    if isinstance(value, dict):
        return _json_safe_dict(value)
    if isinstance(value, list | tuple | set):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _committee_view_payload(
    committee_view: dict[str, Any],
    *,
    prediction_label: str,
    insufficient: bool,
) -> dict[str, Any]:
    if committee_view:
        return {
            "final_committee_label": str(committee_view.get("final_committee_label", "보류")),
            "committee_decision_type": str(
                committee_view.get("committee_decision_type", "review_hold")
            ),
            "committee_decision_type_label": str(
                committee_view.get("committee_decision_type_label", "확인필요 보류")
            ),
            "committee_risk_signal": bool(committee_view.get("committee_risk_signal", True)),
            "risk_hold_reason_tags": [
                str(item) for item in committee_view.get("risk_hold_reason_tags", []) or []
            ],
            "risk_hold_reason_labels": [
                str(item) for item in committee_view.get("risk_hold_reason_labels", []) or []
            ],
            "risk_hold_reason_summary": str(committee_view.get("risk_hold_reason_summary", "")),
            "agent_disagreement_score": _clamp_probability(
                committee_view.get("agent_disagreement_score", 0.0)
            ),
            "agent_disagreement_level": _agent_disagreement_level(
                committee_view.get("agent_disagreement_level")
            ),
            "agent_disagreement_reasons": [
                str(item) for item in committee_view.get("agent_disagreement_reasons", []) or []
            ],
            "agent_disagreement_summary": str(committee_view.get("agent_disagreement_summary", "")),
            "veto_triggered": bool(committee_view.get("veto_triggered", False)),
            "hidden_tail_risk_flag": bool(committee_view.get("hidden_tail_risk_flag", False)),
            "hidden_tail_risk_reason": str(committee_view.get("hidden_tail_risk_reason", "")),
            "conflict_resolution": str(committee_view.get("conflict_resolution", "")),
            "key_risk_factors": [
                str(item) for item in committee_view.get("key_risk_factors", []) or []
            ],
            "mitigating_factors": [
                str(item) for item in committee_view.get("mitigating_factors", []) or []
            ],
            "evidence_summary": _evidence_items(committee_view.get("evidence_summary", [])),
            "final_review_memo": str(committee_view.get("final_review_memo", "")),
        }
    if insufficient:
        label = "보류"
        memo = "필수 입력이 부족해 위원회 검토를 보류했습니다."
    else:
        label = "적격" if prediction_label == "투자적격" else "부적격"
        memo = "Stage 2 committee_view가 생성되지 않아 model_view 기준 라벨만 반영했습니다."
    return {
        "final_committee_label": label,
        "committee_decision_type": "review_hold"
        if label == "보류"
        else "eligible"
        if label == "적격"
        else "reject",
        "committee_decision_type_label": "확인필요 보류" if label == "보류" else label,
        "committee_risk_signal": label == "부적격",
        "risk_hold_reason_tags": [],
        "risk_hold_reason_labels": [],
        "risk_hold_reason_summary": "",
        "agent_disagreement_score": 0.0,
        "agent_disagreement_level": "low",
        "agent_disagreement_reasons": [],
        "agent_disagreement_summary": "",
        "veto_triggered": False,
        "hidden_tail_risk_flag": False,
        "hidden_tail_risk_reason": "",
        "conflict_resolution": memo,
        "key_risk_factors": [],
        "mitigating_factors": [],
        "evidence_summary": [],
        "final_review_memo": memo,
    }


def _evidence_items(raw_items: object) -> list[dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "source": str(item.get("source", "unknown")),
                "summary": str(item.get("summary", "")),
                "reliability": str(item.get("reliability", "unknown")),
            }
        )
    return items


def _build_schema_failure_response(
    state: AgentState,
    payload: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    company_overview = dict(payload.get("company_overview") or {})
    model_result = dict(payload.get("model_result") or {})
    agent_summary = dict(payload.get("agent_summary") or {})
    committee_view = dict(payload.get("committee_view") or {})
    first_error = errors[0] if errors else "unknown schema validation error"
    fallback_payload = {
        "company_overview": {
            "company_id": str(
                company_overview.get("company_id") or state.get("company_id", "unknown")
            ),
            "company_name": str(
                company_overview.get("company_name")
                or state.get("company_name")
                or state.get("company_id", "unknown")
            ),
            "market": str(company_overview.get("market") or state.get("market", "UNKNOWN")),
            "analysis_year": int(
                company_overview.get("analysis_year") or state.get("analysis_year") or 0
            ),
            "summary": "A strict fallback response was generated after schema validation failed.",
        },
        "model_result": {
            "model_name": str(model_result.get("model_name", "xgboost_realtime")),
            "model_version": str(model_result.get("model_version", "unavailable")),
            "prediction_label": str(model_result.get("prediction_label", "unknown")),
            "risk_band": "insufficient_data",
            "probability_speculative": _clamp_probability(
                model_result.get("probability_speculative", 0.0)
            ),
            "top_drivers": [],
            "rule_label": "schema_validation_failed",
        },
        "news_analysis": {
            "status": "schema_validation_failed",
            "summary": (
                "A fallback response was emitted because the generated payload failed "
                f"strict schema validation: {first_error}"
            ),
        },
        "agent_summary": {
            "final_recommendation": str(
                agent_summary.get("final_recommendation")
                or state.get("final_recommendation")
                or "review"
            ),
            "final_confidence": _clamp_probability(
                agent_summary.get("final_confidence", state.get("final_confidence", 0.0))
            ),
            "synthesis": (
                "The generated payload did not satisfy the strict dashboard schema, "
                "so a safe fallback response was emitted."
            ),
            "agents": {
                "synthesis_format": {
                    "summary": "Strict fallback response generated.",
                    "findings": [f"Schema validation error: {first_error}"],
                    "confidence": 0.0,
                }
            },
            "runtime": _runtime_payload(
                agent_summary.get("runtime") or state.get("stage2_runtime_diagnostics")
            ),
        },
        "committee_view": {
            "final_committee_label": str(committee_view.get("final_committee_label") or "보류"),
            "committee_decision_type": "review_hold",
            "committee_decision_type_label": "확인필요 보류",
            "committee_risk_signal": False,
            "risk_hold_reason_tags": [],
            "risk_hold_reason_labels": [],
            "risk_hold_reason_summary": "",
            "agent_disagreement_score": _clamp_probability(
                committee_view.get("agent_disagreement_score", 0.0)
            ),
            "agent_disagreement_level": _agent_disagreement_level(
                committee_view.get("agent_disagreement_level")
            ),
            "agent_disagreement_reasons": [
                str(item) for item in committee_view.get("agent_disagreement_reasons", []) or []
            ],
            "agent_disagreement_summary": str(committee_view.get("agent_disagreement_summary", "")),
            "veto_triggered": bool(committee_view.get("veto_triggered", False)),
            "hidden_tail_risk_flag": False,
            "hidden_tail_risk_reason": "",
            "conflict_resolution": (
                "A fallback committee_view was emitted because strict schema "
                f"validation failed: {first_error}"
            ),
            "key_risk_factors": [
                f"Schema validation error: {first_error}",
            ],
            "mitigating_factors": [],
            "evidence_summary": [],
            "final_review_memo": (
                "The generated payload did not satisfy the strict dashboard schema, "
                "so committee_view was reduced to a safe fallback."
            ),
        },
    }
    validated = DashboardResponse.model_validate(fallback_payload)
    return cast(dict[str, Any], validated.model_dump(mode="json"))


def _news_summary(
    news_cache: dict[str, Any],
    *,
    insufficient: bool,
    company_name: str = "",
    stock_code: str = "",
) -> str:
    if insufficient:
        return "필수 기업 입력값이 부족하여 외부 뉴스·공시 수집을 수행하지 않았습니다."
    status = str(news_cache.get("status", "not_implemented"))
    if status == "disabled":
        return "외부 뉴스·공시 수집이 비활성화되어 확인된 외부근거는 없습니다."
    if status == "ready":
        raw_items = news_cache.get("items", []) or []
        item_count = len(raw_items) if isinstance(raw_items, list) else 0
        direct_count, weak_count = _external_item_match_counts(news_cache)
        verified_count = _int_value(news_cache.get("verified_item_count"), fallback=direct_count)
        high_confidence_critical_count = _int_value(
            news_cache.get("high_confidence_critical_count"),
            fallback=0,
        )
        relevance_text = (
            f"{verified_count} verified, {direct_count} direct-match, {weak_count} weak/indirect"
        )
        critical_terms = ", ".join(str(term) for term in news_cache.get("critical_terms", []) or [])
        if critical_terms:
            if external_evidence_veto_triggered(
                news_cache,
                company_name=company_name,
                stock_code=stock_code,
            ):
                return (
                    f"Collected {item_count} external evidence item(s) "
                    f"({relevance_text}); confirmed high-confidence critical evidence: "
                    f"{critical_terms}."
                )
            return (
                f"Collected {item_count} external evidence item(s) ({relevance_text}); "
                f"unconfirmed keyword hit(s): {critical_terms}; "
                f"{high_confidence_critical_count} item(s) had company-keyword context, "
                "but no high-confidence direct veto evidence."
            )
        return (
            f"Collected {item_count} external evidence item(s) ({relevance_text}); "
            "no configured critical terms were found."
        )
    if status == "missing_credentials":
        return "외부근거 수집은 활성화되었지만 제공자 인증 정보가 없습니다."
    if status == "partial_error":
        return "외부근거 수집 중 일부 제공자 오류가 발생했습니다. 제공자 상태를 확인해야 합니다."
    if status == "no_results":
        return "외부근거 수집은 활성화되었지만 제공자 결과가 없습니다."
    if status == "placeholder":
        return "뉴스·크롤링 연동은 현재 placeholder 상태입니다."
    return "외부근거 수집 상태가 아직 구현되지 않았습니다."


def _external_item_match_counts(news_cache: dict[str, Any]) -> tuple[int, int]:
    direct_count = news_cache.get("direct_match_count")
    weak_count = news_cache.get("weak_evidence_count")
    if isinstance(direct_count, int) and isinstance(weak_count, int):
        return direct_count, weak_count

    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list):
        return 0, 0
    direct = sum(
        1 for item in raw_items if isinstance(item, dict) and item.get("company_match") is True
    )
    weak = sum(
        1 for item in raw_items if isinstance(item, dict) and item.get("company_match") is False
    )
    unknown = sum(
        1
        for item in raw_items
        if isinstance(item, dict) and item.get("company_match") not in {True, False}
    )
    return direct, weak + unknown


def _int_value(value: object, *, fallback: int) -> int:
    try:
        return int(value) if isinstance(value, int | float | str) else fallback
    except (TypeError, ValueError):
        return fallback


def _clamp_probability(value: object) -> float:
    try:
        if value is None:
            numeric = 0.0
        elif isinstance(value, int | float | str):
            numeric = float(value)
        else:
            raise ValueError("unsupported probability type")
    except (TypeError, ValueError):
        numeric = 0.0
    return min(max(numeric, 0.0), 1.0)


def _agent_disagreement_level(value: object) -> str:
    text = str(value or "low").strip().lower()
    return text if text in {"low", "medium", "high"} else "low"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
