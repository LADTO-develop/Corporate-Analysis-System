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
    company_name = _clean_report_text(str(company.get("company_name", s.get("company_id", "?"))))
    news_summary = _clean_report_text(str(news.get("summary", "")))
    conflict_resolution = _clean_report_text(str(committee_view.get("conflict_resolution", "")))
    hidden_tail_reason = _clean_report_text(str(committee_view.get("hidden_tail_risk_reason", "")))
    final_review_memo = _clean_report_text(str(committee_view.get("final_review_memo", "")))

    md_lines = [
        f"# 신용위험 검토 보고서: {company_name}",
        "",
        f"- **기업 ID**: `{company.get('company_id', s.get('company_id', '?'))}`",
        f"- **시장**: {company.get('market', '?')}",
        f"- **평가연도**: {company.get('analysis_year', '?')}",
        (
            f"- **생성시각**: "
            f"{datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z')}"
        ),
        "",
        "## 모델/위원회 요약",
        "",
        f"- **위원회 추천값**: `{agent_summary.get('final_recommendation', 'review')}`",
        f"- **위원회 검토 신뢰도**: {float(agent_summary.get('final_confidence', 0.0) or 0.0):.3f}",
        f"- **모델 위험 밴드**: `{model_result.get('risk_band', 'insufficient_data')}`",
        (
            f"- **투기등급 예측확률**: "
            f"{float(model_result.get('probability_speculative', 0.0) or 0.0):.3f}"
        ),
        "",
        "## 정량 모델 결과",
        "",
        f"- **모델**: {model_result.get('model_name', 'n/a')} "
        f"({model_result.get('model_version', 'n/a')})",
        f"- **모델 라벨**: `{model_result.get('prediction_label', 'unknown')}`",
        f"- **규칙 라벨**: `{model_result.get('rule_label', 'unknown')}`",
        "",
        "### 주요 모델 기여 변수",
        "",
    ]

    drivers = model_result.get("top_drivers", []) or []
    if drivers:
        for driver in drivers:
            md_lines.append(f"- `{driver.get('name', '')}`: {float(driver.get('value', 0.0)):.3f}")
    else:
        md_lines.append("_(표시할 모델 기여 변수가 없습니다)_")
    md_lines += [
        "",
        "## 외부근거 수집 상태",
        "",
        f"- **상태**: `{news.get('status', 'not_implemented')}`",
        f"- **요약**: {news_summary}",
        "",
        "## 위원회 검토 의견",
        "",
        f"- **최종 위원회 라벨**: `{committee_view.get('final_committee_label', '보류')}`",
        f"- **비토 발동 여부**: `{bool(committee_view.get('veto_triggered', False))}`",
        f"- **숨은 꼬리위험 플래그**: `{bool(committee_view.get('hidden_tail_risk_flag', False))}`",
        f"- **숨은 꼬리위험 사유**: {hidden_tail_reason}",
        f"- **충돌 조정 의견**: {conflict_resolution}",
        f"- **최종 검토 메모**: {final_review_memo}",
        "",
    ]

    risk_factors = _clean_report_items(committee_view.get("key_risk_factors", []) or [])
    mitigating_factors = _clean_report_items(committee_view.get("mitigating_factors", []) or [])
    if risk_factors:
        md_lines += ["### 주요 위험 요인", "", *[f"- {item}" for item in risk_factors], ""]
    if mitigating_factors:
        md_lines += [
            "### 완화 요인",
            "",
            *[f"- {item}" for item in mitigating_factors],
            "",
        ]

    md_lines += [
        "## 에이전트 종합 의견",
        "",
        _clean_report_text(str(agent_summary.get("synthesis", ""))),
        "",
    ]
    agents = dict(agent_summary.get("agents") or {})
    if agents:
        md_lines += ["| 에이전트 | 신뢰도 | 요약 |", "|---|---:|---|"]
        for role, payload in agents.items():
            payload_dict = dict(payload)
            summary = _clean_report_text(str(payload_dict.get("summary", ""))).replace("|", r"\|")
            confidence = float(payload_dict.get("confidence", 0.0) or 0.0)
            md_lines.append(f"| `{role}` | {confidence:.3f} | {summary} |")
        md_lines.append("")

    if schema_errors:
        md_lines += [
            "## 스키마 오류",
            "",
            *[f"- {error}" for error in schema_errors],
            "",
        ]

    md_lines += [
        "## 감사 추적",
        "",
        audit_to_md(audit),
        "",
        "---",
        "_본 보고서는 의사결정 보조 자료이며, 자동 투자 판단 또는 공식 신용등급 부여가 아닙니다._",
    ]

    return {**response, "markdown": "\n".join(md_lines)}


def _clean_report_items(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    return [_clean_report_text(str(item)) for item in items]


def _clean_report_text(text: str) -> str:
    """Clean generated prose before rendering the Korean markdown report."""
    cleaned = text.strip()
    replacements = {
        "적격로": "적격으로",
        "부적격로": "부적격으로",
        "투자적격 등급을 확정합니다": "투자적격 검토 의견을 제시합니다",
        "부적격 등급을 확정합니다": "부적격 검토 의견을 제시합니다",
        "신용등급을 확정합니다": "신용위험 검토 의견을 제시합니다",
        "등급을 확정합니다": "검토 의견을 제시합니다",
        "최종 승인합니다": "검토 의견으로 정리합니다",
        "최종 승인": "검토 의견",
        "확정합니다": "검토 의견을 제시합니다",
        "승인합니다": "의견을 제시합니다",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", ".")
    return cleaned


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
