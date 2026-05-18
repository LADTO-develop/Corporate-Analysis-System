"""External evidence signal extraction for EvidenceAuditAgent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExternalEvidenceSignals:
    """External news, disclosure, and search findings used by EvidenceAuditAgent."""

    findings: list[str]


def evaluate_external_evidence(news_cache: dict[str, Any]) -> ExternalEvidenceSignals:
    """Convert collected external evidence items into audit findings."""
    raw_items = news_cache.get("items", [])
    if not isinstance(raw_items, list) or not raw_items:
        return ExternalEvidenceSignals(
            findings=["외부 근거 수집: 현재 연결된 뉴스/공시 항목은 없습니다."]
        )

    findings: list[str] = []
    for item in raw_items[:3]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "external"))
        title = str(item.get("title") or item.get("summary") or "근거 제목 없음")
        reliability = str(item.get("reliability", "unknown"))
        evidence_quality = str(item.get("evidence_quality", "unknown"))
        relevance = _relevance_label(item.get("company_match"))
        keyword_note = _keyword_note(item)
        findings.append(
            f"외부 근거({source}, {relevance}, 품질 {evidence_quality}, "
            f"신뢰도 {reliability}): {title}{keyword_note}"
        )
    if news_cache.get("has_critical_risk"):
        terms = ", ".join(str(term) for term in news_cache.get("critical_terms", []) or [])
        if any(isinstance(item, dict) and item.get("veto_candidate") is True for item in raw_items):
            findings.append(
                f"직접 관련 위험 키워드 후보 감지: {terms or 'critical risk'} "
                "(다중 출처·고신뢰 조건 충족 시 veto 검토)."
            )
        else:
            findings.append(
                f"미확인 위험 키워드 히트: {terms or 'critical risk'} "
                "(기업 직접 관련성과 문맥이 확인되지 않으면 veto 근거로 보지 않음)."
            )
    return ExternalEvidenceSignals(findings=findings)


def _relevance_label(company_match: object) -> str:
    if company_match is True:
        return "직접 관련 확인"
    if company_match is False:
        return "직접 관련성 낮음"
    return "직접 관련성 미확인"


def _keyword_note(item: dict[str, Any]) -> str:
    terms = [str(term) for term in item.get("critical_terms", []) or []]
    if not terms:
        return ""
    if item.get("veto_candidate") is True:
        return f" / 직접 관련 위험 키워드 후보: {', '.join(terms)}"
    return f" / 미확인 키워드 히트: {', '.join(terms)}"
