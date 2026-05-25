"""Structured treatment signals for EvidenceAuditAgent outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cas.agents.committee_utils import safe_float
from cas.agents.signals.materiality_signals import (
    financing_evidence_items,
    has_hard_distress_terms,
    has_substantive_external_risk,
    high_risk_financing_evidence_count,
    material_financing_evidence_blocks_tn_hold,
    substantive_external_risk_item,
)

EvidenceTreatment = Literal[
    "context_only",
    "watch_context",
    "substantive_review",
    "critical_veto_review",
]

_UNAVAILABLE_EVIDENCE_STATUSES = {
    "disabled",
    "missing_credentials",
    "not_implemented",
    "not_requested",
    "placeholder",
}


@dataclass(frozen=True)
class EvidenceTreatmentSignals:
    """Evidence counts and treatment recommendation passed from EvidenceAudit onward."""

    critical_evidence_count: int = 0
    watch_context_count: int = 0
    materiality_summary: dict[str, Any] = field(default_factory=dict)
    hard_distress_detected: bool = False
    recommended_evidence_treatment: EvidenceTreatment = "context_only"

    def as_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload for LLM prompts and Stage 2 outputs."""
        return {
            "critical_evidence_count": self.critical_evidence_count,
            "watch_context_count": self.watch_context_count,
            "materiality_summary": self.materiality_summary,
            "hard_distress_detected": self.hard_distress_detected,
            "recommended_evidence_treatment": self.recommended_evidence_treatment,
        }


def evaluate_evidence_treatment(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None = None,
    materiality_summary: dict[str, Any] | None = None,
) -> EvidenceTreatmentSignals:
    """Summarize whether external evidence is critical, watch-only, or contextual."""
    status = str(news_cache.get("status") or "").strip().lower()
    raw_items = news_cache.get("items", [])
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    summary = materiality_summary or _materiality_summary(
        news_cache,
        source_feature_row=source_feature_row,
    )
    if status in _UNAVAILABLE_EVIDENCE_STATUSES or not items:
        return EvidenceTreatmentSignals(materiality_summary=summary)

    critical_count = 0
    watch_context_count = 0
    hard_distress_detected = False
    veto_or_confirmed_count = 0
    for item in items:
        if item.get("company_match") is False:
            continue
        hard_distress = has_hard_distress_terms(item)
        hard_distress_detected = hard_distress_detected or hard_distress
        veto_or_confirmed = (
            item.get("veto_candidate") is True
            or item.get("critical_context_confirmed") is True
        )
        veto_or_confirmed_count += int(veto_or_confirmed)
        substantive = substantive_external_risk_item(
            item,
            source_feature_row=source_feature_row,
        )
        if veto_or_confirmed or hard_distress or substantive:
            critical_count += 1
        elif _is_watch_context_external_item(item):
            watch_context_count += 1

    if hard_distress_detected or veto_or_confirmed_count > 0:
        treatment: EvidenceTreatment = "critical_veto_review"
    elif critical_count > 0 or _safe_int(summary.get("substantive_external_risk_count")) > 0:
        treatment = "substantive_review"
    elif watch_context_count > 0 or _safe_int(summary.get("materiality_event_count")) > 0:
        treatment = "watch_context"
    else:
        treatment = "context_only"

    return EvidenceTreatmentSignals(
        critical_evidence_count=critical_count,
        watch_context_count=watch_context_count,
        materiality_summary=summary,
        hard_distress_detected=hard_distress_detected,
        recommended_evidence_treatment=treatment,
    )


def _materiality_summary(
    news_cache: dict[str, Any],
    *,
    source_feature_row: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_items = news_cache.get("items", [])
    items = (
        [item for item in raw_items if isinstance(item, dict)]
        if isinstance(raw_items, list)
        else []
    )
    materiality_items = [
        item
        for item in items
        if (
            safe_float(item.get("materiality_ratio")) is not None
            or str(item.get("materiality_basis") or "").strip()
            or str(item.get("disclosure_event_class") or "").strip()
            or str(item.get("disclosure_materiality") or "").strip()
        )
    ]
    top_item = max(
        materiality_items,
        key=lambda item: safe_float(item.get("materiality_ratio")) or 0.0,
        default={},
    )
    event_classes = sorted(
        {
            str(item.get("disclosure_event_class") or "").strip()
            for item in items
            if str(item.get("disclosure_event_class") or "").strip()
        }
    )
    materiality_classes = sorted(
        {
            str(item.get("disclosure_materiality") or "").strip()
            for item in items
            if str(item.get("disclosure_materiality") or "").strip()
        }
    )
    substantive_count = sum(
        1
        for item in items
        if substantive_external_risk_item(item, source_feature_row=source_feature_row)
    )
    hard_distress_count = sum(1 for item in items if has_hard_distress_terms(item))
    return {
        "item_count": len(items),
        "materiality_event_count": len(materiality_items),
        "substantive_external_risk_count": substantive_count,
        "has_substantive_external_risk": has_substantive_external_risk(
            news_cache,
            source_feature_row=source_feature_row,
        ),
        "financing_evidence_count": len(financing_evidence_items(news_cache)),
        "high_risk_financing_evidence_count": high_risk_financing_evidence_count(
            news_cache,
            source_feature_row=source_feature_row,
        ),
        "material_financing_blocks_tn_hold": material_financing_evidence_blocks_tn_hold(
            news_cache,
            source_feature_row=source_feature_row,
        ),
        "hard_distress_item_count": hard_distress_count,
        "max_materiality_ratio": safe_float(top_item.get("materiality_ratio")),
        "top_materiality_basis": str(top_item.get("materiality_basis") or ""),
        "top_materiality_title": str(top_item.get("title") or ""),
        "event_classes": event_classes[:12],
        "materiality_classes": materiality_classes[:12],
    }


def _is_watch_context_external_item(item: dict[str, Any]) -> bool:
    severity = str(item.get("disclosure_severity", "")).lower()
    event_class = str(item.get("disclosure_event_class", "")).lower()
    materiality = str(item.get("disclosure_materiality", "")).lower()
    provider_relevance = str(item.get("provider_relevance", "")).lower()
    return (
        severity in {"routine", "caution"}
        or event_class in {"routine_context", "watch_context", "procedural_or_one_off"}
        or materiality in {"routine_context", "watch_context", "procedural_or_one_off"}
        or provider_relevance in {"routine", "caution", "context"}
    )


def _safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "EvidenceTreatment",
    "EvidenceTreatmentSignals",
    "evaluate_evidence_treatment",
]
