"""Agent disagreement scoring for Stage 2 committee diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput

AgentDisagreementLevel = Literal["low", "medium", "high"]

_EVIDENCE_TREATMENT_RISK = {
    "context_only": 0,
    "watch_context": 1,
    "substantive_review": 2,
    "critical_veto_review": 3,
}
_COMMITTEE_DECISION_RISK = {
    "eligible": 0,
    "mitigation_hold": 1,
    "boundary_hold": 1,
    "review_hold": 1,
    "risk_hold": 2,
    "reject": 3,
}
_REASON_LABELS = {
    "quant_risk_evidence_watch_context": "정량 모델은 위험 신호를 보지만 외부근거는 watch/context 수준입니다.",
    "quant_investment_evidence_substantive": "정량 모델은 투자적격인데 외부근거는 실질 검토 수준입니다.",
    "chair_risk_without_critical_evidence": "최종 판단은 위험 보류이나 EvidenceAudit 치명 근거는 제한적입니다.",
    "chair_reject_without_critical_evidence": "최종 판단은 부적격이나 EvidenceAudit 치명 근거는 제한적입니다.",
    "chair_eligible_with_substantive_evidence": "최종 판단은 적격이나 EvidenceAudit은 실질 외부근거를 봅니다.",
    "chair_escalates_against_investment_model": "정량 모델 투자적격을 최종 위원회가 위험 보류/부적격으로 올렸습니다.",
    "chair_softens_reject_model": "정량 모델 부적격을 최종 위원회가 적격 또는 낮은 보류로 완화했습니다.",
    "committee_label_memo_conflict": "최종 라벨과 메모 문구가 서로 엇갈릴 가능성이 있습니다.",
    "agent_confidence_gap": "역할 agent 간 confidence 차이가 큽니다.",
}
_ELIGIBLE_FINAL_MARKERS = (
    "최종 의견을 적격",
    "최종 의견은 적격",
    "최종 위원회 판단은 적격",
    "최종 라벨은 적격",
    "최종 라벨을 적격",
    "최종 판단은 적격",
    "최종 판단을 적격",
)
_HOLD_FINAL_MARKERS = (
    "최종 의견을 보류",
    "최종 의견은 보류",
    "최종 위원회 판단은 보류",
    "최종 라벨은 보류",
    "최종 라벨을 보류",
    "최종 판단은 보류",
    "최종 판단을 보류",
    "위험 보류입니다",
    "경계등급 보류",
    "확인필요 보류",
)
_REJECT_FINAL_MARKERS = (
    "최종 의견을 부적격",
    "최종 의견은 부적격",
    "최종 위원회 판단은 부적격",
    "최종 라벨은 부적격",
    "최종 라벨을 부적격",
    "최종 판단은 부적격",
    "최종 판단을 부적격",
    "부적격으로 정리",
)


@dataclass(frozen=True)
class AgentDisagreementSignals:
    """Summary of disagreement between QuantCredit, EvidenceAudit, and Chair view."""

    score: float = 0.0
    level: AgentDisagreementLevel = "low"
    reasons: list[str] = field(default_factory=list)
    summary: str = "역할 agent 간 판단 충돌은 낮습니다."

    def as_payload(self) -> dict[str, object]:
        """Return JSON-serializable payload fields for committee_view/runtime diagnostics."""
        return {
            "agent_disagreement_score": self.score,
            "agent_disagreement_level": self.level,
            "agent_disagreement_reasons": list(self.reasons),
            "agent_disagreement_summary": self.summary,
        }


def evaluate_agent_disagreement(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
) -> AgentDisagreementSignals:
    """Score tension between model/quant stance, evidence stance, and final chair view."""
    model_risk = _model_risk_level(bundle)
    evidence_risk = _evidence_risk_level(evidence_audit)
    committee_risk = _committee_risk_level(committee_view)
    has_critical_evidence = _has_critical_evidence(evidence_audit)

    score = 0.0
    reasons: list[str] = []

    if model_risk >= 2 and evidence_risk <= 1:
        score += 0.25
        reasons.append("quant_risk_evidence_watch_context")
    if model_risk == 0 and evidence_risk >= 2:
        score += 0.30
        reasons.append("quant_investment_evidence_substantive")
    if committee_risk == 2 and evidence_risk <= 1 and not has_critical_evidence:
        score += 0.30
        reasons.append("chair_risk_without_critical_evidence")
    if committee_risk == 3 and evidence_risk <= 1 and not has_critical_evidence:
        score += 0.35
        reasons.append("chair_reject_without_critical_evidence")
    if committee_risk == 0 and evidence_risk >= 2:
        score += 0.35
        reasons.append("chair_eligible_with_substantive_evidence")
    if model_risk == 0 and committee_risk >= 2:
        score += 0.20
        reasons.append("chair_escalates_against_investment_model")
    if model_risk >= 2 and committee_risk <= 1:
        score += 0.20
        reasons.append("chair_softens_reject_model")
    if _label_memo_conflict_possible(committee_view):
        score += 0.35
        reasons.append("committee_label_memo_conflict")
    if _confidence_gap(quant_credit, evidence_audit, chair_report) >= 0.30:
        score += 0.10
        reasons.append("agent_confidence_gap")

    unique_reasons = _unique(reasons)
    score = round(min(score, 1.0), 4)
    return AgentDisagreementSignals(
        score=score,
        level=_disagreement_level(score),
        reasons=unique_reasons,
        summary=_summary(unique_reasons, score),
    )


def _model_risk_level(bundle: Stage2InputBundle) -> int:
    probability = _clamp(bundle.probability_speculative)
    threshold = bundle.threshold if bundle.threshold > 0 else 0.315
    if bundle.prediction_label == "부적격":
        if probability >= threshold + 0.25:
            return 3
        if probability >= threshold + 0.10:
            return 2
        return 2
    if 0.0 <= threshold - probability <= 0.05:
        return 1
    return 0


def _evidence_risk_level(evidence_audit: EvidenceAuditOutput) -> int:
    treatment_level = _EVIDENCE_TREATMENT_RISK.get(
        evidence_audit.recommended_evidence_treatment,
        0,
    )
    if evidence_audit.hard_distress_detected:
        return max(treatment_level, 3)
    if evidence_audit.critical_evidence_count > 0:
        return max(treatment_level, 2)
    if evidence_audit.watch_context_count > 0:
        return max(treatment_level, 1)
    return treatment_level


def _committee_risk_level(committee_view: dict[str, Any]) -> int:
    decision_type = str(committee_view.get("committee_decision_type") or "")
    return _COMMITTEE_DECISION_RISK.get(decision_type, 1)


def _has_critical_evidence(evidence_audit: EvidenceAuditOutput) -> bool:
    return bool(
        evidence_audit.recommended_evidence_treatment == "critical_veto_review"
        or evidence_audit.hard_distress_detected
        or evidence_audit.critical_evidence_count > 0
    )


def _label_memo_conflict_possible(committee_view: dict[str, Any]) -> bool:
    final_label = str(committee_view.get("final_committee_label") or "")
    memo = str(committee_view.get("final_review_memo") or "")
    if not memo:
        return False
    if _contains_any(memo, _MEMO_NEGATED_ELIGIBLE_FINAL_MARKERS):
        memo = _remove_markers(memo, _MEMO_NEGATED_ELIGIBLE_FINAL_MARKERS)
    says_eligible = _contains_any(memo, _ELIGIBLE_FINAL_MARKERS)
    says_hold = _contains_any(memo, _HOLD_FINAL_MARKERS)
    says_reject = _contains_any(memo, _REJECT_FINAL_MARKERS)
    if final_label == "적격":
        return says_hold or says_reject
    if final_label == "보류":
        return says_eligible or says_reject
    if final_label == "부적격":
        return says_eligible or says_hold
    return False


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


_MEMO_NEGATED_ELIGIBLE_FINAL_MARKERS = (
    "최종 적격으로 확정하지 않고",
    "최종 적격으로 확정하지 않",
    "최종 라벨을 적격으로 확정하지 않고",
    "최종 라벨을 적격으로 확정하지 않",
)


def _remove_markers(text: str, markers: tuple[str, ...]) -> str:
    updated = text
    for marker in markers:
        updated = updated.replace(marker, "")
    return updated


def _confidence_gap(
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
) -> float:
    values = [
        float(quant_credit.confidence),
        float(evidence_audit.confidence),
        float(chair_report.confidence),
    ]
    return max(values) - min(values)


def _disagreement_level(score: float) -> AgentDisagreementLevel:
    if score >= 0.55:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def _summary(reasons: list[str], score: float) -> str:
    if not reasons:
        return "역할 agent 간 판단 충돌은 낮습니다."
    labels = [_REASON_LABELS[reason] for reason in reasons if reason in _REASON_LABELS]
    if not labels:
        return f"역할 agent 간 판단 충돌 점수는 {score:.2f}입니다."
    return f"역할 agent 간 판단 충돌 점수는 {score:.2f}입니다. " + " ".join(labels[:3])


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


__all__ = [
    "AgentDisagreementLevel",
    "AgentDisagreementSignals",
    "evaluate_agent_disagreement",
]
