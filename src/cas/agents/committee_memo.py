"""Memo and conflict-resolution helpers for committee_view payloads."""

from __future__ import annotations

from typing import cast

from cas.agents.committee_assessments import (
    BoundaryReviewAssessment,
    HiddenTailRiskAssessment,
    OverwarningMitigationAssessment,
    RejectConfirmationAssessment,
    SecondaryReviewRiskAssessment,
)
from cas.agents.committee_schema import CommitteeLabel
from cas.agents.committee_utils import clean_korean_review_text as _clean_korean_review_text
from cas.agents.signals.evidence_treatment_signals import EvidenceTreatmentSignals
from cas.agents.state import AgentOutput


def evidence_limitations_from_agents(agents: list[AgentOutput]) -> list[str]:
    """Extract evidence limitation lines from flattened EvidenceAudit findings."""
    evidence_agent = next((agent for agent in agents if agent.role == "evidence_audit"), None)
    if evidence_agent is None:
        return []
    limitations: list[str] = []
    for finding in evidence_agent.findings:
        text = str(finding)
        if text.startswith("근거 한계:"):
            value = text.removeprefix("근거 한계:").strip()
            if value:
                limitations.append(value)
    return limitations


def conflict_resolution(
    *,
    prediction_label: str,
    committee_label: str,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    boundary_review: BoundaryReviewAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
) -> str:
    """Build the human-readable conflict-resolution summary."""
    if veto_triggered:
        return (
            "치명적 외부 위험 신호가 확인되어 모델 원판단과 무관하게 "
            "위원회 의견을 부적격으로 보수 조정했습니다."
        )
    if hidden_tail_risk.triggered:
        if not hidden_tail_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}이지만, 직접 관련 외부 규모성 공시가 "
                "추가 확인 대상으로 확인되어 위원회 의견은 보류로 정리했습니다. 다만 "
                "치명 문맥이나 실질 부실 전이 근거는 제한적이어서 위험 보류가 아닌 "
                "확인필요 보류로 구분했습니다."
            )
        return (
            f"모델 원판단은 {prediction_label}이지만, 직접 관련 외부 위험 근거가 "
            "모델이 놓칠 수 있는 숨은 꼬리위험을 보완해 위원회 의견은 보류로 정리했습니다."
        )
    if overwarning_mitigation.triggered:
        return (
            f"모델 원판단은 {prediction_label}이지만, 과민 경고 가능성과 완화 근거를 함께 보아 "
            "위원회 의견은 부적격이 아닌 보류로 정리했습니다."
        )
    if reject_confirmation.triggered:
        hold_type = "위험 보류" if reject_confirmation.review_risk_signal else "확인필요 보류"
        return (
            f"모델 원판단은 {prediction_label}이지만, 부적격 확정 게이트를 통과하지 못해 "
            f"위원회 의견은 부적격 확정이 아닌 {hold_type}로 정리했습니다. "
            f"{reject_confirmation.reason}"
        )
    if boundary_review.triggered:
        return (
            f"모델 원판단은 {prediction_label}이지만, 경계등급 확인 신호가 있어 "
            "위원회 의견은 경계등급 보류로 정리했습니다. 이는 위험 확정이나 과민경고 완화가 "
            f"아니라, 추가 근거 확인이 필요한 상태입니다. {boundary_review.reason}"
        )
    if secondary_review_risk.triggered:
        if not secondary_review_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}이지만, full_review_trigger_73 보조 트리거가 "
                "추가 확인 대상으로 "
                "올린 케이스라 위원회 의견은 보류로 정리했습니다. 다만 확률 수준은 위험신호 "
                "표시 기준선보다 낮아 확인필요 보류로 구분합니다."
            )
        return (
            f"모델 원판단은 {prediction_label}이지만, full_review_trigger_73 보조 트리거의 "
            "추가 검토 신호가 "
            "FN 가능성을 보완해 위원회 의견은 보류로 정리했습니다."
        )
    model_label = "적격" if prediction_label == "투자적격" else "부적격"
    if committee_label == "보류":
        return (
            f"모델 원판단은 {prediction_label}이지만, 정량 해석과 외부/유동성 검증 사이에 "
            "추가 점검 여지가 있어 위원회 의견은 보류로 정리했습니다."
        )
    if committee_label != model_label:
        return (
            f"모델 원판단({prediction_label})과 위원회 라벨({committee_label})이 달라, "
            "외부 검증 근거와 완화 요인을 함께 고려해 최종 의견을 조정했습니다."
        )
    return (
        f"모델 원판단({prediction_label})과 위원회 라벨({committee_label})이 대체로 일치하며, "
        "Stage 2는 판단을 덮어쓰기보다 근거와 설명을 보완했습니다."
    )


def final_review_memo(
    *,
    prediction_label: str,
    committee_label: str,
    veto_triggered: bool,
    hidden_tail_risk: HiddenTailRiskAssessment,
    boundary_review: BoundaryReviewAssessment,
    secondary_review_risk: SecondaryReviewRiskAssessment,
    overwarning_mitigation: OverwarningMitigationAssessment,
    reject_confirmation: RejectConfirmationAssessment,
    risk_factors: list[str],
    mitigating_factors: list[str],
) -> str:
    """Build the base final review memo before optional Chair prose is appended."""
    if veto_triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존하지만, 강제 경고 조건을 충족하는 "
            "외부 또는 정책 위험 신호가 있어 위원회 의견을 부적격으로 정리했습니다."
        )
    if hidden_tail_risk.triggered:
        if not hidden_tail_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 직접 관련 "
                "규모성 공시가 확인되어 최종 적격으로 바로 확정하지 않고 보류로 "
                "정리했습니다. 치명 문맥이나 현금흐름 악화가 함께 확인된 실질 부실 "
                f"근거는 제한적이므로 세부 유형은 확인필요 보류입니다. {hidden_tail_risk.reason}"
            )
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 직접 관련 외부 위험 "
            f"근거가 확인되어 재무제표 기반 모델이 놓칠 수 있는 FN 가능성을 보완했습니다. "
            f"위원회는 최종 의견을 {committee_label}로 정리했습니다. {hidden_tail_risk.reason}"
        )
    if overwarning_mitigation.triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 강한 외부 위험 근거가 "
            f"확인되지 않았고 완화 근거가 있어 위원회는 최종 의견을 {committee_label}로 "
            f"낮춰 정리했습니다. {overwarning_mitigation.reason}"
        )
    if reject_confirmation.triggered:
        hold_type = "위험 보류" if reject_confirmation.review_risk_signal else "확인필요 보류"
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 위원회는 부적격을 "
            "확정하기 위한 복수의 강한 근거가 충분하지 않다고 보고 최종 의견을 "
            f"{committee_label}로 정리했으며, 세부 유형은 {hold_type}입니다. "
            f"{reject_confirmation.reason}"
        )
    if boundary_review.triggered:
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 투자적격/투기등급 "
            "경계에서 판단 불확실성이 큰 케이스라 위원회는 최종 의견을 보류로 "
            f"정리하고, 세부 유형은 경계등급 보류로 표시했습니다. {boundary_review.reason}"
        )
    if secondary_review_risk.triggered:
        if not secondary_review_risk.risk_signal:
            return (
                f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 full_review_trigger_73 "
                "보조 트리거가 "
                "추가 확인 대상으로 올린 케이스라 최종 적격으로 바로 확정하지 않고 보류로 "
                "정리했습니다. 확률 수준은 위험신호 표시 기준선보다 낮아 확인필요 보류로 "
                f"구분합니다. {secondary_review_risk.reason}"
            )
        return (
            f"모델 원판단은 {prediction_label}으로 보존합니다. 다만 full_review_trigger_73 "
            "보조 트리거가 "
            f"추가 검토 대상으로 올린 케이스라 FN 가능성을 보수적으로 보완했습니다. "
            f"위원회는 최종 의견을 {committee_label}로 정리했습니다. "
            f"{secondary_review_risk.reason}"
        )
    risk_note = (
        f"주요 위험은 {risk_factors[0]}"
        if risk_factors
        else "추가로 확정된 핵심 위험 요인은 제한적입니다"
    )
    mitigation_note = (
        f"완화 요인은 {mitigating_factors[0]}"
        if mitigating_factors
        else "명시적 완화 요인은 제한적입니다"
    )
    return (
        f"모델 원판단은 {prediction_label}으로 보존합니다. 위원회는 정량 해석, "
        f"부채/유동성 교차 검증, 외부 근거 상태를 함께 검토해 최종 의견을 "
        f"{committee_label}로 정리했습니다. {risk_note}. {mitigation_note}."
    )


def chair_report_memo_seed(
    agents: list[AgentOutput],
    *,
    committee_label: CommitteeLabel,
    evidence_treatment: EvidenceTreatmentSignals,
) -> str:
    """Return the first informative ChairReport memo snippet, if any."""
    chair = next((agent for agent in agents if agent.role == "chair_report"), None)
    if chair is None:
        return ""
    candidates = [*chair.findings[::-1], chair.summary]
    for candidate in candidates:
        cleaned = cast(str, _clean_korean_review_text(str(candidate or "")))
        cleaned = sanitize_chair_report_memo_for_evidence_treatment(
            cleaned,
            committee_label=committee_label,
            evidence_treatment=evidence_treatment,
        )
        if is_informative_chair_report_memo(
            cleaned,
            committee_label=committee_label,
        ):
            return cleaned
    return ""


def is_informative_chair_report_memo(text: str, *, committee_label: CommitteeLabel) -> bool:
    """Return whether Chair prose adds useful context and does not contradict the label."""
    if len(text.strip()) < 40:
        return False
    generic_markers = (
        "ChairReportAgent는 정량 해석과 검증 근거를 사람이 읽는 심사 메모로 연결합니다",
        "정량 판단은 model_view로 보존",
        "committee_view에서는 해석과 보완 의견만 추가합니다",
        "최종 보고서는 적격/보류/부적격 3단 위원회 의견",
        "Agno ",
        "chair label=",
        "ChairReportAgent는 모델 원판단",
        "현재 서비스 recommendation은",
    )
    if any(marker in text for marker in generic_markers):
        return False
    return not chair_report_memo_conflicts_with_final_label(
        text,
        committee_label=committee_label,
    )


def sanitize_chair_report_memo_for_evidence_treatment(
    text: str,
    *,
    committee_label: CommitteeLabel,
    evidence_treatment: EvidenceTreatmentSignals,
) -> str:
    """Soften over-critical Chair prose when structured evidence is not veto-grade."""
    if committee_label != "적격" and (
        evidence_treatment.recommended_evidence_treatment == "critical_veto_review"
        or evidence_treatment.hard_distress_detected
    ):
        return text

    replacements = {
        "치명적 외부 위험 신호": "추가 확인이 필요한 외부 위험 단서",
        "치명적 위험 신호": "추가 확인이 필요한 외부 위험 단서",
        "치명 외부근거": "직접 확인된 중대 외부근거",
        "외부 치명근거": "직접 확인된 중대 외부근거",
        "치명 리스크": "중대 위험",
        "치명 문맥": "중대 위험 문맥",
        "치명급": "중대",
        "외부 증거의 중대성을 감안하여": "외부 증거의 확인 필요성을 감안하여",
        "보수적 관점에서 위원장 단계 추가 검토가 필요": "사후 모니터링 관점에서 추가 확인이 필요",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def chair_report_memo_conflicts_with_final_label(
    text: str, *, committee_label: CommitteeLabel
) -> bool:
    """Avoid appending agent prose that contradicts the resolved committee label."""
    if committee_label == "적격":
        return False
    investment_keep_markers = (
        "투자적격 판단을 유지",
        "투자적격 라벨을 유지",
        "투자적격 등급을 유지",
        "투자적격 분류를 유지",
        "투자적격 유지",
        "모델 라벨을 유지",
        "모델 라벨 유지",
        "모델 라벨을 존중",
        "모델 라벨 존중",
        "최종 라벨은 투자적격",
        "최종 라벨을 투자적격",
        "기존 투자적격",
    )
    return any(marker in text for marker in investment_keep_markers)


def with_chair_report_memo(base_memo: str, chair_memo_seed: str) -> str:
    """Append Chair prose to the deterministic memo when it adds new information."""
    base = cast(str, _clean_korean_review_text(base_memo))
    seed = cast(str, _clean_korean_review_text(chair_memo_seed))
    if not seed:
        return base
    normalized_base = " ".join(base.split())
    normalized_seed = " ".join(seed.split())
    if normalized_seed in normalized_base:
        return base
    if normalized_base in normalized_seed:
        return seed
    return f"{base} 위원회 보강 의견: {seed}"


__all__ = [
    "chair_report_memo_seed",
    "conflict_resolution",
    "evidence_limitations_from_agents",
    "final_review_memo",
    "with_chair_report_memo",
]
