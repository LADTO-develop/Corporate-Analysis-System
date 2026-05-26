"""Assessment value objects used by the Stage 2 committee view."""

from __future__ import annotations

from dataclasses import dataclass

ADVERSE_PROVIDER_RELEVANCE = {"risk"}
ADVERSE_EVIDENCE_QUALITY = {"medium", "high"}
FINANCING_EVIDENCE_TERMS = (
    "전환사채",
    "전환사채권발행",
    "유상증자",
    "신주인수권",
    "신주인수권부사채",
    "증권예탁증권",
    "dr발행",
    "자금조달",
)


@dataclass(frozen=True)
class HiddenTailRiskAssessment:
    """Model-aware external-evidence flag for likely false-negative risk."""

    triggered: bool
    reason: str
    adverse_item_count: int
    verified_adverse_item_count: int
    risk_signal: bool = True


@dataclass(frozen=True)
class SecondaryReviewRiskAssessment:
    """Model-aware Stage 2 review flag for near-threshold false-negative risk."""

    triggered: bool
    reason: str
    review_priority: str
    risk_signal: bool = False


@dataclass(frozen=True)
class BoundaryReviewAssessment:
    """Hold subtype for model decisions close to the investment/speculative boundary."""

    triggered: bool
    reason: str


@dataclass(frozen=True)
class OverwarningMitigationAssessment:
    """Model-aware mitigation flag for likely false-positive review cases."""

    triggered: bool
    reason: str


@dataclass(frozen=True)
class FinancialResilienceAssessment:
    """Financial-defense screen for high-probability over-warning cases."""

    triggered: bool
    reason: str
    support_count: int
    blocker_count: int


@dataclass(frozen=True)
class NoncriticalEvidenceAssessment:
    """External-evidence screen for severe model warnings without decisive corroboration."""

    triggered: bool
    reason: str
    direct_item_count: int
    blocking_item_count: int


@dataclass(frozen=True)
class RejectConfirmationAssessment:
    """Hard-reject gate that separates reject from risk hold."""

    confirmed: bool
    triggered: bool
    reason: str
    signal_count: int
    signals: tuple[str, ...]
    review_risk_signal: bool = False
    review_risk_reason: str = ""
