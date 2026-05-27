"""Shared names and compatibility aliases for Stage 2 review-trigger signals."""

from __future__ import annotations

STAGE2_REVIEW_TRIGGER_DISPLAY_NAME = "full_review_trigger_73"
STAGE2_REVIEW_AUX_ALIAS = "stage2_review_aux"
STAGE2_REVIEW_TRIGGER_POLICY_ID = "feature46_full_review_trigger_73"

STAGE2_REVIEW_AUX_PROB_COLUMN = "prob_speculative_stage2_review_aux"
STAGE2_REVIEW_AUX_THRESHOLD_COLUMN = "threshold_stage2_review_aux"
STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN = "threshold_stage2_review_aux_it_services_review"

LEGACY_STAGE2_REVIEW_AUX_PROB_COLUMN = "prob_speculative_45"
LEGACY_STAGE2_REVIEW_AUX_THRESHOLD_COLUMN = "threshold_45"
LEGACY_STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN = "threshold_45_it_services_review"


def normalize_stage2_review_trigger_reason(value: object) -> str:
    """Return user-facing trigger text without legacy 43/45 feature-set wording."""
    if value is None:
        return ""
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "nat", "<na>"}:
        return ""
    replacements = {
        "45개 보조 레이더": f"{STAGE2_REVIEW_TRIGGER_DISPLAY_NAME} 보조 트리거",
        "45개 보조 변수셋": f"{STAGE2_REVIEW_TRIGGER_DISPLAY_NAME} 보조 트리거",
        "45개 변수셋": f"{STAGE2_REVIEW_TRIGGER_DISPLAY_NAME} 보조 트리거",
        "45개 모델": f"{STAGE2_REVIEW_TRIGGER_DISPLAY_NAME} 보조 모델",
        "43개 모델": "feature_46 공식 모델",
        "43개 baseline": "feature_46 공식 모델",
    }
    for legacy, current in replacements.items():
        text = text.replace(legacy, current)
    return text


__all__ = [
    "LEGACY_STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN",
    "LEGACY_STAGE2_REVIEW_AUX_PROB_COLUMN",
    "LEGACY_STAGE2_REVIEW_AUX_THRESHOLD_COLUMN",
    "STAGE2_REVIEW_AUX_ALIAS",
    "STAGE2_REVIEW_AUX_IT_THRESHOLD_COLUMN",
    "STAGE2_REVIEW_AUX_PROB_COLUMN",
    "STAGE2_REVIEW_AUX_THRESHOLD_COLUMN",
    "STAGE2_REVIEW_TRIGGER_DISPLAY_NAME",
    "STAGE2_REVIEW_TRIGGER_POLICY_ID",
    "normalize_stage2_review_trigger_reason",
]
