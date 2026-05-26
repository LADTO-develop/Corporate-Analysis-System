"""Tests for dashboard Stage 2 live-review cache helpers."""

from __future__ import annotations

import pandas as pd

from cas.dashboard import credit_app


def _selected_row() -> pd.Series:
    return pd.Series(
        {
            "stock_code": "123456",
            "corp_name": "테스트기업",
            "corp_code": "00123456",
            "fiscal_year": 2025,
            "eval_year": 2026,
            "market": "KOSPI",
        }
    )


def _prediction_row() -> pd.Series:
    return pd.Series(
        {
            "prob_speculative": 0.42,
            "threshold": 0.5,
            "predicted_label": "투자적격",
            "risk_band": "near_threshold",
            "stage2_review_priority": "medium",
            "stage2_review_trigger": True,
            "stage2_secondary_trigger": False,
            "stage2_overwarning_filter_candidate": False,
        }
    )


def test_dashboard_committee_cache_key_separates_runner_and_model(monkeypatch) -> None:
    selected_row = _selected_row()
    prediction_row = _prediction_row()
    evidence_snapshot = {"status": "not_requested", "items": []}

    monkeypatch.setenv("CAS_STAGE2_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("CAS_STAGE2_MODEL", "gpt-4.1-mini")
    deterministic_key = credit_app._dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        evidence_snapshot,
        stage2_runner="deterministic",
    )
    agno_key = credit_app._dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        evidence_snapshot,
        stage2_runner="agno",
    )
    assert deterministic_key != agno_key

    monkeypatch.setenv("CAS_STAGE2_MODEL", "gpt-4.1")
    agno_key_with_other_model = credit_app._dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        evidence_snapshot,
        stage2_runner="agno",
    )
    assert agno_key != agno_key_with_other_model


def test_dashboard_live_stage2_trigger_uses_model_and_evidence_flags() -> None:
    assert credit_app._dashboard_needs_live_stage2_from_views(
        {"stage2_review_trigger": True},
        {},
    )
    assert credit_app._dashboard_needs_live_stage2_from_views(
        {"stage2_review_priority": "high"},
        {},
    )
    assert credit_app._dashboard_needs_live_stage2_from_views(
        {"stage2_review_priority": "none"},
        {"veto_candidate_count": 1},
    )
    assert not credit_app._dashboard_needs_live_stage2_from_views(
        {"stage2_review_trigger": False, "stage2_review_priority": "none"},
        {"veto_candidate_count": 0, "high_confidence_critical_count": 0},
    )
