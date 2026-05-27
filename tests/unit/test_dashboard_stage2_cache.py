"""Tests for dashboard Stage 2 live-review cache helpers."""

from __future__ import annotations

import pandas as pd

from cas.agents.nodes import committee_node
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.agents.state import AgentState
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


def test_dashboard_stage2_runtime_config_snapshots_cache_knobs(monkeypatch) -> None:
    selected_row = _selected_row()
    prediction_row = _prediction_row()
    evidence_snapshot = {"status": "not_requested", "items": []}

    monkeypatch.setenv("CAS_STAGE2_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("CAS_STAGE2_MODEL", "gpt-4.1-mini")
    runtime_config = credit_app._dashboard_stage2_runtime_config("agno")
    key_from_snapshot = credit_app._dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        evidence_snapshot,
        runtime_config=runtime_config,
    )

    monkeypatch.setenv("CAS_STAGE2_MODEL", "gpt-4.1")
    key_from_same_snapshot = credit_app._dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        evidence_snapshot,
        runtime_config=runtime_config,
    )
    key_from_current_env = credit_app._dashboard_committee_cache_key(
        selected_row,
        prediction_row,
        evidence_snapshot,
        stage2_runner="agno",
    )

    assert key_from_snapshot == key_from_same_snapshot
    assert key_from_snapshot != key_from_current_env


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


def test_dashboard_stage2_runner_override_does_not_mutate_environment(monkeypatch) -> None:
    monkeypatch.setenv("CAS_ALLOW_LIVE_STAGE2_IN_TESTS", "1")
    monkeypatch.setenv("CAS_STAGE2_RUNNER", "deterministic")

    def fake_committee_run(_state: AgentState) -> dict[str, object]:
        assert committee_node._stage2_runner_name() == "agno"
        return {"committee_view": {"final_committee_label": "보류"}}

    monkeypatch.setattr(credit_app.committee_node, "run", fake_committee_run)

    result = credit_app._run_dashboard_stage2(
        {
            "model_view": {
                "stage2_review_trigger": True,
                "stage2_review_priority": "high",
            },
            "news_cache_snapshot": {},
        },
        requested_runner="agno",
    )

    assert result["committee_view"]["final_committee_label"] == "보류"
    assert committee_node._stage2_runner_name() == "deterministic"


def test_dashboard_stage2_runtime_config_override_does_not_read_mutated_env(monkeypatch) -> None:
    monkeypatch.setenv("CAS_ALLOW_LIVE_STAGE2_IN_TESTS", "1")
    monkeypatch.setenv("CAS_STAGE2_RUNNER", "deterministic")
    monkeypatch.setenv("CAS_STAGE2_MODEL", "env-model")
    runtime_config = Stage2RuntimeConfig(runner="agno", model="snapshot-model")

    def fake_committee_run(_state: AgentState) -> dict[str, object]:
        assert committee_node._stage2_runner_name() == "agno"
        assert committee_node._stage2_runtime_config().model == "snapshot-model"
        return {"committee_view": {"final_committee_label": "보류"}}

    monkeypatch.setattr(credit_app.committee_node, "run", fake_committee_run)

    result = credit_app._run_dashboard_stage2(
        {
            "model_view": {
                "stage2_review_trigger": True,
                "stage2_review_priority": "high",
            },
            "news_cache_snapshot": {},
        },
        runtime_config=runtime_config,
    )

    assert result["committee_view"]["final_committee_label"] == "보류"
    assert committee_node._stage2_runner_name() == "deterministic"
    assert committee_node._stage2_runtime_config().model == "env-model"
