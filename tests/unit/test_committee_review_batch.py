"""Tests for committee review batch execution helpers."""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import pytest
from scripts import run_committee_review_evaluation_batch as batch_module


def test_sample_model_replay_skips_pre_replay_stage2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"pre_stage": 0, "replay": 0}

    def fail_run_once(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("run_once should not be called for sample-model replay")

    def fake_run_graph_until_rule_engine(*, company_selection: dict[str, Any]) -> dict[str, Any]:
        calls["pre_stage"] += 1
        assert company_selection["company"]["stock_code"] == "000001"
        return {
            "xgboost_result": {
                "prediction_label": "투자적격",
                "probability_speculative": 0.12,
            },
            "news_cache_snapshot": {"status": "ready", "items": []},
        }

    def fake_rerun_committee_with_sample_model_view(
        state: dict[str, Any],
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        calls["replay"] += 1
        assert sample["corp_name"] == "테스트기업"
        updated = dict(state)
        updated["xgboost_result"] = {
            "prediction_label": "투자적격",
            "probability_speculative": 0.12,
        }
        updated["committee_view"] = {
            "final_committee_label": "보류",
            "veto_triggered": False,
            "hidden_tail_risk_flag": False,
            "conflict_resolution": "샘플 모델값 기준으로 재검토했습니다.",
            "final_review_memo": "샘플 replay",
        }
        updated["stage2_runtime_diagnostics"] = {
            "backend_name": "agno",
            "cache_hit": False,
            "stage2_total_elapsed_seconds": 12.3,
            "agent_elapsed_seconds_sum": 11.7,
            "agent_elapsed_seconds": {
                "quant_credit": 4.1,
                "evidence_audit": 5.2,
                "chair_report": 2.4,
            },
            "parallel_independent_agents": True,
        }
        return updated

    monkeypatch.setattr(batch_module, "run_once", fail_run_once)
    monkeypatch.setattr(
        batch_module,
        "_run_graph_until_rule_engine",
        fake_run_graph_until_rule_engine,
    )
    monkeypatch.setattr(
        batch_module,
        "_rerun_committee_with_sample_model_view",
        fake_rerun_committee_with_sample_model_view,
    )

    results = batch_module.run_batch(
        _sample_batch_frame(),
        use_sample_model_view=True,
        workers=1,
    )

    assert calls == {"pre_stage": 1, "replay": 1}
    assert results.loc[0, "committee_effect"] == "fn_escalated"
    assert results.loc[0, "final_committee_label"] == "보류"
    assert results.loc[0, "stage2_backend_name"] == "agno"
    assert results.loc[0, "stage2_total_elapsed_seconds"] == 12.3
    assert results.loc[0, "stage2_evidence_audit_elapsed_seconds"] == 5.2
    assert bool(results.loc[0, "stage2_parallel_independent_agents"]) is True
    assert results.loc[0, "prior_credit_rating"] == "BB+"
    assert results.loc[0, "prior_rating_agency"] == "한국신용평가"


def test_parallel_batch_preserves_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_batch_case(
        *,
        index: int,
        total: int,
        row: dict[str, Any],
        use_sample_model_view: bool,
    ) -> dict[str, Any]:
        assert total == 3
        assert use_sample_model_view is False
        if index == 0:
            time.sleep(0.02)
        return {"index": index, "corp_name": row["corp_name"]}

    monkeypatch.setattr(batch_module, "_run_batch_case", fake_run_batch_case)
    frame = pd.DataFrame(
        [
            {"corp_name": "A"},
            {"corp_name": "B"},
            {"corp_name": "C"},
        ]
    )

    results = batch_module.run_batch(frame, use_sample_model_view=False, workers=3)

    assert list(results["corp_name"]) == ["A", "B", "C"]
    assert list(results["index"]) == [0, 1, 2]


def test_failed_result_positions_detect_operational_failures() -> None:
    results = pd.DataFrame(
        [
            {
                "corp_name": "정상",
                "final_committee_label": "적격",
                "committee_effect": "tn_kept_eligible",
                "committee_review_safe_effect": "review_safe_tn_not_rejected",
                "stage2_error_message": "",
                "error_message": "",
                "evidence_status": "ready",
            },
            {
                "corp_name": "그래프실패",
                "final_committee_label": "",
                "committee_effect": "run_failed",
                "committee_review_safe_effect": "run_failed",
                "stage2_error_message": "",
                "error_message": "Rate limit reached",
                "evidence_status": "",
            },
            {
                "corp_name": "스테이지2실패",
                "final_committee_label": "보류",
                "committee_effect": "fn_escalated",
                "committee_review_safe_effect": "review_safe_fn_escalated",
                "stage2_error_message": "timeout",
                "error_message": "",
                "evidence_status": "ready",
            },
            {
                "corp_name": "근거수집실패",
                "final_committee_label": "보류",
                "committee_effect": "fn_escalated",
                "committee_review_safe_effect": "review_safe_fn_escalated",
                "stage2_error_message": "",
                "error_message": "",
                "evidence_status": "error",
            },
        ]
    )

    assert batch_module._failed_result_positions(results) == [1, 2, 3]


def test_retry_failed_cases_replaces_failed_rows_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    batch = pd.DataFrame(
        [
            {"corp_name": "A", "stock_code": "000001"},
            {"corp_name": "B", "stock_code": "000002"},
        ]
    )
    initial_results = pd.DataFrame(
        [
            {
                "corp_name": "A",
                "final_committee_label": "",
                "committee_effect": "run_failed",
                "committee_review_safe_effect": "run_failed",
                "error_message": "Rate limit reached",
                "stage2_error_message": "",
                "evidence_status": "",
            },
            {
                "corp_name": "B",
                "final_committee_label": "적격",
                "committee_effect": "tn_kept_eligible",
                "committee_review_safe_effect": "review_safe_tn_not_rejected",
                "error_message": "",
                "stage2_error_message": "",
                "evidence_status": "ready",
            },
        ]
    )
    retry_calls: list[pd.DataFrame] = []

    def fake_run_batch(
        retry_batch: pd.DataFrame,
        *,
        use_sample_model_view: bool,
        workers: int,
    ) -> pd.DataFrame:
        retry_calls.append(retry_batch.copy())
        assert use_sample_model_view is True
        assert workers == 1
        return pd.DataFrame(
            [
                {
                    "corp_name": "A",
                    "final_committee_label": "보류",
                    "committee_effect": "fn_escalated",
                    "committee_review_safe_effect": "review_safe_fn_escalated",
                    "error_message": "",
                    "stage2_error_message": "",
                    "evidence_status": "ready",
                }
            ]
        )

    monkeypatch.setattr(batch_module, "run_batch", fake_run_batch)

    combined, reports = batch_module.retry_failed_cases(
        batch,
        initial_results,
        use_sample_model_view=True,
        attempts=2,
        workers=1,
        delay_seconds=0,
        output_dir=tmp_path,
        write_artifacts=True,
    )

    assert len(retry_calls) == 1
    assert list(retry_calls[0]["corp_name"]) == ["A"]
    assert list(combined["corp_name"]) == ["A", "B"]
    assert combined.loc[0, "final_committee_label"] == "보류"
    assert combined.loc[0, "retry_attempt"] == 1
    assert combined.loc[1, "final_committee_label"] == "적격"
    assert reports == [
        {
            "attempt": 1,
            "failed_rows_before": 1,
            "retried_rows": 1,
            "recovered_rows": 1,
            "remaining_failed_rows": 0,
            "workers": 1,
            "artifact_paths": {
                "samples": str(tmp_path / "retry_artifacts/retry_attempt_1_samples.csv"),
                "results": str(tmp_path / "retry_artifacts/retry_attempt_1_results.csv"),
            },
        }
    ]
    assert (tmp_path / "retry_artifacts/retry_attempt_1_samples.csv").exists()
    assert (tmp_path / "retry_artifacts/retry_attempt_1_results.csv").exists()


def test_configure_runtime_sets_single_claude_agno_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAS_STAGE2_AGNO_MODE", raising=False)
    monkeypatch.delenv("CAS_STAGE2_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("CAS_STAGE2_MODEL", raising=False)

    batch_module.configure_runtime(
        live_external_evidence=False,
        stage2_runner="agno",
        stage2_agno_mode="single",
        stage2_model_provider="anthropic",
        stage2_model="claude-sonnet-4-5-20250929",
    )

    assert batch_module.os.environ["CAS_STAGE2_RUNNER"] == "agno"
    assert batch_module.os.environ["CAS_STAGE2_AGNO_MODE"] == "single"
    assert batch_module.os.environ["CAS_STAGE2_MODEL_PROVIDER"] == "anthropic"
    assert batch_module.os.environ["CAS_STAGE2_MODEL"] == "claude-sonnet-4-5-20250929"
    assert batch_module.os.environ["CAS_STAGE2_LLM_CACHE_ENABLED"] == "1"
    assert "CAS_ENABLE_EXTERNAL_EVIDENCE" not in batch_module.os.environ


def test_configure_runtime_can_disable_stage2_llm_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAS_STAGE2_LLM_CACHE_ENABLED", raising=False)

    batch_module.configure_runtime(
        live_external_evidence=False,
        stage2_runner="agno",
        stage2_agno_mode="single",
        stage2_model_provider="openai",
        stage2_model="gpt-4.1-mini",
        stage2_llm_cache=False,
    )

    assert batch_module.os.environ["CAS_STAGE2_AGNO_MODE"] == "single"
    assert batch_module.os.environ["CAS_STAGE2_LLM_CACHE_ENABLED"] == "0"


def test_materiality_summary_reports_direct_ratio_fields() -> None:
    summary = batch_module._materiality_summary(
        [
            {
                "company_match": True,
                "disclosure_event_class": "material_debt_guarantee",
                "disclosure_materiality": "substantive_adverse",
                "disclosure_severity": "adverse",
                "materiality_ratio": "0.1280",
                "materiality_basis": "채무보증금액/자기자본: 12.80%",
            },
            {
                "company_match": True,
                "disclosure_event_class": "contract_cancellation_watch",
                "disclosure_materiality": "watch_context",
                "disclosure_severity": "caution",
                "materiality_ratio": "0.0312",
                "materiality_basis": "계약해지 금액 매출액 대비: 3.12%",
            },
            {
                "company_match": False,
                "disclosure_materiality": "substantive_adverse",
                "materiality_ratio": "0.5000",
            },
        ]
    )

    assert summary["event_count"] == 2
    assert summary["substantive_count"] == 1
    assert summary["watch_count"] == 1
    assert summary["max_ratio"] == 0.128
    assert summary["top_basis"] == "채무보증금액/자기자본: 12.80%"
    assert summary["event_classes"] == "material_debt_guarantee / contract_cancellation_watch"


def _sample_batch_frame() -> pd.DataFrame:
    selection = {
        "source": "web_listing",
        "company": {
            "market": "KOSDAQ",
            "stock_code": "000001",
            "corp_name": "테스트기업",
        },
        "analysis": {"fiscal_year": 2023, "eval_year": 2024},
        "as_of_date": "2023-12-31",
    }
    return pd.DataFrame(
        [
            {
                "company_selection_json": json.dumps(selection, ensure_ascii=False),
                "corp_name": "테스트기업",
                "actual_label_name": "투기등급",
                "model_predicted_label_name": "투자적격",
                "model_error_type": "false_negative",
                "prior_credit_rating": "BB+",
                "prior_rating_date": "2023-04-01",
                "prior_rating_age_days": 274,
                "prior_rating_agency": "한국신용평가",
                "sample_category": "fn_caught_by_stage2_review",
                "market": "KOSDAQ",
                "stock_code": "000001",
                "fiscal_year": 2023,
                "eval_year": 2024,
            }
        ]
    )
