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
    assert "CAS_ENABLE_EXTERNAL_EVIDENCE" not in batch_module.os.environ


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
