"""Smoke tests for the LangGraph corporate analysis pipeline."""

from __future__ import annotations

import pytest

from cas.agents.response_schema import DashboardResponse
from cas.agents.state import AgentState, append_audit, append_opinions, merge_dict


class TestReducers:
    def test_append_audit_concatenates(self) -> None:
        from cas.agents.state import AuditEntry

        a = [AuditEntry(node="data", timestamp="t1", summary="loaded")]
        b = [AuditEntry(node="feature", timestamp="t2", summary="normalized")]
        merged = append_audit(a, b)
        assert len(merged) == 2
        assert merged[0].node == "data"
        assert merged[1].node == "feature"

    def test_append_reviews_with_nones(self) -> None:
        assert append_opinions(None, None) == []

    def test_merge_dict(self) -> None:
        merged = merge_dict({"a": 1}, {"b": 2, "a": 3})
        assert merged == {"a": 3, "b": 2}


class TestStateSchema:
    def test_minimal_state_instantiates(self) -> None:
        state: AgentState = {
            "company_id": "sample-company",
            "market": "KOSDAQ",
            "analysis_year": 2025,
            "base_assessments": {},
            "committee_reviews": [],
            "audit": [],
            "artifacts": {},
            "insufficient_data": False,
        }
        assert state["company_id"] == "sample-company"

    def test_schema_node_emits_strict_fallback_on_validation_error(self) -> None:
        from cas.agents.nodes.schema_node import run

        state: AgentState = {
            "company_id": "invalid-company",
            "company_name": "Invalid Company",
            "market": "KOSDAQ",
            "analysis_year": 2025,
            "company_profile": {"company": {"name": "Invalid Company", "market": "KOSDAQ"}},
            "xgboost_result": {
                "model_name": "xgboost_realtime",
                "model_version": "test",
                "prediction_label": "speculative_grade",
                "risk_band": "high_risk",
                "probability_speculative": 1.5,
                "top_drivers": [("profitability_score", 0.9)],
            },
            "rule_result": {"label": "high_risk", "risk_band": "high_risk"},
            "final_recommendation": "review",
            "final_confidence": 0.3,
            "insufficient_data": False,
        }

        result = run(state)

        assert result["json_schema_errors"] != []
        assert result["response_json"]["news_analysis"]["status"] == "schema_validation_failed"
        assert result["response_json"]["model_result"]["rule_label"] == "schema_validation_failed"
        DashboardResponse.model_validate(result["response_json"])


@pytest.mark.integration
class TestGraphBuild:
    def test_graph_compiles(self) -> None:
        from cas.agents.graph import build_graph

        graph = build_graph()
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_graph_runs_on_sample_company(self) -> None:
        from cas.agents.graph import run_once

        state = run_once(company_id="250", market="KOSDAQ", analysis_year=2024)
        assert state["company_name"] == "삼천당제약(주)"
        assert state["final_recommendation"] in {"priority", "watch", "review", "defer"}
        assert state["response_json"].keys() == {
            "company_overview",
            "model_result",
            "news_analysis",
            "agent_summary",
        }
        assert state["json_schema_errors"] == []
        assert state["response_json"]["model_result"]["model_name"] == "credit_43_features"
        assert state["response_json"]["model_result"]["prediction_label"] in {"투자적격", "부적격"}
        assert "financial_model" in state["response_json"]["agent_summary"]["agents"]
        assert "report_md" in state["artifacts"]

    def test_graph_runs_on_2026_inference_company(self) -> None:
        from cas.agents.graph import run_once

        state = run_once(company_id="250", market="KOSDAQ", analysis_year=2026)
        assert state["company_name"] == "삼천당제약(주)"
        assert state["analysis_year"] == 2026
        assert state["processed_company"]["fiscal_year"] == 2025
        assert state["response_json"]["model_result"]["prediction_label"] in {"투자적격", "부적격"}
        assert state["json_schema_errors"] == []

    def test_graph_runs_from_company_selection_contract(self) -> None:
        from cas.agents.graph import run_once

        state = run_once(
            company_selection={
                "request_id": "req-smoke-selection",
                "source": "web_listing",
                "selected_at": "2026-05-11T04:30:00Z",
                "as_of_date": "2026-05-11",
                "company": {
                    "market": "KOSDAQ",
                    "stock_code": "000250",
                    "corp_name": "삼천당제약(주)",
                },
                "analysis": {"fiscal_year": 2023, "eval_year": 2024},
            }
        )

        assert state["company_id"] == "KOSDAQ-000250-2023"
        assert state["company_name"] == "삼천당제약(주)"
        assert state["processed_company"]["source"] == "web_listing"
        assert state["processed_company"]["request_id"] == "req-smoke-selection"
        assert state["processed_company"]["stock_code"] == "000250"
        assert state["response_json"]["company_overview"]["company_id"] == "KOSDAQ-000250-2023"
        assert state["json_schema_errors"] == []

    def test_graph_runs_from_company_selection_without_explicit_year(self) -> None:
        from cas.agents.graph import run_once

        state = run_once(
            company_selection={
                "request_id": "req-smoke-selection-latest",
                "source": "web_listing",
                "selected_at": "2026-05-11T04:30:00Z",
                "as_of_date": "2025-05-11",
                "company": {
                    "market": "KOSDAQ",
                    "stock_code": "000250",
                    "corp_name": "삼천당제약(주)",
                },
                "analysis": {},
            }
        )

        assert state["company_id"] == "KOSDAQ-000250-2023"
        assert state["processed_company"]["fiscal_year"] == 2023
        assert state["processed_company"]["eval_year"] == 2024
        assert state["response_json"]["company_overview"]["company_id"] == "KOSDAQ-000250-2023"
        assert state["json_schema_errors"] == []
