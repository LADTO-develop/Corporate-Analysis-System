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

        state = run_once(company_id="sample-company")
        assert state["company_name"] == "Sample Components"
        assert state["final_recommendation"] in {"priority", "watch", "review", "defer"}
        assert state["response_json"].keys() == {
            "company_overview",
            "model_result",
            "news_analysis",
            "agent_summary",
        }
        assert state["json_schema_errors"] == []
        assert state["response_json"]["model_result"]["model_name"] == "xgboost_realtime"
        assert "news_summary" in state["response_json"]["agent_summary"]["agents"]
        assert "report_md" in state["artifacts"]
