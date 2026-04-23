"""Smoke tests for the LangGraph corporate analysis pipeline."""

from __future__ import annotations

import pytest

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
        assert "report_md" in state["artifacts"]
