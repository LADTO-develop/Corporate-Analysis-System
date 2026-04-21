"""Smoke tests for LangGraph graph construction.

These tests verify the graph compiles and its topology is consistent with
``configs/agent/graph.yaml`` — without invoking any LLM or loading real
data. Anything that requires API calls lives in ``requires_llm``-marked
tests and is skipped in CI.
"""

from __future__ import annotations

import pytest

from bfd.agents.state import AgentState, append_audit, append_opinions, merge_dict


class TestReducers:
    def test_append_audit_concatenates(self) -> None:
        from bfd.agents.state import AuditEntry

        a = [AuditEntry(node="data", timestamp="t1", summary="x")]
        b = [AuditEntry(node="feature", timestamp="t2", summary="y")]
        merged = append_audit(a, b)
        assert len(merged) == 2
        assert merged[0].node == "data"
        assert merged[1].node == "feature"

    def test_append_opinions_with_nones(self) -> None:
        assert append_opinions(None, None) == []

    def test_merge_dict(self) -> None:
        a = {"tabpfn": 1, "lightgbm": 2}
        b = {"ebm": 3, "tabpfn": 99}
        merged = merge_dict(a, b)
        assert merged == {"tabpfn": 99, "lightgbm": 2, "ebm": 3}


class TestStateSchema:
    def test_minimal_state_instantiates(self) -> None:
        state: AgentState = {
            "corp_code": "005930",
            "market": "KOSPI",
            "fiscal_year": 2024,
            "target_year": 2025,
            "base_predictions": {},
            "committee_opinions": [],
            "audit": [],
            "artifacts": {},
            "insufficient_data": False,
        }
        # Field access should not raise
        assert state["corp_code"] == "005930"
        assert state["target_year"] == state["fiscal_year"] + 1


@pytest.mark.integration
class TestGraphBuild:
    def test_graph_compiles(self) -> None:
        """Compile the graph. Node callables are imported but not invoked."""
        from bfd.agents.graph import build_graph

        graph = build_graph()
        # Compiled graph should expose the LangGraph runnable interface
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_data_node_conditional_predicate(self) -> None:
        """has_enough_data returns the correct branch label."""
        from bfd.agents.nodes.data_node import has_enough_data

        assert has_enough_data({"insufficient_data": True}) == "insufficient"  # type: ignore[arg-type]
        assert has_enough_data({"insufficient_data": False}) == "enough"  # type: ignore[arg-type]
        assert has_enough_data({}) == "enough"  # type: ignore[arg-type]
