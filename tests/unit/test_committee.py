"""Unit tests for recommendation thresholds."""

from __future__ import annotations

from cas.agents.nodes.committee_node import _recommendation_from_score


def test_recommendation_thresholds() -> None:
    thresholds = {"priority": 0.75, "watch": 0.60, "review": 0.45}
    assert _recommendation_from_score(0.82, thresholds) == "priority"
    assert _recommendation_from_score(0.65, thresholds) == "watch"
    assert _recommendation_from_score(0.50, thresholds) == "review"
    assert _recommendation_from_score(0.30, thresholds) == "defer"
