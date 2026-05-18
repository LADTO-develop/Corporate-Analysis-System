"""Build provider-neutral input bundles for Stage 2 agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from cas.agents.state import AgentState


@dataclass(frozen=True)
class Stage2InputBundle:
    """Normalized context passed to every Stage 2 agent.

    The bundle keeps Agno/Claude-facing inputs explicit while preserving the
    current deterministic scaffold. Agent implementations should read from this
    object rather than reaching into the full LangGraph state directly.
    """

    company_id: str
    company_name: str
    market: str
    analysis_year: int | None
    company_profile: dict[str, Any]
    model_view: dict[str, Any]
    xgboost_result: dict[str, Any]
    rule_result: dict[str, Any]
    source_feature_row: dict[str, Any]
    peer_comparison_rows: tuple[dict[str, Any], ...]
    news_cache_snapshot: dict[str, Any]

    @property
    def prediction_label(self) -> str:
        """Return the read-only Stage 1 model label used by Stage 2."""
        return str(
            self.xgboost_result.get("prediction_label")
            or self.model_view.get("prediction_label")
            or "unknown"
        )

    @property
    def probability_speculative(self) -> float:
        """Return the speculative-grade probability with a safe default."""
        return _safe_float(
            self.xgboost_result.get("probability_speculative")
            or self.model_view.get("probability_speculative")
            or self.model_view.get("y_proba")
        )

    @property
    def news_status(self) -> str:
        """Return the external evidence collection status."""
        return str(self.news_cache_snapshot.get("status", "not_implemented"))

    @property
    def peer_rows_by_feature(self) -> dict[str, dict[str, Any]]:
        """Index peer comparison rows by feature name for SHAP explanations."""
        return {
            str(row.get("feature")): row
            for row in self.peer_comparison_rows
            if isinstance(row.get("feature"), str)
        }

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return a compact dict that can be passed to future Agno agents."""
        return {
            "company": {
                "company_id": self.company_id,
                "company_name": self.company_name,
                "market": self.market,
                "analysis_year": self.analysis_year,
                "profile": self.company_profile,
            },
            "model_view": self.model_view,
            "xgboost_result": self.xgboost_result,
            "rule_result": self.rule_result,
            "source_feature_row": self.source_feature_row,
            "peer_comparison_rows": list(self.peer_comparison_rows),
            "news_cache_snapshot": self.news_cache_snapshot,
        }


def build_stage2_input_bundle(state: AgentState) -> Stage2InputBundle:
    """Normalize LangGraph state into the Stage 2 agent input contract."""
    company_profile = _as_dict(state.get("company_profile"))
    source_feature_row = _as_dict(state.get("source_feature_row"))
    company_id = str(
        state.get("company_id")
        or company_profile.get("company_id")
        or source_feature_row.get("company_id")
        or "unknown"
    )
    company_name = str(
        state.get("company_name")
        or company_profile.get("company_name")
        or source_feature_row.get("company_name")
        or company_id
    )
    market = str(
        state.get("market")
        or source_feature_row.get("market")
        or company_profile.get("market")
        or "UNKNOWN"
    )

    return Stage2InputBundle(
        company_id=company_id,
        company_name=company_name,
        market=market,
        analysis_year=_optional_int(state.get("analysis_year")),
        company_profile=company_profile,
        model_view=_as_dict(state.get("model_view")),
        xgboost_result=_as_dict(state.get("xgboost_result")),
        rule_result=_as_dict(state.get("rule_result")),
        source_feature_row=source_feature_row,
        peer_comparison_rows=_as_dict_tuple(state.get("peer_comparison_rows")),
        news_cache_snapshot=_as_dict(state.get("news_cache_snapshot")),
    )


def _as_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return cast(dict[str, Any], value.model_dump())
    return {}


def _as_dict_tuple(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _optional_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        if not isinstance(value, int | float | str):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        if not isinstance(value, int | float | str):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["Stage2InputBundle", "build_stage2_input_bundle"]
