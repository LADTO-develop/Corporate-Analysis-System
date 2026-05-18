"""Build and compile the LangGraph pipeline from ``configs/agent/graph.yaml``."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Any, cast

from cas.agents.contracts import CompanySelectionRequest, build_agent_state_seed
from cas.agents.state import AgentState
from cas.utils.io import read_yaml
from cas.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    MemorySaver = None  # type: ignore[misc,assignment]
    StateGraph = None  # type: ignore[misc,assignment]
    START = "__start__"
    END = "__end__"
    HAS_LANGGRAPH = False


def _import_callable(spec: str) -> Callable[..., Any]:
    module_name, attr = spec.split(":")
    module = importlib.import_module(module_name)
    return cast(Callable[..., Any], getattr(module, attr))


def _import_node_fn(
    module_path: str,
    fn_name: str,
) -> Callable[[AgentState], dict[str, Any]]:
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)  # type: ignore[no-any-return]


def build_graph(config_path: str | Path = "configs/agent/graph.yaml") -> object:
    """Build and compile the main StateGraph."""
    cfg = read_yaml(config_path)
    if HAS_LANGGRAPH:
        return _build_langgraph(cfg)
    return _build_fallback_graph(cfg)


def _build_langgraph(cfg: dict[str, Any]) -> object:
    builder: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

    for node in cfg["nodes"]:
        fn = _import_node_fn(node["module"], node["fn"])
        builder.add_node(node["name"], fn)  # type: ignore[call-overload]
        logger.debug("graph_add_node", node=node["name"])

    builder.add_edge(START, cfg["entry_node"])

    conditional_sources = set((cfg.get("conditional_edges") or {}).keys())
    for src, dst in cfg["edges"]:
        if src in conditional_sources:
            continue
        builder.add_edge(src, dst)

    for src, spec in (cfg.get("conditional_edges") or {}).items():
        cond_fn = _import_callable(spec["condition_fn"])
        mapping: dict[Hashable, str] = {
            k: (END if v == "END" else v) for k, v in spec["mapping"].items()
        }
        builder.add_conditional_edges(src, cond_fn, mapping)

    builder.add_edge("report", END)
    graph = builder.compile(checkpointer=MemorySaver())
    logger.info("graph_compiled", n_nodes=len(cfg["nodes"]), backend="langgraph")
    return graph


def _build_fallback_graph(cfg: dict[str, Any]) -> _FallbackGraph:
    logger.info("graph_compiled", n_nodes=len(cfg["nodes"]), backend="fallback")
    return _FallbackGraph(cfg)


class _FallbackGraph:
    """Small sequential graph runner used when LangGraph is unavailable."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._entry = cfg["entry_node"]
        self._nodes = {
            node["name"]: _import_node_fn(node["module"], node["fn"]) for node in cfg["nodes"]
        }
        self._edges = {src: dst for src, dst in cfg["edges"]}
        self._conditionals = {
            src: {
                "fn": _import_callable(spec["condition_fn"]),
                "mapping": spec["mapping"],
            }
            for src, spec in (cfg.get("conditional_edges") or {}).items()
        }

    def invoke(self, initial: AgentState, config: dict[str, Any] | None = None) -> AgentState:
        state: AgentState = dict(initial)
        node_name = self._entry
        while True:
            updates = self._nodes[node_name](state)
            state = _merge_state(state, updates)

            if node_name in self._conditionals:
                conditional = self._conditionals[node_name]
                branch = conditional["fn"](state)
                target = conditional["mapping"][branch]
                if target == "END":
                    break
                node_name = target
                continue

            next_node = self._edges.get(node_name)
            if next_node in (None, "END"):
                break
            node_name = next_node
        return state

    def stream(
        self,
        initial: AgentState,
        config: dict[str, Any] | None = None,
    ) -> list[AgentState]:
        return [self.invoke(initial, config=config)]


def _merge_state(current: AgentState, updates: dict[str, Any]) -> AgentState:
    merged = dict(current)
    for key, value in updates.items():
        if key in {"audit", "committee_reviews", "agent_outputs"}:
            merged[key] = [*(merged.get(key) or []), *(value or [])]
        elif key in {"base_assessments", "artifacts", "agent_summary"}:
            existing = dict(merged.get(key) or {})
            existing.update(value or {})
            merged[key] = existing
        else:
            merged[key] = value
    return merged


def run_once(
    *,
    company_id: str | None = None,
    company_selection: CompanySelectionRequest | Mapping[str, Any] | None = None,
    market: str | None = None,
    analysis_year: int | None = None,
    config_path: str | Path = "configs/agent/graph.yaml",
    thread_id: str | None = None,
) -> AgentState:
    """Compile and run the graph once for a single company."""
    graph = build_graph(config_path)

    if company_selection is not None:
        initial = build_agent_state_seed(company_selection)
        if market is not None:
            initial["market"] = market
        if analysis_year is not None:
            initial["analysis_year"] = analysis_year
    elif company_id is not None:
        initial = {
            "company_id": company_id,
            "market": market or "UNKNOWN",
            "analysis_year": analysis_year or 0,
            "base_assessments": {},
            "committee_reviews": [],
            "agent_outputs": [],
            "agent_summary": {},
            "committee_view": {},
            "audit": [],
            "artifacts": {},
            "insufficient_data": False,
        }
    else:
        raise ValueError("company_id or company_selection is required")

    default_thread_id = (
        thread_id
        or cast(dict[str, Any], initial.get("company_selection") or {}).get("request_id")
        or initial["company_id"]
    )
    config = {"configurable": {"thread_id": default_thread_id}}
    final: AgentState = graph.invoke(initial, config=config)  # type: ignore[attr-defined]
    return final
