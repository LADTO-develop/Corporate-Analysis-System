"""Build and compile the LangGraph pipeline from configs/agent/graph.yaml.

The graph is a straightforward DAG with one fan-out/fan-in:

    data → feature → base_prediction → (macro_overlay ∥ news_overlay) → committee → report

``data_node`` may short-circuit to ``report`` if it flags insufficient data.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from bfd.agents.state import AgentState
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


def _import_callable(spec: str) -> Callable[..., Any]:
    """Import a callable given ``module.path:callable_name``."""
    module_name, attr = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _import_node_fn(module_path: str, fn_name: str) -> Callable[[AgentState], dict[str, Any]]:
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)  # type: ignore[no-any-return]


def build_graph(config_path: str | Path = "configs/agent/graph.yaml") -> Any:
    """Build and compile the main StateGraph."""
    cfg = read_yaml(config_path)

    builder: StateGraph = StateGraph(AgentState)

    # --- nodes --------------------------------------------------------------
    for node in cfg["nodes"]:
        fn = _import_node_fn(node["module"], node["fn"])
        builder.add_node(node["name"], fn)
        logger.debug("graph_add_node", node=node["name"])

    # --- entry --------------------------------------------------------------
    builder.add_edge(START, cfg["entry_node"])

    # --- plain edges --------------------------------------------------------
    # Conditional-edge sources are excluded from plain edges so we don't
    # create a duplicate routing path.
    conditional_sources = set((cfg.get("conditional_edges") or {}).keys())
    for src, dst in cfg["edges"]:
        if src in conditional_sources:
            continue
        builder.add_edge(src, dst)

    # --- conditional edges ---------------------------------------------------
    for src, spec in (cfg.get("conditional_edges") or {}).items():
        cond_fn = _import_callable(spec["condition_fn"])
        mapping: dict[str, Any] = {
            k: (END if v == "END" else v) for k, v in spec["mapping"].items()
        }
        builder.add_conditional_edges(src, cond_fn, mapping)
        logger.debug("graph_add_conditional_edge", src=src, mapping=spec["mapping"])

    # --- terminal edge: report → END ----------------------------------------
    builder.add_edge("report", END)

    # --- checkpointer --------------------------------------------------------
    checkpointer: Any
    cp_cfg = cfg.get("checkpointer", {})
    if cp_cfg.get("kind", "memory") == "memory":
        checkpointer = MemorySaver()
    elif cp_cfg["kind"] == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            checkpointer = SqliteSaver.from_conn_string(cp_cfg["sqlite_path"])
        except ImportError as exc:
            raise RuntimeError(
                "SqliteSaver requested but langgraph-checkpoint-sqlite is not installed."
            ) from exc
    else:
        raise ValueError(f"Unknown checkpointer kind: {cp_cfg['kind']}")

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("graph_compiled", n_nodes=len(cfg["nodes"]))
    return graph


def run_once(
    *,
    corp_code: str,
    market: str,
    fiscal_year: int,
    config_path: str | Path = "configs/agent/graph.yaml",
    thread_id: str | None = None,
) -> AgentState:
    """Convenience helper — compile and run the graph once for a single firm."""
    graph = build_graph(config_path)

    initial: AgentState = {
        "corp_code": corp_code,
        "market": market,  # type: ignore[typeddict-item]
        "fiscal_year": fiscal_year,
        "target_year": fiscal_year + 1,
        "base_predictions": {},
        "committee_opinions": [],
        "audit": [],
        "artifacts": {},
        "insufficient_data": False,
    }
    config = {"configurable": {"thread_id": thread_id or f"{corp_code}-{fiscal_year}"}}
    final: AgentState = graph.invoke(initial, config=config)  # type: ignore[assignment]
    return final
