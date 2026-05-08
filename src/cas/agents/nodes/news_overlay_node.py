"""Placeholder node for future cached-news/crawling integration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cas.agents.state import AgentState, AuditEntry


def run(state: AgentState) -> dict[str, Any]:
    """Keep the news-cache step in the graph without implementing crawling logic."""
    audit = AuditEntry(
        node="news_cache",
        timestamp=_now(),
        summary="News-cache/crawling integration is reserved as a placeholder.",
    )
    return {
        "news_cache_snapshot": {
            "status": "placeholder",
            "source": "not_implemented",
        },
        "audit": [audit],
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
