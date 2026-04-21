"""Audit-trail formatting and export.

The audit trail is built via the ``append_audit`` reducer on the LangGraph
state (see ``bfd.agents.state``). This module turns the accumulated list of
``AuditEntry`` objects into something displayable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bfd.agents.state import AuditEntry
from bfd.utils.io import ensure_dir


def to_dataframe(entries: list[AuditEntry] | list[dict[str, Any]]) -> pd.DataFrame:
    """Coerce a heterogeneous list of audit entries into a DataFrame."""
    rows: list[dict[str, Any]] = []
    for e in entries:
        if isinstance(e, AuditEntry):
            d = e.model_dump()
        else:
            d = dict(e)
        rows.append(d)
    return pd.DataFrame(rows)


def to_markdown(entries: list[AuditEntry] | list[dict[str, Any]]) -> str:
    """Render the audit trail as a markdown table ordered by timestamp."""
    df = to_dataframe(entries)
    if df.empty:
        return "_(감사 기록 없음)_"
    df = df.sort_values("timestamp").reset_index(drop=True)

    lines = ["| 시각 | 노드 | 요약 | 지표 |", "|---|---|---|---|"]
    for _, row in df.iterrows():
        metrics = row.get("metrics", {}) or {}
        metrics_s = ", ".join(f"`{k}={v:.3f}`" for k, v in metrics.items() if isinstance(v, (int, float)))
        summary = str(row.get("summary", "")).replace("|", r"\|")
        lines.append(f"| {row['timestamp']} | `{row['node']}` | {summary} | {metrics_s} |")
    return "\n".join(lines)


def export(
    entries: list[AuditEntry] | list[dict[str, Any]],
    output_dir: str | Path,
    *,
    basename: str = "audit",
) -> dict[str, str]:
    """Write the audit trail to JSON and markdown files. Returns paths."""
    out_dir = ensure_dir(output_dir)
    df = to_dataframe(entries)

    json_path = out_dir / f"{basename}.json"
    md_path = out_dir / f"{basename}.md"
    parquet_path = out_dir / f"{basename}.parquet"

    json_path.write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(to_markdown(entries), encoding="utf-8")
    if not df.empty:
        df.to_parquet(parquet_path, index=False)

    return {"json": str(json_path), "markdown": str(md_path), "parquet": str(parquet_path)}
