"""Audit-trail formatting and export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cas.agents.state import AuditEntry
from cas.utils.io import ensure_dir


def to_dataframe(entries: list[AuditEntry] | list[dict[str, Any]]) -> pd.DataFrame:
    """Coerce a heterogeneous list of audit entries into a DataFrame."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        rows.append(entry.model_dump() if isinstance(entry, AuditEntry) else dict(entry))
    return pd.DataFrame(rows)


def to_markdown(entries: list[AuditEntry] | list[dict[str, Any]]) -> str:
    """Render the audit trail as a markdown table ordered by timestamp."""
    df = to_dataframe(entries)
    if df.empty:
        return "_(감사 추적 항목이 없습니다)_"
    df = df.sort_values("timestamp").reset_index(drop=True)

    lines = ["| 시각 | 노드 | 요약 | 지표 |", "|---|---|---|---|"]
    for _, row in df.iterrows():
        metrics = row.get("metrics", {}) or {}
        metrics_s = ", ".join(
            f"`{k}={v:.3f}`" for k, v in metrics.items() if isinstance(v, int | float)
        )
        summary = _localize_summary(str(row.get("summary", ""))).replace("|", r"\|")
        lines.append(f"| {row['timestamp']} | `{row['node']}` | {summary} | {metrics_s} |")
    return "\n".join(lines)


def _localize_summary(summary: str) -> str:
    """Translate common pipeline audit messages for the Korean report."""
    localized = summary
    replacements = {
        "External evidence collection is disabled; set CAS_ENABLE_EXTERNAL_EVIDENCE=1 to enable it.": (
            "외부근거 수집이 비활성화되어 있습니다. 활성화하려면 CAS_ENABLE_EXTERNAL_EVIDENCE=1로 설정하세요."
        ),
        "Dashboard response JSON validated against strict schema.": (
            "대시보드 응답 JSON이 엄격한 스키마 검증을 통과했습니다."
        ),
    }
    for old, new in replacements.items():
        localized = localized.replace(old, new)
    if localized.startswith("Loaded feature-master row for "):
        return localized.replace(
            "Loaded feature-master row for ", "feature-master 행을 불러왔습니다: "
        )
    if localized.startswith("Loaded dataset-backed feature snapshot with "):
        return localized.replace(
            "Loaded dataset-backed feature snapshot with ",
            "데이터셋 기반 feature snapshot을 불러왔습니다: ",
        )
    if localized.startswith("Stage 1 XGBoost inference completed: "):
        return localized.replace(
            "Stage 1 XGBoost inference completed: ",
            "Stage 1 XGBoost 추론 완료: ",
        )
    if localized.startswith("Rule engine assigned "):
        return localized.replace("Rule engine assigned ", "규칙엔진 판단 완료: ")
    if localized.startswith("Three-agent Stage 2 scaffold completed via "):
        return localized.replace(
            "Three-agent Stage 2 scaffold completed via ",
            "3개 에이전트 Stage 2 실행 완료: ",
        )
    return localized


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
