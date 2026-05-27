"""Diagnostics and text helpers for post-committee QA agents."""

from __future__ import annotations

from typing import Any

from cas.agents.nodes.evidence_profile import _safe_float
from cas.llm.usage import aggregate_role_usage


def _append_sentence(base: str, sentence: str) -> str:
    cleaned = base.strip()
    if not cleaned:
        return sentence
    if sentence in cleaned:
        return cleaned
    if cleaned.endswith((".", "!", "?")):
        return f"{cleaned} {sentence}"
    return f"{cleaned}. {sentence}"


def _prepend_unique_text(raw_items: object, item: str) -> list[str]:
    if isinstance(raw_items, list | tuple | set):
        existing = [str(value) for value in raw_items if str(value)]
    else:
        existing = []
    return [item, *[value for value in existing if value != item]]


def _merge_review_qa_diagnostics(
    runtime_diagnostics: dict[str, Any],
    review_qa_diagnostics: dict[str, Any],
) -> None:
    _merge_post_committee_qa_diagnostics(
        runtime_diagnostics,
        review_qa_diagnostics,
        role="review_qa",
    )


def _merge_post_committee_qa_diagnostics(
    runtime_diagnostics: dict[str, Any],
    qa_diagnostics: dict[str, Any],
    *,
    role: str,
) -> None:
    existing_timings = runtime_diagnostics.get("agent_elapsed_seconds")
    if not isinstance(existing_timings, dict):
        existing_timings = {}
    qa_timings = qa_diagnostics.get("agent_elapsed_seconds")
    qa_elapsed_seconds = 0.0
    if isinstance(qa_timings, dict):
        existing_timings.update(qa_timings)
        qa_elapsed = _safe_float(qa_timings.get(role))
        if qa_elapsed is not None:
            qa_elapsed_seconds = qa_elapsed
    runtime_diagnostics["agent_elapsed_seconds"] = existing_timings
    runtime_diagnostics["agent_elapsed_seconds_sum"] = round(
        sum(float(value) for value in existing_timings.values()),
        4,
    )
    current_total = _safe_float(runtime_diagnostics.get("stage2_total_elapsed_seconds"))
    if current_total is not None and qa_elapsed_seconds:
        runtime_diagnostics["stage2_total_elapsed_seconds"] = round(
            current_total + qa_elapsed_seconds,
            4,
        )
    runtime_diagnostics[f"{role}_cache_hit"] = bool(qa_diagnostics.get(f"{role}_cache_hit", False))
    if qa_diagnostics.get(f"{role}_cache_key"):
        runtime_diagnostics[f"{role}_cache_key"] = qa_diagnostics[f"{role}_cache_key"]
    _merge_role_token_usage(runtime_diagnostics, qa_diagnostics)


def _merge_role_token_usage(
    runtime_diagnostics: dict[str, Any],
    qa_diagnostics: dict[str, Any],
) -> None:
    existing_usage = runtime_diagnostics.get("role_token_usage")
    if not isinstance(existing_usage, dict):
        existing_usage = {}
    qa_usage = qa_diagnostics.get("role_token_usage")
    if isinstance(qa_usage, dict):
        existing_usage.update(qa_usage)
    runtime_diagnostics["role_token_usage"] = existing_usage
    runtime_diagnostics["token_usage_totals"] = aggregate_role_usage(existing_usage)


__all__ = [
    "_append_sentence",
    "_merge_post_committee_qa_diagnostics",
    "_merge_review_qa_diagnostics",
    "_prepend_unique_text",
]
