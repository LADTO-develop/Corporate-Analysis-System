"""Agno-backed Stage 2 triplet agent package."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.agents.state import Recommendation

from .chair_report_agent import run_chair_report_agent
from .evidence_audit_agent import run_evidence_audit_agent
from .quant_credit_agent import run_quant_credit_agent
from .review_qa_agent import run_review_qa_agent
from .risk_recall_qa_agent import run_risk_recall_qa_agent

OutputT = TypeVar("OutputT")


def run_triplet_agents(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    model_name: str,
    model_provider: str = "openai",
    quant_model_provider: str | None = None,
    quant_model_name: str | None = None,
    evidence_model_provider: str | None = None,
    evidence_model_name: str | None = None,
    chair_model_provider: str | None = None,
    chair_model_name: str | None = None,
    max_tokens: int,
    runtime_config: Stage2RuntimeConfig | None = None,
    diagnostics: dict[str, object] | None = None,
) -> tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput]:
    """Run QuantCredit, EvidenceAudit, and ChairReport Agno agents in order."""
    timings: dict[str, float] = {}
    runtime = _resolved_runtime_config(runtime_config)
    parallel_enabled = _parallel_independent_agents_enabled(runtime)
    if parallel_enabled:
        with ThreadPoolExecutor(max_workers=2) as executor:
            quant_future = executor.submit(
                _timed_call,
                "quant_credit",
                run_quant_credit_agent,
                timings,
                bundle=bundle,
                model_provider=quant_model_provider or model_provider,
                model_name=quant_model_name or model_name,
                max_tokens=max_tokens,
                runtime_config=runtime,
            )
            evidence_future = executor.submit(
                _timed_call,
                "evidence_audit",
                run_evidence_audit_agent,
                timings,
                bundle=bundle,
                model_provider=evidence_model_provider or model_provider,
                model_name=evidence_model_name or model_name,
                max_tokens=max_tokens,
                runtime_config=runtime,
            )
            quant_credit = quant_future.result()
            evidence_audit = evidence_future.result()
    else:
        quant_credit = _timed_call(
            "quant_credit",
            run_quant_credit_agent,
            timings,
            bundle=bundle,
            model_provider=quant_model_provider or model_provider,
            model_name=quant_model_name or model_name,
            max_tokens=max_tokens,
            runtime_config=runtime,
        )
        evidence_audit = _timed_call(
            "evidence_audit",
            run_evidence_audit_agent,
            timings,
            bundle=bundle,
            model_provider=evidence_model_provider or model_provider,
            model_name=evidence_model_name or model_name,
            max_tokens=max_tokens,
            runtime_config=runtime,
        )
    chair_report = _timed_call(
        "chair_report",
        run_chair_report_agent,
        timings,
        bundle=bundle,
        recommendation=recommendation,
        confidence=confidence,
        quant_credit=quant_credit,
        evidence_audit=evidence_audit,
        model_provider=chair_model_provider or model_provider,
        model_name=chair_model_name or model_name,
        max_tokens=max_tokens,
        runtime_config=runtime,
    )
    if diagnostics is not None:
        diagnostics["agent_elapsed_seconds"] = dict(timings)
        diagnostics["parallel_independent_agents"] = parallel_enabled
    return quant_credit, evidence_audit, chair_report


def _timed_call(
    role: str,
    fn: Callable[..., OutputT],
    timings: dict[str, float],
    **kwargs: object,
) -> OutputT:
    started_at = time.perf_counter()
    try:
        return fn(**kwargs)
    finally:
        timings[role] = round(time.perf_counter() - started_at, 4)


def _resolved_runtime_config(
    runtime_config: Stage2RuntimeConfig | None,
) -> Stage2RuntimeConfig:
    return runtime_config or Stage2RuntimeConfig.from_env(os.environ)


def _parallel_independent_agents_enabled(runtime_config: Stage2RuntimeConfig) -> bool:
    return bool(runtime_config.parallel_independent_agents)


__all__ = [
    "run_chair_report_agent",
    "run_evidence_audit_agent",
    "run_quant_credit_agent",
    "run_review_qa_agent",
    "run_risk_recall_qa_agent",
    "run_triplet_agents",
]
