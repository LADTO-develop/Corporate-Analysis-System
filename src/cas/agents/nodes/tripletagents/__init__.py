"""Agno-backed Stage 2 triplet agent package."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from pydantic import BaseModel

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import ChairReportOutput, EvidenceAuditOutput, QuantCreditOutput
from cas.agents.stage2_policy import stage2_policy_version
from cas.agents.stage2_prompt_contracts import stage2_prompt_contract_version
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.agents.stage2_specs import Stage2AgentRole
from cas.agents.state import Recommendation
from cas.llm.usage import aggregate_role_usage, normalize_usage_record, usage_for_cache_hit
from cas.utils.live_cache import read_json_cache, stable_cache_key, write_json_cache

from .chair_report_agent import run_chair_report_agent
from .evidence_audit_agent import run_evidence_audit_agent
from .quant_credit_agent import run_quant_credit_agent
from .review_qa_agent import run_review_qa_agent
from .risk_recall_qa_agent import run_risk_recall_qa_agent

OutputT = TypeVar("OutputT")
STAGE2_ROLE_CACHE_VERSION = "stage2_llm_role_response_v1"
_PRIMARY_ROLES: tuple[Stage2AgentRole, ...] = (
    "quant_credit",
    "evidence_audit",
    "chair_report",
)


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
    role_cache_hits: dict[str, bool] = {}
    role_cache_keys: dict[str, str] = {}
    role_token_usage: dict[str, dict[str, Any]] = {}
    runtime = _resolved_runtime_config(runtime_config)
    cache_env = runtime.cache_env()
    parallel_enabled = _parallel_independent_agents_enabled(runtime)
    if parallel_enabled:
        with ThreadPoolExecutor(max_workers=2) as executor:
            quant_future = executor.submit(
                _cached_role_call,
                "quant_credit",
                run_quant_credit_agent,
                QuantCreditOutput,
                timings,
                role_cache_hits,
                role_cache_keys,
                role_token_usage,
                cache_env,
                _role_cache_payload(
                    role="quant_credit",
                    bundle=bundle,
                    model_provider=quant_model_provider or model_provider,
                    model_name=quant_model_name or model_name,
                    max_tokens=max_tokens,
                ),
                bundle=bundle,
                model_provider=quant_model_provider or model_provider,
                model_name=quant_model_name or model_name,
                max_tokens=max_tokens,
                runtime_config=runtime,
            )
            evidence_future = executor.submit(
                _cached_role_call,
                "evidence_audit",
                run_evidence_audit_agent,
                EvidenceAuditOutput,
                timings,
                role_cache_hits,
                role_cache_keys,
                role_token_usage,
                cache_env,
                _role_cache_payload(
                    role="evidence_audit",
                    bundle=bundle,
                    model_provider=evidence_model_provider or model_provider,
                    model_name=evidence_model_name or model_name,
                    max_tokens=max_tokens,
                ),
                bundle=bundle,
                model_provider=evidence_model_provider or model_provider,
                model_name=evidence_model_name or model_name,
                max_tokens=max_tokens,
                runtime_config=runtime,
            )
            quant_credit = quant_future.result()
            evidence_audit = evidence_future.result()
    else:
        quant_credit = _cached_role_call(
            "quant_credit",
            run_quant_credit_agent,
            QuantCreditOutput,
            timings,
            role_cache_hits,
            role_cache_keys,
            role_token_usage,
            cache_env,
            _role_cache_payload(
                role="quant_credit",
                bundle=bundle,
                model_provider=quant_model_provider or model_provider,
                model_name=quant_model_name or model_name,
                max_tokens=max_tokens,
            ),
            bundle=bundle,
            model_provider=quant_model_provider or model_provider,
            model_name=quant_model_name or model_name,
            max_tokens=max_tokens,
            runtime_config=runtime,
        )
        evidence_audit = _cached_role_call(
            "evidence_audit",
            run_evidence_audit_agent,
            EvidenceAuditOutput,
            timings,
            role_cache_hits,
            role_cache_keys,
            role_token_usage,
            cache_env,
            _role_cache_payload(
                role="evidence_audit",
                bundle=bundle,
                model_provider=evidence_model_provider or model_provider,
                model_name=evidence_model_name or model_name,
                max_tokens=max_tokens,
            ),
            bundle=bundle,
            model_provider=evidence_model_provider or model_provider,
            model_name=evidence_model_name or model_name,
            max_tokens=max_tokens,
            runtime_config=runtime,
        )
    chair_report = _cached_role_call(
        "chair_report",
        run_chair_report_agent,
        ChairReportOutput,
        timings,
        role_cache_hits,
        role_cache_keys,
        role_token_usage,
        cache_env,
        _role_cache_payload(
            role="chair_report",
            bundle=bundle,
            recommendation=recommendation,
            confidence=confidence,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            model_provider=chair_model_provider or model_provider,
            model_name=chair_model_name or model_name,
            max_tokens=max_tokens,
        ),
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
        diagnostics["role_cache_hits"] = dict(role_cache_hits)
        diagnostics["role_cache_keys"] = dict(role_cache_keys)
        diagnostics["role_cache_hit_count"] = sum(
            1 for role in _PRIMARY_ROLES if role_cache_hits.get(role) is True
        )
        diagnostics["role_cache_all_hit"] = all(
            role_cache_hits.get(role) is True for role in _PRIMARY_ROLES
        )
        diagnostics["role_token_usage"] = dict(role_token_usage)
        diagnostics["token_usage_totals"] = aggregate_role_usage(role_token_usage)
    return quant_credit, evidence_audit, chair_report


def _cached_role_call(
    role: str,
    fn: Callable[..., OutputT],
    schema: type[BaseModel],
    timings: dict[str, float],
    role_cache_hits: dict[str, bool],
    role_cache_keys: dict[str, str],
    role_token_usage: dict[str, dict[str, Any]],
    cache_env: Mapping[str, str],
    cache_payload: Mapping[str, object],
    **kwargs: object,
) -> OutputT:
    started_at = time.perf_counter()
    model_provider = str(kwargs.get("model_provider") or "")
    model_name = str(kwargs.get("model_name") or "")
    cache_key = stable_cache_key(cache_payload)
    cached_output = _read_cached_role_output(
        cache_key=cache_key,
        schema=schema,
        env=cache_env,
    )
    if cached_output is not None:
        output, cached_usage = cached_output
        role_cache_hits[role] = True
        role_cache_keys[role] = cache_key
        role_token_usage[role] = usage_for_cache_hit(
            cached_usage,
            provider=model_provider,
            model_name=model_name,
            role=role,
        )
        timings[role] = round(time.perf_counter() - started_at, 4)
        return output  # type: ignore[return-value]

    usage: dict[str, object] = {}
    try:
        output = fn(**kwargs, usage=usage)
        normalized_usage = normalize_usage_record(
            usage,
            provider=model_provider,
            model_name=model_name,
            role=role,
            cache_hit=False,
            billable=bool(usage),
        )
        role_cache_hits[role] = False
        role_cache_keys[role] = cache_key
        role_token_usage[role] = normalized_usage
        _write_cached_role_output(
            cache_key=cache_key,
            cache_version=STAGE2_ROLE_CACHE_VERSION,
            output=output,
            usage=normalized_usage,
            env=cache_env,
        )
        return output
    finally:
        timings[role] = round(time.perf_counter() - started_at, 4)


def _role_cache_payload(
    *,
    role: Stage2AgentRole,
    bundle: Stage2InputBundle,
    model_provider: str,
    model_name: str,
    max_tokens: int,
    recommendation: Recommendation | None = None,
    confidence: float | None = None,
    quant_credit: QuantCreditOutput | None = None,
    evidence_audit: EvidenceAuditOutput | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "cache_version": STAGE2_ROLE_CACHE_VERSION,
        "role": role,
        "prompt_contract_version": stage2_prompt_contract_version(role),
        "stage2_policy_version": stage2_policy_version(),
        "model_provider": model_provider,
        "model_name": model_name,
        "max_tokens": max_tokens,
        "stage2_input_bundle": bundle.to_compact_prompt_payload(role=role),
    }
    if role == "chair_report":
        payload.update(
            {
                "recommendation": recommendation,
                "confidence": confidence,
                "agent_outputs": {
                    "quant_credit": (
                        quant_credit.model_dump(mode="json") if quant_credit is not None else {}
                    ),
                    "evidence_audit": (
                        evidence_audit.model_dump(mode="json")
                        if evidence_audit is not None
                        else {}
                    ),
                },
            }
        )
    return payload


def _read_cached_role_output(
    *,
    cache_key: str,
    schema: type[BaseModel],
    env: Mapping[str, str],
) -> tuple[Any, Mapping[str, object] | None] | None:
    cached_payload = read_json_cache(
        "llm_stage2_roles",
        cache_key,
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
        env=env,
    )
    if cached_payload is None:
        return None
    response_payload = cached_payload.get("response", cached_payload)
    try:
        output = schema.model_validate(response_payload)
    except ValueError:
        return None
    usage = cached_payload.get("usage")
    return output, usage if isinstance(usage, Mapping) else None


def _write_cached_role_output(
    *,
    cache_key: str,
    cache_version: str,
    output: BaseModel,
    usage: Mapping[str, object],
    env: Mapping[str, str],
) -> None:
    write_json_cache(
        "llm_stage2_roles",
        cache_key,
        {
            "cache_version": cache_version,
            "response": output.model_dump(mode="json"),
            "usage": dict(usage),
        },
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
        env=env,
    )


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
