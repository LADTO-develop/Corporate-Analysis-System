"""Runtime config and cache helpers for post-committee QA agents."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
    ReviewQAOutput,
    RiskRecallQAOutput,
)
from cas.agents.stage2_policy import stage2_policy_version
from cas.agents.stage2_prompt_contracts import stage2_prompt_contract_version
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.utils.live_cache import read_json_cache, stable_cache_key, write_json_cache

_REVIEW_QA_CACHE_VERSION = "stage2_review_qa_v4"
_RISK_RECALL_QA_CACHE_VERSION = "stage2_risk_recall_qa_v3"
_STAGE2_RUNTIME_CONFIG_OVERRIDE: ContextVar[Stage2RuntimeConfig | None] = ContextVar(
    "post_committee_stage2_runtime_config_override",
    default=None,
)
_QAOutputT = TypeVar("_QAOutputT", bound=BaseModel)


@contextmanager
def post_committee_runtime_config_override(config: Stage2RuntimeConfig) -> Iterator[None]:
    """Temporarily override Stage 2 QA runtime config for the current context."""
    token = _STAGE2_RUNTIME_CONFIG_OVERRIDE.set(config)
    try:
        yield
    finally:
        _STAGE2_RUNTIME_CONFIG_OVERRIDE.reset(token)


def _stage2_runtime_config() -> Stage2RuntimeConfig:
    override = _STAGE2_RUNTIME_CONFIG_OVERRIDE.get()
    if override is not None:
        return override
    return Stage2RuntimeConfig.from_env(os.environ)


def _run_review_qa_agent_with_cache(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
) -> tuple[ReviewQAOutput, dict[str, Any]]:
    runtime_config = _stage2_runtime_config()
    model_provider = str(runtime_config.review_qa_provider_resolved)
    model_name = str(runtime_config.review_qa_model_resolved)
    return run_cached_optional_agent(
        role="review_qa",
        cache_namespace="llm_stage2_review_qa",
        cache_version=_REVIEW_QA_CACHE_VERSION,
        cache_env=runtime_config.cache_env(),
        payload_builder=lambda provider, name: _review_qa_cache_payload(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=provider,
            model_name=name,
        ),
        agent_callable=lambda: _call_review_qa_agent(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=model_provider,
            model_name=model_name,
            max_tokens=int(runtime_config.review_qa_max_tokens),
            runtime_config=runtime_config,
        ),
        schema=ReviewQAOutput,
        model_provider=model_provider,
        model_name=model_name,
    )


def _review_qa_cache_payload(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
    model_provider: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "cache_version": _REVIEW_QA_CACHE_VERSION,
        "prompt_contract_version": stage2_prompt_contract_version("review_qa"),
        "stage2_policy_version": stage2_policy_version(),
        "model_provider": model_provider,
        "model_name": model_name,
        "stage2_input_bundle": bundle.to_compact_prompt_payload(role="review_qa"),
        "committee_view": committee_view,
        "agent_outputs": {
            "quant_credit": quant_credit.model_dump(mode="json"),
            "evidence_audit": evidence_audit.model_dump(mode="json"),
            "chair_report": chair_report.model_dump(mode="json"),
        },
        "trigger_reasons": trigger_reasons,
    }


def _stage2_review_qa_provider() -> str:
    return str(_stage2_runtime_config().review_qa_provider_resolved)


def _stage2_review_qa_model() -> str:
    return str(_stage2_runtime_config().review_qa_model_resolved)


def _stage2_review_qa_max_tokens() -> int:
    return int(_stage2_runtime_config().review_qa_max_tokens)


def _call_review_qa_agent(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
    model_provider: str,
    model_name: str,
    max_tokens: int,
    runtime_config: Stage2RuntimeConfig,
) -> ReviewQAOutput:
    review_module = import_module("cas.agents.nodes.tripletagents.review_qa_agent")
    return cast(
        ReviewQAOutput,
        review_module.run_review_qa_agent(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=model_provider,
            model_name=model_name,
            max_tokens=max_tokens,
            runtime_config=runtime_config,
        ),
    )


def _run_risk_recall_qa_agent_with_cache(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
) -> tuple[RiskRecallQAOutput, dict[str, Any]]:
    runtime_config = _stage2_runtime_config()
    model_provider = str(runtime_config.risk_recall_qa_provider_resolved)
    model_name = str(runtime_config.risk_recall_qa_model_resolved)
    return run_cached_optional_agent(
        role="risk_recall_qa",
        cache_namespace="llm_stage2_risk_recall_qa",
        cache_version=_RISK_RECALL_QA_CACHE_VERSION,
        cache_env=runtime_config.cache_env(),
        payload_builder=lambda provider, name: _risk_recall_qa_cache_payload(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=provider,
            model_name=name,
        ),
        agent_callable=lambda: _call_risk_recall_qa_agent(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=model_provider,
            model_name=model_name,
            max_tokens=int(runtime_config.risk_recall_qa_max_tokens),
            runtime_config=runtime_config,
        ),
        schema=RiskRecallQAOutput,
        model_provider=model_provider,
        model_name=model_name,
    )


def _risk_recall_qa_cache_payload(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
    model_provider: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "cache_version": _RISK_RECALL_QA_CACHE_VERSION,
        "prompt_contract_version": stage2_prompt_contract_version("risk_recall_qa"),
        "stage2_policy_version": stage2_policy_version(),
        "model_provider": model_provider,
        "model_name": model_name,
        "stage2_input_bundle": bundle.to_compact_prompt_payload(role="risk_recall_qa"),
        "committee_view": committee_view,
        "agent_outputs": {
            "quant_credit": quant_credit.model_dump(mode="json"),
            "evidence_audit": evidence_audit.model_dump(mode="json"),
            "chair_report": chair_report.model_dump(mode="json"),
        },
        "trigger_reasons": trigger_reasons,
    }


def _stage2_risk_recall_qa_provider() -> str:
    return str(_stage2_runtime_config().risk_recall_qa_provider_resolved)


def _stage2_risk_recall_qa_model() -> str:
    return str(_stage2_runtime_config().risk_recall_qa_model_resolved)


def _stage2_risk_recall_qa_max_tokens() -> int:
    return int(_stage2_runtime_config().risk_recall_qa_max_tokens)


def _call_risk_recall_qa_agent(
    *,
    bundle: Stage2InputBundle,
    committee_view: dict[str, Any],
    quant_credit: QuantCreditOutput,
    evidence_audit: EvidenceAuditOutput,
    chair_report: ChairReportOutput,
    trigger_reasons: list[str],
    model_provider: str,
    model_name: str,
    max_tokens: int,
    runtime_config: Stage2RuntimeConfig,
) -> RiskRecallQAOutput:
    recall_module = import_module("cas.agents.nodes.tripletagents.risk_recall_qa_agent")
    return cast(
        RiskRecallQAOutput,
        recall_module.run_risk_recall_qa_agent(
            bundle=bundle,
            committee_view=committee_view,
            quant_credit=quant_credit,
            evidence_audit=evidence_audit,
            chair_report=chair_report,
            trigger_reasons=trigger_reasons,
            model_provider=model_provider,
            model_name=model_name,
            max_tokens=max_tokens,
            runtime_config=runtime_config,
        ),
    )


def run_cached_optional_agent(
    *,
    role: str,
    cache_namespace: str,
    cache_version: str,
    cache_env: Mapping[str, str],
    payload_builder: Callable[[str, str], dict[str, Any]],
    agent_callable: Callable[[], _QAOutputT],
    schema: type[_QAOutputT],
    model_provider: str,
    model_name: str,
) -> tuple[_QAOutputT, dict[str, Any]]:
    """Run an optional QA agent through the shared Stage 2 LLM cache boundary."""
    started_at = time.perf_counter()
    cache_payload = payload_builder(model_provider, model_name)
    cache_key = stable_cache_key(cache_payload)
    cached_payload = read_json_cache(
        cache_namespace,
        cache_key,
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
        env=cache_env,
    )
    if cached_payload is not None:
        response_payload = cached_payload.get("response", cached_payload)
        return schema.model_validate(response_payload), _qa_cache_diagnostics(
            role=role,
            cache_key=cache_key,
            cache_hit=True,
            started_at=started_at,
        )

    output = agent_callable()
    write_json_cache(
        cache_namespace,
        cache_key,
        {
            "cache_version": cache_version,
            "response": output.model_dump(mode="json"),
        },
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
        env=cache_env,
    )
    return output, _qa_cache_diagnostics(
        role=role,
        cache_key=cache_key,
        cache_hit=False,
        started_at=started_at,
    )


def _qa_cache_diagnostics(
    *,
    role: str,
    cache_key: str,
    cache_hit: bool,
    started_at: float,
) -> dict[str, Any]:
    return {
        f"{role}_cache_hit": cache_hit,
        f"{role}_cache_key": cache_key,
        "agent_elapsed_seconds": {role: round(time.perf_counter() - started_at, 4)},
    }


__all__ = [
    "_REVIEW_QA_CACHE_VERSION",
    "_RISK_RECALL_QA_CACHE_VERSION",
    "_review_qa_cache_payload",
    "_risk_recall_qa_cache_payload",
    "_run_review_qa_agent_with_cache",
    "_run_risk_recall_qa_agent_with_cache",
    "_stage2_review_qa_max_tokens",
    "_stage2_review_qa_model",
    "_stage2_review_qa_provider",
    "_stage2_risk_recall_qa_max_tokens",
    "_stage2_risk_recall_qa_model",
    "_stage2_risk_recall_qa_provider",
    "_stage2_runtime_config",
    "post_committee_runtime_config_override",
    "run_cached_optional_agent",
]
