"""Stage 2 agent runner adapters.

The default runner is deterministic for CI stability. The Agno runner is wired
as an optional adapter so local demos can switch to LLM-backed structured
outputs without changing committee_node orchestration.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
)
from cas.agents.stage2_policy import stage2_policy_version
from cas.agents.stage2_prompt_contracts import (
    build_stage2_llm_client_prompt_payload,
    stage2_llm_client_prompt_contract_version,
    stage2_llm_client_prompt_contract_versions,
    stage2_prompt_contract_versions,
)
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.agents.stage2_specs import Stage2AgentRole
from cas.agents.state import Recommendation
from cas.llm.model_catalog import (
    DEFAULT_STAGE2_AGNO_MODE,
    DEFAULT_STAGE2_SINGLE_MODEL,
    DEFAULT_STAGE2_SINGLE_PROVIDER,
    is_multi_llm_committee_mode,
    stage2_role_model_default,
)
from cas.llm.usage import aggregate_role_usage, llm_usage_from_response, usage_for_cache_hit
from cas.utils.live_cache import read_json_cache, stable_cache_key, write_json_cache

QuantCreditFn = Callable[[Stage2InputBundle], QuantCreditOutput]
EvidenceAuditFn = Callable[[Stage2InputBundle], EvidenceAuditOutput]
ChairReportFn = Callable[[Stage2InputBundle, Recommendation, float], ChairReportOutput]
Stage2RunnerOutputs = tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput]
STAGE2_LLM_CACHE_VERSION = "stage2_llm_response_v3"
STAGE2_TRIPLET_AGENT_ROLES: tuple[Stage2AgentRole, ...] = (
    "quant_credit",
    "evidence_audit",
    "chair_report",
)


class Stage2LLMResponse(BaseModel):
    """Structured response expected from an LLM-backed Stage 2 run."""

    model_config = ConfigDict(extra="forbid")

    quant_credit: QuantCreditOutput
    evidence_audit: EvidenceAuditOutput
    chair_report: ChairReportOutput

    def as_outputs(self) -> Stage2RunnerOutputs:
        """Return outputs in the fixed Stage 2 role order."""
        return (self.quant_credit, self.evidence_audit, self.chair_report)


class Stage2LLMClient(Protocol):
    """Injectable client used by tests or alternate LLM providers."""

    def run_structured(
        self,
        *,
        prompt_payload: dict[str, Any],
        output_schema: type[Stage2LLMResponse],
    ) -> object:
        """Run Stage 2 with a structured response schema."""


class Stage2AgentRunner(Protocol):
    """Common interface for deterministic and future Agno-backed Stage 2 runners."""

    backend_name: str

    def run(
        self,
        *,
        bundle: Stage2InputBundle,
        recommendation: Recommendation,
        confidence: float,
    ) -> Stage2RunnerOutputs:
        """Run the three Stage 2 agents and return structured outputs."""


class _TripletAgentModule(Protocol):
    """Protocol for the lazily imported Agno triplet package."""

    def run_triplet_agents(
        self,
        *,
        bundle: Stage2InputBundle,
        recommendation: Recommendation,
        confidence: float,
        model_provider: str,
        model_name: str,
        quant_model_provider: str | None,
        quant_model_name: str | None,
        evidence_model_provider: str | None,
        evidence_model_name: str | None,
        chair_model_provider: str | None,
        chair_model_name: str | None,
        max_tokens: int,
        runtime_config: Stage2RuntimeConfig | None,
        diagnostics: dict[str, Any] | None = None,
    ) -> Stage2RunnerOutputs:
        """Run the Agno Stage 2 triplet agents."""


@dataclass(frozen=True)
class DeterministicStage2AgentRunner:
    """Deterministic Stage 2 runner used by local tests and CI."""

    quant_credit_agent: QuantCreditFn
    evidence_audit_agent: EvidenceAuditFn
    chair_report_agent: ChairReportFn
    backend_name: str = "deterministic"

    def run(
        self,
        *,
        bundle: Stage2InputBundle,
        recommendation: Recommendation,
        confidence: float,
    ) -> Stage2RunnerOutputs:
        """Run the deterministic scaffold in the fixed Stage 2 role order."""
        return (
            self.quant_credit_agent(bundle),
            self.evidence_audit_agent(bundle),
            self.chair_report_agent(bundle, recommendation, confidence),
        )


@dataclass
class AgnoStage2AgentRunner:
    """Optional Agno-backed Stage 2 runner.

    This path is intentionally opt-in. If the ``agno`` package is unavailable,
    install the optional LLM dependencies or inject a ``Stage2LLMClient`` in
    tests/local experiments. Without an injected client, the runner executes
    the three Agno triplet agents. The default routing is a single OpenAI model
    so local demos can run with only ``OPENAI_API_KEY``. ``multi_llm_committee``
    remains available for Claude/GPT/Gemini role routing when all provider keys
    are configured.
    """

    deterministic_runner: DeterministicStage2AgentRunner | None = None
    llm_client: Stage2LLMClient | None = None
    routing_mode: str = DEFAULT_STAGE2_AGNO_MODE
    model_provider: str = DEFAULT_STAGE2_SINGLE_PROVIDER
    model_name: str = DEFAULT_STAGE2_SINGLE_MODEL
    quant_model_provider: str | None = None
    quant_model_name: str | None = None
    evidence_model_provider: str | None = None
    evidence_model_name: str | None = None
    chair_model_provider: str | None = None
    chair_model_name: str | None = None
    max_tokens: int = 6000
    backend_name: str = "agno"
    fallback_on_error: bool = True
    runtime_config: Stage2RuntimeConfig | None = None
    last_run_backend_name: str = "agno"
    last_error_message: str = ""
    last_run_diagnostics: dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        *,
        bundle: Stage2InputBundle,
        recommendation: Recommendation,
        confidence: float,
    ) -> Stage2RunnerOutputs:
        """Run Stage 2 through the Agno triplet agents or an injected LLM client."""
        run_started_at = time.perf_counter()
        runtime_config = self._resolved_runtime_config()
        cache_env = runtime_config.cache_env()
        try:
            cache_key = stable_cache_key(
                _stage2_cache_payload(
                    runner=self,
                    bundle=bundle,
                    recommendation=recommendation,
                    confidence=confidence,
                )
            )
            cached_response = _read_stage2_cached_response(cache_key, env=cache_env)
            if cached_response is not None:
                response, cached_backend_name, cached_diagnostics = cached_response
                self.last_run_backend_name = cached_backend_name
                self.last_error_message = ""
                self.last_run_diagnostics = _stage2_run_diagnostics(
                    backend_name=cached_backend_name,
                    cache_hit=True,
                    cache_key=cache_key,
                    started_at=run_started_at,
                    extra={
                        **_cached_response_diagnostics(cached_diagnostics),
                        "prompt_contract_versions": _prompt_contract_versions(self),
                    },
                )
                return response.as_outputs()

            if self.llm_client is not None:
                client_started_at = time.perf_counter()
                prompt_payload = _build_prompt_payload(
                    bundle=bundle,
                    recommendation=recommendation,
                    confidence=confidence,
                    draft_outputs=self._draft_outputs(bundle, recommendation, confidence),
                )
                raw_response = self.llm_client.run_structured(
                    prompt_payload=prompt_payload,
                    output_schema=Stage2LLMResponse,
                )
                client_usage = llm_usage_from_response(
                    raw_response,
                    provider=self.model_provider,
                    model_name=self.model_name,
                )
                successful_backend_name = self.backend_name
                agent_timings = {"llm_client": round(time.perf_counter() - client_started_at, 4)}
                runner_diagnostics: dict[str, Any] = {
                    "agent_elapsed_seconds": agent_timings,
                    "prompt_contract_versions": _prompt_contract_versions(self),
                    "role_token_usage": {"llm_client": client_usage},
                    "token_usage_totals": aggregate_role_usage({"llm_client": client_usage}),
                }
            else:
                runner_diagnostics = {
                    "prompt_contract_versions": _prompt_contract_versions(self),
                }
                raw_response = _run_triplet_agents_with_agno(
                    bundle=bundle,
                    recommendation=recommendation,
                    confidence=confidence,
                    model_provider=self.model_provider,
                    model_name=self.model_name,
                    quant_model_provider=self._role_provider("quant_credit"),
                    quant_model_name=self._role_model_name("quant_credit"),
                    evidence_model_provider=self._role_provider("evidence_audit"),
                    evidence_model_name=self._role_model_name("evidence_audit"),
                    chair_model_provider=self._role_provider("chair_report"),
                    chair_model_name=self._role_model_name("chair_report"),
                    max_tokens=self.max_tokens,
                    runtime_config=runtime_config,
                    diagnostics=runner_diagnostics,
                )
                successful_backend_name = self.backend_name
            response = _coerce_llm_response(raw_response)
            _write_stage2_cached_response(
                cache_key=cache_key,
                backend_name=successful_backend_name,
                response=response,
                diagnostics=runner_diagnostics,
                env=cache_env,
            )
            outputs = response.as_outputs()
            self.last_run_backend_name = successful_backend_name
            self.last_error_message = ""
            self.last_run_diagnostics = _stage2_run_diagnostics(
                backend_name=successful_backend_name,
                cache_hit=False,
                cache_key=cache_key,
                started_at=run_started_at,
                agent_timings=_coerce_agent_timings(
                    runner_diagnostics.get("agent_elapsed_seconds")
                ),
                extra=runner_diagnostics,
            )
            return outputs
        except Exception as error:
            if self.deterministic_runner is None or not self.fallback_on_error:
                raise
            self.last_run_backend_name = "agno_fallback_deterministic"
            self.last_error_message = str(error)
            outputs = self.deterministic_runner.run(
                bundle=bundle,
                recommendation=recommendation,
                confidence=confidence,
            )
            self.last_run_diagnostics = _stage2_run_diagnostics(
                backend_name="agno_fallback_deterministic",
                cache_hit=False,
                cache_key=cache_key if "cache_key" in locals() else "",
                started_at=run_started_at,
                extra={
                    "error_message": str(error),
                    "prompt_contract_versions": _prompt_contract_versions(self),
                },
            )
            return outputs

    def _draft_outputs(
        self,
        bundle: Stage2InputBundle,
        recommendation: Recommendation,
        confidence: float,
    ) -> dict[str, Any]:
        if self.deterministic_runner is None:
            return {}
        outputs = self.deterministic_runner.run(
            bundle=bundle,
            recommendation=recommendation,
            confidence=confidence,
        )
        return {
            "quant_credit": outputs[0].model_dump(mode="json"),
            "evidence_audit": outputs[1].model_dump(mode="json"),
            "chair_report": outputs[2].model_dump(mode="json"),
        }

    def _role_provider(self, role: str) -> str | None:
        explicit = {
            "quant_credit": self.quant_model_provider,
            "evidence_audit": self.evidence_model_provider,
            "chair_report": self.chair_model_provider,
        }[role]
        if explicit:
            return explicit
        if self._uses_multi_llm_committee():
            return cast(str, stage2_role_model_default(role).provider)
        return None

    def _role_model_name(self, role: str) -> str | None:
        explicit = {
            "quant_credit": self.quant_model_name,
            "evidence_audit": self.evidence_model_name,
            "chair_report": self.chair_model_name,
        }[role]
        if explicit:
            return explicit
        if self._uses_multi_llm_committee():
            return cast(str, stage2_role_model_default(role).model)
        return None

    def _uses_multi_llm_committee(self) -> bool:
        return bool(is_multi_llm_committee_mode(self.routing_mode))

    def _resolved_runtime_config(self) -> Stage2RuntimeConfig:
        return self.runtime_config or Stage2RuntimeConfig.from_env(os.environ)


def _build_prompt_payload(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    draft_outputs: dict[str, Any],
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        build_stage2_llm_client_prompt_payload(
            recommendation=recommendation,
            confidence=confidence,
            stage2_input_bundle=bundle.to_compact_prompt_payload(role="stage2"),
            deterministic_draft_outputs=draft_outputs,
        ),
    )


def _stage2_cache_payload(
    *,
    runner: AgnoStage2AgentRunner,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
) -> dict[str, Any]:
    return {
        "cache_version": STAGE2_LLM_CACHE_VERSION,
        "prompt_contract": _prompt_contract_version(runner),
        "prompt_contract_versions": _prompt_contract_versions(runner),
        "stage2_policy_version": stage2_policy_version(),
        "runner": {
            "backend_name": runner.backend_name,
            "routing_mode": runner.routing_mode,
            "model_provider": runner.model_provider,
            "model_name": runner.model_name,
            "quant_model_provider": runner._role_provider("quant_credit"),
            "quant_model_name": runner._role_model_name("quant_credit"),
            "evidence_model_provider": runner._role_provider("evidence_audit"),
            "evidence_model_name": runner._role_model_name("evidence_audit"),
            "chair_model_provider": runner._role_provider("chair_report"),
            "chair_model_name": runner._role_model_name("chair_report"),
            "max_tokens": runner.max_tokens,
            "llm_client_class": (
                type(runner.llm_client).__name__ if runner.llm_client is not None else ""
            ),
        },
        "recommendation": recommendation,
        "confidence": confidence,
        "stage2_input_bundle": bundle.to_compact_prompt_payload(role="stage2"),
        "deterministic_draft_outputs": (
            runner._draft_outputs(bundle, recommendation, confidence)
            if runner.llm_client is not None
            else {}
        ),
    }


def _prompt_contract_version(runner: AgnoStage2AgentRunner) -> str:
    if runner.llm_client is not None:
        return str(stage2_llm_client_prompt_contract_version())
    return "stage2_triplet_prompt_contract_v1"


def _prompt_contract_versions(runner: AgnoStage2AgentRunner) -> dict[str, str]:
    if runner.llm_client is not None:
        return cast(dict[str, str], stage2_llm_client_prompt_contract_versions())
    return cast(dict[str, str], stage2_prompt_contract_versions(STAGE2_TRIPLET_AGENT_ROLES))


def _read_stage2_cached_response(
    cache_key: str,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[Stage2LLMResponse, str, dict[str, Any]] | None:
    cached_payload = read_json_cache(
        "llm_stage2",
        cache_key,
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
        env=env,
    )
    if cached_payload is None:
        return None
    response_payload = cached_payload.get("response", cached_payload)
    try:
        response = Stage2LLMResponse.model_validate(response_payload)
    except ValueError:
        return None
    backend_name = str(cached_payload.get("backend_name") or "agno")
    if not backend_name.endswith("_cache"):
        backend_name = f"{backend_name}_cache"
    diagnostics = cached_payload.get("diagnostics")
    return response, backend_name, dict(diagnostics) if isinstance(diagnostics, dict) else {}


def _write_stage2_cached_response(
    *,
    cache_key: str,
    backend_name: str,
    response: Stage2LLMResponse,
    diagnostics: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    write_json_cache(
        "llm_stage2",
        cache_key,
        {
            "cache_version": STAGE2_LLM_CACHE_VERSION,
            "backend_name": backend_name,
            "response": response.model_dump(mode="json"),
            "diagnostics": dict(diagnostics or {}),
        },
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
        env=env,
    )


def _stage2_run_diagnostics(
    *,
    backend_name: str,
    cache_hit: bool,
    cache_key: str,
    started_at: float,
    agent_timings: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timings = agent_timings or {}
    role_cache_hits = _coerce_role_cache_hits((extra or {}).get("role_cache_hits"))
    role_cache_hit_count = sum(1 for value in role_cache_hits.values() if value is True)
    role_cache_any_hit = bool(role_cache_hit_count)
    role_cache_all_hit = all(
        role_cache_hits.get(role) is True for role in STAGE2_TRIPLET_AGENT_ROLES
    ) if role_cache_hits else False
    diagnostics = {
        "backend_name": backend_name,
        "cache_hit": cache_hit or role_cache_any_hit,
        "response_cache_hit": cache_hit,
        "cache_key": cache_key,
        "stage2_total_elapsed_seconds": round(time.perf_counter() - started_at, 4),
        "agent_elapsed_seconds": timings,
        "agent_elapsed_seconds_sum": round(sum(timings.values()), 4),
        "role_cache_hit_count": role_cache_hit_count,
        "role_cache_any_hit": role_cache_any_hit,
        "role_cache_all_hit": role_cache_all_hit,
    }
    if extra:
        for key, value in extra.items():
            if key != "agent_elapsed_seconds":
                diagnostics[key] = value
    if "role_token_usage" in diagnostics and "token_usage_totals" not in diagnostics:
        role_usage = diagnostics.get("role_token_usage")
        if isinstance(role_usage, Mapping):
            diagnostics["token_usage_totals"] = aggregate_role_usage(role_usage)
    return diagnostics


def _coerce_agent_timings(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    timings: dict[str, float] = {}
    for role, elapsed in value.items():
        try:
            timings[str(role)] = round(float(elapsed), 4)
        except (TypeError, ValueError):
            continue
    return timings


def _cached_response_diagnostics(cached_diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = {
        str(key): value
        for key, value in cached_diagnostics.items()
        if str(key)
        not in {
            "agent_elapsed_seconds",
            "agent_elapsed_seconds_sum",
            "backend_name",
            "cache_hit",
            "cache_key",
            "response_cache_hit",
            "stage2_total_elapsed_seconds",
        }
    }
    role_cache_hits = {
        role: True for role in STAGE2_TRIPLET_AGENT_ROLES
    }
    diagnostics["role_cache_hits"] = role_cache_hits
    diagnostics["role_cache_hit_count"] = len(STAGE2_TRIPLET_AGENT_ROLES)
    diagnostics["role_cache_all_hit"] = True
    diagnostics["role_cache_any_hit"] = True
    raw_role_usage = diagnostics.get("role_token_usage")
    if isinstance(raw_role_usage, Mapping):
        role_usage: dict[str, dict[str, Any]] = {}
        for role, raw_usage in raw_role_usage.items():
            if not isinstance(raw_usage, Mapping):
                continue
            provider = str(raw_usage.get("provider") or "")
            model_name = str(raw_usage.get("model") or raw_usage.get("model_name") or "")
            role_usage[str(role)] = usage_for_cache_hit(
                raw_usage,
                provider=provider,
                model_name=model_name,
                role=str(role),
            )
        diagnostics["role_token_usage"] = role_usage
        diagnostics["token_usage_totals"] = aggregate_role_usage(role_usage)
    return diagnostics


def _coerce_role_cache_hits(value: object) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        return {}
    return {str(role): bool(hit) for role, hit in value.items()}


def _run_triplet_agents_with_agno(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    model_provider: str,
    model_name: str,
    quant_model_provider: str | None,
    quant_model_name: str | None,
    evidence_model_provider: str | None,
    evidence_model_name: str | None,
    chair_model_provider: str | None,
    chair_model_name: str | None,
    max_tokens: int,
    runtime_config: Stage2RuntimeConfig | None,
    diagnostics: dict[str, Any] | None = None,
) -> Stage2LLMResponse:
    try:
        triplet_module = cast(
            _TripletAgentModule,
            import_module("cas.agents.nodes.tripletagents"),
        )
    except ImportError as error:
        raise RuntimeError(
            "CAS_STAGE2_RUNNER=agno could not import the Agno triplet agents."
        ) from error

    raw_outputs = triplet_module.run_triplet_agents(
        bundle=bundle,
        recommendation=recommendation,
        confidence=confidence,
        model_provider=model_provider,
        model_name=model_name,
        quant_model_provider=quant_model_provider,
        quant_model_name=quant_model_name,
        evidence_model_provider=evidence_model_provider,
        evidence_model_name=evidence_model_name,
        chair_model_provider=chair_model_provider,
        chair_model_name=chair_model_name,
        max_tokens=max_tokens,
        runtime_config=runtime_config,
        diagnostics=diagnostics,
    )
    if not isinstance(raw_outputs, tuple) or len(raw_outputs) != 3:
        raise TypeError("Agno triplet agents must return exactly three Stage 2 outputs.")
    return Stage2LLMResponse(
        quant_credit=QuantCreditOutput.model_validate(raw_outputs[0]),
        evidence_audit=EvidenceAuditOutput.model_validate(raw_outputs[1]),
        chair_report=ChairReportOutput.model_validate(raw_outputs[2]),
    )


def _coerce_llm_response(raw_response: object) -> Stage2LLMResponse:
    if isinstance(raw_response, Stage2LLMResponse):
        return raw_response
    if isinstance(raw_response, BaseModel):
        return Stage2LLMResponse.model_validate(raw_response.model_dump(mode="json"))
    if isinstance(raw_response, dict):
        return Stage2LLMResponse.model_validate(raw_response)
    if isinstance(raw_response, str):
        try:
            return Stage2LLMResponse.model_validate_json(raw_response)
        except ValueError as error:
            raise RuntimeError(
                "Agno Stage 2 runner returned text instead of the structured "
                "Stage2LLMResponse payload. Check API credentials, model support, "
                "and structured output availability."
            ) from error
    raise TypeError(
        f"Agno Stage 2 runner returned an unsupported response type: {type(raw_response).__name__}"
    )


__all__ = [
    "AgnoStage2AgentRunner",
    "ChairReportFn",
    "DeterministicStage2AgentRunner",
    "EvidenceAuditFn",
    "QuantCreditFn",
    "Stage2AgentRunner",
    "Stage2LLMClient",
    "Stage2LLMResponse",
    "Stage2RunnerOutputs",
]
