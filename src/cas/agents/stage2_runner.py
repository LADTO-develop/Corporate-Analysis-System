"""Stage 2 agent runner adapters.

The default runner is deterministic for CI stability. The Agno runner is wired
as an optional adapter so local demos can switch to LLM-backed structured
outputs without changing committee_node orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict

from cas.agents.stage2_bundle import Stage2InputBundle
from cas.agents.stage2_outputs import (
    ChairReportOutput,
    EvidenceAuditOutput,
    QuantCreditOutput,
)
from cas.agents.state import Recommendation
from cas.utils.live_cache import read_json_cache, stable_cache_key, write_json_cache

QuantCreditFn = Callable[[Stage2InputBundle], QuantCreditOutput]
EvidenceAuditFn = Callable[[Stage2InputBundle], EvidenceAuditOutput]
ChairReportFn = Callable[[Stage2InputBundle, Recommendation, float], ChairReportOutput]
Stage2RunnerOutputs = tuple[QuantCreditOutput, EvidenceAuditOutput, ChairReportOutput]


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
    routing_mode: str = "single"
    model_provider: str = "openai"
    model_name: str = "gpt-4.1-mini"
    quant_model_provider: str | None = None
    quant_model_name: str | None = None
    evidence_model_provider: str | None = None
    evidence_model_name: str | None = None
    chair_model_provider: str | None = None
    chair_model_name: str | None = None
    max_tokens: int = 6000
    backend_name: str = "agno"
    fallback_on_error: bool = True
    last_run_backend_name: str = "agno"
    last_error_message: str = ""

    def run(
        self,
        *,
        bundle: Stage2InputBundle,
        recommendation: Recommendation,
        confidence: float,
    ) -> Stage2RunnerOutputs:
        """Run Stage 2 through the Agno triplet agents or an injected LLM client."""
        try:
            cache_key = stable_cache_key(
                _stage2_cache_payload(
                    runner=self,
                    bundle=bundle,
                    recommendation=recommendation,
                    confidence=confidence,
                )
            )
            cached_response = _read_stage2_cached_response(cache_key)
            if cached_response is not None:
                response, cached_backend_name = cached_response
                self.last_run_backend_name = cached_backend_name
                self.last_error_message = ""
                return response.as_outputs()

            if self.llm_client is not None:
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
                successful_backend_name = self.backend_name
            else:
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
                )
                successful_backend_name = self.backend_name
            response = _coerce_llm_response(raw_response)
            _write_stage2_cached_response(
                cache_key=cache_key,
                backend_name=successful_backend_name,
                response=response,
            )
            outputs = response.as_outputs()
            self.last_run_backend_name = successful_backend_name
            self.last_error_message = ""
            return outputs
        except Exception as error:
            if self.deterministic_runner is None or not self.fallback_on_error:
                raise
            self.last_run_backend_name = "agno_fallback_deterministic"
            self.last_error_message = str(error)
            return self.deterministic_runner.run(
                bundle=bundle,
                recommendation=recommendation,
                confidence=confidence,
            )

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
            return {
                "quant_credit": "anthropic",
                "evidence_audit": "openai",
                "chair_report": "google",
            }[role]
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
            return {
                "quant_credit": self.model_name,
                "evidence_audit": "gpt-5.4-mini",
                "chair_report": "gemini-flash-latest",
            }[role]
        return None

    def _uses_multi_llm_committee(self) -> bool:
        return self.routing_mode.strip().lower() in {"multi", "multi_llm", "multi_llm_committee"}


def _build_prompt_payload(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    draft_outputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": (
            "Review Stage 1 credit-risk output without overwriting model_view. "
            "Return only the structured Stage2LLMResponse payload."
        ),
        "guardrails": [
            "Preserve the Stage 1 prediction_label as the model's original decision.",
            "Use committee_view logic only to explain, qualify, or escalate the decision.",
            "Do not invent external evidence. If news_cache_snapshot has no items, say evidence is pending.",
            "If external evidence status is disabled or missing, do not infer specific news, DART, macro, or industry events.",
            "For historical replay, respect news_cache_snapshot.as_of_date and do not reason from filtered future/undated evidence.",
            "Treat external items with company_match=false as weak/indirect evidence.",
            "Use evidence_quality, evidence_score, and verification_flags when judging news strength.",
            "Do not describe critical_terms as confirmed events unless veto_candidate=true or veto_triggered=true.",
            "Use hidden_tail_risk_flag when direct, verified external adverse evidence challenges an eligible model decision.",
            "Do not escalate an eligible near-threshold model call to hold from proximity alone when absolute risk is low and severe financial stress is absent.",
            "EvidenceAuditAgent must separate evidence_limitations from confirmed risks.",
            "Do not say the system confirms, approves, assigns, or finalizes an official credit rating.",
            "Treat rule_engine_confidence as a rule-engine review confidence, not model confidence.",
            "If direct_match_count is positive, do not claim all external evidence lacks company relevance.",
            "EvidenceAuditAgent must state evidence_strength, model_challenge, and audit_conclusion.",
            "Keep all confidence values between 0 and 1.",
            "Keep each list to at most 3 items and each Korean sentence concise.",
            "Return compact JSON only; do not include markdown or commentary outside the schema.",
        ],
        "recommendation": recommendation,
        "rule_engine_confidence": confidence,
        "stage2_input_bundle": bundle.to_prompt_payload(),
        "deterministic_draft_outputs": draft_outputs,
    }


def _stage2_cache_payload(
    *,
    runner: AgnoStage2AgentRunner,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
) -> dict[str, Any]:
    return {
        "cache_version": "stage2_llm_response_v1",
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
        "stage2_input_bundle": bundle.to_prompt_payload(),
    }


def _read_stage2_cached_response(cache_key: str) -> tuple[Stage2LLMResponse, str] | None:
    cached_payload = read_json_cache(
        "llm_stage2",
        cache_key,
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
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
    return response, backend_name


def _write_stage2_cached_response(
    *,
    cache_key: str,
    backend_name: str,
    response: Stage2LLMResponse,
) -> None:
    write_json_cache(
        "llm_stage2",
        cache_key,
        {
            "cache_version": "stage2_llm_response_v1",
            "backend_name": backend_name,
            "response": response.model_dump(mode="json"),
        },
        env_var="CAS_STAGE2_LLM_CACHE_ENABLED",
        default=True,
    )


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
