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
        model_name: str,
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
    """Optional Agno/Claude-backed Stage 2 runner.

    This path is intentionally opt-in. If the ``agno`` package is unavailable,
    install the optional LLM dependencies or inject a ``Stage2LLMClient`` in
    tests/local experiments. Without an injected client, the runner executes
    the three Agno triplet agents in sequence.
    """

    deterministic_runner: DeterministicStage2AgentRunner | None = None
    llm_client: Stage2LLMClient | None = None
    model_name: str = "claude-sonnet-4-5-20250929"
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
            else:
                raw_response = _run_triplet_agents_with_agno(
                    bundle=bundle,
                    recommendation=recommendation,
                    confidence=confidence,
                    model_name=self.model_name,
                    max_tokens=self.max_tokens,
                )
            outputs = _coerce_llm_response(raw_response).as_outputs()
            self.last_run_backend_name = self.backend_name
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
            "Treat external items with company_match=false as weak/indirect evidence.",
            "Use evidence_quality, evidence_score, and verification_flags when judging news strength.",
            "Do not describe critical_terms as confirmed events unless veto_candidate=true or veto_triggered=true.",
            "Use hidden_tail_risk_flag when direct, verified external adverse evidence challenges an eligible model decision.",
            "If direct_match_count is positive, do not claim all external evidence lacks company relevance.",
            "EvidenceAuditAgent must state evidence_strength, model_challenge, and audit_conclusion.",
            "Keep all confidence values between 0 and 1.",
            "Keep each list to at most 3 items and each Korean sentence concise.",
            "Return compact JSON only; do not include markdown or commentary outside the schema.",
        ],
        "recommendation": recommendation,
        "confidence": confidence,
        "stage2_input_bundle": bundle.to_prompt_payload(),
        "deterministic_draft_outputs": draft_outputs,
    }


def _run_triplet_agents_with_agno(
    *,
    bundle: Stage2InputBundle,
    recommendation: Recommendation,
    confidence: float,
    model_name: str,
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
        model_name=model_name,
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
