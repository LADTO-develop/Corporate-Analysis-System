"""Stage 2 agent runner adapters.

The default runner is deterministic for CI stability. The Agno runner is wired
as an optional adapter so local demos can switch to LLM-backed structured
outputs without changing committee_node orchestration.
"""

from __future__ import annotations

import json
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
    """Single structured response expected from an Agno-backed Stage 2 run."""

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
    tests/local experiments.
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
        """Run Stage 2 through Agno or an injected structured LLM client."""
        prompt_payload = _build_prompt_payload(
            bundle=bundle,
            recommendation=recommendation,
            confidence=confidence,
            draft_outputs=self._draft_outputs(bundle, recommendation, confidence),
        )
        try:
            raw_response = (
                self.llm_client.run_structured(
                    prompt_payload=prompt_payload,
                    output_schema=Stage2LLMResponse,
                )
                if self.llm_client is not None
                else self._run_with_agno(prompt_payload=prompt_payload)
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

    def _run_with_agno(self, *, prompt_payload: dict[str, Any]) -> object:
        try:
            agent_module = import_module("agno.agent")
            anthropic_module = import_module("agno.models.anthropic")
        except ImportError as error:
            raise RuntimeError(
                "CAS_STAGE2_RUNNER=agno requires the optional Agno/Anthropic runtime. "
                "Install Agno and configure ANTHROPIC_API_KEY, or keep "
                "CAS_STAGE2_RUNNER=deterministic for offline runs."
            ) from error

        agent_cls = cast(Any, agent_module).Agent
        claude_cls = cast(Any, anthropic_module).Claude
        model = claude_cls(
            id=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        agent = agent_cls(
            model=model,
            description=_AGNO_STAGE2_DESCRIPTION,
            output_schema=Stage2LLMResponse,
        )
        response = agent.run(_prompt_text(prompt_payload))
        return getattr(response, "content", response)


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


def _prompt_text(prompt_payload: dict[str, Any]) -> str:
    return (
        "Return a valid Stage2LLMResponse object that satisfies the provided "
        "Pydantic output_schema. Use the following CAS Stage 2 input payload:\n\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2, default=str)}"
    )


_AGNO_STAGE2_DESCRIPTION = """
You are the Stage 2 credit review committee for CAS.
Produce three structured agent outputs:
1. quant_credit: explain model_view, SHAP drivers, and peer context.
2. evidence_audit: audit external evidence, debt/liquidity, and macro/industry signals.
3. chair_report: synthesize without overwriting the Stage 1 model decision.
Use Korean business review language and avoid unsupported claims.
""".strip()


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
