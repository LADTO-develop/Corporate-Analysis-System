"""Shared Agno runtime helpers for Stage 2 triplet agents."""

from __future__ import annotations

import json
import os
import time
from importlib import import_module
from typing import Protocol, TypeVar, cast

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class AgnoAgentLike(Protocol):
    """Minimal Agno Agent protocol used by the adapters."""

    def run(self, query: str) -> object:
        """Run an agent prompt and return the provider response."""


def build_agno_agent(  # noqa: UP047, RUF100
    *,
    name: str,
    model_name: str,
    max_tokens: int,
    response_model: type[ModelT],
    instructions: list[str],
) -> AgnoAgentLike:
    """Create an Agno Agent lazily so importing CAS does not require Agno."""
    api_key = _anthropic_api_key()
    try:
        agent_module = import_module("agno.agent")
        anthropic_module = import_module("agno.models.anthropic")
    except ImportError as error:
        raise RuntimeError(
            "CAS_STAGE2_RUNNER=agno requires the optional Agno/Anthropic runtime. "
            "Install this project with the 'agent' extra and configure ANTHROPIC_API_KEY."
        ) from error

    agent_cls = agent_module.Agent
    claude_cls = anthropic_module.Claude
    model = claude_cls(
        id=model_name,
        max_tokens=max_tokens,
        temperature=0,
        api_key=api_key,
    )

    return cast(
        AgnoAgentLike,
        agent_cls(
            name=name,
            model=model,
            instructions=instructions,
            output_schema=response_model,
            parse_response=True,
            expected_output=f"Return only a valid {response_model.__name__} object.",
            markdown=False,
        ),
    )


def run_structured_agent(  # noqa: UP047, RUF100
    *,
    agent: AgnoAgentLike,
    query: str,
    response_model: type[ModelT],
) -> ModelT:
    """Run an Agno agent and coerce the response into a Pydantic model."""
    attempts = _stage2_agent_retry_attempts()
    for attempt in range(1, attempts + 1):
        try:
            response = agent.run(query)
            content = getattr(response, "content", response)
            return coerce_model_response(content, response_model)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(_stage2_agent_retry_delay_seconds() * attempt)

    raise RuntimeError("Agno agent retry loop exited unexpectedly.")


def coerce_model_response(raw_response: object, response_model: type[ModelT]) -> ModelT:  # noqa: UP047, RUF100
    """Coerce common Agno response shapes into the requested response model."""
    if isinstance(raw_response, response_model):
        return raw_response
    if isinstance(raw_response, BaseModel):
        return response_model.model_validate(raw_response.model_dump(mode="json"))
    if isinstance(raw_response, dict):
        return response_model.model_validate(raw_response)
    if isinstance(raw_response, str):
        return response_model.model_validate_json(_strip_json_fence(raw_response))
    raise TypeError(
        f"Agno agent returned an unsupported response type: {type(raw_response).__name__}"
    )


def json_payload(value: object) -> str:
    """Serialize prompt context with stable formatting."""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def compact_items(*values: str) -> list[str]:
    """Return up to three non-empty findings."""
    return [value for value in values if value][:3]


def clamp(value: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a score to a bounded confidence range."""
    return min(max(value, minimum), maximum)


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def _anthropic_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "CAS_STAGE2_RUNNER=agno requires ANTHROPIC_API_KEY. "
            "Set it in your local .env or environment before running live Agno Stage 2."
        )
    return api_key


def _stage2_agent_retry_attempts() -> int:
    raw_value = os.environ.get("CAS_STAGE2_AGENT_RETRIES", "2").strip()
    try:
        attempts = int(raw_value)
    except ValueError:
        attempts = 2
    return min(max(attempts, 1), 5)


def _stage2_agent_retry_delay_seconds() -> float:
    raw_value = os.environ.get("CAS_STAGE2_AGENT_RETRY_DELAY_SECONDS", "1.5").strip()
    try:
        delay = float(raw_value)
    except ValueError:
        delay = 1.5
    return min(max(delay, 0.0), 10.0)
