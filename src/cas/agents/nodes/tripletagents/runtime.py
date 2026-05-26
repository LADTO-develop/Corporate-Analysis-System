"""Shared Agno runtime helpers for Stage 2 triplet agents."""

from __future__ import annotations

import json
import os
import time
from importlib import import_module
from typing import Protocol, cast

from pydantic import BaseModel


class AgnoAgentLike(Protocol):
    """Minimal Agno Agent protocol used by the adapters."""

    def run(self, query: str) -> object:
        """Run an agent prompt and return the provider response."""


def build_agno_agent[ModelT: BaseModel](
    *,
    name: str,
    model_name: str,
    model_provider: str = "openai",
    max_tokens: int,
    response_model: type[ModelT],
    instructions: list[str],
) -> AgnoAgentLike:
    """Create an Agno Agent lazily so importing CAS does not require Agno."""
    try:
        agent_module = import_module("agno.agent")
    except ImportError as error:
        raise RuntimeError(
            "CAS_STAGE2_RUNNER=agno requires the optional Agno runtime. "
            'Install this project with: python -m pip install -e ".[agent]".'
        ) from error

    agent_cls = agent_module.Agent
    model = _build_agno_model(
        provider=model_provider,
        model_name=model_name,
        max_tokens=max_tokens,
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


def normalize_model_provider(provider: str) -> str:
    """Normalize supported provider aliases used by Stage 2 model routing."""
    normalized = provider.strip().lower().replace("-", "_")
    aliases = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai",
        "gpt": "openai",
        "google": "google",
        "gemini": "google",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported CAS Stage 2 model provider. "
            "Use one of: anthropic/claude, openai/gpt, google/gemini."
        )
    return aliases[normalized]


def provider_label(provider: str) -> str:
    """Return a human-readable provider label for prompts and diagnostics."""
    labels = {
        "anthropic": "Claude",
        "openai": "GPT",
        "google": "Gemini",
    }
    return labels[normalize_model_provider(provider)]


def provider_env_var_names(provider: str) -> tuple[str, ...]:
    """Return accepted API key environment variables for a provider."""
    normalized = normalize_model_provider(provider)
    if normalized == "anthropic":
        return ("ANTHROPIC_API_KEY",)
    if normalized == "openai":
        return ("OPENAI_API_KEY",)
    return ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def _build_agno_model(
    *,
    provider: str,
    model_name: str,
    max_tokens: int,
) -> object:
    normalized_provider = normalize_model_provider(provider)
    api_key = _provider_api_key(normalized_provider)

    if normalized_provider == "anthropic":
        try:
            anthropic_module = import_module("agno.models.anthropic")
        except ImportError as error:
            raise RuntimeError(
                "CAS_STAGE2_RUNNER=agno with Claude requires agno[anthropic] and "
                'the anthropic package. Install with: python -m pip install -e ".[agent]".'
            ) from error
        claude_cls = anthropic_module.Claude
        return claude_cls(
            id=model_name,
            max_tokens=max_tokens,
            temperature=0,
            api_key=api_key,
            timeout=_stage2_agent_timeout_seconds(),
        )

    if normalized_provider == "openai":
        try:
            openai_module = import_module("agno.models.openai")
        except ImportError as error:
            raise RuntimeError(
                "CAS_STAGE2_RUNNER=agno with GPT requires agno[openai] and the openai "
                'package. Install with: python -m pip install -e ".[agent]".'
            ) from error
        openai_cls = openai_module.OpenAIResponses
        return openai_cls(
            id=model_name,
            max_output_tokens=max_tokens,
            temperature=0,
            api_key=api_key,
            timeout=_stage2_agent_timeout_seconds(),
            max_retries=_stage2_provider_max_retries(),
        )

    try:
        google_module = import_module("agno.models.google")
    except ImportError as error:
        raise RuntimeError(
            "CAS_STAGE2_RUNNER=agno with Gemini requires agno[google] and google-genai. "
            'Install with: python -m pip install -e ".[agent]".'
        ) from error
    gemini_cls = google_module.Gemini
    return gemini_cls(
        id=model_name,
        max_output_tokens=max_tokens,
        temperature=0,
        api_key=api_key,
    )


def _provider_api_key(provider: str) -> str:
    for env_var_name in provider_env_var_names(provider):
        api_key = os.environ.get(env_var_name, "").strip()
        if api_key:
            return api_key
    env_var_text = " or ".join(provider_env_var_names(provider))
    raise RuntimeError(
        f"CAS_STAGE2_RUNNER=agno requires {env_var_text}. "
        "Set it in your local .env or environment before running live Agno Stage 2."
    )


def run_structured_agent[ModelT: BaseModel](
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


def coerce_model_response[ModelT: BaseModel](
    raw_response: object, response_model: type[ModelT]
) -> ModelT:
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


def _stage2_agent_timeout_seconds() -> float | None:
    raw_value = os.environ.get("CAS_STAGE2_AGENT_TIMEOUT_SECONDS", "").strip().lower()
    if raw_value in {"", "0", "false", "no", "none", "off"}:
        return None
    try:
        timeout = float(raw_value)
    except ValueError:
        return None
    return min(max(timeout, 1.0), 600.0)


def _stage2_provider_max_retries() -> int:
    raw_value = os.environ.get("CAS_STAGE2_PROVIDER_MAX_RETRIES", "0").strip()
    try:
        retries = int(raw_value)
    except ValueError:
        retries = 0
    return min(max(retries, 0), 5)
