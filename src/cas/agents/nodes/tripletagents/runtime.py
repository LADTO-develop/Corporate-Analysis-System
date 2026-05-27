"""Shared Agno runtime helpers for Stage 2 triplet agents."""

from __future__ import annotations

import json
import os
import time
from importlib import import_module
from typing import Protocol, cast

from pydantic import BaseModel

from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.llm.model_catalog import load_model_catalog, normalize_stage2_provider


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
    runtime_config: Stage2RuntimeConfig | None = None,
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
        runtime_config=runtime_config,
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
    return cast(str, normalize_stage2_provider(provider))


def provider_label(provider: str) -> str:
    """Return a human-readable provider label for prompts and diagnostics."""
    catalog = load_model_catalog()
    config = catalog.provider(provider)
    if config is not None:
        return cast(str, config.label)
    return normalize_model_provider(provider)


def provider_env_var_names(provider: str) -> tuple[str, ...]:
    """Return accepted API key environment variables for a provider."""
    catalog = load_model_catalog()
    config = catalog.provider(provider)
    if config is None:
        normalize_model_provider(provider)
        return ()
    return cast(tuple[str, ...], config.api_key_env_vars)


def _build_agno_model(
    *,
    provider: str,
    model_name: str,
    max_tokens: int,
    runtime_config: Stage2RuntimeConfig | None = None,
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
            timeout=_stage2_agent_timeout_seconds(runtime_config),
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
            timeout=_stage2_agent_timeout_seconds(runtime_config),
            max_retries=_stage2_provider_max_retries(runtime_config),
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
        timeout=_stage2_agent_timeout_seconds(runtime_config),
        retries=_stage2_provider_max_retries(runtime_config),
        delay_between_retries=_stage2_provider_retry_delay_seconds(runtime_config),
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
    runtime_config: Stage2RuntimeConfig | None = None,
) -> ModelT:
    """Run an Agno agent and coerce the response into a Pydantic model."""
    attempts = _stage2_agent_retry_attempts(runtime_config)
    for attempt in range(1, attempts + 1):
        try:
            response = agent.run(query)
            content = getattr(response, "content", response)
            return coerce_model_response(content, response_model)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(_stage2_agent_retry_delay_seconds(runtime_config) * attempt)
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


def _stage2_agent_retry_attempts(runtime_config: Stage2RuntimeConfig | None = None) -> int:
    return int(_resolved_runtime_config(runtime_config).agent_retries)


def _stage2_agent_retry_delay_seconds(runtime_config: Stage2RuntimeConfig | None = None) -> float:
    return float(_resolved_runtime_config(runtime_config).agent_retry_delay_seconds)


def _stage2_agent_timeout_seconds(
    runtime_config: Stage2RuntimeConfig | None = None,
) -> float | None:
    return cast(float | None, _resolved_runtime_config(runtime_config).agent_timeout_seconds)


def _stage2_provider_max_retries(runtime_config: Stage2RuntimeConfig | None = None) -> int:
    return int(_resolved_runtime_config(runtime_config).provider_max_retries)


def _stage2_provider_retry_delay_seconds(runtime_config: Stage2RuntimeConfig | None = None) -> int:
    delay = float(_resolved_runtime_config(runtime_config).agent_retry_delay_seconds)
    return max(1, round(delay))


def _resolved_runtime_config(
    runtime_config: Stage2RuntimeConfig | None,
) -> Stage2RuntimeConfig:
    return runtime_config or Stage2RuntimeConfig.from_env(os.environ)
