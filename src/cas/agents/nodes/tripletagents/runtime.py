"""Shared Agno runtime helpers for Stage 2 triplet agents."""

from __future__ import annotations

import json
import os
import time
from importlib import import_module
# [수정됨] Mypy 방어를 위해 Any 추가
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class AgnoAgentLike(Protocol):
    """Minimal Agno Agent protocol used by the adapters."""

    def run(self, query: str) -> object:
        """Run an agent prompt and return the provider response."""


def _get_api_key(provider: str) -> str:
    """Retrieve the correct API key based on the LLM provider."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY"
    }
    env_var = key_map.get(provider)
    if not env_var:
        raise ValueError(f"🚨 지원하지 않는 LLM 제공자입니다: {provider}")
    
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise RuntimeError(
            f"CAS_STAGE2_RUNNER 에러: '{provider}' 모델을 사용하려면 "
            f"환경 변수에 {env_var} 가 설정되어 있어야 합니다."
        )
    return api_key


# [수정됨] 반환 타입을 -> Any 로 명시하여 Mypy의 동적 타입 추론 에러 원천 차단
def _create_model(model_config: str, max_tokens: int) -> Any:
    """Parse 'provider:model_name' string and return the Agno Model instance."""
    if ":" not in model_config:
        raise ValueError(
            f"🚨 잘못된 모델 설정 포맷입니다: '{model_config}'. "
            f"반드시 'provider:model_name' 형식이어야 합니다. (예: 'openai:gpt-4o')"
        )
        
    provider, model_id = model_config.split(":", 1)
    provider = provider.lower()
    api_key = _get_api_key(provider)

    try:
        if provider == "openai":
            openai_module = import_module("agno.models.openai")
            return openai_module.OpenAIChat(
                id=model_id, max_tokens=max_tokens, temperature=0, api_key=api_key
            )
        elif provider == "anthropic":
            anthropic_module = import_module("agno.models.anthropic")
            return anthropic_module.Claude(
                id=model_id, max_tokens=max_tokens, temperature=0, api_key=api_key
            )
        elif provider == "gemini":
            google_module = import_module("agno.models.google")
            return google_module.Gemini(
                id=model_id, max_tokens=max_tokens, temperature=0, api_key=api_key
            )
        else:
            raise ValueError(f"🚨 지원하지 않는 LLM 제공자입니다: {provider}")
            
    except ImportError as error:
        raise RuntimeError(
            f"{provider} 관련 패키지를 찾을 수 없습니다. "
            f"해당 제공자의 SDK가 설치되어 있는지 확인하세요."
        ) from error


def build_agno_agent(  # noqa: UP047, RUF100
    *,
    name: str,
    model_name: str,
    max_tokens: int,
    response_model: type[ModelT],
    instructions: list[str],
) -> AgnoAgentLike:
    """Create an Agno Agent dynamically based on the requested LLM provider."""
    try:
        agent_module = import_module("agno.agent")
    except ImportError as error:
        raise RuntimeError(
            "CAS_STAGE2_RUNNER=agno requires the optional Agno runtime. "
            "Install this project with the 'agent' extra."
        ) from error

    agent_cls = agent_module.Agent
    
    # 모델 조립 공장 호출
    model = _create_model(model_name, max_tokens)

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