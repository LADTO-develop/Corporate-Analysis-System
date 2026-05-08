"""LLM helpers for TS2000 dashboard explanations."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests  # type: ignore[import-untyped]

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"

LLM_INSTRUCTIONS = """
너는 한국어 신용위험 분석 보조역할이며, 심사역 메모 초안을 정리하는 역할이다.
- 반드시 한국어로만 답변한다.
- 모델의 예측 결과를 뒤집지 말고, 제공된 점수와 SHAP를 바탕으로 설명만 한다.
- 과장하거나 단정하지 않는다.
- 부도, 상장폐지처럼 확정적 표현은 쓰지 않는다.
- 숫자와 단위는 가능한 한 그대로 유지한다.
- payload에 `*_display` 형식으로 제공된 값이 있으면 그 값을 우선 사용한다.
- 제공된 동종업계 비교와 산업 집계가 있으면 그 맥락을 반영한다.
- SHAP 상위 변수와 실제 지표값을 바탕으로 왜 위험 또는 완화로 작용했는지 짧게 설명한다.
- 문장은 짧고 명확하게 쓴다. 불필요한 수식어는 줄인다.
- 모든 서술형 문장은 자연스러운 한국어의 '~합니다.' 체로 끝낸다.
- bullet 안의 문장도 동일하게 '~합니다.' 체를 사용한다.
- 아래 형식을 지킨다.

[한줄 판단]
- 1문장

[핵심 위험 요인]
- 2~3개 bullet

[완화 요인]
- 1~2개 bullet

[종합 의견]
- 1~2문장
""".strip()

OUTPUT_FORMAT_INSTRUCTIONS = {
    "brief": """
- 출력은 최대한 짧고 빠르게 읽히게 정리한다.
- [한줄 판단]은 1문장으로 쓴다.
- [핵심 위험 요인]은 2개 bullet 이내로 쓴다.
- [완화 요인]은 1개 bullet 이내로 쓴다.
- [종합 의견]은 1문장으로 쓴다.
""".strip(),
    "memo": """
- 기본 심사 메모 형식으로 작성한다.
- [한줄 판단]은 1문장으로 쓴다.
- [핵심 위험 요인]은 2~3개 bullet로 쓴다.
- [완화 요인]은 1~2개 bullet로 쓴다.
- [종합 의견]은 1~2문장으로 쓴다.
""".strip(),
    "detailed": """
- 조금 더 자세한 보고서형 메모로 작성한다.
- [한줄 판단]은 1문장으로 쓴다.
- [핵심 위험 요인]은 3개 bullet까지 허용한다.
- [완화 요인]은 2개 bullet까지 허용한다.
- [종합 의견]은 2~3문장으로 쓰되 과장하지 않는다.
- 핵심 숫자나 비교 맥락을 필요할 때 한 번 더 언급해도 된다.
""".strip(),
}


def _extract_response_text(payload: dict[str, Any]) -> str:
    """Extract output text from an OpenAI Responses API payload."""
    output = payload.get("output", [])
    text_chunks: list[str] = []
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if text:
                    text_chunks.append(str(text))
    return "\n".join(text_chunks).strip()


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    """Extract output text from an Anthropic Messages API payload."""
    content = payload.get("content", [])
    text_chunks: list[str] = []
    for item in content:
        if item.get("type") == "text":
            text = item.get("text")
            if text:
                text_chunks.append(str(text))
    return "\n".join(text_chunks).strip()


def _to_jsonable(value: object) -> object:
    """Recursively convert pandas/numpy-like objects into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return value


def build_llm_input(payload: dict[str, Any]) -> str:
    """Render a compact JSON payload as input text for the LLM."""
    return json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)


def _clean_token(value: str) -> str:
    """Remove invisible characters and trim text inputs used in headers or model IDs."""
    return (
        value.replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .strip()
    )


def _normalize_api_key(api_key: str) -> str:
    """Normalize API keys to a compact ASCII token safe for headers."""
    cleaned = _clean_token(api_key)
    cleaned = "".join(ch for ch in cleaned if not ch.isspace())
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned


def _normalize_model_name(model: str) -> str:
    """Normalize model text input to a plain model ID."""
    cleaned = _clean_token(model)
    if "|" in cleaned:
        cleaned = cleaned.split("|", 1)[0].strip()
    return cleaned


def _validate_header_safe(value: str, field_name: str) -> None:
    """Ensure header-bound text is safe to encode."""
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"{field_name}에 인코딩할 수 없는 문자가 포함되어 있습니다. "
            "추천 모델에서 영문 모델 ID를 선택하거나 입력값을 다시 확인해 주세요."
        ) from error


def _build_system_instruction(output_format: str) -> str:
    """Build the provider-agnostic system instruction block."""
    format_instruction = OUTPUT_FORMAT_INSTRUCTIONS.get(
        output_format, OUTPUT_FORMAT_INSTRUCTIONS["memo"]
    )
    return f"{LLM_INSTRUCTIONS}\n\n{format_instruction}"


def _extract_error_detail(response: requests.Response) -> str:
    """Extract a short provider error detail when available."""
    try:
        payload = response.json()
    except ValueError:
        return ""

    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = error_obj.get("message") or error_obj.get("type")
            if message:
                return str(message)
        message = payload.get("message")
        if message:
            return str(message)
    return ""


def _raise_friendly_http_error(response: requests.Response, provider_label: str) -> None:
    """Raise a user-friendly error for common provider HTTP failures."""
    detail = _extract_error_detail(response)
    status = response.status_code

    if status in {401, 403}:
        raise ValueError(
            f"{provider_label} API 키가 올바르지 않거나 현재 모델에 대한 접근 권한이 없습니다."
        )
    if status == 404:
        raise ValueError(
            f"{provider_label} 모델명을 찾지 못했습니다. 추천 모델을 다시 선택해 주세요."
        )
    if status == 429:
        raise ValueError(f"{provider_label} 요청이 잠시 많습니다. 잠시 후 다시 시도해 주세요.")
    if 500 <= status < 600:
        raise ValueError(
            f"{provider_label} 서버에서 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        )

    if detail:
        raise ValueError(f"{provider_label} 요청이 실패했습니다. {detail}")
    raise ValueError(f"{provider_label} 요청이 실패했습니다. 상태 코드: {status}")


def generate_openai_explanation(
    *,
    api_key: str,
    model: str,
    payload: dict[str, Any],
    output_format: str = "memo",
    timeout: int = 60,
) -> str:
    """Generate a Korean explanation using the OpenAI Responses API."""
    provider_label = "OpenAI"
    clean_api_key = _normalize_api_key(api_key)
    clean_model = _normalize_model_name(model)
    system_instruction = _build_system_instruction(output_format)

    if not clean_api_key:
        raise ValueError("API 키를 다시 입력해 주세요.")

    _validate_header_safe(clean_model, "모델명")

    request_body = json.dumps(
        {
            "model": clean_model,
            "instructions": system_instruction,
            "input": build_llm_input(payload),
        },
        ensure_ascii=False,
    ).encode("utf-8")

    try:
        response = requests.post(
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": f"Bearer {clean_api_key}",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            data=request_body,
            timeout=timeout,
        )
    except requests.Timeout as error:
        raise ValueError("응답 시간이 길어지고 있습니다. 잠시 후 다시 시도해 주세요.") from error
    except requests.RequestException as error:
        raise ValueError("네트워크 연결을 확인한 뒤 다시 시도해 주세요.") from error

    if not response.ok:
        _raise_friendly_http_error(response, provider_label)

    response_payload = response.json()
    explanation = _extract_response_text(response_payload)
    if explanation:
        return explanation
    raise ValueError("LLM 응답에서 텍스트를 추출하지 못했습니다.")


def generate_claude_explanation(
    *,
    api_key: str,
    model: str,
    payload: dict[str, Any],
    output_format: str = "memo",
    timeout: int = 60,
) -> str:
    """Generate a Korean explanation using the Anthropic Messages API."""
    provider_label = "Claude"
    clean_api_key = _normalize_api_key(api_key)
    clean_model = _normalize_model_name(model)
    system_instruction = _build_system_instruction(output_format)

    if not clean_api_key:
        raise ValueError("API 키를 다시 입력해 주세요.")

    _validate_header_safe(clean_model, "모델명")

    request_body = json.dumps(
        {
            "model": clean_model,
            "max_tokens": 1400,
            "system": system_instruction,
            "messages": [{"role": "user", "content": build_llm_input(payload)}],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    try:
        response = requests.post(
            ANTHROPIC_MESSAGES_URL,
            headers={
                "x-api-key": clean_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
            },
            data=request_body,
            timeout=timeout,
        )
    except requests.Timeout as error:
        raise ValueError("응답 시간이 길어지고 있습니다. 잠시 후 다시 시도해 주세요.") from error
    except requests.RequestException as error:
        raise ValueError("네트워크 연결을 확인한 뒤 다시 시도해 주세요.") from error

    if not response.ok:
        _raise_friendly_http_error(response, provider_label)

    response_payload = response.json()
    explanation = _extract_anthropic_text(response_payload)
    if explanation:
        return explanation
    raise ValueError("LLM 응답에서 텍스트를 추출하지 못했습니다.")


def generate_llm_explanation(
    *,
    provider: str,
    api_key: str,
    model: str,
    payload: dict[str, Any],
    output_format: str = "memo",
    timeout: int = 60,
) -> str:
    """Generate a Korean explanation using the selected provider."""
    provider_name = _clean_token(provider).lower()
    if provider_name == "openai":
        return generate_openai_explanation(
            api_key=api_key,
            model=model,
            payload=payload,
            output_format=output_format,
            timeout=timeout,
        )
    if provider_name in {"claude", "anthropic"}:
        return generate_claude_explanation(
            api_key=api_key,
            model=model,
            payload=payload,
            output_format=output_format,
            timeout=timeout,
        )
    raise ValueError("지원하지 않는 AI 제공자입니다.")
