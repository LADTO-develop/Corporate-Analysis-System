"""LLM helpers for TS2000 dashboard explanations."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import requests  # type: ignore[import-untyped]

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"

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


def generate_openai_explanation(
    *,
    api_key: str,
    model: str,
    payload: dict[str, Any],
    output_format: str = "memo",
    timeout: int = 60,
) -> str:
    """Generate a Korean explanation using the OpenAI Responses API."""
    clean_api_key = _normalize_api_key(api_key)
    clean_model = _normalize_model_name(model)
    format_instruction = OUTPUT_FORMAT_INSTRUCTIONS.get(
        output_format, OUTPUT_FORMAT_INSTRUCTIONS["memo"]
    )

    if not clean_api_key:
        raise ValueError("API 키를 다시 입력해 주세요.")

    _validate_header_safe(clean_model, "모델명")

    request_body = json.dumps(
        {
            "model": clean_model,
            "instructions": f"{LLM_INSTRUCTIONS}\n\n{format_instruction}",
            "input": build_llm_input(payload),
        },
        ensure_ascii=False,
    ).encode("utf-8")

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
    response.raise_for_status()
    response_payload = response.json()
    explanation = _extract_response_text(response_payload)
    if explanation:
        return explanation
    raise ValueError("LLM 응답에서 텍스트를 추출하지 못했습니다.")
