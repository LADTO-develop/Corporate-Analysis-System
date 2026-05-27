"""LLM token usage and cost helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from pydantic import BaseModel

MODEL_PRICING_USD_PER_1M = {
    "anthropic:claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "openai:gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "google:gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini:gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}

ROLE_TOKEN_ESTIMATES = {
    "quant_credit": {"input": 2600, "output": 550},
    "evidence_audit": {"input": 3200, "output": 700},
    "chair_report": {"input": 2800, "output": 550},
    "review_qa": {"input": 2600, "output": 450},
    "risk_recall_qa": {"input": 2600, "output": 450},
}

_INPUT_TOKEN_KEYS = (
    "input_tokens",
    "prompt_tokens",
    "prompt_token_count",
    "input_token_count",
    "inputTokens",
)
_OUTPUT_TOKEN_KEYS = (
    "output_tokens",
    "completion_tokens",
    "completion_token_count",
    "candidates_token_count",
    "output_token_count",
    "outputTokens",
)
_TOTAL_TOKEN_KEYS = (
    "total_tokens",
    "total_token_count",
    "totalTokens",
)
_CACHED_TOKEN_KEYS = (
    "cached_tokens",
    "cache_read_input_tokens",
    "cached_input_tokens",
)
_COST_KEYS = (
    "cost_usd",
    "total_cost_usd",
    "estimated_cost_usd",
    "total_cost",
    "cost",
)


def llm_usage_from_response(
    response: object,
    *,
    provider: str,
    model_name: str,
) -> dict[str, Any]:
    """Extract token usage from common provider/Agno response shapes."""
    candidates = list(_usage_candidates(response))
    input_tokens = _first_numeric(candidates, _INPUT_TOKEN_KEYS)
    output_tokens = _first_numeric(candidates, _OUTPUT_TOKEN_KEYS)
    total_tokens = _first_numeric(candidates, _TOTAL_TOKEN_KEYS)
    cached_tokens = _first_numeric(candidates, _CACHED_TOKEN_KEYS)
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    cost_usd = _first_float(candidates, _COST_KEYS)
    cost_source = "not_reported"
    if cost_usd is not None:
        cost_source = "provider_reported"
    else:
        estimated = estimate_usage_cost_usd(
            provider=provider,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if estimated is not None:
            cost_usd = estimated
            cost_source = "pricing_estimate_from_actual_tokens"

    return normalize_usage_record(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": cost_usd,
            "cost_source": cost_source,
        },
        provider=provider,
        model_name=model_name,
        cache_hit=False,
        billable=True,
    )


def normalize_usage_record(
    usage: Mapping[str, object] | None,
    *,
    provider: str,
    model_name: str,
    role: str | None = None,
    cache_hit: bool = False,
    billable: bool = True,
) -> dict[str, Any]:
    """Return a stable usage payload for diagnostics, cache files, and CSV rows."""
    source = dict(usage or {})
    resolved_provider = str(source.get("provider") or provider)
    resolved_model = str(source.get("model_name") or source.get("model") or model_name)
    input_tokens = _int_or_none(source.get("input_tokens"))
    output_tokens = _int_or_none(source.get("output_tokens"))
    total_tokens = _int_or_none(source.get("total_tokens"))
    cached_tokens = _int_or_none(source.get("cached_tokens"))
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)

    cost_usd = _float_or_none(source.get("cost_usd"))
    cost_source = str(source.get("cost_source") or "not_reported")
    if cost_usd is None:
        estimated = estimate_usage_cost_usd(
            provider=resolved_provider,
            model_name=resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if estimated is not None:
            cost_usd = estimated
            cost_source = "pricing_estimate_from_actual_tokens"

    effective_billable = billable and not cache_hit
    record: dict[str, Any] = {
        "provider": resolved_provider,
        "model": resolved_model,
        "pricing_model_id": pricing_model_id(resolved_provider, resolved_model),
        "cache_hit": bool(cache_hit),
        "billable": bool(effective_billable),
        "cost_source": cost_source,
    }
    if role:
        record["role"] = role
    for key, value in (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("total_tokens", total_tokens),
        ("cached_tokens", cached_tokens),
    ):
        if value is not None:
            record[key] = value
    if cost_usd is not None:
        record["cost_usd"] = round(float(cost_usd), 8)

    record["billable_input_tokens"] = int(input_tokens or 0) if effective_billable else 0
    record["billable_output_tokens"] = int(output_tokens or 0) if effective_billable else 0
    record["billable_total_tokens"] = int(total_tokens or 0) if effective_billable else 0
    record["billable_cost_usd"] = (
        round(float(cost_usd or 0.0), 8) if effective_billable else 0.0
    )
    return record


def usage_for_cache_hit(
    usage: Mapping[str, object] | None,
    *,
    provider: str,
    model_name: str,
    role: str | None = None,
) -> dict[str, Any]:
    """Return a cached usage record with billable fields zeroed out."""
    return normalize_usage_record(
        usage,
        provider=provider,
        model_name=model_name,
        role=role,
        cache_hit=True,
        billable=False,
    )


def aggregate_role_usage(role_usage: Mapping[str, object]) -> dict[str, Any]:
    """Sum token usage records across Stage 2 roles."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "billable_input_tokens": 0,
        "billable_output_tokens": 0,
        "billable_total_tokens": 0,
        "cost_usd": 0.0,
        "billable_cost_usd": 0.0,
        "usage_role_count": 0,
        "billable_role_count": 0,
    }
    roles: list[str] = []
    cost_seen = False
    for role, raw_record in role_usage.items():
        if not isinstance(raw_record, Mapping):
            continue
        roles.append(str(role))
        totals["usage_role_count"] += 1
        if raw_record.get("billable") is True:
            totals["billable_role_count"] += 1
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "billable_input_tokens",
            "billable_output_tokens",
            "billable_total_tokens",
        ):
            totals[key] += int(_int_or_none(raw_record.get(key)) or 0)
        cost = _float_or_none(raw_record.get("cost_usd"))
        if cost is not None:
            totals["cost_usd"] += cost
            cost_seen = True
        billable_cost = _float_or_none(raw_record.get("billable_cost_usd"))
        if billable_cost is not None:
            totals["billable_cost_usd"] += billable_cost
    output: dict[str, Any] = {
        **{
            key: int(value)
            for key, value in totals.items()
            if key.endswith("_tokens") or key.endswith("_count")
        },
        "billable_cost_usd": round(float(totals["billable_cost_usd"]), 8),
        "roles": roles,
    }
    if cost_seen:
        output["cost_usd"] = round(float(totals["cost_usd"]), 8)
    return output


def estimate_usage_cost_usd(
    *,
    provider: str,
    model_name: str,
    input_tokens: int | float | None,
    output_tokens: int | float | None,
) -> float | None:
    """Estimate USD cost from actual token counts and the local pricing table."""
    if input_tokens is None and output_tokens is None:
        return None
    pricing = MODEL_PRICING_USD_PER_1M.get(pricing_model_id(provider, model_name))
    if pricing is None:
        return None
    cost = (
        float(input_tokens or 0) * float(pricing["input"]) / 1_000_000
        + float(output_tokens or 0) * float(pricing["output"]) / 1_000_000
    )
    return round(cost, 8)


def pricing_model_id(provider: str, model_name: str) -> str:
    """Return the normalized provider:model key used by the pricing table."""
    raw_provider = provider.strip().lower().replace("-", "_")
    raw_model = model_name.strip()
    if ":" in raw_model:
        prefix, model = raw_model.split(":", 1)
        provider_from_model = prefix.strip().lower().replace("-", "_")
        if provider_from_model == "gemini":
            return f"google:{model.strip()}"
        if provider_from_model in {"claude", "anthropic"}:
            return f"anthropic:{model.strip()}"
        if provider_from_model in {"gpt", "openai"}:
            return f"openai:{model.strip()}"
        return f"{provider_from_model}:{model.strip()}"
    provider_aliases = {
        "claude": "anthropic",
        "anthropic": "anthropic",
        "gpt": "openai",
        "openai": "openai",
        "gemini": "google",
        "google": "google",
    }
    return f"{provider_aliases.get(raw_provider, raw_provider)}:{raw_model}"


def _usage_candidates(response: object) -> Iterable[object]:
    seen: set[int] = set()
    stack = [response]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        object_id = id(current)
        if object_id in seen:
            continue
        seen.add(object_id)
        yield current
        if isinstance(current, BaseModel):
            stack.append(current.model_dump(mode="json"))
            continue
        if isinstance(current, Mapping):
            for key in (
                "usage",
                "token_usage",
                "usage_metadata",
                "metrics",
                "response_metadata",
                "metadata",
            ):
                if key in current:
                    stack.append(current[key])
            continue
        if isinstance(current, list | tuple):
            stack.extend(current)
            continue
        for attr in (
            "usage",
            "token_usage",
            "usage_metadata",
            "metrics",
            "response_metadata",
            "metadata",
            "raw_response",
            "model_response",
        ):
            try:
                value = getattr(current, attr)
            except Exception:
                continue
            if value is not None:
                stack.append(value)


def _first_numeric(candidates: Iterable[object], keys: tuple[str, ...]) -> int | None:
    for candidate in candidates:
        value = _value_for_keys(candidate, keys)
        numeric = _int_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _first_float(candidates: Iterable[object], keys: tuple[str, ...]) -> float | None:
    for candidate in candidates:
        value = _value_for_keys(candidate, keys)
        numeric = _float_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _value_for_keys(candidate: object, keys: tuple[str, ...]) -> object | None:
    if isinstance(candidate, Mapping):
        for key in keys:
            if key in candidate:
                return cast(object, candidate[key])
    for key in keys:
        try:
            return cast(object, getattr(candidate, key))
        except Exception:
            continue
    return None


def _int_or_none(value: object) -> int | None:
    numeric = _float_or_none(value)
    if numeric is None:
        return None
    return round(numeric)


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    if isinstance(value, list | tuple):
        values = [_float_or_none(item) for item in value]
        numbers = [item for item in values if item is not None]
        if numbers:
            return float(sum(numbers))
    return None


__all__ = [
    "MODEL_PRICING_USD_PER_1M",
    "ROLE_TOKEN_ESTIMATES",
    "aggregate_role_usage",
    "estimate_usage_cost_usd",
    "llm_usage_from_response",
    "normalize_usage_record",
    "pricing_model_id",
    "usage_for_cache_hit",
]
