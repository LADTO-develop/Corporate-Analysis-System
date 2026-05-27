"""Runtime configuration for Stage 2 agent execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from cas.llm.model_catalog import (
    DEFAULT_STAGE2_AGNO_MODE,
    DEFAULT_STAGE2_RUNNER,
    DEFAULT_STAGE2_SINGLE_MODEL,
    DEFAULT_STAGE2_SINGLE_PROVIDER,
    is_multi_llm_committee_mode,
    normalize_stage2_provider,
    stage2_role_model_default,
    stage2_single_model_default,
)


@dataclass(frozen=True)
class Stage2RuntimeConfig:
    """Explicit Stage 2 runtime knobs captured before execution starts."""

    runner: str = DEFAULT_STAGE2_RUNNER
    agno_mode: str = DEFAULT_STAGE2_AGNO_MODE
    model_provider: str = DEFAULT_STAGE2_SINGLE_PROVIDER
    model: str = DEFAULT_STAGE2_SINGLE_MODEL
    quant_provider: str | None = None
    quant_model: str | None = None
    evidence_provider: str | None = None
    evidence_model: str | None = None
    chair_provider: str | None = None
    chair_model: str | None = None
    max_tokens: int = 6000
    fallback_on_error: bool = True
    review_qa_enabled: bool | None = None
    review_qa_apply_advisory: bool = True
    review_qa_fallback_on_error: bool = True
    review_qa_provider: str | None = None
    review_qa_model: str | None = None
    review_qa_max_tokens: int = 3000
    risk_recall_qa_enabled: bool | None = None
    risk_recall_qa_apply_advisory: bool = True
    risk_recall_qa_fallback_on_error: bool = True
    risk_recall_qa_provider: str | None = None
    risk_recall_qa_model: str | None = None
    risk_recall_qa_max_tokens: int = 3000
    llm_cache_enabled: bool | None = None
    cache_dir: str | None = None
    cache_refresh: bool | None = None
    agent_retries: int = 2
    agent_retry_delay_seconds: float = 1.5
    agent_timeout_seconds: float | None = None
    provider_max_retries: int = 0
    parallel_independent_agents: bool = True

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str],
        *,
        runner: str | None = None,
    ) -> Stage2RuntimeConfig:
        """Build a config from environment-like values."""
        single_default = stage2_single_model_default()
        return cls(
            runner=_normalized_text(runner or env.get("CAS_STAGE2_RUNNER"), DEFAULT_STAGE2_RUNNER),
            agno_mode=_normalized_text(env.get("CAS_STAGE2_AGNO_MODE"), DEFAULT_STAGE2_AGNO_MODE),
            model_provider=normalize_stage2_provider(
                _normalized_text(
                    env.get("CAS_STAGE2_MODEL_PROVIDER"),
                    single_default.provider,
                )
            ),
            model=_normalized_text(env.get("CAS_STAGE2_MODEL"), single_default.model),
            quant_provider=_optional_stage2_provider(env.get("CAS_STAGE2_QUANT_PROVIDER")),
            quant_model=_optional_text(env.get("CAS_STAGE2_QUANT_MODEL")),
            evidence_provider=_optional_stage2_provider(env.get("CAS_STAGE2_EVIDENCE_PROVIDER")),
            evidence_model=_optional_text(env.get("CAS_STAGE2_EVIDENCE_MODEL")),
            chair_provider=_optional_stage2_provider(env.get("CAS_STAGE2_CHAIR_PROVIDER")),
            chair_model=_optional_text(env.get("CAS_STAGE2_CHAIR_MODEL")),
            max_tokens=_int_value(env.get("CAS_STAGE2_MAX_TOKENS"), 6000),
            fallback_on_error=_bool_value(env.get("CAS_STAGE2_FALLBACK_ON_ERROR"), True),
            review_qa_enabled=_optional_bool(env.get("CAS_STAGE2_REVIEW_QA_ENABLED")),
            review_qa_apply_advisory=_bool_value(
                env.get("CAS_STAGE2_REVIEW_QA_APPLY_ADVISORY"),
                True,
            ),
            review_qa_fallback_on_error=_bool_value(
                env.get("CAS_STAGE2_REVIEW_QA_FALLBACK_ON_ERROR"),
                True,
            ),
            review_qa_provider=_optional_text(env.get("CAS_STAGE2_REVIEW_QA_PROVIDER")),
            review_qa_model=_optional_text(env.get("CAS_STAGE2_REVIEW_QA_MODEL")),
            review_qa_max_tokens=_int_value(env.get("CAS_STAGE2_REVIEW_QA_MAX_TOKENS"), 3000),
            risk_recall_qa_enabled=_optional_bool(env.get("CAS_STAGE2_RISK_RECALL_QA_ENABLED")),
            risk_recall_qa_apply_advisory=_bool_value(
                env.get("CAS_STAGE2_RISK_RECALL_QA_APPLY_ADVISORY"),
                True,
            ),
            risk_recall_qa_fallback_on_error=_bool_value(
                env.get("CAS_STAGE2_RISK_RECALL_QA_FALLBACK_ON_ERROR"),
                True,
            ),
            risk_recall_qa_provider=_optional_text(env.get("CAS_STAGE2_RISK_RECALL_QA_PROVIDER")),
            risk_recall_qa_model=_optional_text(env.get("CAS_STAGE2_RISK_RECALL_QA_MODEL")),
            risk_recall_qa_max_tokens=_int_value(
                env.get("CAS_STAGE2_RISK_RECALL_QA_MAX_TOKENS"),
                3000,
            ),
            llm_cache_enabled=_optional_bool(env.get("CAS_STAGE2_LLM_CACHE_ENABLED")),
            cache_dir=_optional_text(env.get("CAS_STAGE2_CACHE_DIR")),
            cache_refresh=_optional_bool(env.get("CAS_STAGE2_CACHE_REFRESH")),
            agent_retries=_bounded_int_value(env.get("CAS_STAGE2_AGENT_RETRIES"), 2, 1, 5),
            agent_retry_delay_seconds=_bounded_float_value(
                env.get("CAS_STAGE2_AGENT_RETRY_DELAY_SECONDS"),
                1.5,
                0.0,
                10.0,
            ),
            agent_timeout_seconds=_optional_timeout_seconds(
                env.get("CAS_STAGE2_AGENT_TIMEOUT_SECONDS")
            ),
            provider_max_retries=_bounded_int_value(
                env.get("CAS_STAGE2_PROVIDER_MAX_RETRIES"),
                0,
                0,
                5,
            ),
            parallel_independent_agents=_bool_value(
                env.get("CAS_STAGE2_PARALLEL_INDEPENDENT_AGENTS"),
                True,
            ),
        )

    def with_runner(self, runner: str | None) -> Stage2RuntimeConfig:
        """Return this config with only the runner changed."""
        return replace(self, runner=_normalized_text(runner, DEFAULT_STAGE2_RUNNER))

    def cache_payload(self, *, cache_version: str) -> dict[str, object]:
        """Return stable fields that should invalidate Stage 2 dashboard caches."""
        return {
            "cache_version": cache_version,
            "runner": self.runner,
            "model_provider": self.model_provider,
            "model": self.model,
            "agno_mode": self.agno_mode,
            "quant_provider": self.quant_provider,
            "quant_model": self.quant_model,
            "evidence_provider": self.evidence_provider,
            "evidence_model": self.evidence_model,
            "chair_provider": self.chair_provider,
            "chair_model": self.chair_model,
            "review_qa_enabled": self.review_qa_enabled,
            "risk_recall_qa_enabled": self.risk_recall_qa_enabled,
            "max_tokens": str(self.max_tokens),
            "multi_role_defaults": self._multi_role_cache_payload(),
        }

    def cache_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return cache-related environment values for cache helpers.

        Cache helpers still accept an environment-like mapping for backwards
        compatibility. Passing this snapshot keeps cache enable/refresh choices
        tied to the runtime config captured for the current job.
        """
        output = dict(env or {})
        if self.llm_cache_enabled is not None:
            output["CAS_STAGE2_LLM_CACHE_ENABLED"] = "1" if self.llm_cache_enabled else "0"
        if self.cache_dir:
            output["CAS_STAGE2_CACHE_DIR"] = self.cache_dir
        if self.cache_refresh is not None:
            output["CAS_STAGE2_CACHE_REFRESH"] = "1" if self.cache_refresh else "0"
        return output

    @property
    def review_qa_provider_resolved(self) -> str:
        """Return the Review QA provider after fallback resolution."""
        return self.review_qa_provider or self.chair_provider or self.model_provider

    @property
    def review_qa_model_resolved(self) -> str:
        """Return the Review QA model after fallback resolution."""
        return self.review_qa_model or self.chair_model or self.model

    @property
    def risk_recall_qa_provider_resolved(self) -> str:
        """Return the risk-recall QA provider after fallback resolution."""
        return self.risk_recall_qa_provider or self.chair_provider or self.model_provider

    @property
    def risk_recall_qa_model_resolved(self) -> str:
        """Return the risk-recall QA model after fallback resolution."""
        return self.risk_recall_qa_model or self.chair_model or self.model

    def multi_role_provider_resolved(self, role: str) -> str | None:
        """Return a role provider when the configured mode uses multi-LLM routing."""
        if not is_multi_llm_committee_mode(self.agno_mode):
            return None
        explicit = {
            "quant_credit": self.quant_provider,
            "evidence_audit": self.evidence_provider,
            "chair_report": self.chair_provider,
        }[role]
        return explicit or stage2_role_model_default(role).provider

    def multi_role_model_resolved(self, role: str) -> str | None:
        """Return a role model when the configured mode uses multi-LLM routing."""
        if not is_multi_llm_committee_mode(self.agno_mode):
            return None
        explicit = {
            "quant_credit": self.quant_model,
            "evidence_audit": self.evidence_model,
            "chair_report": self.chair_model,
        }[role]
        return explicit or stage2_role_model_default(role).model

    def _multi_role_cache_payload(self) -> dict[str, object]:
        """Return resolved multi-role routing fields for cache invalidation."""
        if not is_multi_llm_committee_mode(self.agno_mode):
            return {}
        return {
            role: {
                "provider": self.multi_role_provider_resolved(role),
                "model": self.multi_role_model_resolved(role),
            }
            for role in ("quant_credit", "evidence_audit", "chair_report")
        }


def _normalized_text(value: str | None, default: str) -> str:
    text = (value or "").strip()
    return text or default


def _optional_text(value: str | None) -> str | None:
    text = (value or "").strip()
    return text or None


def _optional_stage2_provider(value: str | None) -> str | None:
    text = _optional_text(value)
    return normalize_stage2_provider(text) if text else None


def _int_value(value: str | None, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _bounded_int_value(value: str | None, default: int, minimum: int, maximum: int) -> int:
    return min(max(_int_value(value, default), minimum), maximum)


def _float_value(value: str | None, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _bounded_float_value(
    value: str | None,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    return min(max(_float_value(value, default), minimum), maximum)


def _optional_timeout_seconds(value: str | None) -> float | None:
    text = (value or "").strip().lower()
    if text in {"", "0", "false", "no", "none", "off"}:
        return None
    try:
        timeout = float(text)
    except ValueError:
        return None
    return min(max(timeout, 1.0), 600.0)


def _bool_value(value: str | None, default: bool) -> bool:
    parsed = _optional_bool(value)
    return default if parsed is None else parsed


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None
