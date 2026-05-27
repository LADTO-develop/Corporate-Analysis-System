"""High-level Stage 2 committee orchestration."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, cast

from cas.agents.committee_view import build_committee_view
from cas.agents.nodes.committee_diagnostics import _runtime_diagnostic_metrics
from cas.agents.nodes.deterministic_agents import (
    _chair_report_agent,
    _evidence_audit_agent,
    _quant_credit_agent,
)
from cas.agents.nodes.evidence_profile import _clamp, _external_evidence_quality
from cas.agents.nodes.post_committee_qa import post_committee_runtime_config_override
from cas.agents.post_committee_pipeline import run_post_committee_pipeline
from cas.agents.signals.credit_policy_signals import evaluate_credit_policy
from cas.agents.stage2_bundle import Stage2InputBundle, build_stage2_input_bundle
from cas.agents.stage2_runner import (
    AgnoStage2AgentRunner,
    DeterministicStage2AgentRunner,
    Stage2AgentRunner,
)
from cas.agents.stage2_runtime_config import Stage2RuntimeConfig
from cas.agents.state import (
    AgentOutput,
    AgentState,
    AuditEntry,
    CommitteeReview,
    Recommendation,
)

_STAGE2_RUNTIME_CONFIG_OVERRIDE: ContextVar[Stage2RuntimeConfig | None] = ContextVar(
    "stage2_runtime_config_override",
    default=None,
)


@contextmanager
def stage2_runner_override(runner_name: str | None) -> Iterator[None]:
    """Temporarily override Stage 2 runner selection for the current execution context."""
    config = _stage2_runtime_config().with_runner(runner_name)
    with stage2_runtime_config_override(config):
        yield


@contextmanager
def stage2_runtime_config_override(config: Stage2RuntimeConfig) -> Iterator[None]:
    """Temporarily override Stage 2 runtime config for the current execution context."""
    token = _STAGE2_RUNTIME_CONFIG_OVERRIDE.set(config)
    try:
        with post_committee_runtime_config_override(config):
            yield
    finally:
        _STAGE2_RUNTIME_CONFIG_OVERRIDE.reset(token)


def run_stage2_committee(state: AgentState) -> dict[str, Any]:
    """Run Stage 2 agents, optional QA, and dashboard-facing audit assembly."""
    credit_policy_snapshot = evaluate_credit_policy(
        source_feature_row=dict(state.get("source_feature_row") or {}),
        peer_comparison_rows=list(state.get("peer_comparison_rows") or []),
    ).model_dump(mode="json")

    state_with_policy = cast(
        AgentState,
        {
            **state,
            "credit_policy_snapshot": credit_policy_snapshot,
        },
    )

    bundle = build_stage2_input_bundle(state_with_policy)
    recommendation = cast(
        Recommendation,
        bundle.rule_result.get("recommendation") or state.get("final_recommendation") or "review",
    )
    rule_confidence = round(
        float(bundle.rule_result.get("confidence", state.get("final_confidence", 0.0)) or 0.0),
        4,
    )

    runner = _stage2_runner()
    structured_outputs = runner.run(
        bundle=bundle,
        recommendation=recommendation,
        confidence=rule_confidence,
    )
    runtime_backend_name = str(getattr(runner, "last_run_backend_name", runner.backend_name))
    runtime_diagnostics = dict(getattr(runner, "last_run_diagnostics", {}) or {})
    runtime_diagnostics.setdefault("backend_name", runtime_backend_name)
    runtime_diagnostics.setdefault("cache_hit", False)

    initial_committee_view = build_committee_view(
        bundle=bundle,
        recommendation=recommendation,
        agents=[output.to_agent_output() for output in structured_outputs],
    )
    post_committee = run_post_committee_pipeline(
        bundle=bundle,
        committee_view=initial_committee_view,
        structured_outputs=structured_outputs,
        runtime_backend_name=runtime_backend_name,
        runtime_diagnostics=runtime_diagnostics,
    )
    agents = post_committee.agents
    committee_view = post_committee.committee_view
    runtime_diagnostics = post_committee.runtime_diagnostics

    reviews = [
        CommitteeReview(
            perspective=agent.role,
            recommendation=recommendation,
            confidence=agent.confidence,
            rationale=agent.summary,
        )
        for agent in agents
    ]
    committee_confidence = _committee_confidence(
        bundle=bundle,
        agents=agents,
        committee_view=committee_view,
        rule_confidence=rule_confidence,
        runtime_backend_name=runtime_backend_name,
    )
    agent_summary = {
        "final_recommendation": recommendation,
        "final_confidence": committee_confidence,
        "synthesis": _chair_summary(agents),
        "agents": {
            agent.role: {
                "summary": agent.summary,
                "findings": agent.findings,
                "confidence": agent.confidence,
            }
            for agent in agents
        },
        "runtime": runtime_diagnostics,
    }

    audit_metrics = {
        "n_agents": float(len(agents)),
        "rule_confidence": rule_confidence,
        "final_confidence": committee_confidence,
    }
    audit_metrics.update(_runtime_diagnostic_metrics(runtime_diagnostics))
    audit = AuditEntry(
        node="agno_agents",
        timestamp=_now(),
        summary=(
            f"Stage 2 scaffold completed via {runtime_backend_name} runner: "
            f"{', '.join(agent.role for agent in agents)}"
        ),
        metrics=audit_metrics,
    )
    return {
        "agent_outputs": agents,
        "committee_reviews": reviews,
        "agent_summary": agent_summary,
        "committee_view": committee_view,
        "stage2_runtime_diagnostics": runtime_diagnostics,
        "credit_policy_snapshot": credit_policy_snapshot,
        "final_recommendation": recommendation,
        "final_confidence": committee_confidence,
        "audit": [audit],
    }


def _chair_summary(agents: list[AgentOutput]) -> str:
    for agent in agents:
        if agent.role == "chair_report":
            return str(agent.summary)
    return str(agents[-1].summary) if agents else ""


def _committee_confidence(
    *,
    bundle: Stage2InputBundle,
    agents: list[AgentOutput],
    committee_view: dict[str, Any],
    rule_confidence: float,
    runtime_backend_name: str,
) -> float:
    """Blend model certainty, evidence quality, and agent certainty into one score."""
    probability = _clamp(bundle.probability_speculative)
    model_confidence = 0.45 + 0.35 * min(abs(probability - 0.5) * 2.0, 1.0)
    agent_confidence = _average_agent_confidence(agents)
    evidence_confidence = _external_evidence_quality(
        bundle.news_cache_snapshot,
        veto_triggered=bool(committee_view.get("veto_triggered", False)),
    )
    alignment_adjustment = _committee_alignment_adjustment(
        bundle=bundle,
        committee_label=str(committee_view.get("final_committee_label", "")),
    )
    fallback_penalty = -0.07 if "fallback" in runtime_backend_name else 0.0
    score = (
        0.35 * _clamp(rule_confidence)
        + 0.35 * model_confidence
        + 0.20 * agent_confidence
        + 0.10 * evidence_confidence
        + alignment_adjustment
        + fallback_penalty
    )
    return float(round(_clamp(score, minimum=0.2, maximum=0.95), 4))


def _average_agent_confidence(agents: list[AgentOutput]) -> float:
    if not agents:
        return 0.35
    return float(_clamp(sum(agent.confidence for agent in agents) / len(agents)))


def _committee_alignment_adjustment(
    *,
    bundle: Stage2InputBundle,
    committee_label: str,
) -> float:
    model_label = "적격" if bundle.prediction_label == "투자적격" else "부적격"
    if committee_label == model_label:
        return 0.08
    if committee_label == "보류":
        risk_band = str(bundle.rule_result.get("risk_band", ""))
        return 0.03 if risk_band == "watch" else 0.0
    return -0.06


def _stage2_runner() -> Stage2AgentRunner:
    deterministic_runner = DeterministicStage2AgentRunner(
        quant_credit_agent=_quant_credit_agent,
        evidence_audit_agent=_evidence_audit_agent,
        chair_report_agent=_chair_report_agent,
    )
    config = _stage2_runtime_config()
    runner_name = _stage2_runner_name()
    if runner_name in {"", "deterministic", "local", "offline"}:
        return deterministic_runner
    if runner_name == "agno":
        return AgnoStage2AgentRunner(
            deterministic_runner=deterministic_runner,
            routing_mode=config.agno_mode,
            model_provider=config.model_provider,
            model_name=config.model,
            quant_model_provider=config.quant_provider,
            quant_model_name=config.quant_model,
            evidence_model_provider=config.evidence_provider,
            evidence_model_name=config.evidence_model,
            chair_model_provider=config.chair_provider,
            chair_model_name=config.chair_model,
            max_tokens=config.max_tokens,
            fallback_on_error=config.fallback_on_error,
            runtime_config=config,
        )
    raise ValueError(
        f"Unsupported CAS_STAGE2_RUNNER value. Use 'deterministic' or 'agno', got {runner_name!r}."
    )


def _stage2_max_tokens() -> int:
    return int(_stage2_runtime_config().max_tokens)


def _stage2_runner_name() -> str:
    if "PYTEST_CURRENT_TEST" in os.environ and os.environ.get(
        "CAS_ALLOW_LIVE_STAGE2_IN_TESTS", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        return "deterministic"
    return str(_stage2_runtime_config().runner).strip().lower()


def _stage2_fallback_on_error() -> bool:
    return bool(_stage2_runtime_config().fallback_on_error)


def _stage2_runtime_config() -> Stage2RuntimeConfig:
    override = _STAGE2_RUNTIME_CONFIG_OVERRIDE.get()
    if override is not None:
        return override
    return Stage2RuntimeConfig.from_env(os.environ)


def _recommendation_from_score(score: float, thresholds: dict[str, float]) -> Recommendation:
    """Map a numeric suitability score to the legacy recommendation buckets."""
    if score >= float(thresholds["priority"]):
        return "priority"
    if score >= float(thresholds["watch"]):
        return "watch"
    if score >= float(thresholds["review"]):
        return "review"
    return "defer"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "_chair_summary",
    "_committee_confidence",
    "_recommendation_from_score",
    "_stage2_fallback_on_error",
    "_stage2_max_tokens",
    "_stage2_runner",
    "_stage2_runner_name",
    "_stage2_runtime_config",
    "run_stage2_committee",
    "stage2_runner_override",
    "stage2_runtime_config_override",
]
