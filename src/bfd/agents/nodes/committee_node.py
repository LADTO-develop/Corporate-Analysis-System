"""Committee node — orchestrates bias-removal techniques to reach a verdict.

Each enabled technique (악마의 대변인, 6 모자, 명목집단법, 스텝래더) is
instantiated as a strategy object in ``bfd.agents.committee`` and produces a
list of ``CommitteeOpinion`` entries appended to the state's audit-style
reducer channel. After all strategies run, a weighted vote yields the
``final_verdict`` and ``final_confidence``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic

from bfd.agents.committee.devils_advocate import DevilsAdvocateStrategy
from bfd.agents.committee.nominal_group import NominalGroupStrategy
from bfd.agents.committee.six_hats import SixHatsStrategy
from bfd.agents.committee.step_ladder import StepLadderStrategy
from bfd.agents.state import AgentState, AuditEntry, CommitteeOpinion
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)

STRATEGY_CLASSES = {
    "devils_advocate": DevilsAdvocateStrategy,
    "six_hats": SixHatsStrategy,
    "nominal_group": NominalGroupStrategy,
    "step_ladder": StepLadderStrategy,
}


def run(state: AgentState) -> dict[str, Any]:
    """Run the committee, aggregate opinions, set final verdict."""
    cfg = read_yaml(Path("configs/agent/committee.yaml"))
    llm = _build_llm(cfg["committee_llm"])

    opinions: list[CommitteeOpinion] = []
    for spec in cfg["strategies"]:
        if not spec.get("enabled", True):
            continue
        kind = spec["kind"]
        strategy_cls = STRATEGY_CLASSES[kind]
        strategy = strategy_cls(llm=llm, **spec.get("params", {}))
        logger.info("committee_run_strategy", kind=kind)
        strategy_opinions = strategy.deliberate(state)
        opinions.extend(strategy_opinions)

    verdict, confidence = _aggregate(opinions, cfg["aggregation"])

    audit = AuditEntry(
        node="committee",
        timestamp=_now(),
        summary=(
            f"Committee verdict={verdict} (confidence={confidence:.3f}) "
            f"from {len(opinions)} opinions"
        ),
        metrics={"final_confidence": confidence, "n_opinions": float(len(opinions))},
    )

    return {
        "committee_opinions": opinions,
        "final_verdict": verdict,
        "final_confidence": confidence,
        "audit": [audit],
    }


def _build_llm(llm_cfg: dict[str, Any]) -> ChatAnthropic:
    return ChatAnthropic(
        model=llm_cfg["model"],
        temperature=float(llm_cfg.get("temperature", 0.3)),
        max_tokens=int(llm_cfg.get("max_tokens", 1024)),
    )


def _aggregate(
    opinions: list[CommitteeOpinion],
    agg_cfg: dict[str, Any],
) -> tuple[str, float]:
    """Produce a final verdict + confidence from all committee opinions."""
    if not opinions:
        return "uncertain", 0.0

    method = agg_cfg.get("method", "weighted_vote")
    threshold = float(agg_cfg.get("uncertainty_threshold", 0.15))

    if method == "majority":
        votes: dict[str, int] = {}
        for op in opinions:
            votes[op.verdict] = votes.get(op.verdict, 0) + 1
        verdict = max(votes, key=lambda k: votes[k])
        confidence = votes[verdict] / len(opinions)
    elif method == "weighted_vote":
        weights = agg_cfg.get("weights", {})
        scores: dict[str, float] = {"borderline": 0.0, "healthy": 0.0, "uncertain": 0.0}
        total = 0.0
        for op in opinions:
            w = float(weights.get(op.technique, 0.1)) * op.confidence
            scores[op.verdict] = scores.get(op.verdict, 0.0) + w
            total += w
        if total > 0:
            for k in scores:
                scores[k] /= total
        verdict = max(scores, key=lambda k: scores[k])
        confidence = scores[verdict]
    else:
        # Fallback: simple average
        verdict = "uncertain"
        confidence = 0.0

    if confidence < threshold:
        return "uncertain", confidence
    return verdict, confidence


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
