"""News overlay node — LLM-driven adjustment from retrieved news/disclosures."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from bfd.agents.state import AgentState, AuditEntry, NewsOverlay
from bfd.rag.llm_features import NewsLLMFeaturizer
from bfd.utils.logging import get_logger
from bfd.utils.time import fiscal_year_end

logger = get_logger(__name__)


def run(state: AgentState) -> dict[str, Any]:
    """Pull LLM-extracted news features and derive a bounded adjustment."""
    corp_code = state["corp_code"]
    fiscal_year = state["fiscal_year"]
    as_of = fiscal_year_end(fiscal_year)

    try:
        featurizer = NewsLLMFeaturizer()
        features = featurizer.featurize(corp_code=corp_code, as_of=as_of)
    except Exception as exc:  # noqa: BLE001
        audit = AuditEntry(
            node="news_overlay",
            timestamp=_now(),
            summary=f"News featurization failed: {exc}. Skipping.",
        )
        return {"news_overlay": NewsOverlay().model_dump(), "audit": [audit]}

    # Heuristic: sentiment pushes probability in opposite direction, hard
    # flags push toward borderline. Bounded in [-0.07, +0.07].
    adjustment = 0.0
    rationale: list[str] = []

    adjustment += -0.02 * features.firm_specific_sentiment
    adjustment += -0.01 * features.industry_sentiment

    if features.event_risk_flag:
        adjustment += 0.02
        rationale.append("이벤트 리스크 언급")
    if features.going_concern_mention:
        adjustment += 0.03
        rationale.append("계속기업 의심 언급")
    if features.governance_issue_mention:
        adjustment += 0.015
        rationale.append("지배구조 이슈 언급")

    adjustment = float(max(-0.07, min(0.07, adjustment)))
    if not rationale:
        rationale.append(features.summary_ko or "주요 이슈 없음")

    overlay = NewsOverlay(
        firm_specific_sentiment=features.firm_specific_sentiment,
        industry_sentiment=features.industry_sentiment,
        event_risk_flag=features.event_risk_flag,
        going_concern_mention=features.going_concern_mention,
        adjustment=adjustment,
        rationale_ko=" / ".join(rationale),
    )

    audit = AuditEntry(
        node="news_overlay",
        timestamp=_now(),
        summary=f"News adjustment={adjustment:+.3f} ({overlay.rationale_ko})",
        metrics={"news_adjustment": adjustment},
    )
    return {"news_overlay": overlay.model_dump(), "audit": [audit]}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
