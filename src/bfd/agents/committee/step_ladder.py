"""Step-ladder technique — panel grows incrementally with each round.

Round 1: panel of ``initial_panel_size`` LLM calls produces an initial
consensus summary. Each subsequent round adds one new "member" who
receives the summary so far plus the base evidence, and re-opines.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bfd.agents.state import AgentState, CommitteeOpinion


class _StepResponse(BaseModel):
    verdict: str = Field(..., description="'borderline' | 'healthy' | 'uncertain'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    new_insight_ko: str = Field(..., description="이번 라운드에서 추가된 새로운 관점 (<= 200자)")
    consensus_ko: str = Field(..., description="현재까지의 합의 요약 (<= 200자)")


class StepLadderStrategy:
    """Panel-size-growing deliberation."""

    def __init__(
        self,
        llm: ChatAnthropic,
        *,
        initial_panel_size: int = 2,
        max_panel_size: int = 5,
        **_: Any,
    ) -> None:
        self.llm = llm
        self.initial_panel_size = initial_panel_size
        self.max_panel_size = max_panel_size
        self.structured = llm.with_structured_output(_StepResponse)

    # ------------------------------------------------------------------
    def deliberate(self, state: AgentState) -> list[CommitteeOpinion]:
        opinions: list[CommitteeOpinion] = []
        running_summary = "(아직 없음)"

        for round_idx in range(self.initial_panel_size, self.max_panel_size + 1):
            role = f"round_{round_idx}"
            sys_msg = (
                f"당신은 {round_idx}번째 라운드의 스텝래더 참가자입니다. "
                "지금까지의 합의 요약을 받고, 새로운 관점을 추가한 뒤 판정을 갱신하세요."
            )
            prompt = _round_prompt(state, round_idx, running_summary)
            resp: _StepResponse = self.structured.invoke(
                [SystemMessage(content=sys_msg), HumanMessage(content=prompt)]
            )

            running_summary = resp.consensus_ko
            opinions.append(
                CommitteeOpinion(
                    technique="step_ladder",
                    role=role,
                    verdict=_coerce(resp.verdict),  # type: ignore[arg-type]
                    confidence=resp.confidence,
                    rationale_ko=resp.new_insight_ko + " | 합의: " + resp.consensus_ko,
                )
            )
        return opinions


def _round_prompt(state: AgentState, round_idx: int, running_summary: str) -> str:
    ens = state.get("ensemble_proba", 0.5)
    macro = state.get("macro_overlay", {}) or {}
    news = state.get("news_overlay", {}) or {}
    lines = [
        f"라운드 {round_idx}",
        f"기업 {state.get('corp_code')}, {state.get('market')}, FY{state.get('fiscal_year')}",
        f"AI 앙상블 P(borderline)={ens:.3f}",
        f"거시 {macro.get('adjustment', 0.0):+.3f} | 뉴스 {news.get('adjustment', 0.0):+.3f}",
        "",
        "지금까지의 합의 요약:",
        running_summary,
    ]
    return "\n".join(lines)


def _coerce(raw: str) -> str:
    low = raw.strip().lower()
    if "border" in low:
        return "borderline"
    if "heal" in low:
        return "healthy"
    return "uncertain"
