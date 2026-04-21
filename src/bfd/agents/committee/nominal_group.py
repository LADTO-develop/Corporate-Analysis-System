"""Nominal-group technique — independent votes then aggregation.

``num_voters`` independent LLM calls (each with a different framing prompt)
produce independent verdicts, which are then aggregated by one of:
    * majority vote
    * Borda count (weights higher-ranked preferences more)
    * weighted average of confidences
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bfd.agents.state import AgentState, CommitteeOpinion

# Each voter gets a slightly different role framing to simulate diversity
VOTER_ROLES_KO = [
    "은행의 기업여신 심사역",
    "자산운용사 크레딧 애널리스트",
    "회계법인의 감사 파트너",
    "산업 전문 저널리스트",
    "기업 구조조정 전문 컨설턴트",
    "재무부장 출신 임원",
    "스타트업 액셀러레이터 파트너",
]


class _VoteResponse(BaseModel):
    verdict: str = Field(..., description="'borderline' | 'healthy' | 'uncertain'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    preference_ranking: list[str] = Field(
        default_factory=lambda: ["borderline", "healthy", "uncertain"],
        description="세 옵션의 선호 순위 (앞쪽이 더 선호). Borda 집계에 사용.",
    )
    rationale_ko: str = Field(..., description="판정 근거 (<= 200자)")


class NominalGroupStrategy:
    """Run N independent voter prompts, then aggregate."""

    def __init__(
        self,
        llm: ChatAnthropic,
        *,
        num_voters: int = 5,
        voting_method: str = "borda",
        **_: Any,
    ) -> None:
        self.llm = llm
        self.num_voters = min(num_voters, len(VOTER_ROLES_KO))
        self.voting_method = voting_method
        self.structured = llm.with_structured_output(_VoteResponse)

    # ------------------------------------------------------------------
    def deliberate(self, state: AgentState) -> list[CommitteeOpinion]:
        responses: list[tuple[str, _VoteResponse]] = []
        for role in VOTER_ROLES_KO[: self.num_voters]:
            sys_msg = (
                f"당신은 {role}입니다. 주어진 기업 데이터를 독립적으로 평가하여 "
                "한계기업 여부에 대한 판정을 내리세요. 다른 평가자의 의견은 모릅니다."
            )
            resp: _VoteResponse = self.structured.invoke(
                [
                    SystemMessage(content=sys_msg),
                    HumanMessage(content=_voter_prompt(state)),
                ]
            )
            responses.append((role, resp))

        opinions = [
            CommitteeOpinion(
                technique="nominal_group",
                role=role,
                verdict=_coerce_verdict(resp.verdict),  # type: ignore[arg-type]
                confidence=resp.confidence,
                rationale_ko=resp.rationale_ko,
            )
            for role, resp in responses
        ]
        return opinions


def _voter_prompt(state: AgentState) -> str:
    ens = state.get("ensemble_proba", 0.5)
    macro = state.get("macro_overlay", {}) or {}
    news = state.get("news_overlay", {}) or {}
    feats = state.get("features", {}) or {}
    lines = [
        f"기업 {state.get('corp_code')}, {state.get('market')}, FY{state.get('fiscal_year')}",
        f"AI 앙상블 P(borderline)={ens:.3f}",
        f"거시 {macro.get('adjustment', 0.0):+.3f} | {macro.get('rationale_ko', '')}",
        f"뉴스 {news.get('adjustment', 0.0):+.3f} | {news.get('rationale_ko', '')}",
        "",
        "주요 피처:",
    ]
    for name, value in list(feats.items())[:12]:
        lines.append(f"  - {name}: {value}")
    return "\n".join(lines)


def _coerce_verdict(raw: str) -> str:
    low = raw.strip().lower()
    if "border" in low:
        return "borderline"
    if "heal" in low:
        return "healthy"
    return "uncertain"
