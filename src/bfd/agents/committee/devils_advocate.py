"""Devil's advocate strategy.

Given the current leaning implied by ``state['ensemble_proba']`` plus any
macro/news adjustments, the devil's advocate generates ``num_objections``
counter-arguments and then takes a verdict against the majority view.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bfd.agents.state import AgentState, CommitteeOpinion


_SYSTEM_KO = """당신은 '악마의 대변인(Devil's Advocate)'입니다.
현재 판정에 반대되는 논거만을 체계적으로 제기하여, 집단사고(groupthink)를 방지하고
간과된 위험/근거를 드러내는 것이 역할입니다.

원칙:
1. 현재 다수 의견과 반대되는 입장을 먼저 정한 뒤, 그 입장에서만 말합니다.
2. 가능한 근거는 재무지표, 거시 지표, 뉴스, 공시, 역사적 유사 사례 등 무엇이든 사용합니다.
3. 주장은 출력 JSON 스키마의 필드에 압축적으로 요약합니다."""


class _DAResponse(BaseModel):
    counter_verdict: str = Field(..., description="'borderline' 또는 'healthy'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    objections_ko: list[str] = Field(..., description="반대 논거 요약 리스트")
    rationale_ko: str = Field(..., description="한 문단으로 된 종합 반대의견 (<= 300자)")


class DevilsAdvocateStrategy:
    """One-shot devil's advocate pass."""

    def __init__(
        self,
        llm: ChatAnthropic,
        *,
        num_objections: int = 3,
        force_contrarian: bool = True,
        **_: Any,
    ) -> None:
        self.llm = llm
        self.num_objections = num_objections
        self.force_contrarian = force_contrarian
        self.structured = llm.with_structured_output(_DAResponse)

    # ------------------------------------------------------------------
    def deliberate(self, state: AgentState) -> list[CommitteeOpinion]:
        prompt = _format_prompt(state, self.num_objections, self.force_contrarian)
        resp: _DAResponse = self.structured.invoke(
            [SystemMessage(content=_SYSTEM_KO), HumanMessage(content=prompt)]
        )
        verdict = "borderline" if resp.counter_verdict.lower() == "borderline" else "healthy"
        return [
            CommitteeOpinion(
                technique="devils_advocate",
                role="devils_advocate",
                verdict=verdict,  # type: ignore[arg-type]
                confidence=resp.confidence,
                rationale_ko=resp.rationale_ko + " | 반론: " + " | ".join(resp.objections_ko),
            )
        ]


def _format_prompt(state: AgentState, k: int, force_contrarian: bool) -> str:
    ens = state.get("ensemble_proba", 0.5)
    majority = "borderline" if ens >= 0.5 else "healthy"
    contra = "healthy" if majority == "borderline" else "borderline"
    macro = state.get("macro_overlay", {}) or {}
    news = state.get("news_overlay", {}) or {}
    feats = state.get("features", {}) or {}

    lines = [
        f"대상 기업: {state.get('corp_code')}  시장: {state.get('market')}  회계연도: {state.get('fiscal_year')}",
        f"앙상블 P(borderline)={ens:.3f}  (다수의견: {majority})",
        f"거시 조정={macro.get('adjustment', 0.0):+.3f} ({macro.get('rationale_ko','')})",
        f"뉴스 조정={news.get('adjustment', 0.0):+.3f} ({news.get('rationale_ko','')})",
        "",
        "주요 재무 피처 (상위 15개):",
    ]
    for name, value in list(feats.items())[:15]:
        lines.append(f"  - {name}: {value}")

    if force_contrarian:
        lines.append("")
        lines.append(f"반드시 '{contra}' 입장을 취하고, 반론 {k}건을 제시하세요.")
    else:
        lines.append(f"반론 {k}건을 제시하고, 귀하의 판정을 선택하세요.")
    return "\n".join(lines)
