"""Edward de Bono's Six Thinking Hats strategy.

Each hat is a separate LLM invocation with a dedicated system prompt. The
final ``blue`` hat receives the summaries of the other five and produces a
synthesized verdict. All six opinions are appended to the committee trail.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bfd.agents.state import AgentState, CommitteeOpinion

HAT_ORDER = ["white", "red", "black", "yellow", "green", "blue"]

HAT_SYSTEM_PROMPTS_KO: dict[str, str] = {
    "white": (
        "당신은 '백색 모자'입니다. 오직 객관적 사실과 수치만 말합니다. "
        "의견/해석/감정 표현은 금지합니다."
    ),
    "red": (
        "당신은 '적색 모자'입니다. 직관과 감정적 반응만 말합니다. "
        "논리적 근거 제시 없이 느낌을 표현합니다. 단, 판정은 반드시 내립니다."
    ),
    "black": (
        "당신은 '흑색 모자'입니다. 위험, 결함, 비판적 관점만 말합니다. "
        "낙관적 요소는 의도적으로 무시합니다."
    ),
    "yellow": (
        "당신은 '황색 모자'입니다. 긍정적 요소, 기회, 강점만 말합니다. "
        "위험 요소는 의도적으로 무시합니다."
    ),
    "green": (
        "당신은 '녹색 모자'입니다. 창의적 대안과 시나리오를 제시합니다. "
        "'만약 ~라면'의 형태로 미래 경로를 탐색합니다."
    ),
    "blue": (
        "당신은 '청색 모자'입니다. 앞선 다섯 모자의 발언을 통합하여 최종 판정을 내립니다. "
        "절차적 통제와 종합이 역할입니다."
    ),
}


class _HatResponse(BaseModel):
    verdict: str = Field(..., description="'borderline' | 'healthy' | 'uncertain'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale_ko: str = Field(..., description="해당 모자 관점의 핵심 요지 (<= 200자)")


class SixHatsStrategy:
    """Six-hats thinking with sequential hat invocations."""

    def __init__(self, llm: ChatAnthropic, *, hats: list[str] | None = None, **_: Any) -> None:
        self.llm = llm
        self.hats = hats or HAT_ORDER
        self.structured = llm.with_structured_output(_HatResponse)

    # ------------------------------------------------------------------
    def deliberate(self, state: AgentState) -> list[CommitteeOpinion]:
        summaries: list[tuple[str, _HatResponse]] = []
        opinions: list[CommitteeOpinion] = []
        base_context = _base_context(state)

        for hat in self.hats:
            if hat == "blue":
                prompt = _blue_prompt(base_context, summaries)
            else:
                prompt = base_context

            resp: _HatResponse = self.structured.invoke(
                [SystemMessage(content=HAT_SYSTEM_PROMPTS_KO[hat]), HumanMessage(content=prompt)]
            )
            summaries.append((hat, resp))

            verdict = _coerce_verdict(resp.verdict)
            opinions.append(
                CommitteeOpinion(
                    technique="six_hats",
                    role=hat,
                    verdict=verdict,  # type: ignore[arg-type]
                    confidence=resp.confidence,
                    rationale_ko=resp.rationale_ko,
                )
            )
        return opinions


def _base_context(state: AgentState) -> str:
    ens = state.get("ensemble_proba", 0.5)
    macro = state.get("macro_overlay", {}) or {}
    news = state.get("news_overlay", {}) or {}
    feats = state.get("features", {}) or {}

    lines = [
        f"대상 기업: {state.get('corp_code')}  시장: {state.get('market')}  회계연도: {state.get('fiscal_year')}",
        f"앙상블 P(borderline)={ens:.3f}",
        f"거시 조정={macro.get('adjustment', 0.0):+.3f} | {macro.get('rationale_ko', '')}",
        f"뉴스 조정={news.get('adjustment', 0.0):+.3f} | {news.get('rationale_ko', '')}",
        "",
        "주요 피처 (상위 15):",
    ]
    for name, value in list(feats.items())[:15]:
        lines.append(f"  - {name}: {value}")
    return "\n".join(lines)


def _blue_prompt(base: str, summaries: list[tuple[str, _HatResponse]]) -> str:
    parts = [base, "", "지금까지 다섯 모자의 발언:"]
    for hat, resp in summaries:
        parts.append(f"[{hat}] verdict={resp.verdict}, conf={resp.confidence:.2f}, {resp.rationale_ko}")
    parts.append("")
    parts.append("위 발언을 통합하여 최종 판정을 내리세요.")
    return "\n".join(parts)


def _coerce_verdict(raw: str) -> str:
    low = raw.strip().lower()
    if "border" in low:
        return "borderline"
    if "heal" in low:
        return "healthy"
    return "uncertain"
