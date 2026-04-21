"""Turn retrieved news chunks into scalar features via an LLM.

The LLM (Claude) is called once per firm-year with the top-K retrieved
chunks, and is asked to emit a structured JSON that matches the schema
in ``configs/data/news.yaml``. Structured output is enforced by Pydantic.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from bfd.rag.retrieval import NewsRetriever, RetrievedChunk
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema — matches output_schema in configs/data/news.yaml
# ---------------------------------------------------------------------------
class NewsFeatures(BaseModel):
    """Structured features extracted by the LLM from news chunks."""

    firm_specific_sentiment: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="해당 기업에 대한 뉴스의 톤 (-1 매우 부정, 0 중립, 1 매우 긍정).",
    )
    industry_sentiment: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="해당 산업 전반에 대한 뉴스의 톤.",
    )
    event_risk_flag: bool = Field(
        ...,
        description="M&A, 소송, 리콜, 횡령 등 이벤트 리스크 언급 여부.",
    )
    going_concern_mention: bool = Field(
        ...,
        description="'계속기업' 의심 또는 유사한 표현 존재 여부.",
    )
    governance_issue_mention: bool = Field(
        ...,
        description="지배구조/횡령/감사의견 관련 이슈 언급 여부.",
    )
    summary_ko: str = Field(
        ...,
        max_length=200,
        description="핵심 논지를 200자 이내로 요약 (한국어).",
    )


SYSTEM_PROMPT_KO = """당신은 한국 기업의 신용 위험을 평가하는 금융 애널리스트입니다.
제공된 뉴스 토막들을 종합적으로 읽고, 지정된 JSON 스키마에 정확히 맞춰 출력하세요.

원칙:
1. 근거 없는 과장/축소를 피하고, 뉴스에 명시된 내용만 반영합니다.
2. sentiment는 -1 ~ 1 실수, flag는 불리언, summary는 200자 이내 한국어.
3. 이벤트 리스크·계속기업·지배구조 이슈는 약한 의혹이어도 true로 표시하되,
   summary에 그 근거를 짧게 명시합니다.
4. 뉴스가 무관하거나 정보가 없으면 sentiment=0.0, flag=false로 중립 처리."""


class NewsLLMFeaturizer:
    """Converts retrieved chunks into a ``NewsFeatures`` record via Claude."""

    def __init__(self, config_path: str | Path = "configs/data/news.yaml") -> None:
        cfg = read_yaml(config_path)["llm_featurizer"]
        self.top_k = int(cfg.get("top_k", 15))
        self.model = ChatAnthropic(
            model=cfg.get("model", "claude-sonnet-4-6"),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg.get("max_tokens", 512)),
        )
        # LangChain's .with_structured_output routes Claude's output through
        # Pydantic validation for deterministic JSON.
        self.structured = self.model.with_structured_output(NewsFeatures)

    # ------------------------------------------------------------------
    def featurize(
        self,
        corp_code: str,
        as_of: date,
        query: str = "재무 위험, 신용, 실적, 공시, 소송",
        retriever: NewsRetriever | None = None,
    ) -> NewsFeatures:
        """Run the full retrieval + LLM extraction for one firm at one point in time."""
        if retriever is None:
            retriever = NewsRetriever()
        chunks = retriever.query(
            text=query,
            corp_code=corp_code,
            as_of=as_of,
            top_k=self.top_k,
        )
        if not chunks:
            return NewsFeatures(
                firm_specific_sentiment=0.0,
                industry_sentiment=0.0,
                event_risk_flag=False,
                going_concern_mention=False,
                governance_issue_mention=False,
                summary_ko="관련 뉴스 없음.",
            )

        prompt = self._format_prompt(chunks, corp_code, as_of)
        logger.info("llm_featurize", corp_code=corp_code, as_of=str(as_of), n_chunks=len(chunks))
        result: Any = self.structured.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT_KO),
                HumanMessage(content=prompt),
            ]
        )
        return result  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    @staticmethod
    def _format_prompt(chunks: list[RetrievedChunk], corp_code: str, as_of: date) -> str:
        lines = [
            f"대상 기업 코드: {corp_code}",
            f"기준일: {as_of.isoformat()}",
            "",
            "아래 뉴스/공시 토막들을 종합하여 JSON으로 출력하세요.",
            "",
        ]
        for i, c in enumerate(chunks, start=1):
            lines.append(f"[{i}] ({c.article_date}) {c.title}")
            lines.append(c.text.strip())
            lines.append("")
        return "\n".join(lines)
