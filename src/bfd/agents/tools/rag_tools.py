"""RAG-layer tools exposed to agents."""

from __future__ import annotations

from datetime import date

from langchain_core.tools import tool

from bfd.rag.llm_features import NewsLLMFeaturizer
from bfd.rag.retrieval import NewsRetriever, parse_as_of


@tool
def search_news(
    query: str,
    corp_code: str | None = None,
    as_of: str | None = None,
    top_k: int = 10,
) -> list[dict[str, object]]:
    """Semantic search over the ingested news/disclosure corpus.

    Args:
        query: natural-language query.
        corp_code: optional KRX 6-digit code to restrict the search.
        as_of: optional ISO date; exclude news published after this date.
        top_k: number of hits to return.
    """
    retriever = NewsRetriever()
    as_of_date: date | None = parse_as_of(as_of) if as_of else None
    chunks = retriever.query(text=query, corp_code=corp_code, as_of=as_of_date, top_k=top_k)
    return [
        {
            "chunk_id": c.chunk_id,
            "title": c.title,
            "article_date": c.article_date,
            "corp_code": c.corp_code,
            "distance": c.distance,
            "text": c.text,
        }
        for c in chunks
    ]


@tool
def extract_llm_features(corp_code: str, as_of: str) -> dict[str, object]:
    """Produce structured news features for a firm as of a given date.

    Args:
        corp_code: KRX 6-digit code.
        as_of: ISO date (YYYY-MM-DD).
    """
    featurizer = NewsLLMFeaturizer()
    features = featurizer.featurize(corp_code=corp_code, as_of=parse_as_of(as_of))
    return features.model_dump()
