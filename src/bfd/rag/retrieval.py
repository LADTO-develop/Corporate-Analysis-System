"""Retrieval-side of the news RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from bfd.utils.io import read_yaml


@dataclass
class RetrievedChunk:
    """A single retrieved text chunk."""

    chunk_id: str
    text: str
    corp_code: str
    article_date: str
    title: str
    distance: float


class NewsRetriever:
    """Queries the ChromaDB collection produced by ``NewsIngestor``."""

    def __init__(self, config_path: str | Path = "configs/data/news.yaml") -> None:
        self.config = read_yaml(config_path)
        vs = self.config["vector_store"]
        self.client = chromadb.PersistentClient(
            path=vs["persist_directory"],
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=vs["collection_name"])

    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        *,
        corp_code: str | None = None,
        as_of: date | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Semantic + metadata filter search.

        Args:
            text: query string.
            corp_code: restrict to one firm.
            as_of: drop chunks published after this date (prevents look-ahead).
            top_k: number of hits to return.
        """
        where: dict[str, Any] = {}
        if corp_code is not None:
            where["corp_code"] = corp_code
        if as_of is not None:
            where["article_date"] = {"$lte": as_of.isoformat()}

        results = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where or None,
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc,
                    corp_code=meta.get("corp_code", ""),
                    article_date=meta.get("article_date", ""),
                    title=meta.get("title", ""),
                    distance=float(dist),
                )
            )
        return out


def parse_as_of(value: str) -> date:
    """Parse an ISO date string into a date."""
    return datetime.fromisoformat(value).date()
