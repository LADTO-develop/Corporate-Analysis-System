"""Chunk and embed the news corpus into ChromaDB.

Reference: https://docs.trychroma.com/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
from chromadb.config import Settings

from bfd.data.loaders.news import NewsCorpusLoader
from bfd.utils.io import read_yaml
from bfd.utils.logging import get_logger

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Very simple character-level chunker; upgrade to token-based later."""
    chunks: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


class NewsIngestor:
    """Ingests the news corpus into a ChromaDB collection."""

    def __init__(self, config_path: str | Path = "configs/data/news.yaml") -> None:
        self.config = read_yaml(config_path)
        vs = self.config["vector_store"]
        self.persist_dir = vs["persist_directory"]
        self.collection_name = vs["collection_name"]
        self.chunk_size = int(self.config["corpus"]["chunk_size"])
        self.chunk_overlap = int(self.config["corpus"]["chunk_overlap"])

        # Client with explicit persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    # ------------------------------------------------------------------
    def get_or_create_collection(self) -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.config["vector_store"].get("distance_metric", "cosine")},
        )

    # ------------------------------------------------------------------
    def ingest(self, df: pd.DataFrame | None = None) -> int:
        """Ingest either ``df`` (if given) or the full corpus from disk.

        Returns the number of chunks added.
        """
        if df is None:
            df = NewsCorpusLoader(Path("configs/data/news.yaml")).load_all()
        if df.empty:
            logger.warning("news_corpus_empty")
            return 0

        coll = self.get_or_create_collection()
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            chunks = _chunk_text(str(row["text"]), self.chunk_size, self.chunk_overlap)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{row['corp_code']}__{row['article_date']}__{idx}"
                ids.append(doc_id)
                docs.append(chunk)
                metas.append(
                    {
                        "corp_code": row["corp_code"],
                        "article_date": str(row["article_date"]),
                        "title": row.get("title", ""),
                        "chunk_index": idx,
                    }
                )
        if not ids:
            return 0

        logger.info("news_ingest", n_chunks=len(ids), collection=self.collection_name)
        # Chroma accepts documents without explicit embeddings iff an embedding
        # function is attached to the collection; for simplicity here we leave
        # that to the caller. The ingest just loads raw text.
        coll.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)
