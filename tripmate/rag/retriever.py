"""Destination knowledge retriever -- the business logic behind the RAG tool.

This layer knows about destinations and relevance. It knows nothing about the
LLM, tool schemas, or the agent loop.
"""

from __future__ import annotations

import re
from pathlib import Path

from tripmate.logging_setup import get_logger
from tripmate.rag.embedder import build_embedder
from tripmate.rag.ingest import load_chunks
from tripmate.rag.store import ScoredChunk, VectorStore

log = get_logger(__name__)


class DestinationRetriever:
    """Search the destination guide knowledge base."""

    def __init__(
        self,
        data_dir: Path,
        embedding_model: str,
        *,
        min_similarity: float | None = None,
        prefer_semantic: bool = True,
    ) -> None:
        chunks = load_chunks(data_dir)
        corpus = [c.embedding_text() for c in chunks]
        embedder = build_embedder(embedding_model, corpus, prefer_semantic=prefer_semantic)

        self._store = VectorStore(chunks, embedder)
        # An explicit override wins; otherwise take the floor that suits
        # whichever embedder we actually ended up with.
        self._min_similarity = (
            min_similarity if min_similarity is not None else embedder.default_min_similarity
        )
        self._cities = sorted({c.city for c in chunks})

        # Users name countries as often as cities ("a visa for Japan"), so
        # both resolve to the same guide. Longest alias first, so "Reykjavik"
        # is never shadowed by a shorter partial match.
        aliases: dict[str, str] = {c.city: c.city for c in chunks}
        aliases.update({c.country: c.city for c in chunks})
        self._alias_patterns = [
            (city, re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE))
            for alias, city in sorted(aliases.items(), key=lambda kv: len(kv[0]), reverse=True)
        ]

        log.info(
            "rag.index_built",
            extra={
                "chunks": len(self._store),
                "cities": self._cities,
                "embedder": self._store.embedder_name,
            },
        )

    @property
    def cities(self) -> list[str]:
        """Cities this knowledge base covers. Used to report scope honestly."""
        return list(self._cities)

    @property
    def chunk_count(self) -> int:
        return len(self._store)

    @property
    def embedder_name(self) -> str:
        return self._store.embedder_name

    def detect_city(self, query: str) -> str | None:
        """Return the covered city `query` refers to, by city or country name."""
        for city, pattern in self._alias_patterns:
            if pattern.search(query):
                return city
        return None

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        """Retrieve the most relevant chunks, filtered by the similarity floor.

        Returning nothing is a valid, useful answer -- it lets the agent say
        "the guides do not cover that" instead of paraphrasing a weak match.
        """
        city = self.detect_city(query)
        hits = self._store.search(query, top_k=top_k, city_filter=city)
        kept = [h for h in hits if h.score >= self._min_similarity]

        log.info(
            "rag.search",
            extra={
                "query": query,
                "city_filter": city,
                "candidates": len(hits),
                "kept": len(kept),
                "top_score": round(hits[0].score, 4) if hits else None,
                "chunk_ids": [h.chunk.chunk_id for h in kept],
            },
        )
        return kept
