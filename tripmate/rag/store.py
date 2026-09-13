"""In-memory vector store.

The knowledge base is 20 chunks. A brute-force cosine scan over a
(20 x dim) numpy matrix is a single matrix-vector product -- microseconds --
so FAISS or Chroma would add a dependency, a build step, and a persistence
story to solve a problem this corpus does not have. See the README's
"Scalability" section for what changes at a few hundred cities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tripmate.rag.embedder import Embedder
from tripmate.rag.ingest import Chunk


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """Holds chunk vectors and answers similarity queries."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            raise ValueError("Cannot build a VectorStore from zero chunks")
        self._chunks = chunks
        self._embedder = embedder
        self._matrix = embedder.encode([c.embedding_text() for c in chunks])

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        city_filter: str | None = None,
    ) -> list[ScoredChunk]:
        """Return the `top_k` most similar chunks, highest score first.

        `city_filter` restricts the candidate set before scoring, which is how
        we keep "what do I pack for Bangkok" from surfacing Reykjavik's
        packing section -- the two are lexically and semantically close.
        """
        if not query.strip():
            return []

        candidates = list(range(len(self._chunks)))
        if city_filter:
            wanted = city_filter.casefold()
            candidates = [
                i for i, c in enumerate(self._chunks) if c.city.casefold() == wanted
            ]
            if not candidates:
                return []

        query_vector = self._embedder.encode([query])[0]
        # Both sides are L2-normalised, so the dot product is cosine similarity.
        scores = self._matrix[candidates] @ query_vector

        k = min(top_k, len(candidates))
        # argpartition finds the top-k without fully sorting, then we sort those.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            ScoredChunk(chunk=self._chunks[candidates[i]], score=float(scores[i]))
            for i in top
        ]
