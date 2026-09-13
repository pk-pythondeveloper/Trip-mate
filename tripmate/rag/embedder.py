"""Pluggable text embedding.

Two implementations behind one interface:

* `SentenceTransformerEmbedder` -- real semantic embeddings (all-MiniLM-L6-v2),
  running locally with no API key. This is the default.
* `LexicalEmbedder` -- a dependency-free TF-IDF fallback in pure numpy, used
  automatically when sentence-transformers is unavailable. It keeps the test
  suite fast and lets a reviewer clone and run without a model download.

Both return L2-normalised vectors, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from tripmate.logging_setup import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Crude suffix stripper.

    Not linguistically principled, but it collapses the inflections that
    actually matter here -- a user asking what to "pack" must reach the
    "PACKING TIPS" section, and "customs"/"custom" must be one term. Applied
    identically to documents and queries, so both sides stay comparable.
    """
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", "")):
        if token.endswith(suffix) and len(token) - len(suffix) + len(replacement) >= 3:
            return token[: -len(suffix)] + replacement
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


class Embedder(Protocol):
    """Anything that can turn text into normalised vectors.

    `default_min_similarity` travels with the embedder because cosine scores
    from sparse and dense models are not on the same scale -- a 0.25 TF-IDF
    match is strong, while a 0.25 MiniLM match is weak. A single hardcoded
    floor would be wrong for one of them.
    """

    name: str
    default_min_similarity: float

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Semantic embeddings via sentence-transformers (downloaded once, ~80MB)."""

    default_min_similarity = 0.25

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.name = f"sentence-transformers/{model_name}"
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class LexicalEmbedder:
    """TF-IDF vectoriser in pure numpy.

    Fitted on the corpus at build time; queries are projected into the same
    vocabulary. Unknown query terms are ignored, which is the behaviour we
    want: a question about a city we hold no guide for has no vocabulary
    overlap at all, scores an exact 0.0, and trips the retriever's floor
    instead of returning noise.

    The floor is set low on purpose. Measured on this corpus, genuine
    questions score 0.15-0.35 while off-topic ones reach 0.17, so the ranges
    overlap and no threshold separates them. Deciding a request is out of
    scope is the agent's job (see agent/prompts.py), not the retriever's; the
    floor exists only to suppress the zero-overlap case.
    """

    default_min_similarity = 0.10

    def __init__(self) -> None:
        self.name = "lexical-tfidf"
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [_stem(t) for t in _TOKEN_RE.findall(text.lower())]

    def fit(self, texts: list[str]) -> "LexicalEmbedder":
        doc_freq: Counter[str] = Counter()
        for text in texts:
            doc_freq.update(set(self._tokenize(text)))

        self._vocab = {term: i for i, term in enumerate(sorted(doc_freq))}
        n_docs = max(len(texts), 1)
        idf = np.zeros(len(self._vocab), dtype=np.float32)
        for term, index in self._vocab.items():
            # Smoothed IDF; the +1 stops terms appearing in every document
            # from collapsing to exactly zero weight.
            idf[index] = math.log((1 + n_docs) / (1 + doc_freq[term])) + 1.0
        self._idf = idf
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._vocab:
            raise RuntimeError("LexicalEmbedder.fit() must be called before encode()")
        matrix = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(self._tokenize(text)).items():
                index = self._vocab.get(term)
                if index is not None:
                    matrix[row, index] = count * self._idf[index]
        return _normalise(matrix)


def build_embedder(
    model_name: str, corpus: list[str], *, prefer_semantic: bool = True
) -> Embedder:
    """Return the best embedder available, degrading gracefully.

    `corpus` is consumed only by the lexical fallback, which must be fitted
    before it can encode.
    """
    if prefer_semantic:
        try:
            embedder = SentenceTransformerEmbedder(model_name)
            log.info(
                "embedder.selected",
                extra={"embedder": embedder.name, "mode": "semantic"},
            )
            return embedder
        except Exception as exc:  # ImportError, download failure, offline
            log.warning(
                "embedder.semantic_unavailable",
                extra={
                    "requested_model": model_name,
                    "reason": str(exc),
                    "fallback": "lexical-tfidf",
                },
            )

    fallback = LexicalEmbedder().fit(corpus)
    log.info("embedder.selected", extra={"embedder": fallback.name, "mode": "lexical"})
    return fallback
