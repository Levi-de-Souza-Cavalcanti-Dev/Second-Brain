"""Cross-encoder reranking for retrieval results."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from secondbrain.embeddings.base import EmbeddingError
from secondbrain.models import SearchHit


class RerankerProtocol(Protocol):
    async def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]: ...


class BgeReranker:
    """Sentence-transformers cross-encoder reranker."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers necessário para reranker. "
                    "Instale: pip install -e '.[embeddings-st]'",
                ) from e
            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        if not hits:
            return []
        model = self._load()
        pairs = [[query, h.text] for h in hits]

        def _score() -> list[float]:
            scores = cast(Any, model).predict(pairs)
            return [float(s) for s in scores]

        raw_scores = await asyncio.to_thread(_score)
        ranked = sorted(
            zip(hits, raw_scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        max_score = max((s for _, s in ranked), default=1.0) or 1.0
        return [
            SearchHit(text=h.text, score=float(s / max_score), metadata=h.metadata)
            for h, s in ranked
        ]


class NoopReranker:
    async def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        _ = query
        return hits[:top_n]
