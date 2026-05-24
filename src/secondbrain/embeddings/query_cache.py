from __future__ import annotations

from collections import OrderedDict
from typing import Any

from secondbrain.config import Settings
from secondbrain.embeddings.base import EmbedderProtocol


def _cache_key(settings: Settings, query: str) -> tuple[Any, ...]:
    return (
        query,
        settings.embedding_provider,
        settings.ollama_embed_model if settings.embedding_provider == "ollama" else "",
        settings.sentence_transformer_model
        if settings.embedding_provider == "sentence_transformers"
        else "",
        settings.openai_embed_model if settings.embedding_provider == "openai" else "",
    )


class CachedQueryEmbedder:
    """Wraps an embedder with LRU cache for single-query embeddings."""

    def __init__(self, inner: EmbedderProtocol, settings: Settings) -> None:
        self._inner = inner
        self._settings = settings
        self._maxsize = max(0, settings.query_embed_cache_size)
        self._cache: OrderedDict[tuple[Any, ...], list[float]] = OrderedDict()

    def _get_cached(self, key: tuple[Any, ...]) -> list[float] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _set_cached(self, key: tuple[Any, ...], value: list[float]) -> None:
        if self._maxsize <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) == 1 and self._maxsize > 0:
            key = _cache_key(self._settings, texts[0])
            hit = self._get_cached(key)
            if hit is not None:
                return [hit]
            vec = (await self._inner.embed_many(texts))[0]
            self._set_cached(key, vec)
            return [vec]
        return await self._inner.embed_many(texts)

    @property
    def inner(self) -> EmbedderProtocol:
        return self._inner
