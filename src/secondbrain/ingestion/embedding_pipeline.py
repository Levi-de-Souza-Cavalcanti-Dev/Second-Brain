"""Embedding batch pipeline for vault indexing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from secondbrain.config import Settings
from secondbrain.embeddings.base import EmbedderProtocol

T = TypeVar("T")
R = TypeVar("R")


class EmbeddingPipeline:
    def __init__(self, settings: Settings, embedder: EmbedderProtocol) -> None:
        self._settings = settings
        self._embedder = embedder
        self._sem = asyncio.Semaphore(max(1, settings.embedding_batch_concurrency))
        self._batch_sz = max(1, settings.embedding_request_batch_size)

    async def embed_batches(self, texts: list[str]) -> list[list[float]]:
        async def one_batch(batch: list[str]) -> list[list[float]]:
            async with self._sem:
                return await self._embedder.embed_many(batch)

        batches = [texts[i : i + self._batch_sz] for i in range(0, len(texts), self._batch_sz)]
        if not batches:
            return []
        chunks_embedded = await asyncio.gather(*(one_batch(b) for b in batches))
        out: list[list[float]] = []
        for chunk in chunks_embedded:
            out.extend(chunk)
        return out

    async def map_parallel(
        self,
        items: list[T],
        fn: Callable[[T], Awaitable[R]],
    ) -> list[R]:
        sem = asyncio.Semaphore(max(1, self._settings.embedding_batch_concurrency))

        async def wrapped(item: T) -> R:
            async with sem:
                return await fn(item)

        return list(await asyncio.gather(*(wrapped(i) for i in items)))
