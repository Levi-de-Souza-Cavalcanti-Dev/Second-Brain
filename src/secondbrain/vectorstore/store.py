from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from secondbrain.models import DocumentChunk, SearchHit


class VectorStoreError(RuntimeError):
    """Vector storage operations failed."""


class VectorStoreProtocol(Protocol):
    async def upsert_documents(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None: ...

    async def delete_by_source_path(self, source_path: str) -> None: ...

    async def semantic_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
    ) -> list[SearchHit]: ...

    async def close(self) -> None: ...
