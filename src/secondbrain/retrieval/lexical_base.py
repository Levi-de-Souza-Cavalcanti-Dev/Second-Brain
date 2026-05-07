from __future__ import annotations

from typing import Protocol

from secondbrain.models import SearchHit


class LexicalRetriever(Protocol):
    """Optional lexical retrieval (noop default)."""

    async def lexical_search(self, query: str, top_k: int) -> list[SearchHit]: ...


class NoopLexicalRetriever:
    async def lexical_search(self, query: str, top_k: int) -> list[SearchHit]:
        _, _ = query, top_k
        return []
