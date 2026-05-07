"""Embedding providers (protocol + implementations)."""

from __future__ import annotations

from typing import Protocol


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class EmbedderProtocol(Protocol):
    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
