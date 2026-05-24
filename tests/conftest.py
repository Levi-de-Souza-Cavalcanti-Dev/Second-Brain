"""Pytest configuration / fixtures."""

from __future__ import annotations

import pytest


class FakeEmbedder:
    dimension: int = 16

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [
            [(idx + offset) / 1000.0 for offset in range(self.dimension)]
            for idx, _ in enumerate(texts)
        ]


@pytest.fixture
def monkeypatch_fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    fake = FakeEmbedder()

    def _maker(_settings: object) -> FakeEmbedder:
        return fake

    monkeypatch.setattr("secondbrain.embeddings.factory.make_embedder", _maker)
    monkeypatch.setattr("secondbrain.retrieval.retriever.make_embedder", _maker)
    return fake


@pytest.fixture(autouse=True)
def _clear_runtime_cache() -> None:
    import asyncio

    from secondbrain.runtime_cache import clear_runtime_cache

    yield
    asyncio.run(clear_runtime_cache())
