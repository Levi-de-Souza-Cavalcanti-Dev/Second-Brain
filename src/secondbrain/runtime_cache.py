"""Process-local lazy cache for embedder and vector store (CLI search/ask)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from secondbrain.config import Settings
from secondbrain.constants import DEFAULT_COLLECTION_NAME
from secondbrain.embeddings.base import EmbedderProtocol
from secondbrain.vectorstore.factory import VectorStoreDeps, build_vector_store
from secondbrain.vectorstore.store import VectorStoreProtocol

_CACHE_KEY: tuple[Any, ...] | None = None
_EMBEDDER: EmbedderProtocol | None = None
_STORE: VectorStoreProtocol | None = None
_DIM: int = 0


def _cache_key(settings: Settings) -> tuple[Any, ...]:
    return (
        str(Path(settings.vectorstore_path).expanduser().resolve()),
        settings.embedding_provider,
        settings.ollama_host if settings.embedding_provider == "ollama" else "",
        settings.ollama_embed_model if settings.embedding_provider == "ollama" else "",
        settings.sentence_transformer_model
        if settings.embedding_provider == "sentence_transformers"
        else "",
    )


async def get_embedder_and_store(
    settings: Settings,
    *,
    dimension: int,
    cache_queries: bool = True,
) -> tuple[EmbedderProtocol, VectorStoreProtocol]:
    global _CACHE_KEY, _EMBEDDER, _STORE, _DIM  # noqa: PLW0603

    key = _cache_key(settings)
    if _EMBEDDER is not None and _STORE is not None and key == _CACHE_KEY and dimension == _DIM:
        return _EMBEDDER, _STORE

    from secondbrain.embeddings.factory import make_embedder  # noqa: PLC0415

    await clear_runtime_cache()

    embedder = make_embedder(settings, cache_queries=cache_queries)
    vs_path = str(Path(settings.vectorstore_path).expanduser().resolve())
    deps = VectorStoreDeps(dimension=dimension, collection_name=DEFAULT_COLLECTION_NAME)
    store = await build_vector_store(vs_path, deps)

    _CACHE_KEY = key
    _EMBEDDER = embedder
    _STORE = store
    _DIM = dimension
    return embedder, store


async def clear_runtime_cache() -> None:
    global _CACHE_KEY, _EMBEDDER, _STORE, _DIM  # noqa: PLW0603

    from secondbrain.embeddings.factory import aclose_embedder  # noqa: PLC0415

    if _EMBEDDER is not None:
        await aclose_embedder(_EMBEDDER)
    if _STORE is not None:
        await _STORE.close()
    _CACHE_KEY = None
    _EMBEDDER = None
    _STORE = None
    _DIM = 0
