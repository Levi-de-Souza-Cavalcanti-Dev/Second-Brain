from __future__ import annotations

from secondbrain.config import Settings
from secondbrain.embeddings.base import EmbedderProtocol
from secondbrain.embeddings.ollama_embedder import OllamaEmbedder
from secondbrain.embeddings.openai_embedder import OpenAIEmbedder
from secondbrain.embeddings.query_cache import CachedQueryEmbedder
from secondbrain.embeddings.sentence_transformers_embedder import SentenceTransformerEmbedder


def make_embedder(settings: Settings, *, cache_queries: bool = False) -> EmbedderProtocol:
    if settings.embedding_provider == "ollama":
        embedder: EmbedderProtocol = OllamaEmbedder(
            base_url=settings.ollama_host,
            model=settings.ollama_embed_model,
            timeout_seconds=settings.ollama_embed_timeout_seconds,
        )
    elif settings.embedding_provider == "openai":
        base_url = settings.openai_embed_base_url or settings.openai_compat_base_url
        if not base_url:
            raise ValueError(
                "OPENAI_EMBED_BASE_URL ou OPENAI_COMPAT_BASE_URL é obrigatório para EMBEDDING_PROVIDER=openai.",
            )
        embedder = OpenAIEmbedder(
            base_url=base_url,
            api_key=settings.openai_compat_api_key,
            model=settings.openai_embed_model,
            timeout_seconds=settings.openai_compat_timeout_seconds,
        )
    else:
        embedder = SentenceTransformerEmbedder(model_name=settings.sentence_transformer_model)

    if cache_queries and settings.query_embed_cache_size > 0:
        return CachedQueryEmbedder(embedder, settings)
    return embedder


async def aclose_embedder(embedder: EmbedderProtocol) -> None:
    target = embedder.inner if isinstance(embedder, CachedQueryEmbedder) else embedder
    if isinstance(target, OllamaEmbedder | OpenAIEmbedder):
        await target.aclose()
