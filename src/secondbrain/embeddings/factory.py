from __future__ import annotations

from secondbrain.config import Settings
from secondbrain.embeddings.base import EmbedderProtocol
from secondbrain.embeddings.ollama_embedder import OllamaEmbedder
from secondbrain.embeddings.sentence_transformers_embedder import SentenceTransformerEmbedder


def make_embedder(settings: Settings) -> EmbedderProtocol:
    if settings.embedding_provider == "ollama":
        return OllamaEmbedder(
            base_url=settings.ollama_host,
            model=settings.ollama_embed_model,
            timeout_seconds=settings.ollama_embed_timeout_seconds,
        )
    return SentenceTransformerEmbedder(model_name=settings.sentence_transformer_model)


async def aclose_embedder(embedder: EmbedderProtocol) -> None:
    if isinstance(embedder, OllamaEmbedder):
        await embedder.aclose()
