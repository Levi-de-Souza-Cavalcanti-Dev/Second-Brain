from __future__ import annotations

import asyncio

from secondbrain.embeddings.base import EmbeddingError, EmbedderProtocol


class SentenceTransformerEmbedder(EmbedderProtocol):
    def __init__(self, *, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "`sentence-transformers` não encontrado. Rode `pip install -r requirements.txt` (pacote já listado lá).",
            ) from e
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        def _run() -> list[list[float]]:
            vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
            return [[float(y) for y in row] for row in vecs.tolist()]

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            raise EmbeddingError(f"Falha no sentence-transformers ({self._model_name}): {e}") from e
