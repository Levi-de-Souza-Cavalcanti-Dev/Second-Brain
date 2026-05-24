from __future__ import annotations

import asyncio
from typing import Any

from secondbrain.embeddings.base import EmbedderProtocol, EmbeddingError

_MODEL_CACHE: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "`sentence-transformers` não encontrado. Rode `pip install -r requirements.txt` (pacote já listado lá).",
            ) from e
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class SentenceTransformerEmbedder(EmbedderProtocol):
    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name
        self._model = _get_model(model_name)

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
