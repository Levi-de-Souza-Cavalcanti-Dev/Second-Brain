from __future__ import annotations

import httpx

import structlog

from secondbrain.embeddings.base import EmbeddingError, EmbedderProtocol

_LOG = structlog.get_logger()


class OllamaEmbedder(EmbedderProtocol):
    """Cliente da API de embeddings do Ollama.

    Usa ``POST /api/embed`` (documentado; suporta ``input`` como string ou lista).
    A rota legada ``/api/embeddings`` usa ``prompt`` e devolve formatos inconsistentes
    quando se envia ``input`` — evitamos essa combinação.
    """

    def __init__(self, *, base_url: str, model: str, timeout_seconds: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _parse_vectors(self, body: dict[str, object], *, n_texts: int) -> list[list[float]]:
        raw_batch = body.get("embeddings")
        single = body.get("embedding")

        def _rows_from_list_of_lists(rows: list) -> list[list[float]]:
            out: list[list[float]] = []
            for row in rows:
                if not isinstance(row, list) or not row:
                    raise EmbeddingError("Vetor de embedding vazio ou inválido na resposta do Ollama.")
                if not isinstance(row[0], (int | float)):
                    raise EmbeddingError(
                        "Formato aninhado de embeddings não suportado pela resposta do Ollama.",
                    )
                out.append([float(x) for x in row])
            return out

        if isinstance(raw_batch, list) and raw_batch:
            if isinstance(raw_batch[0], list):
                vecs = _rows_from_list_of_lists(raw_batch)
                if len(vecs) != n_texts:
                    raise EmbeddingError(
                        f"Ollama retornou {len(vecs)} vetores para {n_texts} textos.",
                    )
                return vecs
            if isinstance(raw_batch[0], (int | float)) and n_texts == 1:
                return [[float(x) for x in raw_batch]]  # type: ignore[arg-type]

        if isinstance(single, list) and single:
            if isinstance(single[0], (int | float)):
                if n_texts != 1:
                    raise EmbeddingError(
                        "Ollama retornou vetor único para requisição com múltiplos textos.",
                    )
                return [[float(x) for x in single]]  # type: ignore[arg-type]
            if isinstance(single[0], list):
                vecs = _rows_from_list_of_lists(single)
                if len(vecs) != n_texts:
                    raise EmbeddingError(
                        f"Ollama retornou {len(vecs)} vetores para {n_texts} textos "
                        "(chave 'embedding' aninhada).",
                    )
                return vecs

        _LOG.warning("ollama.embedding.unexpected_shape", keys=list(body.keys()))
        raise EmbeddingError("Formato inesperado de embeddings na resposta do Ollama.")

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/api/embed"
        payload: dict[str, object] = {"model": self.model, "input": texts}
        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.ConnectError as e:
            raise EmbeddingError(
                f"Não foi possível conectar ao Ollama em {self.base_url}. Serviço ativo?",
            ) from e
        except httpx.TimeoutException as e:
            raise EmbeddingError("Timeout ao gerar embeddings no Ollama.") from e
        except httpx.HTTPStatusError as e:
            msg = getattr(e.response, "text", "") or ""
            raise EmbeddingError(
                f"Falha HTTP na API de embeddings do Ollama ({url}): "
                f"{e.response.status_code} {msg}",
            ) from e
        if not isinstance(body, dict):
            raise EmbeddingError("Resposta JSON inválida do Ollama (embed).")
        return self._parse_vectors(body, n_texts=len(texts))
