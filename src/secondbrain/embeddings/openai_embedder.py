from __future__ import annotations

import httpx
import structlog

from secondbrain.embeddings.base import EmbedderProtocol, EmbeddingError

_LOG = structlog.get_logger()


class OpenAIEmbedder(EmbedderProtocol):
    """OpenAI-compatible embeddings API (/v1/embeddings)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/v1/embeddings"
        payload: dict[str, object] = {"model": self.model, "input": texts}
        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.ConnectError as e:
            raise EmbeddingError(
                f"Não foi possível conectar ao endpoint de embeddings em {self.base_url}.",
            ) from e
        except httpx.TimeoutException as e:
            raise EmbeddingError("Timeout ao gerar embeddings (OpenAI-compat).") from e
        except httpx.HTTPStatusError as e:
            msg = getattr(e.response, "text", "") or ""
            raise EmbeddingError(
                f"Falha HTTP na API de embeddings ({url}): {e.response.status_code} {msg}",
            ) from e

        data = body.get("data")
        if not isinstance(data, list) or not data:
            raise EmbeddingError("Resposta inválida da API de embeddings (sem data).")

        ordered: list[list[float] | None] = [None] * len(texts)
        for item in data:
            if not isinstance(item, dict):
                continue
            idx = item.get("index", 0)
            emb = item.get("embedding")
            if isinstance(idx, int) and isinstance(emb, list):
                ordered[idx] = [float(x) for x in emb]

        if any(v is None for v in ordered):
            _LOG.warning("openai.embedding.index_mismatch", expected=len(texts), got=len(data))
            raise EmbeddingError("Embeddings incompletos na resposta OpenAI-compat.")
        return [v for v in ordered if v is not None]
