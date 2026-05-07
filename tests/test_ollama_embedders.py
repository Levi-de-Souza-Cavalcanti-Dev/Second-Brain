import httpx
import pytest
import respx

from secondbrain.embeddings.ollama_embedder import EmbeddingError, OllamaEmbedder


@pytest.mark.asyncio
async def test_ollama_embedder_batch_vectors(respx_mock: respx.Router) -> None:
    embedder = OllamaEmbedder(base_url="http://fixture:7777", model="fake")
    respx_mock.post("http://fixture:7777/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.9], [-0.2, 2.5]]}),
    )
    try:
        vectors = await embedder.embed_many(["a", "b"])
        assert vectors[1][1] == pytest.approx(2.5)
    finally:
        await embedder.aclose()


@pytest.mark.asyncio
async def test_ollama_embedder_single_wrapped_in_embeddings(respx_mock: respx.Router) -> None:
    """Um texto → resposta documentada: embeddings com uma linha."""

    embedder = OllamaEmbedder(base_url="http://fixture:7778", model="fake")
    respx_mock.post("http://fixture:7778/api/embed").mock(
        return_value=httpx.Response(
            200,
            json={"embeddings": [[0.0, 1.25]], "model": "fake"},
        ),
    )
    try:
        vectors = await embedder.embed_many(["only"])
        assert vectors == [[0.0, pytest.approx(1.25)]]
    finally:
        await embedder.aclose()


@pytest.mark.asyncio
async def test_ollama_embedder_legacy_embedding_key(respx_mock: respx.Router) -> None:
    """Compat: corpo com chave singular ``embedding`` (apis antigas / proxies)."""

    embedder = OllamaEmbedder(base_url="http://fixture:7779", model="fake")
    respx_mock.post("http://fixture:7779/api/embed").mock(
        return_value=httpx.Response(200, json={"embedding": [[0.5, 2.0]]}),
    )
    try:
        vectors = await embedder.embed_many(["only"])
        assert len(vectors) == 1
        assert vectors[0] == [pytest.approx(0.5), pytest.approx(2.0)]
    finally:
        await embedder.aclose()


@pytest.mark.asyncio
async def test_ollama_embedder_connect_error_raises(respx_mock: respx.Router) -> None:
    embedder = OllamaEmbedder(base_url="http://fixture:65432", model="fake")

    async def explode(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=_request)

    respx_mock.post("http://fixture:65432/api/embed").mock(side_effect=explode)
    try:
        with pytest.raises(EmbeddingError):
            await embedder.embed_many(["x"])
    finally:
        await embedder.aclose()
