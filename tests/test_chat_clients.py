from __future__ import annotations

import httpx
import pytest
import respx

from secondbrain.llm.base import ChatError
from secondbrain.llm.ollama_chat import OllamaChatClient
from secondbrain.llm.openai_compat import OpenAICompatChatClient


@pytest.mark.asyncio
@respx.mock
async def test_ollama_chat_success() -> None:
    respx.post("http://127.0.0.1:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": " Olá "}}),
    )
    client = OllamaChatClient(base_url="http://127.0.0.1:11434", model="llama3.2", timeout_seconds=5.0)
    try:
        out = await client.complete([{"role": "user", "content": "oi"}])
        assert out == "Olá"
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_ollama_chat_http_error() -> None:
    respx.post("http://127.0.0.1:11434/api/chat").mock(return_value=httpx.Response(500, text="boom"))
    client = OllamaChatClient(base_url="http://127.0.0.1:11434", model="llama3.2", timeout_seconds=5.0)
    try:
        with pytest.raises(ChatError, match="Falha HTTP"):
            await client.complete([{"role": "user", "content": "oi"}])
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_success_no_header_leak(caplog: pytest.LogCaptureFixture) -> None:
    route = respx.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Resposta"}}]},
        ),
    )
    client = OpenAICompatChatClient(
        base_url="https://api.example.com",
        api_key="secret-token-xyz",
        model="gpt-test",
        timeout_seconds=5.0,
    )
    try:
        out = await client.complete([{"role": "user", "content": "q"}])
        assert out == "Resposta"
        assert route.called
        sent = route.calls[0].request
        assert sent.headers.get("Authorization") == "Bearer secret-token-xyz"
        assert "secret-token-xyz" not in caplog.text
    finally:
        await client.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_openai_compat_timeout() -> None:
    respx.post("https://api.example.com/v1/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    client = OpenAICompatChatClient(
        base_url="https://api.example.com",
        api_key="k",
        model="m",
        timeout_seconds=0.1,
    )
    try:
        with pytest.raises(ChatError, match="Timeout"):
            await client.complete([{"role": "user", "content": "q"}])
    finally:
        await client.aclose()
