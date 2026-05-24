from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from secondbrain.llm.base import ChatCompletionClient, ChatError


class OllamaChatClient(ChatCompletionClient):
    def __init__(self, *, base_url: str, model: str, timeout_seconds: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.ConnectError as e:
            raise ChatError(f"Não foi possível conectar ao Ollama em {self.base_url}.") from e
        except httpx.TimeoutException as e:
            raise ChatError("Timeout ao chamar o modelo de chat no Ollama.") from e
        except httpx.HTTPStatusError as e:
            msg = getattr(e.response, "text", "") or ""
            raise ChatError(
                f"Falha HTTP no chat do Ollama: {e.response.status_code} {msg}",
            ) from e
        msg = body.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ChatError("Resposta vazia do Ollama (/api/chat).")
        return content.strip()

    async def complete_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = chunk.get("message") or {}
                    if isinstance(msg, dict):
                        part = msg.get("content")
                        if isinstance(part, str) and part:
                            yield part
        except httpx.ConnectError as e:
            raise ChatError(f"Não foi possível conectar ao Ollama em {self.base_url}.") from e
        except httpx.TimeoutException as e:
            raise ChatError("Timeout ao chamar o modelo de chat no Ollama.") from e
        except httpx.HTTPStatusError as e:
            msg = getattr(e.response, "text", "") or ""
            raise ChatError(
                f"Falha HTTP no chat do Ollama: {e.response.status_code} {msg}",
            ) from e
