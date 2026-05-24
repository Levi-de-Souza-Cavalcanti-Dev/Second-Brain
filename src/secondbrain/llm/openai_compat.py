from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from secondbrain.llm.base import ChatCompletionClient, ChatError


class OpenAICompatChatClient(ChatCompletionClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        try:
            resp = await self._client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.ConnectError as e:
            raise ChatError(f"Não foi possível conectar ao endpoint {self.base_url}.") from e
        except httpx.TimeoutException as e:
            raise ChatError("Timeout ao chamar API compatível com OpenAI.") from e
        except httpx.HTTPStatusError as e:
            msg = getattr(e.response, "text", "") or ""
            raise ChatError(
                f"Falha HTTP no chat (OpenAI-compat): {e.response.status_code} {msg}",
            ) from e
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ChatError("Resposta inválida (sem choices) da API OpenAI-compat.")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ChatError("Conteúdo vazio na resposta do modelo remoto.")
        return content.strip()

    async def complete_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if isinstance(delta, dict):
                        part = delta.get("content")
                        if isinstance(part, str) and part:
                            yield part
        except httpx.ConnectError as e:
            raise ChatError(f"Não foi possível conectar ao endpoint {self.base_url}.") from e
        except httpx.TimeoutException as e:
            raise ChatError("Timeout ao chamar API compatível com OpenAI.") from e
        except httpx.HTTPStatusError as e:
            msg = getattr(e.response, "text", "") or ""
            raise ChatError(
                f"Falha HTTP no chat (OpenAI-compat): {e.response.status_code} {msg}",
            ) from e
