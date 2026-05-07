from __future__ import annotations

import httpx

from secondbrain.llm.base import ChatError, ChatCompletionClient


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
