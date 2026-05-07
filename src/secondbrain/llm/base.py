from __future__ import annotations

from typing import Protocol


class ChatError(RuntimeError):
    """Erro ao chamar o provedor de chat."""


class ChatCompletionClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
    ) -> str: ...
