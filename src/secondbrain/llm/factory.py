from __future__ import annotations

from secondbrain.config import Settings
from secondbrain.llm.base import ChatCompletionClient
from secondbrain.llm.ollama_chat import OllamaChatClient
from secondbrain.llm.openai_compat import OpenAICompatChatClient


def make_chat_client(settings: Settings) -> ChatCompletionClient:
    if settings.chat_provider == "ollama":
        return OllamaChatClient(
            base_url=settings.ollama_host,
            model=settings.ollama_chat_model,
            timeout_seconds=settings.ollama_chat_timeout_seconds,
        )
    if not settings.openai_compat_base_url or not settings.openai_compat_model:
        raise ValueError("OPENAI_COMPAT_BASE_URL e OPENAI_COMPAT_MODEL são obrigatórios para chat remoto.")
    return OpenAICompatChatClient(
        base_url=settings.openai_compat_base_url,
        api_key=settings.openai_compat_api_key,
        model=settings.openai_compat_model,
        timeout_seconds=settings.openai_compat_timeout_seconds,
    )


async def aclose_chat_client(client: ChatCompletionClient) -> None:
    """Fecha cliente HTTP quando disponível."""

    closer = getattr(client, "aclose", None)
    if callable(closer):
        await closer()
