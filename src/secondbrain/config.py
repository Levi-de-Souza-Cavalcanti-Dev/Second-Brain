from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    obsidian_vault_path: str = Field(
        ...,
        alias="OBSIDIAN_VAULT_PATH",
        description="Absolute path to the Obsidian vault root.",
    )
    vectorstore_path: str = Field(
        default="./data/vectorstore",
        alias="VECTORSTORE_PATH",
    )

    embedding_provider: Literal["ollama", "sentence_transformers"] = Field(
        default="ollama",
        alias="EMBEDDING_PROVIDER",
    )
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        alias="OLLAMA_EMBED_MODEL",
    )
    sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="SENTENCE_TRANSFORMER_MODEL",
    )

    chat_provider: Literal["ollama", "openai_compat"] = Field(
        default="ollama",
        alias="CHAT_PROVIDER",
    )
    ollama_chat_model: str = Field(default="llama3.2", alias="OLLAMA_CHAT_MODEL")
    ollama_embed_timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        le=3600.0,
        alias="OLLAMA_EMBED_TIMEOUT_SECONDS",
        description="Timeout HTTP para /api/embed (CPU costuma precisar de vários segundos).",
    )
    ollama_chat_timeout_seconds: float = Field(
        default=900.0,
        ge=10.0,
        le=7200.0,
        alias="OLLAMA_CHAT_TIMEOUT_SECONDS",
        description="Timeout HTTP para /api/chat — primeira chamada pode carregar o modelo (CPU: minutos).",
    )

    openai_compat_base_url: str = Field(default="", alias="OPENAI_COMPAT_BASE_URL")
    openai_compat_api_key: str = Field(default="", alias="OPENAI_COMPAT_API_KEY")
    openai_compat_model: str = Field(default="", alias="OPENAI_COMPAT_MODEL")
    openai_compat_timeout_seconds: float = Field(
        default=600.0,
        ge=10.0,
        le=7200.0,
        alias="OPENAI_COMPAT_TIMEOUT_SECONDS",
        description="Timeout HTTP para POST /v1/chat/completions.",
    )

    chunk_size_chars: int = Field(default=2000, alias="CHUNK_SIZE_CHARS")
    chunk_overlap_chars: int = Field(default=200, alias="CHUNK_OVERLAP_CHARS")

    embedding_batch_concurrency: int = Field(default=4, alias="EMBEDDING_BATCH_CONCURRENCY")
    embedding_request_batch_size: int = Field(default=16, alias="EMBEDDING_REQUEST_BATCH_SIZE")

    ignore_globs: str = Field(
        default=(
            ".obsidian/**,"
            "**/node_modules/**,"
            "**/*.excalidraw.md,"
            ".trash/**"
        ),
        alias="IGNORE_GLOBS",
        description="Comma-separated glob patterns relative to vault root.",
    )


def parse_ignore_globs(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]
