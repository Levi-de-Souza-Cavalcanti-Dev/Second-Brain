from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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

    embedding_provider: Literal["ollama", "sentence_transformers", "openai"] = Field(
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
    openai_embed_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBED_MODEL",
    )
    openai_embed_base_url: str = Field(default="", alias="OPENAI_EMBED_BASE_URL")

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
    chunk_size_tokens: int = Field(default=512, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=64, alias="CHUNK_OVERLAP_TOKENS")
    chunk_by_tokens: bool = Field(default=False, alias="CHUNK_BY_TOKENS")

    embedding_batch_concurrency: int = Field(default=4, alias="EMBEDDING_BATCH_CONCURRENCY")
    embedding_request_batch_size: int = Field(default=16, alias="EMBEDDING_REQUEST_BATCH_SIZE")

    hybrid_search: bool = Field(default=False, alias="HYBRID_SEARCH")
    hybrid_rrf_k: int = Field(default=60, alias="HYBRID_RRF_K")

    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        alias="RERANKER_MODEL",
    )
    reranker_top_n: int = Field(default=50, alias="RERANKER_TOP_N")

    mmr_lambda: float | None = Field(default=None, alias="MMR_LAMBDA")
    query_embed_cache_size: int = Field(default=128, alias="QUERY_EMBED_CACHE_SIZE")

    rag_link_expansion: bool = Field(default=False, alias="RAG_LINK_EXPANSION")
    rag_link_expansion_top_k: int = Field(default=3, alias="RAG_LINK_EXPANSION_TOP_K")

    otel_exporter_otlp_endpoint: str = Field(default="", alias="OTEL_EXPORTER_OTLP_ENDPOINT")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")

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

    @field_validator("obsidian_vault_path")
    @classmethod
    def validate_vault_path(cls, v: str) -> str:
        path = Path(v).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"OBSIDIAN_VAULT_PATH não é um diretório válido: {path}")
        return str(path)

    @field_validator("vectorstore_path")
    @classmethod
    def validate_vectorstore_path(cls, v: str) -> str:
        path = Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return str(path)


def parse_ignore_globs(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]
