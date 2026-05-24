"""Shared ingestion dataclasses and helpers."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from secondbrain.config import Settings


@dataclass(slots=True)
class IndexSummary:
    files_total: int
    skipped_unchanged: int
    files_indexed: int
    chunks_written: int


@dataclass(slots=True)
class VaultChangeReport:
    files_to_index: list[pathlib.Path]
    removed_posix_paths: list[str]
    current_posix_paths: set[str]
    force_reindex_all: bool = False

    @property
    def has_changes(self) -> bool:
        return bool(self.files_to_index or self.removed_posix_paths or self.force_reindex_all)


def embed_model_fingerprint(settings: Settings) -> str:
    if settings.embedding_provider == "ollama":
        return f"ollama:{settings.ollama_embed_model}"
    if settings.embedding_provider == "openai":
        return f"openai:{settings.openai_embed_model}"
    return f"sentence_transformers:{settings.sentence_transformer_model}"
