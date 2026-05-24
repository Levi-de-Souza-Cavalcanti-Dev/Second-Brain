from __future__ import annotations

import pytest

from secondbrain.config import Settings
from secondbrain.ingestion.indexer import index_vault
from secondbrain.models import SearchRequest
from secondbrain.retrieval.lexical_base import NoopLexicalRetriever
from secondbrain.retrieval.retriever import semantic_search_service
from tests.helpers import configure_basic_env, write_sample_note


@pytest.mark.asyncio
async def test_index_twice_then_search(tmp_path, monkeypatch_fake_embedder: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch_fake_embedder
    vault = tmp_path / "vault"
    vault.mkdir()

    configure_basic_env(monkeypatch, vault, tmp_path / "vector-store")

    write_sample_note(vault)

    settings = Settings()
    summary_first = await index_vault(settings)
    summary_second = await index_vault(settings)

    assert summary_first.skipped_unchanged == 0
    assert summary_second.skipped_unchanged >= 1
    hits = await semantic_search_service(settings, SearchRequest(query="Lista final", top_k=5))
    assert hits
    assert any(str(h.metadata.get("source_path", "")).endswith("notes/sample.md") for h in hits)


@pytest.mark.asyncio
async def test_noop_lexical_empty() -> None:
    lexical = NoopLexicalRetriever()
    assert await lexical.lexical_search("qualquer coisa", 5) == []
