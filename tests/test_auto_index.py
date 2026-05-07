from __future__ import annotations

from pathlib import Path

import pytest

from secondbrain.config import Settings
from secondbrain.ingestion.hashing import compute_file_content_hash_from_path
from secondbrain.ingestion.indexer import IndexSummary, auto_index_if_needed, save_manifest, vault_has_changes
from secondbrain.models import AskRequest, SearchHit
from secondbrain.rag.pipeline import answer_question


def _settings(vault_path: Path, vectorstore_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault_path))
    monkeypatch.setenv("VECTORSTORE_PATH", str(vectorstore_path))
    return Settings()


def test_vault_has_changes_false_when_manifest_matches(monkeypatch, tmp_path) -> None:
    vault = tmp_path / "vault"
    vectorstore = tmp_path / "vectorstore"
    vault.mkdir(parents=True)
    vectorstore.mkdir(parents=True)

    note = vault / "ai.md"
    note.write_text("# IA\n\nRedes neurais.\n", encoding="utf-8")

    settings = _settings(vault, vectorstore, monkeypatch)
    rel = "ai.md"
    save_manifest(vectorstore / "manifest.json", {rel: compute_file_content_hash_from_path(note)})

    assert vault_has_changes(settings) is False


def test_vault_has_changes_true_when_file_removed(monkeypatch, tmp_path) -> None:
    vault = tmp_path / "vault"
    vectorstore = tmp_path / "vectorstore"
    vault.mkdir(parents=True)
    vectorstore.mkdir(parents=True)

    settings = _settings(vault, vectorstore, monkeypatch)
    save_manifest(vectorstore / "manifest.json", {"old.md": "abc123"})

    assert vault_has_changes(settings) is True


@pytest.mark.asyncio
async def test_auto_index_if_needed_calls_index_only_on_changes(monkeypatch, tmp_path) -> None:
    vault = tmp_path / "vault"
    vectorstore = tmp_path / "vectorstore"
    vault.mkdir(parents=True)
    vectorstore.mkdir(parents=True)
    settings = _settings(vault, vectorstore, monkeypatch)

    called: dict[str, int] = {"n": 0}

    async def fake_index(_settings: Settings) -> IndexSummary:
        called["n"] += 1
        return IndexSummary(files_total=1, skipped_unchanged=0, files_indexed=1, chunks_written=2)

    monkeypatch.setattr("secondbrain.ingestion.indexer.index_vault", fake_index)
    monkeypatch.setattr("secondbrain.ingestion.indexer.vault_has_changes", lambda _s: False)

    out = await auto_index_if_needed(settings)
    assert out is None
    assert called["n"] == 0

    monkeypatch.setattr("secondbrain.ingestion.indexer.vault_has_changes", lambda _s: True)
    out2 = await auto_index_if_needed(settings)
    assert out2 is not None
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_answer_question_runs_auto_index(monkeypatch, tmp_path) -> None:
    vault = tmp_path / "vault"
    vectorstore = tmp_path / "vectorstore"
    vault.mkdir(parents=True)
    vectorstore.mkdir(parents=True)
    settings = _settings(vault, vectorstore, monkeypatch)

    calls: dict[str, int] = {"auto": 0}

    async def fake_auto_index(_settings: Settings) -> None:
        calls["auto"] += 1
        return None

    async def fake_retrieve(_settings: Settings, *, query: str, top_k: int) -> list[SearchHit]:
        assert query == "onde falo de redes neurais?"
        assert top_k == 8
        return [
            SearchHit(
                text="Conteudo sobre redes neurais.",
                score=0.99,
                metadata={"source_path": "IA.md", "heading_path": "ML/Redes"},
            ),
        ]

    class FakeChat:
        async def complete(self, _messages: list[dict[str, str]]) -> str:
            return "Voce fala em IA.md."

    async def fake_close_chat(_chat: FakeChat) -> None:
        return None

    monkeypatch.setattr("secondbrain.rag.pipeline.auto_index_if_needed", fake_auto_index)
    monkeypatch.setattr("secondbrain.rag.pipeline.retrieve_top_hits", fake_retrieve)
    monkeypatch.setattr("secondbrain.rag.pipeline.make_chat_client", lambda _s: FakeChat())
    monkeypatch.setattr("secondbrain.rag.pipeline.aclose_chat_client", fake_close_chat)

    out = await answer_question(settings, AskRequest(query="onde falo de redes neurais?"))
    assert calls["auto"] == 1
    assert out.answer == "Voce fala em IA.md."
    assert out.sources[0].path == "IA.md"
