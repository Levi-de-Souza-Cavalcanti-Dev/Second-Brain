from __future__ import annotations

import pytest

from secondbrain.config import Settings
from secondbrain.models import AskRequest, SearchHit
from secondbrain.rag.pipeline import EMPTY_CONTEXT_ANSWER, answer_question, build_context_with_citations


def test_build_context_dedupes_identical_text() -> None:
    hits = [
        SearchHit(text="same paragraph", score=1.0, metadata={"source_path": "a.md", "heading_path": ""}),
        SearchHit(text="same paragraph", score=0.9, metadata={"source_path": "b.md", "heading_path": ""}),
    ]
    ctx, cites = build_context_with_citations(hits, max_context_chars=10_000)
    assert ctx.count("same paragraph") == 1
    assert len(cites) == 2


def test_build_context_includes_title_in_header() -> None:
    hits = [
        SearchHit(
            text="body",
            score=1.0,
            metadata={"source_path": "n.md", "heading_path": "S", "title": "My Title"},
        ),
    ]
    ctx, cites = build_context_with_citations(hits, max_context_chars=10_000)
    assert "My Title · n.md" in ctx
    assert cites[0].title == "My Title"


@pytest.mark.asyncio
async def test_answer_question_empty_context_short_circuit(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    monkeypatch.setenv("VECTORSTORE_PATH", str(tmp_path / "vs"))
    (tmp_path / "vault").mkdir()
    (tmp_path / "vs").mkdir()

    async def fake_auto(_s: Settings) -> None:
        return None

    async def fake_retrieve(_s: Settings, *, query: str, top_k: int) -> list[SearchHit]:
        return []

    chat_called = {"n": 0}

    class FakeChat:
        async def complete(self, _messages: list[dict[str, str]]) -> str:
            chat_called["n"] += 1
            return "should not run"

    monkeypatch.setattr("secondbrain.rag.pipeline.auto_index_if_needed", fake_auto)
    monkeypatch.setattr("secondbrain.rag.pipeline.retrieve_top_hits", fake_retrieve)
    monkeypatch.setattr("secondbrain.rag.pipeline.make_chat_client", lambda _s: FakeChat())

    out = await answer_question(Settings(), AskRequest(query="algo?"))
    assert out.answer == EMPTY_CONTEXT_ANSWER
    assert out.sources == []
    assert chat_called["n"] == 0
