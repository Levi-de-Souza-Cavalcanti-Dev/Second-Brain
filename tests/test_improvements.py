"""Tests for config validation, BM25, MMR, diagnostics, and new CLI commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from secondbrain.cli.main import app_cli
from secondbrain.config import Settings
from secondbrain.models import SearchHit
from secondbrain.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from secondbrain.retrieval.mmr import mmr_diversify


def test_settings_validates_vault_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    vs = tmp_path / "vs"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("VECTORSTORE_PATH", str(vs))
    s = Settings()
    assert s.obsidian_vault_path == str(vault.resolve())


def test_settings_rejects_missing_vault(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("VECTORSTORE_PATH", str(tmp_path / "vs"))
    with pytest.raises(ValueError, match="OBSIDIAN_VAULT_PATH"):
        Settings()


def test_rrf_merges_rankings() -> None:
    dense = [
        SearchHit(text="a", score=0.9, metadata={"source_path": "a.md", "heading_path": ""}),
        SearchHit(text="b", score=0.8, metadata={"source_path": "b.md", "heading_path": ""}),
    ]
    lexical = [
        SearchHit(text="b", score=1.0, metadata={"source_path": "b.md", "heading_path": ""}),
        SearchHit(text="c", score=0.7, metadata={"source_path": "c.md", "heading_path": ""}),
    ]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    assert len(fused) == 3
    assert fused[0].metadata["source_path"] == "b.md"


def test_mmr_diversify() -> None:
    hits = [
        SearchHit(text="same topic one", score=0.9, metadata={}),
        SearchHit(text="same topic two", score=0.85, metadata={}),
        SearchHit(text="different topic", score=0.7, metadata={}),
    ]
    qv = [1.0] * 4
    vecs = {
        "same topic one": [1.0, 0.0, 0.0, 0.0],
        "same topic two": [0.99, 0.01, 0.0, 0.0],
        "different topic": [0.0, 1.0, 0.0, 0.0],
    }
    out = mmr_diversify(hits, qv, vecs, lambda_mult=0.5, top_k=2)
    assert len(out) == 2


def test_bm25_index_upsert_and_delete() -> None:
    from secondbrain.models import DocumentChunk

    idx = BM25Index()
    chunk = DocumentChunk(
        chunk_id="c1",
        source_path="notes/a.md",
        heading_path="",
        text="neural networks are cool",
        tags=(),
        wikilinks=(),
        file_hash="h",
    )
    idx.upsert_chunks([chunk])
    assert idx.bm25 is not None
    idx.delete_by_source_path("notes/a.md")
    assert idx.chunk_ids == []


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app_cli, ["version"])
    assert result.exit_code == 0


def test_logging_redacts_secrets() -> None:
    from secondbrain.logging_config import _redact_secrets

    out = _redact_secrets(None, "", {"api_key": "secret123", "message": "ok"})
    assert out["api_key"] == "***REDACTED***"
    assert out["message"] == "ok"
