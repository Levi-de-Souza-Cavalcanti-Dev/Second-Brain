from __future__ import annotations

import json

import pytest

from secondbrain.config import Settings
from secondbrain.ingestion.indexer import index_vault, load_manifest, scan_vault_changes
from secondbrain.ingestion.manifest import VaultManifest, manifest_path

from tests.helpers import configure_basic_env, write_sample_note


@pytest.mark.asyncio
async def test_index_change_delete_and_v0_migration(
    tmp_path,
    monkeypatch_fake_embedder: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch_fake_embedder
    vault = tmp_path / "vault"
    vault.mkdir()
    vectorstore = tmp_path / "vectorstore"
    configure_basic_env(monkeypatch, vault, vectorstore)
    write_sample_note(vault)

    settings = Settings()
    first = await index_vault(settings)
    assert first.files_indexed >= 1
    assert first.chunks_written >= 1

    manifest_file = manifest_path(vectorstore)
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data.get("version") == 1
    assert data.get("embed_dim") == 16

    second = await index_vault(settings)
    assert second.skipped_unchanged >= 1

    note = vault / "notes" / "sample.md"
    note.write_text(note.read_text(encoding="utf-8") + "\nNova linha extra.\n", encoding="utf-8")
    assert (await scan_vault_changes(settings)).has_changes is True

    third = await index_vault(settings)
    assert third.files_indexed >= 1

    note.unlink()
    assert (await scan_vault_changes(settings)).has_changes is True
    fourth = await index_vault(settings)
    assert fourth.files_indexed == 0
    manifest = load_manifest(manifest_file)
    assert "notes/sample.md" not in manifest.entries


@pytest.mark.asyncio
async def test_v0_manifest_migrates_on_save(tmp_path, monkeypatch_fake_embedder: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch_fake_embedder
    vault = tmp_path / "vault"
    vault.mkdir()
    vectorstore = tmp_path / "vectorstore"
    configure_basic_env(monkeypatch, vault, vectorstore)
    write_sample_note(vault)

    mf = manifest_path(vectorstore)
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps({"notes/sample.md": "deadbeef" * 8}), encoding="utf-8")

    settings = Settings()
    await index_vault(settings)
    data = json.loads(mf.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "entries" in data
    assert isinstance(data["entries"], dict)
