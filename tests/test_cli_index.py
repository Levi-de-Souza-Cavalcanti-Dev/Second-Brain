from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from secondbrain.cli.main import app_cli
from secondbrain.ingestion.indexer import IndexSummary


def test_cli_index_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("VECTORSTORE_PATH", str(tmp_path / "vs"))

    async def fake_index(_settings: object) -> IndexSummary:
        return IndexSummary(files_total=3, skipped_unchanged=1, files_indexed=2, chunks_written=5)

    monkeypatch.setattr("secondbrain.cli.main.index_vault", fake_index)

    result = CliRunner().invoke(app_cli, ["index"])
    assert result.exit_code == 0
    assert "Indexação concluída" in result.stdout
    assert "total arquivos: 3" in result.stdout


def test_cli_index_settings_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "missing-vault"))
    monkeypatch.setenv("VECTORSTORE_PATH", str(tmp_path / "vs"))
    result = CliRunner().invoke(app_cli, ["index"])
    assert result.exit_code != 0
