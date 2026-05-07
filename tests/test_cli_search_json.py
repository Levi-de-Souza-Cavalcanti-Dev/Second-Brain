"""CLI search --json usa Rich para JSON legível."""

from __future__ import annotations

from typer.testing import CliRunner

from secondbrain.models import SearchHit


def test_search_json_flag(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    async def fake_semantic(_s, _body):
        return [
            SearchHit(
                text="trecho único da nota",
                score=0.75,
                metadata={"source_path": "x.md"},
            )
        ]

    monkeypatch.setattr("secondbrain.cli.main.semantic_search_service", fake_semantic)

    from secondbrain.cli.main import app_cli

    runner = CliRunner()
    result = runner.invoke(app_cli, ["search", "q", "--json", "--top-k", "3"])

    assert result.exit_code == 0
    out = result.stdout
    assert "hits" in out
    assert "trecho" in out
    assert "x.md" in out
