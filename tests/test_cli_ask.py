"""CLI ask chama o pipeline RAG (mock)."""

from __future__ import annotations

from typer.testing import CliRunner

from secondbrain.models import AskResponse, SourceCitation


def test_ask_command_human_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    async def fake_answer(_s, req):
        assert req.query == "qual é a capital?"
        assert req.top_k == 8
        return AskResponse(
            answer="Resposta curta.",
            sources=[
                SourceCitation(path="a.md", heading_path="Intro"),
            ],
        )

    monkeypatch.setattr("secondbrain.cli.main.answer_question", fake_answer)

    from secondbrain.cli.main import app_cli

    runner = CliRunner()
    result = runner.invoke(app_cli, ["ask", "qual", "é", "a", "capital?"])

    assert result.exit_code == 0
    assert "Resposta curta" in result.stdout
    assert "a.md" in result.stdout
    assert "Intro" in result.stdout


def test_ask_command_json_output(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    async def fake_answer(_s, req):
        return AskResponse(answer="x", sources=[])

    monkeypatch.setattr("secondbrain.cli.main.answer_question", fake_answer)

    from secondbrain.cli.main import app_cli

    runner = CliRunner()
    result = runner.invoke(app_cli, ["ask", "hi", "--json"])

    assert result.exit_code == 0
    assert '"answer"' in result.stdout
    assert "x" in result.stdout
