from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch


def write_sample_note(vault_root: Path) -> None:
    note = vault_root / "notes" / "sample.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "---\ntitle: MetaTitulo\ntags:\n  - yaml-a\n  - yaml-b\n---\n"
        "# Topo\nUm paragrafo inicial #hash-tag dentro da linha.\n\n"
        "## BetaSecao\nConteudo com [[WikiOne]] ligacao.\n"
        "### Gamma\nLista final bem descritiva.\n",
        encoding="utf-8",
    )


def configure_basic_env(monkeypatch: MonkeyPatch, vault: Path, vectorstore: Path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("VECTORSTORE_PATH", str(vectorstore))
