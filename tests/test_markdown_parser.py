from __future__ import annotations

from pathlib import Path

from secondbrain.ingestion.markdown_parser import parse_markdown_file, stable_chunk_id
from tests.helpers import write_sample_note


def test_parse_frontmatter_and_inline_signals(tmp_path: Path) -> None:
    vp = tmp_path / "vault"
    vp.mkdir()
    write_sample_note(vp)
    p = vp / "notes" / "sample.md"
    parsed = parse_markdown_file(p, p.read_text(encoding="utf-8"))
    assert parsed.title == "MetaTitulo"
    assert {"yaml-a", "yaml-b", "hash-tag"}.issubset(set(parsed.tags))
    assert "WikiOne" in parsed.wikilinks


def test_stable_chunk_id_stable() -> None:
    cid1 = stable_chunk_id("notes/x.md", "Intro > Detail", 0)
    cid2 = stable_chunk_id("notes/x.md", "Intro > Detail", 0)
    assert cid1 == cid2
