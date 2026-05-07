from __future__ import annotations

from pathlib import Path

from secondbrain.chunking.splitter import chunk_markdown_into_documents
from secondbrain.ingestion.hashing import compute_file_content_hash_from_path


def test_chunk_heading_metadata(tmp_path: Path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(
        "Preâmbulo curto antes do primeiro heading para validar breadcrumbs vazios.\n\n"
        "# Root\n"
        + ("Primeiro corpo bem longo suficiente para virar segundo chunk.\n" * 3)
        + "## Subido\nTrecho menor.\n",
        encoding="utf-8",
    )
    body = md_path.read_text(encoding="utf-8")
    h = compute_file_content_hash_from_path(md_path)

    chunks = chunk_markdown_into_documents(
        "doc.md",
        vault_relative_body=body,
        file_hash=h,
        tags=("t1",),
        wikilinks=("x",),
        chunk_size_chars=80,
        chunk_overlap_chars=10,
    )

    assert chunks
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    headings = [c.heading_path for c in chunks]
    assert any(hp == "" for hp in headings)  # preamble
    assert any("Root" in hp for hp in headings)
    assert any("Subido" in hp for hp in headings)


def test_empty_body_returns_empty() -> None:
    chunks = chunk_markdown_into_documents(
        "empty.md",
        vault_relative_body="",
        file_hash="abc",
        tags=(),
        wikilinks=(),
        chunk_size_chars=200,
        chunk_overlap_chars=50,
    )
    assert chunks == []
