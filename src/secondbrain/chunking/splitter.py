"""Split Markdown body into overlapping chunks anchored on headings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from secondbrain.ingestion.markdown_parser import stable_chunk_id
from secondbrain.models import DocumentChunk

_HEADING_LINE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


def _split_large_text(text: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    if chunk_size_chars <= 0:
        return [text] if text else []
    trimmed = text.strip()
    if not trimmed:
        return []
    if len(trimmed) <= chunk_size_chars:
        return [trimmed]
    parts: list[str] = []
    start = 0
    overlap = max(0, overlap_chars)
    while start < len(trimmed):
        end = min(len(trimmed), start + chunk_size_chars)
        chunk = trimmed[start:end].strip()
        if chunk:
            parts.append(chunk)
        if end >= len(trimmed):
            break
        start = max(0, end - overlap)
    return parts


def _heading_path(stack: Iterable[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def chunk_markdown_into_documents(
    source_path_posix: str,
    *,
    vault_relative_body: str,
    file_hash: str,
    tags: tuple[str, ...],
    wikilinks: tuple[str, ...],
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> list[DocumentChunk]:
    """Produce DocumentChunk records with deterministic chunk_id ordering."""

    lines = vault_relative_body.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    sections: list[tuple[str, int, list[str]]] = []  # heading_path, level, segment lines raw
    stack: list[tuple[int, str]] = []
    acc: list[str] = []

    def flush() -> None:
        if acc:
            path = _heading_path(stack)
            level = stack[-1][0] if stack else 0
            sections.append((path, level, [*acc]))
            acc.clear()

    for line in lines:
        hm = _HEADING_LINE.match(line)
        if hm:
            flush()
            level = len(hm.group(1))
            title = hm.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        acc.append(line)
    flush()

    chunks: list[DocumentChunk] = []
    ordinal = 0

    if not sections and vault_relative_body.strip():
        sections = [("", 0, lines)]

    for heading_path, heading_level, seg_lines in sections:
        joined = "\n".join(seg_lines).strip()
        if not joined:
            continue
        for sub_i, chunk_text in enumerate(
            _split_large_text(joined, chunk_size_chars, chunk_overlap_chars),
        ):
            cid = stable_chunk_id(source_path_posix, heading_path or "__root__", ordinal)
            meta: dict[str, object] = {
                "title_path": heading_path,
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    source_path=source_path_posix,
                    heading_path=heading_path,
                    text=chunk_text,
                    tags=tags,
                    wikilinks=wikilinks,
                    file_hash=file_hash,
                    heading_level=heading_level,
                    chunk_index_in_section=sub_i,
                    extra_metadata=meta,
                ),
            )
            ordinal += 1

    return chunks


def filepath_to_posix_source(path_abs: Path, vault_root: Path) -> str:
    return path_abs.resolve().relative_to(vault_root.resolve()).as_posix()
