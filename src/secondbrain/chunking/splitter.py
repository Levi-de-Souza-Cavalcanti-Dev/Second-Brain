"""Split Markdown body into overlapping chunks anchored on headings."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from secondbrain.chunking.tokenizer import TokenCounter, make_token_counter
from secondbrain.ingestion.markdown_parser import stable_chunk_id
from secondbrain.models import DocumentChunk

_HEADING_LINE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


def _split_large_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        return [text] if text else []
    trimmed = text.strip()
    if not trimmed:
        return []
    if len(trimmed) <= chunk_size:
        return [trimmed]
    parts: list[str] = []
    start = 0
    overlap = max(0, overlap)
    while start < len(trimmed):
        end = min(len(trimmed), start + chunk_size)
        chunk = trimmed[start:end].strip()
        if chunk:
            parts.append(chunk)
        if end >= len(trimmed):
            break
        start = max(0, end - overlap)
    return parts


def _split_large_text_tokens(
    text: str,
    counter: TokenCounter,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    tokens = counter.encode(text.strip())
    if not tokens:
        return []
    if len(tokens) <= chunk_size_tokens:
        return [text.strip()]
    parts: list[str] = []
    start = 0
    overlap = max(0, overlap_tokens)
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size_tokens)
        chunk_tokens = tokens[start:end]
        decoded = counter.decode(chunk_tokens).strip()
        if decoded:
            parts.append(decoded)
        if end >= len(tokens):
            break
        start = max(0, end - overlap)
    return parts


def _line_range_for_text(full_lines: list[str], chunk_text: str) -> tuple[int, int]:
    """Best-effort line range for a chunk within section lines."""
    joined = "\n".join(full_lines)
    idx = joined.find(chunk_text[: min(80, len(chunk_text))])
    if idx < 0:
        return (1, max(1, len(full_lines)))
    prefix = joined[:idx]
    line_start = prefix.count("\n") + 1
    line_end = line_start + chunk_text.count("\n")
    return (line_start, max(line_start, line_end))


def _heading_path(stack: Iterable[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def chunk_markdown_into_documents(
    source_path_posix: str,
    *,
    vault_relative_body: str,
    file_hash: str,
    tags: tuple[str, ...],
    wikilinks: tuple[str, ...],
    title: str = "",
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    chunk_by_tokens: bool = False,
    token_model_hint: str = "",
) -> list[DocumentChunk]:
    """Produce DocumentChunk records with deterministic chunk_id ordering."""

    lines = vault_relative_body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    counter = make_token_counter(by_tokens=chunk_by_tokens, model_hint=token_model_hint)

    sections: list[tuple[str, int, list[str]]] = []
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
            heading_title = hm.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_title))
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

        if chunk_by_tokens:
            text_parts = _split_large_text_tokens(
                joined,
                counter,
                chunk_size_tokens,
                chunk_overlap_tokens,
            )
        else:
            text_parts = _split_large_text(joined, chunk_size_chars, chunk_overlap_chars)

        for sub_i, chunk_text in enumerate(text_parts):
            line_start, line_end = _line_range_for_text(seg_lines, chunk_text)
            cid = stable_chunk_id(source_path_posix, heading_path or "__root__", ordinal)
            meta: dict[str, object] = {
                "title": title,
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
                    line_start=line_start,
                    line_end=line_end,
                    extra_metadata=meta,
                ),
            )
            ordinal += 1

    return chunks


def filepath_to_posix_source(path_abs: Path, vault_root: Path) -> str:
    return path_abs.resolve().relative_to(vault_root.resolve()).as_posix()
