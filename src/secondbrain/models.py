from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Optional metadata filters for search."""

    tag: str | None = None
    path_prefix: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters | None = None


class SearchHit(BaseModel):
    text: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class AskRequest(BaseModel):
    query: str
    top_k: int = Field(default=8, ge=1, le=50)
    max_context_chars: int = Field(default=12_000, ge=500, le=100_000)
    stream: bool = False


class SourceCitation(BaseModel):
    path: str
    heading_path: str
    title: str = ""
    line_start: int | None = None
    line_end: int | None = None
    obsidian_uri: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


@dataclass(frozen=True, slots=True)
class ParsedMarkdown:
    """Result of parsing a vault markdown file."""

    title: str
    body_markdown: str
    tags: tuple[str, ...]
    wikilinks: tuple[str, ...]
    frontmatter_raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A chunk ready for embedding and vector storage."""

    chunk_id: str
    source_path: str
    heading_path: str
    text: str
    tags: tuple[str, ...]
    wikilinks: tuple[str, ...]
    file_hash: str
    heading_level: int = 0
    chunk_index_in_section: int = 0
    line_start: int = 0
    line_end: int = 0
    extra_metadata: dict[str, Any] = field(default_factory=dict)
