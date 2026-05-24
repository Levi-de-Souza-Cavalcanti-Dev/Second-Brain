from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import frontmatter

from secondbrain.models import ParsedMarkdown

_WIKILINK_RE = re.compile(
    r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]",
)
INLINE_TAG_RE = re.compile(r"(?<![#\w])(#)([^\s#\[\]`]+)")  # #tag excluding ## headers

FIRST_H1_RE = re.compile(r"^\s{0,3}#\s+(.+?)\s*$", re.MULTILINE)


def _extract_inline_tags(markdown_body: str) -> list[str]:
    tags: list[str] = []
    for line in markdown_body.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s", line):
            continue
        for m in INLINE_TAG_RE.finditer(line):
            tag = (m.group(2) or "").strip()
            if tag:
                tags.append(tag.strip("/ "))
    return list(dict.fromkeys(tags))


def _normalize_tags(raw: Any) -> tuple[str, ...]:
    """Frontmatter tags: list, comma string, or single string."""

    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = [p.strip() for p in re.split(r"[,\s]+", raw.strip()) if p.strip()]
        return tuple(dict.fromkeys(parts))
    if isinstance(raw, list | tuple):
        out: list[str] = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return tuple(dict.fromkeys(out))
    return (str(raw).strip(),) if str(raw).strip() else ()


def _extract_wikilinks(text: str) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for m in _WIKILINK_RE.finditer(text):
        target = (m.group(1) or "").strip()
        if target:
            seen.setdefault(target, None)
    return tuple(seen.keys())


def _first_h1_title(body: str) -> str | None:
    m = FIRST_H1_RE.search(body)
    return m.group(1).strip() if m else None


def parse_markdown_file(path: Path, raw_text: str) -> ParsedMarkdown:
    post = frontmatter.loads(raw_text)
    front = dict(post.metadata) if post.metadata else {}
    body = post.content or ""

    fm_tags = _normalize_tags(front.get("tags"))
    inline_tags = tuple(_extract_inline_tags(body))
    tags = tuple(dict.fromkeys([*fm_tags, *inline_tags]))

    wikilinks = _extract_wikilinks(body)
    if front:
        for _k, v in front.items():
            if isinstance(v, str):
                wikilinks = tuple(dict.fromkeys([*wikilinks, *_extract_wikilinks(v)]))

    title = ""
    if isinstance(front.get("title"), str) and front["title"].strip():
        title = front["title"].strip()
    else:
        h1 = _first_h1_title(body)
        title = h1 if h1 else path.stem

    return ParsedMarkdown(
        title=title,
        body_markdown=body,
        tags=tags,
        wikilinks=wikilinks,
        frontmatter_raw=front,
    )


def stable_chunk_id(source_path: str, heading_path: str, ordinal: int) -> str:
    """Stable ids for chunk upserts (deterministic SHA-256 hex)."""

    payload = f"{source_path}\0{heading_path}\0{ordinal}".encode()
    return hashlib.sha256(payload).hexdigest()
