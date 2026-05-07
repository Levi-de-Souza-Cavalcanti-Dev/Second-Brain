"""Normalize vault file content for stable hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path


def normalize_text_for_hash(raw: str) -> bytes:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    canonical = ("\n".join(lines).strip() + ("\n" if raw else "")).encode("utf-8")
    return canonical


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_content_hash_utf8(raw_text: str) -> str:
    """Whole-file fingerprint (YAML frontmatter incluído).

    Preferimos hashing do arquivo completo Unicode normalizado a mtime/size
    para refletir qualquer edição também no frontmatter.
    """

    return hash_bytes(normalize_text_for_hash(raw_text))


def compute_file_content_hash_from_path(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return compute_file_content_hash_utf8(raw)
