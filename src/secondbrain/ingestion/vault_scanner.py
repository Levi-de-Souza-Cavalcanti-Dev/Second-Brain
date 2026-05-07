from __future__ import annotations

from pathlib import Path

import pathspec

from secondbrain.config import parse_ignore_globs


def vault_markdown_paths(vault_root: Path, ignore_globs_str: str) -> list[Path]:
    vault_root = vault_root.expanduser().resolve()
    patterns = parse_ignore_globs(ignore_globs_str)
    spec = pathspec.PathSpec.from_lines("gitignore", patterns) if patterns else None
    md_files: list[Path] = sorted(vault_root.rglob("*.md"))
    out: list[Path] = []
    for path in md_files:
        try:
            rel = path.relative_to(vault_root)
        except ValueError:
            continue
        posix = rel.as_posix()
        if spec is not None and spec.match_file(posix):
            continue
        out.append(path)
    return out


def posix_relative_path(path: Path, vault_root: Path) -> str:
    return path.resolve().relative_to(vault_root.resolve()).as_posix()
