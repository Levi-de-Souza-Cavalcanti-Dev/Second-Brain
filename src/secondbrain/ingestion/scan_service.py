"""Vault scan service for incremental indexing."""

from __future__ import annotations

import asyncio
import pathlib

from secondbrain.chunking.splitter import filepath_to_posix_source
from secondbrain.config import Settings
from secondbrain.ingestion.hashing import compute_file_content_hash_from_path
from secondbrain.ingestion.manifest import (
    file_stat_meta,
    load_manifest,
    load_manifest_meta,
    manifest_meta_path,
    manifest_path,
)
from secondbrain.ingestion.types import VaultChangeReport, embed_model_fingerprint
from secondbrain.ingestion.vault_scanner import vault_markdown_paths


class ScanService:
    async def scan_vault_changes(self, settings: Settings) -> VaultChangeReport:
        vr = pathlib.Path(settings.obsidian_vault_path).expanduser().resolve()
        vs = pathlib.Path(settings.vectorstore_path).expanduser().resolve()
        mf = manifest_path(vs)
        manifest = load_manifest(mf)
        meta_path = manifest_meta_path(vs)
        stat_meta = load_manifest_meta(meta_path)

        current_model = embed_model_fingerprint(settings)
        force_all = False
        if manifest.embed_model and manifest.embed_model != current_model:
            force_all = True
        if manifest.version < 1:
            force_all = True

        files = vault_markdown_paths(vr, settings.ignore_globs)
        current_posix: set[str] = {filepath_to_posix_source(p, vr) for p in files}
        removed = sorted(posix for posix in manifest.entries if posix not in current_posix)

        if force_all:
            return VaultChangeReport(
                files_to_index=list(files),
                removed_posix_paths=removed,
                current_posix_paths=current_posix,
                force_reindex_all=True,
            )

        concurrency = max(1, settings.embedding_batch_concurrency)
        sem = asyncio.Semaphore(concurrency)
        to_index: list[pathlib.Path] = []

        async def needs_reindex(md_path: pathlib.Path) -> bool:
            posix = filepath_to_posix_source(md_path, vr)
            async with sem:
                st = await asyncio.to_thread(file_stat_meta, md_path)
                prev = stat_meta.get(posix)
                if (
                    prev is not None
                    and prev.mtime_ns == st.mtime_ns
                    and prev.size == st.size
                    and manifest.entries.get(posix) is not None
                ):
                    return False
                content_hash = await asyncio.to_thread(compute_file_content_hash_from_path, md_path)
            return manifest.entries.get(posix) != content_hash

        checks = await asyncio.gather(*(needs_reindex(p) for p in files))
        for md_path, changed in zip(files, checks, strict=True):
            if changed:
                to_index.append(md_path)

        return VaultChangeReport(
            files_to_index=to_index,
            removed_posix_paths=removed,
            current_posix_paths=current_posix,
            force_reindex_all=False,
        )
