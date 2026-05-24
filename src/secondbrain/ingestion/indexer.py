"""Ingest markdown files → chunks → embeddings → vector store."""

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass

import structlog

from secondbrain.chunking.splitter import chunk_markdown_into_documents, filepath_to_posix_source
from secondbrain.config import Settings
from secondbrain.ingestion.hashing import compute_file_content_hash_from_path
from secondbrain.ingestion.manifest import (
    VaultManifest,
    file_stat_meta,
    load_manifest,
    load_manifest_meta,
    manifest_meta_path,
    manifest_path,
    save_manifest,
    save_manifest_meta,
)

# Re-exported for tests and callers using flat manifest helpers.
__all__ = [
    "IndexSummary",
    "VaultChangeReport",
    "VaultManifest",
    "auto_index_if_needed",
    "embed_model_fingerprint",
    "index_vault",
    "load_manifest",
    "save_manifest",
    "scan_vault_changes",
    "vault_has_changes",
]
from secondbrain.ingestion.markdown_parser import parse_markdown_file
from secondbrain.ingestion.vault_scanner import vault_markdown_paths
from secondbrain.vectorstore.factory import VectorStoreDeps, build_vector_store

_LOG = structlog.get_logger()


@dataclass(slots=True)
class IndexSummary:
    files_total: int
    skipped_unchanged: int
    files_indexed: int
    chunks_written: int


@dataclass(slots=True)
class VaultChangeReport:
    """Result of a single vault scan (used by auto-index and index_vault)."""

    files_to_index: list[pathlib.Path]
    removed_posix_paths: list[str]
    current_posix_paths: set[str]
    force_reindex_all: bool = False

    @property
    def has_changes(self) -> bool:
        return bool(self.files_to_index or self.removed_posix_paths or self.force_reindex_all)


def embed_model_fingerprint(settings: Settings) -> str:
    if settings.embedding_provider == "ollama":
        return f"ollama:{settings.ollama_embed_model}"
    return f"sentence_transformers:{settings.sentence_transformer_model}"


async def _probe_embed_dim(embedder: object) -> int:
    embed_many = getattr(embedder, "embed_many", None)
    if embed_many is None:
        raise TypeError("embedder must provide embed_many")
    vecs = await embed_many([" "])
    return len(vecs[0])


async def scan_vault_changes(settings: Settings) -> VaultChangeReport:
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
            if prev is not None and prev.mtime_ns == st.mtime_ns and prev.size == st.size:
                if manifest.entries.get(posix) is not None:
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


def vault_has_changes(settings: Settings) -> bool:
    """Fast filesystem check to decide if auto-index is necessary."""

    return asyncio.run(scan_vault_changes(settings)).has_changes


async def index_vault(
    settings: Settings,
    *,
    change_report: VaultChangeReport | None = None,
) -> IndexSummary:
    from secondbrain.embeddings.factory import aclose_embedder, make_embedder  # noqa: PLC0415

    vr = pathlib.Path(settings.obsidian_vault_path).expanduser().resolve()
    vs = pathlib.Path(settings.vectorstore_path).expanduser().resolve()
    vs.mkdir(parents=True, exist_ok=True)

    mf = manifest_path(vs)
    meta_path = manifest_meta_path(vs)
    manifest = load_manifest(mf)
    stat_meta = load_manifest_meta(meta_path)

    report = change_report if change_report is not None else await scan_vault_changes(settings)

    embedder = make_embedder(settings)
    try:
        current_model = embed_model_fingerprint(settings)
        if manifest.embed_model != current_model or manifest.embed_dim <= 0:
            dim = await _probe_embed_dim(embedder)
        else:
            dim = manifest.embed_dim

        manifest.embed_model = current_model
        manifest.embed_dim = dim
        manifest.version = 1

        deps = VectorStoreDeps(dimension=dim, collection_name="secondbrain_notes")
        store = await build_vector_store(str(vs), deps)

        files = vault_markdown_paths(vr, settings.ignore_globs)
        current_paths = report.current_posix_paths or {filepath_to_posix_source(p, vr) for p in files}
        paths_to_index = (
            set(files)
            if report.force_reindex_all
            else {p for p in report.files_to_index}
        )
        skipped = 0
        indexed = 0
        chunks_total = 0

        sem = asyncio.Semaphore(max(1, settings.embedding_batch_concurrency))
        batch_sz = max(1, settings.embedding_request_batch_size)

        async def embed_batches(texts: list[str]) -> list[list[float]]:
            async def one_batch(batch: list[str]) -> list[list[float]]:
                async with sem:
                    return await embedder.embed_many(batch)

            batches = [texts[i : i + batch_sz] for i in range(0, len(texts), batch_sz)]
            if not batches:
                return []
            chunks_embedded = await asyncio.gather(*(one_batch(b) for b in batches))
            out: list[list[float]] = []
            for chunk in chunks_embedded:
                out.extend(chunk)
            return out

        try:
            for removed_path in report.removed_posix_paths:
                await store.delete_by_source_path(removed_path)
                manifest.entries.pop(removed_path, None)
                stat_meta.pop(removed_path, None)

            for md_path in files:
                posix = filepath_to_posix_source(md_path, vr)
                st = await asyncio.to_thread(file_stat_meta, md_path)

                if md_path not in paths_to_index:
                    skipped += 1
                    stat_meta[posix] = st
                    continue

                h = await asyncio.to_thread(compute_file_content_hash_from_path, md_path)
                raw_text = md_path.read_text(encoding="utf-8", errors="replace")
                parsed = parse_markdown_file(md_path, raw_text)

                chunks = chunk_markdown_into_documents(
                    posix,
                    vault_relative_body=parsed.body_markdown,
                    file_hash=h,
                    tags=parsed.tags,
                    wikilinks=parsed.wikilinks,
                    title=parsed.title,
                    chunk_size_chars=settings.chunk_size_chars,
                    chunk_overlap_chars=settings.chunk_overlap_chars,
                )

                await store.delete_by_source_path(posix)

                indexed += 1
                chunks_total += len(chunks)

                if chunks:
                    embeddings = await embed_batches([c.text for c in chunks])
                    await store.upsert_documents(chunks, embeddings)

                manifest.entries[posix] = h
                stat_meta[posix] = st

            for orphan in [p for p in list(manifest.entries) if p not in current_paths]:
                await store.delete_by_source_path(orphan)
                manifest.entries.pop(orphan, None)
                stat_meta.pop(orphan, None)

            save_manifest(mf, manifest)
            save_manifest_meta(meta_path, stat_meta)

            _LOG.info(
                "index.complete",
                files_total=len(files),
                skipped_unchanged=skipped,
                files_indexed=indexed,
                chunks_written=chunks_total,
            )

            return IndexSummary(
                files_total=len(files),
                skipped_unchanged=skipped,
                files_indexed=indexed,
                chunks_written=chunks_total,
            )
        finally:
            await store.close()
    finally:
        await aclose_embedder(embedder)


async def auto_index_if_needed(settings: Settings) -> IndexSummary | None:
    """Run incremental indexing only when vault content changed."""

    report = await scan_vault_changes(settings)
    if not report.has_changes:
        _LOG.info("index.auto.skipped", reason="vault_unchanged")
        return None

    _LOG.info("index.auto.triggered", reason="vault_changed")
    return await index_vault(settings, change_report=report)
