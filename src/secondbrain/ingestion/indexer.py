"""Ingest markdown files → chunks → embeddings → vector store."""

from __future__ import annotations

import asyncio
import pathlib
from typing import TYPE_CHECKING

import structlog
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from secondbrain.chunking.splitter import chunk_markdown_into_documents, filepath_to_posix_source
from secondbrain.config import Settings
from secondbrain.constants import DEFAULT_COLLECTION_NAME
from secondbrain.ingestion.embedding_pipeline import EmbeddingPipeline
from secondbrain.ingestion.hashing import compute_file_content_hash_utf8
from secondbrain.ingestion.manifest import (
    VaultManifest,
    file_stat_meta,
    load_manifest,
    save_manifest,
)
from secondbrain.ingestion.markdown_parser import parse_markdown_file
from secondbrain.ingestion.persistence_service import PersistenceService
from secondbrain.ingestion.scan_service import ScanService
from secondbrain.ingestion.types import IndexSummary, VaultChangeReport, embed_model_fingerprint
from secondbrain.ingestion.vault_scanner import vault_markdown_paths
from secondbrain.retrieval.bm25 import BM25LexicalRetriever
from secondbrain.vectorstore.factory import VectorStoreDeps, build_vector_store

if TYPE_CHECKING:
    from secondbrain.embeddings.base import EmbedderProtocol

_LOG = structlog.get_logger()

__all__ = [
    "IndexSummary",
    "VaultChangeReport",
    "VaultManifest",
    "auto_index_if_needed",
    "embed_model_fingerprint",
    "get_bm25_retriever",
    "index_vault",
    "load_manifest",
    "save_manifest",
    "scan_vault_changes",
    "vault_has_changes",
]


async def _probe_embed_dim(embedder: EmbedderProtocol) -> int:
    vecs = await embedder.embed_many([" "])
    return len(vecs[0])


async def scan_vault_changes(settings: Settings) -> VaultChangeReport:
    return await ScanService().scan_vault_changes(settings)


def vault_has_changes(settings: Settings) -> bool:
    return asyncio.run(scan_vault_changes(settings)).has_changes


def _token_model_hint(settings: Settings) -> str:
    if settings.embedding_provider == "sentence_transformers":
        return settings.sentence_transformer_model
    if settings.embedding_provider == "openai":
        return settings.openai_embed_model
    return ""


async def index_vault(
    settings: Settings,
    *,
    change_report: VaultChangeReport | None = None,
    show_progress: bool = True,
) -> IndexSummary:
    from secondbrain.embeddings.factory import aclose_embedder, make_embedder  # noqa: PLC0415

    vr = pathlib.Path(settings.obsidian_vault_path).expanduser().resolve()
    vs = pathlib.Path(settings.vectorstore_path).expanduser().resolve()
    vs.mkdir(parents=True, exist_ok=True)

    persistence = PersistenceService(vs)
    manifest = persistence.load_manifest()
    stat_meta = persistence.load_stat_meta()
    bm25_index = persistence.load_bm25()

    report = change_report if change_report is not None else await scan_vault_changes(settings)

    embedder = make_embedder(settings)
    pipeline = EmbeddingPipeline(settings, embedder)

    try:
        current_model = embed_model_fingerprint(settings)
        if manifest.embed_model != current_model or manifest.embed_dim <= 0:
            dim = await _probe_embed_dim(embedder)
        else:
            dim = manifest.embed_dim

        manifest.embed_model = current_model
        manifest.embed_dim = dim
        manifest.version = 1

        deps = VectorStoreDeps(dimension=dim, collection_name=DEFAULT_COLLECTION_NAME)
        store = await build_vector_store(str(vs), deps)

        files = vault_markdown_paths(vr, settings.ignore_globs)
        current_paths = report.current_posix_paths or {filepath_to_posix_source(p, vr) for p in files}
        paths_to_index = (
            set(files) if report.force_reindex_all else {p for p in report.files_to_index}
        )
        skipped = 0
        indexed = 0
        chunks_total = 0
        token_hint = _token_model_hint(settings)

        try:
            for removed_path in report.removed_posix_paths:
                await store.delete_by_source_path(removed_path)
                manifest.entries.pop(removed_path, None)
                stat_meta.pop(removed_path, None)
                bm25_index.delete_by_source_path(removed_path)

            files_to_process = [p for p in files if p in paths_to_index]

            async def process_file(md_path: pathlib.Path) -> tuple[pathlib.Path, int]:
                posix = filepath_to_posix_source(md_path, vr)
                st = await asyncio.to_thread(file_stat_meta, md_path)
                raw_text = await asyncio.to_thread(
                    md_path.read_text,
                    encoding="utf-8",
                    errors="replace",
                )
                h = compute_file_content_hash_utf8(raw_text)
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
                    chunk_size_tokens=settings.chunk_size_tokens,
                    chunk_overlap_tokens=settings.chunk_overlap_tokens,
                    chunk_by_tokens=settings.chunk_by_tokens,
                    token_model_hint=token_hint,
                )
                await store.delete_by_source_path(posix)
                if chunks:
                    embeddings = await pipeline.embed_batches([c.text for c in chunks])
                    await store.upsert_documents(chunks, embeddings)
                    if settings.hybrid_search:
                        bm25_index.upsert_chunks(chunks)
                manifest.entries[posix] = h
                stat_meta[posix] = st
                return md_path, len(chunks)

            if files_to_process:
                if show_progress:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                    ) as progress:
                        task = progress.add_task("A indexar ficheiros…", total=len(files_to_process))
                        for md_path in files_to_process:
                            _, n_chunks = await process_file(md_path)
                            indexed += 1
                            chunks_total += n_chunks
                            progress.advance(task)
                else:
                    results = await pipeline.map_parallel(files_to_process, process_file)
                    for _, n_chunks in results:
                        indexed += 1
                        chunks_total += n_chunks

            for md_path in files:
                if md_path not in paths_to_index:
                    skipped += 1
                    posix = filepath_to_posix_source(md_path, vr)
                    st = await asyncio.to_thread(file_stat_meta, md_path)
                    stat_meta[posix] = st

            for orphan in [p for p in list(manifest.entries) if p not in current_paths]:
                await store.delete_by_source_path(orphan)
                manifest.entries.pop(orphan, None)
                stat_meta.pop(orphan, None)
                bm25_index.delete_by_source_path(orphan)

            persistence.save_manifest(manifest)
            persistence.save_stat_meta(stat_meta)
            if settings.hybrid_search:
                persistence.save_bm25(bm25_index)

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
    report = await scan_vault_changes(settings)
    if not report.has_changes:
        _LOG.info("index.auto.skipped", reason="vault_unchanged")
        return None
    _LOG.info("index.auto.triggered", reason="vault_changed")
    return await index_vault(settings, change_report=report, show_progress=False)


def get_bm25_retriever(settings: Settings) -> BM25LexicalRetriever | None:
    if not settings.hybrid_search:
        return None
    vs = pathlib.Path(settings.vectorstore_path).expanduser().resolve()
    persistence = PersistenceService(vs)
    return BM25LexicalRetriever(persistence.load_bm25())
