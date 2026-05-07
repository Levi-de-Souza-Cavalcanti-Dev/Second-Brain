"""Ingest markdown files → chunks → embeddings → vector store."""

from __future__ import annotations

import asyncio
import json
import pathlib
from dataclasses import dataclass

import structlog

from secondbrain.chunking.splitter import chunk_markdown_into_documents, filepath_to_posix_source
from secondbrain.config import Settings
from secondbrain.ingestion.hashing import compute_file_content_hash_from_path
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


def _manifest_path(vectorstore_root: pathlib.Path) -> pathlib.Path:
    return vectorstore_root / "manifest.json"


def load_manifest(path: pathlib.Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
    return out


def save_manifest(path: pathlib.Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def index_vault(settings: Settings) -> IndexSummary:
    from secondbrain.embeddings.factory import aclose_embedder, make_embedder  # noqa: PLC0415

    vr = pathlib.Path(settings.obsidian_vault_path).expanduser().resolve()
    vs = pathlib.Path(settings.vectorstore_path).expanduser().resolve()
    vs.mkdir(parents=True, exist_ok=True)

    manifest = _manifest_path(vs)
    hashes = load_manifest(manifest)

    embedder = make_embedder(settings)
    try:
        probe_dim_vec = await embedder.embed_many(["secondbrain.dimension.probe"])
        dim = len(probe_dim_vec[0])
        deps = VectorStoreDeps(dimension=dim, collection_name="secondbrain_notes")

        store = await build_vector_store(str(vs), deps)

        files = vault_markdown_paths(vr, settings.ignore_globs)
        current_paths = {filepath_to_posix_source(p, vr) for p in files}
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
            removed_paths = [posix for posix in hashes if posix not in current_paths]
            for removed_path in removed_paths:
                await store.delete_by_source_path(removed_path)
                hashes.pop(removed_path, None)

            for md_path in files:
                posix = filepath_to_posix_source(md_path, vr)
                h = compute_file_content_hash_from_path(md_path)
                if hashes.get(posix) == h:
                    skipped += 1
                    continue

                raw_text = md_path.read_text(encoding="utf-8", errors="replace")
                parsed = parse_markdown_file(md_path, raw_text)

                chunks = chunk_markdown_into_documents(
                    posix,
                    vault_relative_body=parsed.body_markdown,
                    file_hash=h,
                    tags=parsed.tags,
                    wikilinks=parsed.wikilinks,
                    chunk_size_chars=settings.chunk_size_chars,
                    chunk_overlap_chars=settings.chunk_overlap_chars,
                )

                await store.delete_by_source_path(posix)

                indexed += 1
                chunks_total += len(chunks)

                if chunks:
                    embeddings = await embed_batches([c.text for c in chunks])
                    await store.upsert_documents(chunks, embeddings)

                hashes[posix] = h

            save_manifest(manifest, hashes)

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


def vault_has_changes(settings: Settings) -> bool:
    """Fast filesystem check to decide if auto-index is necessary."""

    vr = pathlib.Path(settings.obsidian_vault_path).expanduser().resolve()
    vs = pathlib.Path(settings.vectorstore_path).expanduser().resolve()

    manifest = _manifest_path(vs)
    hashes = load_manifest(manifest)
    files = vault_markdown_paths(vr, settings.ignore_globs)

    seen_paths: set[str] = set()
    for md_path in files:
        posix = filepath_to_posix_source(md_path, vr)
        seen_paths.add(posix)
        current_hash = compute_file_content_hash_from_path(md_path)
        if hashes.get(posix) != current_hash:
            return True

    return any(path not in seen_paths for path in hashes)


async def auto_index_if_needed(settings: Settings) -> IndexSummary | None:
    """Run incremental indexing only when vault content changed."""

    if not vault_has_changes(settings):
        _LOG.info("index.auto.skipped", reason="vault_unchanged")
        return None

    _LOG.info("index.auto.triggered", reason="vault_changed")
    return await index_vault(settings)
