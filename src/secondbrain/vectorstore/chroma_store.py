"""ChromaDB persistent vector store."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings

from secondbrain.models import DocumentChunk, SearchHit
from secondbrain.vectorstore.store import VectorStoreError

_LOG = structlog.get_logger()


def chroma_where_from_filters(*, tag: str | None) -> dict[str, Any] | None:
    """Chroma-native metadata filter on tag list (path prefix is post-filtered in Python)."""

    if not tag:
        return None
    return {"tags": {"$contains": tag}}


class ChromaVectorStore:
    """Thread-offloaded Chroma wrapper (client is synchronous)."""

    def __init__(
        self,
        persistence_path: str,
        collection_name: str,
        *,
        expected_dimension: int | None = None,
    ) -> None:
        self._persistence_path = persistence_path
        self._client = chromadb.PersistentClient(
            path=persistence_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if expected_dimension is not None and expected_dimension > 0:
            self._assert_dimension(expected_dimension)

    def _collection_embedding_dim(self) -> int | None:
        if self._collection.count() == 0:
            return None
        peek = self._collection.get(limit=1, include=["embeddings"])
        embeddings = peek.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        first = embeddings[0]
        if first is None:
            return None
        return len(first)

    def _assert_dimension(self, expected: int) -> None:
        existing = self._collection_embedding_dim()
        if existing is not None and existing != expected:
            raise VectorStoreError(
                f"embedding dimension changed: {existing} → {expected}; "
                "reindex the vault (`secondbrain index`) after changing the embed model.",
            )

    def _tags_pipe(self, tags: tuple[str, ...]) -> str:
        if not tags:
            return ""
        lowered = [t.lower() for t in tags if t]
        return "|" + "|".join(lowered) + "|"

    def _chunk_metadata_dict(self, c: DocumentChunk) -> dict[str, str | int | float | bool | list[str]]:
        title = str(c.extra_metadata.get("title") or "")
        meta: dict[str, str | int | float | bool | list[str]] = {
            "source_path": c.source_path,
            "heading_path": c.heading_path,
            "file_hash": c.file_hash,
            "heading_level": int(c.heading_level),
            "chunk_index_in_section": int(c.chunk_index_in_section),
            "tags_joined": ",".join(c.tags),
            "tags_pipe": self._tags_pipe(c.tags),
            "wikilinks_joined": ",".join(c.wikilinks),
            "title": title,
            "line_start": int(c.line_start),
            "line_end": int(c.line_end),
        }
        if c.tags:
            meta["tags"] = list(c.tags)
        return meta

    def _upsert_sync(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunks e embeddings com tamanhos diferentes.")
        if not chunks:
            return
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [self._chunk_metadata_dict(c) for c in chunks]
        embs = [list(map(float, e)) for e in embeddings]
        self._collection.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

    def _delete_sync(self, source_path: str) -> None:
        try:
            self._collection.delete(where={"source_path": source_path})
        except Exception as exc:
            _LOG.warning("chroma.delete_failed", source_path=source_path, error=str(exc))
            raise VectorStoreError(f"Falha ao apagar chunks de {source_path!r}: {exc}") from exc

    def _query_sync(
        self,
        vec: Sequence[float],
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[SearchHit]:
        if top_k <= 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(map(float, vec))],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        raw = self._collection.query(**kwargs)
        hits: list[SearchHit] = []
        distances = raw.get("distances") or [[]]
        documents = raw.get("documents") or [[]]
        metadatas = raw.get("metadatas") or [[]]

        docs0 = documents[0] if documents else []
        dist0 = distances[0] if distances else []
        meta0 = metadatas[0] if metadatas else []
        for idx, txt in enumerate(docs0):
            if txt is None:
                continue
            d = float(dist0[idx]) if idx < len(dist0) else 0.0
            score = max(0.0, min(1.0, 1.0 - d))
            md = dict(meta0[idx]) if idx < len(meta0) else {}
            hits.append(SearchHit(text=txt or "", score=score, metadata=md))
        return hits

    async def upsert_documents(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        await asyncio.to_thread(self._upsert_sync, chunks, embeddings)

    async def delete_by_source_path(self, source_path: str) -> None:
        await asyncio.to_thread(self._delete_sync, source_path)

    async def semantic_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        return await asyncio.to_thread(self._query_sync, query_embedding, top_k, where)

    async def close(self) -> None:
        return
