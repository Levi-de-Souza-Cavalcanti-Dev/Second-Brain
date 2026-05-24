from __future__ import annotations

import pytest

from secondbrain.models import DocumentChunk
from secondbrain.vectorstore.chroma_store import ChromaVectorStore
from secondbrain.vectorstore.store import VectorStoreError


@pytest.mark.asyncio
async def test_chroma_upsert_search_delete(tmp_path) -> None:
    store = ChromaVectorStore(str(tmp_path / "chroma"), "test_coll", expected_dimension=4)
    chunk = DocumentChunk(
        chunk_id="c1",
        source_path="notes/a.md",
        heading_path="H",
        text="hello world",
        tags=("ml",),
        wikilinks=(),
        file_hash="abc",
        extra_metadata={"title": "Note A"},
    )
    emb = [[0.1, 0.2, 0.3, 0.4]]
    await store.upsert_documents([chunk], emb)
    hits = await store.semantic_search(emb[0], top_k=1)
    assert hits
    assert hits[0].metadata.get("title") == "Note A"
    assert hits[0].metadata.get("tags_pipe") == "|ml|"

    filtered = await store.semantic_search(
        emb[0],
        top_k=5,
        where={"tags": {"$contains": "ml"}},
    )
    assert len(filtered) == 1

    await store.delete_by_source_path("notes/a.md")
    hits_after = await store.semantic_search(emb[0], top_k=5)
    assert hits_after == []


def test_chroma_dimension_mismatch(tmp_path) -> None:
    store = ChromaVectorStore(str(tmp_path / "chroma2"), "dim_coll", expected_dimension=4)
    chunk = DocumentChunk(
        chunk_id="c1",
        source_path="x.md",
        heading_path="",
        text="t",
        tags=(),
        wikilinks=(),
        file_hash="h",
    )
    import asyncio

    asyncio.run(store.upsert_documents([chunk], [[0.1, 0.2, 0.3, 0.4]]))

    with pytest.raises(VectorStoreError, match="embedding dimension changed"):
        ChromaVectorStore(str(tmp_path / "chroma2"), "dim_coll", expected_dimension=8)
