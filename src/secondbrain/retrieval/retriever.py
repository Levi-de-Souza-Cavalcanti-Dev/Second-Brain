from __future__ import annotations

from pathlib import Path

from secondbrain.config import Settings
from secondbrain.embeddings.factory import aclose_embedder, make_embedder
from secondbrain.ingestion.indexer import load_manifest, manifest_path
from secondbrain.models import SearchFilters, SearchHit, SearchRequest
from secondbrain.runtime_cache import get_embedder_and_store
from secondbrain.vectorstore.chroma_store import chroma_where_from_filters


async def _embedding_dimension(settings: Settings) -> int:
    vs = Path(settings.vectorstore_path).expanduser().resolve()
    manifest = load_manifest(manifest_path(vs))
    if manifest.embed_dim > 0:
        return manifest.embed_dim
    embedder = make_embedder(settings)
    try:
        return len((await embedder.embed_many([" "]))[0])
    finally:
        await aclose_embedder(embedder)


def apply_search_filters(hits: list[SearchHit], filters: SearchFilters | None) -> list[SearchHit]:
    """Post-filter for path prefix and case-insensitive tag match."""

    if not filters or (not filters.tag and not filters.path_prefix):
        return hits
    tag = filters.tag
    pref = filters.path_prefix or ""
    out: list[SearchHit] = []
    for h in hits:
        md = h.metadata or {}
        if pref and not str(md.get("source_path", "")).startswith(pref):
            continue
        if tag:
            joined = str(md.get("tags_joined", ""))
            known = [t for t in joined.split(",") if t]
            if not any(t == tag or t.lower() == tag.lower() for t in known):
                continue
        out.append(h)
    return out


async def semantic_search_service(
    settings: Settings,
    req: SearchRequest,
    *,
    overfetch_factor: int = 5,
    max_overfetch: int = 200,
) -> list[SearchHit]:
    dim = await _embedding_dimension(settings)
    embedder, store = await get_embedder_and_store(settings, dimension=dim)
    qv = (await embedder.embed_many([req.query]))[0]

    where = chroma_where_from_filters(tag=req.filters.tag if req.filters else None)

    fetch_n = req.top_k
    if req.filters and (req.filters.tag or req.filters.path_prefix):
        fetch_n = min(max_overfetch, max(req.top_k * overfetch_factor, req.top_k))

    raw_hits = await store.semantic_search(qv, top_k=fetch_n, where=where)
    filtered = apply_search_filters(raw_hits, req.filters)
    filtered.sort(key=lambda h: h.score, reverse=True)
    return filtered[: req.top_k]


async def retrieve_top_hits(
    settings: Settings,
    *,
    query: str,
    top_k: int,
) -> list[SearchHit]:
    """Shared retrieval para ask (sem filtros adicionais)."""

    return await semantic_search_service(
        settings,
        SearchRequest(query=query, top_k=top_k, filters=None),
    )
