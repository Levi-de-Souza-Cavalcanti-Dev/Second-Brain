from __future__ import annotations

from pathlib import Path

from secondbrain.config import Settings
from secondbrain.embeddings.factory import aclose_embedder, make_embedder
from secondbrain.models import SearchFilters, SearchHit, SearchRequest
from secondbrain.vectorstore.factory import VectorStoreDeps, build_vector_store


def apply_search_filters(hits: list[SearchHit], filters: SearchFilters | None) -> list[SearchHit]:
    if not filters or (not filters.tag and not filters.path_prefix):
        return hits
    out: list[SearchHit] = []
    tag = filters.tag
    pref = filters.path_prefix or ""
    for h in hits:
        md = h.metadata or {}
        if pref:
            p = str(md.get("source_path", ""))
            if not p.startswith(pref):
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
    embedder = make_embedder(settings)
    try:
        qv = (await embedder.embed_many([req.query]))[0]
    finally:
        await aclose_embedder(embedder)

    deps = VectorStoreDeps(dimension=len(qv), collection_name="secondbrain_notes")
    store = await build_vector_store(str(Path(settings.vectorstore_path).expanduser().resolve()), deps)
    try:
        fetch_n = min(max_overfetch, max(req.top_k * overfetch_factor, req.top_k))
        raw_hits = await store.semantic_search(qv, top_k=fetch_n)
        filtered = apply_search_filters(raw_hits, req.filters)
        filtered.sort(key=lambda h: h.score, reverse=True)
        return filtered[: req.top_k]
    finally:
        await store.close()


async def retrieve_top_hits(
    settings: Settings,
    *,
    query: str,
    top_k: int,
) -> list[SearchHit]:
    """Shared retrieval para /ask (sem filtros adicionais)."""

    return await semantic_search_service(
        settings,
        SearchRequest(query=query, top_k=top_k, filters=None),
    )
