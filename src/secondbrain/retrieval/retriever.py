from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from secondbrain.config import Settings
from secondbrain.embeddings.factory import aclose_embedder, make_embedder
from secondbrain.ingestion.indexer import get_bm25_retriever
from secondbrain.ingestion.manifest import load_manifest, manifest_path
from secondbrain.models import SearchFilters, SearchHit, SearchRequest
from secondbrain.retrieval.bm25 import reciprocal_rank_fusion
from secondbrain.retrieval.mmr import mmr_diversify
from secondbrain.retrieval.reranker import BgeReranker, NoopReranker, RerankerProtocol
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


def _make_reranker(settings: Settings) -> RerankerProtocol:
    if settings.reranker_enabled:
        return BgeReranker(settings.reranker_model)
    return NoopReranker()


async def semantic_search_service(
    settings: Settings,
    req: SearchRequest,
    *,
    overfetch_factor: int = 5,
    max_overfetch: int = 200,
) -> list[SearchHit]:
    dim = await _embedding_dimension(settings)
    embedder, store = await get_embedder_and_store(settings, dimension=dim, cache_queries=True)
    qv = (await embedder.embed_many([req.query]))[0]

    where = chroma_where_from_filters(tag=req.filters.tag if req.filters else None)

    fetch_n = req.top_k
    if req.filters and (req.filters.tag or req.filters.path_prefix):
        fetch_n = min(max_overfetch, max(req.top_k * overfetch_factor, req.top_k))
    if settings.reranker_enabled:
        fetch_n = min(max_overfetch, max(fetch_n, settings.reranker_top_n))

    dense_hits = await store.semantic_search(qv, top_k=fetch_n, where=where)

    if settings.hybrid_search:
        bm25 = get_bm25_retriever(settings)
        if bm25 is not None:
            lexical_hits = await bm25.lexical_search(req.query, top_k=fetch_n)
            raw_hits = reciprocal_rank_fusion(
                [dense_hits, lexical_hits],
                k=settings.hybrid_rrf_k,
            )
        else:
            raw_hits = dense_hits
    else:
        raw_hits = dense_hits

    reranker = _make_reranker(settings)
    raw_hits = await reranker.rerank(req.query, raw_hits, top_n=settings.reranker_top_n)

    if settings.mmr_lambda is not None:
        chunk_vectors = {h.text[:80]: qv for h in raw_hits}
        raw_hits = mmr_diversify(
            raw_hits,
            qv,
            chunk_vectors,
            lambda_mult=settings.mmr_lambda,
            top_k=fetch_n,
        )

    filtered = apply_search_filters(raw_hits, req.filters)
    filtered.sort(key=lambda h: h.score, reverse=True)
    return filtered[: req.top_k]


async def retrieve_top_hits(
    settings: Settings,
    *,
    query: str,
    top_k: int,
) -> list[SearchHit]:
    return await semantic_search_service(
        settings,
        SearchRequest(query=query, top_k=top_k, filters=None),
    )


def build_obsidian_uri(settings: Settings, path: str, line_start: int | None = None) -> str:
    vault = Path(settings.obsidian_vault_path).name
    file_q = quote(path)
    uri = f"obsidian://open?vault={quote(vault)}&file={file_q}"
    if line_start is not None and line_start > 0:
        uri += f"&line={line_start}"
    return uri
