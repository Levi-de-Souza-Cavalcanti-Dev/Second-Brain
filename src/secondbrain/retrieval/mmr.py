"""Maximal Marginal Relevance diversification."""

from __future__ import annotations

import math

from secondbrain.models import SearchHit


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mmr_diversify(
    hits: list[SearchHit],
    query_vec: list[float],
    chunk_vectors: dict[str, list[float]],
    *,
    lambda_mult: float = 0.5,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Select diverse hits using MMR. chunk_vectors keyed by chunk text prefix."""

    if not hits or lambda_mult is None:
        return hits

    k = top_k or len(hits)
    selected: list[SearchHit] = []
    remaining = list(hits)

    def vec_for(hit: SearchHit) -> list[float]:
        key = hit.text[:80]
        return chunk_vectors.get(key, query_vec)

    while remaining and len(selected) < k:
        best_idx = 0
        best_score = float("-inf")
        for i, hit in enumerate(remaining):
            rel = _cosine(vec_for(hit), query_vec) * hit.score
            div = 0.0
            if selected:
                div = max(_cosine(vec_for(hit), vec_for(s)) for s in selected)
            mmr = lambda_mult * rel - (1.0 - lambda_mult) * div
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected
