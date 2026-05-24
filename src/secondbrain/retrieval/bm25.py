"""BM25 lexical retrieval with persisted sidecar index."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from secondbrain.constants import BM25_INDEX_FILENAME
from secondbrain.models import DocumentChunk, SearchHit


@dataclass(slots=True)
class BM25Index:
    chunk_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    metadatas: list[dict[str, object]] = field(default_factory=list)
    tokenized_corpus: list[list[str]] = field(default_factory=list)
    bm25: BM25Okapi | None = None

    def rebuild(self) -> None:
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        by_path = {c.chunk_id: c for c in chunks}
        for cid in list(by_path):
            if cid in self.chunk_ids:
                idx = self.chunk_ids.index(cid)
                self.chunk_ids.pop(idx)
                self.texts.pop(idx)
                self.metadatas.pop(idx)
                self.tokenized_corpus.pop(idx)
        for c in chunks:
            self.chunk_ids.append(c.chunk_id)
            self.texts.append(c.text)
            self.metadatas.append(
                {
                    "source_path": c.source_path,
                    "heading_path": c.heading_path,
                    "title": str(c.extra_metadata.get("title") or ""),
                    "tags_joined": ",".join(c.tags),
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                },
            )
            self.tokenized_corpus.append(_tokenize(c.text))
        self.rebuild()

    def delete_by_source_path(self, source_path: str) -> None:
        keep_ids: list[str] = []
        keep_texts: list[str] = []
        keep_meta: list[dict[str, object]] = []
        keep_tok: list[list[str]] = []
        for i, cid in enumerate(self.chunk_ids):
            meta = self.metadatas[i]
            if str(meta.get("source_path", "")) == source_path:
                continue
            keep_ids.append(cid)
            keep_texts.append(self.texts[i])
            keep_meta.append(meta)
            keep_tok.append(self.tokenized_corpus[i])
        self.chunk_ids = keep_ids
        self.texts = keep_texts
        self.metadatas = keep_meta
        self.tokenized_corpus = keep_tok
        self.rebuild()


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.strip()]


def bm25_index_path(vectorstore_root: Path) -> Path:
    return vectorstore_root / BM25_INDEX_FILENAME


def load_bm25_index(path: Path) -> BM25Index:
    if not path.is_file():
        return BM25Index()
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, BM25Index):
        if data.bm25 is None and data.tokenized_corpus:
            data.rebuild()
        return data
    return BM25Index()


def save_bm25_index(path: Path, index: BM25Index) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


class BM25LexicalRetriever:
    def __init__(self, index: BM25Index) -> None:
        self._index = index

    async def lexical_search(self, query: str, top_k: int) -> list[SearchHit]:
        if self._index.bm25 is None or not self._index.texts:
            return []
        scores = self._index.bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        hits: list[SearchHit] = []
        max_score = max((s for _, s in ranked), default=1.0) or 1.0
        for idx, score in ranked:
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    text=self._index.texts[idx],
                    score=float(score / max_score),
                    metadata=dict(self._index.metadatas[idx]),
                ),
            )
        return hits


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchHit]],
    *,
    k: int = 60,
) -> list[SearchHit]:
    """Combine multiple ranked lists via RRF."""

    scores: dict[str, float] = {}
    by_key: dict[str, SearchHit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            md = hit.metadata or {}
            key = f"{md.get('source_path', '')}|{md.get('heading_path', '')}|{hit.text[:80]}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            by_key.setdefault(key, hit)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[SearchHit] = []
    for key, score in fused:
        base = by_key[key]
        out.append(SearchHit(text=base.text, score=score, metadata=base.metadata))
    return out
