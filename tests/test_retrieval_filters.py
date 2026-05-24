from __future__ import annotations

from secondbrain.models import SearchFilters, SearchHit
from secondbrain.retrieval.retriever import apply_search_filters
from secondbrain.vectorstore.chroma_store import chroma_where_from_filters


def test_chroma_where_tag_only() -> None:
    where = chroma_where_from_filters(tag="ML")
    assert where == {"tags": {"$contains": "ML"}}


def test_chroma_where_none_without_tag() -> None:
    assert chroma_where_from_filters(tag=None) is None


def test_apply_search_filters_tag_case_insensitive() -> None:
    hits = [
        SearchHit(text="a", score=1.0, metadata={"tags_joined": "ml,python"}),
        SearchHit(text="b", score=0.9, metadata={"tags_joined": "other"}),
    ]
    out = apply_search_filters(hits, SearchFilters(tag="ML"))
    assert len(out) == 1
    assert out[0].text == "a"


def test_apply_search_filters_path_prefix() -> None:
    hits = [
        SearchHit(text="a", score=1.0, metadata={"source_path": "notes/x.md"}),
        SearchHit(text="b", score=0.9, metadata={"source_path": "other/x.md"}),
    ]
    out = apply_search_filters(hits, SearchFilters(path_prefix="notes/"))
    assert len(out) == 1
    assert out[0].text == "a"
