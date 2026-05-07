"""Testes dos endpoints FastAPI relacionados ao formato JSON (pretty)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from secondbrain.models import SearchHit


def test_search_pretty_indent_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    async def fake_semantic_search(_s, _body):
        return [
            SearchHit(
                text="linha uma",
                score=0.9,
                metadata={"source_path": "x.md"},
            )
        ]

    monkeypatch.setattr("secondbrain.api.main.semantic_search_service", fake_semantic_search)

    from secondbrain.api.main import app

    with TestClient(app) as client:
        r = client.post("/search", json={"query": "q", "pretty": True})
        assert r.status_code == 200
        assert "\n  " in r.text
        assert r.headers.get("content-type", "").startswith("application/json")
        assert r.headers.get("X-SecondBrain-Pretty") == "1"

        r_compact = client.post("/search", json={"query": "q", "pretty": False})
        assert r_compact.status_code == 200
        lines = [ln for ln in r_compact.text.split("\n") if ln.strip()]
        assert len(lines) == 1


def test_search_pretty_query_false_does_not_override_body_true(monkeypatch, tmp_path) -> None:
    """`?pretty=false` não deve silenciar `"pretty": true` no JSON do body."""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    async def fake_semantic_search(_s, _body):
        return [SearchHit(text="a", score=0.5, metadata={})]

    monkeypatch.setattr("secondbrain.api.main.semantic_search_service", fake_semantic_search)

    from secondbrain.api.main import app

    with TestClient(app) as client:
        r = client.post("/search?pretty=false", json={"query": "q", "pretty": True})
        assert r.status_code == 200
        assert "\n  " in r.text
        assert r.headers.get("X-SecondBrain-Pretty") == "1"


def test_search_pretty_via_header_when_body_omits(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    async def fake_semantic_search(_s, _body):
        return [SearchHit(text="b", score=0.5, metadata={})]

    monkeypatch.setattr("secondbrain.api.main.semantic_search_service", fake_semantic_search)

    from secondbrain.api.main import app

    with TestClient(app) as client:
        r = client.post(
            "/search",
            json={"query": "q", "pretty": False},
            headers={"X-Secondbrain-Json-Pretty": "true"},
        )
        assert r.status_code == 200
        assert "\n  " in r.text
        assert r.headers.get("X-SecondBrain-Pretty") == "1"


def test_search_pretty_query_true_forces_even_if_body_false(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))

    async def fake_semantic_search(_s, _body):
        return [SearchHit(text="z", score=1.0, metadata={})]

    monkeypatch.setattr("secondbrain.api.main.semantic_search_service", fake_semantic_search)

    from secondbrain.api.main import app

    with TestClient(app) as client:
        r = client.post("/search?pretty=true", json={"query": "q", "pretty": False})
        assert r.status_code == 200
        assert '\n  "hits"' in r.text
