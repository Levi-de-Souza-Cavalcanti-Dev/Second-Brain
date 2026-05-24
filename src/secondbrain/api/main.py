"""FastAPI HTTP server for search and ask."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel

from secondbrain.config import Settings
from secondbrain.models import AskRequest, SearchRequest
from secondbrain.rag.pipeline import answer_question
from secondbrain.retrieval.retriever import semantic_search_service

app = FastAPI(title="Second Brain API", version="0.1.0")


class HealthResponse(BaseModel):
    status: str


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/search")
async def search(req: SearchRequest) -> dict[str, object]:
    settings = Settings()
    hits = await semantic_search_service(settings, req)
    return {"hits": [h.model_dump() for h in hits]}


@app.post("/ask")
async def ask(req: AskRequest) -> dict[str, object]:
    settings = Settings()
    out = await answer_question(settings, req)
    return out.model_dump()


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> AsyncIterator[str]:
    from sse_starlette.sse import EventSourceResponse

    settings = Settings()
    stream_req = AskRequest(
        query=req.query,
        top_k=req.top_k,
        max_context_chars=req.max_context_chars,
        stream=True,
    )

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        out = await answer_question(settings, stream_req)
        yield {"event": "message", "data": out.answer}

    return EventSourceResponse(event_generator())
