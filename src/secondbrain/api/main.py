from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import structlog

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response

from secondbrain.config import Settings
from secondbrain.embeddings.base import EmbeddingError
from secondbrain.ingestion.indexer import index_vault
from secondbrain.llm.base import ChatError
from secondbrain.logging_config import configure_structlog
from secondbrain.models import (
    AskRequest,
    AskResponse,
    ReindexResponse,
    SearchRequest,
    SearchResponse,
)
from secondbrain.rag.pipeline import answer_question
from secondbrain.retrieval.retriever import semantic_search_service
from secondbrain.vectorstore.store import VectorStoreError

_LOG = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_structlog()
    try:
        settings = Settings()
    except Exception as e:
        _LOG.error("settings.load_failed", error=str(e))
        raise
    app.state.settings = settings
    _LOG.info("secondbrain.api.loaded", main_file=str(Path(__file__).resolve()))
    yield


def create_app() -> FastAPI:
    api = FastAPI(title="secondbrain", version="0.1.0", lifespan=lifespan)

    @api.get("/health")
    async def health(request: Request):
        s: Settings = request.app.state.settings
        return {
            "status": "ok",
            "embedding_provider": s.embedding_provider,
            "chat_provider": s.chat_provider,
            "vectorstore_path": s.vectorstore_path,
        }

    @api.post("/search", response_model=None)
    async def search(
        request: Request,
        body: SearchRequest,
        pretty_query: Annotated[bool | None, Query(alias="pretty")] = None,
    ) -> SearchResponse | Response:
        s: Settings = request.app.state.settings
        try:
            hits = await semantic_search_service(s, body)
        except EmbeddingError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except VectorStoreError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        payload = SearchResponse(hits=hits)
        hdr = request.headers.get("x-secondbrain-json-pretty", "").strip().lower()
        hdr_on = hdr in {"1", "true", "yes"}
        pretty = True if pretty_query is True else (bool(body.pretty) or hdr_on)
        if pretty:
            return Response(
                content=json.dumps(
                    payload.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                media_type="application/json; charset=utf-8",
                headers={"X-SecondBrain-Pretty": "1"},
            )
        return payload

    @api.post("/ask", response_model=None)
    async def ask(
        request: Request,
        body: AskRequest,
        pretty_query: Annotated[bool | None, Query(alias="pretty")] = None,
    ) -> AskResponse | Response:
        s: Settings = request.app.state.settings
        try:
            out = await answer_question(s, body)
        except EmbeddingError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except VectorStoreError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except ChatError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        hdr = request.headers.get("x-secondbrain-json-pretty", "").strip().lower()
        hdr_on = hdr in {"1", "true", "yes"}
        pretty = True if pretty_query is True else (bool(body.pretty) or hdr_on)
        if pretty:
            return Response(
                content=json.dumps(out.model_dump(), ensure_ascii=False, indent=2) + "\n",
                media_type="application/json; charset=utf-8",
                headers={"X-SecondBrain-Pretty": "1"},
            )
        return out

    async def _reindex_job(settings_snapshot: Settings) -> None:
        try:
            await index_vault(settings_snapshot)
        except Exception:
            _LOG.exception("reindex.background_failed")

    @api.post("/reindex", response_model=ReindexResponse)
    async def reindex(request: Request, background_tasks: BackgroundTasks) -> ReindexResponse:
        s: Settings = request.app.state.settings
        background_tasks.add_task(_reindex_job, s)
        _LOG.info("reindex.scheduled")
        return ReindexResponse(
            ok=True,
            message="Reindex disparado em segundo plano (pode demorar conforme o vault).",
        )

    @api.get("/", response_class=PlainTextResponse)
    async def root() -> str:
        return "secondbrain API — use /health, /search, /ask, /reindex"

    return api


app = create_app()
