from __future__ import annotations

import structlog

from secondbrain.config import Settings
from secondbrain.ingestion.indexer import auto_index_if_needed
from secondbrain.llm.factory import aclose_chat_client, make_chat_client
from secondbrain.models import AskRequest, AskResponse, SearchHit, SearchRequest, SourceCitation
from secondbrain.rag.prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
from secondbrain.retrieval.retriever import (
    build_obsidian_uri,
    retrieve_top_hits,
    semantic_search_service,
)

_LOG = structlog.get_logger()

EMPTY_CONTEXT_ANSWER = "Sem contexto suficiente no vault."


def build_context_with_citations(
    hits: list[SearchHit],
    *,
    max_context_chars: int,
    settings: Settings | None = None,
) -> tuple[str, list[SourceCitation]]:
    blocks: list[str] = []
    used = 0
    citations: list[SourceCitation] = []
    seen_cite: set[tuple[str, str]] = set()
    seen_text: set[str] = set()

    for h in hits:
        md = h.metadata or {}
        path = str(md.get("source_path", ""))
        heading = str(md.get("heading_path", ""))
        title = str(md.get("title") or "")
        line_start = md.get("line_start")
        line_end = md.get("line_end")
        ls = int(line_start) if isinstance(line_start, int | str) and str(line_start).isdigit() else None
        le = int(line_end) if isinstance(line_end, int | str) and str(line_end).isdigit() else None
        key = (path, heading)
        if key not in seen_cite:
            obs_uri = build_obsidian_uri(settings, path, ls) if settings else ""
            citations.append(
                SourceCitation(
                    path=path,
                    heading_path=heading,
                    title=title,
                    line_start=ls,
                    line_end=le,
                    obsidian_uri=obs_uri,
                ),
            )
            seen_cite.add(key)

        body = h.text.strip()
        if not body or body in seen_text:
            continue
        seen_text.add(body)

        line_suffix = ""
        if ls is not None:
            line_suffix = f":{ls}" + (f"-{le}" if le and le != ls else "")
        header = f"--- fonte: {path}{line_suffix} | heading_path: {heading}"
        if title:
            header = f"--- fonte: {title} · {path}{line_suffix} | heading_path: {heading}"
        block = f"{header}\n{body}\n"
        if used + len(block) > max_context_chars:
            break
        blocks.append(block)
        used += len(block)

    return "\n".join(blocks), citations


async def _expand_wikilinks(
    settings: Settings,
    hits: list[SearchHit],
) -> list[SearchHit]:
    if not settings.rag_link_expansion:
        return hits
    expanded = list(hits)
    seen_paths: set[str] = {str(h.metadata.get("source_path", "")) for h in hits}
    for hit in hits:
        links_raw = str(hit.metadata.get("wikilinks_joined", ""))
        for link in [x.strip() for x in links_raw.split(",") if x.strip()]:
            if link in seen_paths:
                continue
            extra = await semantic_search_service(
                settings,
                SearchRequest(query=link, top_k=settings.rag_link_expansion_top_k),
            )
            for e in extra:
                p = str(e.metadata.get("source_path", ""))
                if p not in seen_paths:
                    expanded.append(e)
                    seen_paths.add(p)
    return expanded


async def answer_question(settings: Settings, req: AskRequest) -> AskResponse:
    await auto_index_if_needed(settings)
    _LOG.info(
        "rag.ask.start",
        query_preview=req.query[:120],
        top_k=req.top_k,
        max_context_chars=req.max_context_chars,
    )
    hits = await retrieve_top_hits(settings, query=req.query, top_k=req.top_k)
    hits = await _expand_wikilinks(settings, hits)
    _LOG.info("rag.ask.retrieval_done", n_hits=len(hits))

    if not hits:
        _LOG.info("rag.ask.empty_context")
        return AskResponse(answer=EMPTY_CONTEXT_ANSWER, sources=[])

    context, citations = build_context_with_citations(
        hits,
        max_context_chars=req.max_context_chars,
        settings=settings,
    )

    chat = make_chat_client(settings)
    user_msg = USER_MESSAGE_TEMPLATE.format(
        context=context if context.strip() else "(vazio)",
        query=req.query,
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        _LOG.info(
            "rag.ask.chat_request",
            context_chars=len(context),
            chat_provider=str(settings.chat_provider),
            stream=req.stream,
        )
        if req.stream:
            parts: list[str] = []
            async for token in chat.complete_stream(messages):
                parts.append(token)
            answer_text = "".join(parts)
        else:
            answer_text = await chat.complete(messages)
    finally:
        await aclose_chat_client(chat)

    _LOG.info("rag.ask.done", answer_chars=len(answer_text))
    return AskResponse(answer=answer_text, sources=citations)
