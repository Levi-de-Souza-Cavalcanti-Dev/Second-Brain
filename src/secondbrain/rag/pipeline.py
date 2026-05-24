from __future__ import annotations

import structlog

from secondbrain.config import Settings
from secondbrain.ingestion.indexer import auto_index_if_needed
from secondbrain.llm.factory import aclose_chat_client, make_chat_client
from secondbrain.models import AskRequest, AskResponse, SearchHit, SourceCitation
from secondbrain.rag.prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
from secondbrain.retrieval.retriever import retrieve_top_hits

_LOG = structlog.get_logger()

EMPTY_CONTEXT_ANSWER = "Sem contexto suficiente no vault."


def build_context_with_citations(
    hits: list[SearchHit],
    *,
    max_context_chars: int,
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
        key = (path, heading)
        if key not in seen_cite:
            citations.append(SourceCitation(path=path, heading_path=heading, title=title))
            seen_cite.add(key)

        body = h.text.strip()
        if not body or body in seen_text:
            continue
        seen_text.add(body)

        header = f"--- fonte: {path} | heading_path: {heading}"
        if title:
            header = f"--- fonte: {title} · {path} | heading_path: {heading}"
        block = f"{header}\n{body}\n"
        if used + len(block) > max_context_chars:
            break
        blocks.append(block)
        used += len(block)

    return "\n".join(blocks), citations


async def answer_question(settings: Settings, req: AskRequest) -> AskResponse:
    await auto_index_if_needed(settings)
    _LOG.info(
        "rag.ask.start",
        query_preview=req.query[:120],
        top_k=req.top_k,
        max_context_chars=req.max_context_chars,
    )
    hits = await retrieve_top_hits(settings, query=req.query, top_k=req.top_k)
    _LOG.info("rag.ask.retrieval_done", n_hits=len(hits))

    if not hits:
        _LOG.info("rag.ask.empty_context")
        return AskResponse(answer=EMPTY_CONTEXT_ANSWER, sources=[])

    context, citations = build_context_with_citations(hits, max_context_chars=req.max_context_chars)

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
        )
        answer_text = await chat.complete(messages)
    finally:
        await aclose_chat_client(chat)

    _LOG.info("rag.ask.done", answer_chars=len(answer_text))
    return AskResponse(answer=answer_text, sources=citations)
