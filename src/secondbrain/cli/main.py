"""CLI segundo cérebro (Typer)."""

from __future__ import annotations

import asyncio
import importlib.metadata
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from secondbrain.chat.session import chat_turn, new_session
from secondbrain.config import Settings
from secondbrain.diagnostics import collect_stats, doctor_sync
from secondbrain.ingestion.indexer import index_vault
from secondbrain.ingestion.watch import run_watch
from secondbrain.logging_config import configure_structlog, resolve_log_level
from secondbrain.models import AskRequest, SearchFilters, SearchRequest
from secondbrain.rag.pipeline import answer_question
from secondbrain.retrieval.retriever import semantic_search_service
from secondbrain.telemetry import setup_otel

_LOG_LEVEL_OPTION: str | None = None
console = Console()


def _print_json_pretty(payload: dict[str, Any]) -> None:
    Console().print_json(data=payload, indent=2)


def _load_settings() -> Settings:
    settings = Settings()
    setup_otel(settings)
    return settings


app_cli = typer.Typer(no_args_is_help=True, help="Obsidian vault → segundo cérebro.")


@app_cli.command("version")
def version_command() -> None:
    """Mostra a versão instalada."""
    try:
        ver = importlib.metadata.version("secondbrain")
    except importlib.metadata.PackageNotFoundError:
        ver = "0.1.0"
    typer.echo(ver)


@app_cli.callback()
def _configure_logs(
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Nível de log (DEBUG, INFO, WARNING, …). Sobrescreve LOG_LEVEL do .env.",
    ),
) -> None:
    global _LOG_LEVEL_OPTION  # noqa: PLW0603
    _LOG_LEVEL_OPTION = log_level
    try:
        settings = Settings()
    except ValidationError:
        settings = None
    level_name = (log_level or (settings.log_level if settings else "INFO") or "INFO").upper()
    log_json = settings.log_json if settings else False
    configure_structlog(level=resolve_log_level(level_name), json_output=log_json)


@app_cli.command("index")
def index_command() -> None:
    settings = _load_settings()
    summary = asyncio.run(index_vault(settings))
    typer.echo(
        "\n".join(
            [
                "Indexação concluída.",
                f"total arquivos: {summary.files_total}",
                f"inalterados (skip): {summary.skipped_unchanged}",
                f"arquivos atualizados: {summary.files_indexed}",
                f"chunks gravados nesta corrida: {summary.chunks_written}",
            ],
        ),
    )


@app_cli.command("search")
def search_command(
    query: str,
    top_k: int = typer.Option(10),
    tag: str | None = typer.Option(None),
    path_prefix: str | None = typer.Option(None, help='Prefixo POSIX (ex.: "notas/projeto/")'),
    json_out: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Saída JSON indentada no terminal (via rich; não precisa do executável jq).",
    ),
) -> None:
    settings = _load_settings()
    filt: SearchFilters | None = None
    if tag or path_prefix:
        filt = SearchFilters(tag=tag, path_prefix=path_prefix)
    req = SearchRequest(query=query, top_k=top_k, filters=filt)
    hits = asyncio.run(semantic_search_service(settings, req))
    if json_out:
        _print_json_pretty({"hits": [h.model_dump() for h in hits]})
        return
    typer.echo(f"{len(hits)} hits")
    typer.echo("")
    for hit in hits:
        md = hit.metadata or {}
        path = str(md.get("source_path", ""))
        title = str(md.get("title") or "")
        label = f"{title} · {path}" if title else path
        score_txt = str(round(hit.score, 4))
        typer.echo(f"[{score_txt}] {label}")
        excerpt = (hit.text or "").replace("\n", " ")[:260]
        typer.echo(f"  {excerpt}...")
        typer.echo("")


@app_cli.command("ask")
def ask_command(
    query: list[str] = typer.Argument(
        ...,
        help="Pergunta; várias palavras sem aspas são juntadas (ex.: secondbrain ask onde falo de IA).",
    ),
    top_k: int = typer.Option(8, "--top-k", "-k"),
    max_context_chars: int = typer.Option(12_000, "--max-context-chars", "-c"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Streaming de tokens na resposta."),
    json_out: bool = typer.Option(
        False,
        "--json",
        "-j",
        help='Resposta como JSON (campos "answer" e "sources"); senão imprime texto legível.',
    ),
) -> None:
    """RAG: recupera trechos do vault e responde com o LLM configurado no .env."""
    q = " ".join(query).strip()
    if not q:
        raise typer.BadParameter("Escreve uma pergunta após «ask».")
    settings = _load_settings()
    try:
        req = AskRequest(
            query=q,
            top_k=top_k,
            max_context_chars=max_context_chars,
            stream=stream,
        )
    except ValidationError as exc:
        raise typer.BadParameter("; ".join(f"{e['loc']}: {e['msg']}" for e in exc.errors())) from exc

    if stream and not json_out:
        async def _stream_ask() -> AskRequest:
            return req

        async def _run_stream() -> None:
            from secondbrain.ingestion.indexer import auto_index_if_needed
            from secondbrain.llm.factory import aclose_chat_client, make_chat_client
            from secondbrain.rag.pipeline import build_context_with_citations
            from secondbrain.rag.prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE
            from secondbrain.retrieval.retriever import retrieve_top_hits

            await auto_index_if_needed(settings)
            hits = await retrieve_top_hits(settings, query=req.query, top_k=req.top_k)
            if not hits:
                typer.echo("Sem contexto suficiente no vault.")
                return
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
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            parts: list[str] = []
            try:
                with Live(Spinner("dots", text="A gerar resposta…"), console=console, refresh_per_second=12):
                    async for token in chat.complete_stream(messages):
                        parts.append(token)
                answer = "".join(parts)
                console.print(answer.strip())
            finally:
                await aclose_chat_client(chat)
            typer.echo("")
            typer.secho("Fontes (trechos usados):", dim=True)
            for cite in citations:
                h = cite.heading_path.strip() if cite.heading_path else ""
                heading_suffix = f" · {h}" if h else ""
                line_suffix = ""
                if cite.line_start:
                    line_suffix = f":{cite.line_start}"
                    if cite.line_end and cite.line_end != cite.line_start:
                        line_suffix += f"-{cite.line_end}"
                if cite.title:
                    typer.echo(f"  • {cite.title} · {cite.path}{line_suffix}{heading_suffix}")
                else:
                    typer.echo(f"  • {cite.path}{line_suffix}{heading_suffix}")
                if cite.obsidian_uri:
                    typer.echo(f"    {cite.obsidian_uri}")

        asyncio.run(_run_stream())
        return

    out = asyncio.run(answer_question(settings, req))
    if json_out:
        _print_json_pretty(out.model_dump())
        return
    typer.echo(out.answer.strip())
    typer.echo("")
    typer.secho("Fontes (trechos usados):", dim=True)
    if not out.sources:
        typer.echo("  (nenhuma citação registada)")
        return
    for cite in out.sources:
        h = cite.heading_path.strip() if cite.heading_path else ""
        heading_suffix = f" · {h}" if h else ""
        line_suffix = ""
        if cite.line_start:
            line_suffix = f":{cite.line_start}"
            if cite.line_end and cite.line_end != cite.line_start:
                line_suffix += f"-{cite.line_end}"
        if cite.title:
            typer.echo(f"  • {cite.title} · {cite.path}{line_suffix}{heading_suffix}")
        else:
            typer.echo(f"  • {cite.path}{line_suffix}{heading_suffix}")
        if cite.obsidian_uri:
            typer.echo(f"    {cite.obsidian_uri}")


@app_cli.command("doctor")
def doctor_command() -> None:
    """Verifica vault, vectorstore e conectividade Ollama."""
    settings = _load_settings()
    report = doctor_sync(settings)
    typer.echo(f"Vault OK: {report.vault_ok}")
    typer.echo(f"Vectorstore OK: {report.vectorstore_ok}")
    if report.ollama_ok is not None:
        typer.echo(f"Ollama OK: {report.ollama_ok}")
    if report.embed_model_available is not None:
        typer.echo(f"Modelo embed disponível: {report.embed_model_available}")
    if report.chat_model_available is not None:
        typer.echo(f"Modelo chat disponível: {report.chat_model_available}")
    typer.echo(f"Manifest embed_dim: {report.manifest_embed_dim}")
    if report.issues:
        typer.secho("Problemas:", fg=typer.colors.RED)
        for issue in report.issues:
            typer.echo(f"  - {issue}")
        raise typer.Exit(code=1)
    typer.secho("Tudo OK.", fg=typer.colors.GREEN)


@app_cli.command("stats")
def stats_command() -> None:
    """Estatísticas do índice e vectorstore."""
    settings = _load_settings()
    stats = collect_stats(settings)
    _print_json_pretty(
        {
            "files_in_vault": stats.files_in_vault,
            "files_in_manifest": stats.files_in_manifest,
            "embed_model": stats.embed_model,
            "embed_dim": stats.embed_dim,
            "vectorstore_bytes": stats.vectorstore_bytes,
        },
    )


@app_cli.command("watch")
def watch_command() -> None:
    """Observa o vault e re-indexa incrementalmente (Ctrl-C para sair)."""
    settings = _load_settings()
    typer.echo(f"A observar {settings.obsidian_vault_path} …")
    run_watch(settings)


@app_cli.command("chat")
def chat_command(
    persist: bool = typer.Option(False, "--persist", help="Grava sessão em data/sessions/."),
) -> None:
    """REPL conversacional sobre o vault."""
    settings = _load_settings()
    session = new_session()
    typer.echo("Modo chat (exit/quit para sair).")
    while True:
        try:
            user_text = typer.prompt("Tu")
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.strip().lower() in {"exit", "quit"}:
            break
        if not user_text.strip():
            continue
        answer = asyncio.run(chat_turn(settings, session, user_text))
        typer.echo(f"Assistente: {answer}\n")
    if persist:
        from pathlib import Path

        base = Path(settings.vectorstore_path).parent
        session.save(base)
        typer.echo(f"Sessão gravada: {session.session_id}")


@app_cli.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8765),
) -> None:
    """Servidor HTTP FastAPI (search, ask, healthz)."""
    try:
        import uvicorn
    except ImportError as e:
        raise typer.BadParameter(
            "Instale deps API: pip install -e '.[api]'",
        ) from e
    uvicorn.run("secondbrain.api.main:app", host=host, port=port, reload=False)


def app_main() -> None:
    app_cli()


if __name__ == "__main__":
    app_cli()
